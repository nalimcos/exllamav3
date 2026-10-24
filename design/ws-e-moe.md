# Workstream E — MoE / routing / HC + reconstruct (Maxwell port)

**Target:** Maxwell M40 (CC 5.2, 24 SM, 288 GB/s) / M10 (CC 5.0, 5 SM, 83 GB/s). No tensor cores, no `dp4a`, no native fp16 arithmetic (`__hfma2`/`hexp` are sm_53+), no `cp.async`; all MoE/HC compute is fp32 FMA.

Worktree: `exllamav3-wt-maxwell` (branch `maxwell`). Builds on Workstream B (fp32 dequant core + dense GEMM), C (GEMV/decode), and F (TP sync + MoE CPU-offload kernel path). Scope: the routed-expert (MoE) path, the routing (router) GEMM, the hyperconnection (HC) mix, and the reconstruct (dequant→GEMM) path all run correctly on Maxwell.

## Hardware & arch boundary

- **Device tier (C++):** `EXL3_MAXWELL` (`__CUDA_ARCH__ < 600`, `exllamav3_ext/arch.cuh`).
- **Host tier (C++):** `CC_MAXWELL` (`prop.major == 5`, `exllamav3_ext/quant/exl3_devctx.cuh`); dispatch via `DevCtx::instance().get_cc(device) == CC_MAXWELL`.
- **Python tier:** `is_maxwell(device)` in `exllamav3/util/backend.py` — `torch.cuda.get_device_capability(device)[0] == 5` (matches `CC_MAXWELL`; NOT `< 6`), `False` for non-CUDA devices, cached per device index (like `gfx_arch`).

## E1 — MoE multi-row redirect + never-nullptr backstop

The fused-MoE FMA matmul is 16-row-only on Maxwell: `exl3_gemm_inner.cuh` has `static_assert(!EXL3_MAXWELL || TILEBLOCKS_M == 1)`, and the m32/m64 MoE tile getters (`comp_units/exl3_moe_inst_*_m32/m64.cu`) `return nullptr` under `#if EXL3_MAXWELL`. Multi-row tiles are declined and redirected, not built.

- **Python gate at both call sites** (the only two callers of `ext.exl3_moe`):
  - `exllamav3/modules/block_sparse_mlp.py` — `self.mtile_ok = MTILE and bool(self.multi_up.mul1) and not is_maxwell(device)`.
  - `exllamav3/model/moe_cpu_host.py` — `maxwell = is_maxwell(y.device)` computed once above the slot-group loop; the tier condition is `TUNING.mtile and not maxwell and (t1 or t2)`, else the `[(n_fused, 1, fused_t, 16)]` 16-row fallback.
- **C++ never-nullptr backstop** in `exllamav3_ext/quant/exl3_moe.cu`: `TORCH_CHECK(kernel != nullptr, ...)` immediately before the `moe_kernel_attr_set`/`cudaFuncSetAttribute` launch-setup block, after both the ROCm and CUDA dispatch branches assign `kernel`. Defense-in-depth — with the Python gate the null is unreachable, but a caller that bypasses the gate fails here instead of launching a null kernel.

On Maxwell, `exl3_moe` is never called with `m_tile > 16`: experts with ≤128 rows (the fused tier's row capacity, `MOE_FUSED_ROWS`) run a single 16-row FMA launch; experts with >128 rows route to the batched-reconstruct / per-expert reconstruct path (E5 provides its GEMM).

## E2 — MoE-coop decode backstop

The coop selectors `moe_coop_kernel_a`/`_b` (wrappers at `exl3_moe_coop.cu:20/47`) always return a real `MoeCoopKernel`: every `exl3_moe_coop_kernel_a_*`/`_b_*` in `exl3_moe_coop_instances.cuh` returns `{ (void*) exl3_moe_coop_a_kernel<K, cb, wide>, smem }` (or the `_b_kernel` form) — no nullptr branch on any CC. The coop `gemv_tile` (`exl3_moe_coop_kernel.cuh:341`) carries an `if constexpr (EXL3_MAXWELL)` fp32-FMA path (fragment gather, cross-warp reduction), so on Maxwell it is a plain fp32 per-row FMA with no `cp.async`/`mma` dependency.

A symmetric defense-in-depth backstop sits immediately after the two selector lines (`exl3_moe_coop.cu:141-142`): `TORCH_CHECK(ka.kernel != nullptr, ...)` and the `(b)` counterpart, so a future selector that declines fails cleanly instead of launching a null kernel. Pass-through on Maxwell today.

Coverage: `tests/test_moe_coop.py` (existing, GPU; exercises the coop fp32-FMA kernel and the E5 FP32-SGEMM path) runs the backstop as a pass-through.

## E3 — routing (router) fallback

Routing dispatch in `routing.cu`:
- bsz 1 → `routing_gemv_kernel` (CUDA-core fp32 FMA GEMV, safe on every arch) launched at `routing.cu:282`.
- bsz > 1 → `hgemm(hidden, gate, scores)` fallback at `routing.cu:294`, which on Maxwell runs the E5 FP32-SGEMM path (`hgemm.cu`).
- The int8 TC `routing_gemm_det` is gated off: `routing_gemm_det_fits` (`routing_gemm.cu:285`) returns `false` for `cc_major < 8`, so that branch is never taken on Maxwell.

## E4 — HC mix: clean-decline CC gate

`hc_mix_tiled` is declined on Maxwell: the Python gate in `hyperconnections.py` sets `self.tiled` only for CC major `>= 8`, so Maxwell (CC 5) is never routed to `gr_mix_tiled`. The tiling kernels' TC primitives (`det_mma3`, `cp.async`, `ldmatrix` in `det_gemm.cuh`) are `#if __CUDA_ARCH__ >= 800` and silently no-op below 800 → silent garbage if launched on Maxwell.

A CUDA-side guard in the host entry `gr_mix_tiled` (`hc_mix_tiled.cu`) reads the compute-capability major and `TORCH_CHECK(cc_major >= 8, ...)` before the large dynamic-smem attribute / TC-kernel launches — a caller that bypasses the Python gate fails cleanly. The prefill non-tiled `else` branch in `hyperconnections.py` is untouched — its FP16 `torch.matmul` dispatches to `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F)`, supported on Maxwell.

## E5 — reconstruct → cuBLAS SGEMM

Two explicit FP32 GEMM helpers in `exllamav3_ext/hgemm.cu`, gated on `DevCtx::instance().get_cc(device) == CC_MAXWELL` and wrapped in `#if !defined(USE_ROCM)`. On Maxwell the reconstruct GEMM does not depend on the mixed-precision 16F-in / C32-accumulate cuBLAS behavior:

- **`hgemm_sgemm_impl(a, b, c, stream)`** — dense. Casts `a`, `b` to FP32; `c` must be 2D (`TORCH_CHECK(c.dim() == 2)` — a 3D `c` routes to `hgemm_batched`). Allocates an FP32 output laid out to match `c`'s row stride (`at::empty({size_m, c_stride_m}, fp32).narrow(1, 0, size_n)`), calls `cublasSgemm` with the same operand roles / dims / leading dims / transpose as the `cublasGemmEx` it mirrors, then `c.copy_(cf)` (downcast for an FP16 `c`, strided for a non-contiguous `c`).
- **`hgemm_batched_sgemm_impl(a, w, c, stream)`** — batched. Same idea; `cublasGemmStridedBatchedEx` with `CUDA_R_32F` + `CUBLAS_COMPUTE_32F`, FP32-cast inputs, identical dims / leading dims / batch strides to the call it mirrors, then `c.copy_(cf)`.

Both — and the dense/batched GemmEx sites — share one `cublas_setup(stream, &alpha_, &beta_)` helper (current handle + stream + host pointer mode + per-device workspace from `DevCtx::instance().get_ws(device)` + alpha=1.0/beta=0.0); all four cuBLAS sites in the file call it.

Dispatch on Maxwell:
- Dense prefill: `exl3.py:reconstruct_hgemm` → `ext.hgemm_recon` → C++ `hgemm_recon` (`hgemm_f16acc.cu`) → `hgemm_f16acc_try` (false on Maxwell: `f16acc::covered` requires `major >= 8`) → `hgemm` → `hgemm_gr` → `hgemm_gemmex_impl`, which calls `hgemm_sgemm_impl` and returns under the Maxwell gate.
- Batched-expert MoE: `moe_batch_recon.py` → `ext.hgemm_batched` → `hgemm_batched`, which calls `hgemm_batched_sgemm_impl` and returns under the Maxwell gate (after the f16acc try, false on Maxwell).

The existing `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F)` path also works on Maxwell CC 5.0/5.2 (16F-in / 32F-accumulate is supported; only FP16 *compute* / `COMPUTE_16F` needs CC 5.3+), but the explicit `cublasSgemm` helpers are the guaranteed-correct path per the workstream requirement. The general `hgemm`/`hgemm_batched` are untouched on non-Maxwell devices. `reconstruct.cu` is unchanged (still emits FP16 weights); the FP16→FP32 cast lives in the GEMM layer.

## Files touched

- **E1:** `exllamav3/util/backend.py` (`is_maxwell`), `exllamav3/modules/block_sparse_mlp.py` (`mtile_ok` gate), `exllamav3/model/moe_cpu_host.py` (tier gate), `exllamav3/exllamav3_ext/quant/exl3_moe.cu` (never-nullptr `TORCH_CHECK`).
- **E2:** `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu` (`TORCH_CHECK` backstop after the selectors).
- **E3:** `exllamav3/exllamav3_ext/routing.cu` (comment at the `hgemm` fallback).
- **E4:** `exllamav3/exllamav3_ext/hc_mix_tiled.cu` (clean-decline `cc_major >= 8` guard in `gr_mix_tiled`).
- **E5:** `exllamav3/exllamav3_ext/hgemm.cu` (explicit `cublasSgemm` reconstruct path + shared `cublas_setup`).

## Compile plan

Incremental, no GPU workloads: `TORCH_CUDA_ARCH_LIST="5.0;5.2" python setup.py build_ext --inplace` from the worktree root (do not `rm -rf build`). Confirm it compiles for sm_50 + sm_52.
