# Workstream E — MoE / routing / HC + reconstruct (Maxwell port)

**Target:** Maxwell M40 (CC 5.2, 24 SM, 288 GB/s) / M10 (CC 5.0, 5 SM, 83 GB/s). No tensor cores, no `dp4a`, no native fp16 arithmetic (`__hfma2`/`hexp` are sm_53+), no `cp.async`; all MoE/HC compute is fp32 FMA.

Worktree: `exllamav3-wt-maxwell` (branch `maxwell`). Builds on Workstream B (fp32 dequant core + dense GEMM), C (GEMV/decode), and F (TP sync + MoE CPU-offload kernel path). Scope: make the routed-expert (MoE) path, the routing (router) GEMM, the hyperconnection (HC) mix, and the reconstruct (dequant→GEMM) path all run correctly on Maxwell.

## Hardware & arch boundary

- **Device tier (C++):** `EXL3_MAXWELL` (`__CUDA_ARCH__ < 600`, `exllamav3_ext/arch.cuh`).
- **Host tier (C++):** `CC_MAXWELL` (`prop.major == 5`, `exllamav3_ext/quant/exl3_devctx.cuh`); detect via `DevCtx::instance().get_cc(device) == CC_MAXWELL`.
- **Python tier:** `is_maxwell(device)` in `exllamav3/util/backend.py` — `torch.cuda.get_device_capability(device)[0] == 5` (matches `CC_MAXWELL`; NOT `< 6`), `False` for non-CUDA devices.

## E1 — MoE multi-row gate/redirect (this task)

The fused-MoE FMA matmul is **16-row-only**: `exl3_gemm_inner.cuh` has `static_assert(!EXL3_MAXWELL || TILEBLOCKS_M == 1)`, and the m32/m64 MoE tile getters (`comp_units/exl3_moe_inst_*_m32/m64.cu`) each `return nullptr` under `#if EXL3_MAXWELL`. This is **not** a new multi-row FMA kernel — the multi-row tiles are declined on Maxwell and redirected:

- **Python gate at both call sites** (the only two callers of `ext.exl3_moe`):
  - `exllamav3/modules/block_sparse_mlp.py` — `self.mtile_ok = MTILE and bool(self.multi_up.mul1) and not is_maxwell(device)`.
  - `exllamav3/model/moe_cpu_host.py` — inline tier `TUNING.mtile and not is_maxwell(y.device) and (t1 or t2)`, keeping the `else [(n_fused, 1, fused_t, 16)]` 16-row fallback.
  The two are left in their existing distinct forms (one stored `mtile_ok` with a mul1 check, one inline `TUNING.mtile` condition) — not unified.
- **C++ never-nullptr backstop** in `exllamav3_ext/quant/exl3_moe.cu`: a `TORCH_CHECK(kernel != nullptr, ...)` immediately before the `moe_kernel_attr_set`/`cudaFuncSetAttribute` launch-setup block, after both the ROCm and CUDA dispatch branches assign `kernel`. Pure defense-in-depth — with the Python gate the null is unreachable, but a future caller that bypasses the gate fails cleanly instead of launching a null kernel at `cudaFuncSetAttribute`/`cudaLaunchKernel`.

**Net effect on Maxwell:** `exl3_moe` is never called with `m_tile > 16`. Experts with ≤128 rows (the fused tier's row capacity, `MOE_FUSED_ROWS`) run a single 16-row FMA launch; experts with >128 rows route to the batched-reconstruct / per-expert reconstruct path. The GEMM side of reconstruct is a separate workstream item (E5); this task does not touch `hgemm.cu`.

## E2 — MoE-coop decode confirm + backstop

**Confirmed** (`exl3_moe_coop.cu` + `comp_units/exl3_moe_coop_instances.cuh` + `exl3_moe_coop_kernel.cuh`):
- The selectors `moe_coop_kernel_a`/`_b` (wrappers at `exl3_moe_coop.cu:20/47`) always return a real `MoeCoopKernel`: every `exl3_moe_coop_kernel_a_*`/`_b_*` in `exl3_moe_coop_instances.cuh` returns `{ (void*) exl3_moe_coop_a_kernel<K, cb, wide>, smem }` (or the `_b_kernel` form) — there is no nullptr branch, on any CC. (The multi-row `exl3_moe_inst_*` decline is a different kernel family; the coop `gemv_tile` is instantiated unconditionally.)
- `gemv_tile` (`exl3_moe_coop_kernel.cuh:341`) carries an `if constexpr (EXL3_MAXWELL)` fp32-FMA path (fragment gather at :466/:490, cross-warp reduction at :583): on Maxwell it is a plain fp32 per-row FMA with no `cp.async`/`mma` dependency. The launch (`cudaLaunchKernel(ka.kernel, …)` at :159) receives these real pointers.

**Change:** symmetric defense-in-depth backstop immediately after the two kernel-selection lines (`exl3_moe_coop.cu:141-142`): `TORCH_CHECK(ka.kernel != nullptr, "exl3_moe_coop: no kernel selected (a)")` and the `(b)` counterpart, so a future selector that does decline fails cleanly instead of launching a null kernel.

**Test:** `tests/test_moe_coop.py` is the natural decode-MoE / `exl3_moe_coop` coverage — end-to-end parity of every K 1..8 × codebook, mixed K, all activations, biases, padded dims, expert-range masking, shared-expert merge and fp32 gate/up scratch against a torch reference built from reconstructed weights. It is GPU-bound (exercises `ext.reconstruct`/`ext.hgemm`, the E5 FP32-SGEMM path, and the coop fp32-FMA kernel) and is **STAGED for the Maxwell target box**; it runs the new backstop as a pass-through. No new test added (it would be redundant and GPU-only, and would not run in this no-GPU worktree).

## E3 — routing (router) confirm + comment

**Confirmed** (`routing.cu` + `routing_gemm.cu`):
- bsz 1 → `routing_gemv_kernel` (CUDA-core fp32 FMA GEMV, safe on every arch) at `routing.cu:282`.
- bsz > 1 → `hgemm(hidden, gate, scores)` fallback at `routing.cu:294`, which on Maxwell runs the E5 FP32-SGEMM path (`hgemm.cu`).
- The int8 TC `routing_gemm_det` is cleanly skipped: `routing_gemm_det_fits` (`routing_gemm.cu:285`) returns `false` for `cc_major < 8`, so the `routing_gemm_det_` branch is never taken on Maxwell.

**Change:** a two-line comment at the `hgemm` fallback noting that pre-CC7 (Maxwell) devices reach this cuBLAS path (now FP32-SGEMM via `hgemm.cu`) and that the TC det_gemm path is gated off in `routing_gemm.cu`. Comment only — no kernel change.

## E4 — HC mix: clean-decline CC gate

`hc_mix_tiled` (HC mix tiling) is declined on Maxwell: the Python gate `hyperconnections.py` sets `self.tiled` only for CC major `>= 8`, so Maxwell (CC 5) is never routed to `gr_mix_tiled`. The TC primitives in the tiling kernels (`det_mma3`, `cp.async`, `ldmatrix` in `det_gemm.cuh`) are `#if __CUDA_ARCH__ >= 800` and silently no-op below 800 → silent garbage if launched on Maxwell.

**Change:** a single minimal CUDA-side defense-in-depth guard in the host entry `gr_mix_tiled` (`hc_mix_tiled.cu`), after the stream is in scope and before the large dynamic-smem attribute / TC-kernel launches: read the compute-capability major and `TORCH_CHECK(cc_major >= 8, "gr_mix_tiled: needs sm_80+ (cp.async + mma.m16n8k32 s8); pre-Ampere (e.g. Maxwell) devices take the CUDA-core gr_mix / cuBLAS path")`. The prefill non-tiled `else` branch in `hyperconnections.py` is untouched — its FP16 `torch.matmul` dispatches to `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F)`, supported on Maxwell.

## E5 — reconstruct → cuBLAS SGEMM (guaranteed-correct Maxwell path)

Two explicit FP32 GEMM helpers in `exllamav3_ext/hgemm.cu`, gated on `DevCtx::instance().get_cc(device) == CC_MAXWELL` and wrapped in `#if !defined(USE_ROCM)`, so the reconstruct GEMM on Maxwell does not depend on the mixed-precision 16F-in / C32-accumulate cuBLAS behavior (matches the GTX 16xx `hgemm_recon_fp32` reference).

- **`hgemm_sgemm_impl(a, b, c, stream)`** — dense. Casts `a`, `b` to FP32 (`to(at::kFloat)`), allocates an FP32 output laid out to match `c`'s row stride (`at::empty({size_m, c_stride_m}, fp32).narrow(1, 0, size_n)`), calls `cublasSgemm` with the SAME operand roles / dims / leading dims / transpose as the `cublasGemmEx` it mirrors (`CUBLAS_OP_N` both; m/n/k = `size_n, size_m, size_k`; b ld `size_n`, a ld `size_k`, c ld `c_stride_m`), then `c.copy_(cf)` (downcast for an FP16 `c`, strided for a non-contiguous `c`).
- **`hgemm_batched_sgemm_impl(a, w, c, stream)`** — batched. Same idea; `cublasGemmStridedBatchedEx` with `CUDA_R_32F` + `CUBLAS_COMPUTE_32F`, FP32-cast inputs, identical dims / leading dims / batch strides to the `cublasGemmStridedBatchedEx` it mirrors, then `c.copy_(cf)`.

**Dispatch (hot path confirmed for both cases on Maxwell):**
- Dense prefill: `exl3.py:reconstruct_hgemm` → `ext.hgemm_recon` → C++ `hgemm_recon` (`hgemm_f16acc.cu`) → `hgemm_f16acc_try` (false on Maxwell: `f16acc::covered` requires `major >= 8`) → `hgemm` → `hgemm_gr` → **`hgemm_gemmex_impl`**, which now calls `hgemm_sgemm_impl` and returns under the Maxwell gate.
- Batched-expert MoE: `moe_batch_recon.py` → `ext.hgemm_batched` → **`hgemm_batched`**, which now calls `hgemm_batched_sgemm_impl` and returns under the Maxwell gate (after the f16acc try, false on Maxwell).

Both gates sit after the existing shape/contiguity checks; the existing `cublasGemmEx` path is untouched on non-Maxwell devices. `reconstruct.cu` is unchanged (still emits FP16 weights); the FP16→FP32 cast lives in the GEMM layer.

**Nuance (kept deliberately):** the existing `hgemm`/`hgemm_batched` use `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F)`, which **is** supported on Maxwell CC 5.0/5.2 (CUDA 9.0 release notes: "Maxwell or later"). What is unsupported on Maxwell is FP16 *compute* (`COMPUTE_16F` / `cublasHgemm`, needs CC 5.3+). We do not switch the general `hgemm` to it; we add the explicit `cublasSgemm` reconstruct path per the workstream requirement so this path is guaranteed-correct by construction on Maxwell.

## Files touched per sub-task

- **E1:** `exllamav3/util/backend.py` (new `is_maxwell`), `exllamav3/modules/block_sparse_mlp.py` (import + `mtile_ok` gate), `exllamav3/model/moe_cpu_host.py` (import + tier gate), `exllamav3/exllamav3_ext/quant/exl3_moe.cu` (never-nullptr `TORCH_CHECK`).
- **E2:** `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu` (confirm + `TORCH_CHECK` backstop after the selectors). Test: `tests/test_moe_coop.py` (existing, GPU, STAGED for the target box).
- **E3:** `exllamav3/exllamav3_ext/routing.cu` (confirm + a two-line comment at the `hgemm` fallback).
- **E4:** `exllamav3/exllamav3_ext/hc_mix_tiled.cu` — clean-decline `cc_major >= 8` guard in `gr_mix_tiled`.
- **E5:** `exllamav3/exllamav3_ext/hgemm.cu` — explicit `cublasSgemm` reconstruct path gated on `CC_MAXWELL`.

## Compile plan

Incremental, no GPU workloads: `TORCH_CUDA_ARCH_LIST="5.0;5.2" python setup.py build_ext --inplace` from the worktree root after each C++ sub-task (do not `rm -rf build`). E1's only C++ change is the `TORCH_CHECK` in `exl3_moe.cu`; the Python gates do not affect the build. Confirm it compiles for sm_50 + sm_52.
