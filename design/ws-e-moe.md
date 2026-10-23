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

## E2 — MoE-coop decode confirm

Confirm the cooperative MoE decode kernel (`exl3_moe_coop`) uses the fp32 FMA `gemv_tile` (single-row path) with real weight pointers on Maxwell, and add a decode test. The decode (bsz 1) path is a plain fp32 FMA gemv launch, co-residency-bounded — confirm no wide-row / TC dependency is exercised.

## E3 — routing (router) confirm

Confirm the router GEMM runs on Maxwell: bsz 1 through the fp32 FMA GEMV path, bsz > 1 through `hgemm` (cublas, E5 path), and the `det_gemm` (deterministic) branch gated on `cc < 8` so it is not taken on the non-relevant arches. No kernel change expected — confirm the dispatch picks a working path.

## E4 — HC mix: clean-decline CC gate

`hc_mix_tiled` (HC mix tiling) is declined on Maxwell: the tiling gate uses `>= 8` (CC major 8+), so Maxwell (CC 5) takes the non-tiled path. The prefill non-tiled `else` branch already uses `torch.matmul`, which is correct on Maxwell via `CUBLAS_COMPUTE_32F` — **no change** needed there. Add the clean-decline gate so the tiling path is never entered on CC 5.

## E5 — reconstruct → cuBLAS SGEMM (guaranteed-correct Maxwell path)

Add an explicit `cublasSgemm` path to `exllamav3_ext/hgemm.cu`, gated on `CC_MAXWELL`, so the reconstruct GEMM on Maxwell is guaranteed-correct and matches the GTX 16xx reference, rather than relying on the mixed-precision path.

**Nuance (kept deliberately):** the existing `hgemm`/`hgemm_batched` use `cublasGemmEx(CUDA_R_16F, CUBLAS_COMPUTE_32F)`, which **is** supported on Maxwell CC 5.0/5.2 (CUDA 9.0 release notes: "Maxwell or later"). What is unsupported on Maxwell is FP16 *compute* (`COMPUTE_16F` / `cublasHgemm`, needs CC 5.3+). We do not switch the general `hgemm` to it; we add the explicit `cublasSgemm` reconstruct path per the workstream requirement so this path is guaranteed-correct by construction on Maxwell.

## Files touched per sub-task

- **E1:** `exllamav3/util/backend.py` (new `is_maxwell`), `exllamav3/modules/block_sparse_mlp.py` (import + `mtile_ok` gate), `exllamav3/model/moe_cpu_host.py` (import + tier gate), `exllamav3/exllamav3_ext/quant/exl3_moe.cu` (never-nullptr `TORCH_CHECK`).
- **E2:** `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu` (confirm only) + a decode test.
- **E3:** router dispatch (confirm only) — `exllamav3/modules/block_sparse_mlp_routing.py` / the `det_gemm` branch.
- **E4:** `exllamav3/exllamav3_ext/hc_mix_tiled.cu` (+ `hc_fuse.cuh`) — clean-decline `>= 8` gate.
- **E5:** `exllamav3/exllamav3_ext/hgemm.cu` — explicit `cublasSgemm` reconstruct path gated on `CC_MAXWELL`.

## Compile plan

Incremental, no GPU workloads: `TORCH_CUDA_ARCH_LIST="5.0;5.2" python setup.py build_ext --inplace` from the worktree root after each C++ sub-task (do not `rm -rf build`). E1's only C++ change is the `TORCH_CHECK` in `exl3_moe.cu`; the Python gates do not affect the build. Confirm it compiles for sm_50 + sm_52.
