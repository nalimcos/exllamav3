# Workstream F — Tensor-Parallel Sync Layer (Maxwell TP=4)

## Goal
Make the native TP=4 collectives (all_reduce, broadcast, gather) run correctly and efficiently on Maxwell (Tesla M10: 4× GM107, CC 5.0 / sm_50, PCIe Gen3 x16, no NVLink). Workstream A already made `parallel/*` COMPILE for sm_50/52 (PTX-compat primitives in `ptx.cuh`/`ptx_portable.cuh`). F makes them actually WORK and be efficient on top of that substrate.

## Substrate (verified inventory)
Cooperative launches: 8 `cudaLaunchCooperativeKernel` in the tree; all `parallel/*` collectives launch cooperatively, unconditional, no Maxwell fallback.
- `pg_all_reduce_kernel` (all_reduce.cu:285) — **DEAD** (only Python ref commented out, model_tp_backend.py:474). 2 blocks, per-round `grid.sync()` in a loop.
- `pg_all_reduce_cpu_kernel` (all_reduce_cpu.cu:633/645/657) — HOT (prefill MoE reduce). Grid = 2×CPUREDUCE_MB_BLOCKS = 8 blocks × 1024 threads. **No `grid.sync`** — coop used ONLY for the co-residency guarantee (all 8 blocks live for the elastic CPU ring).
- `pg_gather_kernel` (gather.cu:467) — HOT. grid = num_blocks (num_ranks on the consumer device, 1 on producers). ONE `grid.sync()` just before `pg_barrier_inner`.
- `pg_gather_small_kernel` (gather.cu:388) — HOT. 1 block, no `grid.sync`.
- broadcast.cu, barrier.cu: plain `<<<1,...>>>`, no coop.

Reusable non-coop primitives:
- `exl3_launch_grid_bounded` (coop_autotune.cuh) — bounded plain `cudaLaunchKernel` to `maxb×num_sms`.
- `exl3_gemm_launch` (coop_autotune.cuh) — host arch dispatch: `CC_MAXWELL` → bounded plain, else `cudaLaunchCooperativeKernel`. This is the launch entry the collectives should call.
- `group_barrier(group_id, group_size, int* counters_sense)` (ptx.cuh) — sense-reversing inter-block barrier: `__threadfence` → `atomicAdd(counter,1)` (plain/unscoped, sm_50-safe) → last block resets counter=0 and flips sense; others spin on sense via `exl3_sleep`. Counter layout: `counters_sense[group_id*2]`=counter, `[group_id*2+1]`=sense. GEMM uses the per-device `locks` buffer at `EXL3_GRID_BARRIER_OFFSET`.

Maxwell tier = `CC_MAXWELL` (`prop.major==5`, covers sm_50 AND sm_52). Host detect: `DevCtx::instance().get_cc(device)==CC_MAXWELL` (or `g_get_cc`).

ptx.cuh Maxwell branches (pre-F): `stg_wt_u32/u128` = plain store + `__threadfence` (device scope); `ldg_cv_u32/u128` = **plain (non-volatile) load + `__threadfence`** (stale-L1 hazard); `ldg_acquire_sys_u32/u64` = volatile load + `__threadfence_system`; `stg_release_sys_u32/u64` = `__threadfence_system` + volatile store; `globaltimer_ns` = `clock64` (per-SM, best-effort watchdog only); `exl3_sleep` = 32-iteration volatile loop (ns ignored).

## Risks found (cross-reviewed)
- **R1 (coop launch):** all 4 coop kernels fail to launch at runtime on sm_50. De-cooperativize.
- **R2 (stale read):** `ldg_cv_*` Maxwell branch = plain load + fence. `barrier_inner.cuh` epoch spin reads cross-device flags (`barrier_epoch_device[]`) from pinned host memory shared across 4 GPUs + CPU over PCIe; a plain load can be served stale from L1 → the epoch barrier can spin forever. Fix at the primitive.
- **R3 (under-scoped write):** `stg_wt_*` (used by barrier_inner's arrival publish + epoch bump) is device-scoped (`__threadfence`); the cross-GPU publish should be sys-scoped to be correct-by-construction.
- **R4 (co-residency):** de-cooperativized kernels that need all blocks live (all_reduce_cpu's elastic ring; gather's `group_barrier`) MUST be co-resident or they deadlock. Guarantee via bounded launch + a `TORCH_CHECK` (no silent clamp).
- **R5 (barrier buffer sharing — P0 hang):** PGContext is ONE shared registered region mapped by all 4 GPUs. A single shared barrier counter would be atomically contended by all 4 GPUs at once → corrupt → guaranteed hang on every gather. The `group_barrier` buffer MUST be per-GPU.
- **R6 (MoE memop):** stream-memop path is runtime-selected; on sm_50 `cuStreamWaitValue32` → NOT_SUPPORTED → kernel fallback. Make it deterministic on Maxwell.
- **R7 (`__grid_constant__` in gather.cu):** CC7-only kernel-param qualifier, unguarded. Drop it (keep `all_offsets` by value).
- **R8 (watchdog):** `globaltimer_ns` = `clock64` (per-SM cycles). The `sync_deadline()` multiplier is ns-calibrated; on Maxwell the effective wall-time timeout is still tens of seconds (≫ ms PCIe transfers), so no false-positive bricking in practice. Best-effort limitation, documented, no code change.

## Design decisions
1. **De-cooperativize the live cooperative kernels**, arch-gated so non-Maxwell keeps `cudaLaunchCooperativeKernel` (byte-identical) and Maxwell uses the bounded plain launch:
   - `pg_gather_small_kernel` → plain `<<<1, 256>>>` (trivial; single block, no barrier).
   - `pg_all_reduce_cpu_kernel` → `exl3_gemm_launch` (arch-gated bounded plain; NO `group_barrier` — it has no `grid.sync`). Add a `TORCH_CHECK` that the requested grid (8 blocks) ≤ `maxb×num_sms`, computing `maxb` with the SAME `cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, block, smem)` on the SAME templated kernel pointer, BEFORE the launch (belt-and-suspenders against the helper's internal clamp launching the wrong block count).
   - `pg_gather_kernel` → `exl3_gemm_launch` + replace the ONE `grid.sync()` with `group_barrier(0, gridDim.x, ctx->grid_barrier[this_device])` (per-GPU buffer), KEEPING the existing "barrier, then check abort" order (a timed-out block must still reach the barrier so co-resident peers don't deadlock). `TORCH_CHECK` co-residency. Drop `__grid_constant__`, keep `const Offsets all_offsets` by value.
   - broadcast.cu / barrier.cu: **no change** (already plain, no coop).
   - **`pg_all_reduce_kernel` (dead): OUT OF SCOPE.** It is never launched (dead Python path), so leaving it cooperative creates no runtime failure. De-cooperativizing it adds maintenance surface + per-round `group_barrier` complexity for zero runtime benefit; it is left as-is and documented as dead.
2. **Per-GPU barrier buffer (R5):** add to PGContext a per-device barrier state, `alignas(64) uint32_t grid_barrier[MAX_DEVICES][2]` (or an equivalent `{counter, sense}` per device), initialized to 0 in `pg_init_context`. Each GPU's `group_barrier` uses its own slot (index `this_device`), consistent with what `grid.sync` does implicitly per-GPU.
3. **Primitive ordering fixes (R2/R3), in ptx.cuh:**
   - `ldg_cv_u32/u128` Maxwell branch: plain load + `__threadfence` → **volatile load + `__threadfence_system`** (real acquire, sys scope — the flags are shared across multiple GPUs + CPU over PCIe). This makes it converge with `ldg_acquire_sys_*` on Maxwell (intended; the names/semantics stay distinct for the general sm_70+ path).
   - `stg_wt_u32/u128` Maxwell branch: plain store + `__threadfence` → **plain store + `__threadfence_system`** (release, sys scope) for the cross-GPU publish/epoch-bump path.
4. **MoE CPU-offload (R6):** on `CC_MAXWELL` (host `DevCtx::get_cc(device)`, device-guarded), FORCE the kernel fallback path deterministically — skip the `cuStreamWaitValue32/WriteValue32` dlopen probe (no reliance on the `NVreg_EnableStreamMemOPs` driver knob). The kernel fallback spins with `ldg_acquire_sys_u32` + `exl3_sleep` and sets `*abort_flag` on a 30 s timeout; verify its acquire/release ordering is correct. (Non-Maxwell keeps the memop-first behavior.)
5. **Efficiency / per-layer stall:** the decode (single-chunk ≤128 KB) path is ALREADY two plain stream-ordered kernels (send + recv), CUDA-graph-capturable → the primary per-token path, no change. The prefill/multi-chunk striped 8-block kernel is de-cooperativized to a co-resident bounded launch so the elastic CPU pipeline does not stall. The per-layer stall (2 PCIe traversals + CPU round-trip + CPU spin) is inherent to the host-pinned zero-copy protocol and cannot be removed without P2P/NVLink (unavailable on Maxwell). Minimize, don't eliminate: keep decode graph-capturable, keep the CPU AVX-sum overlapping PCIe (existing elastic ring), keep co-residency so no extra stalls. No minimum-GPU-count or NVLink assumption — TP=4 is architecturally fine once the sync substrate is correct.

## Invariant (documented)
`group_barrier` has no timeout; its safety net is the co-residency bound + stream ordering. The de-cooperativized collectives launch the grid bounded to `maxb×num_sms` (guaranteeing all blocks are scheduled/resident) in the same stream; a co-residency violation would be a silent hang, so the `TORCH_CHECK` fails loudly instead.

## Files touched
- `exllamav3_ext/ptx.cuh` — `ldg_cv_*` + `stg_wt_*` Maxwell branches (R2/R3).
- `exllamav3_ext/parallel/context.cuh` + `context.cu` — per-GPU `grid_barrier` field + init (R5).
- `exllamav3_ext/parallel/all_reduce_cpu.cu` — de-coop `pg_all_reduce_cpu_kernel` (arch-gated launch, co-residency check); drop vestigial `cooperative_groups` include if unused.
- `exllamav3_ext/parallel/gather.cu` — de-coop both gather kernels (`group_barrier` on the multi-block one; plain on the small one); drop `__grid_constant__`; co-residency check.
- `exllamav3_ext/quant/coop_autotune.cuh` — rename `exl3_gemm_launch` to a generic arch-aware name (now used by GEMM + collectives); keep behavior identical.
- `exllamav3_ext/cpu/moe_handoff.cu` — Maxwell → kernel-path gate (R6).
- `design/ws-f-tp-sync.md` — this doc.

## Coder task split (one at a time, compile-check sm_50+sm_52 after each)
- **T1a** — ptx.cuh `ldg_cv_*` + `stg_wt_*` Maxwell primitive fixes.
- **T1b** — PGContext per-GPU `grid_barrier` field + init.
- **T2** — de-coop `pg_all_reduce_cpu_kernel` (arch-gated launch, co-residency check); rename `exl3_gemm_launch` tree-wide; drop vestigial `cooperative_groups` include.
- **T3** — de-coop gather (both kernels + `group_barrier` on the multi-block one + drop `__grid_constant__` + co-residency check).
- **T4** — MoE-offload Maxwell kernel-path gate.

## Correctness review requirement
After coding: an unbriefed Reviewer (complexity/clarity) AND an Architect correctness review of: the volatile+`__threadfence_system` acquire/release ordering in each handoff; the de-cooperativized barrier (per-GPU buffer, `group_barrier` re-entrancy + reset + sense flip); the co-residency bound (no silent clamp); the MoE-offload fallback ordering. Fix any issues and re-review.

## Definition of done
- Live TP=4 collectives (all_reduce, broadcast, gather) work on Maxwell: cooperative kernels de-cooperativized (no runtime cooperative launch on the live paths), sync-primitive ordering correct, MoE-offload ported/gated.
- Whole library still compiles for sm_50 + sm_52 (incremental build, no GPU).
- Per-layer stall reasoned about / minimized.
- Design doc committed; unbriefed Reviewer + Architect correctness review done.
