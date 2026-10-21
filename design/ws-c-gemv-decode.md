# Workstream C — GEMV / decode (Maxwell, fp32)

Target: Tesla M40 (CC 5.2) / M10 (CC 5.0). No tensor cores, no dp4a, no native fp16 arithmetic, no cp.async, no cooperative launch, no gpu-scope ordered atomics. All compute is fp32 FMA. Worktree: `exllamav3-wt-maxwell` (branch `maxwell`). Builds on Workstream A (PTX-compat substrate) and Workstream B (procedural fp32 dequant core + dense GEMM matmul-swap). This doc is the design for the EXL3 GEMV (decode), the SK GEMV, the int8-GEMV coverage note, and the shared primitive that also covers MoE-coop.

## 0. Scope and definition of done

- EXL3 GEMV (decode) runs on Maxwell in fp32 for m==1 (MMODE 0) and m=2..8 (MMODE 1), NO tensor cores / dp4a / cp.async, using the Workstream B fp32 dequant core.
- SK GEMV ported + re-gated (no latent `__trap`).
- int8-GEMV skip documented (K=1 / residual-mode coverage gap).
- Shared mma→FMA primitive covers MoE-coop `gemv_tile`.
- GEMV synchronization via A's `exl3_launch_grid_bounded` + `group_barrier`.
- Whole library still compiles for sm_50 + sm_52 (incremental build, no GPU).
- Design doc committed; an unbriefed Reviewer reviews the coding task.

## 1. The core decision: matmul-swap HYBRID (not a pure swap, not a full port)

The existing worktree GEMV (`quant/exl3_gemv_kernel.cuh`) is the **upstream MMA version**: it calls `exl3_gemv_ns::mma_ab_h` (an fp16 tensor-core `mma.sync`), whose `EXL3_MAXWELL` branch is an **empty stub** (the kernel computes nothing on Maxwell today). The gtx16-full reference has a separate `CORE` branch (a direct fp16 `__hfma2` CUDA-core inner loop).

A "pure" matmul-swap (just replace `mma_ab_h` with a scalar FMA, keep the existing FOLD/sh_red) is **INCORRECT**: the mma path and the scalar-FMA path use **different accumulator layouts**.

- **mma layout (existing worktree):** `ch[t][f][0]` = two *adjacent* N-columns, each *fully reduced over k*. Lane `l&3` picks the column pair; the k-sum is done inside the mma. Reduction writes `sh_red[warp][sr0][t*16+f*8+2*(l&3)]` and `+1` with **no shuffle**.
- **CORE / scalar-FMA layout (gtx16-full):** `ch[t][f][0]` = *one* N-column `g = l>>2`, split into even-k / odd-k partials. The full k-sum requires summing the 4 k-pair lanes of the group via `shfl_xor(1)` then `shfl_xor(2)`.

**Verdict (traced in source, lane 5, MMODE 0, WNT=1):** the CORE FMA is a **direct per-lane FMA with NO B-gather**. Each lane multiplies its own A fragment by its own B fragment (`f0`/`f1`) in place; the only cross-lane `shfl` in the compute path is the **accumulator reduction**. The dequant already emits `f0`/`f1` in the `mma.m16n8k16` B-fragment layout where lane `l` (`t=l&3`, `g=l>>2`) holds `B[2t][g], B[2t+1][g], B[2t+8][g], B[2t+9][g]` (f0) and the same for `g+8` (f1) — exactly the weights lane `l` needs for its own column at its own k-pair.

**Design:** modify the EXISTING worktree GEMV in place (keep its `__ldcs` prefetch ring, shuffle-window resolution, Hadamard in/out, grid-stride group loop, dispatch), and add an `EXL3_MAXWELL` (or a new `FMA` template param) path that:
1. Dequantizes B to **fp32** (Workstream B core: `decode_3inst_f<cb>`).
2. Does a **direct per-lane fp32 FMA** (CORE layout, no B-gather).
3. Uses the **CORE reduction** (`shfl_xor(1)` + `shfl_xor(2)`, lane `l&3==0` writes `sh_red[warp][r][t*16+g]` and `+8`).
4. Accumulates **directly into fp32** accumulators (drop the fp16 `ch` + FOLD machinery — fp32 accumulators have no fp16 error-growth concern).

The non-Maxwell (AMPERE/TC) path keeps the existing mma + fp16 dequant + mma reduction. Selected at compile time (template param or `if constexpr (EXL3_MAXWELL)`).

## 2. The fp32 FMA inner loop (exact)

Per lane `l` (`t = l&3`, `g = l>>2`), per warp-tile `t` (warp tile index 0..WNT-1), per k-slice `i`:

**A fragment (fp32).** A is stored as `half` in memory (`A2 = (const half2*) A`); load the half2 and convert to fp32.
- MMODE 0: **all lanes read row 0** (`a_row0 = 0`, `r0_ok = true`). `a_col = (ks0+i)*8 + (l&3)`.
  - `a01_x = A[0][2*(l&3)]`, `a01_y = A[0][2*(l&3)+1]` (from `A2[a_col]`)
  - `a23_x = A[0][8+2*(l&3)]`, `a23_y = A[0][8+2*(l&3)+1]` (from `A2[a_col+4]`)
- MMODE 1: per row `r` (0..ROWS-1, `ok = r < size_m`), same k-pair math with `A[r][...]`. Preload all rows' A fragments (`ar01[r]`, `ar23[r]`).

**B fragment (fp32)** — from the fp32 dequant (section 3): `frag0[0..3]`, `frag1[0..3]` (each `Vec<float,4>`):
- `frag0[0]=B[2t][g]`, `frag0[1]=B[2t+1][g]`, `frag0[2]=B[8+2t][g]`, `frag0[3]=B[8+2t+1][g]`
- `frag1[0]=B[2t][g+8]`, `frag1[1]=B[2t+1][g+8]`, `frag1[2]=B[8+2t][g+8]`, `frag1[3]=B[8+2t+1][g+8]`

**FMA (MMODE 0)** — accumulate directly into fp32 `acc0[t][f]` (`float2`), no fold:
```
acc0[t][0].x += a01_x * frag0[0];   acc0[t][0].y += a01_y * frag0[1];
acc0[t][0].x += a23_x * frag0[2];   acc0[t][0].y += a23_y * frag0[3];
acc0[t][1].x += a01_x * frag1[0];   acc0[t][1].y += a01_y * frag1[1];
acc0[t][1].x += a23_x * frag1[2];   acc0[t][1].y += a23_y * frag1[3];
```
`acc0[t][0]` = column `g` (x=even-k, y=odd-k); `acc0[t][1]` = column `g+8`.

**FMA (MMODE 1)** — same 8 FMA per row `r`, into `accm[t][f][r]` (`float2[WNT][2][ROWS]`), guarded by `ok = r < size_m`.

No B-gather. The A k-index equals the B k-index in every product (traced: lane 5 → k∈{2,3,10,11} for column g=1).

## 3. The fp32 dequant (reuse Workstream B core)

The GEMV's register dequant (`dq8_regs_4bits/2bits/3bits<cb>`, `dq8_regs_half`) builds 8 windows `w0..w7` from the shuffle-window words (funnel-shift / BFE / fshift). **This window extraction is unchanged** — it is bit-identical between fp16 and fp32.

Only the **decode** changes. Replace `decode8<cb>(w0..w7, f0, f1)` (4× `decode_3inst_2<cb>` → 4 half2) with **8× `decode_3inst_f<cb>(wN & 0xffff)`** into `Vec<float,4> frag0, frag1`:
```
frag0[0]=decode_3inst_f<cb>(w0&0xffff); frag0[1]=decode_3inst_f<cb>(w1&0xffff);
frag0[2]=decode_3inst_f<cb>(w2&0xffff); frag0[3]=decode_3inst_f<cb>(w3&0xffff);
frag1[0]=decode_3inst_f<cb>(w4&0xffff); frag1[1]=decode_3inst_f<cb>(w5&0xffff);
frag1[2]=decode_3inst_f<cb>(w6&0xffff); frag1[3]=decode_3inst_f<cb>(w7&0xffff);
```
This is exactly the decode body of `exl3_dq_dispatch_f32` (`quant/exl3_dq.cuh:188-195`). `decode_3inst_f<cb>` (`codebook.cuh:142-146`) runs the identical fp16 codebook math and upcasts → bit-identical to the half path. `cb==2` (mul1) uses the same `decode_3inst_f<2>` (Workstream B already replaced `__dp4a` with the portable byte-sum inside `decode_3inst<2>`).

**Implementation:** add fp32 sibling dequant functions (e.g. `dq8_regs_4bits_f32<cb>`, etc.) that keep the existing window extraction and emit `Vec<float,4> frag0/frag1`. Do NOT mutate the existing fp16 `dq8_regs_*` (MoE-coop's `gemv_tile` still needs them for the non-Maxwell path; on Maxwell the `gemv_tile` switches to the fp32 siblings — see section 7).

**Do NOT materialize dequantized weights** — the fp32 weights feed the FMA directly, in registers.

## 4. The reduction (CORE, shfl_xor)

MMODE 0, per warp-tile `t`:
```
float v0 = acc0[t][0].x + acc0[t][0].y;   // even-k + odd-k for column g
float v1 = acc0[t][1].x + acc0[t][1].y;   // for column g+8
v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
if ((lane & 3) == 0) {
    const int g = lane >> 2;
    sh_red[warp][0][t*16 + g]     = v0;
    sh_red[warp][0][t*16 + g + 8] = v1;
}
```
MMODE 1: identical per row `r`, writing `sh_red[warp][r][...]`.

The existing `sh_red[WK][ROWS][COLS]` shared buffer and the cross-warp sum (`for j<WK: sum += sh_red[j][r][c]`, then write C, `__syncthreads`) are **unchanged** — only the per-warp write changes from the mma layout to the CORE layout. `c_fp32` output (float) vs fp16 output (`__float2half_rn`) both supported as today.

## 5. GEMV synchronization (Workstream A primitives)

The GEMV kernel currently uses cooperative launch + two `grid.sync()` (after input Hadamard ~L245, before output Hadamard ~L459). On Maxwell cooperative launch is unavailable. Replace:
- **Launch:** `cudaLaunchCooperativeKernel` → `exl3_launch_grid_bounded` (1-D / dim3) on Maxwell. The GEMV grid is already capped at co-residency (`exl3_gemv.cu` ~L219-220), so the bound holds. Use the arch-aware `exl3_gemm_launch` / `EXL3_COOP_LAUNCH` macro (CC_MAXWELL → bounded plain launch; else cooperative).
- **Barrier:** the two `grid.sync()` → `EXL3_GRID_SYNC` macro (which is `group_barrier(0, gridDim.x, locks + EXL3_GRID_BARRIER_OFFSET)` on Maxwell, `grid.sync()` elsewhere). The GEMV args already carry `locks` (`EXL3_GEMM_ARGS`).

**Do NOT** use a bare volatile+fence spin — a bare spin deadlocks on an evicted block; the co-residency bound is mandatory (Workstream A design).

## 6. Host dispatch (exl3_gemv.cu)

- Remove the `if (exl3_no_tc(cc)) return false;` gate at L127 (it currently excludes Maxwell). On Maxwell the GEMV is now eligible. This affects **only** Maxwell (`exl3_no_tc` is `cc == CC_MAXWELL`); no other tier is touched, and the separate `has_su_sv` check stays.
- Instantiate the FMA variant of the kernel for Maxwell (template param / arch-aware selection in `exl3_gemv_select_kernel`).
- **Bitrate scope: the main GEMV covers 2–4 bpw only.** The maxwell worktree GEMV has **no** 5-bpw machinery (`static_assert bits==2||3||4` at L202; no `TSTRIDE`/`dq4`/2-load `ld_b`; dispatch declines `K>4` at L118). The 5–8 bpw path is the **SK GEMV's** job (§8). Do NOT port 5-bpw into the main GEMV (it is an absent gtx16-full feature, not "existing machinery").
- **Shape envelope must be re-tuned for Maxwell.** `exl3_gemv_cfg` (L47-75) is Ampere-tuned and declines most decode shapes (e.g. k=4096, n=4096, K=4 → `return -1` → falls back to the dense GEMM). The GEMV is the decode-speed lever the user wants, so re-tune the envelope (its `narrow_coresident`/occupancy differ on Maxwell) so the GEMV is actually selected for the m≤8 decode shapes it is designed for; otherwise the GEMV is dead code on the target. This is a must-fix for the performance goal, not a nice-to-have.
- Launch via `EXL3_COOP_LAUNCH` (bounded on Maxwell). The grid is already co-residency-capped (`exl3_gemv.cu` L169-170: `grid = MIN(size_n/cols, occupancy*num_sms)`), so the bound holds.
- **`EXL3_GRID_SYNC` reachability:** the macro is defined in `exl3_gemm_kernel.cuh`, which the GEMV kernel does **not** include. The Coder must either include it or inline the `group_barrier(0, gridDim.x, locks + EXL3_GRID_BARRIER_OFFSET)` call directly. The GEMV args carry `locks` (`EXL3_GEMM_ARGS`), so the barrier is reachable.

## 7. Shared primitive + MoE-coop (port once)

The shared unit is the **CORE fp32 FMA + fp32 dequant** in `exl3_gemv_ns` (`exl3_gemv_kernel.cuh`). `exl3_moe_coop_kernel.cuh`'s `gemv_tile<bits,cb,WIDE,HALF>` (a near-copy of the GEMV body, MMODE 1, CFG 0) hard-calls `exl3_gemv_ns::mma_ab_h` and `dq8_regs_*`.

**Important (Architect finding):** `gemv_tile` today uses the **MMA layout** — per-row A fragment (`a_row0 = rows[r0]*a_stride2`, `r0 = lane>>2`) and a **direct-write** reduction (`red[col] = acc0.x/.y`, `col = t*16+f*8+2*(l&3)`, row `r0`). The main GEMV's CORE path uses a **different** layout: all-rows A fragment (`ar01[r]`, `r=0..ROWS-1`) + `shfl_xor` reduction. A per-lane FMA cannot reproduce the MMA layout (the mma's full k-sum is a warp-level cross-lane combine that a scalar FMA recovers only via `shfl_xor`). So "port once" is valid only if the shared primitive is the **CORE** FMA and `gemv_tile` is **restructured** to the CORE layout — not merely given fp32 dequant.

**Port once:** add the CORE fp32 FMA + fp32 dequant siblings in `exl3_gemv_ns`; the GEMV kernel and MoE-coop's `gemv_tile` both consume them. For MoE-coop on Maxwell:
- **Restructure `gemv_tile` to the CORE layout:** all-rows A fragment via `rows[]` (all lanes read each row, not `r0 = lane>>2`), per-lane FMA on column `g = lane>>2`, `shfl_xor(1)+shfl_xor(2)` reduction, per-row `accm` accumulator. The `sh_red`/`red` indexing changes from the MMA `[wk][r0][col]` form to the CORE `[warp][r][t*16+g(,+8)]` form.
- `exl3_moe_coop.cu:130` (`if (exl3_no_tc(...)) return;`) is re-gated so MoE-coop's `gemv_tile` is eligible on Maxwell.
- **Scope limit:** cover MoE-coop's `gemv_tile` (the single-row / decode tile). The multi-row MoE GEMM tiles (m32/m64) remain the current `nullptr` decline — that is Workstream E, NOT built here. Do not build multi-row FMA.

## 8. SK GEMV (full fp32-FMA port + dispatch re-gate)

**Source:** the file is **absent from the maxwell worktree**. Source it from the Turing snapshot `/home/mc/AI/workspace/exllamav3/build/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemv_sk_kernel.cuh` (471 lines) + companion `comp_units/exl3_gemv_sk_inst.cu` (29 lines). Copy both into the worktree.

**Current state:** device gate `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 750) __trap();` (L65-68) — on sm_52 the whole body compiles to `__trap()`. Host dispatch `sk_eligible = cc == CC_OLD && ...` (`exl3_gemv.cu:149` in the Turing source) — Maxwell maps to `CC_OLD`, so it is a **latent trap**. Note: the maxwell worktree's own `exl3_gemv.cu` has **no** SK dispatch at all (it never inherited it), so the dispatch must be **added**, not just re-gated.

**Port (structural):**
- Re-gate the device gate from `__CUDA_ARCH__ != 750` to `EXL3_MAXWELL` (i.e. the real code compiles on Maxwell).
- Replace the tensor-core mma: `exl3_gemv_sk_mma_wa` (`mma.sync.aligned.m16n8k8.row.col.f16`, L50-60, used in the TRANS / weights-as-A path) and `exl3_gemv_ns::mma_ab_h` (non-TRANS path) → fp32 FMA.
- Replace the fp16 dequant (`dq8_regs_*` / `dq_dispatch` → `FragB` half2) → fp32 dequant (`decode_3inst_f`).
- fp16 accumulators (`FragC_h ch`, packed `uint32_t cw`) + the per-ring `__low2float/__high2float` fold → direct fp32 accumulators.
- Cooperative `grid.sync()` (L152, L417) → `exl3_launch_grid_bounded` + `group_barrier` (Workstream A).
- Allocate the SK fp32 workspace on Maxwell too (currently `alloc_gemv_sk_ws` only on sm_75, `exl3_devctx.cu:109`).
- Re-gate the host dispatch: add a Maxwell-eligible SK path (CC_MAXWELL) alongside the existing CC_OLD path.

**Per-MMODE scope (Architect finding — the SK selector maps MMODE → TRANS differently):**
- **MMODE 0 (m=1):** TRANS=false, activation-as-A, k16 — **reuses the main GEMV's CORE fp32 FMA** (same layout).
- **MMODE 1 (m=2..8):** TRANS=true, **weights-as-A, k8** (`exl3_gemv_sk_mma_wa`, `mma.m16n8k8`) — a **separate** fp32 FMA derived from the k8 D-row=D-column mapping. The main GEMV's FMA cannot substitute (it has no TRANS path). This is the common decode case and the largest hidden chunk of C3.
- **MMODE 2 (m=9..16):** TRANS=true, NB=2 (two 8-row halves). **Decline to the dense GEMM** — it is outside the m≤8 decode range this workstream targets. Do not port MMODE 2.

**RISK (compound):** the SK port is **two** FMA derivations (TRANS k8 + non-TRANS k16), plus fp32 dequant, fp32 accumulator restructure, `__trap`→real-code re-gate, bounded-launch sync, and SK workspace alloc. The Coder must study the SK kernel's exact A/B/accumulator lane mapping for BOTH the k8 and k16 paths before writing the FMA — do not assume either matches the main GEMV. This is the riskiest sub-task.

## 9. int8-GEMV — skip, document the coverage gap

Do NOT port. `exl3_gemv_int8_kernel.cuh` is pure `__dp4a` (`dp4a_us` → `EXL3_DP4A`), and its value on a real target is the dp4a integer dot — absent/low-value on Maxwell. It is already gated off by the smem check (`exl3_gemv_int8.cu:279`: `if (get_smem_max(device) < 80*1024) return false;` — Maxwell max smem/block = 48 KB < 80 KB). Its cooperative launch is also not Maxwell-gated.

**Coverage gap to document (in a comment + this doc):** the fp32 GEMV covers K≥2 (EXL3 codebooks need 2-bit minimum). The int8 GEMV covered the **K=1** case and the **residual / error-feedback quality mode** (`exl3_gemv_int8_mode() == 1`). On Maxwell, K=1 and the int8 residual mode are **not covered** — decode falls back to the fp32 GEMM. No action beyond documenting.

## 10. Performance design (decode is memory-bound)

- Decode GEMV streams weights; it is **memory-bound**. The procedural fp32 dequant is **ALU-bound** and must overlap the FMA.
- **Latency hiding without cp.async:** the existing `__ldcs` (`ld.global.cs`) prefetch ring (`pf[PF][LOADS]`, register array, compile-time bounds so LDgs stay live) is the mechanism. Keep `PF` deep enough (CFG 0: PF=4; CFG 1: PF=2) to keep in-flight LDGs under the dequant+FMA.
- **Fuse dequant+FMA:** the fp32 weights feed the FMA directly (no materialization). 1 decode + 1 FMA per weight.
- **Instruction-level overlap:** the dequant of tile `i+PF` overlaps the FMA of tile `i` via the ring; the `shfl` window resolution and the FMA are ILP-parallel.
- **M40 (24 SM) vs M10 (5 SM, bandwidth-limited):** the split-K across `WK` warps and the cross-warp `sh_red` reduction are the levers. Both CFGs (0/1) are available; the launch/autotune picks occupancy. MMODE 1 has higher register pressure (fp32 `accm[WNT][2][ROWS]` + per-row A fragments) — prefer CFG 0 (WNT=2) for register headroom; `__launch_bounds__` + occupancy autotune handle the rest.
- fp32 FMA throughput and the ALU dequant are the compute cost centers; the design does not add redundant conversions (A half→fp32 once per fragment; dequant upcasts once per weight).

## 11. Compile-check (no GPU)

`TORCH_CUDA_ARCH_LIST="5.0;5.2" /home/mc/AI/workspace/exllamav3/venv/bin/python setup.py build_ext --inplace` (incremental; do NOT `rm -rf build`). The real GEMV / SK GEMV kernels must compile for `-arch=sm_50` and `-arch=sm_52` under CUDA 12.4. No **fused fp16 FMA / fp16 transcendentals** (`__hfma2`/`hexp`/`h2exp` are sm_53+ and do NOT compile on sm_50/52) — this is why the fp32 FMA is **mandatory**, not a precision choice: the CORE `__hfma2` path would not compile. (The Hadamard's `__hadd2`/`__hmul2` DO compile emulated and are fine.) No tensor-core / dp4a on the Maxwell path.

## 12. Task split for the Coder (pre-split — the B GEMM task was too large for one pass)

- **C1 — Main GEMV fp32 FMA (m==1 + m>1, 2–4 bpw):** the `exl3_gemv_kernel.cuh` FMA path (fp32 dequant siblings + per-lane FMA + CORE reduction + A fragment half→fp32 + drop FOLD), the `exl3_gemv.cu` dispatch (remove no-TC gate, instantiate FMA variant, **re-tune the shape envelope for Maxwell**, `EXL3_COOP_LAUNCH`), and the sync (include `EXL3_GRID_SYNC` or inline `group_barrier`). This is the primary deliverable.
- **C2 — Shared primitive + MoE-coop `gemv_tile`:** the CORE fp32 FMA + dequant siblings in `exl3_gemv_ns` are shared; **restructure** `exl3_moe_coop_kernel.cuh` `gemv_tile` to the CORE layout (all-rows A, per-lane FMA, `shfl_xor` reduction, per-row `accm`); re-gate `exl3_moe_coop.cu:130`. (Builds on C1's primitives.)
- **C3 — SK GEMV port + re-gate (MMODE 0 + MMODE 1 only):** source the file + companion into the worktree; MMODE 0 reuses the main GEMV's CORE FMA; MMODE 1 needs a new weights-as-A k8 fp32 FMA (study the k8 layout first); fp32 dequant, fp32 accumulators, re-gate device + host dispatch, workspace alloc, sync. **Decline MMODE 2 (m=9..16) to the dense GEMM.** Riskiest; do after C1's primitives exist.
- **C4 — int8-GEMV coverage note + design doc commit:** the K=1/residual gap comment; commit the design doc.

C1 must land and compile before C2/C3 build on its primitives. C3 depends on C1's FMA/dequant primitives.

## 13. Files touched

- `exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh` — FMA path + fp32 dequant siblings + CORE reduction (C1, C2).
- `exllamav3/exllamav3_ext/quant/exl3_gemv.cu` — dispatch: remove no-TC gate, FMA instantiation, SK dispatch add, `EXL3_COOP_LAUNCH` (C1, C3).
- `exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh` — `gemv_tile` fp32 path (C2).
- `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu` — re-gate no-TC (C2).
- `exllamav3/exllamav3_ext/quant/exl3_gemv_sk_kernel.cuh` (NEW, sourced) + `comp_units/exl3_gemv_sk_inst.cu` (NEW) — SK port (C3).
- `exllamav3/exllamav3_ext/quant/exl3_devctx.cu` — SK workspace alloc on Maxwell (C3).
- `exllamav3/exllamav3_ext/quant/exl3_gemv_int8*.cu/.cuh` — coverage note only (C4).
- Reuse (do NOT modify): `quant/codebook.cuh` (`decode_3inst_f`), `quant/exl3_dq.cuh` (`exl3_dq_dispatch_f32` reference), Workstream A primitives (`ptx.cuh` `group_barrier`, `coop_autotune.cuh` `exl3_launch_grid_bounded`/`exl3_gemm_launch`/`EXL3_COOP_LAUNCH`, `EXL3_GRID_SYNC`).
