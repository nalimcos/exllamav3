# Workstream B — Dequant core + dense prefill GEMM for Maxwell

Scope: make `exl3_gemm` actually run on Maxwell (M40 CC5.2, M10 CC5.0). This is the
dequant core (a shared fp32, K-dependent LUT) plus the dense prefill GEMM (fp32 FMA),
the synchronization rework it needs, and the shared-memory tile/stage ladder. The dequant
core is also the substrate Workstream C (GEMV) reuses.

---

## Hardware & baseline

Maxwell (sm_50/sm_52) has none of the tensor-core or wide-integer features the rest of
EXL3 is written against:

- No tensor cores (no `mma`, no `ldmatrix`).
- No `dp4a` (sm_61+). The dequant byte-sum runs on the portable loop `exl3_dp4a_us`
  (`ptx_portable.cuh`).
- No fp16 arithmetic. All math is fp32; fp16 is storage only. The fused half FMA /
  exp / reciprocal intrinsics are sm_53+ and are stood in by the portable forms
  (`exl3_hfma`, `exl3_hexp`, ...) on `EXL3_MAXWELL`.
- No `cp.async` (sm_80+). The A/B tile loads are synchronous shared-memory stores.
- No cooperative launch, no `grid.sync()`.
- No gpu-scope ordered atomics (`acquire`/`release`/scoped, sm_60+). Barriers fall back
  to a volatile load + `__threadfence` spin.
- Funnel shift (`shf`) IS available, so the bit-extraction (`dq_dispatch`, shf/bfe) is
  unchanged on Maxwell.

Two tiers are defined: the device tier `EXL3_MAXWELL` (`__CUDA_ARCH__ < 600`,
`exllamav3_ext/arch.cuh`) and the host tier `CC_MAXWELL` (`exllamav3_ext/quant/exl3_devctx.cuh`).

Devices:

- M40 (CC5.2): 24 SM, 288 GB/s, 48 KB max shared memory per block.
- M10 (CC5.0): 5 SM, 83 GB/s. GM107 exposes 48 KB max shared memory but only 32 KB by
  default; the host filter (`get_smem_request`) handles the opt-in.

### The actual baseline (verified)

On Maxwell, `exl3_gemm` IS reached: the Python routing is purely
`rows <= AUTO_RECONSTRUCT_THRESHOLD (144)` with no compute-capability check
(`exllamav3/modules/quant/exl3.py`), so small prefill (rows <= 144) goes to the fused
`exl3_gemm` path and large prefill (rows > 144) goes to `reconstruct_hgemm` (which works).

The autotune path finds smem-compatible shapes and launches via `EXL3_COOP_LAUNCH` =
`cudaLaunchCooperativeKernel`, which is unavailable on CC 5.x. The launch fails,
`gpu_assert` fires, and the process `exit()`s. So the model CRASHES on the first GEMM.

This is not a "cuBLAS fallback" problem: there is no cuBLAS fallback in the EXL3 GEMM
path. Workstream B makes `exl3_gemm` actually run on Maxwell.

---

## Dequant core (fp32, K-dependent LUT)

### Design

One shared-memory LUT per (K, cb), size `2^K` floats:

    lut[code] = float( decode_3inst<cb>(code) )

The table is built by calling the EXISTING `decode_3inst<cb>` / `decode_3inst_2` and
upcasting the half result to float (`decode_3inst_f`). This guarantees the LUT holds the
exact fp16 codebook values: it replicates the fp16 roundings of the mul1 path (cb2) and
the lop3/hadd of cb0/cb1, bit-for-bit. It UNIFIES all three codebooks (cb0/cb1/cb2) into
one `lut[code]` lookup — no per-cb ALU decode path is needed on the hot loop.

- Built once per block, in shared memory, by running the existing dequant math for each
  code `0 .. 2^K - 1`. Max size is `2^8 = 256` floats (1 KB) at K=8.
- Per weight in the consumer (GEMM/GEMV): extract the K-bit code (bit-extraction via
  `dq_dispatch`, unchanged), then `w = lut[code]` (1 LDS) and `acc += a * w` (1 FMA).
  This is the "1 LUT-load + 1 FMA per weight" the plan requires.
- K-dependent size: K <= 4 gives a LUT of <= 16 floats (64 B); K > 4 grows up to 256
  (1 KB). Both fit in shared memory.

The LUT value table must match the EXACT fp16 codebook math, not a fresh fp32
`byte_sum * k_inv + k_bias`: the fresh fp32 form drifts by one fp16 ulp (for cb2, 255 of
256 codes differ and the result is not fp16-representable). Every LUT value is an fp16
result upcast to fp32, so every value is exactly fp16-representable — a property the CPU
test asserts.

### Implementation (shipped)

- `exl3_dq_lut_build<cb, K>(float* lut)` in `quant/codebook.cuh`: fills
  `lut[code] = decode_3inst_f<cb>(code)` for `code < 2^K`. Device-only, unrolled.
- `exl3_dq_lut(lut, code)` in `quant/exl3_dq.cuh`: the fp32 dequant of a K-bit code,
  one shared-memory load. This is what the GEMM/GEMV call.
- `decode_3inst` / `decode_3inst_2` are unchanged (the source of truth); the LUT is a
  precomputation over them, not a replacement. `decode_3inst_2` stays intact for the
  non-Maxwell mma path.
- Bit-extraction (`dq_dispatch`, shf/bfe) is unchanged.

### Documented alternative (not implemented — no GPU to profile)

For K <= 2 the shared-memory LUT has bank conflicts (K=1 -> 16-way, K=2 -> 8-way). There,
the portable ALU byte-sum (`exl3_dp4a_us`, on the integer pipe, overlapping fp32) can win.
Ship LUT-for-all-K as the base; add a K<=2 -> ALU dispatch only if on-hardware profiling
shows the dequant is the bottleneck. Do NOT use `__constant__` memory: random per-lane
access serializes.

---

## Dense prefill GEMM (fp32 FMA)

### Decision

Option (a): a fp32 FMA GEMM. Not Option (b) (reconstruct B to fp32, then cuBLAS SGEMM).

Prefill with large M is memory-sensitive on the B matrix. Option (b) reconstructs B to
fp32 (K=4, N=4096 -> 64 MB), which exceeds the 48 MB L2 and re-reads from DRAM per M-slab
— memory-bound, catastrophic on M10's 2 MB L2. Option (a) keeps B quantized (8 MB,
L2-resident) and fuses dequant + Hadamard. On pure FLOPS cuBLAS may edge the FMA GEMM by
~10%, but the memory win dominates; the FMA GEMM targets ~80-85% of peak fp32.

### Structure (key correctness point)

Do NOT port the gtx16-full MoE CORE inner. It is MoE-hardcoded:
`static_assert(!CORE || TILESIZE_M==16)`, `(!CORE || !c_fp32)`, `(!CORE || !shmem_out_had)`,
16-row accumulators, fp16-out.

Instead, in the EXISTING dense `exl3_gemm_inner.cuh`, REPLACE ONLY the `matmul` lambda
(mma -> fp32 FMA) and the B-dequant (half2 -> fp32 LUT). Keep the existing M-slab loop,
input/output Hadamard, split-k (`barrier_acquire`/`barrier_release`), `c_fp32` output, and
the double-buffered pipeline — they are arch-independent and already present. This is a
matmul-swap, not a merge.

The fp32 FMA inner loop borrows the gtx16 CORE path's read-A-from-swizzled-shared structure
(`koff = 2*(lane_id&3)`, `kc = sub_k*2`, a01/a23 half2 reads) but:

- A is cvt fp16 -> fp32 in-loop (no fp16 FMA).
- B is dequantized to fp32 via the LUT.
- Accumulation is DIRECT fp32 (drop the fp16 partial + fold — more accurate than the gtx16
  fp16-partial CORE, matches `EXL3_GEMM_H_ACC = 0`).

Per lane per k-tile: 128 fp32 FMA (the same count as the fp16 `__hfma2` version — fp32 does
NOT halve the count) + ~64 A-cvt + ~8 LUT+FMA dequant (amortized over 16 rows).

A is stored fp16 in shared memory (storage only) and cvt -> fp32 in the loop: this halves
the A smem footprint versus fp32-in-smem, and the cvt is lossless. B stays quantized in
shared memory (`TILE_U16` uint16); the fp32 dequant lives in registers.

The Hadamard epilogue stays fp32 (`had_ff`/`had_fh` are already dtype-parametric and read
the fp32 `sh_c` tile; the tile scale `svh`/`suh` is applied there, not in the codebook). No
fp16/fp32 mismatch.

---

## Synchronization (owned here)

The rework is multi-site (verified):

- `EXL3_COOP_LAUNCH` has 7 call sites: 5 in `quant/coop_autotune.cu` (the autotuner) and 2
  in `exl3_gemm.cu` (the forced paths). It is `cudaLaunchCooperativeKernel` on CUDA.
- `EXL3_GRID_SYNC()` has 7 call sites, all in `exl3_gemm_kernel.cuh`, currently
  `grid.sync()` on CUDA with no Maxwell branch.
- `exl3_launch_grid_bounded` (`quant/coop_autotune.cuh`) is defined but never called.

Plan, on Maxwell:

- Route `EXL3_COOP_LAUNCH` to a non-cooperative, co-residency-bounded launch via
  `exl3_launch_grid_bounded`, which clamps the grid to `maxb * num_sms` using
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and issues a plain `cudaLaunchKernel`.
- Route `EXL3_GRID_SYNC` to
  `group_barrier(0, gridDim.x*gridDim.y*gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET)` —
  the A-substrate atomicAdd sense-reversing barrier (already in `ptx.cuh`).
- The split-k ordered-atomic lock (`barrier_acquire`/`barrier_release`) already has a
  Maxwell volatile + `__threadfence` branch in `ptx.cuh`.

Co-residency is MANDATORY: a bare spin on an evicted block deadlocks. The bounded grid
(<= `maxb * num_sms`) guarantees the whole grid is resident, so both the whole-grid
barrier and the split-k lock's co-residency requirement are satisfied.

---

## 48 KB tile/stage ladder

Shared memory is not binding (there is headroom), but `SH_STAGES` tunes occupancy. The
shared-memory footprint is

    SH_STAGES * (2*sh_a_stage + 2*sh_b_stage) + 4*sh_c_size

where the `2*` / `4*` are the half/uint16/float -> byte conversions (one A stage, one B
stage, the C tile). For the 16x32x128 tile at K=4 (`TILE_U16 = 64`):

- `sh_a_stage` = 512 halves = 1024 B
- `sh_b_stage` = 1024 uint16 = 2048 B
- `sh_c_size`  = 2048 floats = 8192 B
- per stage = 1024 + 2048 = 3072 B

| SH_STAGES | total | blocks/SM (512-thread block) | occupancy (2048 max threads/SM) |
|-----------|-------|------------------------------|----------------------------------|
| 2         | 14 KB | 3                            | 75%                               |
| 3         | 17 KB | 2                            | 50%                               |
| 4         | 20 KB | 2                            | 50%                               |

`cp.async` is a synchronous load on Maxwell (no async overlap), so latency hiding is
occupancy-based: prefer higher occupancy. SH_STAGES=2 (75%) beats SH_STAGES=4 (50%). The
shape filter (`exl3_gemm_shape_compat` vs `get_smem_request` = 48 KB on Maxwell) already
gates shapes; a Maxwell shape set must be enabled in `select_gemm_shape` (the forced path
currently `case CC_MAXWELL: return 0`; the autotune path filters by smem). The ladder
derives the choice: pick `SH_STAGES` that maximizes blocks/SM while keeping >= 2 for the
double-buffered pipeline.

Verify M10 smem opt-in (GM107: 48 KB max / 32 KB default) — the host filter handles it.

---

## 48 KB ladder (finalized)

Concrete numbers for the shipped Maxwell shape set. The per-shape shared-memory footprint is

    SH_STAGES * (2*sh_a_stage + 2*sh_b_stage) + 4*sh_c_size + 4*lut_size

where the `2*` / `4*` are the half/uint16/float -> byte conversions and `lut_size = 1<<bits`
floats (the Maxwell fp32 dequant LUT; the mma path allocates none). `sh_b_stage` grows with the
bitrate (`TILE_U16 = 16*K`) and the LUT grows `2..256` floats over K=1..8.

Shape 8 (16x32x128, SH_STAGES=2, FRAG_STAGES=3) — the CC_MAXWELL default:

| K  | smem     | blocks/SM (512-thread) | occupancy (2048 max threads/SM) |
|----|----------|------------------------|----------------------------------|
| 1  | 11272 B  | 4                      | 100%                             |
| 2  | 12304 B  | 3                      | 75%                              |
| 3  | 13344 B  | 3                      | 75%                              |
| 4  | 14400 B  | 3                      | 75%                              |
| 5  | 15488 B  | 3                      | 75%                              |
| 6  | 16640 B  | 2                      | 50%                              |
| 7  | 17920 B  | 2                      | 50%                              |
| 8  | 19456 B  | 2                      | 50%                              |

At K=4 the per-shape smem is 14400 B -> 3 blocks/SM -> 75% occupancy. That is the value the
per-shape smem request (Fix 1) feeds to `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; the 48 KB
device-max request would read as 1 block/SM (25%).

Why SH_STAGES=2 (not 4): `cp.async` is a synchronous load on Maxwell (no async overlap), so latency
hiding is occupancy-based. Higher occupancy (3 blocks, 75%) beats a deeper pipeline (2 blocks, 50%);
the extra stage does not overlap the synchronous tile load.

Why shape 3 (16x32x256, SH_STAGES=4) is excluded: it overflows the 48 KB budget at K>=7 (49664 B at
K=7, 54272 B at K=8) and, even where it fits (K<=6), runs at 1 block/SM (25%) — far below shape 8's
3 blocks (75%). It is never returned by `select_gemm_shape` for CC_MAXWELL.

Two distinct mechanisms gate the launch:

- The host shape filter (`exl3_gemm_shape_compat` vs `get_smem_request` = 48 KB on Maxwell) decides
  WHICH shapes are candidates.
- The per-shape smem request (Fix 1) sets the OCCUPANCY: it is the value `exl3_launch_grid_bounded`
  passes to `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, so the grid bound reflects the real
  footprint (3 blocks/SM for shape 8) rather than the 48 KB device max (1 block/SM).

---

## CPU dequant unit tests

`tests/test_maxwell_dequant.py` is a pure-Python/NumPy pytest that replicates the exact
fp16 codebook math (byte-sum / lop3 / scale for cb0/cb1/cb2) and checks the LUT value
table against the per-code decode, across K=1..8 and cb=0/1/2.

Scope: it tests the code -> value MAPPING (the LUT table), NOT the CUDA bit-extraction
(shf/bfe) — that path is architecture-dependent and is validated on-hardware. It matches
the EXACT fp16 math (the fp32-upcasted values), not a fresh fp32 formula. The module
imports only `numpy` and `pytest` (no `torch.cuda`, no CUDA tensors, no device skip).

Assertions:

- `test_lut_matches_decode`: the vectorized LUT build equals the scalar per-code decode,
  exactly, for every code (two independent code paths, so it is not circular).
- `test_lut_values_are_fp16`: every LUT value round-trips through fp16 (it is an fp16
  result upcast to fp32, not a fresh fp32 value).
- `test_cb2_differs_from_fresh_fp32`: the mul1 path is not a fresh fp32
  `byte_sum*k_inv+k_bias` (which is not fp16-representable).
- `test_anchor_values`: frozen code -> value pairs guard against accidental changes.
