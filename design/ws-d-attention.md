# Workstream D — Attention (Maxwell port)

**Target:** Maxwell M40 (CC 5.2, 24 SM, 288 GB/s, ~6.83 TFLOPS fp32) / M10 (CC 5.0, 5 SM, 83 GB/s). No tensor cores, no dp4a, no native fp16 arithmetic (`__hfma2`/`hexp`/`h2exp` are sm_53+ → DO NOT COMPILE on sm_50/52); all compute is fp32 FMA. No cp.async.

Worktree: `exllamav3-wt-maxwell` (branch `maxwell`). Builds on Workstream A (PTX-compat substrate + Triton sm_50/52 make-or-break verdict), Workstream B (fp32 dequant core + dense GEMM), and Workstream C (GEMV/decode). This doc is the design for the attention (QK) kernels: paged decode, flash-decode fast path, paged prefill, and the native-CUDA / SDPA fallbacks.

## Decisions

- **Decode — Triton paged decode is the primary/perf path** (broadest: any head_dim≤512, any GQA, 4-bit quant KV via int-shift dequant, split-KV across CTAs (2·SM target), in-flight LDGs, fp32 online softmax). Compile-verified for sm_50/52 (0 mma). It already runs end-to-end through dispatch.

- **Decode — fa75_decode (opt-in fast path):** ported the gtx16-full CUDA-core flash-decode (`fa75_decode.cu`, packed 4-bit KV, head_dim 256, GQA 8, 8 warps = 8 GQA heads, 1 CTA/SM, 37696 B smem) to fp32. Single arithmetic change: the Q·Kᵀ inner loop `__hfma2` → fp32 FMA (`float2 acc`, `fmaf` per lane); smem stays fp16 (37696 B, odd 258 stride, banks unchanged); the `<750 __trap` guard removed. It is the tightest bandwidth machine for the flagship (dim=256/GQA=8/4-bit) config. Gated: `EXL3_ATTN_FA75` env flag (default OFF = falls through to Triton), `dim==256`, GQA==8, `q_len==1`, causal, 4-bit K/V, `cc_major==5`. First in `_fns_qc`. Compile-verified: SASS 1026 FFMA / 0 HFMA2 / 0 MMA / 0 DP4A. **Open item:** A/B vs Triton decode on a real Maxwell before flipping `EXL3_ATTN_FA75` default to 1 (path is compile-verified but functionally unverified here).

- **Prefill — Triton paged prefill is the primary/perf path** (compile-verified, any dim≤512/GQA/q_len, 4-bit). "Triton as an option" = available and preferred when it covers the shape.

- **Prefill — bighead generalized to all instantiated dims** (native-CUDA fp32 fallback, tried after Triton): the `bighead_scalar.py` gate relaxed from `dim>=512` to `{64,128,256,512}` with an instantiated-G guard (G∈{1,2,4,8} for dim<512, {1,2,4,8,16} for dim==512) and q_len relaxed (workspace bounded at 16 MB by the internal kv_chunk_size doubling loop). Pure fp32 FMA + shfl + `__expf`; no TC/dp4a/cp.async.

- **Prefill — SDPA-math threshold (explicit):** the SDPA math backend is acceptable only while the O(N²) fp32 score buffer (`n_q_heads·N²·4` bytes) fits `SDPA_MATH_MAX_SCORE_BYTES` (default 256 MB = 1<<28, env `EXL3_SDPA_MATH_MAX_SCORE_BYTES`). VRAM-bounded, not FLOP-bounded: 256 MB → N≈2048 @ 32 heads, N≈1400 @ 64 heads. Applied to both `fn_torch_sdpa_fallback_cache` (N = `cache_seqlens.max()+kv_len`) and `fn_torch_sdpa_fallback_nocache` (N = `kv_len`); over budget → decline to the next dispatch candidate. SDPA-math is a bounded last-resort fallback (Triton handles varlen/swa/softcap/sinks).

## Triton verdict (Workstream-A make-or-break, cleared)

Triton 3.5.0 JIT-compiles the production attention kernels to real sm_50/52 cubins with 0 `mma` (tl.dot → fp32 FMA). Re-verified in `tests/triton_maxwell_compile_check.py`: decode-split, decode-combine, prefill — all PASS for sm_50 and sm_52.

## Files changed

- `exllamav3/exllamav3_ext/rope.cu` — fp16-weight RMS-norm `else` branch: `__hadd2`/`__hmul2` → fp32 add+mul + single downcast (mirrors the bf16 branch). Consistency/robustness; `__hadd2`/`__hmul2` would still compile emulated on sm_50/52.
- `exllamav3/modules/attention_fn/smem.py` — `smem_limit()`: hard-cap at 48 KB for CC major==5 (Maxwell) so the ladder shrinks even if the driver reports 0 opt-in.
- `exllamav3/modules/attention_fn/dispatch.py` — CC-5.x guard drops the two xformers (cutlass FMHA needs CC≥7.0) entries on Maxwell; `fn_fa75_decode_qc` first in `_fns_qc`. Base order otherwise unchanged (Triton first).
- `exllamav3/exllamav3_ext/fa75_decode.cu` + `.cuh` (new) — ported flash-decode, fp32 Q·Kᵀ, no `<750 __trap`.
- `exllamav3/exllamav3_ext/bindings.cpp` — `m.def("fa75_decode_qc", ...)` + include.
- `exllamav3/modules/attention_fn/triton_paged.py` — `paged_attn_fa75_decode_qc` + `fn_fa75_decode_qc` (opt-in Maxwell gate; multi-split reuses `_paged_attn_decode_combine_kernel`).
- `exllamav3/modules/attention_fn/bighead_scalar.py` — gate generalized to instantiated dims + G guard, q_len relaxed.
- `exllamav3/modules/attention_fn/torch.py` — `SDPA_MATH_MAX_SCORE_BYTES` threshold on both SDPA fallbacks.
- `tests/triton_maxwell_compile_check.py` (new) — sm_50/52 JIT gate.

## Correctness notes (to be confirmed by the Architect correctness review)

- Q·Kᵀ: fa75 accumulates the per-32-key dot in a `float2` with `fmaf` per lane (exactly the `__hfma2` semantics in fp32); smem holds the dequantized rotated-domain K/V and rotated Q (orthogonal H32 → score preserved); fp32 acc ≥ fp16 acc.
- attn·V: fp32 `fmaf` (unchanged). Online softmax: fp32 `__expf`, `alpha=expf(m−m_new)`, `o=alpha·o+expf(s−m_new)·v`, `l=alpha·l+expf(s−m_new)` (unchanged; bighead/fa75/Triton all use this form).
- KV dequant: `(nibble − 7.5)·(scale·0.125)`, rotated domain — bit-for-bit identical to the Triton qc dequant.
- rope fix: fp32 add+bias then fp32 mul then single `__floats2half2_rn` downcast (no fp16 arithmetic).

## Open items

- fa75 A/B vs Triton decode on a real Maxwell before default-on.
- bighead-generalized is a fallback (reached only when Triton declines for a shape); not the primary perf path.
