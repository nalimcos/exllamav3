#pragma once

#include <ATen/Tensor.h>

// Flash-decode attention (q_len == 1) over the packed 4-bit KV cache for Turing (sm_75) CUDA cores.
//
// q     [bsz, 1, n_q_heads, 256] fp16 contiguous
// out   [bsz, 1, n_q_heads, 256] fp16 contiguous
// k_cache/v_cache [pages, 256, n_kv_heads * 64] int32 packed (4-bit, groups of 32)
// k_scales/v_scales [pages, 256, n_kv_heads * 16] fp16
// block_table [bsz, num_pages_per_seq] int32, cache_seqlens [bsz] int32
// partial_o  [programs * num_splits * 8 * 256] fp32, partial_ml [programs * num_splits * 8 * 2] fp32
// (Triton decode layout, BLOCK_ROWS = 8, HD_PAD = 256; unused when num_splits == 1)
//
// Values are read in the cache's rotated domain; q and the output are rotated by H32/sqrt(32),
// matching the Triton quantized-cache reference.
void fa75_decode_qc
(
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor k_scales,
    at::Tensor v_cache,
    at::Tensor v_scales,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    at::Tensor out,
    at::Tensor partial_o,
    at::Tensor partial_ml,
    int64_t num_splits,
    int64_t max_k_len,
    int64_t num_pages_per_seq,
    int64_t kv_append_len,
    double scale,
    bool causal
);
