#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include "graph.cuh"

// Native Q4 attention for the fixed 24Q/4KV, head-dim-256 Maxwell shape.
void q4m_prepare(
    at::Tensor q, at::Tensor q_rot, double q_scale
);

void q4m_attention(
    at::Tensor q,
    at::Tensor k_cache, at::Tensor k_scales,
    at::Tensor v_cache, at::Tensor v_scales,
    at::Tensor block_table, at::Tensor cache_seqlens,
    at::Tensor out, at::Tensor q_rot,
    at::Tensor partial_o, at::Tensor partial_ml,
    double softmax_scale, int64_t splits
);

std::vector<int64_t> q4m_resource_info();

void q4m_prepare_gr(
    const at::Tensor& q, const at::Tensor& q_rot, int total_rows,
    float q_scale, cudaStream_t stream
);

void q4m_decode_gr(
    const at::Tensor& q,
    const at::Tensor& k_cache, const at::Tensor& k_scales,
    const at::Tensor& v_cache, const at::Tensor& v_scales,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens,
    const at::Tensor& out, const at::Tensor& q_rot,
    const at::Tensor& partial_o, const at::Tensor& partial_ml,
    int bsz, int num_pages_per_seq, int splits,
    cudaStream_t stream, Graph* graph
);
