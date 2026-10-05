// fa75_decode: flash-decode attention (q_len == 1) for the packed 4-bit KV cache on Turing (sm_75)
// CUDA cores (__hfma2), replacing the Triton tl.dot decode path for head_dim 256 / GQA 8.
//
// One CTA per (batch, kv_head); its 8 warps are the 8 GQA query heads. K and V tiles of BLOCK_N keys
// are dequantized once into shared memory (the rotated domain the cache stores), so the 8 sibling q
// heads read them without re-touching global memory. S = Q K^T is computed per warp with lane <-> key
// (no cross-lane reduction in the inner loop); softmax is a warp reduction; P V re-maps the warp to a
// 32-wide dim slice and accumulates all 8 heads. Split-K partials follow the Triton decode layout so
// the existing combine kernel reduces them.
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define FAD_HD 256
#define FAD_GQA 8
// Row stride in halves for the key-major K/V tiles: 256 used + 2 padding. The odd multiple of 4 bytes
// (516) makes the lane<->key half2 reads and the per-lane 16-byte dequant stores land on distinct banks.
#define FAD_KROW 258
#define FAD_R32 0.17677669529663688110f   // 1/sqrt(32)

// H32 on a 32-lane vector (lane l holds element l). The Sylvester matrix is invariant under any
// permutation of the index bits, so this equals the 4-in-register + 8-subgroup factorization the
// quantizer uses. Unnormalized; callers multiply by FAD_R32.
__device__ __forceinline__ float fad_had32(float x, int lane)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        float y = __shfl_xor_sync(0xffffffffu, x, i);
        x = (lane & i) ? (y - x) : (x + y);
    }
    return x;
}

// Occupancy tuning note (GTX 1650 Max-Q, sm_75): this 256-thread / 8-warp / BLOCK_N=32 kernel runs
// 1 CTA/SM (37696 B smem is the cap, not the 128 registers) yet beats every higher-occupancy variant
// measured: 512-thread 16-warp (2 warps per head, dim-split with a cross-warp S reduction) 0.50 vs
// 0.42 ms @ctx 16384 bsz1; a 256-thread head-local PV that drops the softmax->PV barrier (127 regs,
// 37120 B) and a BLOCK_N=16 variant (80-106 regs, 2-3 CTA/SM) are both slower too. More resident
// warps do not lower the per-tile cost, so the kernel is instruction-throughput bound rather than
// latency/occupancy bound, and exp2 softmax (folded via log2 e) is neutral. Keep BLOCK_N=32 here.
template <int BLOCK_N>
__global__ void __launch_bounds__(256, 1)
fa75_decode_kernel
(
    const half* __restrict__ q,
    const uint32_t* __restrict__ k_cache,
    const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_cache,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int n_kv_heads,
    int num_pages_per_seq,
    int kv_append_len,
    int split_len,
    int num_splits,
    float scale,
    int causal
)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    constexpr int KP = BLOCK_N / 32;
    static_assert(KP * 32 == BLOCK_N, "BLOCK_N must be a multiple of 32");

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int batch = pid / n_kv_heads;
    const int kv_head = pid - batch * n_kv_heads;
    const int n_q_heads = n_kv_heads * FAD_GQA;

    const int total_k_len = cache_seqlens[batch] + kv_append_len;
    const int q_abs = total_k_len - 1;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    __shared__ __align__(16) half sK[BLOCK_N][FAD_KROW];
    __shared__ __align__(16) half sV[BLOCK_N][FAD_KROW];
    __shared__ __align__(16) half sQ[FAD_GQA][FAD_HD];
    __shared__ __align__(16) half sP[FAD_GQA][BLOCK_N];
    __shared__ float sAlpha[FAD_GQA];
    __shared__ float sL[FAD_GQA];

    // ---- rotate q into sQ: warp w <-> head w, lane l <-> dim l within each 32-group ----
    {
        const half* qh = q + ((int64_t)(batch * n_q_heads + kv_head * FAD_GQA + warp)) * FAD_HD;
        #pragma unroll
        for (int g = 0; g < FAD_HD / 32; ++g)
        {
            float x = __half2float(qh[g * 32 + lane]);
            x = fad_had32(x, lane) * FAD_R32;
            sQ[warp][g * 32 + lane] = __float2half_rn(x);
        }
    }
    __syncthreads();

    float m_run = -CUDART_INF_F;
    float l_run = 0.0f;
    float O[FAD_GQA];
    #pragma unroll
    for (int r = 0; r < FAD_GQA; ++r) O[r] = 0.0f;

    // Prefetch: the next tile's global loads are issued before this tile's P V so their DRAM latency
    // overlaps the P V compute and the following barrier
    uint32_t lkw[BLOCK_N / 8], lvw[BLOCK_N / 8];
    float lks[BLOCK_N / 8], lvs[BLOCK_N / 8];
    auto fad_load = [&](int n0v)
    {
        #pragma unroll
        for (int kk = 0; kk < BLOCK_N / 8; ++kk)
        {
            const int idx = tid + kk * 256;
            const int key = idx >> 5;
            const int word = idx & 31;
            const int logical = n0v + key;
            const bool valid = logical < n_end;
            const int lg = valid ? logical : 0;
            const int page = lg >> 8;
            const int page_off = lg & 255;
            const int phys = block_table[batch * num_pages_per_seq + page];
            const int token_pos = phys * 256 + page_off;
            const int grp = word >> 2;
            lks[kk] = valid ? __half2float(k_scales[(int64_t) token_pos * (n_kv_heads * 8) + kv_head * 8 + grp]) * 0.125f : 0.0f;
            lvs[kk] = valid ? __half2float(v_scales[(int64_t) token_pos * (n_kv_heads * 8) + kv_head * 8 + grp]) * 0.125f : 0.0f;
            lkw[kk] = valid ? k_cache[(int64_t) token_pos * (n_kv_heads * 32) + kv_head * 32 + word] : 0u;
            lvw[kk] = valid ? v_cache[(int64_t) token_pos * (n_kv_heads * 32) + kv_head * 32 + word] : 0u;
        }
    };
    auto fad_store = [&]()
    {
        // 4 half2 writes: the 516-byte row stride is only 4-byte aligned, and the odd stride is what
        // keeps the lane<->key half2 reads of S = Q K^T bank-conflict free
        #pragma unroll
        for (int kk = 0; kk < BLOCK_N / 8; ++kk)
        {
            const int idx = tid + kk * 256;
            const int key = idx >> 5;
            const int word = idx & 31;
            half2* kd = reinterpret_cast<half2*>(&sK[key][word * 8]);
            half2* vd = reinterpret_cast<half2*>(&sV[key][word * 8]);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int a = (lkw[kk] >> (i * 8)) & 15;
                const int b = (lkw[kk] >> (i * 8 + 4)) & 15;
                kd[i] = __floats2half2_rn(((float) a - 7.5f) * lks[kk], ((float) b - 7.5f) * lks[kk]);
                const int c = (lvw[kk] >> (i * 8)) & 15;
                const int d = (lvw[kk] >> (i * 8 + 4)) & 15;
                vd[i] = __floats2half2_rn(((float) c - 7.5f) * lvs[kk], ((float) d - 7.5f) * lvs[kk]);
            }
        }
    };

    if (n_start < n_end) fad_load(n_start);

    for (int n0 = n_start; n0 < n_end; n0 += BLOCK_N)
    {
        __syncthreads();   // previous tile's PV finished reading sK/sV
        fad_store();
        __syncthreads();

        // ---- S = Q K^T: warp w <-> head w, lane l <-> key (lane + 32*m) ----
        float s[KP];
        #pragma unroll
        for (int m = 0; m < KP; ++m)
        {
            const half* krow = sK[lane + 32 * m];
            const half* qrow = sQ[warp];
            half2 acc = __floats2half2_rn(0.0f, 0.0f);
            float sm = 0.0f;
            #pragma unroll
            for (int g = 0; g < FAD_HD / 32; ++g)
            {
                #pragma unroll
                for (int i = 0; i < 16; ++i)
                {
                    const int k2 = g * 32 + i * 2;
                    acc = __hfma2(*reinterpret_cast<const half2*>(qrow + k2),
                                  *reinterpret_cast<const half2*>(krow + k2), acc);
                }
                float2 f = __half22float2(acc);
                sm += f.x + f.y;
                acc = __floats2half2_rn(0.0f, 0.0f);
            }
            s[m] = sm * scale;
        }

        // ---- online softmax (natural exp), mirroring the Triton decode kernel ----
        #pragma unroll
        for (int m = 0; m < KP; ++m)
        {
            const int logical = n0 + lane + 32 * m;
            const bool valid = (logical < n_end) && (!causal || logical <= q_abs);
            if (!valid) s[m] = -CUDART_INF_F;
        }
        float smax = -CUDART_INF_F;
        #pragma unroll
        for (int m = 0; m < KP; ++m) smax = fmaxf(smax, s[m]);
        #pragma unroll
        for (int i = 1; i < 32; i <<= 1) smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, i));
        const float m_new = fmaxf(m_run, smax);
        const float m_exp = (m_new == -CUDART_INF_F) ? 0.0f : m_new;
        const float alpha = (m_run == -CUDART_INF_F) ? 0.0f : __expf(m_run - m_exp);
        float p[KP];
        float psum = 0.0f;
        #pragma unroll
        for (int m = 0; m < KP; ++m)
        {
            p[m] = __expf(s[m] - m_exp);
            psum += p[m];
        }
        #pragma unroll
        for (int i = 1; i < 32; i <<= 1) psum += __shfl_xor_sync(0xffffffffu, psum, i);
        l_run = l_run * alpha + psum;
        m_run = m_new;
        #pragma unroll
        for (int m = 0; m < KP; ++m) sP[warp][lane + 32 * m] = __float2half_rn(p[m]);
        sAlpha[warp] = alpha;
        __syncthreads();

        // Issue the next tile's loads while P V runs
        if (n0 + BLOCK_N < n_end) fad_load(n0 + BLOCK_N);

        // ---- O += P V: warp w <-> dims [w*32, w*32+32), lane l <-> dim, all 8 heads ----
        {
            const int d = warp * 32 + lane;
            #pragma unroll
            for (int r = 0; r < FAD_GQA; ++r) O[r] *= sAlpha[r];
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n)
            {
                const float v = __half2float(sV[n][d]);
                #pragma unroll
                for (int r = 0; r < FAD_GQA; ++r)
                    O[r] = fmaf(__half2float(sP[r][n]), v, O[r]);
            }
        }
    }

    // ---- finalize ----
    if (num_splits == 1)
    {
        sL[warp] = l_run;
        __syncthreads();
        const int d = warp * 32 + lane;
        #pragma unroll
        for (int r = 0; r < FAD_GQA; ++r)
        {
            float lv = sL[r];
            if (lv == 0.0f) lv = 1.0f;
            float y = fad_had32(O[r] / lv, lane) * FAD_R32;
            out[((int64_t)(batch * n_q_heads + kv_head * FAD_GQA + r)) * FAD_HD + d] = __float2half_rn(y);
        }
    }
    else
    {
        const int64_t base_o = ((int64_t)(pid * num_splits + split)) * (FAD_GQA * FAD_HD);
        const int64_t base_ml = ((int64_t)(pid * num_splits + split)) * (FAD_GQA * 2);
        const int d = warp * 32 + lane;
        #pragma unroll
        for (int r = 0; r < FAD_GQA; ++r)
            partial_o[base_o + r * FAD_HD + d] = O[r];
        if (lane == 0)
        {
            partial_ml[base_ml + warp * 2] = m_run;
            partial_ml[base_ml + warp * 2 + 1] = l_run;
        }
    }
#endif
}

// q [bsz,1,H,256], packed cache, partials in the Triton decode layout
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
)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "cuda tensors");
    TORCH_CHECK(q.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf, "fp16 q/out");
    TORCH_CHECK(k_cache.scalar_type() == at::kInt && v_cache.scalar_type() == at::kInt, "int32 packed cache");
    TORCH_CHECK(k_scales.scalar_type() == at::kHalf && v_scales.scalar_type() == at::kHalf, "fp16 scales");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && cache_seqlens.scalar_type() == at::kInt, "int32 indices");
    TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && q.size(3) == FAD_HD, "q must be [bsz,1,H,256]");
    TORCH_CHECK(q.is_contiguous() && out.is_contiguous(), "contiguous q/out");
    TORCH_CHECK(k_cache.size(1) == 256 && k_cache.size(2) % 32 == 0, "packed cache shape");
    const int bsz = (int) q.size(0);
    const int n_q_heads = (int) q.size(2);
    const int n_kv_heads = n_q_heads / FAD_GQA;
    TORCH_CHECK(n_kv_heads * FAD_GQA == n_q_heads && n_kv_heads > 0, "GQA must be exactly 8");
    TORCH_CHECK(k_cache.size(2) == n_kv_heads * 32, "packed cache head count");
    TORCH_CHECK(k_scales.size(2) == n_kv_heads * 8, "scale head count");
    TORCH_CHECK(block_table.size(0) == bsz && block_table.size(1) == num_pages_per_seq, "block_table shape");
    TORCH_CHECK(cache_seqlens.size(0) == bsz, "cache_seqlens shape");
    TORCH_CHECK(out.size(0) == bsz && out.size(1) == 1 && out.size(2) == n_q_heads && out.size(3) == FAD_HD, "out shape");
    const int splits = (int) num_splits;
    TORCH_CHECK(splits >= 1, "num_splits");
    int split_len = (int) ((max_k_len + splits - 1) / splits);
    if (split_len < 1) split_len = 1;

    dim3 grid(bsz * n_kv_heads, splits);
    fa75_decode_kernel<32><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) q.data_ptr(), (const uint32_t*) k_cache.data_ptr(), (const half*) k_scales.data_ptr(),
        (const uint32_t*) v_cache.data_ptr(), (const half*) v_scales.data_ptr(),
        (const int*) block_table.data_ptr(), (const int*) cache_seqlens.data_ptr(),
        (half*) out.data_ptr(), (float*) partial_o.data_ptr(), (float*) partial_ml.data_ptr(),
        n_kv_heads, (int) num_pages_per_seq, (int) kv_append_len, split_len, splits,
        (float) scale, causal ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void fa75_decode_qc
(
    at::Tensor q, at::Tensor k_cache, at::Tensor k_scales, at::Tensor v_cache, at::Tensor v_scales,
    at::Tensor block_table, at::Tensor cache_seqlens, at::Tensor out, at::Tensor partial_o,
    at::Tensor partial_ml, int64_t num_splits, int64_t max_k_len, int64_t num_pages_per_seq,
    int64_t kv_append_len, double scale, bool causal
)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}

#endif
