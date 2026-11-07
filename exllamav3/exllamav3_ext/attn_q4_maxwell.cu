#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "attn_q4_maxwell.cuh"
#include "cache/h32.cuh"

namespace
{
constexpr int QH = 24;
constexpr int KVH = 4;
constexpr int HD = 256;
constexpr int PAGE = 256;
constexpr int TILE = 128;
constexpr float R32 = 0.17677669529663688110f;
constexpr float LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ void cohort_barrier(int cohort)
{
    asm volatile("bar.sync %0, 64;" :: "r"(cohort + 1) : "memory");
}

__device__ __forceinline__ uint32_t word_at(const uint4& v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ float component_at(const float4& v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ float half_at(const uint4& v, int i)
{
    uint32_t pair = word_at(v, i >> 1);
    return __half2float(__ushort_as_half((uint16_t) (pair >> (16 * (i & 1)))));
}

__global__ __launch_bounds__(128)
void q4m_prepare_kernel(const half* __restrict__ q, float* __restrict__ q_rot,
                        int total_rows, float q_scale)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= total_rows) return;

    int sg = lane >> 3;
    int sl = lane & 7;
    #pragma unroll
    for (int g4 = 0; g4 < 8; g4 += 4)
    {
        int d = 32 * (g4 + sg) + 4 * sl;
        const half* src = q + (size_t) row * HD + d;
        uint2 packed = *reinterpret_cast<const uint2*>(src);
        const half* h = reinterpret_cast<const half*>(&packed);
        float v0 = __half2float(h[0]);
        float v1 = __half2float(h[1]);
        float v2 = __half2float(h[2]);
        float v3 = __half2float(h[3]);
        had_4_inreg(v0, v1, v2, v3);
        had_8_subgroup(v0, v1, v2, v3, lane);
        float scale = q_scale * R32;
        float4 out = make_float4(v0 * scale, v1 * scale, v2 * scale, v3 * scale);
        *reinterpret_cast<float4*>(q_rot + (size_t) row * HD + d) = out;
    }
}

template<int R>
struct PVArena
{
    float p_lo[64][4];
    float p_hi[64][R - 4];
    float2 scale_ab[8][65];
};

// Each cohort reuses only its own 8 KiB K/PV storage. A CTA-wide union here would
// let one cohort overwrite the other cohort's live tile between named barriers.
template<int R>
union CohortArena
{
    uint4 kbuf[2][32][8];
    PVArena<R> pv;
};

template<int R>
union SplitArena
{
    CohortArena<R> cohorts[2];
    float stage[R][HD];
};
static_assert(sizeof(CohortArena<6>) == 8192 && sizeof(CohortArena<12>) == 8192);

template<int R, int M, bool TAIL>
__global__ __launch_bounds__(128, R == 6 ? 5 : 4)
void q4m_split_kernel(
    const float* __restrict__ q_rot,
    const uint32_t* __restrict__ K,
    const uint32_t* __restrict__ V,
    const half* __restrict__ SK,
    const half* __restrict__ SV,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    int num_pages_per_seq,
    int S)
{
    constexpr int NRB = R == 6 ? 1 : M / 2;
    static_assert(R == 6 || R == 12);
    static_assert(R != 6 || TAIL);
    static_assert(R != 12 || (M >= 2 && !TAIL));

    __shared__ __align__(16) SplitArena<R> arena;
    __shared__ __align__(16) float base[2][R];
    __shared__ __align__(16) float warp_max[4][R];
    __shared__ __align__(16) float alpha[2][R];
    __shared__ __align__(16) float warp_l[4][R];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int cohort = warp >> 1;
    int pv_half = warp & 1;
    int program = blockIdx.x;
    int split = blockIdx.y;
    int rb = program % NRB;
    int bh = program / NRB;
    int h = bh & 3;
    int b = bh >> 2;
    int past = cache_seqlens[b];
    int last_qpos = R == 6 ? (M == 1 ? 0 : M - 1) : 2 * rb + 1;
    int kv_end = past + last_qpos + 1;
    int T = (kv_end + TILE - 1) / TILE;
    int t0 = T * split / S;
    int t1 = T * (split + 1) / S;
    size_t ml_base = (((size_t) program * S + split) * R) * 2;

    if (t0 == t1)
    {
        if (tid < R)
        {
            partial_ml[ml_base + 2 * tid] = -CUDART_INF_F;
            partial_ml[ml_base + 2 * tid + 1] = 0.0f;
        }
        return;
    }

    float acc[R][4];
    float den[R];
    #pragma unroll
    for (int r = 0; r < R; ++r)
    {
        den[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = 0.0f;
    }
    if (tid < 2 * R) (&base[0][0])[tid] = -CUDART_INF_F;

    const float* q_base;
    if constexpr (R == 6)
        q_base = q_rot + (((size_t) (b * M + last_qpos) * QH + 6 * h) * HD);
    else
        q_base = q_rot + (((size_t) (b * M + 2 * rb) * QH + 6 * h) * HD);

    __syncthreads();

    for (int it = t0; it < t1; ++it)
    {
        int n0 = TILE * it;
        int page = n0 >> 8;
        int page_offset = n0 & 255;
        int physical_page = block_table[b * num_pages_per_seq + page];
        int physical_row0 = physical_page * PAGE + page_offset;

        // Each warp privately stages 32 packed K rows, with an XOR group swizzle.
        int stage_token = n0 + 32 * warp;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int token = 4 * i + (lane >> 3);
            int g = lane & 7;
            int logical = stage_token + token;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (logical < kv_end)
            {
                const uint32_t* src = K + (size_t) (physical_row0 + 32 * warp + token) * (KVH * 32)
                                    + h * 32 + 4 * g;
                packed = *reinterpret_cast<const uint4*>(src);
            }
            arena.cohorts[cohort].kbuf[pv_half][token][(g + token) & 7] = packed;
        }
        __syncwarp();

        int token = n0 + 32 * warp + lane;
        bool token_valid = token < kv_end;
        uint4 ks = make_uint4(0, 0, 0, 0);
        if (token_valid)
        {
            const half* src = SK + (size_t) (physical_row0 + 32 * warp + lane) * (KVH * 8) + h * 8;
            ks = *reinterpret_cast<const uint4*>(src);
        }

        float score[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) score[r] = 0.0f;

        #pragma unroll 4
        for (int g = 0; g < 8; ++g)
        {
            uint4 packed = arena.cohorts[cohort].kbuf[pv_half][lane][(g + lane) & 7];
            float a = half_at(ks, g) * 0.125f;
            float bb = -7.5f * a;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint32_t word = word_at(packed, j);
                #pragma unroll
                for (int n4 = 0; n4 < 2; ++n4)
                {
                    int shift = 16 * n4;
                    float k0 = fmaf((float) ((word >> shift) & 15), a, bb);
                    float k1 = fmaf((float) ((word >> (shift + 4)) & 15), a, bb);
                    float k2 = fmaf((float) ((word >> (shift + 8)) & 15), a, bb);
                    float k3 = fmaf((float) ((word >> (shift + 12)) & 15), a, bb);
                    int d = 32 * g + 8 * j + 4 * n4;
                    #pragma unroll 2
                    for (int r = 0; r < R; ++r)
                    {
                        int qr = R == 6 ? r : (r < 6 ? r : 24 + r - 6);
                        float4 qv = *reinterpret_cast<const float4*>(q_base + qr * HD + d);
                        score[r] = fmaf(qv.x, k0, score[r]);
                        score[r] = fmaf(qv.y, k1, score[r]);
                        score[r] = fmaf(qv.z, k2, score[r]);
                        score[r] = fmaf(qv.w, k3, score[r]);
                    }
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            int qpos = R == 6 ? last_qpos : 2 * rb + r / 6;
            if (!token_valid || token > past + qpos) score[r] = -CUDART_INF_F;
            float mx = score[r];
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
            if (lane == 0) warp_max[warp][r] = mx;
        }
        cohort_barrier(cohort);

        if (!pv_half && lane < R)
        {
            int r = lane;
            float tile_max = fmaxf(warp_max[2 * cohort][r], warp_max[2 * cohort + 1][r]);
            float old_m = base[cohort][r];
            float new_m = old_m;
            float a = 1.0f;
            if (tile_max != -CUDART_INF_F)
            {
                if (old_m == -CUDART_INF_F)
                {
                    new_m = tile_max;
                    a = 0.0f;
                }
                else if (tile_max > old_m + 12.0f)
                {
                    new_m = tile_max;
                    a = exp2f(old_m - new_m);
                }
                base[cohort][r] = new_m;
            }
            alpha[cohort][r] = a;
        }
        cohort_barrier(cohort);

        int t = 32 * pv_half + lane;
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            float a = alpha[cohort][r];
            if (a != 1.0f)
            {
                den[r] *= a;
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[r][j] *= a;
            }
            float m = base[cohort][r];
            float p = (score[r] == -CUDART_INF_F || m == -CUDART_INF_F) ? 0.0f : exp2f(score[r] - m);
            den[r] += p;
            score[r] = p;
        }

        uint4 vs = make_uint4(0, 0, 0, 0);
        int cohort_token = n0 + 64 * cohort + t;
        if (cohort_token < kv_end)
        {
            const half* src = SV + (size_t) (physical_row0 + 64 * cohort + t) * (KVH * 8) + h * 8;
            vs = *reinterpret_cast<const uint4*>(src);
        }
        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            float a = half_at(vs, g) * 0.125f;
            arena.cohorts[cohort].pv.scale_ab[g][t] = make_float2(a, -7.5f * a);
        }
        *reinterpret_cast<float4*>(&arena.cohorts[cohort].pv.p_lo[t][0]) =
            make_float4(score[0], score[1], score[2], score[3]);
        if constexpr (R == 6)
        {
            *reinterpret_cast<float2*>(&arena.cohorts[cohort].pv.p_hi[t][0]) = make_float2(score[4], score[5]);
        }
        else
        {
            *reinterpret_cast<float4*>(&arena.cohorts[cohort].pv.p_hi[t][0]) =
                make_float4(score[4], score[5], score[6], score[7]);
            *reinterpret_cast<float4*>(&arena.cohorts[cohort].pv.p_hi[t][4]) =
                make_float4(score[8], score[9], score[10], score[11]);
        }
        cohort_barrier(cohort);

        int word_offset = h * 32 + 16 * pv_half + (lane >> 1);
        int nibble0 = 16 * (lane & 1);
        int group = 4 * pv_half + (lane >> 3);
        #pragma unroll 4
        for (int pt = 0; pt < 64; ++pt)
        {
            int logical = n0 + 64 * cohort + pt;
            uint32_t word = 0;
            if (logical < kv_end)
                word = V[(size_t) (physical_row0 + 64 * cohort + pt) * (KVH * 32) + word_offset];
            float2 ab = arena.cohorts[cohort].pv.scale_ab[group][pt];
            float v0 = fmaf((float) ((word >> nibble0) & 15), ab.x, ab.y);
            float v1 = fmaf((float) ((word >> (nibble0 + 4)) & 15), ab.x, ab.y);
            float v2 = fmaf((float) ((word >> (nibble0 + 8)) & 15), ab.x, ab.y);
            float v3 = fmaf((float) ((word >> (nibble0 + 12)) & 15), ab.x, ab.y);
            float4 p0 = *reinterpret_cast<const float4*>(&arena.cohorts[cohort].pv.p_lo[pt][0]);
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                float p = component_at(p0, r);
                acc[r][0] = fmaf(p, v0, acc[r][0]);
                acc[r][1] = fmaf(p, v1, acc[r][1]);
                acc[r][2] = fmaf(p, v2, acc[r][2]);
                acc[r][3] = fmaf(p, v3, acc[r][3]);
            }
            if constexpr (R == 6)
            {
                float2 p1 = *reinterpret_cast<const float2*>(&arena.cohorts[cohort].pv.p_hi[pt][0]);
                #pragma unroll
                for (int r = 0; r < 2; ++r)
                {
                    float p = r == 0 ? p1.x : p1.y;
                    acc[4 + r][0] = fmaf(p, v0, acc[4 + r][0]);
                    acc[4 + r][1] = fmaf(p, v1, acc[4 + r][1]);
                    acc[4 + r][2] = fmaf(p, v2, acc[4 + r][2]);
                    acc[4 + r][3] = fmaf(p, v3, acc[4 + r][3]);
                }
            }
            else
            {
                float4 p1 = *reinterpret_cast<const float4*>(&arena.cohorts[cohort].pv.p_hi[pt][0]);
                float4 p2 = *reinterpret_cast<const float4*>(&arena.cohorts[cohort].pv.p_hi[pt][4]);
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                {
                    float p = r < 4 ? component_at(p1, r) : component_at(p2, r - 4);
                    acc[4 + r][0] = fmaf(p, v0, acc[4 + r][0]);
                    acc[4 + r][1] = fmaf(p, v1, acc[4 + r][1]);
                    acc[4 + r][2] = fmaf(p, v2, acc[4 + r][2]);
                    acc[4 + r][3] = fmaf(p, v3, acc[4 + r][3]);
                }
            }
        }
        cohort_barrier(cohort);
    }

    #pragma unroll
    for (int r = 0; r < R; ++r)
    {
        float l = den[r];
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
            l += __shfl_xor_sync(0xffffffffu, l, offset);
        if (lane == 0) warp_l[warp][r] = l;
    }
    __syncthreads();

    if (warp == 0 && lane < R)
    {
        int r = lane;
        float m0 = base[0][r];
        float m1 = base[1][r];
        float l0 = warp_l[0][r] + warp_l[1][r];
        float l1 = warp_l[2][r] + warp_l[3][r];
        float m = fmaxf(m0, m1);
        float a0 = (l0 == 0.0f || m0 == -CUDART_INF_F) ? 0.0f : exp2f(m0 - m);
        float a1 = (l1 == 0.0f || m1 == -CUDART_INF_F) ? 0.0f : exp2f(m1 - m);
        float l = l0 * a0 + l1 * a1;
        if (l == 0.0f) m = -CUDART_INF_F;
        partial_ml[ml_base + 2 * r] = m;
        partial_ml[ml_base + 2 * r + 1] = l;
        alpha[0][r] = a0;
        alpha[1][r] = a1;
        base[0][r] = l;
    }
    if (cohort == 1)
    {
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            float4 v = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
            *reinterpret_cast<float4*>(&arena.stage[r][128 * pv_half + 4 * lane]) = v;
        }
    }
    __syncthreads();

    if (cohort == 0)
    {
        size_t o_base = (((size_t) program * S + split) * R) * HD;
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            if (base[0][r] != 0.0f)
            {
                float a0 = alpha[0][r];
                float a1 = alpha[1][r];
                float4 s = *reinterpret_cast<const float4*>(&arena.stage[r][128 * pv_half + 4 * lane]);
                float4 o = make_float4(
                    fmaf(s.x, a1, acc[r][0] * a0), fmaf(s.y, a1, acc[r][1] * a0),
                    fmaf(s.z, a1, acc[r][2] * a0), fmaf(s.w, a1, acc[r][3] * a0));
                *reinterpret_cast<float4*>(partial_o + o_base + r * HD + 128 * pv_half + 4 * lane) = o;
            }
        }
    }
}

template<int M>
__global__ __launch_bounds__(32)
void q4m_combine_kernel(
    const float* __restrict__ po0,
    const float* __restrict__ ml0,
    const float* __restrict__ po1,
    const float* __restrict__ ml1,
    half* __restrict__ out,
    int S0,
    int S1)
{
    static_assert(M == 1);
    int lane = threadIdx.x;
    int q_head = blockIdx.x % QH;
    int b = blockIdx.x / (QH * M);
    int h = q_head / 6;
    int sibling = q_head % 6;
    int program = b * 4 + h;
    int S = S0;
    (void) po1; (void) ml1; (void) S1;

    float m = -CUDART_INF_F;
    float l = 0.0f;
    if (lane < S)
    {
        size_t idx = (((size_t) program * S + lane) * 6 + sibling) * 2;
        m = ml0[idx];
        l = ml0[idx + 1];
    }
    bool valid = lane < S && l != 0.0f && m != -CUDART_INF_F;
    float mm = valid ? m : -CUDART_INF_F;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, offset));
    float w = valid && mm != -CUDART_INF_F ? exp2f(m - mm) : 0.0f;
    float ll = valid ? w * l : 0.0f;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        ll += __shfl_xor_sync(0xffffffffu, ll, offset);

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int d0 = 4 * lane;
    int d1 = 128 + 4 * lane;
    for (int s = 0; s < S; s += 2)
    {
        float wa = __shfl_sync(0xffffffffu, w, s);
        float wb = s + 1 < S ? __shfl_sync(0xffffffffu, w, s + 1) : 0.0f;
        if (wa != 0.0f)
        {
            const float* p = po0 + (((size_t) program * S + s) * 6 + sibling) * HD;
            float4 a = *reinterpret_cast<const float4*>(p + d0);
            float4 b4 = *reinterpret_cast<const float4*>(p + d1);
            acc0[0] = fmaf(a.x, wa, acc0[0]); acc0[1] = fmaf(a.y, wa, acc0[1]);
            acc0[2] = fmaf(a.z, wa, acc0[2]); acc0[3] = fmaf(a.w, wa, acc0[3]);
            acc1[0] = fmaf(b4.x, wa, acc1[0]); acc1[1] = fmaf(b4.y, wa, acc1[1]);
            acc1[2] = fmaf(b4.z, wa, acc1[2]); acc1[3] = fmaf(b4.w, wa, acc1[3]);
        }
        if (wb != 0.0f)
        {
            const float* p = po0 + (((size_t) program * S + s + 1) * 6 + sibling) * HD;
            float4 a = *reinterpret_cast<const float4*>(p + d0);
            float4 b4 = *reinterpret_cast<const float4*>(p + d1);
            acc0[0] = fmaf(a.x, wb, acc0[0]); acc0[1] = fmaf(a.y, wb, acc0[1]);
            acc0[2] = fmaf(a.z, wb, acc0[2]); acc0[3] = fmaf(a.w, wb, acc0[3]);
            acc1[0] = fmaf(b4.x, wb, acc1[0]); acc1[1] = fmaf(b4.y, wb, acc1[1]);
            acc1[2] = fmaf(b4.z, wb, acc1[2]); acc1[3] = fmaf(b4.w, wb, acc1[3]);
        }
    }

    float scale = ll != 0.0f ? R32 / ll : 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { acc0[j] *= scale; acc1[j] *= scale; }
    had_4_inreg(acc0[0], acc0[1], acc0[2], acc0[3]);
    had_4_inreg(acc1[0], acc1[1], acc1[2], acc1[3]);
    had_8_subgroup(acc0[0], acc0[1], acc0[2], acc0[3], lane);
    had_8_subgroup(acc1[0], acc1[1], acc1[2], acc1[3], lane);

    union { half h[4]; uint2 u; } o0, o1;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        o0.h[j] = __float2half_rn(acc0[j]);
        o1.h[j] = __float2half_rn(acc1[j]);
    }
    half* dst = out + (size_t) blockIdx.x * HD;
    *reinterpret_cast<uint2*>(dst + d0) = o0.u;
    *reinterpret_cast<uint2*>(dst + d1) = o1.u;
}

void validate_q_rot(const at::Tensor& q, const at::Tensor& q_rot)
{
    TORCH_CHECK(q.is_cuda() && q_rot.is_cuda(), "q4m: CUDA tensors required");
    TORCH_CHECK(q.scalar_type() == at::kHalf && q_rot.scalar_type() == at::kFloat, "q4m: q/out dtypes");
    TORCH_CHECK(q.dim() == 4 && q.size(1) >= 1 && q.size(1) <= 8 && q.size(2) == QH && q.size(3) == HD,
                "q4m: q must be [B,M,24,256]");
    TORCH_CHECK(q.is_contiguous() && q_rot.is_contiguous(), "q4m: contiguous q/q_rot required");
    TORCH_CHECK(q_rot.numel() == q.numel(), "q4m: bad q_rot size");
    TORCH_CHECK(q.device() == q_rot.device(), "q4m: q/q_rot device mismatch");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(q_rot.data_ptr()) & 15) == 0, "q4m: 16-byte alignment required");
}

} // namespace

void q4m_prepare_gr(const at::Tensor& q, const at::Tensor& q_rot, int total_rows,
                    float q_scale, cudaStream_t stream)
{
    dim3 grid((total_rows + 3) / 4);
    q4m_prepare_kernel<<<grid, 128, 0, stream>>>(
        (const half*) q.data_ptr(), (float*) q_rot.data_ptr(), total_rows, q_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void q4m_decode_gr(
    const at::Tensor& q,
    const at::Tensor& k_cache, const at::Tensor& k_scales,
    const at::Tensor& v_cache, const at::Tensor& v_scales,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens,
    const at::Tensor& out, const at::Tensor& q_rot,
    const at::Tensor& partial_o, const at::Tensor& partial_ml,
    int bsz, int num_pages_per_seq, int splits,
    cudaStream_t stream, Graph* graph)
{
    TORCH_CHECK(q.size(1) == 1, "q4m increment 1 supports q_len=1 only");
    dim3 grid(bsz * 4, splits);
    auto split_fn = q4m_split_kernel<6, 1, true>;
    split_fn<<<grid, 128, 0, stream>>>(
        (const float*) q_rot.data_ptr(), (const uint32_t*) k_cache.data_ptr(),
        (const uint32_t*) v_cache.data_ptr(), (const half*) k_scales.data_ptr(),
        (const half*) v_scales.data_ptr(), (const int*) block_table.data_ptr(),
        (const int*) cache_seqlens.data_ptr(), (float*) partial_o.data_ptr(),
        (float*) partial_ml.data_ptr(), num_pages_per_seq, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (graph)
    {
        graph->record_param((void*) split_fn, GP_attn_block_table, 5);
        graph->record_param((void*) split_fn, GP_attn_seqlens, 6);
        graph->record_param((void*) split_fn, GP_attn_num_pages, 9, 4);
        graph->record_param((void*) split_fn, GP_end, 0);
    }
    q4m_combine_kernel<1><<<bsz * QH, 32, 0, stream>>>(
        (const float*) partial_o.data_ptr(), (const float*) partial_ml.data_ptr(),
        (const float*) partial_o.data_ptr(), (const float*) partial_ml.data_ptr(),
        (half*) out.data_ptr(), splits, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void q4m_prepare(at::Tensor q, at::Tensor q_rot, double q_scale)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    validate_q_rot(q, q_rot);
    q4m_prepare_gr(q, q_rot, (int) q.numel() / HD, (float) q_scale,
                   at::cuda::getCurrentCUDAStream());
}

void q4m_attention(
    at::Tensor q,
    at::Tensor k_cache, at::Tensor k_scales,
    at::Tensor v_cache, at::Tensor v_scales,
    at::Tensor block_table, at::Tensor cache_seqlens,
    at::Tensor out, at::Tensor q_rot,
    at::Tensor partial_o, at::Tensor partial_ml,
    double softmax_scale, int64_t splits)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    validate_q_rot(q, q_rot);
    auto* properties = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(properties->major == 5 && properties->minor == 2, "q4m: sm_52 required");
    int bsz = (int) q.size(0);
    int M = (int) q.size(1);
    TORCH_CHECK(bsz == 1 && M == 1, "q4m increment 1 supports batch=1, q_len=1");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous() && out.sizes() == q.sizes(),
                "q4m: out must match q");
    TORCH_CHECK(k_cache.scalar_type() == at::kInt && v_cache.scalar_type() == at::kInt,
                "q4m: packed caches must be int32");
    TORCH_CHECK(k_cache.dim() == 3 && k_cache.size(1) == PAGE && k_cache.size(2) == KVH * 32 &&
                v_cache.sizes() == k_cache.sizes(), "q4m: packed cache shape");
    TORCH_CHECK(k_scales.scalar_type() == at::kHalf && v_scales.scalar_type() == at::kHalf &&
                k_scales.dim() == 3 && k_scales.size(0) == k_cache.size(0) &&
                k_scales.size(1) == PAGE && k_scales.size(2) == KVH * 8 &&
                v_scales.sizes() == k_scales.sizes(), "q4m: scale cache shape");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && k_scales.is_contiguous() &&
                v_scales.is_contiguous() && block_table.is_contiguous() && cache_seqlens.is_contiguous() &&
                partial_o.is_contiguous() && partial_ml.is_contiguous(), "q4m: contiguous tensors required");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && cache_seqlens.scalar_type() == at::kInt &&
                block_table.dim() == 2 && block_table.size(0) == bsz && cache_seqlens.numel() == bsz,
                "q4m: bad block table/sequence lengths");
    TORCH_CHECK(splits >= 1 && splits <= 32, "q4m: splits must be 1..32");
    int64_t programs = (int64_t) bsz * 4;
    TORCH_CHECK(partial_o.scalar_type() == at::kFloat && partial_ml.scalar_type() == at::kFloat &&
                partial_o.numel() >= programs * splits * 6 * HD &&
                partial_ml.numel() >= programs * splits * 6 * 2, "q4m: partial workspace too small");
    auto device = q.device();
    TORCH_CHECK(out.device() == device && k_cache.device() == device && v_cache.device() == device &&
                k_scales.device() == device && v_scales.device() == device && block_table.device() == device &&
                cache_seqlens.device() == device && partial_o.device() == device && partial_ml.device() == device,
                "q4m: device mismatch");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(k_cache.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(v_cache.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(k_scales.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(v_scales.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(block_table.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(partial_o.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(partial_ml.data_ptr()) & 15) == 0,
                "q4m: 16-byte alignment required");
    auto stream = at::cuda::getCurrentCUDAStream();
    q4m_prepare_gr(q, q_rot, bsz * M * QH, (float) softmax_scale * LOG2E, stream);
    q4m_decode_gr(q, k_cache, k_scales, v_cache, v_scales, block_table, cache_seqlens,
                  out, q_rot, partial_o, partial_ml, bsz, (int) block_table.size(1), (int) splits,
                  stream, nullptr);
}

std::vector<int64_t> q4m_resource_info()
{
    cudaFuncAttributes attr{};
    auto fn = q4m_split_kernel<6, 1, true>;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*) fn));
    int occupancy = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, (const void*) fn, 128, 0));
    return {6, 1, attr.numRegs, (int64_t) attr.sharedSizeBytes, (int64_t) attr.localSizeBytes, occupancy};
}

#else

#include <ATen/ATen.h>
#include "attn_q4_maxwell.cuh"
void q4m_prepare(at::Tensor, at::Tensor, double) { TORCH_CHECK(false, "q4m Maxwell unavailable on ROCm"); }
void q4m_attention(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                   at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                   double, int64_t) { TORCH_CHECK(false, "q4m Maxwell unavailable on ROCm"); }
std::vector<int64_t> q4m_resource_info() { return {}; }

#endif
