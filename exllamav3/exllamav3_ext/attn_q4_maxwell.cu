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


// Six-row QK software pipeline. F identifies one (four-dimension, row) Q fragment.
// The four-entry Q ring loads F+4 before row F's FFMAs. The two-entry K ring
// decodes U+2 before row block U. All indices are compile-time constants.
template<int F>
__device__ __forceinline__ float4 q4m_pipe_load_q(const float* q_base)
{
    static_assert(F >= 0 && F < 64 * 6);
    constexpr int U = F / 6;
    constexpr int r = F % 6;
    return *reinterpret_cast<const float4*>(q_base + r * HD + 4 * U);
}

template<int U>
__device__ __forceinline__ float4 q4m_pipe_load_k(const uint4* k_row, const uint4& ks, int lane)
{
    static_assert(U >= 0 && U < 64);
    constexpr int g = U / 8;
    constexpr int j = (U % 8) / 2;
    constexpr int n4 = U % 2;
    uint4 packed = k_row[(g + lane) & 7];
    uint32_t word = word_at(packed, j);
    float a = half_at(ks, g) * 0.125f;
    float b = -7.5f * a;
    constexpr int shift = 16 * n4;
    return make_float4(
        fmaf((float) ((word >> shift) & 15), a, b),
        fmaf((float) ((word >> (shift + 4)) & 15), a, b),
        fmaf((float) ((word >> (shift + 8)) & 15), a, b),
        fmaf((float) ((word >> (shift + 12)) & 15), a, b));
}

template<int U, int r>
__device__ __forceinline__ void q4m_pipe_rows(
    const float* q_base, const uint4* k_row, const uint4& ks, int lane,
    float4 (&q_ring)[4], float (&score)[6], const float4& kv)
{
    constexpr int F = U * 6 + r;
    float4 qv = q_ring[F & 3];
    if constexpr (F + 4 < 64 * 6)
        q_ring[F & 3] = q4m_pipe_load_q<F + 4>(q_base);
    score[r] = fmaf(qv.x, kv.x, score[r]);
    score[r] = fmaf(qv.y, kv.y, score[r]);
    score[r] = fmaf(qv.z, kv.z, score[r]);
    score[r] = fmaf(qv.w, kv.w, score[r]);
    if constexpr (r < 5)
        q4m_pipe_rows<U, r + 1>(q_base, k_row, ks, lane, q_ring, score, kv);
}

template<int U>
__device__ __forceinline__ void q4m_pipe_steps(
    const float* q_base, const uint4* k_row, const uint4& ks, int lane,
    float4 (&q_ring)[4], float4 (&k_ring)[2], float (&score)[6])
{
    float4 kv = k_ring[U & 1];
    if constexpr (U + 2 < 64)
        k_ring[U & 1] = q4m_pipe_load_k<U + 2>(k_row, ks, lane);
    q4m_pipe_rows<U, 0>(q_base, k_row, ks, lane, q_ring, score, kv);
    if constexpr (U < 63)
        q4m_pipe_steps<U + 1>(q_base, k_row, ks, lane, q_ring, k_ring, score);
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
__global__ __launch_bounds__(128, 4)
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
        const uint32_t* k_tile = K + (size_t) physical_row0 * (KVH * 32) + h * 32;
        const uint32_t* v_tile = V + (size_t) physical_row0 * (KVH * 32) + h * 32;
        const half* sk_tile = SK + (size_t) physical_row0 * (KVH * 8) + h * 8;
        const half* sv_tile = SV + (size_t) physical_row0 * (KVH * 8) + h * 8;

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
                const uint32_t* src = k_tile + (32 * warp + token) * (KVH * 32) + 4 * g;
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
            const half* src = sk_tile + (32 * warp + lane) * (KVH * 8);
            ks = *reinterpret_cast<const uint4*>(src);
        }

        float score[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) score[r] = 0.0f;

        const uint4* k_row = arena.cohorts[cohort].kbuf[pv_half][lane];
        if constexpr (R == 6)
        {
            float4 q_ring[4] = {
                q4m_pipe_load_q<0>(q_base), q4m_pipe_load_q<1>(q_base),
                q4m_pipe_load_q<2>(q_base), q4m_pipe_load_q<3>(q_base)};
            float4 k_ring[2] = {
                q4m_pipe_load_k<0>(k_row, ks, lane), q4m_pipe_load_k<1>(k_row, ks, lane)};
            q4m_pipe_steps<0>(q_base, k_row, ks, lane, q_ring, k_ring, score);
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                uint4 packed = k_row[(g + lane) & 7];
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
                        #pragma unroll
                        for (int r = 0; r < R; ++r)
                        {
                            int qr = r < 6 ? r : 24 + r - 6;
                            float4 qv = *reinterpret_cast<const float4*>(q_base + qr * HD + d);
                            score[r] = fmaf(qv.x, k0, score[r]);
                            score[r] = fmaf(qv.y, k1, score[r]);
                            score[r] = fmaf(qv.z, k2, score[r]);
                            score[r] = fmaf(qv.w, k3, score[r]);
                        }
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
            const half* src = sv_tile + (64 * cohort + t) * (KVH * 8);
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

        int word_offset = 16 * pv_half + (lane >> 1);
        int nibble0 = 16 * (lane & 1);
        int group = 4 * pv_half + (lane >> 3);
        const uint32_t* v_ptr = v_tile + 64 * cohort * (KVH * 32) + word_offset;
        #pragma unroll 4
        for (int pt = 0; pt < 64; ++pt)
        {
            int logical = n0 + 64 * cohort + pt;
            uint32_t word = 0;
            if (logical < kv_end)
                word = *v_ptr;
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
            v_ptr += KVH * 32;
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

struct LaneDHot
{
    float red03[8][36];
    float red45[8][36];
    float score[8][6];
    float2 pa[8][6];
    uint32_t flags[8];
};

struct LaneDHotArena
{
    LaneDHot warps[4];
    float q[6][HD];
};

union LaneDArena
{
    LaneDHotArena hot;
    float stage[2][6][HD];
};

static_assert(sizeof(LaneDHot) == 2912);
static_assert(sizeof(LaneDHotArena) == 17792);
static_assert(sizeof(LaneDArena) == 17792);

__device__ __forceinline__ void laneD_qk_row(
    float& score, const float (&k)[8], const float4& q0, const float4& q1)
{
    score = fmaf(q0.x, k[0], 0.0f);
    score = fmaf(q0.y, k[1], score);
    score = fmaf(q0.z, k[2], score);
    score = fmaf(q0.w, k[3], score);
    score = fmaf(q1.x, k[4], score);
    score = fmaf(q1.y, k[5], score);
    score = fmaf(q1.z, k[6], score);
    score = fmaf(q1.w, k[7], score);
}

template<int N>
__device__ __forceinline__ void laneD_pv_dimension(
    float (&o)[6][8], const float (&p)[6], uint32_t word, float a, float b)
{
    static_assert(N >= 0 && N < 8);
    float x = fmaf((float) ((word >> (4 * N)) & 15), a, b);
    #pragma unroll
    for (int r = 0; r < 6; ++r)
        o[r][N] = fmaf(p[r], x, o[r][N]);
    if constexpr (N < 7)
        laneD_pv_dimension<N + 1>(o, p, word, a, b);
}

template<int R>
__device__ __forceinline__ void laneD_rescale_pair(
    float (&o)[6][8], float a0, float a1)
{
    static_assert(R == 0 || R == 2 || R == 4);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        o[R][j] *= a0;
        o[R + 1][j] *= a1;
    }
}

__global__ __launch_bounds__(128, 5)
void q4m_laneD_split_kernel(
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
    __shared__ __align__(16) LaneDArena arena;
    __shared__ __align__(16) float warp_m[4][6];
    __shared__ __align__(16) float warp_l[4][6];
    __shared__ __align__(16) float weights[2][2][6];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int cohort = warp >> 1;
    int warp_half = warp & 1;
    int program = blockIdx.x;
    int split = blockIdx.y;
    int h = program & 3;
    int b = program >> 2;
    int past = cache_seqlens[b];
    int kv_end = past + 1;
    int T = (kv_end + TILE - 1) / TILE;
    int t0 = T * split / S;
    int t1 = T * (split + 1) / S;

    if (t0 == t1)
    {
        if (tid < 6)
        {
            size_t ml_base = (((size_t) program * S + split) * 6) * 2;
            partial_ml[ml_base + 2 * tid] = -CUDART_INF_F;
            partial_ml[ml_base + 2 * tid + 1] = 0.0f;
        }
        return;
    }

    const float* q_base = q_rot + (((size_t) b * QH + 6 * h) * HD);
    float4* q_dst = reinterpret_cast<float4*>(&arena.hot.q[0][0]);
    const float4* q_src = reinterpret_cast<const float4*>(q_base);
    #pragma unroll
    for (int i = 0; i < 3; ++i) q_dst[tid + 128 * i] = q_src[tid + 128 * i];
    float o[6][8];
    #pragma unroll
    for (int r = 0; r < 6; ++r)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[r][j] = 0.0f;
    }
    float m = -CUDART_INF_F;
    float l = 0.0f;

    LaneDHot& hot = arena.hot.warps[warp];
    int n0 = TILE * t0;
    int tiles_left = t1 - t0;
    int valid_left = kv_end - n0 - 32 * warp;
    __syncthreads();
    for (; tiles_left > 0; --tiles_left, n0 += TILE, valid_left -= TILE - 32)
    {
        int physical_page = block_table[b * num_pages_per_seq + (n0 >> 8)];
        int physical_row0 = physical_page * PAGE + (n0 & 255);
        const uint32_t* k_ptr = K + ((size_t) physical_row0 * KVH + h) * 32
                                + 32 * warp * (KVH * 32) + lane;
        const uint32_t* v_ptr = V + ((size_t) physical_row0 * KVH + h) * 32
                                + 32 * warp * (KVH * 32) + lane;
        const half* sk_ptr = SK + ((size_t) physical_row0 * KVH + h) * 8
                             + 32 * warp * (KVH * 8) + (lane >> 2);
        const half* sv_ptr = SV + ((size_t) physical_row0 * KVH + h) * 8
                             + 32 * warp * (KVH * 8) + (lane >> 2);

        #pragma unroll 1
        for (int batch = 0; batch < 4; ++batch)
        {
            #pragma unroll
            for (int token = 0; token < 8; ++token)
            {
                bool valid = token < valid_left;
                uint32_t word = valid ? k_ptr[token * (KVH * 32)] : 0u;
                float a = valid ? __half2float(sk_ptr[token * (KVH * 8)]) * 0.125f : 0.0f;
                float bmid = -7.5f * a;
                float k[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    k[j] = fmaf((float) ((word >> (4 * j)) & 15), a, bmid);
                float score[6];
                // Adjacent quads load opposite halves first to cover all 32 banks.
                int half = (lane >> 2) & 1;
                #pragma unroll
                for (int r = 0; r < 6; ++r)
                {
                    const float* qr = &arena.hot.q[r][8 * lane];
                    float4 q0 = *reinterpret_cast<const float4*>(qr + 4 * half);
                    float4 q1 = *reinterpret_cast<const float4*>(qr + 4 * (half ^ 1));
                    if (half) laneD_qk_row(score[r], k, q1, q0);
                    else laneD_qk_row(score[r], k, q0, q1);
                }
                #pragma unroll
                for (int r = 0; r < 6; ++r)
                {
                    score[r] += __shfl_xor_sync(0xffffffffu, score[r], 1);
                    score[r] += __shfl_xor_sync(0xffffffffu, score[r], 2);
                }
                if ((lane & 3) == 0)
                {
                    int group = lane >> 2;
                    *reinterpret_cast<float4*>(&hot.red03[token][4 * group]) =
                        make_float4(score[0], score[1], score[2], score[3]);
                    *reinterpret_cast<float2*>(&hot.red45[token][4 * group]) =
                        make_float2(score[4], score[5]);
                }
            }
            __syncwarp();

            {
                int r = lane >> 3;
                int token = lane & 7;
                float v[8];
                #pragma unroll
                for (int g = 0; g < 8; ++g) v[g] = hot.red03[token][4 * g + r];
                v[0] += v[1]; v[2] += v[3]; v[4] += v[5]; v[6] += v[7];
                v[0] += v[2]; v[4] += v[6];
                v[0] += v[4];
                if (lane < 16) *reinterpret_cast<volatile float*>(&hot.score[token][r]) = v[0];
                if (lane >= 16) *reinterpret_cast<volatile float*>(&hot.score[token][r]) = v[0];
            }
            if (lane < 16)
            {
                int r = 4 + (lane >> 3);
                int token = lane & 7;
                float v[8];
                #pragma unroll
                for (int g = 0; g < 8; ++g) v[g] = hot.red45[token][4 * g + r - 4];
                v[0] += v[1]; v[2] += v[3]; v[4] += v[5]; v[6] += v[7];
                v[0] += v[2]; v[4] += v[6];
                v[0] += v[4];
                hot.score[token][r] = v[0];
            }
            __syncwarp();

            float threshold = m + 12.0f;
            #pragma unroll
            for (int token = 0; token < 8; ++token)
            {
                bool valid = token < valid_left;
                bool updated = false;
                float p = 0.0f;
                float alpha = 1.0f;
                if (lane < 6 && valid)
                {
                    float x = hot.score[token][lane];
                    if (m == -CUDART_INF_F)
                    {
                        m = x;
                        threshold = m + 12.0f;
                        alpha = 0.0f;
                        p = 1.0f;
                        l = 1.0f;
                        updated = true;
                    }
                    else if (x > threshold)
                    {
                        alpha = exp2f(m - x);
                        m = x;
                        threshold = m + 12.0f;
                        p = 1.0f;
                        l = fmaf(l, alpha, 1.0f);
                        updated = true;
                    }
                    else
                    {
                        p = exp2f(x - m);
                        l += p;
                    }
                }
                uint32_t ballot = __ballot_sync(0xffffffffu, updated);
                if (lane < 6) hot.pa[token][lane] = make_float2(p, alpha);
                if (lane == 0) hot.flags[token] = ballot;
            }
            __syncwarp();

            #pragma unroll
            for (int token = 0; token < 8; ++token)
            {
                uint32_t flag = hot.flags[token];
                float p[6];
                float4 pa01 = *reinterpret_cast<const float4*>(&hot.pa[token][0]);
                p[0] = pa01.x; p[1] = pa01.z;
                if (flag) laneD_rescale_pair<0>(o, pa01.y, pa01.w);
                float4 pa23 = *reinterpret_cast<const float4*>(&hot.pa[token][2]);
                p[2] = pa23.x; p[3] = pa23.z;
                if (flag) laneD_rescale_pair<2>(o, pa23.y, pa23.w);
                float4 pa45 = *reinterpret_cast<const float4*>(&hot.pa[token][4]);
                p[4] = pa45.x; p[5] = pa45.z;
                if (flag) laneD_rescale_pair<4>(o, pa45.y, pa45.w);

                bool valid = token < valid_left;
                uint32_t word = valid ? v_ptr[token * (KVH * 32)] : 0u;
                float a = valid ? __half2float(sv_ptr[token * (KVH * 8)]) * 0.125f : 0.0f;
                float bmid = -7.5f * a;
                laneD_pv_dimension<0>(o, p, word, a, bmid);
            }
            valid_left -= 8;
            k_ptr += 8 * (KVH * 32);
            v_ptr += 8 * (KVH * 32);
            sk_ptr += 8 * (KVH * 8);
            sv_ptr += 8 * (KVH * 8);
        }
    }

    // All hot phases finish before reduction/Q storage becomes final staging.
    __syncthreads();
    if (lane < 6)
    {
        warp_m[warp][lane] = m;
        warp_l[warp][lane] = l;
    }
    if (warp_half)
    {
        #pragma unroll
        for (int r = 0; r < 6; ++r)
        {
            float4 o0 = make_float4(o[r][0], o[r][1], o[r][2], o[r][3]);
            float4 o1 = make_float4(o[r][4], o[r][5], o[r][6], o[r][7]);
            *reinterpret_cast<float4*>(&arena.stage[cohort][r][8 * lane]) = o0;
            *reinterpret_cast<float4*>(&arena.stage[cohort][r][8 * lane + 4]) = o1;
        }
    }
    cohort_barrier(cohort);

    float cohort_a0 = 0.0f;
    float cohort_a1 = 0.0f;
    if (!warp_half && lane < 6)
    {
        int r = lane;
        float m0 = warp_m[2 * cohort][r];
        float m1 = warp_m[2 * cohort + 1][r];
        float l0 = warp_l[2 * cohort][r];
        float l1 = warp_l[2 * cohort + 1][r];
        float mc = fmaxf(m0, m1);
        cohort_a0 = (l0 == 0.0f || m0 == -CUDART_INF_F) ? 0.0f : exp2f(m0 - mc);
        cohort_a1 = (l1 == 0.0f || m1 == -CUDART_INF_F) ? 0.0f : exp2f(m1 - mc);
        float lc = fmaf(l0, cohort_a0, l1 * cohort_a1);
        if (lc == 0.0f) mc = -CUDART_INF_F;
        warp_m[2 * cohort][r] = mc;
        warp_l[2 * cohort][r] = lc;
    }
    if (!warp_half)
    {
        #pragma unroll
        for (int r = 0; r < 6; ++r)
        {
            float a0 = __shfl_sync(0xffffffffu, cohort_a0, r);
            float a1 = __shfl_sync(0xffffffffu, cohort_a1, r);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                o[r][j] = fmaf(arena.stage[cohort][r][8 * lane + j], a1, o[r][j] * a0);
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 6)
    {
        int r = lane;
        size_t ml_base = (((size_t) blockIdx.x * S + blockIdx.y) * 6) * 2;
        float m0 = warp_m[0][r];
        float m1 = warp_m[2][r];
        float l0 = warp_l[0][r];
        float l1 = warp_l[2][r];
        float mf = fmaxf(m0, m1);
        float a0 = (l0 == 0.0f || m0 == -CUDART_INF_F) ? 0.0f : exp2f(m0 - mf);
        float a1 = (l1 == 0.0f || m1 == -CUDART_INF_F) ? 0.0f : exp2f(m1 - mf);
        float lf = fmaf(l0, a0, l1 * a1);
        if (lf == 0.0f) mf = -CUDART_INF_F;
        partial_ml[ml_base + 2 * r] = mf;
        partial_ml[ml_base + 2 * r + 1] = lf;
        weights[0][0][r] = a0;
        weights[0][1][r] = a1;
        warp_l[0][r] = lf;
    }
    if (warp == 2)
    {
        #pragma unroll
        for (int r = 0; r < 6; ++r)
        {
            float4 o0 = make_float4(o[r][0], o[r][1], o[r][2], o[r][3]);
            float4 o1 = make_float4(o[r][4], o[r][5], o[r][6], o[r][7]);
            *reinterpret_cast<float4*>(&arena.stage[1][r][8 * lane]) = o0;
            *reinterpret_cast<float4*>(&arena.stage[1][r][8 * lane + 4]) = o1;
        }
    }
    __syncthreads();

    if (warp == 0)
    {
        size_t o_base = (((size_t) blockIdx.x * S + blockIdx.y) * 6) * HD;
        #pragma unroll
        for (int r = 0; r < 6; ++r)
        {
            if (warp_l[0][r] != 0.0f)
            {
                float a0 = weights[0][0][r];
                float a1 = weights[0][1][r];
                float4 s0 = *reinterpret_cast<const float4*>(&arena.stage[1][r][8 * lane]);
                float4 s1 = *reinterpret_cast<const float4*>(&arena.stage[1][r][8 * lane + 4]);
                float4 o0 = make_float4(
                    fmaf(s0.x, a1, o[r][0] * a0), fmaf(s0.y, a1, o[r][1] * a0),
                    fmaf(s0.z, a1, o[r][2] * a0), fmaf(s0.w, a1, o[r][3] * a0));
                float4 o1 = make_float4(
                    fmaf(s1.x, a1, o[r][4] * a0), fmaf(s1.y, a1, o[r][5] * a0),
                    fmaf(s1.z, a1, o[r][6] * a0), fmaf(s1.w, a1, o[r][7] * a0));
                *reinterpret_cast<float4*>(partial_o + o_base + r * HD + 8 * lane) = o0;
                *reinterpret_cast<float4*>(partial_o + o_base + r * HD + 8 * lane + 4) = o1;
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

void validate_q4m_args(
    const at::Tensor& q,
    const at::Tensor& k_cache, const at::Tensor& k_scales,
    const at::Tensor& v_cache, const at::Tensor& v_scales,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens,
    const at::Tensor& out, const at::Tensor& q_rot,
    const at::Tensor& partial_o, const at::Tensor& partial_ml,
    int64_t splits)
{
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

void q4m_laneD_decode_gr(
    const at::Tensor& q,
    const at::Tensor& k_cache, const at::Tensor& k_scales,
    const at::Tensor& v_cache, const at::Tensor& v_scales,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens,
    const at::Tensor& out, const at::Tensor& q_rot,
    const at::Tensor& partial_o, const at::Tensor& partial_ml,
    int bsz, int num_pages_per_seq, int splits,
    cudaStream_t stream, Graph* graph)
{
    TORCH_CHECK(q.size(1) == 1, "q4m laneD increment 1 supports q_len=1 only");
    dim3 grid(bsz * 4, splits);
    auto split_fn = q4m_laneD_split_kernel;
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
    validate_q4m_args(q, k_cache, k_scales, v_cache, v_scales, block_table, cache_seqlens,
                      out, q_rot, partial_o, partial_ml, splits);
    int bsz = (int) q.size(0);
    int M = (int) q.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    q4m_prepare_gr(q, q_rot, bsz * M * QH, (float) softmax_scale * LOG2E, stream);
    q4m_decode_gr(q, k_cache, k_scales, v_cache, v_scales, block_table, cache_seqlens,
                  out, q_rot, partial_o, partial_ml, bsz, (int) block_table.size(1), (int) splits,
                  stream, nullptr);
}

void q4m_laneD_attention(
    at::Tensor q,
    at::Tensor k_cache, at::Tensor k_scales,
    at::Tensor v_cache, at::Tensor v_scales,
    at::Tensor block_table, at::Tensor cache_seqlens,
    at::Tensor out, at::Tensor q_rot,
    at::Tensor partial_o, at::Tensor partial_ml,
    double softmax_scale, int64_t splits)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    validate_q4m_args(q, k_cache, k_scales, v_cache, v_scales, block_table, cache_seqlens,
                      out, q_rot, partial_o, partial_ml, splits);
    int bsz = (int) q.size(0);
    int M = (int) q.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    q4m_prepare_gr(q, q_rot, bsz * M * QH, (float) softmax_scale * LOG2E, stream);
    q4m_laneD_decode_gr(q, k_cache, k_scales, v_cache, v_scales, block_table, cache_seqlens,
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

std::vector<int64_t> q4m_laneD_resource_info()
{
    cudaFuncAttributes attr{};
    auto fn = q4m_laneD_split_kernel;
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
void q4m_laneD_attention(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                         at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                         double, int64_t) { TORCH_CHECK(false, "q4m Maxwell unavailable on ROCm"); }
std::vector<int64_t> q4m_resource_info() { return {}; }
std::vector<int64_t> q4m_laneD_resource_info() { return {}; }

#endif
