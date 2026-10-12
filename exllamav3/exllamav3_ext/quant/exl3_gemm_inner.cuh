#pragma once

#include "../ptx.cuh"
#include "exl3_dq.cuh"
// EXL3_GEMM_BASE_THREADS, SMEM_MAX and exl3_gemm_smem_bytes(), the shared definition of this kernel's
// shared memory footprint
#include "exl3_kernel_map.cuh"

// On GA10x, HMMA with fp32 accumulation runs at half rate and dominates the m=1 (decode-bound) case.
// Accumulate MMA results in fp16 instead and fold into the fp32 accumulators once per k-slice: ~14%
// faster at bsz 1 on RTX 3090 together with the codebook.cuh IMUL change (see benchmarks/exl3_m1_bench).
// Max observed error vs fp32 accumulation is ~1% of output RMS at k=4096, well below quantization noise.
// Only enabled for sm_86 for now; unvalidated on other archs where fp32-acc HMMA is also half rate.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    #define EXL3_GEMM_H_ACC 1
#else
    #define EXL3_GEMM_H_ACC 0
#endif

// TILESIZE_M == 16 is the dense / decode shape: one m16 row fragment, A fragments double-buffered
// across the fragment stages (codegen unchanged by the multi-row support below). TILESIZE_M > 16
// is the fused-MoE prefill shape (32 / 64 rows per tile): TILEBLOCKS_M row fragments share each
// dequantized B fragment instead of re-running the whole B pipeline per 16 rows, with A
// single-buffered so the row fragments fit alongside the prefetched B fragments.
template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int size_n_stride = 0     // full width of B and C when computing a column slice (0: = size_n)
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    if (size_n_stride == 0) size_n_stride = size_n;
    // Column blocks of the full-width B row: slices index B relative to their own column offset,
    // but a k-tile row still spans the whole matrix
    const int blocks_n_full = size_n_stride / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    // const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    constexpr int TILE_U16 = 16 * bits + (half_k ? 8 : 0);                        // uint16 per 16x16 tile
    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * TILE_U16;   // in uint16s
    const int sh_c_size = MAX  // in floats
    (
        4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * TILEBLOCKS_M,
        shmem_out_had ? TILESIZE_N * TILESIZE_M : 0
    );

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M >= 16 && TILESIZE_M % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    // The Maxwell fp32 FMA inner builds a 2^bits-float dequant LUT in shared memory (exl3_dq_lut_build);
    // the mma path allocates none. Must match the lut_bytes term in exl3_gemm_smem_bytes.
    constexpr int lut_size = EXL3_MAXWELL ? (1 << bits) : 0;   // floats
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size + 4 * lut_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );
    // The host filters shapes by asking exl3_gemm_smem_bytes() what this layout costs; if that
    // function and the layout above ever diverge, the filter would vet a footprint the kernel
    // does not actually have. Assert they agree, per instantiation, at compile time.
    static_assert
    (
        exl3_gemm_smem_bytes(TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES,
                             bits, half_k, shmem_out_had)
            == SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size + 4 * lut_size,
        "exl3_gemm_smem_bytes() disagrees with the kernel's shared memory layout"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);
    // fp32 dequant LUT (Maxwell FMA path only); lut[code] = float(decode_3inst<cb>(code))
    float* lut = sh_c + sh_c_size;

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    //int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    //int blocks_m = 1;
    //int blocks_k = tiles_k * TILEBLOCKS_K;
    int blocks_n = tiles_n * TILEBLOCKS_N;

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    // int slice_m = index_m(slice_beg);
    // int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);
    const int slice_m = 0;

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * TILE_U16;
    const int gl_b_stride_n = TILEBLOCKS_N * TILE_U16;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * TILE_U16;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n_full * TILE_U16 / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n_stride;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    // TILEBLOCKS_M == 1 (dense / decode): A fragments double-buffered across the fragment stages,
    // as before. TILEBLOCKS_M > 1 (fused MoE prefill tiles): one A fragment per 16-row block,
    // single-buffered, so the row fragments fit alongside the prefetched B fragments
    register FragA frag_a[TILEBLOCKS_M == 1 ? FRAG_STAGES : TILEBLOCKS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #if EXL3_GEMM_H_ACC
        register FragC_h frag_c_h[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #endif
    // Maxwell fp32 FMA path: the dequantized B fragment is fp32 (a LUT load per weight), not the
    // half2 the mma path feeds to the tensor cores. The mma B fragment is 4 values (b[0][0], b[0][1],
    // b[1][0], b[1][1]); same per-lane (k, column) mapping.
    using FragB_f32 = Vec<float, 4>;
    // Maxwell-only arrays; sized to 1 on other arches so they are eliminated (the FMA branch that
    // reads them is compile-discarded there).
    register FragB_f32 frag_b_f32[EXL3_MAXWELL ? FRAG_STAGES : 1][FRAGS_N_PER_WARP];
    // The shared A stage a fragment's B was loaded from, stashed in load_frags so the FMA reads A from
    // the same stage as that B (sh1_a_ptr advances after load_frags, so it must not be read live)
    half* maxwell_a_ptr[EXL3_MAXWELL ? FRAG_STAGES : 1];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout). The Maxwell FMA matmul reads A straight from the
        // swizzled shared tile, so no ldmatrix fragment is loaded; stash the stage for the FMA (it must
        // not read sh1_a_ptr live, which advances after this point).
        if constexpr (!EXL3_MAXWELL)
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[TILEBLOCKS_M == 1 ? buf : m], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }
        else
        {
            maxwell_a_ptr[buf] = sh1_a_ptr;
        }

        // B fragments: half2 for the mma path, fp32 (a LUT load per weight) for the Maxwell FMA path
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * TILE_U16);

            if constexpr (EXL3_MAXWELL)
                exl3_dq_dispatch_f32<bits, cb, half_k>(shb, lane_id << 3, lut, frag_b_f32[buf][n2], frag_b_f32[buf][n2 + 1]);
            else
                dq_dispatch<bits, cb, half_k>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[0][n] = {};
            #if EXL3_GEMM_H_ACC
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    frag_c_h[0][n] = {};
            #endif
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    frag_c[m][n] = {};
            #if EXL3_GEMM_H_ACC
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        frag_c_h[m][n] = {};
            #endif
        }
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i)
        {
            if (sub_k == i)
            {
                if constexpr (TILEBLOCKS_M == 1)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[0][n][j];
                    }
                }
                else
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[m][n][j];
                        }
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i)
        {
            if (sub_k == i)
            {
                if constexpr (TILEBLOCKS_M == 1)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) frag_c[0][n][j] += *sh_red++;
                    }
                }
                else
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) frag_c[m][n][j] += *sh_red++;
                        }
                }
            }
        };

        auto store_small = [&] (int i)
        {
            if constexpr (TILEBLOCKS_M == 1)
            {
                if (sub_k == i && lane_id / 4 < size_m)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        *sh_red++ = frag_c[0][n][0];
                        *sh_red++ = frag_c[0][n][1];
                    }
                }
                __syncthreads();
            }
            else
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                    {
                        bool row_ok = (m * 16 + lane_id / 4) < size_m;
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            if (row_ok) { *sh_red = frag_c[m][n][0]; *(sh_red + 1) = frag_c[m][n][1]; }
                            sh_red += 2;
                        }
                    }
                }
                __syncthreads();
            }
        };

        auto add_small = [&] (int i)
        {
            if constexpr (TILEBLOCKS_M == 1)
            {
                if (sub_k == i && lane_id / 4 < size_m)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        frag_c[0][n][0] += *sh_red++;
                        frag_c[0][n][1] += *sh_red++;
                    }
                }
            }
            else
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4 * TILEBLOCKS_M) * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                    {
                        bool row_ok = (m * 16 + lane_id / 4) < size_m;
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                        {
                            if (row_ok) { frag_c[m][n][0] += *sh_red; frag_c[m][n][1] += *(sh_red + 1); }
                            sh_red += 2;
                        }
                    }
                }
            }
        };

        if (size_m <= 8)
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store_small(1);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store_small(3);
                add_small(2);
                store_small(1);
                add_small(0);
                store_small(2);
                add_small(0);
            }
        }
        else
        {
            if constexpr (TILEBLOCKS_K == 2)
            {
                store(1);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 3)
            {
                store(1);
                add(0);
                store(2);
                add(0);
            }
            if constexpr (TILEBLOCKS_K == 4)
            {
                store(3);
                add(2);
                store(1);
                add(0);
                store(2);
                add(0);
            }
        }
    };

    // Pre-hadamard: Write final output tile to shmem
    auto write_sum_tile_sh = [&]()
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (TILEBLOCKS_M == 1)
        {
            const int r0 = lane_id / 4;
            const int r1 = r0 + 8;
            if (r0 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[0][n][0];
                    *c_ptr++ = frag_c[0][n][1];
                }
            }
            if (r1 < size_m)
            {
                const int c = (lane_id % 4) * 2;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[0][n][2];
                    *c_ptr++ = frag_c[0][n][3];
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                const int r0 = m * 16 + lane_id / 4;
                const int r1 = r0 + 8;
                if (r0 < size_m)
                {
                    const int c = (lane_id % 4) * 2;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[m][n][0];
                        *c_ptr++ = frag_c[m][n][1];
                    }
                }
                if (r1 < size_m)
                {
                    const int c = (lane_id % 4) * 2;
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[m][n][2];
                        *c_ptr++ = frag_c[m][n][3];
                    }
                }
            }
        }
    };

    // Copy output tile to global with hadamard transform and out scale
    auto output_had_sh_gl = [&]()
    {
        int sh_warp = warp_id;
        constexpr int active_warps = EXL3_GEMM_BASE_THREADS / 32;
        for (;; sh_warp += active_warps)
        {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= size_m) break;

            const float* had_in = sh_c + row * TILESIZE_N + col * 128;
            const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;

            if constexpr (c_fp32)
            {
                float* had_out = gl_c_ptr_32 + row * size_n_stride + col * 128;
                had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
            else
            {
                half* had_out = gl_c_ptr_16 + row * size_n_stride + col * 128;
                had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                int r0 = lane_id / 4;
                int r1 = r0 + 8;
                int c = (lane_id % 4) * 2;
                if (r0 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                        frag_c[0][n][0] += *c_ptr++;
                        frag_c[0][n][1] += *c_ptr++;
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[0][n][0] += interm.x;
                        frag_c[0][n][1] += interm.y;
                    }
                }
                if (r1 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                        frag_c[0][n][2] += *c_ptr++;
                        frag_c[0][n][3] += *c_ptr++;
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                        float2 interm = __half22float2(*c_ptr);
                        frag_c[0][n][2] += interm.x;
                        frag_c[0][n][3] += interm.y;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                            frag_c[m][n][0] += *c_ptr++;
                            frag_c[m][n][1] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][0] += interm.x;
                            frag_c[m][n][1] += interm.y;
                        }
                    }
                    if (r1 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                            frag_c[m][n][2] += *c_ptr++;
                            frag_c[m][n][3] += *c_ptr++;
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                            float2 interm = __half22float2(*c_ptr);
                            frag_c[m][n][2] += interm.x;
                            frag_c[m][n][3] += interm.y;
                        }
                    }
                }
        }
    };

    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                int r0 = lane_id / 4;
                int r1 = r0 + 8;
                int c = (lane_id % 4) * 2;
                if (r0 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[0][n][0];
                        *c_ptr++ = frag_c[0][n][1];
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[0][n][0], frag_c[0][n][1]);
                        *c_ptr = sum;
                    }
                }
                if (r1 < size_m)
                {
                    if constexpr (c_fp32)
                    {
                        float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                        *c_ptr++ = frag_c[0][n][2];
                        *c_ptr++ = frag_c[0][n][3];
                    }
                    else
                    {
                        half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                        half2 sum = __floats2half2_rn(frag_c[0][n][2], frag_c[0][n][3]);
                        *c_ptr = sum;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    int r0 = m * 16 + lane_id / 4;
                    int r1 = r0 + 8;
                    int c = (lane_id % 4) * 2;
                    if (r0 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r0 * size_n_stride + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][0];
                            *c_ptr++ = frag_c[m][n][1];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n_stride + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                            *c_ptr = sum;
                        }
                    }
                    if (r1 < size_m)
                    {
                        if constexpr (c_fp32)
                        {
                            float* c_ptr = gl_c_ptr_32 + r1 * size_n_stride + (n0 + n) * 8 + c;
                            *c_ptr++ = frag_c[m][n][2];
                            *c_ptr++ = frag_c[m][n][3];
                        }
                        else
                        {
                            half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n_stride + (n0 + n) * 8 + c);
                            half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                            *c_ptr = sum;
                        }
                    }
                }
        }
    };

    // Output reduction
    auto reduce = [&] ()
    {
        #if EXL3_GEMM_H_ACC
            // Fold the fp16 MMA accumulators into the fp32 accumulators once per k-slice
            if constexpr (TILEBLOCKS_M == 1)
            {
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    float2 f0 = __half22float2(frag_c_h[0][n][0]);
                    float2 f1 = __half22float2(frag_c_h[0][n][1]);
                    frag_c[0][n][0] += f0.x; frag_c[0][n][1] += f0.y;
                    frag_c[0][n][2] += f1.x; frag_c[0][n][3] += f1.y;
                }
            }
            else
            {
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m)
                    #pragma unroll
                    for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    {
                        float2 f0 = __half22float2(frag_c_h[m][n][0]);
                        float2 f1 = __half22float2(frag_c_h[m][n][1]);
                        frag_c[m][n][0] += f0.x; frag_c[m][n][1] += f0.y;
                        frag_c[m][n][2] += f1.x; frag_c[m][n][3] += f1.y;
                    }
            }
        #endif

        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first)
        {
            read_sum_gl();
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            write_sum_gl();
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            if constexpr (shmem_out_had)
                write_sum_tile_sh();
            else
                write_sum_gl();
        }

        if constexpr (shmem_out_had)
        {
            if (last) __syncthreads();
            if (!sub_k && last)
                output_had_sh_gl();
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        if constexpr (EXL3_MAXWELL)
        {
            static_assert(!EXL3_MAXWELL || TILEBLOCKS_M == 1, "Maxwell FMA matmul requires TILESIZE_M == 16");
            // fp32 FMA matmul (no tensor cores), accumulating DIRECTLY in fp32 (frag_c). Output layout
            // matches the mma C fragment the downstream reduce/write expects: a lane owns rows g = lane/4
            // and g+8, columns 2t and 2t+1 (t = lane%4) per n-block.
            //
            // A is read straight from the swizzled shared tile for the FULL k=16 range (both the kc and
            // kc+1 int4 columns), cvt fp16 -> fp32 lossless. B is the fp32 dequantized fragment in the mma B
            // layout; the mma C columns 2t/2t+1 are not the lane's own B column, so they are gathered across
            // the warp with __shfl_xor_sync (the same lane routing the mma does internally). The k=16 dot is
            // partitioned across the four (lane%4) lanes of a (lane/4) group and summed back with a
            // __shfl_xor_sync k-reduce.
            const int t = lane_id & 3;
            const int g = lane_id >> 2;
            const int kc = sub_k * 2;   // int4 column of the 16-half k-half this sub_k owns
            const int sw0 = (g >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK;
            const int sw1 = ((g + 8) >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK;
            const half* row0 = maxwell_a_ptr[buf] + g * TILESIZE_K;
            const half* row1 = maxwell_a_ptr[buf] + (g + 8) * TILESIZE_K;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                // Gather the mma C columns (2t, 2t+1) for all k from the dequantized B fragment. b_c[j] =
                // B[j][2t], b_d[j] = B[j][2t+1], j = 0..15 (the 8 k-pairs as even/odd float2 pairs)
                float2 b_c[8], b_d[8];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    b_c[i].x     = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][0], 8 * t + i);
                    b_c[i].y     = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][1], 8 * t + i);
                    b_c[4 + i].x = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][2], 8 * t + i);
                    b_c[4 + i].y = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][3], 8 * t + i);
                    b_d[i].x     = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][0], 8 * t + 4 + i);
                    b_d[i].y     = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][1], 8 * t + 4 + i);
                    b_d[4 + i].x = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][2], 8 * t + 4 + i);
                    b_d[4 + i].y = __shfl_xor_sync(0xffffffffu, frag_b_f32[buf][n][3], 8 * t + 4 + i);
                }
                // A rows g and g+8 for all k, from the swizzled shared tile (fp16 -> fp32)
                float2 a0[8], a1[8];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    a0[i]     = __half22float2(*(const half2*) (row0 + ((kc + 0) ^ sw0) * 8 + 2 * i));
                    a0[4 + i] = __half22float2(*(const half2*) (row0 + ((kc + 1) ^ sw0) * 8 + 2 * i));
                    a1[i]     = __half22float2(*(const half2*) (row1 + ((kc + 0) ^ sw1) * 8 + 2 * i));
                    a1[4 + i] = __half22float2(*(const half2*) (row1 + ((kc + 1) ^ sw1) * 8 + 2 * i));
                }
                // Four mma C elements: dot of each A row against the two B columns
                float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    c0 = fmaf(a0[i].x, b_c[i].x, c0);
                    c0 = fmaf(a0[i].y, b_c[i].y, c0);
                    c1 = fmaf(a0[i].x, b_d[i].x, c1);
                    c1 = fmaf(a0[i].y, b_d[i].y, c1);
                    c2 = fmaf(a1[i].x, b_c[i].x, c2);
                    c2 = fmaf(a1[i].y, b_c[i].y, c2);
                    c3 = fmaf(a1[i].x, b_d[i].x, c3);
                    c3 = fmaf(a1[i].y, b_d[i].y, c3);
                }
                // Warp k-reduce: the four (lane%4) lanes split the k=16 range for the same (row, col)
                c0 += __shfl_xor_sync(0xffffffffu, c0, 1);
                c0 += __shfl_xor_sync(0xffffffffu, c0, 2);
                c1 += __shfl_xor_sync(0xffffffffu, c1, 1);
                c1 += __shfl_xor_sync(0xffffffffu, c1, 2);
                c2 += __shfl_xor_sync(0xffffffffu, c2, 1);
                c2 += __shfl_xor_sync(0xffffffffu, c2, 2);
                c3 += __shfl_xor_sync(0xffffffffu, c3, 1);
                c3 += __shfl_xor_sync(0xffffffffu, c3, 2);
                frag_c[0][n][0] += c0;
                frag_c[0][n][1] += c1;
                frag_c[0][n][2] += c2;
                frag_c[0][n][3] += c3;
            }
        }
        else if constexpr (TILEBLOCKS_M == 1)
        {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                #if EXL3_GEMM_H_ACC
                    ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c_h[0][n]);
                #else
                    ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[0][n]);
                #endif
            }
        }
        else
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #if EXL3_GEMM_H_ACC
                        ptx_mma_m16n8k16(frag_a[m], frag_b[buf][n], frag_c_h[m][n]);
                    #else
                        ptx_mma_m16n8k16(frag_a[m], frag_b[buf][n], frag_c[m][n]);
                    #endif
                }
        }
    };

    // Build the fp32 dequant LUT once per block (Maxwell FMA path). Every thread fills the same 2^bits
    // entries; the barrier makes the table visible before the dequant reads it in load_frags.
    if constexpr (EXL3_MAXWELL)
    {
        exl3_dq_lut_build<cb, bits>(lut);
        __syncthreads();
    }

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE_OLD(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}
