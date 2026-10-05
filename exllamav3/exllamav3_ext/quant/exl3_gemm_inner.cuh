#pragma once

#include "../ptx.cuh"

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

#include "exl3_dq.cuh"
// For exl3_gemm_smem_bytes(), the shared definition of this kernel's shared memory footprint
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
//
// CORE selects the sm_75 CUDA-core inner loop instead of the (emulated) tensor-core mma: the
// dequantized B fragment is consumed by packed __hfma2 instead of ptx_mma_m16n8k16. It is only
// instantiated for the fused-MoE 16-row tile (TILESIZE_M == 16, TILEBLOCKS_M == 1) and fp16
// output, which is the only shape the MoE host ever asks for; every other instantiation keeps
// CORE = false and is byte-for-byte the previous kernel.
template<EXL3_GEMM_T_ARGS, bool shmem_out_had, bool CORE = false>
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
    // CORE is the sm_75 CUDA-core MoE inner loop: one 16-row tile, fp16 output, k-tile of 32
    // (the A-tile XOR swizzle below is written for 8-half rows), no fused output hadamard
    static_assert(!CORE || TILESIZE_M == 16, "CORE inner GEMM requires the 16-row tile");
    static_assert(!CORE || TILESIZE_K == 32, "CORE inner GEMM requires TILESIZE_K = 32");
    static_assert(!CORE || !c_fp32, "CORE inner GEMM writes fp16 output");
    static_assert(!CORE || !shmem_out_had, "CORE inner GEMM has no fused output hadamard");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );
    // The host filters shapes by asking exl3_gemm_smem_bytes() what this layout costs; if that
    // function and the layout above ever diverge, the filter would vet a footprint the kernel
    // does not actually have. Assert they agree, per instantiation, at compile time.
    static_assert
    (
        exl3_gemm_smem_bytes(TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES,
                             bits, half_k, shmem_out_had)
            == SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "exl3_gemm_smem_bytes() disagrees with the kernel's shared memory layout"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

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

    // CORE accumulators: per (8-column block, tile row) one fp16 k-partial half2 plus the fp32
    // folded sum. Sized 1 when CORE is off so the arrays are eliminated in every other
    // instantiation. core_a_ptr is the shared A tile the CORE matmul reads from (stashed in
    // load_frags, which for CORE runs immediately before matmul with FRAG_STAGES == 1).
    constexpr int CORE_NB = CORE ? FRAGS_N_PER_WARP : 1;
    constexpr int CORE_NR = CORE ? 16 : 1;
    register half2 core_ch[CORE_NB][CORE_NR];
    register float core_acc[CORE_NB][CORE_NR];
    const half* core_a_ptr = nullptr;
    int core_fold = 0;

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

        // A fragments (XOR-swizzled shared memory layout)
        if constexpr (!CORE)
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
            // CORE reads A straight from the shared tile in matmul. Hand it the stage this
            // fragment load is about to advance past (FRAG_STAGES == 1 for CORE, so matmul runs
            // on this same k-tile immediately afterwards)
            core_a_ptr = sh1_a_ptr;
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * TILE_U16);

            dq_dispatch<bits, cb, half_k>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        // The barrier below retires the shared A/B stage this load read. The mma path copies A
        // into registers here, so its shared reads are all done and the barrier is placed before
        // advance1(). CORE reads A in matmul, i.e. after this point, so it takes the barrier at
        // the end of matmul instead (see matmul); skipping it here keeps one barrier per k-tile.
        // On sm_75 the global->shared "async" copy is a synchronous store, so without the barrier
        // on the read side a fast warp's next-tile store can land in the stage a slow warp is
        // still reading, which is exactly the decode run-to-run non-reproducibility this fixes.
        if constexpr (!CORE) __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        if constexpr (CORE)
        {
            #pragma unroll
            for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                #pragma unroll
                for (int r = 0; r < 16; ++r)
                {
                    core_ch[b][r] = __float2half2_rn(0.0f);
                    core_acc[b][r] = 0.0f;
                }
            core_fold = 0;
            return;
        }
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

    // CORE store / accumulate of the tile's final column values (written in reduce below).
    // Only the (lane_id & 3) == 0 lane of each group holds the completed sum; the global C
    // layout is the same row-major tile the mma path produces, so the split-k chain and the
    // MoE consumer see no difference.
    auto core_write_sum_gl = [&]()
    {
        if ((lane_id & 3) != 0) return;
        const int ncol = warp_id * FRAGS_N_PER_WARP * 8 + (lane_id >> 2);
        #pragma unroll
        for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
            #pragma unroll
            for (int r = 0; r < 16; ++r)
                if (r < size_m)
                    gl_c_ptr_16[r * size_n_stride + ncol + b * 8] = __float2half_rn(core_acc[b][r]);
    };

    auto core_read_sum_gl = [&]()
    {
        if ((lane_id & 3) != 0) return;
        const int ncol = warp_id * FRAGS_N_PER_WARP * 8 + (lane_id >> 2);
        #pragma unroll
        for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
            #pragma unroll
            for (int r = 0; r < 16; ++r)
                if (r < size_m)
                    core_acc[b][r] += __half2float(gl_c_ptr_16[r * size_n_stride + ncol + b * 8]);
    };

    // Output reduction
    auto reduce = [&] ()
    {
        if constexpr (CORE)
        {
            // Fold the residual fp16 k-partials, then complete each column's k=16 dot product
            // across the four (lane_id & 3) lanes that split its k range
            #pragma unroll
            for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                #pragma unroll
                for (int r = 0; r < 16; ++r)
                {
                    float v = core_acc[b][r] + __low2float(core_ch[b][r]) + __high2float(core_ch[b][r]);
                    v += __shfl_xor_sync(0xffffffffu, v, 1);
                    v += __shfl_xor_sync(0xffffffffu, v, 2);
                    core_acc[b][r] = v;
                }
            core_fold = 0;

            // Cross-sub_k sum through the reduction scratch (sub_k = 1 stages, sub_k = 0 adds)
            if (sub_k == 1 && (lane_id & 3) == 0)
            {
                #pragma unroll
                for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                    #pragma unroll
                    for (int r = 0; r < 16; ++r)
                        sh_c[r * TILESIZE_N + warp_id * FRAGS_N_PER_WARP * 8 + b * 8 + (lane_id >> 2)] =
                            core_acc[b][r];
            }
            __syncthreads();
            if (sub_k == 0 && (lane_id & 3) == 0)
            {
                #pragma unroll
                for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                    #pragma unroll
                    for (int r = 0; r < 16; ++r)
                        core_acc[b][r] +=
                            sh_c[r * TILESIZE_N + warp_id * FRAGS_N_PER_WARP * 8 + b * 8 + (lane_id >> 2)];
            }
            __syncthreads();

            // Same split-k chain as the mma path: process partial slices in reverse column order
            // so the bottom-slice threadblock is free to move on to the next column
            int lock_i = tiles_k - slice2_k - 1;
            int lock_d = slice2_k - slice2_k0 + 1;
            int* lock = &locks[slice_m * blocks_n + slice2_n];

            barrier_acquire(lock, lock_i);

            bool first = lock_i == 0;
            bool last = lock_i + lock_d == tiles_k;

            if (!sub_k && !first) core_read_sum_gl();
            if (!sub_k && !last) core_write_sum_gl();
            if (!sub_k && last) core_write_sum_gl();

            barrier_release(lock, lock_d, last);

            clear_frag_c();
            return;
        }

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
        if constexpr (CORE)
        {
            // CUDA-core dot product. Lane l owns tile rows 0..15 and, inside each 8-column
            // block b, column l/4; its A k values are 2*(l%4), 2*(l%4)+1 (a01) and
            // 2*(l%4)+8, +9 (a23) - exactly the k values the dequantized B fragment holds for
            // that column (see dq_dispatch). The four lanes of each (l%4) group cover the
            // whole k=16 range and are summed in reduce(). A comes straight from the swizzled
            // shared tile; the swizzle moves whole 8-half groups, so a lane's two half2 words
            // sit at cols (r>>1)&3 and 1^((r>>1)&3), offsets 2*(l%4).
            const int koff = 2 * (lane_id & 3);
            const int kc = sub_k * 2;   // int4 columns this k-half owns (16 halves per sub_k)
            #pragma unroll
            for (int r = 0; r < 16; ++r)
            {
                const int sw = (r >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK;
                const half* row = core_a_ptr + r * TILESIZE_K;
                half2 a01 = *reinterpret_cast<const half2*>(row + ((kc ^ sw) * 8) + koff);
                half2 a23 = *reinterpret_cast<const half2*>(row + (((kc + 1) ^ sw) * 8) + koff);
                #pragma unroll
                for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                {
                    core_ch[b][r] = __hfma2(a01, frag_b[buf][b][0], core_ch[b][r]);
                    core_ch[b][r] = __hfma2(a23, frag_b[buf][b][1], core_ch[b][r]);
                }
            }

            // Fold the fp16 partials into the fp32 accumulators every 4 k-tiles (64 k), like the
            // decode GEMV's CORE path; folding every tile would cost as much as the hfma2s
            if (++core_fold == 4)
            {
                core_fold = 0;
                #pragma unroll
                for (int b = 0; b < FRAGS_N_PER_WARP; ++b)
                    #pragma unroll
                    for (int r = 0; r < 16; ++r)
                    {
                        core_acc[b][r] += __low2float(core_ch[b][r]) + __high2float(core_ch[b][r]);
                        core_ch[b][r] = __float2half2_rn(0.0f);
                    }
            }
            // Retire the shared A/B stage this matmul (and the preceding load_frags, which read
            // the B tile) consumed, before the next iteration's global->shared store overwrites
            // it. The mma path takes this barrier at the end of load_frags instead; see there.
            __syncthreads();
            return;
        }
        if constexpr (TILEBLOCKS_M == 1)
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
