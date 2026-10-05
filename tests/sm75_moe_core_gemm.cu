// Standalone A/B of the fused-MoE inner GEMM's sm_75 CUDA-core (__hfma2) path against the
// emulated-mma path, driven by tests/test_sm75_moe_core.py. Both instantiations are called on
// identical packed-trellis input and their fp16 outputs compared.
//
// The inner GEMM is not reachable from Python on its own (the CORE variant is only selected by
// exl3_moe), so this exercises it directly. TILESIZE_M/TILESIZE_K/N and the stage counts match
// the CORE MoE instance exactly (16, 32, 128, 3, 1); the grid width sweeps the split-k, the
// size_m sweep the row guard, and the size_n sweep the multi-n-tile slice partition.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

#include "exllamav3/exllamav3_ext/util.h"
#include "exllamav3/exllamav3_ext/util.cuh"
#include "exllamav3/exllamav3_ext/quant/hadamard_inner.cuh"
#include "exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh"

#define TM 16
#define TK 32
#define TN 128
#define SH 3
#define FS 1   // CORE reads A from shared in matmul, so load_frags must run immediately before it

template<bool CORE>
__global__ void __launch_bounds__(512)
test_kernel(const half* A, const uint16_t* B, half* C, int size_m, int size_k, int size_n, int* locks)
{
    exl3_gemm_kernel_inner<2, false, false, 2, TM, TK, TN, SH, FS, false, CORE>
        (A, B, C, size_m, size_k, size_n, locks, nullptr);
}

static int run_case(int size_m, int size_k, int size_n, int grid)
{
    int smem = exl3_gemm_smem_bytes(TM, TK, TN, SH, FS, 2, false, false);
    half *A = nullptr, *C0 = nullptr, *C1 = nullptr;
    uint16_t* B = nullptr;
    int* locks = nullptr;
    size_t b_words = (size_t)(size_k / 16) * (size_n / 16) * (16 * 2);
    cudaMalloc(&A, (size_t)size_m * size_k * sizeof(half));
    cudaMalloc(&B, b_words * sizeof(uint16_t));
    cudaMalloc(&C0, (size_t)size_m * size_n * sizeof(half));
    cudaMalloc(&C1, (size_t)size_m * size_n * sizeof(half));
    cudaMalloc(&locks, 4 * 1024 * 1024);

    half* hA = (half*)malloc((size_t)size_m * size_k * sizeof(half));
    uint16_t* hB = (uint16_t*)malloc(b_words * sizeof(uint16_t));
    srand(1234 + size_m * 7 + size_n * 13 + grid);
    for (size_t i = 0; i < (size_t)size_m * size_k; ++i)
        hA[i] = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.5f);
    for (size_t i = 0; i < b_words; ++i) hB[i] = (uint16_t)(rand() & 0xffff);
    cudaMemcpy(A, hA, (size_t)size_m * size_k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, b_words * sizeof(uint16_t), cudaMemcpyHostToDevice);

    double s0 = 0, sd = 0;
    int n = size_m * size_n;
    for (int pass = 0; pass < 2; ++pass)
    {
        half* C = pass == 0 ? C0 : C1;
        cudaMemset(locks, 0, 4 * 1024 * 1024);
        cudaMemset(C, 0, (size_t)size_m * size_n * sizeof(half));
        if (pass == 0) test_kernel<false><<<grid, 512, smem>>>(A, B, C, size_m, size_k, size_n, locks);
        else           test_kernel<true ><<<grid, 512, smem>>>(A, B, C, size_m, size_k, size_n, locks);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    }
    half* h0 = (half*)malloc((size_t)n * sizeof(half));
    half* h1 = (half*)malloc((size_t)n * sizeof(half));
    cudaMemcpy(h0, C0, (size_t)n * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h1, C1, (size_t)n * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        double a = __half2float(h0[i]), b = __half2float(h1[i]);
        s0 += a * a; sd += (a - b) * (a - b);
    }
    double rel = s0 > 0 ? sqrt(sd / s0) : 0.0;
    printf("m=%d k=%d n=%d grid=%d  rel_rms=%.6f\n", size_m, size_k, size_n, grid, rel);
    free(hA); free(hB); free(h0); free(h1);
    cudaFree(A); cudaFree(B); cudaFree(C0); cudaFree(C1); cudaFree(locks);
    // fp16 partial accumulation over k, folded every 4 k-tiles, against an fp32-accumulate mma:
    // anything above 1% is a mapping bug, not rounding
    return rel < 0.01 ? 0 : 1;
}

int main()
{
    int rc = 0;
    rc |= run_case(16, 2048, 128, 1);
    rc |= run_case(16, 2048, 128, 8);
    rc |= run_case(16, 2048, 512, 8);
    rc |= run_case(3,  2048, 512, 8);
    rc |= run_case(5,  2048, 512, 8);
    rc |= run_case(16, 512, 2048, 8);
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
