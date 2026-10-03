#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "exl3_devctx.cuh"

// CUDA-core GEMV decision for Turing GeForce (sm_75, TU117/TU116).
//
// The m==1 GEMV kernel has two ways to drive the 16x16 weight tiles: the tensor-core path
// issues mma.m16n8k8 (the emulated m16n8k16, see ptx.cuh) and the CORE path issues packed
// __hfma2 on the CUDA cores (exl3_gemv_kernel.cuh). On the GeForce GTX 16-series the tensor
// cores run at a fraction of the FP16 CUDA-core rate (measured ~0.09x FP16 GFLOPS here), so
// the emulated MMA is the slower of the two and decode should use CORE. On TU104/TU106 (T4,
// RTX 20xx) the tensor cores are full-rate and the MMA path stays.
//
// The decision is probed once per device and cached: time an m16n8k8 f16-accumulate chain
// against a same-MAC-count __hfma2 chain and enable CORE iff the MMA chain is less than half
// the CUDA-core rate. The shapes mirror the measured GEMV inner loop (8 independent
// accumulator chains per thread, 32 MACs per mma per thread).
//
// Env EXL3_GEMV_CORE: "auto" (default) probes, "0"/"1" force the decision. The probe is
// gated to cc 7.5 before the env is consulted, so every other architecture returns 0.

#define GEMV_CORE_ILP 8

__device__ __forceinline__ unsigned gemv_core_pack2(__half lo, __half hi)
{
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// mma.m16n8k8 with f16 accumulation, 8 independent accumulator chains (256 MACs/thread/iter)
__global__ void gemv_core_probe_mma_kernel(const __half* A, const __half* B, __half* out, int iters)
{
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const unsigned a0 = gemv_core_pack2(A[group * 8 + tig * 2 + 0], A[group * 8 + tig * 2 + 1]);
    const unsigned a1 = gemv_core_pack2(A[(group + 8) * 8 + tig * 2 + 0], A[(group + 8) * 8 + tig * 2 + 1]);
    const unsigned b0 = gemv_core_pack2(B[(tig * 2 + 0) * 8 + group], B[(tig * 2 + 1) * 8 + group]);
    unsigned c[GEMV_CORE_ILP][2];
    #pragma unroll
    for (int j = 0; j < GEMV_CORE_ILP; ++j) { c[j][0] = 0; c[j][1] = 0; }
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int j = 0; j < GEMV_CORE_ILP; ++j)
        {
            asm volatile
            (
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                : "+r"(c[j][0]), "+r"(c[j][1])
                :  "r"(a0), "r"(a1), "r"(b0)
            );
        }
    }
    unsigned s = 0;
    #pragma unroll
    for (int j = 0; j < GEMV_CORE_ILP; ++j) s ^= c[j][0] ^ c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = __ushort_as_half((unsigned short) s);
}

// Packed __hfma2 chain, same MAC count: 8 chains x 16 __hfma2 = 256 fp16 MACs/thread/iter
__global__ void gemv_core_probe_hfma2_kernel(const __half* in, __half* out, int iters)
{
    __half2 acc[GEMV_CORE_ILP];
    const __half2 one = __halves2half2(__float2half(1.0f), __float2half(1.0f));
    #pragma unroll
    for (int j = 0; j < GEMV_CORE_ILP; ++j)
    {
        __half v = in[threadIdx.x & 31];
        acc[j] = __halves2half2(v, v);
    }
    const __half2 m = __halves2half2(__float2half(1.001f), __float2half(1.001f));
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int j = 0; j < GEMV_CORE_ILP; ++j)
        {
            #pragma unroll
            for (int k = 0; k < 16; ++k) acc[j] = __hfma2(acc[j], m, one);
        }
    }
    __half2 s = __halves2half2(__float2half(0.f), __float2half(0.f));
    #pragma unroll
    for (int j = 0; j < GEMV_CORE_ILP; ++j) s = __hadd2(s, acc[j]);
    out[blockIdx.x * blockDim.x + threadIdx.x] = __low2half(s);
}

// Time 3 runs of a launch, after one warmup run. Returns elapsed ms.
template <typename F>
static float gemv_core_probe_ms(F&& run, cudaStream_t stream)
{
    run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 3; ++r) run();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

static bool gemv_core_stream_capturing(int device)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t e = cudaStreamIsCapturing(stream, &st);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return st != cudaStreamCaptureStatusNone;
}

// Per-device decision: -1 undecided, 0 use mma, 1 use CORE
static int g_gemv_core[MAX_DEVICES];
static bool g_gemv_core_init = false;
static std::mutex g_gemv_core_mutex;

int g_get_gemv_core_cached(int device)
{
    std::lock_guard<std::mutex> lock(g_gemv_core_mutex);
    if (!g_gemv_core_init || device < 0 || device >= MAX_DEVICES) return -1;
    return g_gemv_core[device];
}

int g_get_gemv_core(int device)
{
    std::lock_guard<std::mutex> lock(g_gemv_core_mutex);
    if (!g_gemv_core_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_gemv_core[i] = -1;
        g_gemv_core_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return 0;
    if (g_gemv_core[device] >= 0) return g_gemv_core[device];

    // Capture safety: the probe allocates a sink with cudaMalloc, which is illegal inside a
    // CUDA graph capture. Leave the decision undecided so the caller stays on the mma path;
    // the next uncaptured launch (or prepare_ctx) resolves it.
    if (gemv_core_stream_capturing(device)) return 0;

    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, device);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        g_gemv_core[device] = 0;
        return 0;
    }
    if (!(prop.major == 7 && prop.minor == 5))
    {
        g_gemv_core[device] = 0;
        return 0;
    }

    int on = 0;
    const char* env = std::getenv("EXL3_GEMV_CORE");
    if (env && std::strcmp(env, "0") == 0) on = 0;
    else if (env && std::strcmp(env, "1") == 0) on = 1;
    else
    {
        const c10::cuda::CUDAGuard guard(device);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        const int blocks = prop.multiProcessorCount * 4;
        const int threads = 256;
        const int iters = 20000;

        __half* dA = nullptr;
        __half* dB = nullptr;
        __half* dOut = nullptr;
        cudaMalloc(&dA, 256 * sizeof(__half));
        cudaMalloc(&dB, 64 * sizeof(__half));
        cudaMalloc(&dOut, (size_t) blocks * threads * sizeof(__half));
        cudaMemset(dA, 0, 256 * sizeof(__half));
        cudaMemset(dB, 0, 64 * sizeof(__half));

        float t_mma = gemv_core_probe_ms([&] {
            gemv_core_probe_mma_kernel<<<blocks, threads, 0, stream>>>(dA, dB, dOut, iters);
        }, stream);
        float t_hfma2 = gemv_core_probe_ms([&] {
            gemv_core_probe_hfma2_kernel<<<blocks, threads, 0, stream>>>(dA, dOut, iters);
        }, stream);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dOut);
        cudaGetLastError();

        // Total FLOPs over the 3 timed runs of each loop
        double mma_flop = (double) blocks * 8.0 * GEMV_CORE_ILP * (2.0 * 16 * 8 * 8) * (double) iters * 3.0;
        double hfma2_flop = (double) blocks * (double) threads * GEMV_CORE_ILP * 16.0 * 2.0 * 2.0 * (double) iters * 3.0;
        double mma_gflops = t_mma > 0.f ? mma_flop / (t_mma * 1e-3) / 1e9 : 0.0;
        double hfma2_gflops = t_hfma2 > 0.f ? hfma2_flop / (t_hfma2 * 1e-3) / 1e9 : 0.0;
        on = (t_mma > 0.f && t_hfma2 > 0.f && mma_gflops < 0.5 * hfma2_gflops) ? 1 : 0;
    }
    g_gemv_core[device] = on;
    return on;
}
