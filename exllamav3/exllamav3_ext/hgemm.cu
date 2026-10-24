#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#if defined(USE_ROCM)
    #include "rocm/wmma_gemm.cuh"
    #include "rocm/hgemm_narrow_rdna.cuh"
#endif

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

// Shared cuBLAS setup: the current handle bound to the stream with host pointer mode, the per-device
// cuBLAS workspace, and the GEMM alpha/beta scalars (1.0 / 0.0). Returns the handle.
static cublasHandle_t cublas_setup(cudaStream_t stream, float* alpha, float* beta)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(handle, ws, WORKSPACE_SIZE);
    *alpha = 1.0f;
    *beta = 0.0f;
    return handle;
}

#if !defined(USE_ROCM)
// Maxwell (CC 5) reconstruct GEMM: an explicit FP32 SGEMM path so the result does not depend on the
// mixed-precision 16F-in / C32-accumulate cuBLAS behavior. FP16 inputs are cast to FP32 and the FP32
// result is written back into c (downcast for an FP16 c, strided for a non-contiguous c). The layout
// (operand roles, dims, leading dims, transpose) mirrors the GemmEx each path already uses, so the
// numbers match that path. CUDA-only: the ROCm build always uses hipBLAS here.

// Dense: mirrors the cublasGemmEx call in hgemm_gemmex_impl, using cublasSgemm.
static void hgemm_sgemm_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    TORCH_CHECK(c.dim() == 2, "hgemm_sgemm: dense path expects a 2D output (use hgemm_batched for batched/3D c)");
    auto af = a.to(at::kFloat);
    auto bf = b.to(at::kFloat);

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    // FP32 output laid out to match c's row stride, so the GEMM writes exactly where GemmEx would
    auto cf = at::empty({size_m, c_stride_m}, c.options().dtype(at::kFloat)).narrow(1, 0, size_n);

    float alpha_, beta_;
    cublasHandle_t cublas_handle = cublas_setup(stream, &alpha_, &beta_);
    auto r = cublasSgemm
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, (const float*) bf.data_ptr(), size_n,
                 (const float*) af.data_ptr(), size_k,
        &beta_,  (float*) cf.data_ptr(), (int) c_stride_m
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
    c.copy_(cf);
}

// Batched: mirrors the cublasGemmStridedBatchedEx call in hgemm_batched, using CUDA_R_32F.
static void hgemm_batched_sgemm_impl
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c,
    cudaStream_t stream
)
{
    auto af = a.to(at::kFloat);
    auto wf = w.to(at::kFloat);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);

    float alpha_, beta_;
    cublasHandle_t cublas_handle = cublas_setup(stream, &alpha_, &beta_);
    auto cf = at::empty({batch, size_m, size_n}, c.options().dtype(at::kFloat));
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, wf.data_ptr(), CUDA_R_32F, size_n, (long long) size_k * size_n,
                 af.data_ptr(), CUDA_R_32F, size_k, (long long) size_m * size_k,
        &beta_,  cf.data_ptr(), CUDA_R_32F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
    c.copy_(cf);
}
#endif

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    int device;
    cudaGetDevice(&device);
#if !defined(USE_ROCM)
    if (DevCtx::instance().get_cc(device) == CC_MAXWELL)
    {
        hgemm_sgemm_impl(a, b, c, stream);
        return;
    }
#endif

#if defined(USE_ROCM)
    // RDNA: narrow outputs at decode-class row counts on a split-K GEMV (rocm/hgemm_narrow_rdna.cuh)
    if (hgemm_narrow_try(a_ptr, b_ptr, c.data_ptr(), output_fp32, size_m, size_k, size_n, c_stride_m, a.get_device(), stream))
        return;

    // RDNA: the WMMA GEMM (rocm/wmma_gemm.cu) for the shapes its per-arch table routes there, mainly fp32
    // output, for which rocBLAS/hipBLASLt have no matrix-core kernels on these parts
    if (wmma_gemm_try(a_ptr, b_ptr, c.data_ptr(), output_fp32, size_m, size_k, size_n, c_stride_m, a.get_device(), stream))
    {
        cuda_check(cudaPeekAtLastError());
        return;
    }
#endif

    float alpha_, beta_;
    cublasHandle_t cublas_handle = cublas_setup(stream, &alpha_, &beta_);
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

#if defined(USE_ROCM)
// Reconstruct-path GEMM. On CUDA this lives in hgemm_f16acc.cu, which ROCm builds leave out: always hipBLAS
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    hgemm(a, b, c);
}
#endif

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS. The ROCm build
    // has no fp16-accumulator kernel (hgemm_f16acc.cu is CUDA-only)
#if !defined(USE_ROCM)
    if (hgemm_f16acc_try(a, w, c)) return;
#endif

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    int device;
    cudaGetDevice(&device);
#if !defined(USE_ROCM)
    if (DevCtx::instance().get_cc(device) == CC_MAXWELL)
    {
        hgemm_batched_sgemm_impl(a, w, c, stream);
        return;
    }
#endif

    float alpha_, beta_;
    cublasHandle_t cublas_handle = cublas_setup(stream, &alpha_, &beta_);
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
