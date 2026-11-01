#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// Largest row count the small-m GEMV path handles (shared by exl3_gemv_kernel.cuh and the
// exl3_mgemm small-m dispatch)
#ifndef EXL3_GEMV_MAX_M
#define EXL3_GEMV_MAX_M 8
#endif

// QTIP-style small-m GEMV path (see exl3_gemv_kernel.cuh). Launched from exl3_gemm when the
// heuristic applies; also exposed directly for testing. Same kernel arguments as
// exl3_gemm_kernel, so graph recording/patching is identical.

// Try to dispatch a GEMM call to the GEMV kernel. Returns false (launching nothing) if the
// call is not eligible. On success *launched_kernel receives the kernel pointer for graph
// recording. `force` bypasses the shape heuristic but not the hard constraints.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
);

// Non-launching eligibility probe for the batched multi-matrix path below: the identical gate
// to exl3_gemv_try_launch (same shape checks, env knobs and heuristic), additionally requiring
// that the table-indirect mkernel is instantiated for the rate (integer bitrates only).
bool exl3_gemv_probe
(
    int cc,
    int device,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32
);

// Batched (multi-matrix) launch: ONE launch of the table-indirect multi-matrix GEMV over a
// 1D grid of gx * bszm blocks. kernel_args is the 12-slot mkernel argument block whose first
// three slots (A, B_table, C) match the plain GEMV/GEMM kernel layout, so graph recording and
// caller patching are identical; slot 12 is the per-slice grid width, written to *gx_out
// before launch. Returns false (launching nothing) on decline.
bool exl3_gemv_batch_launch
(
    void** kernel_args,
    int* gx_out,
    int size_m,
    int size_k,
    int size_n,
    int bszm,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    void** launched_kernel
);

// Kernel instances for the half-integer bitrates (comp_units/exl3_gemv_half_inst.cu)
void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg, bool smem);

// Split-k GEMV instances (comp_units/exl3_gemv_sk_inst.cu); 2-8 bpw, m = 1 (MMODE 0) on Maxwell
void* exl3_gemv_sk_select_kernel(int bits, int cb, bool c_fp32, bool c_bf16, int mmode);

// Direct entry point (testing): errors if the call is not hard-eligible
void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
);
