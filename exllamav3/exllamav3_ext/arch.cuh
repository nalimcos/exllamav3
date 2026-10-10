#pragma once

// Architecture feature gates.
//
// EXL3 was written against sm_80+. Two sub-Turing tiers are handled:
//
//   Maxwell (sm_50/sm_52)  -> EXL3_MAXWELL. Has no tensor cores at all (no mma, no ldmatrix),
//                             no dp4a (sm_61+), no cp.async (sm_80+), and no ordered atomics
//                             (acquire/release/scoped, sm_60+). The host declines the tensor-core
//                             paths (exl3_no_tc) and their device bodies are compile-only stubs;
//                             the dp4a and fp16 math instead run on portable forms. ptx.cuh supplies
//                             the ordered-atomic and async-copy fallbacks (volatile + __threadfence,
//                             sync store).
//   Turing (sm_75)         -> EXL3_SM75. Has tensor cores but only the k=8 mma and no cp.async:
//                             ptx.cuh issues two mma.m16n8k8 and falls back to synchronous 16 B
//                             smem stores (the pipeline's __syncthreads already order the reads).
//
// Both tiers also cap dynamic shared memory below the 90 KB the sm_86 kernels request; the cap
// is a host-side launch parameter resolved per device at runtime (DevCtx::get_smem_max), so
// shapes that do not fit are filtered out of kernel selection rather than failing at launch.
//
// ldmatrix, __nanosleep and __dp4a are sm_75-capable but NOT Maxwell-capable, so each is gated
// on EXL3_MAXWELL separately (see ptx.cuh / the quant kernels). tanh.approx.f32 needs no gate.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
    #define EXL3_MAXWELL 1
    #define EXL3_SM75    0
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    #define EXL3_MAXWELL 0
    #define EXL3_SM75    1
#else
    #define EXL3_MAXWELL 0
    #define EXL3_SM75    0
#endif

// Dynamic shared memory per block the EXL3 kernels are compiled against (SMEM_MAX in
// quant/exl3_kernel_map.cuh): the sm_86 value on CUDA, which only bounds the compile-time
// static_asserts in the inner kernels and must keep accepting every template instantiation, and
// the RDNA workgroup LDS on ROCm. What actually gets requested at launch is the per-device value.
#if defined(USE_ROCM)
#define EXL3_SMEM_MAX_DEFAULT (64 * 1024)
#else
#define EXL3_SMEM_MAX_DEFAULT (90 * 1024)
#endif
#define EXL3_SMEM_MAX_SM75    (64 * 1024)
