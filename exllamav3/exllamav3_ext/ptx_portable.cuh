#pragma once

// Plain C++ stand-ins for the integer and bit-field PTX instructions used in the kernels: the CUDA
// build keeps its inline PTX (ptx.cuh, codebook.cuh), these replace it on
// ROCm, where inline PTX does not exist. Always defined, so tests/test_ptx_portable.py can check each
// one against the PTX instruction it stands in for on an NVIDIA device. The fp16 section below is
// CUDA-only: portable fused-half-FMA / exp / reciprocal forms for Maxwell, where those intrinsics
// (sm_53+) are not declared.

#include <cstdint>
#include "arch.cuh"

// lop3.b32 d, a, b, c, 0x6a: the lookup table 0x6a is (a & b) ^ c (with a = 0xf0, b = 0xcc, c = 0xaa,
// (0xf0 & 0xcc) ^ 0xaa = 0x6a)
__device__ __forceinline__ uint32_t exl3_lop3_6a(uint32_t a, uint32_t b, uint32_t c)
{
    return (a & b) ^ c;
}

// shf.r.wrap.b32 d, lo, hi, n: the low 32 bits of the 64-bit value hi:lo shifted right by n mod 32
__device__ __forceinline__ uint32_t exl3_shf_r_wrap(uint32_t lo, uint32_t hi, uint32_t n)
{
    uint64_t v = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    return static_cast<uint32_t>(v >> (n & 31));
}

// bfe.u32 d, a, pos, 16, for pos < 32: bits beyond the top of a read as zero
__device__ __forceinline__ uint32_t exl3_bfe_u32_16(uint32_t a, uint32_t pos)
{
    return (a >> pos) & 0xffffu;
}

// bfe.u64 d, a, pos, len, returning the low 32 bits. A zero length, or a start position at or past the
// top of the value, yields zero; bits beyond the top read as zero
__device__ __forceinline__ uint32_t exl3_bfe_u64(uint64_t a, int pos, int len)
{
    if (len <= 0 || pos >= 64) return 0;
    uint64_t v = a >> pos;
    if (len < 64) v &= (1ull << len) - 1ull;
    return static_cast<uint32_t>(v);
}

// mul.lo.u32 / mul.hi.u32
__device__ __forceinline__ uint32_t exl3_mul_lo_u32(uint32_t x, uint32_t y)
{
    return x * y;
}

__device__ __forceinline__ uint32_t exl3_mul_hi_u32(uint32_t x, uint32_t y)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(x) * static_cast<uint64_t>(y)) >> 32);
}

// dp4a.u32.s32 d, a, b, c: the four unsigned bytes of a times the four signed bytes of b, summed onto c
__device__ __forceinline__ int exl3_dp4a_us(uint32_t a, uint32_t b, int c)
{
    // v_dot4_u32_i8 is a gfx11+ instruction (dot8-insts); the invocability test is per target, where
    // __has_builtin would be true for every AMD device pass
    #if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__) && !defined(EXL3_FORCE_SCALAR_DOT)
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4))
            return __builtin_amdgcn_sudot4(false, static_cast<int>(a), true, static_cast<int>(b), c, false);
    #endif
    int d = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        d += static_cast<int>((a >> (8 * i)) & 0xffu) * static_cast<int>(static_cast<int8_t>(b >> (8 * i)));
    return d;
}

#if !defined(USE_ROCM)
#include <cuda_fp16.h>
#include <cmath>

// dp4a.u32.s32 as inline PTX: the __dp4a intrinsic is overload-ambiguous for (u32, u32, int) args
__device__ __forceinline__ int exl3_dp4a_us_asm(uint32_t a, uint32_t b, int c)
{
    int d;
    asm ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// The fused half FMA and the half exp / reciprocal intrinsics are sm_53+ in CUDA; on Maxwell
// (sm_50/sm_52) they are not declared, so the kernels that use them cannot compile. These portable
// forms stand in for them on EXL3_MAXWELL only: the fused forms are a two-step multiply+add (one extra
// rounding) and the transcendental forms round-trip through fp32. They execute at runtime on Maxwell;
// only the mma/ldmatrix stubs in ptx.cuh are compile-only.
__device__ __forceinline__ half exl3_hfma(half a, half b, half c)
{
    return __hadd(__hmul(a, b), c);
}
__device__ __forceinline__ half2 exl3_hfma2(half2 a, half2 b, half2 c)
{
    return __hadd2(__hmul2(a, b), c);
}
__device__ __forceinline__ half exl3_hexp(half x)
{
    return __float2half(expf(__half2float(x)));
}
__device__ __forceinline__ half2 exl3_h2exp(half2 a)
{
    return __halves2half2(__float2half(expf(__half2float(__low2half(a)))),
                          __float2half(expf(__half2float(__high2half(a)))));
}
__device__ __forceinline__ half exl3_hrcp(half x)
{
    return __float2half(1.0f / __half2float(x));
}
__device__ __forceinline__ half2 exl3_h2rcp(half2 a)
{
    return __halves2half2(__float2half(1.0f / __half2float(__low2half(a))),
                          __float2half(1.0f / __half2float(__high2half(a))));
}
#endif

// Call-site selectors: the sm_53+ intrinsic on sm_53+ (and ROCm), the portable form on Maxwell
#if EXL3_MAXWELL
    #define EXL3_HFMA(a, b, c) exl3_hfma(a, b, c)
    #define EXL3_HFMA2(a, b, c) exl3_hfma2(a, b, c)
    #define EXL3_HEXP(x) exl3_hexp(x)
    #define EXL3_H2EXP(x) exl3_h2exp(x)
    #define EXL3_HRCP(x) exl3_hrcp(x)
    #define EXL3_H2RCP(x) exl3_h2rcp(x)
#else
    #define EXL3_HFMA(a, b, c) __hfma(a, b, c)
    #define EXL3_HFMA2(a, b, c) __hfma2(a, b, c)
    #define EXL3_HEXP(x) hexp(x)
    #define EXL3_H2EXP(x) h2exp(x)
    #define EXL3_HRCP(x) hrcp(x)
    #define EXL3_H2RCP(x) h2rcp(x)
#endif
// dp4a is sm_61+; the portable byte-sum stands in on Maxwell and ROCm
#if defined(USE_ROCM) || EXL3_MAXWELL
    #define EXL3_DP4A(a, b, c) exl3_dp4a_us(a, b, c)
#else
    #define EXL3_DP4A(a, b, c) exl3_dp4a_us_asm(a, b, c)
#endif
