#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Unnormalized Sylvester H32, factored into H4 over the low two index bits and
// H8 over the next three. Each eight-lane subgroup owns one 32-value group.
__device__ __forceinline__ void had_4_inreg(float& v0, float& v1, float& v2, float& v3)
{
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v0 = s0 + s1;
    v1 = d0 + d1;
    v2 = s0 - s1;
    v3 = d0 - d1;
}

__device__ __forceinline__ void had_8_subgroup(float& v0, float& v1, float& v2, float& v3, int lane)
{
    #pragma unroll
    for (int i = 1; i < 8; i <<= 1)
    {
        uint64_t p01 = ((uint64_t) __float_as_uint(v0)) | (((uint64_t) __float_as_uint(v1)) << 32);
        uint64_t p23 = ((uint64_t) __float_as_uint(v2)) | (((uint64_t) __float_as_uint(v3)) << 32);
        p01 = __shfl_xor_sync(0xffffffff, p01, i);
        p23 = __shfl_xor_sync(0xffffffff, p23, i);
        uint32_t sfm = (uint32_t) (-(int32_t)(lane & i) >> 31) & 0x80000000u;
        v0 = __uint_as_float(__float_as_uint(v0) ^ sfm) + __uint_as_float((uint32_t) p01);
        v1 = __uint_as_float(__float_as_uint(v1) ^ sfm) + __uint_as_float((uint32_t) (p01 >> 32));
        v2 = __uint_as_float(__float_as_uint(v2) ^ sfm) + __uint_as_float((uint32_t) p23);
        v3 = __uint_as_float(__float_as_uint(v3) ^ sfm) + __uint_as_float((uint32_t) (p23 >> 32));
    }
}
