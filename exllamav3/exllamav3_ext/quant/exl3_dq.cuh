#pragma once

#include "codebook.cuh"

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
     uint64_t merged = ((uint64_t)a << 32) | (uint64_t) b;
     return (uint32_t)(merged >> shift);

    // Conditional funnel shift is somehow no longer faster
    // if (shift < 32) return __funnelshift_r(b, a, shift);
    // return a >> (shift - 32);
}

template <int bits, int cb>
__device__ __forceinline__ half dq(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // bit index, start of word0
    int b1 = b0 + 16;                                   // bit index, end of word0
    int i0 = b0 / 32;                                   // uint32 containing first bit of word0
    int i1 = (b1 - 1) / 32;                             // uint32 containing last bit of word0, may be == i0
    int s0 = (i1 + 1) * 32 - b1;                        // shift value to align word1 to 32-bit boundary

    // Load 32 or 64 bits containing word0
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];

    // Shift into place
    uint32_t w0 = __funnelshift_r(b, a, s0) & 0xffff;
    return decode_3inst<cb>(w0);
}

template <int bits, int cb>
__device__ __forceinline__ half2 dq2(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // bit index, start of word0
    int b1 = b0 + 16;                                   // bit index, end of word0
    int i0 = b0 / 32;                                   // uint32 containing first bit of word0
    int i1 = (b1 - 1) / 32;                             // uint32 containing last bit of word0, may be == i0
    int s0 = (i1 + 1) * 32 - b1;                        // shift value to align word1 to 32-bit boundary

    // Load 32 or 64 bits containing word0
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];

    // Shift into place
    uint32_t w1 = __funnelshift_r(b, a, s0)        & 0xffff;
    uint32_t w0 = __funnelshift_r(b, a, s0 + bits) & 0xffff;
    return decode_3inst_2<cb>(w0, w1);
}

// Extract 4 K-bit codes (w0..w3) at a dequant offset, the dq4/dq2x2 bit-extraction (bits 5..8).
template <int bits>
__device__ __forceinline__ void exl3_dq_codes_4(const uint32_t* ptr, int idx, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3)
{
    if constexpr (bits == 7)
    {
        // dq2x2: two 2-code windows; each window gives two codes
        {
            int b0 = (idx + 257) * bits - 16;
            int b2 = b0 + bits + 16;
            int i0 = b0 / 32, i2 = (b2 - 1) / 32;
            int s2 = (i2 + 1) * 32 - b2;
            uint32_t a = ptr[i0 % (bits * 256 / 32)], b = ptr[i2 % (bits * 256 / 32)];
            w1 = fshift(b, a, s2) & 0xffff;
            w0 = fshift(b, a, s2 + bits) & 0xffff;
        }
        {
            int b0 = (idx + 2 + 257) * bits - 16;
            int b2 = b0 + bits + 16;
            int i0 = b0 / 32, i2 = (b2 - 1) / 32;
            int s2 = (i2 + 1) * 32 - b2;
            uint32_t a = ptr[i0 % (bits * 256 / 32)], b = ptr[i2 % (bits * 256 / 32)];
            w3 = fshift(b, a, s2) & 0xffff;
            w2 = fshift(b, a, s2 + bits) & 0xffff;
        }
    }
    else
    {
        // dq4: one 4-code window
        int b0 = (idx + 257) * bits - 16;
        int b1 = b0 + 3 * bits;
        int b2 = b1 + 16;
        int i0 = b0 / 32, i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[i0 % (bits * 256 / 32)], b = ptr[i2 % (bits * 256 / 32)];
        w3 = fshift(b, a, s2) & 0xffff;
        w2 = fshift(b, a, s2 + bits) & 0xffff;
        w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
        w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    }
}

// Shared bit-extraction: the 8 16-bit sliding windows (w0..w7) at a dequant offset, one shf/BFE pass.
// This is the single extraction the half2 dq_dispatch and the fp32 exl3_dq_dispatch_f32 both consume;
// only the decode differs (decode_3inst_2 vs decode_3inst_f). Bits 5..8 reuse exl3_dq_codes_4. Each
// window holds its 16 bits in the low 16 bits; the consumers re-mask with & 0xffff (a no-op where the
// source already masked).
template <int bits, bool half_k = false>
__device__ __forceinline__ void exl3_dq_codes(const uint32_t* ptr, int idx, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3, uint32_t& w4, uint32_t& w5, uint32_t& w6, uint32_t& w7)
{
    if constexpr (half_k)
    {
        // dq8_half window layout: e7/e3 groups; one funnel shift per group, the other three windows
        // derived by shifting the 32-bit result
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int e7 = ((idx >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        {
            int hi = (e7 - 1) / 32, lo = (e7 - gspan) / 32, s = (hi + 1) * 32 - e7;
            uint32_t a = ptr[lo % (4 * bits2)], b = ptr[hi % (4 * bits2)];
            uint32_t t7 = fshift(b, a, s); w7 = t7; w6 = t7 >> (bits + 1); w5 = w6 >> bits; w4 = w5 >> (bits + 1);
        }
        {
            int hi = (e3 - 1) / 32, lo = (e3 - gspan) / 32, s = (hi + 1) * 32 - e3;
            uint32_t a = ptr[lo % (4 * bits2)], b = ptr[hi % (4 * bits2)];
            uint32_t t3 = fshift(b, a, s); w3 = t3; w2 = t3 >> (bits + 1); w1 = w2 >> bits; w0 = w1 >> (bits + 1);
        }
    }
    else if constexpr (bits == 1)
    {
        int i1 = idx >> 5, i0 = (i1 + 7) & 7;
        uint32_t a = ptr[i0], b = ptr[i1];
        b = fshift(b, a, ((~idx) & 24));
        w7 = b & 0xffff; BFE16_IMM(w6, b, 1); BFE16_IMM(w5, b, 2); BFE16_IMM(w4, b, 3);
        BFE16_IMM(w3, b, 4); BFE16_IMM(w2, b, 5); BFE16_IMM(w1, b, 6); BFE16_IMM(w0, b, 7);
    }
    else if constexpr (bits == 2)
    {
        int i1 = idx >> 4, i0 = (i1 + 15) & 15;
        uint32_t a = ptr[i0], b = ptr[i1];
        b = fshift(b, a, ((~idx) & 8) << 1);
        w7 = b & 0xffff; BFE16_IMM(w6, b, 2); BFE16_IMM(w5, b, 4); BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8); BFE16_IMM(w2, b, 10); BFE16_IMM(w1, b, 12); BFE16_IMM(w0, b, 14);
    }
    else if constexpr (bits == 3)
    {
        int b1 = (idx + 257) * bits;
        int b0 = b1 - 16;
        int b2 = b1 + bits * 7;
        int i0 = b0 / 32, i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[i0 % (bits * 256 / 32)], b = ptr[i2 % (bits * 256 / 32)];
        w7 = fshift(b, a, s2); w6 = w7 >> bits; w5 = w6 >> bits; w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4); w2 = w3 >> bits; w1 = w2 >> bits; w0 = w1 >> bits;
    }
    else if constexpr (bits == 4)
    {
        int i1 = idx >> 3, i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0], b = ptr[i1], s;
        FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff; BFE16_IMM(w6, b, 4); BFE16_IMM(w5, b, 8); BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16); w2 = s & 0xffff; BFE16_IMM(w1, s, 4); BFE16_IMM(w0, s, 8);
    }
    else if constexpr (bits == 5 || bits == 6 || bits == 7 || bits == 8)
    {
        // 4 codes per n-block: w0..w3 at idx (frag0), w4..w7 at idx+4 (frag1)
        exl3_dq_codes_4<bits>(ptr, idx, w0, w1, w2, w3);
        exl3_dq_codes_4<bits>(ptr, idx + 4, w4, w5, w6, w7);
    }
}

// half2 B-fragment dequant (the mma path). Shared bit-extraction (exl3_dq_codes) + the half2
// decode_3inst_2 per code pair.
template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    exl3_dq_codes<bits, half_k>(ptr, idx, w0, w1, w2, w3, w4, w5, w6, w7);
    frag0[0] = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    frag0[1] = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    frag1[0] = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    frag1[1] = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
}

// fp32 B-fragment dequant for the Maxwell FMA GEMM. Same shared bit-extraction as dq_dispatch
// (exl3_dq_codes), but each extracted 16-bit sliding window is dequantized PROCEDURALLY with
// decode_3inst_f<cb> (fp32, bit-identical to the half path) instead of a half2 decode_3inst_2. The
// window spans 16/K consecutive K-bit codes, so it takes up to 2^16 values and a LUT is infeasible.
// The windows land where dq_dispatch would: frag0 (n-block n2) = the mma B fragment (w0..w3),
// frag1 (n-block n2+1) = the next block (w4..w7).
template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void exl3_dq_dispatch_f32(const uint32_t* ptr, int idx, Vec<float, 4>& frag0, Vec<float, 4>& frag1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    exl3_dq_codes<bits, half_k>(ptr, idx, w0, w1, w2, w3, w4, w5, w6, w7);
    frag0[0] = decode_3inst_f<cb>(w0 & 0xffff);
    frag0[1] = decode_3inst_f<cb>(w1 & 0xffff);
    frag0[2] = decode_3inst_f<cb>(w2 & 0xffff);
    frag0[3] = decode_3inst_f<cb>(w3 & 0xffff);
    frag1[0] = decode_3inst_f<cb>(w4 & 0xffff);
    frag1[1] = decode_3inst_f<cb>(w5 & 0xffff);
    frag1[2] = decode_3inst_f<cb>(w6 & 0xffff);
    frag1[3] = decode_3inst_f<cb>(w7 & 0xffff);
}
