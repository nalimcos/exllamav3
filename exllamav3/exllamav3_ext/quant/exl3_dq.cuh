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

template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag)
{
    int b0 = (t_offset + 257) * bits - 16;      // start of first word
    int b1 = b0 + 3 * bits;                     // start of last word
    int b2 = b1 + 16;                           // end of last word
    int i0 = b0 / 32;                           // uint32 containing first bit of first word
    int i2 = (b2 - 1) / 32;                     // uint32 containing last bit of last word, may be == i0
    int s2 = (i2 + 1) * 32 - b2;                // shift value to align last word to 32-bit boundary

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2)            & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits)     & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    half2 d0d1 = decode_3inst_2<cb>(w0, w1);
    half2 d2d3 = decode_3inst_2<cb>(w2, w3);
    frag[0] = d0d1;
    frag[1] = d2d3;
}

template <int bits, int cb>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;  // start of first word
        int b1 = b0 + 1 * bits;                         // start of last word
        int b2 = b1 + 16;                               // end of last word
        int i0 = b0 / 32;                               // uint32 containing first bit of first word
        int i2 = (b2 - 1) / 32;                         // uint32 containing last bit of last word, may be == i0
        int s2 = (i2 + 1) * 32 - b2;                    // shift value to align last word to 32-bit boundary

        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        uint32_t w1 = fshift(b, a, s2)        & 0xffff;
        uint32_t w0 = fshift(b, a, s2 + bits) & 0xffff;
        half2 d0d1 = decode_3inst_2<cb>(w0, w1);
        frag[i] = d0d1;
    }
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int b1 = (t_offset + 257) * bits;               // end of first word
    int b0 = b1 - 16;                               // start of first word
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;                               // uint32 containing first bit of word0
    int i2 = (b2 - 1) / 32;                         // uint32 containing last bit of word0, may be == i0
    int s2 = (i2 + 1) * 32 - b2;                    // shift value to align last word to 32-bit boundary

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1)
    {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;
    i0 = (i1 + 7) & 7;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 24));
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);
    BFE16_IMM(w5, b, 2);
    BFE16_IMM(w4, b, 3);
    BFE16_IMM(w3, b, 4);
    BFE16_IMM(w2, b, 5);
    BFE16_IMM(w1, b, 6);
    BFE16_IMM(w0, b, 7);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}


template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits_bfe64(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int i1 = t_offset / 8;
    int i0 = (i1 + 31) % 32;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    uint32_t w7 = bfe64(b, a, 0, 16);
    uint32_t w6 = bfe64(b, a, 4, 16);
    uint32_t w5 = bfe64(b, a, 8, 16);
    uint32_t w4 = bfe64(b, a, 12, 16);
    uint32_t w3 = bfe64(b, a, 16, 16);
    uint32_t w2 = bfe64(b, a, 20, 16);
    uint32_t w1 = bfe64(b, a, 24, 16);
    uint32_t w0 = bfe64(b, a, 28, 16);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

// Half-integer bitrate KA + 0.5 (mul1 codebook): positions alternate KA and KA + 1 bits (odd positions carry the
// extra bit), so 8 consecutive positions starting at an even index take 8 * KA + 4 bits and a lane's window
// ends are fixed. Two groups of four windows, each spanning 18 + 3 * KA <= 27 bits = at most two 32-bit words:
// one funnel shift per group, the other three windows derived by shifting the 32-bit result
template <int KA, int cb>
__device__ __forceinline__ void dq8_half(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    constexpr int bits2 = 2 * KA + 1;                       // bits per two positions
    constexpr int words = 4 * bits2;                        // uint32 per 256-weight tile
    constexpr int gspan = 18 + 3 * KA;                      // bits spanned by four windows
    const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;   // end bit of window 7 (+ one tile for the wrap)
    const int e3 = e7 - 2 * bits2;                                // end bit of window 3
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    {
        int hi = (e7 - 1) / 32, lo = (e7 - gspan) / 32, s = (hi + 1) * 32 - e7;
        uint32_t a = ptr[lo % words], b = ptr[hi % words];
        w7 = fshift(b, a, s); w6 = w7 >> (KA + 1); w5 = w6 >> KA; w4 = w5 >> (KA + 1);
    }
    {
        int hi = (e3 - 1) / 32, lo = (e3 - gspan) / 32, s = (hi + 1) * 32 - e3;
        uint32_t a = ptr[lo % words], b = ptr[hi % words];
        w3 = fshift(b, a, s); w2 = w3 >> (KA + 1); w1 = w2 >> KA; w0 = w1 >> (KA + 1);
    }
    half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    if constexpr (half_k)
    {
        dq8_half<bits, cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 1)
    {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 2)
    {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 3)
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 4)
    {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 5)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 6)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 7)
    {
        dq2x2<bits, cb>(ptr, idx, frag0);
        dq2x2<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 8)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}

// fp32 dequant of a K-bit code via the shared LUT (exl3_dq_lut_build, codebook.cuh): one smem load.
// This is the per-weight dequant the fp32 GEMM/GEMV call on Maxwell, replacing the half2
// decode_3inst_2 the mma path uses. The bit-extraction above (dq_dispatch) is unchanged; the
// consumer extracts the raw K-bit code and feeds it here.
__device__ __forceinline__ float exl3_dq_lut(const float* lut, uint32_t code)
{
    return lut[code];
}

// Extract 4 K-bit codes (w0..w3) at a dequant offset, the dq4/dq2x2 bit-extraction (bits 5..8).
template <int bits>
__device__ __forceinline__ void exl3_dq_codes_4(const uint32_t* ptr, int idx, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3)
{
    if constexpr (bits == 7)
    {
        // dq2x2: two 2-code windows (i = 0, 1); each window gives one half2
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            int b0 = (idx + 2 * i + 257) * bits - 16;
            int b1 = b0 + 1 * bits;
            int b2 = b1 + 16;
            int i0 = b0 / 32, i2 = (b2 - 1) / 32;
            int s2 = (i2 + 1) * 32 - b2;
            uint32_t a = ptr[i0 % (bits * 256 / 32)], b = ptr[i2 % (bits * 256 / 32)];
            uint32_t w1_ = fshift(b, a, s2) & 0xffff;
            uint32_t w0_ = fshift(b, a, s2 + bits) & 0xffff;
            if (i == 0) { w0 = w0_; w1 = w1_; } else { w2 = w0_; w3 = w1_; }
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

// fp32 B-fragment dequant for the Maxwell FMA GEMM. Same bit-extraction as dq_dispatch (the shf/bfe
// code is unchanged) but each extracted K-bit code becomes a fp32 LUT load (exl3_dq_lut) instead of a
// half2 decode_3inst_2. The codes land where dq_dispatch would: frag0 (n-block n2) = the mma B
// fragment {k=2t, k=2t+1, k=2t+8, k=2t+9} (w0..w3), frag1 (n-block n2+1) = the next block (w4..w7).
template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void exl3_dq_dispatch_f32(const uint32_t* ptr, int idx, const float* lut, Vec<float, 4>& frag0, Vec<float, 4>& frag1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (half_k)
    {
        // dq8_half window layout: e7/e3 groups; reuse its funnel-shift extraction
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
        w7 = fshift(b, a, s2); w6 = fshift(b, a, s2 + bits); w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3); w3 = fshift(b, a, s2 + bits * 4); w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6); w0 = fshift(b, a, s2 + bits * 7);
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

    // frag0 (n-block n2) = w0..w3, frag1 (n-block n2+1) = w4..w7
    frag0[0] = exl3_dq_lut(lut, w0 & 0xffff);
    frag0[1] = exl3_dq_lut(lut, w1 & 0xffff);
    frag0[2] = exl3_dq_lut(lut, w2 & 0xffff);
    frag0[3] = exl3_dq_lut(lut, w3 & 0xffff);
    frag1[0] = exl3_dq_lut(lut, w4 & 0xffff);
    frag1[1] = exl3_dq_lut(lut, w5 & 0xffff);
    frag1[2] = exl3_dq_lut(lut, w6 & 0xffff);
    frag1[3] = exl3_dq_lut(lut, w7 & 0xffff);
}