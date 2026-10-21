#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_gemv_sk_kernel.cuh"
#include "../exl3_gemv.cuh"

// Instances of the sm_75 split-k GEMV (exl3_gemv_sk_kernel.cuh), 4 tiles per warp. m = 1: activation-as-A MMAs,
// prefetch depth 2; 2 <= m <= 8: weights-as-A MMAs, depth 1 (measured best at m = 1, 4 and 8); 9 <= m <= 16:
// weights-as-A over two batch halves, 4-warp blocks, depth 2 (measured best at m = 16)
void* exl3_gemv_sk_select_kernel(int bits, int cb, bool c_fp32, bool c_bf16, int mmode)
{
    #define SEL(bits_, cb_, fp32_, bf16_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && c_bf16 == bf16_) \
            return mmode == 0 ? (void*) exl3_gemv_sk_kernel<bits_, fp32_, bf16_, cb_, 0, EXL3_GEMV_SK_WNT, EXL3_GEMV_SK_PF(0), false> : \
                   mmode == 1 ? (void*) exl3_gemv_sk_kernel<bits_, fp32_, bf16_, cb_, 1, EXL3_GEMV_SK_WNT, EXL3_GEMV_SK_PF(1), true> \
                              : (void*) exl3_gemv_sk_kernel<bits_, fp32_, bf16_, cb_, 2, EXL3_GEMV_SK_WNT_M(2), EXL3_GEMV_SK_PF(2), true>;
    #define SEL3(bits_, cb_) SEL(bits_, cb_, false, false) SEL(bits_, cb_, false, true) SEL(bits_, cb_, true, false)
    SEL3(2, 1) SEL3(2, 2)
    SEL3(3, 1) SEL3(3, 2)
    SEL3(4, 0) SEL3(4, 1) SEL3(4, 2)
    SEL3(5, 0) SEL3(5, 1) SEL3(5, 2)
    SEL3(6, 0) SEL3(6, 1) SEL3(6, 2)
    SEL3(8, 0) SEL3(8, 1) SEL3(8, 2)
    #undef SEL3
    #undef SEL
    return nullptr;
}
