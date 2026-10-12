#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

// The FMA matmul is dense-only (TILEBLOCKS_M == 1); the multi-row MoE tile is not instantiated on Maxwell.
#if EXL3_MAXWELL
fp_exl3_moe_kernel exl3_moe_kernel_h3_n128_cb2_m32() { return nullptr; }
#else
fp_exl3_moe_kernel exl3_moe_kernel_h3_n128_cb2_m32() { return exl3_moe_kernel<3, 128, 2, 32, true>; }
#endif
