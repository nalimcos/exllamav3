#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"
#include <map>

/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from exl3_gemm via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the exl3_gemv entry
point. Kernel arguments and graph parameter offsets are identical to exl3_gemm_kernel.

Env: EXL3_GEMV = 0 disables the path, 1/unset = heuristic (default), 2 = use wherever the hard
constraints allow (testing).

Heuristic envelope (measured, RTX 3090, 4 bpw, m <= 8, vs the tuned regular kernel): the narrow
config wins 15-60% at attention-projection sizes (n <= 4096), the wide config wins ~8% at
large-n/small-k FFN sizes. Big-k x big-n shapes lose slightly and fall through to the regular
kernel, as do other architectures (Ada/Blackwell are memory-bound here and keep the regular
kernel), bpw != 4, and m > 8.
*/

static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

// -1 = default per bits, 0 = force shuffle extraction, 1 = force smem staging (testing)
static int exl3_gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return atoi(env);
}

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident, bool core)
{
    if (mode == 0) return -1;
    if (K < 2 || K > 5) return -1;
    if (K != 4 && cb == 0) return -1;
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // CUDA-core GEMV (sm_75 GeForce, g_get_gemv_core): the emulated-mma GEMM is several times
    // slower at every m <= 8 decode shape, so the Ampere-tuned envelope below does not apply.
    // Accept anything the hard constraints allowed, keeping the config/grid split.
    if (core) return size_n <= 8192 ? 0 : 1;

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem, bool core)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_, core_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_ && core == core_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_, false, core_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_, false) SEL(bits_, cb_, false, 0, 1, sm_, false) \
        SEL(bits_, cb_, false, 1, 0, sm_, false) SEL(bits_, cb_, false, 1, 1, sm_, false) \
        SEL(bits_, cb_, true,  0, 0, sm_, false) SEL(bits_, cb_, true,  0, 1, sm_, false) \
        SEL(bits_, cb_, true,  1, 0, sm_, false) SEL(bits_, cb_, true,  1, 1, sm_, false)
    // CORE (CUDA-core __hfma2) instances: MMODE 0 is the m == 1 GEMV, MMODE 1 the 2 <= m <= 8 M-loop
    #define SEL_GRID_CORE(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_, true) SEL(bits_, cb_, false, 0, 1, sm_, true) \
        SEL(bits_, cb_, true,  0, 0, sm_, true) SEL(bits_, cb_, true,  0, 1, sm_, true)
    #define SEL_GRID_CORE_M1(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_, true) SEL(bits_, cb_, false, 1, 1, sm_, true) \
        SEL(bits_, cb_, true,  1, 0, sm_, true) SEL(bits_, cb_, true,  1, 1, sm_, true)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    SEL_GRID_CORE(4, 0, false) SEL_GRID_CORE(4, 1, false) SEL_GRID_CORE(4, 2, false)
    SEL_GRID_CORE(2, 1, false) SEL_GRID_CORE(2, 2, false) SEL_GRID_CORE(2, 1, true) SEL_GRID_CORE(2, 2, true)
    SEL_GRID_CORE(3, 1, false) SEL_GRID_CORE(3, 2, false) SEL_GRID_CORE(3, 1, true) SEL_GRID_CORE(3, 2, true)
    // 5 bpw: 40-word tiles require the shared-memory staging path (SMEM_STAGE true), so only those
    // instances exist. Used from the CUDA-core GEMV (sm_75 GeForce, see exl3_gemv_try_launch).
    SEL_GRID(5, 1, true) SEL_GRID(5, 2, true)
    SEL_GRID_CORE(5, 1, true) SEL_GRID_CORE(5, 2, true)
    // CORE MMODE 1 (2 <= m <= 8), same (bits, cb, smem) coverage the shape heuristic can select
    SEL_GRID_CORE_M1(4, 0, false) SEL_GRID_CORE_M1(4, 1, false) SEL_GRID_CORE_M1(4, 2, false)
    SEL_GRID_CORE_M1(2, 1, false) SEL_GRID_CORE_M1(2, 2, false)
    SEL_GRID_CORE_M1(3, 1, false) SEL_GRID_CORE_M1(3, 2, false)
    SEL_GRID_CORE_M1(5, 1, true) SEL_GRID_CORE_M1(5, 2, true)
    #undef SEL_GRID_CORE_M1
    #undef SEL_GRID_CORE
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

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
)
{
    // Free integer checks first; the env read (~64 ns) and device queries only run for calls
    // that could actually take this path
    if (!has_su_sv) return false;
    if (half_k)
    {
        if (K < 1 || K > 3 || cb != 2) return false;
    }
    else
    {
        if (K < 2 || K > 5) return false;
        if (K != 4 && cb == 0) return false;
    }
    if (size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    int mode = force ? 2 : exl3_gemv_env_mode();
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);
    // if (cc != CC_AMPERE) return false;
    int mmode = size_m == 1 ? 0 : 1;
    int num_sms = DevCtx::instance().get_num_sms(device);

    // Device decision: pick the CUDA-core __hfma2 inner loop on devices where the probe found it
    // faster (sm_75 GeForce). During graph capture the decision must already be cached - probing
    // calls cudaMalloc, which is illegal there - so an uncached decision falls back to mma without
    // being cached, and the captured node keeps the mma kernel.
    bool core_dev = false;
    {
        int cached = g_get_gemv_core_cached(device);
        if (cached >= 0)
        {
            core_dev = cached == 1;
        }
        else
        {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) cudaGetLastError();
            else if (capture == cudaStreamCaptureStatusNone) core_dev = g_get_gemv_core(device) == 1;
        }
    }
    // 5 bpw tiles only exist as smem-staged instances on the CUDA-core path; leave sm_80+ (and the
    // tensor-core mma path) byte-identical by declining them there, as before.
    if (K == 5 && !core_dev) return false;
    // CORE (CUDA-core __hfma2): the m == 1 GEMV (MMODE 0) and the 2 <= m <= 8 M-loop (MMODE 1)
    bool core = core_dev;

    // Cooperative launch: grids are capped at full co-residency (cached per kernel), and the
    // narrow config's co-residency also feeds the shape heuristic
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
        cache[kernel] = blocks_per_sm;
        return blocks_per_sm;
    };

    // Extraction style: shuffle by default, smem staging selectable per call for evaluation.
    // 5 bpw always stages (a 40-word tile has no lane->word shuffle mapping).
    bool smem = K == 5 || exl3_gemv_env_smem() == 1;

    auto select = [&] (int cfg_) -> void*
    {
        return half_k ? exl3_gemv_select_kernel_half(K, c_fp32, mmode, cfg_, smem, core)
                      : exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg_, smem, core);
    };
    void* narrow_kernel = select(0);
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    // Shape heuristic: a half-integer rate K + 0.5 is handled like the integer rate above it (its tile is
    // between the two in bytes; unmeasured, so it inherits the K + 1 envelope)
    int cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, half_k ? K + 1 : K, cb, mode, narrow_coresident, core);
    if (cfg < 0) return false;

    void* kernel = cfg == 0 ? narrow_kernel : select(cfg);
    if (!kernel) return false;

    int block_dim = cfg == 0 ? 512 : 256;
    int cols = cfg == 0 ? 32 : 64;

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    int grid = MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    cuda_check(cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args,
        0,
        stream
    ));

    if (launched_kernel) *launched_kernel = kernel;
    return true;
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")

    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = (half*) OPTPTR(A_had);
    const half* svh_ptr = (const half*) OPTPTR(svh);
    TORCH_CHECK(suh_ptr && A_had_ptr && svh_ptr, "exl3_gemv requires suh, A_had and svh");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    const int tile_u16 = B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1), "exl3_gemv: half-integer bitrates require the mul1 codebook");

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int device;
    cudaGetDevice(&device);
    int* locks = DevCtx::instance().get_locks(device);

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    void* kernel_args[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    bool ok = exl3_gemv_try_launch
    (
        kernel_args, size_m, size_k, size_n, K, half_k, cb, c_fp32,
        true, device, stream, nullptr, true
    );
    TORCH_CHECK(ok, "exl3_gemv: call is not eligible for the GEMV kernel");

    cuda_check(cudaPeekAtLastError());
}
