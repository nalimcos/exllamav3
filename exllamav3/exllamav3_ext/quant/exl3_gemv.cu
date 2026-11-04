#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_gemv_sk_kernel.cuh"
#include "exl3_devctx.cuh"
#include "coop_autotune.cuh"
#include <map>

/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from exl3_gemm via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the exl3_gemv entry
point. Kernel arguments and graph parameter offsets are identical to exl3_gemm_kernel.

Env: EXL3_GEMV = 0 disables the path, 1/unset = heuristic (default), 2 = use wherever the hard
constraints allow (testing). EXL3_GEMV_GEO = 1 routes Maxwell m = 1 integer-bitrate calls to the
CFG 2 low-warp geometry (4 warps/block, 4 blocks/SM, 4-deep prefetch ring); unset/0 = default.

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

// Maxwell split-k GEMV (exl3_gemv_sk_kernel.cuh). Measured on Tesla M40 (Qwen3.8-27B 3bpw,
// decode bsz 1): the plain GEMV path matches or beats the split-k kernel on every bitrate
// (e.g. 3 bpw 5120x12288: 48.6 vs 40.7 GB/s) once the split-k grid-sync + reduction overhead
// is counted, so the split-k path is opt-in here: EXL3_GEMV_SK = 1 enables it, unset/0 = plain
// GEMV. It remains the only small-m path for 5-8 bpw when enabled.
static int exl3_gemv_sk_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV_SK");
    if (!env) return 0;
    return atoi(env);
}

// -1 = default per bits, 0 = force shuffle extraction, 1 = force smem staging (testing)
static int exl3_gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return atoi(env);
}

// 1 = route Maxwell m=1 integer-bitrate GEMV calls to CFG 2 (4 warps/block, 4 blocks/SM,
// 4-deep prefetch ring) instead of CFG 0. Measured +8% sustained on M40 at 5120x17408 3bpw
// (tmp/geo_bench.py geometry sweep). Unset/0 = default geometry.
static int exl3_gemv_env_geo()
{
    const char* env = std::getenv("EXL3_GEMV_GEO");
    if (!env) return 0;
    return atoi(env);
}

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
    if (K < 2 || K > 4) return -1;
    if (K != 4 && cb == 0) return -1;
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // Maxwell: the GEMV is the decode-speed lever (m <= 8, K 2..4) and is selected for every shape in
    // range — a declined GEMV is dead code on the target, and the grid is capped at co-residency above
    // so correctness holds regardless. CFG 0 (narrow) is the default: it has the register headroom the
    // fp32 MMODE 1 per-row accumulators need. MMODE 0 uses the wide config for large-n / small-k FFN
    // shapes, where its smaller block fits more blocks per SM.
    if (cc == CC_MAXWELL)
    {
        if (size_m > 1) return 0;
        if (size_n >= 8192 && size_k <= 4096) return 1;
        return 0;
    }

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

static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    // CFG 2 (Maxwell low-warp geometry) is instantiated for MMODE 0 only, extraction style
    // fixed to the vectorized smem staging (SMEM_STAGE is inert on the Maxwell VEC_SMEM path)
    if (mmode == 0 && cfg == 2 && !smem)
    {
        SEL(4, 1, false, 0, 2, false) SEL(4, 2, false, 0, 2, false)
        SEL(3, 1, false, 0, 2, false) SEL(3, 2, false, 0, 2, false)
        SEL(3, 1, true,  0, 2, false) SEL(3, 2, true,  0, 2, false)
        SEL(2, 1, false, 0, 2, false) SEL(2, 2, false, 0, 2, false)
        SEL(2, 1, true,  0, 2, false) SEL(2, 2, true,  0, 2, false)
    }
    #undef SEL
    return nullptr;
}

static void* exl3_gemv_select_mkernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_mkernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    // CFG 2 (Maxwell low-warp geometry), MMODE 0 only — see exl3_gemv_select_kernel
    if (mmode == 0 && cfg == 2 && !smem)
    {
        SEL(4, 1, false, 0, 2, false) SEL(4, 2, false, 0, 2, false)
        SEL(3, 1, false, 0, 2, false) SEL(3, 2, false, 0, 2, false)
        SEL(3, 1, true,  0, 2, false) SEL(3, 2, true,  0, 2, false)
        SEL(2, 1, false, 0, 2, false) SEL(2, 2, false, 0, 2, false)
        SEL(2, 1, true,  0, 2, false) SEL(2, 2, true,  0, 2, false)
    }
    #undef SEL
    return nullptr;
}

// Cooperative launch: grids are capped at full co-residency (cached per kernel), and the
// narrow config's co-residency also feeds the shape heuristic
static std::map<void*, int> exl3_gemv_occ_cache[MAX_DEVICES];
static int exl3_gemv_occupancy(int device, void* kernel, int block_dim)
{
    auto& cache = exl3_gemv_occ_cache[device];
    auto it = cache.find(kernel);
    if (it != cache.end()) return it->second;
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
    cache[kernel] = blocks_per_sm;
    return blocks_per_sm;
}

// Shared selection logic for exl3_gemv_try_launch / exl3_gemv_probe / exl3_gemv_batch_launch:
// applies the hard constraints and the shape heuristic, selects the kernel(s) and computes the
// launch geometry. Returns false when the GEMV path does not take this call. Never launches.
// *mkernel_out receives the table-indirect multi-matrix variant when one is instantiated for
// the rate (integer bitrates only; the batched path declines half-integer rates to the MGEMM).
static bool exl3_gemv_pick
(
    int cc,
    int device,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    int mode,
    void** kernel_out,
    void** mkernel_out,
    int* block_dim_out,
    int* cols_out,
    int* max_blocks_out
)
{
    if (half_k)
    {
        if (K < 1 || K > 3 || cb != 2) return false;
    }
    else
    {
        if (K < 2 || K > 8) return false;      // 5-8 bpw (output heads) go to the split-k GEMV
        if (K != 4 && cb == 0) return false;
    }
    if (size_m < 1 || size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    int mmode = size_m == 1 ? 0 : 1;
    int num_sms = DevCtx::instance().get_num_sms(device);

    // Extraction style: shuffle by default, smem staging selectable per call for evaluation
    bool smem = exl3_gemv_env_smem() == 1;

    auto select = [&] (int cfg_) -> void*
    {
        // CFG 2 fixes its extraction style (vectorized smem staging); the env SMEM_STAGE toggle is inert there
        bool smem_sel = cfg_ == 2 ? false : smem;
        return half_k ? exl3_gemv_select_kernel_half(K, c_fp32, mmode, cfg_, smem_sel)
                      : exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg_, smem_sel);
    };
    void* narrow_kernel = select(0);
    if (!narrow_kernel) return false;
    int narrow_coresident = exl3_gemv_occupancy(device, narrow_kernel, 512) * num_sms;

    // Shape heuristic: a half-integer rate K + 0.5 is handled like the integer rate above it (its tile is
    // between the two in bytes; unmeasured, so it inherits the K + 1 envelope)
    int cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, half_k ? K + 1 : K, cb, mode, narrow_coresident);
    if (cfg < 0) return false;

    // EXL3_GEMV_GEO: Maxwell m=1 low-warp geometry (4 warps/block, 4 blocks/SM, 4-deep ring).
    // Only MMODE 0 integer bitrates are instantiated; everything else keeps the default cfg.
    if (cfg >= 0 && cc == CC_MAXWELL && mmode == 0 && !half_k && exl3_gemv_env_geo() == 1)
        cfg = 2;

    void* kernel = cfg == 0 ? narrow_kernel : select(cfg);
    if (!kernel) return false;

    int block_dim = cfg == 0 ? 512 : cfg == 1 ? 256 : 128;
    int cols = cfg == 1 ? 64 : 32;

    int max_blocks = exl3_gemv_occupancy(device, kernel, block_dim) * num_sms;
    if (max_blocks < 1) return false;

    *kernel_out = kernel;
    *mkernel_out = half_k ? nullptr : exl3_gemv_select_mkernel(K, cb, c_fp32, mmode, cfg, cfg == 2 ? false : smem);
    *block_dim_out = block_dim;
    *cols_out = cols;
    *max_blocks_out = max_blocks;
    return true;
}

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
)
{
    void* kernel;
    void* mkernel;
    int block_dim, cols, max_blocks;
    if (!exl3_gemv_pick(cc, device, size_m, size_k, size_n, K, half_k, cb, c_fp32,
                        exl3_gemv_env_mode(), &kernel, &mkernel, &block_dim, &cols, &max_blocks))
        return false;
    // The probe's only consumer is the batched (multi-matrix) path, which needs the mkernel
    return mkernel != nullptr;
}

// Batched (multi-matrix) launch for exl3_mgemm's small-m path: ONE launch over a ONE-dimensional
// grid of gx * bszm blocks of the table-indirect multi-matrix GEMV (slice j = blockIdx.x / gx),
// bounded to full co-residency so the whole-grid barrier inside the kernel is safe. kernel_args
// must be the 15-slot mkernel argument block (A, B_table, C, size_m, size_k, size_n, locks,
// suh_table, A_had, svh_table, bszm_in, gx, n_stride_list, had_src_list, c_list) — the first
// three slots match the plain GEMV/GEMM kernel layout, so graph sites and caller patching are
// identical; the trailing metadata pointers are null for uniform batches. *gx_out receives the
// chosen per-slice grid width (it is an unfixed kernel argument, so the caller supplies storage).
// Returns false (launching nothing) when the shape gate or the co-residency bound declines.
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
)
{
    int cc = DevCtx::instance().get_cc(device);
    void* kernel;
    void* mkernel;
    int block_dim, cols, max_blocks;
    if (!exl3_gemv_pick(cc, device, size_m, size_k, size_n, K, false, cb, c_fp32,
                        exl3_gemv_env_mode(), &kernel, &mkernel, &block_dim, &cols, &max_blocks))
        return false;
    if (!mkernel) return false;

    // The grid barrier inside the mkernel requires every block of the launch co-resident; its
    // register count can differ from the plain kernel's, so re-query occupancy on it directly
    int num_sms = DevCtx::instance().get_num_sms(device);
    max_blocks = exl3_gemv_occupancy(device, mkernel, block_dim) * num_sms;
    if (max_blocks < bszm) return false;

    int groups = size_n / cols;
    int gx = MIN(groups, max_blocks / bszm);
    if (gx < 1) return false;

    *gx_out = gx;
    dim3 grid(gx * bszm);
    cuda_check(EXL3_COOP_LAUNCH(mkernel, grid, block_dim, kernel_args, 0, stream));
    if (launched_kernel) *launched_kernel = mkernel;
    return true;
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

    int mode = force ? 2 : exl3_gemv_env_mode();
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);

    // Maxwell split-k GEMV (MMODE 0, m = 1, activation-as-A; MMODE 1, m = 2..8, weights-as-A k8 TRANS): the
// decode-speed path for 2-8 bpw (output heads), balancing the work per SM for any shape. MMODE 2 (m = 9..16)
// is declined to the dense GEMM; the main GEMV below remains the fallback when the split-k workspace does
// not fit. arg 6 stays the lock buffer — the split-k partials live in it at EXL3_SK_WS_OFFSET, so no
// graph-parameter patching is needed.
    if (cc == CC_MAXWELL && !half_k && !force && exl3_gemv_sk_env_mode() != 0 && size_m >= 1 && size_m <= 8 && K >= 2 && K <= 8)
    {
        const int mmode = size_m == 1 ? 0 : 1;
        void* sk_kernel = exl3_gemv_sk_select_kernel(K, cb, c_fp32, false, mmode);
        if (sk_kernel)
        {
            const int num_sms = DevCtx::instance().get_num_sms(device);
            const int pf = EXL3_GEMV_SK_PF(mmode);
            const int threads = EXL3_GEMV_SK_THREADS_M(mmode, EXL3_GEMV_SK_WNT_M(mmode));
            const int grid = exl3_gemv_occupancy(device, sk_kernel, threads) * num_sms;
            const int warps = grid * (threads / 32);
            const int cols = EXL3_GEMV_SK_WNT_M(mmode) * 16;
            const int kslices = size_k / 16;
            const int total = size_n / cols * kslices;
            // Same geometry as the kernel: the block partials (grid x groups per block range x rows x cols)
            // must fit the workspace
            const int span_w = CEIL_DIVIDE(CEIL_DIVIDE(total, MAX(warps, 1)), pf) * pf;
            const int max_gpb = CEIL_DIVIDE(span_w * (threads / 32), kslices) + 1;
            const size_t ws_floats = (size_t) grid * max_gpb * EXL3_GEMV_SK_ROWS(mmode) * cols;
            if (grid > 0 && size_n % cols == 0 && ws_floats <= EXL3_GEMV_SK_WS_FLOATS)
            {
                cuda_check(EXL3_COOP_LAUNCH(sk_kernel, grid, threads, kernel_args, 0, stream));
                if (launched_kernel) *launched_kernel = sk_kernel;
                return true;
            }
        }
    }

    void* kernel;
    void* mkernel;
    int block_dim, cols, max_blocks;
    if (!exl3_gemv_pick(cc, device, size_m, size_k, size_n, K, half_k, cb, c_fp32, mode,
                        &kernel, &mkernel, &block_dim, &cols, &max_blocks))
        return false;

    int grid = MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    cuda_check(EXL3_COOP_LAUNCH
    (
        kernel,
        grid,
        block_dim,
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
