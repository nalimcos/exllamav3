#pragma once

// Launch of the grid-synchronizing EXL3 GEMM kernels. ROCm's cooperative launch faults in the runtime when these
// are submitted back to back, so ROCm launches them as plain kernels and the kernels synchronize through a device
// barrier instead (EXL3_GRID_SYNC, exl3_gemm_kernel.cuh). Maxwell (CC 5.x) has no cooperative launch either, so it
// takes the same plain-launch path, bounding the grid to the co-resident count so every block is resident. The
// other CUDA tiers use the cooperative launch and grid.sync().
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "exl3_devctx.cuh"

// Bounded non-cooperative launch of a whole-grid-co-resident kernel. The co-residency bound is MANDATORY: a bare
// volatile+fence spin (group_barrier) deadlocks if a block is evicted, so bounding the grid to maxb * num_sms
// guarantees every block is resident and the barrier always terminates. Two forms: a 1-D int grid (the dense
// GEMM) and a dim3 grid (the mgemm, whose y/z shape is kept and only x is scaled to meet the bound).
inline cudaError_t exl3_launch_grid_bounded(void* kernel, int grid, int block, void** args, size_t smem, cudaStream_t stream, int device)
{
    int maxb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, kernel, block, smem);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    if (maxb > 0 && grid > maxb * num_sms) grid = maxb * num_sms;
    return cudaLaunchKernel(kernel, dim3(grid), block, args, smem, stream);
}
inline cudaError_t exl3_launch_grid_bounded(void* kernel, dim3 grid, int block, void** args, size_t smem, cudaStream_t stream, int device)
{
    int maxb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, kernel, block, smem);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    int total = grid.x * grid.y * grid.z;
    if (maxb > 0 && total > maxb * num_sms)
    {
        int yz = grid.y * grid.z;
        grid.x = (yz > 0) ? (maxb * num_sms) / yz : 0;
    }
    return cudaLaunchKernel(kernel, grid, block, args, smem, stream);
}

// Arch-aware launch. Decided at runtime by the device tier (host code has no __CUDA_ARCH__): Maxwell (CC_MAXWELL)
// launches plain with the grid bounded to the co-resident count; the other CUDA tiers use the cooperative launch.
inline cudaError_t exl3_gemm_launch(void* kernel, int grid, int block, void** args, size_t smem, cudaStream_t stream)
{
    int device;
    cudaGetDevice(&device);
    if (DevCtx::instance().get_cc(device) == CC_MAXWELL)
        return exl3_launch_grid_bounded(kernel, grid, block, args, smem, stream, device);
    return cudaLaunchCooperativeKernel(kernel, grid, block, args, smem, stream);
}
inline cudaError_t exl3_gemm_launch(void* kernel, dim3 grid, int block, void** args, size_t smem, cudaStream_t stream)
{
    int device;
    cudaGetDevice(&device);
    if (DevCtx::instance().get_cc(device) == CC_MAXWELL)
        return exl3_launch_grid_bounded(kernel, grid, block, args, smem, stream, device);
    return cudaLaunchCooperativeKernel(kernel, grid, block, args, smem, stream);
}

#define EXL3_COOP_LAUNCH(kernel, grid, block, args, smem, stream) exl3_gemm_launch(kernel, grid, block, args, smem, stream)

struct CoopAutotuneCandidate
{
    void* kernel;
    int block_dim;
    int max_num_sms;
    int max_concurrency;
    int total_sms;
    int tag;
};

struct CoopAutotuneLaunch
{
    void* kernel;
    int block_dim;
    int num_sms;
    int concurrency;
    int tag;
};

class CoopKernelAutotuner
{
public:
    static bool launch_locked
    (
        uint64_t hash,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        CoopAutotuneLaunch* launch_config = nullptr
    );

    static CoopAutotuneLaunch launch
    (
        uint64_t hash,
        const std::vector<CoopAutotuneCandidate>& candidates,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        size_t numel_B = 1e9
    );
};
