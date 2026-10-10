#pragma once

// Launch of the grid-synchronizing EXL3 GEMM kernels. ROCm's cooperative launch faults in the runtime when these
// are submitted back to back, so ROCm launches them as plain kernels and the kernels synchronize through a device
// barrier instead (EXL3_GRID_SYNC, exl3_gemm_kernel.cuh). Every candidate grid is at most one block per
// multiprocessor, so all blocks are resident either way
#if defined(USE_ROCM)
    #define EXL3_COOP_LAUNCH cudaLaunchKernel
#else
    #define EXL3_COOP_LAUNCH cudaLaunchCooperativeKernel
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Non-cooperative launch of a whole-grid-co-resident kernel, with the grid bounded to the max co-resident
// block count. The co-residency bound is MANDATORY: a bare volatile+fence spin (group_barrier) deadlocks if a
// block is evicted, so bounding the grid to maxb * num_sms guarantees every block is resident and the barrier
// always terminates. GEMM already launches at grid = num_sms (within the bound).
// Used by later workstreams (B/C/F); provided here as the non-coop launch primitive (no caller in A).
inline cudaError_t exl3_launch_grid_bounded(void* kernel, int* grid, dim3 block, void** args, size_t smem, cudaStream_t stream, int device)
{
    int maxb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, kernel, block.x, smem);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    *grid = (*grid < maxb * num_sms) ? *grid : maxb * num_sms;
    return cudaLaunchKernel(kernel, dim3(*grid), block, args, smem, stream);
}

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
