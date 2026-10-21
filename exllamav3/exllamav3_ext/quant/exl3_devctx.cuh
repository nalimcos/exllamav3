#pragma once

#include <tuple>
#include <mutex>
#include "../arch.cuh"

// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1] retired groups,
// [2 + group] ticket published to group. Self-resetting, zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)
// Whole-grid barrier of the EXL3 GEMM kernels (counter + sense). ROCm and Maxwell launch without the
// cooperative launch (see EXL3_COOP_LAUNCH in coop_autotune.cuh) and synchronize through this device
// barrier (EXL3_GRID_SYNC in exl3_gemm_kernel.cuh); the other CUDA tiers use grid.sync() and do not
// touch it, but the buffer is always allocated so the offset is uniform.
#define EXL3_GRID_BARRIER_OFFSET (MOE_SCHED_OFFSET + MOE_SCHED_INTS)
#define EXL3_GRID_BARRIER_INTS 2

// Split-k GEMV partials. On Maxwell they live in the lock buffer at EXL3_SK_WS_OFFSET (a single combined
// buffer, so the kernel's arg 6 stays the lock buffer and the whole-grid barrier still reads
// locks + EXL3_GRID_BARRIER_OFFSET); on sm_75 they are a standalone allocation.
#define EXL3_SK_WS_OFFSET (EXL3_GRID_BARRIER_OFFSET + EXL3_GRID_BARRIER_INTS)
#define EXL3_GEMV_SK_WS_FLOATS (5 * 512 * 1024)

// Workspace size
#define WORKSPACE_SIZE (16*1024*1024)

#define MAX_DEVICES 16
// 0 is reserved as the "not cached yet" sentinel of the DevCtx cc cache; tiers start at 1
#define CC_MAXWELL    1
#define CC_OLD        2
#define CC_AMPERE     3
#define CC_ADA        4
#define CC_HOPPER     5
#define CC_BLACKWELL  6

// Maxwell has no tensor cores: decline every mma GEMM/GEMV and the coop MoE.
inline bool exl3_no_tc(int cc) { return cc == CC_MAXWELL; }

// Singleton to manage context for each device. Stores device attributes and a large-enough lock buffer per device
class DevCtx
{
private:
    int num_sms[MAX_DEVICES] = {};
    int cc[MAX_DEVICES] = {};
    int smem_max[MAX_DEVICES] = {};
    void* locks[MAX_DEVICES] = {};
    void* ws[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance();
    int get_num_sms(int device);
    int get_cc(int device);
    // Device capability: dynamic shared memory per block the driver will grant, unclamped
    // (sm_86 reports 99 KB, Turing 64 KB). Use this to decide what a device can do.
    int get_smem_max(int device);
    // What an EXL3 kernel may actually request: the above, capped at the SMEM_MAX the kernels
    // are written against. Use this for cudaFuncSetAttribute and launch parameters.
    int get_smem_request(int device);
    void* get_ws(int device);
    int* get_locks(int device);

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);
int g_get_smem_max(int device);

void prepare_ctx(int device);