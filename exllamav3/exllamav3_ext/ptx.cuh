#pragma once
#include "arch.cuh"
#include "ptx_portable.cuh"

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

#if defined(USE_ROCM)
    // Tensor-core fragment ops and the async copies are emulated on ROCm (see the file for the layouts)
    #include "rocm/ptx_rocm.cuh"
#endif

// m8n8k4 tensor core matmul (emulated on Ampere and later), don't use
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type

#if !defined(USE_ROCM)
__device__ inline void ptx_mma_m8n8k4
(
    const Vec<half2, 2>& frag_a,
    const Vec<half2, 2>& frag_b,
    Vec<float, 8>& frag_c
)
{
#if EXL3_MAXWELL
    // no tensor cores on Maxwell; exl3_no_tc keeps this path unlaunched
#else
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),"=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])

        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7])
    );
#endif
}
#endif

// m16n8k16 tensor core matmul
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

// FP16 @ FP16 + FP32 -> FP32
//
// Turing has no m16n8k16. The k=16 operation is the sum of two k=8 operations over disjoint
// halves of the k dimension, and the register-to-element mapping of m16n8k8 is exactly the
// low half of m16n8k16's: A {a0,a1} covers k=0..7 and {a2,a3} covers k=8..15, B b0 then b1,
// C identical. Chaining two mmas through the same accumulator computes the same products over
// the same operands.
//
// It is not guaranteed bit-identical: PTX does not specify the internal accumulation order of
// a k=16 step, and the split forces a rounding of the partial sum at the k=8 boundary that the
// fused form need not perform. The difference is at most one extra rounding per 16-element dot
// product, far below the quantization noise EXL3 already carries. tests/test_sm75_gemm.py
// bounds it against a dequantize-then-matmul reference.
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
#if defined(USE_ROCM)
    exl3_mma_m16n8k16_f32(frag_a, frag_b, frag_c);
#elif EXL3_MAXWELL
    // no tensor cores on Maxwell; exl3_no_tc keeps this path unlaunched
#else
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

#if EXL3_SM75
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[2]), "r"(a[3]),
           "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
#else
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
#endif
#endif
}

// FP16 @ FP16 + FP16 -> FP16
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
#if defined(USE_ROCM)
    exl3_mma_m16n8k16_f16(frag_a, frag_b, frag_c);
#elif EXL3_MAXWELL
    // no tensor cores on Maxwell; exl3_no_tc keeps this path unlaunched
#else
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    const uint32_t* d = reinterpret_cast<const uint32_t*>(&frag_c);

#if EXL3_SM75
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]),
           "r"(d[0]), "r"(d[1])
    );
    asm
    (
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[2]), "r"(a[3]),
           "r"(b[1]),
           "r"(d[0]), "r"(d[1])
    );
#else
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "r"(d[0]), "r"(d[1])
    );
#endif
#endif
}

// Global barrier

__device__ inline void barrier_acquire
(
    int* lock,
    int stage
)
{
#if defined(USE_ROCM)
    if (threadIdx.x == 0)
    {
        while (__hip_atomic_load(lock, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != stage)
            __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    // Every wave acquires at device scope, so none reads the other blocks' data through a stale per-CU cache
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#elif EXL3_MAXWELL
    if (threadIdx.x == 0)
    {
        volatile int* lp = lock;
        volatile int state = -1;
        do
        {
            state = *lp;  // volatile load bypasses L1, no ordered atomic
        }
        while (state != stage);
    }
    __syncthreads();
    // the fence gives acquire ordering: it orders every thread's post-barrier reads after the wait
    __threadfence();
#else
    if (threadIdx.x == 0)
    {
        volatile int state = -1;
        do
        {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        }
        while (state != stage);
    }
    __syncthreads();
#endif
}

__device__ inline void barrier_release
(
    int* lock,
    int val,
    bool reset
)
{
#if defined(USE_ROCM)
    // Every wave releases its stores at device scope before the block barrier, rather than relying on the one
    // thread that signals to publish the whole block's writes
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            __hip_atomic_store(lock, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            return;
        }
        __hip_atomic_fetch_add(lock, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
#elif EXL3_MAXWELL
    // __threadfence() publishes this block's writes before the counter is read (release ordering);
    // atomicAdd is a relaxed device atomic (sm_50)
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            *lock = 0;
            return;
        }
        atomicAdd(lock, val);
    }
#else
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            *lock = 0;
            return;
        }
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
#endif
}

// Load global to shared memory, predicated. Seems to produce incorrect code when compiling for Blackwell, but
// `if (...) cp_async(...)` compiles to a predicated instruction anyway

__device__ inline void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
#if EXL3_MAXWELL || EXL3_SM75 || defined(USE_ROCM)
    if (pred)
    {
        uint4 v = *reinterpret_cast<const uint4*>(glob_ptr);
        *reinterpret_cast<uint4*>(smem_ptr) = v;
    }
#else
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
#endif
}

// Load global to shared memory
//
// Turing (sm_75) has no cp.async, so the copy degrades to a synchronous 16 B load/store pair
// routed through registers. Correctness is unaffected: cp_async_wait() below becomes a no-op,
// but every consumer of a staged tile already passes a __syncthreads() (see wait_stage() in
// exl3_gemm_inner.cuh) before reading it, and a synchronous store has completed by then. The
// cost is the lost global->shared overlap, which is a throughput loss, not a hazard.

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
#if EXL3_MAXWELL || EXL3_SM75 || defined(USE_ROCM)
    uint4 v = *reinterpret_cast<const uint4*>(glob_ptr);
    *reinterpret_cast<uint4*>(smem_ptr) = v;
#else
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
#endif
}

// Load global to shared memory with cache hint to evict data from L2 ASAP

__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
#if EXL3_MAXWELL || EXL3_SM75 || defined(USE_ROCM)
    // No cp.async and no createpolicy on Maxwell/Turing; the L2 hint is only an optimization
    uint4 v = *reinterpret_cast<const uint4*>(glob_ptr);
    *reinterpret_cast<uint4*>(smem_ptr) = v;
#else
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = 16;
    asm volatile
    (
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
#endif
}

// Async copy fence, commit all pending async copies

__device__ inline void cp_async_fence()
{
#if !EXL3_MAXWELL && !EXL3_SM75 && !defined(USE_ROCM)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until at most n async groups are still pending.

template <int n>
__device__ inline void cp_async_wait()
{
#if !EXL3_MAXWELL && !EXL3_SM75 && !defined(USE_ROCM)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
#endif
}

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
#if defined(USE_ROCM)
    exl3_ldsm4(frag_a, smem_ptr);
#elif EXL3_MAXWELL
    // no ldmatrix on Maxwell; exl3_no_tc keeps this path unlaunched
#else
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
#endif
}

__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y)
{
#if defined(USE_ROCM)
    return exl3_mul_lo_u32(x, y);
#else
    uint32_t w;
    asm volatile
    (
        "mul.lo.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
#endif
}

__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
#if defined(USE_ROCM)
    return exl3_mul_hi_u32(x, y);
#else
    uint32_t w;
    asm volatile
    (
        "mul.hi.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
#endif
}

// Memory ops

#if defined(USE_ROCM)

// .wt (write-through) stores and .cv (don't-cache) loads exist so a flag or payload written here is seen by, or
// re-read from, another device or the host. Relaxed system-scope atomics give that property on AMD. The 128-bit
// forms are four 32-bit accesses; the PTX vector access is not single-copy atomic as a whole either.

__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
    uint32_t* q = reinterpret_cast<uint32_t*>(p);
    stg_wt_u32(q + 0, v.x);
    stg_wt_u32(q + 1, v.y);
    stg_wt_u32(q + 2, v.z);
    stg_wt_u32(q + 3, v.w);
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
    return make_uint4(ldg_cv_u32(q + 0), ldg_cv_u32(q + 1), ldg_cv_u32(q + 2), ldg_cv_u32(q + 3));
}

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Global time in nanoseconds. wall_clock64() is the constant-rate counter (100 MHz on RDNA)

__device__ __forceinline__ uint64_t globaltimer_ns()
{
    return wall_clock64() * 10;
}

#else

__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
#if EXL3_MAXWELL
    // Maxwell emulation of the system-scope write-through: a system-scope release (fence before the
    // store), consistent with stg_release_sys
    __threadfence_system();
    *(volatile uint32_t*) p = v;
#else
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
#endif
}

__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
#if EXL3_MAXWELL
    // Maxwell emulation of the system-scope write-through: a system-scope release (fence before the
    // store), consistent with stg_release_sys. Lane-wise: a volatile uint4 lvalue can't bind to the
    // copy-assignment (nvcc: no operator = matches these operands)
    __threadfence_system();
    volatile uint4* pv = reinterpret_cast<volatile uint4*>(p);
    pv->x = v.x;
    pv->y = v.y;
    pv->z = v.z;
    pv->w = v.w;
#else
    asm volatile ("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                  :: "l"(p),
                     "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#endif
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
#if EXL3_MAXWELL
    // volatile load + sys-fence: real acquire at system scope, no stale L1 (cross-GPU pinned-host flag)
    uint32_t v = *(volatile uint32_t*) p;
    __threadfence_system();
    return v;
#else
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
#endif
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
#if EXL3_MAXWELL
    // volatile per-lane load + sys-fence: real acquire at system scope, no stale L1 (cross-GPU
    // pinned-host flag). Lane-wise: a volatile uint4 lvalue can't bind to the const uint4& copy ctor.
    const volatile uint4* pv = reinterpret_cast<const volatile uint4*>(p);
    uint4 v;
    v.x = pv->x;
    v.y = pv->y;
    v.z = pv->z;
    v.w = pv->w;
    __threadfence_system();
    return v;
#else
    uint4 v;
    asm volatile ("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
#endif
}

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
#if EXL3_MAXWELL
    // volatile load + sys-fence: acquire ordering at system scope
    uint32_t v = *(volatile uint32_t*) p;
    __threadfence_system();
    return v;
#else
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
                 : "=r"(v) : "l"(p) : "memory");
    return v;
#endif
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
#if EXL3_MAXWELL
    // volatile load + sys-fence: acquire ordering at system scope
    uint64_t v = *(volatile uint64_t*) p;
    __threadfence_system();
    return v;
#else
    uint64_t v;
    asm volatile("ld.global.acquire.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#endif
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
#if EXL3_MAXWELL
    // sys-fence + volatile store: release ordering at system scope
    __threadfence_system();
    *(volatile uint32_t*) p = v;
#else
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#endif
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
#if EXL3_MAXWELL
    // sys-fence + volatile store: release ordering at system scope
    __threadfence_system();
    *(volatile uint64_t*) p = v;
#else
    asm volatile("st.global.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
#endif
}

// Global time in nanoseconds

__device__ __forceinline__ uint64_t globaltimer_ns()
{
#if EXL3_MAXWELL
    // best-effort watchdog ONLY - clock64 is a per-SM cycle counter and is NOT a cross-SM/cross-GPU
    // equivalent of %%globaltimer; acceptable because this is compile-only and the watchdog is best-effort
    return clock64();
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

#endif

// Bitfield stuff

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#if defined(USE_ROCM)
    return exl3_bfe_u64(value, offset, length);
#else
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
#endif
}

#if defined(USE_ROCM)
    #define FSHF_IMM(dst, lo, hi, imm) (dst) = exl3_shf_r_wrap((lo), (hi), (imm))
    #define BFE16_IMM(dst, src, imm) (dst) = exl3_bfe_u32_16((src), (imm))
#else
    #define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
    #define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif

// Backoff/yield between barrier polls. On Maxwell this is a duration-agnostic bounded spin
// (__nanosleep is sm_70+); the surrounding while-loop provides correctness, the spin is only a
// backoff hint. The ns argument is the pacing duration on sm_70+ and is ignored on Maxwell.
__device__ __forceinline__ void exl3_sleep(uint32_t ns)
{
#if EXL3_MAXWELL
    for (volatile uint32_t i = 0; i < 32; ++i) {}
#else
    __nanosleep(ns);
#endif
}

// Inter-block barrier. Co-residency-bounded: the grid must be sized so every group is resident
// (see exl3_launch_grid_bounded), otherwise a bare volatile+fence spin deadlocks on an evicted block.
// Non-cooperative (no grid.sync), device-atomic, sense-reversing.

__device__ inline void group_barrier
(
    int group_id,
    int group_size,
    int* barrier_counters_sense  // length 2*max(group_id). odd positions are flipped after sync (sense)
)
{
#if defined(USE_ROCM)
    // As in barrier_release/barrier_acquire: every wave publishes and refreshes at device scope
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#endif
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int* counter = &barrier_counters_sense[group_id * 2];
        volatile int* sense = &barrier_counters_sense[group_id * 2 + 1];

        // Release: publish this block's pre-barrier writes before the arrival
        __threadfence();
        int old_sense = *sense;
        int old = atomicAdd(counter, 1);  // red.global.add, unscoped (sm_50); the scoped form needs sm_60

        if (old == group_size - 1)
        {
            *counter = 0;
            *sense = 1 - old_sense;
        }
        else
        {
            while (*sense == old_sense) exl3_sleep(32);
        }
        // Acquire: order this block's post-barrier reads after the wait/flip
        __threadfence();
    }

    __syncthreads();
#if defined(USE_ROCM)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
}
