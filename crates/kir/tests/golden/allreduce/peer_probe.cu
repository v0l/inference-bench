typedef unsigned short bf16_t;

__device__ __forceinline__ unsigned ar_load_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void ar_store_release(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned long long ar_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

extern "C" __global__ void __launch_bounds__(1) ar_probe_wait(const unsigned* flag, unsigned magic, unsigned long long timeout_ns, unsigned* result) {
    unsigned long long start = ar_now_ns();
    while (ar_load_acquire(flag) != magic) {
        if (ar_now_ns() - start > timeout_ns) {
            *result = 2;
            return;
        }
    }
    *result = 1;
}

extern "C" __global__ void __launch_bounds__(1) ar_probe_store(unsigned* flag, unsigned magic) {
    __threadfence_system();
    ar_store_release(flag, magic);
}

