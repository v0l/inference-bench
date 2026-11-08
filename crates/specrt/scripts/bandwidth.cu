#include <cstdio>
#include <cuda_runtime.h>

__global__ void read_sum(const uint4* __restrict__ src, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

__global__ void copy(const uint4* __restrict__ src, uint4* dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}

int main() {
    size_t bytes = 2ull << 30;
    size_t n = bytes / 16;
    uint4 *a, *b;
    unsigned* o;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&o, 4);
    cudaMemset(a, 1, bytes);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int pass = 0; pass < 2; pass++) {
        float best_read = 1e9f, best_copy = 1e9f;
        for (int r = 0; r < 10; r++) {
            float ms;
            cudaEventRecord(s);
            read_sum<<<sms * 8, 512>>>(a, n, o);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms, s, e);
            best_read = ms < best_read ? ms : best_read;
            cudaEventRecord(s);
            copy<<<sms * 8, 512>>>(a, b, n);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms, s, e);
            best_copy = ms < best_copy ? ms : best_copy;
        }
        if (pass == 1)
            printf("read %.0f GB/s  copy %.0f GB/s (read + write bytes)\n", bytes / best_read / 1e6, 2.0 * bytes / best_copy / 1e6);
    }
    return 0;
}
