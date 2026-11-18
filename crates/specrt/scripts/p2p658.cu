#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__device__ int timed_out;

__global__ void spin_flag(volatile int* flag, int want) {
    long long start = clock64();
    while (*flag != want) {
        if (clock64() - start > 2000000000LL) { atomicExch(&timed_out, 1); return; }
    }
}

__global__ void bump(volatile int* flag, int v) {
    *flag = v;
    __threadfence_system();
}

static double copy_gbs(int dst, int src, bool p2p, size_t bytes, int reps) {
    void *a, *b;
    cudaSetDevice(src);
    cudaMalloc(&a, bytes);
    if (p2p) cudaDeviceEnablePeerAccess(dst, 0);
    cudaSetDevice(dst);
    cudaMalloc(&b, bytes);
    if (p2p) cudaDeviceEnablePeerAccess(src, 0);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaMemcpyPeerAsync(b, dst, a, src, bytes, s);
    cudaEventRecord(t0, s);
    for (int i = 0; i < reps; i++) cudaMemcpyPeerAsync(b, dst, a, src, bytes, s);
    cudaEventRecord(t1, s);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    if (p2p) { cudaDeviceDisablePeerAccess(src); cudaSetDevice(src); cudaDeviceDisablePeerAccess(dst); }
    cudaFree(b);
    cudaSetDevice(src);
    cudaFree(a);
    return (double)bytes * reps / (ms / 1e3) / 1e9;
}

static double bidir_gbs(size_t bytes, int reps) {
    void *a0, *b0, *a1, *b1;
    cudaStream_t s0, s1;
    cudaSetDevice(0); cudaMalloc(&a0, bytes); cudaMalloc(&b0, bytes); cudaDeviceEnablePeerAccess(1, 0); cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaMalloc(&a1, bytes); cudaMalloc(&b1, bytes); cudaDeviceEnablePeerAccess(0, 0); cudaStreamCreate(&s1);
    cudaDeviceSynchronize(); cudaSetDevice(0); cudaDeviceSynchronize();
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        cudaMemcpyPeerAsync(b1, 1, a0, 0, bytes, s0);
        cudaMemcpyPeerAsync(b0, 0, a1, 1, bytes, s1);
    }
    cudaStreamSynchronize(s0); cudaStreamSynchronize(s1);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    return 2.0 * bytes * reps / secs / 1e9;
}

static double flag_latency_us(int reps) {
    int *f0, *f1;
    cudaSetDevice(0); cudaMalloc(&f0, 4); cudaMemset(f0, 0, 4); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaMalloc(&f1, 4); cudaMemset(f1, 0, 4); cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceSynchronize(); cudaSetDevice(0); cudaDeviceSynchronize();
    cudaStream_t s0, s1;
    cudaSetDevice(0); cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaStreamCreate(&s1);
    auto t = std::chrono::steady_clock::now();
    for (int i = 1; i <= reps; i++) {
        cudaSetDevice(0);
        bump<<<1, 1, 0, s0>>>(f1, i);
        spin_flag<<<1, 1, 0, s0>>>(f0, i);
        cudaSetDevice(1);
        spin_flag<<<1, 1, 0, s1>>>(f1, i);
        bump<<<1, 1, 0, s1>>>(f0, i);
    }
    cudaStreamSynchronize(s0); cudaStreamSynchronize(s1);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    int bad0 = 0, bad1 = 0;
    cudaSetDevice(0); cudaMemcpyFromSymbol(&bad0, timed_out, 4);
    cudaSetDevice(1); cudaMemcpyFromSymbol(&bad1, timed_out, 4);
    if (bad0 || bad1) { printf("flag ping-pong timed out: a peer store never became visible (gpu0 %d, gpu1 %d)\n", bad0, bad1); return -1; }
    return secs / reps * 1e6;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 2) { printf("need two GPUs, found %d\n", n); return 1; }
    int can01 = 0, can10 = 0;
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("cudaDeviceCanAccessPeer 0->1 %d, 1->0 %d\n", can01, can10);
    size_t big = 256 << 20;
    printf("unidirectional copy GB/s (256 MiB, dst rows, src cols)\n");
    for (int p2p = 0; p2p < 2; p2p++) {
        printf("  p2p %s\n", p2p ? "on" : "off");
        for (int d = 0; d < 2; d++) {
            printf("   ");
            for (int s = 0; s < 2; s++) printf(" %8.1f", copy_gbs(d, s, p2p, big, 10));
            printf("\n");
        }
    }
    printf("bidirectional p2p GB/s (both directions together): %.1f\n", bidir_gbs(big, 10));
    printf("4 KiB p2p copy GB/s 0->1: %.2f\n", copy_gbs(1, 0, true, 4096, 2000));
    printf("cross-GPU flag ping-pong round trip: %.1f us\n", flag_latency_us(200));
    return 0;
}
