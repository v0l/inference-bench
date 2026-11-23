#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ uint4 ld16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ld8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

template <int MODE>
__global__ void __launch_bounds__(256) k(const unsigned char* w, int n, unsigned* out) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned acc = 0;
    constexpr int RPW = 2;
    for (int row = (blockIdx.x * 8 + warp) * RPW; row < n; row += gridDim.x * 8 * RPW) {
        if (MODE == 0) {
            for (int r = 0; r < 2; r++) {
                const unsigned char* sb = w + (size_t)(row + r) * 896 + (lane >> 3) * 224;
                int h = (lane & 7) >> 2, l0 = (lane & 3) * 8;
                uint2 a = ld8(sb + 64 * h + l0), b = ld8(sb + 64 * h + 32 + l0), q = ld8(sb + 128 + 32 * h + l0);
                unsigned s1 = *(const unsigned*)(sb + 192 + 8 * h), s2 = *(const unsigned*)(sb + 196 + 8 * h);
                unsigned short d = *(const unsigned short*)(sb + 208);
                acc ^= a.x ^ a.y ^ b.x ^ b.y ^ q.x ^ q.y ^ s1 ^ s2 ^ d;
            }
        } else {
            int r = lane >> 4, ll = lane & 15;
            const unsigned char* sb = w + (size_t)(row + r) * 896 + (ll >> 2) * 224;
            int h = (ll & 3) >> 1, l0 = (ll & 1) * 16;
            uint4 a = ld16(sb + 64 * h + l0), b = ld16(sb + 64 * h + 32 + l0), q = ld16(sb + 128 + 32 * h + l0);
            unsigned s1 = *(const unsigned short*)(sb + 192 + 8 * h + l0 / 16 * 2), s2 = *(const unsigned short*)(sb + 196 + 8 * h + l0 / 16 * 2);
            unsigned short d = *(const unsigned short*)(sb + 208);
            acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ q.x ^ q.y ^ q.z ^ q.w ^ s1 ^ s2 ^ d;
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}

template <int MODE>
void run(const unsigned char* w, int n, unsigned* o, int sms, const char* name) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int per : {2, 4, 8}) {
        int blocks = (n + 15) / 16;
        if (blocks > sms * per) blocks = sms * per;
        float best = 1e9f;
        for (int r = 0; r < 30; r++) {
            float ms;
            cudaEventRecord(s);
            k<MODE><<<blocks, 256>>>(w, n, o);
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms, s, e);
            best = ms < best ? ms : best;
        }
        printf("%s blocks/sm=%d: %.0f us %.0f GB/s\n", name, per, best * 1000, (double)n * 896 / best / 1e6);
    }
}

int main() {
    int n = 151936;
    unsigned char* w;
    unsigned* o;
    cudaMalloc(&w, (size_t)n * 896);
    cudaMalloc(&o, 4);
    cudaMemset(w, 3, (size_t)n * 896);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    run<0>(w, n, o, sms, "8B  layout ");
    run<1>(w, n, o, sms, "16B layout ");
    return 0;
}
