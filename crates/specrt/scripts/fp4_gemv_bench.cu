#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

constexpr int HIDDEN = 5120;
constexpr int INTER = 17408;
constexpr int FP4_GROUP = 16;
constexpr int FP4_HEADER = 16;
constexpr int LINEAR_WARPS = 8;

__device__ __forceinline__ float e4m3_to_f32(unsigned b) {
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * __uint_as_float(0x7b800000u);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float e2m1_to_f32(unsigned n) {
    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);
}
__device__ __forceinline__ uint2 load_stream8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 load_stream16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <int K, int ROWS, int U>
struct Tile { uint2 q[ROWS][U]; unsigned char sc[ROWS][U]; };

template <int K, int ROWS, int U>
__device__ __forceinline__ void tile_load(Tile<K, ROWS, U>& t, const unsigned char* w, const unsigned char* scales, const int* rows, int g0, int lane) {
    constexpr int KG = K / FP4_GROUP;
    const unsigned char* sb = scales + FP4_HEADER;
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int u = 0; u < U; u++) {
            int g = g0 + u * 32 + lane;
            if (g < KG) {
                t.q[r][u] = load_stream8(w + (size_t)rows[r] * (K / 2) + (size_t)g * 8);
                t.sc[r][u] = __ldg(sb + (size_t)rows[r] * KG + g);
            } else {
                t.q[r][u] = make_uint2(0, 0);
                t.sc[r][u] = 0;
            }
        }
}

template <int K, int ROWS, int U>
__device__ __forceinline__ void tile_acc(const Tile<K, ROWS, U>& t, const float* xt, int g0, int lane, float* acc) {
    constexpr int KG = K / FP4_GROUP;
#pragma unroll
    for (int u = 0; u < U; u++) {
        int g = g0 + u * 32 + lane;
        if (g >= KG) break;
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i++) xv[i] = xt[i * KG + g];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned byte = ((i < 4 ? t.q[r][u].x : t.q[r][u].y) >> (8 * (i & 3))) & 0xffu;
                part += e2m1_to_f32(byte & 15u) * xv[2 * i];
                part += e2m1_to_f32(byte >> 4) * xv[2 * i + 1];
            }
            acc[r] += part * e4m3_to_f32(t.sc[r][u]);
        }
    }
}

template <int ROWS, int U, int T>
__global__ void gemv_gate_pf(const unsigned char* w, const unsigned char* s, const float* xt, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    constexpr int WPB = T / 32;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * WPB + warp) * (ROWS / 2);
    int rows[ROWS];
    for (int r = 0; r < ROWS; r++) rows[r] = min(2 * row0 + r, 2 * INTER - 1);
    Tile<HIDDEN, ROWS, U> t;
    tile_load<HIDDEN, ROWS, U>(t, w, s, rows, 0, lane);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[(i & 15) * (HIDDEN / 16) + (i >> 4)] = xt[i];
    __syncthreads();
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;
    tile_acc<HIDDEN, ROWS, U>(t, xs, 0, lane, acc);
    for (int g0 = 32 * U; g0 < HIDDEN / 16; g0 += 32 * U) {
        tile_load<HIDDEN, ROWS, U>(t, w, s, rows, g0, lane);
        tile_acc<HIDDEN, ROWS, U>(t, xs, g0, lane, acc);
    }
    float res[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) res[r] = warp_sum(acc[r]);
    float inv = __ldg((const float*)s);
    if (lane == 0 && row0 < INTER)
        for (int r = 0; r < ROWS; r += 2) out[row0 + r / 2] = res[r] * inv * res[r + 1] * inv;
}

template <int K, int ROWS, int U>
__device__ __forceinline__ void fp4_rows_dot(const unsigned char* w, const unsigned char* scales, const int* rows, const float* xt,
                                             int lane, float* result) {
    constexpr int KG = K / FP4_GROUP;
    const unsigned char* sb = scales + FP4_HEADER;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;
    for (int g0 = 0; g0 < KG; g0 += 32 * U) {
        uint2 q[ROWS][U];
        unsigned char sc[ROWS][U];
#pragma unroll
        for (int r = 0; r < ROWS; r++)
#pragma unroll
            for (int u = 0; u < U; u++) {
                int g = g0 + u * 32 + lane;
                if (g < KG) {
                    q[r][u] = load_stream8(w + (size_t)rows[r] * (K / 2) + (size_t)g * 8);
                    sc[r][u] = __ldg(sb + (size_t)rows[r] * KG + g);
                } else {
                    q[r][u] = make_uint2(0, 0);
                    sc[r][u] = 0;
                }
            }
#pragma unroll
        for (int u = 0; u < U; u++) {
            int g = g0 + u * 32 + lane;
            if (g >= KG) break;
            float xv[16];
#pragma unroll
            for (int i = 0; i < 16; i++) xv[i] = xt[i * KG + g];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float part = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    unsigned byte = ((i < 4 ? q[r][u].x : q[r][u].y) >> (8 * (i & 3))) & 0xffu;
                    part += e2m1_to_f32(byte & 15u) * xv[2 * i];
                    part += e2m1_to_f32(byte >> 4) * xv[2 * i + 1];
                }
                acc[r] += part * e4m3_to_f32(sc[r][u]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) result[r] = warp_sum(acc[r]);
}

template <int ROWS, int U>
__global__ void gemv_down(const unsigned char* w, const unsigned char* s, const float* xt, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * LINEAR_WARPS + warp) * ROWS;
    if (row0 >= HIDDEN) return;
    int rows[ROWS];
    for (int r = 0; r < ROWS; r++) rows[r] = row0 + r;
    float res[ROWS];
    fp4_rows_dot<INTER, ROWS, U>(w, s, rows, xt, lane, res);
    float inv = __ldg((const float*)s);
    if (lane == 0)
        for (int r = 0; r < ROWS; r++) out[row0 + r] = res[r] * inv;
}

template <int ROWS, int U, int P>
__global__ void gemv_gate(const unsigned char* w, const unsigned char* s, const float* xt, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[(i & 15) * (HIDDEN / 16) + (i >> 4)] = xt[i];
    __syncthreads();
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int wpb = blockDim.x >> 5;
    for (int p = 0; p < P; p++) {
    int row0 = ((blockIdx.x * P + p) * wpb + warp) * (ROWS / 2);
    if (row0 >= INTER) return;
    int rows[ROWS];
    for (int r = 0; r < ROWS; r++) rows[r] = 2 * row0 + r;
    float res[ROWS];
    fp4_rows_dot<HIDDEN, ROWS, U>(w, s, rows, xs, lane, res);
    float inv = __ldg((const float*)s);
    if (lane == 0)
        for (int r = 0; r < ROWS; r += 2) out[row0 + r / 2] = res[r] * inv * res[r + 1] * inv;
    }
}

template <typename F>
float time_it(F f) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    return best;
}

int main() {
    const int LAYERS = 16;
    size_t gate_bytes = (size_t)2 * INTER * (HIDDEN / 2), gate_sc = (size_t)2 * INTER * (HIDDEN / 16) + 16;
    size_t down_bytes = (size_t)HIDDEN * (INTER / 2), down_sc = (size_t)HIDDEN * (INTER / 16) + 16;
    unsigned char *gw, *gs, *dw, *ds;
    float *x, *out;
    cudaMalloc(&gw, gate_bytes * LAYERS);
    cudaMalloc(&gs, gate_sc * LAYERS);
    cudaMalloc(&dw, down_bytes * LAYERS);
    cudaMalloc(&ds, down_sc * LAYERS);
    cudaMalloc(&x, INTER * 4);
    cudaMalloc(&out, 2 * INTER * 4);
    cudaMemset(gw, 0x23, gate_bytes * LAYERS);
    cudaMemset(gs, 0x38, gate_sc * LAYERS);
    cudaMemset(dw, 0x23, down_bytes * LAYERS);
    cudaMemset(ds, 0x38, down_sc * LAYERS);
    std::vector<float> hx(INTER);
    for (auto& v : hx) v = rand() / (float)RAND_MAX - 0.5f;
    cudaMemcpy(x, hx.data(), INTER * 4, cudaMemcpyHostToDevice);
    cudaMemset(gs, 0, 4);
    float one = 1.0f;
    for (int l = 0; l < LAYERS; l++) {
        cudaMemcpy(gs + l * gate_sc, &one, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ds + l * down_sc, &one, 4, cudaMemcpyHostToDevice);
    }
    auto report = [&](const char* name, size_t bytes, float ms) { printf("%-28s %7.1f us  %6.0f GB/s\n", name, ms * 1000, bytes / ms / 1e6); };
    int layer = 0;
#define RUN_GATE(R, U, P, T)                                                                                              \
    {                                                                                                                      \
        int rows_per_block = (T / 32) * (R / 2) * P;                                                                       \
        float ms = time_it([&] {                                                                                           \
            for (int l = 0; l < LAYERS; l++)                                                                               \
                gemv_gate<R, U, P><<<(INTER + rows_per_block - 1) / rows_per_block, T>>>(gw + l * gate_bytes, gs + l * gate_sc, x, out); \
        }) / LAYERS;                                                                                                       \
        report("gate_up R=" #R " U=" #U " P=" #P " T=" #T, gate_bytes + gate_sc, ms);                                      \
    }
#define RUN_PF(R, U, T)                                                                                                    \
    {                                                                                                                      \
        int rows_per_block = (T / 32) * (R / 2);                                                                           \
        float ms = time_it([&] {                                                                                           \
            for (int l = 0; l < LAYERS; l++)                                                                               \
                gemv_gate_pf<R, U, T><<<(INTER + rows_per_block - 1) / rows_per_block, T>>>(gw + l * gate_bytes, gs + l * gate_sc, x, out); \
        }) / LAYERS;                                                                                                       \
        report("gate_pf R=" #R " U=" #U " T=" #T, gate_bytes + gate_sc, ms);                                               \
    }
#define RUN_DOWN(R, U)                                                                                                     \
    {                                                                                                                      \
        int rows_per_block = LINEAR_WARPS * R;                                                                             \
        float ms = time_it([&] {                                                                                           \
            for (int l = 0; l < LAYERS; l++)                                                                               \
                gemv_down<R, U><<<(HIDDEN + rows_per_block - 1) / rows_per_block, 256>>>(dw + l * down_bytes, ds + l * down_sc, x, out); \
        }) / LAYERS;                                                                                                       \
        report("down R=" #R " U=" #U, down_bytes + down_sc, ms);                                                           \
    }
    (void)layer;
    RUN_GATE(2,2,1,256) RUN_GATE(2,2,2,256) RUN_GATE(2,2,4,256) RUN_GATE(2,2,1,512) RUN_GATE(2,2,2,512) RUN_GATE(2,2,1,1024) RUN_GATE(2,2,4,128) RUN_GATE(2,2,8,128) RUN_GATE(2,2,1,128) RUN_GATE(2,2,2,64)
    RUN_PF(2,1,256) RUN_PF(2,2,256) RUN_PF(2,2,512) RUN_PF(2,2,1024) RUN_PF(2,4,512) RUN_PF(4,1,512) RUN_PF(2,1,1024)
    RUN_DOWN(1, 2) RUN_DOWN(1, 4) RUN_DOWN(1, 8) RUN_DOWN(2, 2) RUN_DOWN(2, 4) RUN_DOWN(4, 2) RUN_DOWN(4, 4)
    return 0;
}
