constexpr int MOE_E = {{MOE_E}};
constexpr int MOE_K = {{MOE_K}};
constexpr int MOE_PAIRS = {{MOE_PAIRS}};
constexpr int MOE_INTER = {{MOE_INTER}};
constexpr bool MOE_SHARED = {{MOE_SHARED}};
constexpr bool MOE_NORM_TOPK = {{MOE_NORM_TOPK}};
constexpr bool MOE_SHARED_GATE = {{MOE_SHARED_GATE}};
constexpr bool MOE_SIGMOID = {{MOE_SIGMOID}};
constexpr int MOE_WARPS = 8;

__device__ __forceinline__ float moe_e2m1(unsigned n) {
    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);
}

extern "C" __global__ void __launch_bounds__(256) moe_route(const float* x, const float* ln, const bf16_t* router, const float* shared_gate, float* xn, int* idx, float* coef) {
    __shared__ float xs[HIDDEN];
    __shared__ float probs[MOE_E];
    __shared__ float red[32];
    __shared__ float top_v[MOE_K];
    __shared__ int top_i[MOE_K];
    __shared__ float gate_logit;
    int t = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const float* xr = x + (size_t)t * HIDDEN;
    float ss = 0.0f;
    for (int i = tid; i < HIDDEN; i += 256) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < MOE_WARPS; w++) total += red[w];
    float inv = rsqrtf(total / HIDDEN + EPS);
    for (int i = tid; i < HIDDEN; i += 256) {
        float v = xr[i] * inv * ln[i];
        xs[i] = v;
        xn[(size_t)t * HIDDEN + i] = v;
    }
    __syncthreads();
    for (int e = warp; e < MOE_E; e += MOE_WARPS) {
        const bf16_t* r = router + (size_t)e * HIDDEN;
        float a = 0.0f;
        for (int i = lane; i < HIDDEN; i += 32) a += bf16_to_f32(r[i]) * xs[i];
        a = warp_sum(a);
        if (lane == 0) probs[e] = a;
    }
    if (MOE_SHARED_GATE && warp == MOE_WARPS - 1) {
        float a = 0.0f;
        for (int i = lane; i < HIDDEN; i += 32) a += shared_gate[i] * xs[i];
        a = warp_sum(a);
        if (lane == 0) gate_logit = a;
    }
    __syncthreads();
    if (warp == 0) {
        if (MOE_SIGMOID) {
            for (int e = lane; e < MOE_E; e += 32) probs[e] = 1.0f / (1.0f + expf(-probs[e]));
        } else {
            float m = NEG_INF;
            for (int e = lane; e < MOE_E; e += 32) m = fmaxf(m, probs[e]);
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
            float s = 0.0f;
            for (int e = lane; e < MOE_E; e += 32) {
                float p = expf(probs[e] - m);
                probs[e] = p;
                s += p;
            }
            s = warp_sum(s);
            for (int e = lane; e < MOE_E; e += 32) probs[e] /= s;
        }
        __syncwarp();
        float picked = 0.0f;
        for (int k = 0; k < MOE_K; k++) {
            float bv = -1.0f;
            int bi = MOE_E;
            for (int e = lane; e < MOE_E; e += 32) {
                if (probs[e] > bv) { bv = probs[e]; bi = e; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, bv, o);
                int oi = __shfl_xor_sync(0xffffffff, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { top_v[k] = bv; top_i[k] = bi; probs[bi] = -2.0f; }
            picked += bv;
            __syncwarp();
        }
        for (int k = lane; k < MOE_K; k += 32) {
            idx[t * MOE_PAIRS + k] = top_i[k];
            coef[t * MOE_PAIRS + k] = MOE_NORM_TOPK ? top_v[k] / picked : top_v[k];
        }
        if (MOE_SHARED && lane == 0) {
            idx[t * MOE_PAIRS + MOE_K] = MOE_E;
            coef[t * MOE_PAIRS + MOE_K] = MOE_SHARED_GATE ? 1.0f / (1.0f + expf(-gate_logit)) : 1.0f;
        }
    }
}

template <bool FP4, int K, int ROWS>
__device__ __forceinline__ float moe_dot(const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane) {
    size_t at = (size_t)expert * ROWS + row;
    float acc = 0.0f;
    if (FP4) {
        const unsigned char* wb = (const unsigned char*)w + at * (K / 2);
        const unsigned char* sb = s + at * (K / 16);
        for (int grp = lane; grp < K / 16; grp += 32) {
            uint2 q = *(const uint2*)(wb + (size_t)grp * 8);
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned byte = ((j < 4 ? q.x : q.y) >> (8 * (j & 3))) & 0xffu;
                part += moe_e2m1(byte & 0xfu) * xs[grp * 16 + 2 * j] + moe_e2m1(byte >> 4) * xs[grp * 16 + 2 * j + 1];
            }
            acc += part * e4m3_to_f32(sb[grp]);
        }
        acc *= g[expert];
    } else {
        const bf16_t* wb = (const bf16_t*)w + at * K;
        for (int k = lane; k < K; k += 32) acc += bf16_to_f32(wb[k]) * xs[k];
    }
    return warp_sum(acc);
}

template <bool FP4>
__device__ __forceinline__ void moe_up_impl(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                            const float* xn, const int* idx, float* act) {
    __shared__ float xs[HIDDEN];
    int pair = blockIdx.x;
    int t = pair / MOE_PAIRS;
    int expert = idx[pair];
    for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[i] = xn[(size_t)t * HIDDEN + i];
    __syncthreads();
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= MOE_INTER) return;
    int stack = MOE_SHARED ? MOE_E + 1 : MOE_E;
    (void)stack;
    float g = moe_dot<FP4, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane);
    float u = moe_dot<FP4, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane);
    if (lane == 0) act[(size_t)pair * MOE_INTER + row] = activation(g) * u;
}

extern "C" __global__ void __launch_bounds__(256) moe_up_bf16(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                                              const float* xn, const int* idx, float* act) {
    moe_up_impl<false>(gw, gs, gg, uw, us, ug, xn, idx, act);
}

extern "C" __global__ void __launch_bounds__(256) moe_up_fp4(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                                             const float* xn, const int* idx, float* act) {
    moe_up_impl<true>(gw, gs, gg, uw, us, ug, xn, idx, act);
}

template <bool FP4>
__device__ __forceinline__ void moe_down_impl(const void* w, const unsigned char* s, const float* g, const float* act, const int* idx, float* out) {
    __shared__ float xs[MOE_INTER];
    int pair = blockIdx.x;
    int expert = idx[pair];
    for (int i = threadIdx.x; i < MOE_INTER; i += 256) xs[i] = act[(size_t)pair * MOE_INTER + i];
    __syncthreads();
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.y * MOE_WARPS + warp;
    if (row >= HIDDEN) return;
    float v = moe_dot<FP4, MOE_INTER, HIDDEN>(w, s, g, expert, row, xs, lane);
    if (lane == 0) out[(size_t)pair * HIDDEN + row] = v;
}

extern "C" __global__ void __launch_bounds__(256) moe_down_bf16(const void* w, const unsigned char* s, const float* g, const float* act, const int* idx, float* out) {
    moe_down_impl<false>(w, s, g, act, idx, out);
}

extern "C" __global__ void __launch_bounds__(256) moe_down_fp4(const void* w, const unsigned char* s, const float* g, const float* act, const int* idx, float* out) {
    moe_down_impl<true>(w, s, g, act, idx, out);
}

extern "C" __global__ void moe_combine(float* x, const float* out, const float* coef) {
    int t = blockIdx.y;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= HIDDEN) return;
    float acc = 0.0f;
    for (int s = 0; s < MOE_PAIRS; s++) acc += coef[t * MOE_PAIRS + s] * out[((size_t)t * MOE_PAIRS + s) * HIDDEN + i];
    x[(size_t)t * HIDDEN + i] += acc;
}

constexpr int MOE_TJ = 2;
constexpr int MOE_SLOTS = MOE_E + (MOE_SHARED ? 1 : 0);

extern "C" __global__ void __launch_bounds__(1024) moe_permute(const int* idx, int* offsets, int* sorted, int pairs) {
    __shared__ int count[MOE_SLOTS + 1];
    __shared__ int cursor[MOE_SLOTS];
    for (int e = threadIdx.x; e <= MOE_SLOTS; e += blockDim.x) count[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) atomicAdd(&count[idx[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int e = 0; e < MOE_SLOTS; e++) {
            int c = count[e];
            offsets[e] = run;
            cursor[e] = run;
            run += c;
        }
        offsets[MOE_SLOTS] = run;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        int at = atomicAdd(&cursor[idx[p]], 1);
        sorted[at] = p;
    }
}

template <bool FP4, int K, int ROWS>
__device__ __forceinline__ void moe_dot2(const void* w, const unsigned char* s, const float* g, int expert, int row, const float* xs, int lane, int live, float* result) {
    size_t at = (size_t)expert * ROWS + row;
#pragma unroll
    for (int j = 0; j < MOE_TJ; j++) result[j] = 0.0f;
    if (FP4) {
        const unsigned char* wb = (const unsigned char*)w + at * (K / 2);
        const unsigned char* sb = s + at * (K / 16);
        for (int grp = lane; grp < K / 16; grp += 32) {
            uint2 q = *(const uint2*)(wb + (size_t)grp * 8);
            float v[16];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned byte = ((j < 4 ? q.x : q.y) >> (8 * (j & 3))) & 0xffu;
                v[2 * j] = moe_e2m1(byte & 0xfu);
                v[2 * j + 1] = moe_e2m1(byte >> 4);
            }
            float sc = e4m3_to_f32(sb[grp]);
#pragma unroll
            for (int j = 0; j < MOE_TJ; j++) {
                if (j < live) {
                    float part = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; i++) part += v[i] * xs[j * K + grp * 16 + i];
                    result[j] += part * sc;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < MOE_TJ; j++) result[j] *= g[expert];
    } else {
        const bf16_t* wb = (const bf16_t*)w + at * K;
        for (int k = lane; k < K; k += 32) {
            float wv = bf16_to_f32(wb[k]);
#pragma unroll
            for (int j = 0; j < MOE_TJ; j++) if (j < live) result[j] += wv * xs[j * K + k];
        }
    }
#pragma unroll
    for (int j = 0; j < MOE_TJ; j++) result[j] = warp_sum(result[j]);
}

template <bool FP4>
__device__ __forceinline__ void moe_group_up_impl(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                                  const float* xn, const int* offsets, const int* sorted, float* act) {
    __shared__ float xs[MOE_TJ * HIDDEN];
    int expert = blockIdx.x;
    int begin = offsets[expert];
    int end = offsets[expert + 1];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.y * MOE_WARPS + warp;
    for (int base = begin; base < end; base += MOE_TJ) {
        int live = min(MOE_TJ, end - base);
        __syncthreads();
        for (int j = 0; j < live; j++) {
            int t = sorted[base + j] / MOE_PAIRS;
            for (int i = threadIdx.x; i < HIDDEN; i += 256) xs[j * HIDDEN + i] = xn[(size_t)t * HIDDEN + i];
        }
        __syncthreads();
        if (row < MOE_INTER) {
            float gv[MOE_TJ], uv[MOE_TJ];
            moe_dot2<FP4, HIDDEN, MOE_INTER>(gw, gs, gg, expert, row, xs, lane, live, gv);
            moe_dot2<FP4, HIDDEN, MOE_INTER>(uw, us, ug, expert, row, xs, lane, live, uv);
            if (lane == 0) {
                for (int j = 0; j < live; j++) act[(size_t)sorted[base + j] * MOE_INTER + row] = activation(gv[j]) * uv[j];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) moe_group_up_bf16(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                                                    const float* xn, const int* offsets, const int* sorted, float* act) {
    moe_group_up_impl<false>(gw, gs, gg, uw, us, ug, xn, offsets, sorted, act);
}

extern "C" __global__ void __launch_bounds__(256) moe_group_up_fp4(const void* gw, const unsigned char* gs, const float* gg, const void* uw, const unsigned char* us, const float* ug,
                                                                   const float* xn, const int* offsets, const int* sorted, float* act) {
    moe_group_up_impl<true>(gw, gs, gg, uw, us, ug, xn, offsets, sorted, act);
}

template <bool FP4>
__device__ __forceinline__ void moe_group_down_impl(const void* w, const unsigned char* s, const float* g, const float* act, const int* offsets, const int* sorted, float* out) {
    __shared__ float xs[MOE_TJ * MOE_INTER];
    int expert = blockIdx.x;
    int begin = offsets[expert];
    int end = offsets[expert + 1];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.y * MOE_WARPS + warp;
    for (int base = begin; base < end; base += MOE_TJ) {
        int live = min(MOE_TJ, end - base);
        __syncthreads();
        for (int j = 0; j < live; j++) {
            size_t p = sorted[base + j];
            for (int i = threadIdx.x; i < MOE_INTER; i += 256) xs[j * MOE_INTER + i] = act[p * MOE_INTER + i];
        }
        __syncthreads();
        if (row < HIDDEN) {
            float v[MOE_TJ];
            moe_dot2<FP4, MOE_INTER, HIDDEN>(w, s, g, expert, row, xs, lane, live, v);
            if (lane == 0) {
                for (int j = 0; j < live; j++) out[(size_t)sorted[base + j] * HIDDEN + row] = v[j];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) moe_group_down_bf16(const void* w, const unsigned char* s, const float* g, const float* act, const int* offsets, const int* sorted, float* out) {
    moe_group_down_impl<false>(w, s, g, act, offsets, sorted, out);
}

extern "C" __global__ void __launch_bounds__(256) moe_group_down_fp4(const void* w, const unsigned char* s, const float* g, const float* act, const int* offsets, const int* sorted, float* out) {
    moe_group_down_impl<true>(w, s, g, act, offsets, sorted, out);
}
