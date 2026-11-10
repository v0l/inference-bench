constexpr int FP4_GROUP = 16;
constexpr int FP4_HEADER = 16;
constexpr int FP4_GEMV_THREADS = 512;

__device__ __forceinline__ float fp4_inv_global(const unsigned char* scales) {
    return __ldg((const float*)scales);
}

__device__ __forceinline__ float e2m1_to_f32(unsigned n) {
    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);
}

__device__ __forceinline__ int fp4_t(int k, int kg) {
    return (k & (FP4_GROUP - 1)) * kg + (k >> 4);
}

template <int K, int ROWS, int U>
__device__ __forceinline__ void fp4_rows_dot(const void* w, const unsigned char* scales, int row0, int stride, const float* xt, int lane, float* result) {
    constexpr int KG = K / FP4_GROUP;
    const unsigned char* wb = (const unsigned char*)w;
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
                    size_t row = (size_t)(row0 + r * stride);
                    q[r][u] = load_stream8(wb + row * (K / 2) + (size_t)g * 8);
                    sc[r][u] = __ldg(sb + row * KG + g);
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
    float inv = fp4_inv_global(scales);
#pragma unroll
    for (int r = 0; r < ROWS; r++) result[r] = warp_sum(acc[r]) * inv;
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ unsigned fp4_pair_bf16(unsigned byte, float scale) {
    float lo = e2m1_to_f32(byte & 15u) * scale;
    float hi = e2m1_to_f32(byte >> 4) * scale;
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}

constexpr int F4W_BK = 64;
constexpr int F4W_ROW = 48;

struct DecFp4 : DecBase<DecFp4> {
    static constexpr int ACC = ACC_DIRECT;
    static constexpr bool LDSM = false;
    static constexpr int WARPS = FP4_GEMV_THREADS / 32;
    template <int BK> static constexpr int wrow = F4W_ROW;
    template <int BK> static constexpr int w_chunks = 3;
    template <int W> static __device__ __forceinline__ int x_at(int i) { return fp4_t(i, W / FP4_GROUP); }
    template <int K, int ROWS>
    static __device__ __forceinline__ void rows(const void* w, const void* s, int row0, int stride, const float* x, int lane, float* out) {
        fp4_rows_dot<K, ROWS, 4 / ROWS>(w, (const unsigned char*)s, row0, stride, x, lane, out);
    }
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void* s, int r, int n, int kt, int ck) {
        static_assert(BK == F4W_BK, "nvfp4 tiles are 64 wide");
        if (ck < 2) {
            cp_async16(ws + r * F4W_ROW + ck * 16, (const unsigned char*)w + (size_t)n * (K / 2) + kt * (BK / 2) + ck * 16);
        } else {
            cp_async4(ws + r * F4W_ROW + 32, (const unsigned char*)s + FP4_HEADER + (size_t)n * (K / FP4_GROUP) + kt * (BK / FP4_GROUP));
        }
    }
    static __device__ __forceinline__ void frag(unsigned* bw, const unsigned char* p, int ks, int t) {
        float sc = e4m3_to_f32(p[32 + ks]);
        bw[0] = fp4_pair_bf16(p[ks * 8 + t], sc);
        bw[1] = fp4_pair_bf16(p[ks * 8 + 4 + t], sc);
    }
    template <int MI, int NJ> static __device__ __forceinline__ void finish(float (&acc)[MI][NJ][4], const void* s) {
        float inv = fp4_inv_global((const unsigned char*)s);
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] *= inv;
    }
};

__device__ __forceinline__ void load_normed_t(const float* x, const float* nw, float* xt) {
    load_normed<DecFp4>(x, nw, xt);
}

extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) gate_up_act_f4(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    gate_up_act_norm<DecFp4>(w, s, x, nw, out);
}

extern "C" __global__ void __launch_bounds__(FP4_GEMV_THREADS) down_proj_f4(const void* w, const unsigned char* s, const float* b, const float* xt, float* out) {
    linear_w<DecFp4, HIDDEN, INTER, false, true>(w, s, b, xt, out);
}

{{F4W_KERNELS}}
{{F4A_SOURCE}}
