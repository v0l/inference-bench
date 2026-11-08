constexpr int FP4_GROUP = 16;
constexpr int FP4_HEADER = 16;

__device__ __forceinline__ float fp4_inv_global(const unsigned char* scales) {
    return __ldg((const float*)scales);
}

__device__ __forceinline__ float e2m1_to_f32(unsigned n) {
    return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22)) * __uint_as_float(0x7e800000u);
}

__device__ __forceinline__ uint2 load_stream8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

__device__ __forceinline__ int fp4_t(int k, int kg) {
    return (k & (FP4_GROUP - 1)) * kg + (k >> 4);
}

template <int K>
__device__ __forceinline__ float fp4_row_dot(const void* w, const unsigned char* scales, int row, const float* xt, int lane) {
    constexpr int KG = K / FP4_GROUP;
    const uint2* wr = (const uint2*)((const unsigned char*)w + (size_t)row * (K / 2));
    const unsigned char* sr = scales + FP4_HEADER + (size_t)row * KG;
    float acc = 0.0f;
#pragma unroll 4
    for (int g = lane; g < KG; g += 32) {
        uint2 q = load_stream8(wr + g);
        float s = e4m3_to_f32(__ldg(sr + g));
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned byte = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xffu;
            part += e2m1_to_f32(byte & 15u) * xt[(2 * i) * KG + g];
            part += e2m1_to_f32(byte >> 4) * xt[(2 * i + 1) * KG + g];
        }
        acc += part * s;
    }
    return warp_sum(acc) * fp4_inv_global(scales);
}

__device__ __forceinline__ void load_normed_t(const float* x, const float* nw, float* xt) {
    __shared__ float scratch[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xt[fp4_t(i, HIDDEN / FP4_GROUP)] = nw[i] * (x[i] * inv);
    __syncthreads();
}

extern "C" __global__ void gate_up_act_f4(const void* w, const unsigned char* s, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xt[HIDDEN];
    load_normed_t(x, nw, xt);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= INTER) return;
    float g = fp4_row_dot<HIDDEN>(w, s, 2 * row, xt, lane);
    float u = fp4_row_dot<HIDDEN>(w, s, 2 * row + 1, xt, lane);
    if (lane == 0) out[fp4_t(row, INTER / FP4_GROUP)] = activation(g) * u;
}

extern "C" __global__ void down_proj_f4(const void* w, const unsigned char* s, const float* b, const float* xt, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= HIDDEN) return;
    float acc = fp4_row_dot<INTER>(w, s, row, xt, lane);
    if (lane == 0) out[row] += acc;
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

template <int N, int K, int BM, int BN, int WM, int WN, int STAGES, int EPI>
__device__ __forceinline__ void gemm_f4w(const void* w, const unsigned char* scales, const bf16_t* x_hi,
                                         const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int BK = F4W_BK;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 16;
    constexpr int CPR = BK / 8;
    constexpr int A_TILE = BM * BK;
    constexpr int A_BYTES = A_TILE * 2 * 2;
    constexpr int STAGE_BYTES = A_BYTES + BN * F4W_ROW;
    constexpr int KT = K / BK;
    constexpr int KG = K / FP4_GROUP;
    constexpr int A_TOTAL = BM * CPR * 2;
    constexpr int A_ITERS = (A_TOTAL + THREADS - 1) / THREADS;
    constexpr int W_TOTAL = BN * 3;
    constexpr int W_ITERS = (W_TOTAL + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of 64");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    int last_row = max(tokens - 1, 0);
    const unsigned char* sbase = scales + FP4_HEADER;
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != A_TOTAL && idx >= A_TOTAL) break;
            int p = idx / (BM * CPR);
            int rem = idx % (BM * CPR);
            int r = rem / CPR;
            int c = rem % CPR;
            int m = m0 + r;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last_row) * K + kt * BK + c * 8;
            cp_async16_zfill(base + p * A_TILE * 2 + swz<CPR>(r, c) * 16, src, m < tokens);
        }
        unsigned char* ws = base + A_BYTES;
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != W_TOTAL && idx >= W_TOTAL) break;
            int r = idx / 3;
            int c = idx % 3;
            int n = min(n0 + r, N - 1);
            if (c < 2) {
                cp_async16(ws + r * F4W_ROW + c * 16, (const unsigned char*)w + (size_t)n * (K / 2) + kt * (BK / 2) + c * 16);
            } else {
                cp_async4(ws + r * F4W_ROW + 32, sbase + (size_t)n * KG + kt * (BK / FP4_GROUP));
            }
        }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < KT) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* base = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const bf16_t* a_hi = (const bf16_t*)base;
        const bf16_t* a_lo = a_hi + A_TILE;
        const unsigned char* ws = base + A_BYTES;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned bw[NJ][2];
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                const unsigned char* p = ws + (wn * TN + j * 8 + g) * F4W_ROW;
                float s = e4m3_to_f32(p[32 + ks]);
                bw[j][0] = fp4_pair_bf16(p[ks * 8 + t], s);
                bw[j][1] = fp4_pair_bf16(p[ks * 8 + 4 + t], s);
            }
            unsigned ah[MI][4], al[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int row = wm * TM + i * 16 + (lane & 15);
                int at = swz<CPR>(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah[i], &a_hi[at]);
                ldsm_x4(al[i], &a_lo[at]);
            }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    mma_bf16(acc[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                    mma_bf16(acc[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                }
        }
    }
    cp_async_wait<0>();
    float inv = fp4_inv_global(scales);
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] *= inv;
    gemm_epilogue<N, BM, BN, WM, WN, EPI, false>(acc, nullptr, out, out_hi, out_lo, tokens);
}

{{F4W_KERNELS}}
