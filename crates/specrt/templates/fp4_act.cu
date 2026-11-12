__device__ __forceinline__ void mma_fp4(float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa, unsigned sfb) {
    unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

__device__ __forceinline__ unsigned f32x2_to_e2m1x2(float lo, float hi) {
    unsigned short r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }" : "=h"(r) : "f"(hi), "f"(lo));
    return r & 0xffu;
}

__device__ __forceinline__ unsigned f32_to_e4m3_bits(float v) {
    return f32x2_to_e4m3x2(v, 0.0f) & 0xffu;
}

__device__ __forceinline__ void cp_async8_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

extern "C" __global__ void quant_rows_fp4(const bf16_t* hi, const bf16_t* lo, const unsigned char* wscales, unsigned char* q,
                                          unsigned char* qs, int width) {
    int groups = width / FP4_GROUP;
    int grp = blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= groups) return;
    size_t row = blockIdx.y;
    size_t at = row * width + (size_t)grp * FP4_GROUP;
    float igs = __ldg((const float*)wscales + 1);
    float v[FP4_GROUP];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i++) {
        v[i] = bf16_to_f32(hi[at + i]) + bf16_to_f32(lo[at + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    unsigned sbits = f32_to_e4m3_bits(fminf(igs * (amax * (1.0f / 6.0f)), 448.0f));
    float sx = e4m3_to_f32(sbits);
    float mul = sx > 0.0f ? igs / sx : 0.0f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i += 2) {
        unsigned b = f32x2_to_e2m1x2(v[i] * mul, v[i + 1] * mul);
        if (i < 8) w0 |= b << (4 * i); else w1 |= b << (4 * (i - 8));
    }
    *(uint2*)(q + at / 2) = make_uint2(w0, w1);
    qs[row * groups + grp] = (unsigned char)sbits;
}

constexpr int F4A_BK = 128;
constexpr int F4A_DATA = F4A_BK / 2;
constexpr int F4A_SCALES = F4A_BK / FP4_GROUP;
constexpr int F4A_ROW = F4A_DATA + F4A_SCALES;

__device__ __forceinline__ int f4a_chunk(int r, int c) {
    return r * F4A_DATA + ((c ^ (r >> 1)) & 3) * 16;
}

template <int N, int K, int BM, int BN, int WM, int WN, int STAGES, int EPI>
__device__ __forceinline__ void gemm_f4a(const void* w, const unsigned char* wscales, const unsigned char* xq,
                                         const unsigned char* xs, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int BK = F4A_BK;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 64;
    constexpr int STAGE_BYTES = (BM + BN) * F4A_ROW;
    constexpr int SCALE_OFF = (BM + BN) * F4A_DATA;
    constexpr int KT = K / BK;
    constexpr int KG = K / FP4_GROUP;
    constexpr int CHUNKS = 5;
    constexpr int A_ITERS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of 128");
    static_assert(KG % 8 == 0, "scale rows are copied 8 bytes at a time");
    static_assert(NJ % 2 == 0, "B fragments come two n tiles per ldmatrix");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int wm = warp / WN;
    int wn = warp % WN;
    int2 tile = grouped_tile<N, BN>();
    int m0 = tile.x * BM;
    int n0 = tile.y * BN;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    int last_row = max(tokens - 1, 0);
    const unsigned char* wsc = wscales + FP4_HEADER;
    auto load_stage = [&](int buf, int kt) {
        unsigned char* as = gemm_smem + buf * STAGE_BYTES;
        unsigned char* ws = as + BM * F4A_DATA;
        unsigned char* ass = as + SCALE_OFF;
        unsigned char* wss = ass + BM * F4A_SCALES;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != BM * CHUNKS && idx >= BM * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int m = m0 + r;
            size_t row = (size_t)min(m, last_row);
            if (c < 4) {
                cp_async16_zfill(as + f4a_chunk(r, c), xq + row * (K / 2) + kt * (BK / 2) + c * 16, m < tokens);
            } else {
                cp_async8_zfill(ass + r * F4A_SCALES, xs + row * KG + kt * (BK / FP4_GROUP), m < tokens);
            }
        }
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * CHUNKS && idx >= BN * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            size_t n = (size_t)min(n0 + r, N - 1);
            if (c < 4) {
                cp_async16(ws + f4a_chunk(r, c), (const unsigned char*)w + n * (K / 2) + kt * (BK / 2) + c * 16);
            } else {
                cp_async8_zfill(wss + r * F4A_SCALES, wsc + n * KG + kt * (BK / FP4_GROUP), true);
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
        const unsigned char* as = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const unsigned char* ws = as + BM * F4A_DATA;
        const unsigned char* ass = as + SCALE_OFF;
        const unsigned char* wss = ass + BM * F4A_SCALES;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned a[MI][4], sfa[MI];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 15);
                ldsm_x4(a[i], as + f4a_chunk(r, ks * 2 + (lane >> 4)));
                sfa[i] = *(const unsigned*)(ass + (wm * TM + i * 16 + g + 8 * (lane & 1)) * F4A_SCALES + ks * 4);
            }
#pragma unroll
            for (int j = 0; j < NJ; j += 2) {
                unsigned b[4];
                int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, ws + f4a_chunk(r, ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
                for (int jj = 0; jj < 2; jj++) {
                    unsigned sfb = *(const unsigned*)(wss + (wn * TN + (j + jj) * 8 + g) * F4A_SCALES + ks * 4);
#pragma unroll
                    for (int i = 0; i < MI; i++) mma_fp4(acc[i][j + jj], a[i], b[2 * jj], b[2 * jj + 1], sfa[i], sfb);
                }
            }
        }
    }
    cp_async_wait<0>();
    float alpha = fp4_inv_global(wscales) / __ldg((const float*)wscales + 1);
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] *= alpha;
    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, false>(acc, nullptr, out, out_hi, out_lo, tokens, m0, n0);
}

