// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// vision head_dim=128 hdp=128 precise=false gemm_smem=73728 flash_smem=69632
typedef unsigned short bf16_t;

__device__ __forceinline__ float bf(bf16_t v) { return __uint_as_float(((unsigned)v) << 16); }

__device__ __forceinline__ float activate(float x, int kind) {
    if (kind == 1) {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + tanhf(u));
    }
    if (kind == 2) {
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
    return x;
}

__device__ __forceinline__ unsigned bf16_bits_rn(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ void split_bf16(float v, unsigned& hi, unsigned& lo) {
    hi = bf16_bits_rn(v);
    lo = bf16_bits_rn(v - __uint_as_float(hi << 16));
}

__device__ __forceinline__ void split_store2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha, la, hb, lb;
    split_bf16(a, ha, la);
    split_bf16(b, hb, lb);
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

__device__ __forceinline__ unsigned f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float f16_float(unsigned bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ unsigned pack_f16_hi(float a, float b) { return f16_bits(a) | (f16_bits(b) << 16); }

__device__ __forceinline__ unsigned pack_f16_lo(float a, float b) {
    return f16_bits(a - f16_float(f16_bits(a))) | (f16_bits(b - f16_float(f16_bits(b))) << 16);
}

constexpr bool GEMM_SPLIT = false;
constexpr bool ATT_SPLIT = false;

__device__ __forceinline__ void store_act2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    if (GEMM_SPLIT) {
        split_store2(hi, lo, at, a, b);
    } else {
        *(unsigned*)(hi + at) = pack_f16_hi(a, b);
    }
}

extern "C" __global__ void vis_to_f16(bf16_t* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = (bf16_t)f16_bits(bf(w[i]));
}

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(PENDING));
}

__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16_hi_lo(float* c, const unsigned* ah, const unsigned* al, unsigned bh0, unsigned bh1,
                                              unsigned bl0, unsigned bl1) {
    mma_f16(c, al, bh0, bh1);
    mma_f16(c, ah, bl0, bl1);
    mma_f16(c, ah, bh0, bh1);
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ int swz4(int r, int c) { return r * 4 + (c ^ ((r >> 1) & 3)); }

constexpr int GM_BM = 128;
constexpr int GM_BN = 128;
constexpr int GM_BK = 32;
constexpr int GM_CPR = GM_BK / 8;
constexpr int GM_PLANES = GEMM_SPLIT ? 2 : 1;
constexpr int GM_WM = 2;
constexpr int GM_WN = 4;
constexpr int GM_STAGES = 3;
constexpr int GM_THREADS = GM_WM * GM_WN * 32;
constexpr int GM_TM = GM_BM / GM_WM;
constexpr int GM_TN = GM_BN / GM_WN;
constexpr int GM_MI = GM_TM / 16;
constexpr int GM_NJ = GM_TN / 8;
constexpr int GM_A_TILE = GM_BM * GM_BK;
constexpr int GM_A_BYTES = GM_A_TILE * 2 * GM_PLANES;
constexpr int GM_STAGE_BYTES = GM_A_BYTES + GM_BN * GM_BK * 2;

__device__ __forceinline__ int gswz(int r, int c) {
    if (GM_CPR == 8) return r * 8 + (c ^ (r & 7));
    return r * 4 + (c ^ ((r >> 1) & 3));
}

extern "C" __global__ void __launch_bounds__(GM_THREADS) vis_mma(const bf16_t* a_hi, const bf16_t* a_lo, const bf16_t* w,
                                                                const bf16_t* bias, const float* resid, float* out,
                                                                bf16_t* out_hi, bf16_t* out_lo, int M, int N, int K, int act) {
    extern __shared__ __align__(16) unsigned char gm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / GM_WN;
    int wn = warp % GM_WN;
    int m0 = blockIdx.y * GM_BM;
    int n0 = blockIdx.x * GM_BN;
    int kt_count = (K + GM_BK - 1) / GM_BK;
    float acc[GM_MI][GM_NJ][4];
#pragma unroll
    for (int i = 0; i < GM_MI; i++)
#pragma unroll
        for (int j = 0; j < GM_NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gm_smem + buf * GM_STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < (GM_PLANES * GM_BM * GM_CPR) / GM_THREADS; i++) {
            int idx = tid + i * GM_THREADS;
            int p = idx / (GM_BM * GM_CPR);
            int rem = idx % (GM_BM * GM_CPR);
            int r = rem / GM_CPR;
            int c = rem % GM_CPR;
            int m = m0 + r;
            int k = kt * GM_BK + c * 8;
            const bf16_t* src = (p ? a_lo : a_hi) + (size_t)min(m, M - 1) * K + min(k, K - 8);
            cp_async16_zfill(base + p * GM_A_TILE * 2 + gswz(r, c) * 16, src, m < M && k < K);
        }
        unsigned char* ws = base + GM_A_BYTES;
#pragma unroll
        for (int i = 0; i < (GM_BN * GM_CPR) / GM_THREADS; i++) {
            int idx = tid + i * GM_THREADS;
            int r = idx / GM_CPR;
            int c = idx % GM_CPR;
            int n = n0 + r;
            int k = kt * GM_BK + c * 8;
            const bf16_t* src = w + (size_t)min(n, N - 1) * K + min(k, K - 8);
            cp_async16_zfill(ws + gswz(r, c) * 16, src, n < N && k < K);
        }
    };
#pragma unroll
    for (int st = 0; st < GM_STAGES - 1; st++) {
        if (st < kt_count) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < kt_count; kt++) {
        cp_async_wait<GM_STAGES - 2>();
        __syncthreads();
        int next = kt + GM_STAGES - 1;
        if (next < kt_count) load_stage(next % GM_STAGES, next);
        cp_async_commit();
        const unsigned char* base = gm_smem + (kt % GM_STAGES) * GM_STAGE_BYTES;
        const bf16_t* sa_hi = (const bf16_t*)base;
        const bf16_t* sa_lo = sa_hi + GM_A_TILE;
        const unsigned char* ws = base + GM_A_BYTES;
#pragma unroll
        for (int ks = 0; ks < GM_BK / 16; ks++) {
            unsigned bw[GM_NJ][2];
#pragma unroll
            for (int jp = 0; jp < GM_NJ / 2; jp++) {
                unsigned r[4];
                int row = wn * GM_TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                ldsm_x4(r, ws + gswz(row, ks * 2 + ((lane >> 3) & 1)) * 16);
                bw[2 * jp][0] = r[0];
                bw[2 * jp][1] = r[1];
                bw[2 * jp + 1][0] = r[2];
                bw[2 * jp + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < GM_MI; i++) {
                unsigned ah[4], al[4];
                int row = wm * GM_TM + i * 16 + (lane & 15);
                int at = gswz(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah, &sa_hi[at]);
                if (GEMM_SPLIT) ldsm_x4(al, &sa_lo[at]);
#pragma unroll
                for (int j = 0; j < GM_NJ; j++) {
                    if (GEMM_SPLIT) {
                        mma_bf16(acc[i][j], al, bw[j][0], bw[j][1]);
                        mma_bf16(acc[i][j], ah, bw[j][0], bw[j][1]);
                    } else {
                        mma_f16(acc[i][j], ah, bw[j][0], bw[j][1]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int i = 0; i < GM_MI; i++)
#pragma unroll
        for (int j = 0; j < GM_NJ; j++) {
            int col = n0 + wn * GM_TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = bias ? bf(bias[col]) : 0.0f;
            float b1 = bias ? bf(bias[col + 1]) : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * GM_TM + i * 16 + g + h * 8;
                if (m >= M) continue;
                float v0 = activate(acc[i][j][h * 2] + b0, act);
                float v1 = activate(acc[i][j][h * 2 + 1] + b1, act);
                size_t at = (size_t)m * N + col;
                if (out_hi) {
                    store_act2(out_hi, out_lo, at, v0, v1);
                } else {
                    if (resid) {
                        v0 += resid[at];
                        v1 += resid[at + 1];
                    }
                    *(float2*)(out + at) = make_float2(v0, v1);
                }
            }
        }
}

extern "C" __global__ void vis_split(const float* x, bf16_t* hi, bf16_t* lo, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    float2 v = *(const float2*)(x + i);
    store_act2(hi, lo, i, v.x, v.y);
}

extern "C" __global__ void vis_layernorm(const float* x, const bf16_t* w, const bf16_t* b, bf16_t* hi, bf16_t* lo, int width, float eps) {
    __shared__ float red[2][32];
    const float* row = x + (size_t)blockIdx.x * width;
    float s = 0.0f, s2 = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = row[i];
        s += v;
        s2 += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        red[0][warp] = s;
        red[1][warp] = s2;
    }
    __syncthreads();
    s = 0.0f;
    s2 = 0.0f;
    for (int i = 0; i < blockDim.x / 32; i++) {
        s += red[0][i];
        s2 += red[1][i];
    }
    float mean = s / width;
    float var = s2 / width - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    size_t base = (size_t)blockIdx.x * width;
    for (int i = threadIdx.x * 2; i < width; i += blockDim.x * 2) {
        float a = (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
        float c = (row[i + 1] - mean) * inv * bf(w[i + 1]) + bf(b[i + 1]);
        store_act2(hi, lo, base + i, a, c);
    }
}

extern "C" __global__ void vis_pos(const float* table, const int* taps, const float* weights, float* out, int hidden) {
    int tok = blockIdx.x;
    const int* ti = taps + tok * 4;
    const float* tw = weights + tok * 4;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < 4; k++) v += tw[k] * table[(size_t)ti[k] * hidden + d];
        out[(size_t)tok * hidden + d] = v;
    }
}

constexpr int AT_HDP = 128;
constexpr int AT_STRIDE = AT_HDP + 8;
constexpr int AT_WARPS = 8;
constexpr int AT_BQ = 16 * AT_WARPS;
constexpr int AT_BK = ATT_SPLIT ? 32 : 64;
constexpr int AT_KB = AT_HDP / 16;
constexpr int AT_NB = AT_HDP / 8;
constexpr int AT_CHUNKS = AT_HDP / 8;
constexpr int AT_TILE = AT_BK * AT_STRIDE;
constexpr int AT_STAGE = (ATT_SPLIT ? 4 : 2) * AT_TILE;

extern "C" __global__ void vis_rope_pack(const float* qkv, const int* hw, bf16_t* planes, int tokens, int heads, int head_dim, float theta) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int half = head_dim / 2;
    int quarter = head_dim / 4;
    size_t width = (size_t)3 * heads * head_dim;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    size_t dst = ((size_t)h * tokens + tok) * AT_HDP;
    int ph = hw[2 * tok], pw = hw[2 * tok + 1];
    for (int part = 0; part < 3; part++) {
        const float* src = qkv + tok * width + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        bf16_t* hi = planes + (2 * part) * plane + dst;
        bf16_t* lo = planes + (2 * part + 1) * plane + dst;
        for (int d = threadIdx.x; d < AT_HDP; d += blockDim.x) {
            float v = 0.0f;
            if (d < head_dim) {
                v = src[d];
                if (part < 2 && theta > 0.0f) {
                    int i = d < half ? d : d - half;
                    int fi = i < quarter ? i : i - quarter;
                    float inv = exp2f(-(2.0f * fi / (float)half) * log2f(theta));
                    float angle = (float)(i < quarter ? ph : pw) * inv;
                    float s, c;
                    sincosf(angle, &s, &c);
                    float other = d < half ? src[d + half] : src[d - half];
                    v = d < half ? v * c - other * s : v * c + other * s;
                }
            }
            unsigned b = f16_bits(v);
            hi[d] = (bf16_t)b;
            if (ATT_SPLIT) lo[d] = (bf16_t)f16_bits(v - f16_float(b));
        }
    }
}

extern "C" __global__ void __launch_bounds__(32 * AT_WARPS) vis_flash(const bf16_t* planes, const int* seg_lo, const int* seg_hi,
                                                                     bf16_t* out_hi, bf16_t* out_lo, int tokens, int heads,
                                                                     int head_dim, float scale_log2) {
    extern __shared__ __align__(16) unsigned char at_smem[];
    bf16_t* sm = (bf16_t*)at_smem;
    int head = blockIdx.y;
    int q0 = blockIdx.x * AT_BQ;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    const bf16_t* head_base = planes + (size_t)head * tokens * AT_HDP;
    for (int i = tid; i < 2 * AT_BQ * AT_CHUNKS; i += 32 * AT_WARPS) {
        int p = i / (AT_BQ * AT_CHUNKS);
        int rem = i % (AT_BQ * AT_CHUNKS);
        int r = rem / AT_CHUNKS;
        int c = rem % AT_CHUNKS;
        int q = q0 + r;
        const bf16_t* src = head_base + p * plane + (size_t)min(q, tokens - 1) * AT_HDP + c * 8;
        cp_async16_zfill(sm + (p * AT_BQ + r) * AT_STRIDE + c * 8, src, q < tokens);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned qh[AT_KB][4], ql[AT_KB][4];
#pragma unroll
    for (int kb = 0; kb < AT_KB; kb++) {
        int row = warp * 16 + (lane & 15);
        int col = kb * 16 + (lane >> 4) * 8;
        ldsm_x4(qh[kb], sm + row * AT_STRIDE + col);
        if (ATT_SPLIT) ldsm_x4(ql[kb], sm + (AT_BQ + row) * AT_STRIDE + col);
    }
    __syncthreads();
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int lo0 = row0 < tokens ? seg_lo[row0] : 0, hi0 = row0 < tokens ? seg_hi[row0] : 0;
    int lo1 = row1 < tokens ? seg_lo[row1] : 0, hi1 = row1 < tokens ? seg_hi[row1] : 0;
    int kstart = seg_lo[q0];
    int kend = seg_hi[min(q0 + AT_BQ, tokens) - 1];
    int tiles = (kend - kstart + AT_BK - 1) / AT_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = sm + buf * AT_STAGE;
        constexpr int KV_LOADS = ATT_SPLIT ? 4 : 2;
        for (int i = tid; i < KV_LOADS * AT_BK * AT_CHUNKS; i += 32 * AT_WARPS) {
            int slot = i / (AT_BK * AT_CHUNKS);
            int p = ATT_SPLIT ? slot : slot * 2;
            int rem = i % (AT_BK * AT_CHUNKS);
            int r = rem / AT_CHUNKS;
            int c = rem % AT_CHUNKS;
            int key = k0 + r;
            const bf16_t* src = head_base + (size_t)(2 + p) * plane + (size_t)min(key, tokens - 1) * AT_HDP + c * 8;
            cp_async16_zfill(st + slot * AT_TILE + r * AT_STRIDE + c * 8, src, key < kend);
        }
    };
    float o[AT_NB][4];
#pragma unroll
    for (int i = 0; i < AT_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    const float NEG = -3.0e38f;
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    if (tiles > 0) load_kv(0, kstart);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kstart + kt * AT_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) & 1, k0 + AT_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* kh = sm + (kt & 1) * AT_STAGE;
        const bf16_t* kl = kh + AT_TILE;
        const bf16_t* vh = ATT_SPLIT ? kl + AT_TILE : kh + AT_TILE;
        const bf16_t* vl = vh + AT_TILE;
        constexpr int SB = AT_BK / 8;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < AT_KB; kb++) {
                unsigned bh[4], bl[4];
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = kr * AT_STRIDE + kb * 16 + ((lane >> 3) & 1) * 8;
                ldsm_x4(bh, kh + at);
                if (ATT_SPLIT) {
                    ldsm_x4(bl, kl + at);
                    mma_f16_hi_lo(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                    mma_f16_hi_lo(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
                } else {
                    mma_f16(s[2 * np], qh[kb], bh[0], bh[1]);
                    mma_f16(s[2 * np + 1], qh[kb], bh[2], bh[3]);
                }
            }
        float mx0 = NEG, mx1 = NEG;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                bool ok = e < 2 ? (key >= lo0 && key < hi0) : (key >= lo1 && key < hi1);
                float v = ok ? s[nb][e] * scale_log2 : NEG;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG ? 0.0f : fast_exp2(s[nb][e] - nm);
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < AT_NB; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_f16_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_f16_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_f16_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_f16_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            pl[0] = pack_f16_lo(s[2 * kb][0], s[2 * kb][1]);
            pl[1] = pack_f16_lo(s[2 * kb][2], s[2 * kb][3]);
            pl[2] = pack_f16_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            pl[3] = pack_f16_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
#pragma unroll
            for (int np = 0; np < AT_NB / 2; np++) {
                unsigned bh[4], bl[4];
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = kr * AT_STRIDE + np * 16 + (lane >> 4) * 8;
                ldsm_x4_t(bh, vh + at);
                if (ATT_SPLIT) {
                    ldsm_x4_t(bl, vl + at);
                    mma_f16_hi_lo(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                    mma_f16_hi_lo(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
                } else {
                    mma_f16(o[2 * np], ph, bh[0], bh[1]);
                    mma_f16(o[2 * np + 1], ph, bh[2], bh[3]);
                }
            }
        }
        __syncthreads();
    }
    float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    float i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    size_t width = (size_t)heads * head_dim;
#pragma unroll
    for (int nb = 0; nb < AT_NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (d >= head_dim) continue;
        size_t col = (size_t)head * head_dim + d;
        if (row0 < tokens) store_act2(out_hi, out_lo, row0 * width + col, o[nb][0] * i0, o[nb][1] * i0);
        if (row1 < tokens) store_act2(out_hi, out_lo, row1 * width + col, o[nb][2] * i1, o[nb][3] * i1);
    }
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < blockDim.x / 32; i++) t += red[i];
    return t;
}

extern "C" __global__ void vis_pool(const float* x, const bf16_t* w, const bf16_t* b, const float* soft, bf16_t* hi, bf16_t* lo,
                                    int hidden, int side, int kernel, int out_side, float eps, float soft_eps) {
    extern __shared__ float acc[];
    __shared__ float red[32];
    int o = blockIdx.x;
    int per_tile = out_side * out_side;
    int base = (o / per_tile) * side * side;
    int local = o % per_tile;
    int oy = local / out_side, ox = local % out_side;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();
    for (int dy = 0; dy < kernel; dy++) {
        for (int dx = 0; dx < kernel; dx++) {
            const float* row = x + (size_t)(base + (oy * kernel + dy) * side + ox * kernel + dx) * hidden;
            float s = 0.0f;
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) s += row[i];
            float mean = block_sum(s, red) / hidden;
            float s2 = 0.0f;
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
                float d = row[i] - mean;
                s2 += d * d;
            }
            float inv = rsqrtf(block_sum(s2, red) / hidden + eps);
            for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] += (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
        }
    }
    float scale = 1.0f / (kernel * kernel);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        acc[i] *= scale;
        ss += acc[i] * acc[i];
    }
    float rinv = rsqrtf(block_sum(ss, red) / hidden + soft_eps);
    __syncthreads();
    size_t at = (size_t)o * hidden;
    for (int i = threadIdx.x * 2; i < hidden; i += blockDim.x * 2) store_act2(hi, lo, at + i, acc[i] * rinv * soft[i], acc[i + 1] * rinv * soft[i + 1]);
}
