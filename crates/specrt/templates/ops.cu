typedef unsigned short bf16_t;
typedef signed char i8_t;

constexpr int HIDDEN = {{HIDDEN}};
constexpr int INTER = {{INTER}};
constexpr int VOCAB = {{VOCAB}};
constexpr int HEADS = {{HEADS}};
constexpr int KV_HEADS = {{KV_HEADS}};
constexpr int HEAD_DIM = {{HEAD_DIM}};
constexpr int GROUP = {{GROUP}};
constexpr int Q_DIM = HEADS * HEAD_DIM;
constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
constexpr int QKV_ROW = {{QKV_ROW}};
constexpr int GATE_OFF = Q_DIM + 2 * KV_DIM;
constexpr bool OUT_GATE = {{OUT_GATE}};
constexpr int ROT = {{ROT}};
constexpr int ROT_HALF = ROT / 2;
constexpr bool QK_NORM = {{QK_NORM}};
constexpr float EPS = {{EPS}};
constexpr float THETA = {{THETA}};
constexpr int ROPE_KIND = {{ROPE_KIND}};
constexpr float ROPE_FACTOR = {{ROPE_FACTOR}};
constexpr float ROPE_LOW_FREQ = {{ROPE_LOW_FREQ}};
constexpr float ROPE_HIGH_FREQ = {{ROPE_HIGH_FREQ}};
constexpr float ROPE_ORIG_POS = {{ROPE_ORIG_POS}};
constexpr int ACT_KIND = {{ACT_KIND}};
constexpr int MAX_CTX = {{MAX_CTX}};
constexpr int PAGE = {{PAGE}};
constexpr int PAGES = (MAX_CTX + PAGE - 1) / PAGE;
constexpr int Q8_GROUP = {{Q8_GROUP}};
constexpr int WF_BF16 = 0;
constexpr int WF_Q8 = 1;
constexpr int WF_FP8 = 2;
constexpr int FP8_BLOCK = {{FP8_BLOCK}};
constexpr int LINEAR_WARPS = 8;
constexpr int ATTN_THREADS = 512;
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float bf16_to_f32(bf16_t v) {
    return __uint_as_float(((unsigned)v) << 16);
}

__device__ __forceinline__ float e4m3_to_f32(unsigned b) {
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * __uint_as_float(0x7b800000u);
}

__device__ __forceinline__ unsigned e4m3_pair_bf16(unsigned two_bytes) {
    return (__float_as_uint(e4m3_to_f32(two_bytes & 0xffu)) >> 16) |
           (__float_as_uint(e4m3_to_f32((two_bytes >> 8) & 0xffu)) & 0xffff0000u);
}

__device__ __forceinline__ float e4m3_dot4(unsigned word, float4 x) {
    return e4m3_to_f32(word & 0xffu) * x.x + e4m3_to_f32((word >> 8) & 0xffu) * x.y +
           e4m3_to_f32((word >> 16) & 0xffu) * x.z + e4m3_to_f32(word >> 24) * x.w;
}

__device__ __forceinline__ float out_gate(const float* qkv, size_t row, int at) {
    if (!OUT_GATE) return 1.0f;
    return 1.0f / (1.0f + expf(-qkv[row * QKV_ROW + GATE_OFF + at]));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

template <bool IS_MAX>
__device__ float block_reduce(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = IS_MAX ? NEG_INF : 0.0f;
    for (int i = 0; i < warps; i++) r = IS_MAX ? fmaxf(r, scratch[i]) : r + scratch[i];
    return r;
}

__device__ __forceinline__ uint4 load_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <int N, int K, bool BIAS, bool RESIDUAL>
__device__ void linear_bf16(const void* w, const float* scales, const float* bias, const float* x, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    const bf16_t* wr = (const bf16_t*)w + (size_t)row * K;
    float acc = 0.0f;
#pragma unroll 8
    for (int k = lane * 8; k < K; k += 256) {
        uint4 packed = load_stream(wr + k);
        float4 xa = *(const float4*)(x + k);
        float4 xb = *(const float4*)(x + k + 4);
        acc += __uint_as_float(packed.x << 16) * xa.x + __uint_as_float(packed.x & 0xffff0000u) * xa.y;
        acc += __uint_as_float(packed.y << 16) * xa.z + __uint_as_float(packed.y & 0xffff0000u) * xa.w;
        acc += __uint_as_float(packed.z << 16) * xb.x + __uint_as_float(packed.z & 0xffff0000u) * xb.y;
        acc += __uint_as_float(packed.w << 16) * xb.z + __uint_as_float(packed.w & 0xffff0000u) * xb.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (BIAS) acc += bias[row];
        out[row] = RESIDUAL ? out[row] + acc : acc;
    }
}

template <int N, int K, bool BIAS, bool RESIDUAL>
__device__ void linear_q8(const void* w, const float* scales, const float* bias, const float* x, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    const i8_t* wr = (const i8_t*)w + (size_t)row * K;
    const float* sr = scales + (size_t)row * (K / Q8_GROUP);
    float acc = 0.0f;
#pragma unroll 2
    for (int k = lane * 16; k < K; k += 512) {
        int4 packed = *(const int4*)(wr + k);
        const i8_t* q = (const i8_t*)&packed;
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; j += 4) {
            float4 xv = *(const float4*)(x + k + j);
            part += (float)q[j] * xv.x + (float)q[j + 1] * xv.y + (float)q[j + 2] * xv.z + (float)q[j + 3] * xv.w;
        }
        acc += part * sr[k / Q8_GROUP];
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (BIAS) acc += bias[row];
        out[row] = RESIDUAL ? out[row] + acc : acc;
    }
}

template <int WF, int K>
__device__ __forceinline__ float row_dot(const void* w, const float* scales, int row, const float* x, int lane) {
    float acc = 0.0f;
    if constexpr (WF == WF_FP8) {
        static_assert(K % 16 == 0, "fp8 rows load 16 bytes per lane");
        const unsigned char* wr = (const unsigned char*)w + (size_t)row * K;
        const float* sr = scales + (size_t)row * (K / FP8_BLOCK);
#pragma unroll 2
        for (int k = lane * 16; k < K; k += 512) {
            uint4 packed = load_stream(wr + k);
            float part = e4m3_dot4(packed.x, *(const float4*)(x + k)) + e4m3_dot4(packed.y, *(const float4*)(x + k + 4)) +
                         e4m3_dot4(packed.z, *(const float4*)(x + k + 8)) + e4m3_dot4(packed.w, *(const float4*)(x + k + 12));
            acc += part * sr[k / FP8_BLOCK];
        }
    } else if constexpr (WF == WF_Q8) {
        const i8_t* wr = (const i8_t*)w + (size_t)row * K;
        const float* sr = scales + (size_t)row * (K / Q8_GROUP);
#pragma unroll 2
        for (int k = lane * 16; k < K; k += 512) {
            int4 packed = *(const int4*)(wr + k);
            const i8_t* q = (const i8_t*)&packed;
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; j += 4) {
                float4 xv = *(const float4*)(x + k + j);
                part += (float)q[j] * xv.x + (float)q[j + 1] * xv.y + (float)q[j + 2] * xv.z + (float)q[j + 3] * xv.w;
            }
            acc += part * sr[k / Q8_GROUP];
        }
    } else {
        const bf16_t* wr = (const bf16_t*)w + (size_t)row * K;
#pragma unroll 8
        for (int k = lane * 8; k < K; k += 256) {
            uint4 packed = load_stream(wr + k);
            float4 xa = *(const float4*)(x + k);
            float4 xb = *(const float4*)(x + k + 4);
            acc += __uint_as_float(packed.x << 16) * xa.x + __uint_as_float(packed.x & 0xffff0000u) * xa.y;
            acc += __uint_as_float(packed.y << 16) * xa.z + __uint_as_float(packed.y & 0xffff0000u) * xa.w;
            acc += __uint_as_float(packed.z << 16) * xb.x + __uint_as_float(packed.z & 0xffff0000u) * xb.y;
            acc += __uint_as_float(packed.w << 16) * xb.z + __uint_as_float(packed.w & 0xffff0000u) * xb.w;
        }
    }
    return warp_sum(acc);
}

template <int N, int K, bool BIAS, bool RESIDUAL>
__device__ void linear_fp8(const void* w, const float* scales, const float* bias, const float* x, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    float acc = row_dot<WF_FP8, K>(w, scales, row, x, lane);
    if (lane == 0) {
        if (BIAS) acc += bias[row];
        out[row] = RESIDUAL ? out[row] + acc : acc;
    }
}

__device__ __forceinline__ void load_normed(const float* x, const float* nw, float* xs) {
    __shared__ float scratch[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = x[i];
        xs[i] = v;
        ss += v * v;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[i] = nw[i] * (xs[i] * inv);
    __syncthreads();
}

template <int WF, int N, bool BIAS>
__device__ void linear_norm(const void* w, const float* scales, const float* bias, const float* x,
                            const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    float acc = row_dot<WF, HIDDEN>(w, scales, row, xs, lane);
    if (lane == 0) out[row] = BIAS ? acc + bias[row] : acc;
}

__device__ __forceinline__ float activation(float g) {
    return ACT_KIND == 1
        ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
        : g / (1.0f + expf(-g));
}

template <int WF>
__device__ void gate_up_act_norm(const void* w, const float* scales, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= INTER) return;
    float g = row_dot<WF, HIDDEN>(w, scales, 2 * row, xs, lane);
    float u = row_dot<WF, HIDDEN>(w, scales, 2 * row + 1, xs, lane);
    if (lane == 0) out[row] = activation(g) * u;
}

constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 16;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;

template <int N, int K, bool BIAS, bool RESIDUAL, int WF>
__device__ void gemm(const void* w, const float* scales, const float* bias, const float* x, float* out, int tokens) {
    __shared__ float xs[GEMM_BK][GEMM_BM];
    __shared__ float ws[GEMM_BK][GEMM_BN];
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    int m0 = blockIdx.y * GEMM_BM;
    int n0 = blockIdx.x * GEMM_BN;
    float acc[GEMM_TM][GEMM_TN];
#pragma unroll
    for (int i = 0; i < GEMM_TM; i++)
#pragma unroll
        for (int j = 0; j < GEMM_TN; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int r = idx / GEMM_BK;
            int kk = idx % GEMM_BK;
            int m = m0 + r;
            xs[kk][r] = m < tokens ? x[(size_t)m * K + k0 + kk] : 0.0f;
            int n = n0 + r;
            float wv = 0.0f;
            if (n < N) {
                if (WF == WF_Q8) {
                    wv = (float)((const i8_t*)w)[(size_t)n * K + k0 + kk] *
                         scales[(size_t)n * (K / Q8_GROUP) + (k0 + kk) / Q8_GROUP];
                } else if (WF == WF_FP8) {
                    wv = e4m3_to_f32(((const unsigned char*)w)[(size_t)n * K + k0 + kk]) *
                         scales[(size_t)n * (K / FP8_BLOCK) + (k0 + kk) / FP8_BLOCK];
                } else {
                    wv = bf16_to_f32(((const bf16_t*)w)[(size_t)n * K + k0 + kk]);
                }
            }
            ws[kk][r] = wv;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GEMM_BK; kk++) {
            float a[GEMM_TM];
            float b[GEMM_TN];
#pragma unroll
            for (int i = 0; i < GEMM_TM; i++) a[i] = xs[kk][ty * GEMM_TM + i];
#pragma unroll
            for (int j = 0; j < GEMM_TN; j++) b[j] = ws[kk][tx * GEMM_TN + j];
#pragma unroll
            for (int i = 0; i < GEMM_TM; i++)
#pragma unroll
                for (int j = 0; j < GEMM_TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < GEMM_TM; i++) {
        int m = m0 + ty * GEMM_TM + i;
        if (m >= tokens) continue;
#pragma unroll
        for (int j = 0; j < GEMM_TN; j++) {
            int n = n0 + tx * GEMM_TN + j;
            if (n >= N) continue;
            float v = acc[i][j];
            if (BIAS) v += bias[n];
            float* o = out + (size_t)m * N + n;
            *o = RESIDUAL ? *o + v : v;
        }
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(PENDING));
}

__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ unsigned bf16_bits_rn(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ void split_hi_lo(float v, unsigned& hi, unsigned& lo) {
    hi = bf16_bits_rn(v);
    lo = bf16_bits_rn(v - __uint_as_float(hi << 16));
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

__device__ __forceinline__ unsigned q8_pair_bf16(unsigned two_bytes) {
    int q0 = (signed char)(two_bytes & 0xff);
    int q1 = (signed char)((two_bytes >> 8) & 0xff);
    return (__float_as_uint((float)q0) >> 16) | (__float_as_uint((float)q1) & 0xffff0000u);
}

constexpr int EPI_STORE = 0;
constexpr int EPI_RESIDUAL = 1;
constexpr int EPI_ACT = 2;

__device__ __forceinline__ void split_store2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha, la, hb, lb;
    split_hi_lo(a, ha, la);
    split_hi_lo(b, hb, lb);
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR >= 4, "rows of at least 64 bytes");
    constexpr int GROUP_ROWS = CPR >= 8 ? 1 : 8 / CPR;
    constexpr int MASK = CPR >= 8 ? 7 : CPR - 1;
    return r * CPR + (c ^ ((r / GROUP_ROWS) & MASK));
}

template <int N, int K, int BM, int BN, int WM, int WN, int BK, int STAGES, int EPI, bool BIAS, int WF, bool SPLIT>
__device__ __forceinline__ void gemm_tc(const void* w, const float* scales, const float* bias, const bf16_t* x_hi,
                                        const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr bool BYTES = WF != WF_BF16;
    constexpr int GROUP_K = WF == WF_FP8 ? FP8_BLOCK : Q8_GROUP;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 16;
    constexpr int CPR = BK / 8;
    constexpr int PLANES = SPLIT ? 2 : 1;
    constexpr int A_TILE = BM * BK;
    constexpr int A_BYTES = A_TILE * 2 * PLANES;
    constexpr int WROW = BYTES ? BK + 16 : BK * 2;
    constexpr int STAGE_BYTES = A_BYTES + BN * WROW;
    constexpr int KT = K / BK;
    constexpr int A_CHUNKS = BK / 8;
    constexpr int A_TOTAL = BM * A_CHUNKS * PLANES;
    constexpr int A_ITERS = (A_TOTAL + THREADS - 1) / THREADS;
    constexpr int W_CHUNKS = BYTES ? BK / 16 : BK / 8;
    constexpr int W_ITERS = (BN * W_CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of BK");
    static_assert(!BYTES || GROUP_K % BK == 0, "byte weight tiles stay inside one scale group");
    static_assert(NJ % 2 == 0, "warp n tile");
    static_assert(STAGES >= 2, "pipeline depth");
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
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != A_TOTAL && idx >= A_TOTAL) break;
            int p = idx / (BM * A_CHUNKS);
            int rem = idx % (BM * A_CHUNKS);
            int r = rem / A_CHUNKS;
            int c = rem % A_CHUNKS;
            int m = m0 + r;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last_row) * K + kt * BK + c * 8;
            cp_async16_zfill(base + p * A_TILE * 2 + swz<CPR>(r, c) * 16, src, m < tokens);
        }
        unsigned char* ws = base + A_BYTES;
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * W_CHUNKS && idx >= BN * W_CHUNKS) break;
            int r = idx / W_CHUNKS;
            int ck = idx % W_CHUNKS;
            int n = min(n0 + r, N - 1);
            const char* src = BYTES ? (const char*)w + (size_t)n * K + kt * BK + ck * 16
                                    : (const char*)w + ((size_t)n * K + kt * BK) * 2 + ck * 16;
            cp_async16(ws + (BYTES ? r * WROW + ck * 16 : swz<CPR>(r, ck) * 16), src);
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
        float part[SPLIT ? MI : 1][SPLIT ? NJ : 1][4];
        if (SPLIT) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) part[i][j][e] = 0.0f;
        }
        float s0[NJ], s1[NJ];
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            s0[j] = 1.0f;
            s1[j] = 1.0f;
            if (BYTES) {
                int col = n0 + wn * TN + j * 8 + 2 * t;
                int group = kt * BK / GROUP_K;
                s0[j] = scales[(size_t)min(col, N - 1) * (K / GROUP_K) + group];
                s1[j] = scales[(size_t)min(col + 1, N - 1) * (K / GROUP_K) + group];
            }
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned bw[NJ][2];
            if (BYTES) {
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    const unsigned char* p = ws + (wn * TN + j * 8 + g) * WROW + ks * 16 + 2 * t;
                    unsigned lo = *(const unsigned short*)p;
                    unsigned hi = *(const unsigned short*)(p + 8);
                    bw[j][0] = WF == WF_FP8 ? e4m3_pair_bf16(lo) : q8_pair_bf16(lo);
                    bw[j][1] = WF == WF_FP8 ? e4m3_pair_bf16(hi) : q8_pair_bf16(hi);
                }
            } else {
#pragma unroll
                for (int jp = 0; jp < NJ / 2; jp++) {
                    unsigned r[4];
                    int row = wn * TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                    ldsm_x4(r, ws + swz<CPR>(row, ks * 2 + ((lane >> 3) & 1)) * 16);
                    bw[2 * jp][0] = r[0];
                    bw[2 * jp][1] = r[1];
                    bw[2 * jp + 1][0] = r[2];
                    bw[2 * jp + 1][1] = r[3];
                }
            }
            unsigned ah[MI][4], al[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int row = wm * TM + i * 16 + (lane & 15);
                int at = swz<CPR>(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah[i], &a_hi[at]);
                if (SPLIT) ldsm_x4(al[i], &a_lo[at]);
            }
            if (SPLIT && BYTES) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            } else if (SPLIT) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma_bf16(part[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(part[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                    }
            } else {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            }
        }
        if (SPLIT && !BYTES) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] += part[i][j][e];
        }
    }
    cp_async_wait<0>();
    if (EPI == EPI_ACT) {
        static_assert(EPI != EPI_ACT || N % 4 == 0, "gate/up pairs");
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                int col = n0 + wn * TN + j * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int m = m0 + wm * TM + i * 16 + g + h * 8;
                    float a = activation(acc[i][j][h * 2]) * acc[i][j][h * 2 + 1];
                    float b = __shfl_xor_sync(0xffffffff, a, 1);
                    if ((t & 1) == 0 && col < N && m < tokens)
                        split_store2(out_hi, out_lo, (size_t)m * (N / 2) + col / 2, a, b);
                }
            }
        return;
    }
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = n0 + wn * TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = 0.0f, b1 = 0.0f;
            if (BIAS) {
                b0 = bias[col];
                b1 = col + 1 < N ? bias[col + 1] : 0.0f;
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * TM + i * 16 + g + h * 8;
                if (m >= tokens) continue;
                float v0 = acc[i][j][h * 2] + b0;
                float v1 = acc[i][j][h * 2 + 1] + b1;
                {
                    float* o = out + (size_t)m * N + col;
                    if (EPI == EPI_RESIDUAL) {
                        v0 += o[0];
                        if (col + 1 < N) v1 += o[1];
                    }
                    o[0] = v0;
                    if (col + 1 < N) o[1] = v1;
                }
            }
        }
}

extern "C" __global__ void rms_split(const float* x, const float* weight, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    static_assert(HIDDEN % 4 == 0, "hidden size");
    size_t row = (size_t)blockIdx.x * HIDDEN;
    const float4* xv = (const float4*)(x + row);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN / 4; i += blockDim.x) {
        float4 v = xv[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN / 4; i += blockDim.x) {
        float4 v = xv[i];
        float4 nv = ((const float4*)weight)[i];
        split_store2(hi, lo, row + 4 * i, nv.x * (v.x * inv), nv.y * (v.y * inv));
        split_store2(hi, lo, row + 4 * i + 2, nv.z * (v.z * inv), nv.w * (v.w * inv));
    }
}

extern "C" __global__ void embed(const bf16_t* table, const int* token, float* x) {
    const bf16_t* row = table + (size_t)token[blockIdx.y] * HIDDEN;
    x += (size_t)blockIdx.y * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x)
        x[i] = bf16_to_f32(row[i]);
}

extern "C" __global__ void rmsnorm(const float* x, const float* weight, float* out) {
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * HIDDEN;
    out += (size_t)blockIdx.x * HIDDEN;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ss += x[i] * x[i];
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) out[i] = weight[i] * (x[i] * inv);
}

__device__ __forceinline__ float rope_inv_freq(int i) {
    float inv_freq = 1.0f / powf(THETA, (float)(2 * i) / (float)ROT);
    if (ROPE_KIND == 1) {
        inv_freq /= ROPE_FACTOR;
    } else if (ROPE_KIND == 2) {
        float wavelen = 6.283185307179586f / inv_freq;
        float low_wavelen = ROPE_ORIG_POS / ROPE_LOW_FREQ;
        float high_wavelen = ROPE_ORIG_POS / ROPE_HIGH_FREQ;
        if (wavelen > low_wavelen) {
            inv_freq /= ROPE_FACTOR;
        } else if (wavelen >= high_wavelen) {
            float smooth = (ROPE_ORIG_POS / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
            inv_freq = (1.0f - smooth) * inv_freq / ROPE_FACTOR + smooth * inv_freq;
        }
    }
    return inv_freq;
}

__device__ __forceinline__ unsigned f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float f16_to_f32(unsigned bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ unsigned attn_bits(float v) {
    return f16_bits(v);
}

__device__ __forceinline__ float attn_float(unsigned bits) {
    return f16_to_f32(bits & 0xffffu);
}

__device__ __forceinline__ void attn_split(float v, unsigned& hi, unsigned& lo) {
    hi = attn_bits(v);
    lo = attn_bits(v - attn_float(hi));
}

__device__ __forceinline__ size_t kv_slot(const int* block_table, int layer, int kv_head, int pos) {
    return (((size_t)layer * PAGES + block_table[pos / PAGE]) * KV_HEADS + kv_head) * PAGE + pos % PAGE;
}

__device__ __forceinline__ bf16_t* kv_row(bf16_t* cache, const int* block_table, int layer, int kv_head, int pos) {
    return cache + kv_slot(block_table, layer, kv_head, pos) * 2 * HEAD_DIM;
}

__device__ __forceinline__ const bf16_t* kv_row(const bf16_t* cache, const int* block_table, int layer, int kv_head,
                                                int pos) {
    return cache + kv_slot(block_table, layer, kv_head, pos) * 2 * HEAD_DIM;
}

__device__ __forceinline__ float kv_at(const bf16_t* row, int d) {
    return attn_float(row[d]) + attn_float(row[HEAD_DIM + d]);
}

__device__ __forceinline__ float4 kv_at4(const bf16_t* row, int d) {
    uint2 h = *(const uint2*)(row + d);
    uint2 l = *(const uint2*)(row + HEAD_DIM + d);
    return make_float4(attn_float(h.x) + attn_float(l.x), attn_float(h.x >> 16) + attn_float(l.x >> 16),
                       attn_float(h.y) + attn_float(l.y), attn_float(h.y >> 16) + attn_float(l.y >> 16));
}

__device__ __forceinline__ void norm_rope_head(float* qkv, const float* q_norm, const float* k_norm,
                                               bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                               int pos, int layer, int head, int row) {
    __shared__ float scratch[32];
    __shared__ float vec[HEAD_DIM];
    int d = threadIdx.x;
    bool is_q = head < HEADS;
    bool is_k = !is_q && head < HEADS + KV_HEADS;
    float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
    float v = d < HEAD_DIM ? src[d] : 0.0f;
    if (QK_NORM && (is_q || is_k)) {
        float inv = rsqrtf(block_reduce<false>(v * v, scratch) / HEAD_DIM + EPS);
        const float* w = is_q ? q_norm : k_norm;
        if (d < HEAD_DIM) v = w[d] * (v * inv);
    }
    if (d < HEAD_DIM) vec[d] = v;
    __syncthreads();
    if ((is_q || is_k) && d < ROT) {
        float angle = (float)pos * rope_inv_freq(d < ROT_HALF ? d : d - ROT_HALF);
        float c = cosf(angle);
        float s = sinf(angle);
        v = d < ROT_HALF ? vec[d] * c - vec[d + ROT_HALF] * s : vec[d] * c + vec[d - ROT_HALF] * s;
    }
    if (d >= HEAD_DIM) return;
    if (is_q) {
        src[d] = v;
        return;
    }
    int kv_head = is_k ? head - HEADS : head - HEADS - KV_HEADS;
    bf16_t* dst = kv_row(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos);
    unsigned hi, lo;
    attn_split(v, hi, lo);
    dst[d] = (bf16_t)hi;
    dst[HEAD_DIM + d] = (bf16_t)lo;
}

extern "C" __global__ void qk_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer, int seq_layers) {
    int pos = pos_ptr[0] + (seq_layers ? 0 : blockIdx.y);
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, block_table, pos, layer + blockIdx.y * seq_layers,
                   blockIdx.x, blockIdx.y);
}

extern "C" __global__ void kv_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer) {
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, block_table, pos_ptr[0] + blockIdx.y, layer,
                   HEADS + blockIdx.x, blockIdx.y);
}

constexpr int ATTN_SPLITS = 64;
constexpr int ATTN_TILE = 16;
constexpr int ATTN_MIN_CHUNK = 16;
constexpr int ATTN_BLOCK = 128;
constexpr int ATTN_V4 = (HEAD_DIM / 4 + 31) / 32;
constexpr int ATTN_PART = HEAD_DIM + 2;

__device__ __forceinline__ int attn_chunk(int len) {
    return max(ATTN_MIN_CHUNK, (len + ATTN_SPLITS - 1) / ATTN_SPLITS);
}

extern "C" __global__ void attention_split(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* part) {
    __shared__ float q_s[HEAD_DIM];
    __shared__ float p_s[ATTN_TILE];
    int head = blockIdx.x;
    int split = blockIdx.y;
    int kv_head = head / GROUP;
    int len = pos_ptr[0] + 1;
    int chunk = attn_chunk(len);
    int start = split * chunk;
    if (start >= len) return;
    int end = min(len, start + chunk);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const float* q = qkv + head * HEAD_DIM;
    for (int d = threadIdx.x; d < HEAD_DIM; d += ATTN_BLOCK) q_s[d] = q[d];
    __syncthreads();
    float scale = rsqrtf((float)HEAD_DIM);
    float m = NEG_INF;
    float l = 0.0f;
    float4 acc[ATTN_V4];
    for (int j = 0; j < ATTN_V4; j++) acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    static_assert(HEAD_DIM % 4 == 0 && HEAD_DIM <= 256, "unsupported head_dim");
    for (int t0 = start; t0 < end; t0 += ATTN_TILE) {
        int n = min(ATTN_TILE, end - t0);
        for (int i = warp; i < n; i += ATTN_BLOCK / 32) {
            int t = t0 + i;
            const bf16_t* k = kv_row(k_cache, block_table, layer, kv_head, t);
            float dot = 0.0f;
            for (int d = lane; d < HEAD_DIM / 4; d += 32) {
                float4 kv = kv_at4(k, 4 * d);
                dot += q_s[4 * d] * kv.x + q_s[4 * d + 1] * kv.y + q_s[4 * d + 2] * kv.z + q_s[4 * d + 3] * kv.w;
            }
            dot = warp_sum(dot);
            if (lane == 0) p_s[i] = dot * scale;
        }
        __syncthreads();
        float tmax = NEG_INF;
        for (int i = 0; i < n; i++) tmax = fmaxf(tmax, p_s[i]);
        float m_new = fmaxf(m, tmax);
        float corr = expf(m - m_new);
        __syncthreads();
        if (threadIdx.x < n) p_s[threadIdx.x] = expf(p_s[threadIdx.x] - m_new);
        __syncthreads();
        float psum = 0.0f;
        for (int i = 0; i < n; i++) psum += p_s[i];
        l = l * corr + psum;
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++) {
            int c = lane + 32 * j;
            float4 pv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < HEAD_DIM / 4) {
#pragma unroll 4
                for (int i = warp; i < n; i += ATTN_BLOCK / 32) {
                    int t = t0 + i;
                    float p = p_s[i];
                    float4 v = kv_at4(kv_row(v_cache, block_table, layer, kv_head, t), 4 * c);
                    pv.x += p * v.x;
                    pv.y += p * v.y;
                    pv.z += p * v.z;
                    pv.w += p * v.w;
                }
            }
            acc[j].x = acc[j].x * corr + pv.x;
            acc[j].y = acc[j].y * corr + pv.y;
            acc[j].z = acc[j].z * corr + pv.z;
            acc[j].w = acc[j].w * corr + pv.w;
        }
        m = m_new;
        __syncthreads();
    }
    float* dst = part + ((size_t)head * ATTN_SPLITS + split) * ATTN_PART;
    __shared__ float4 acc_s[ATTN_BLOCK / 32][HEAD_DIM / 4];
    for (int j = 0; j < ATTN_V4; j++)
        if (lane + 32 * j < HEAD_DIM / 4) acc_s[warp][lane + 32 * j] = acc[j];
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += ATTN_BLOCK) {
        float sum = 0.0f;
        for (int w = 0; w < ATTN_BLOCK / 32; w++) sum += ((const float*)acc_s[w])[d];
        dst[d] = sum;
    }
    if (threadIdx.x == 0) {
        dst[HEAD_DIM] = m;
        dst[HEAD_DIM + 1] = l;
    }
}

extern "C" __global__ void attention_reduce(const float* part, const int* pos_ptr, const float* qkv, float* out) {
    int head = blockIdx.x;
    int len = pos_ptr[0] + 1;
    int chunk = attn_chunk(len);
    int active = (len + chunk - 1) / chunk;
    const float* base = part + (size_t)head * ATTN_SPLITS * ATTN_PART;
    __shared__ float w_s[ATTN_SPLITS];
    int d = threadIdx.x;
    float ms = d < active ? base[d * ATTN_PART + HEAD_DIM] : NEG_INF;
    float ls = d < active ? base[d * ATTN_PART + HEAD_DIM + 1] : 0.0f;
    float m = warp_max(ms);
    if (ATTN_SPLITS > 32) {
        __shared__ float wm[ATTN_SPLITS / 32 + 1];
        if ((d & 31) == 0 && d < ATTN_SPLITS) wm[d >> 5] = m;
        __syncthreads();
        m = NEG_INF;
        for (int i = 0; i <= (ATTN_SPLITS - 1) / 32; i++) m = fmaxf(m, wm[i]);
    }
    float w = d < active ? expf(ms - m) : 0.0f;
    if (d < ATTN_SPLITS) w_s[d] = w;
    __syncthreads();
    float l = 0.0f;
    for (int i = 0; i < active; i++) l += w_s[i] * base[i * ATTN_PART + HEAD_DIM + 1];
    float total = 0.0f;
    if (d < HEAD_DIM) {
#pragma unroll 8
        for (int i = 0; i < active; i++) total += w_s[i] * base[i * ATTN_PART + d];
        out[head * HEAD_DIM + d] = total / l * out_gate(qkv, 0, head * HEAD_DIM + d);
    }
}

extern "C" __global__ void attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* out, float* scores_global,
                                     int seq_layers) {
    __shared__ float scratch[32];
    __shared__ float scores_shared[MAX_CTX > 4096 ? 1 : MAX_CTX];
    int head = blockIdx.x;
    float* scores = MAX_CTX > 4096 ? scores_global + ((size_t)blockIdx.y * HEADS + head) * MAX_CTX : scores_shared;
    int kv_head = head / GROUP;
    int len = pos_ptr[0] + (seq_layers ? 0 : blockIdx.y) + 1;
    layer += blockIdx.y * seq_layers;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const float* q = qkv + (size_t)blockIdx.y * QKV_ROW + head * HEAD_DIM;
    out += (size_t)blockIdx.y * Q_DIM;
    float scale = rsqrtf((float)HEAD_DIM);
    for (int t = warp; t < len; t += ATTN_THREADS / 32) {
        const bf16_t* k = kv_row(k_cache, block_table, layer, kv_head, t);
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) dot += q[d] * kv_at(k, d);
        dot = warp_sum(dot);
        if (lane == 0) scores[t] = dot * scale;
    }
    __syncthreads();
    float m = NEG_INF;
    for (int t = threadIdx.x; t < len; t += blockDim.x) m = fmaxf(m, scores[t]);
    m = block_reduce<true>(m, scratch);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        float e = expf(scores[t] - m);
        scores[t] = e;
        sum += e;
    }
    sum = block_reduce<false>(sum, scratch);
    __syncthreads();
    constexpr int PER_LANE = (HEAD_DIM + 31) / 32;
    constexpr int WARPS = ATTN_THREADS / 32;
    __shared__ float partial[WARPS][HEAD_DIM];
    float acc[PER_LANE];
#pragma unroll
    for (int i = 0; i < PER_LANE; i++) acc[i] = 0.0f;
#pragma unroll 4
    for (int t = warp; t < len; t += WARPS) {
        const bf16_t* v = kv_row(v_cache, block_table, layer, kv_head, t);
        float p = scores[t];
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            if (d < HEAD_DIM) acc[i] += p * kv_at(v, d);
        }
    }
#pragma unroll
    for (int i = 0; i < PER_LANE; i++) {
        int d = lane + i * 32;
        if (d < HEAD_DIM) partial[warp][d] = acc[i];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float total = 0.0f;
        for (int w = 0; w < WARPS; w++) total += partial[w][d];
        out[head * HEAD_DIM + d] = total / sum * out_gate(qkv, blockIdx.y, head * HEAD_DIM + d);
    }
}

constexpr int BD_WARPS = 8;
constexpr int BD_SPLITS = {{BD_SPLITS}};
constexpr int BD_UNROLL = 8;

extern "C" __global__ void batch_decode_attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                                  const int* block_table, const int* pos_ptr, int layer,
                                                  float* part_acc, float* part_ml, int seq_layers) {
    constexpr int PER_LANE = (HEAD_DIM + 31) / 32;
    __shared__ float sm[BD_WARPS][GROUP];
    __shared__ float sl[BD_WARPS][GROUP];
    __shared__ float sacc[BD_WARPS][GROUP][HEAD_DIM];
    int kv_head = blockIdx.x;
    int seq = blockIdx.y;
    int split = blockIdx.z;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int len = pos_ptr[0] + 1;
    int chunk = (len + BD_SPLITS - 1) / BD_SPLITS;
    int t0 = split * chunk;
    int t1 = min(len, t0 + chunk);
    int seq_layer = layer + seq * seq_layers;
    const float* q = qkv + (size_t)seq * QKV_ROW + kv_head * GROUP * HEAD_DIM;
    float scale = rsqrtf((float)HEAD_DIM);
    float qr[GROUP][PER_LANE];
#pragma unroll
    for (int g = 0; g < GROUP; g++)
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            qr[g][i] = d < HEAD_DIM ? q[g * HEAD_DIM + d] * scale : 0.0f;
        }
    float m[GROUP], l[GROUP], acc[GROUP][PER_LANE];
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        m[g] = NEG_INF;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) acc[g][i] = 0.0f;
    }
    constexpr int U = BD_UNROLL;
    for (int tb = t0 + warp; tb < t1; tb += BD_WARPS * U) {
        float kr[U][PER_LANE], vr[U][PER_LANE];
#pragma unroll
        for (int u = 0; u < U; u++) {
            int t = tb + u * BD_WARPS;
            const bf16_t* k = kv_row(k_cache, block_table, seq_layer, kv_head, min(t, t1 - 1));
            const bf16_t* v = kv_row(v_cache, block_table, seq_layer, kv_head, min(t, t1 - 1));
#pragma unroll
            for (int i = 0; i < PER_LANE; i++) {
                int d = lane + i * 32;
                kr[u][i] = d < HEAD_DIM ? kv_at(k, d) : 0.0f;
                vr[u][i] = d < HEAD_DIM ? kv_at(v, d) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (tb + u * BD_WARPS >= t1) break;
#pragma unroll
            for (int g = 0; g < GROUP; g++) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < PER_LANE; i++) dot += qr[g][i] * kr[u][i];
                dot = warp_sum(dot);
                float nm = fmaxf(m[g], dot);
                float corr = expf(m[g] - nm);
                float p = expf(dot - nm);
                l[g] = l[g] * corr + p;
#pragma unroll
                for (int i = 0; i < PER_LANE; i++) acc[g][i] = acc[g][i] * corr + p * vr[u][i];
                m[g] = nm;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        if (lane == 0) {
            sm[warp][g] = m[g];
            sl[warp][g] = l[g];
        }
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            if (d < HEAD_DIM) sacc[warp][g][d] = acc[g][i];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < GROUP * HEAD_DIM; idx += blockDim.x) {
        int g = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        float M = NEG_INF;
        for (int w = 0; w < BD_WARPS; w++) M = fmaxf(M, sm[w][g]);
        float L = 0.0f, A = 0.0f;
        if (M != NEG_INF) {
            for (int w = 0; w < BD_WARPS; w++) {
                float c = sm[w][g] == NEG_INF ? 0.0f : expf(sm[w][g] - M);
                L += sl[w][g] * c;
                A += sacc[w][g][d] * c;
            }
        }
        int head = kv_head * GROUP + g;
        size_t base = ((size_t)seq * HEADS + head) * BD_SPLITS + split;
        part_acc[base * HEAD_DIM + d] = A;
        if (d == 0) {
            part_ml[base * 2] = M;
            part_ml[base * 2 + 1] = L;
        }
    }
}

extern "C" __global__ void batch_decode_combine(const float* part_acc, const float* part_ml, const float* qkv, bf16_t* out_hi, bf16_t* out_lo) {
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int d = 2 * threadIdx.x;
    size_t base = ((size_t)seq * HEADS + head) * BD_SPLITS;
    float M = NEG_INF;
    for (int s = 0; s < BD_SPLITS; s++) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, A0 = 0.0f, A1 = 0.0f;
    for (int s = 0; s < BD_SPLITS; s++) {
        float ms = part_ml[(base + s) * 2];
        float c = ms == NEG_INF ? 0.0f : expf(ms - M);
        L += part_ml[(base + s) * 2 + 1] * c;
        A0 += part_acc[(base + s) * HEAD_DIM + d] * c;
        A1 += part_acc[(base + s) * HEAD_DIM + d + 1] * c;
    }
    int at = head * HEAD_DIM + d;
    split_store2(out_hi, out_lo, (size_t)seq * Q_DIM + at, A0 / L * out_gate(qkv, seq, at), A1 / L * out_gate(qkv, seq, at + 1));
}

constexpr int FA_BQ = 16;
constexpr int FA_BK = 32;
constexpr int FA_THREADS = 128;
constexpr int FA_STRIDE = HEAD_DIM + 1;
constexpr int FA_OUT = HEAD_DIM / 8;

extern "C" __global__ void flash_attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                           const int* block_table, const int* pos_ptr, int layer, float* out, int tokens) {
    __shared__ float qs[FA_BQ][FA_STRIDE];
    __shared__ float ks[FA_BK][FA_STRIDE];
    __shared__ float vs[FA_BK][HEAD_DIM];
    __shared__ float ps[FA_BQ][FA_BK + 1];
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = blockIdx.x * FA_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int qr = tid >> 3;
    int sub = tid & 7;
    int qi = q0 + qr;
    int pos = base + qi;
    float scale = rsqrtf((float)HEAD_DIM);
    for (int i = tid; i < FA_BQ * HEAD_DIM; i += FA_THREADS) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q = q0 + r;
        qs[r][d] = q < tokens ? qkv[(size_t)q * QKV_ROW + head * HEAD_DIM + d] : 0.0f;
    }
    float o[FA_OUT];
#pragma unroll
    for (int i = 0; i < FA_OUT; i++) o[i] = 0.0f;
    float m = NEG_INF;
    float l = 0.0f;
    int last = min(q0 + FA_BQ, tokens) - 1;
    int key_end = base + last + 1;
    for (int k0 = 0; k0 < key_end; k0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * HEAD_DIM; i += FA_THREADS) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int t = k0 + r;
            float kv = 0.0f, vv = 0.0f;
            if (t < key_end) {
                kv = kv_at(kv_row(k_cache, block_table, layer, kv_head, t), d);
                vv = kv_at(kv_row(v_cache, block_table, layer, kv_head, t), d);
            }
            ks[r][d] = kv;
            vs[r][d] = vv;
        }
        __syncthreads();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++) {
            float qv = qs[qr][d];
#pragma unroll
            for (int j = 0; j < 4; j++) s[j] += qv * ks[sub * 4 + j][d];
        }
        float tile_max = NEG_INF;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = k0 + sub * 4 + j;
            s[j] = (t <= pos && t < key_end) ? s[j] * scale : NEG_INF;
            tile_max = fmaxf(tile_max, s[j]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
        float m_new = fmaxf(m, tile_max);
        float alpha = m == NEG_INF ? 0.0f : expf(m - m_new);
        float row_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float p = s[j] == NEG_INF ? 0.0f : expf(s[j] - m_new);
            ps[qr][sub * 4 + j] = p;
            row_sum += p;
        }
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);
        l = l * alpha + row_sum;
        m = m_new;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) o[i] *= alpha;
        __syncwarp();
#pragma unroll 8
        for (int j = 0; j < FA_BK; j++) {
            float p = ps[qr][j];
#pragma unroll
            for (int i = 0; i < FA_OUT; i++) o[i] += p * vs[j][sub + 8 * i];
        }
    }
    if (qi < tokens) {
        float inv = 1.0f / l;
        float* dst = out + (size_t)qi * Q_DIM + head * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) dst[sub + 8 * i] = o[i] * inv * out_gate(qkv, qi, head * HEAD_DIM + sub + 8 * i);
    }
}

constexpr int FM_WARPS = {{FM_WARPS}};
constexpr int FM_BQ = 16 * FM_WARPS;
constexpr int FM_BK = {{FM_BK}};
constexpr int FM_STAGES = {{FM_STAGES}};
constexpr int FM_KB = HEAD_DIM / 16;
constexpr int FM_NB = HEAD_DIM / 8;

__device__ __forceinline__ unsigned pack_hi(float a, float b) {
    return attn_bits(a) | (attn_bits(b) << 16);
}

__device__ __forceinline__ unsigned pack_lo(float a, float b) {
    return attn_bits(a - attn_float(attn_bits(a))) | (attn_bits(b - attn_float(attn_bits(b))) << 16);
}

__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma_hi_lo(float* c, const unsigned* ah, const unsigned* al,
                                          unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma_f16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

extern "C" __global__ void __launch_bounds__(32 * FM_WARPS) flash_mma(const float* qkv, const float* q_norm, const bf16_t* k_cache,
                                     const bf16_t* v_cache, const int* block_table, const int* pos_ptr, int layer,
                                     bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = FM_BK * HEAD_DIM;
    static_assert(FM_BQ * HEAD_DIM * 4 <= {{FM_SMEM}}, "Q staging fits");
    extern __shared__ __align__(16) unsigned char fm_smem[];
    bf16_t* kvs = (bf16_t*)fm_smem;
    float* qs = (float*)fm_smem;
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = (gridDim.x - 1 - blockIdx.x) * FM_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int pos0 = base + row0;
    int pos1 = base + row1;
    float scale_log2 = rsqrtf((float)HEAD_DIM) * 1.4426950408889634f;
    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
    constexpr int PAIRS = (ROT_HALF + 31) / 32;
    float* wq = qs + warp * 16 * HEAD_DIM;
    float freq[PAIRS];
#pragma unroll
    for (int i = 0; i < PAIRS; i++) freq[i] = rope_inv_freq(min(lane + 32 * i, ROT_HALF - 1));
    for (int r = 0; r < 16; r++) {
        int row = q0 + warp * 16 + r;
        const float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
        float* dst = wq + r * HEAD_DIM;
        float ss = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            float v = row < tokens ? src[d] : 0.0f;
            dst[d] = v;
            ss += v * v;
        }
        float inv = QK_NORM ? rsqrtf(warp_sum(ss) / HEAD_DIM + EPS) : 1.0f;
        __syncwarp();
        if (QK_NORM)
            for (int d = lane; d < HEAD_DIM; d += 32) dst[d] = q_norm[d] * (dst[d] * inv);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < PAIRS; i++) {
            int d = lane + 32 * i;
            if (d < ROT_HALF) {
                float x = dst[d], y = dst[d + ROT_HALF];
                float sn, c;
                sincosf((float)(base + row) * freq[i], &sn, &c);
                dst[d] = x * c - y * sn;
                dst[d + ROT_HALF] = y * c + x * sn;
            }
        }
        __syncwarp();
    }
    __syncwarp();
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 v[4];
        v[0] = *(const float2*)&wq[g * HEAD_DIM + d];
        v[1] = *(const float2*)&wq[(g + 8) * HEAD_DIM + d];
        v[2] = *(const float2*)&wq[g * HEAD_DIM + d + 8];
        v[3] = *(const float2*)&wq[(g + 8) * HEAD_DIM + d + 8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            qh[kb][i] = pack_hi(v[i].x, v[i].y);
            ql[kb][i] = pack_lo(v[i].x, v[i].y);
        }
    }
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    int last = min(q0 + FM_BQ, tokens) - 1;
    int key_end = base + last + 1;
    int warp_last = base + q0 + warp * 16 + 15;
    int tiles = (key_end + FM_BK - 1) / FM_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = kvs + buf * 4 * PLANE;
        constexpr int ROW_CHUNKS = 2 * CPR;
        constexpr int TOTAL = 2 * FM_BK * ROW_CHUNKS;
#pragma unroll
        for (int i = tid; i < TOTAL; i += 32 * FM_WARPS) {
            int which = i / (FM_BK * ROW_CHUNKS);
            int rem = i % (FM_BK * ROW_CHUNKS);
            int r = rem / ROW_CHUNKS;
            int c = rem % ROW_CHUNKS;
            int tk = k0 + r;
            const bf16_t* src = kv_row(which ? v_cache : k_cache, block_table, layer, kv_head, min(tk, key_end - 1));
            bf16_t* dst = st + (which * 2 + c / CPR) * PLANE + swz<CPR>(r, c % CPR) * 8;
            cp_async16_zfill(dst, src + c * 8, tk < key_end);
        }
    };
    __syncthreads();
    load_kv(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kt * FM_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) % FM_STAGES, k0 + FM_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* k_hi = kvs + (kt % FM_STAGES) * 4 * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_lo + PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
        if (k0 > warp_last) {
            __syncthreads();
            continue;
        }
        constexpr int SB = FM_BK / 8;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < FM_KB; kb++) {
                unsigned bh[4], bl[4];
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = swz<CPR>(kr, kb * 2 + ((lane >> 3) & 1)) * 8;
                ldsm_x4(bh, k_hi + at);
                ldsm_x4(bl, k_lo + at);
                mma_hi_lo(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                mma_hi_lo(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
            }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int pos = e < 2 ? pos0 : pos1;
                float v = (key <= pos && key < key_end) ? s[nb][e] * scale_log2 : NEG_INF;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG_INF ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG_INF ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG_INF ? 0.0f : fast_exp2(s[nb][e] - nm);
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
        for (int i = 0; i < FM_NB; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            pl[0] = pack_lo(s[2 * kb][0], s[2 * kb][1]);
            pl[1] = pack_lo(s[2 * kb][2], s[2 * kb][3]);
            pl[2] = pack_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            pl[3] = pack_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
#pragma unroll
            for (int np = 0; np < FM_NB / 2; np++) {
                unsigned bh[4], bl[4];
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = swz<CPR>(kr, np * 2 + (lane >> 4)) * 8;
                ldsm_x4_t(bh, v_hi + at);
                ldsm_x4_t(bl, v_lo + at);
                mma_hi_lo(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                mma_hi_lo(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
            }
        }
        __syncthreads();
    }
    float i0 = 1.0f / l0;
    float i1 = 1.0f / l1;
#pragma unroll
    for (int nb = 0; nb < FM_NB; nb++) {
        int d = nb * 8 + 2 * t;
        int at = head * HEAD_DIM + d;
        if (row0 < tokens)
            split_store2(out_hi, out_lo, (size_t)row0 * Q_DIM + at, o[nb][0] * i0 * out_gate(qkv, row0, at),
                         o[nb][1] * i0 * out_gate(qkv, row0, at + 1));
        if (row1 < tokens)
            split_store2(out_hi, out_lo, (size_t)row1 * Q_DIM + at, o[nb][2] * i1 * out_gate(qkv, row1, at),
                         o[nb][3] * i1 * out_gate(qkv, row1, at + 1));
    }
}

extern "C" __global__ void swiglu(const float* gate_up, float* out) {
    gate_up += (size_t)blockIdx.y * 2 * INTER;
    out += (size_t)blockIdx.y * INTER;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < INTER; i += gridDim.x * blockDim.x) {
        float g = gate_up[2 * i];
        float a = ACT_KIND == 1
            ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
            : g / (1.0f + expf(-g));
        out[i] = a * gate_up[2 * i + 1];
    }
}

__device__ float gumbel(unsigned seed, unsigned step, unsigned i) {
    unsigned x = seed * 747796405u + step * 2891336453u + i * 2246822519u + 1u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    float u = ((float)(x >> 8) + 0.5f) / 16777216.0f;
    return -logf(-logf(u));
}

extern "C" __global__ void argmax_advance(const float* logits, int* token, int* pos, int* history, int* count, float inv_temp, unsigned seed) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) v = v * inv_temp + gumbel(seed, (unsigned)count[0], (unsigned)i);
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = best_v[threadIdx.x + s];
            int oi = best_i[threadIdx.x + s];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        token[0] = best_i[0];
        history[count[0]] = best_i[0];
        count[0] += 1;
        pos[0] += 1;
    }
}

extern "C" __global__ void argmax_rows(const float* logits, int* tokens, int* history, const int* count, int stride) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    logits += (size_t)blockIdx.x * VOCAB;
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = best_v[threadIdx.x + s];
            int oi = best_i[threadIdx.x + s];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        tokens[blockIdx.x] = best_i[0];
        history[(size_t)blockIdx.x * stride + count[0]] = best_i[0];
    }
}

extern "C" __global__ void bump(int* pos, int* count) {
    pos[0] += 1;
    count[0] += 1;
}

extern "C" __global__ void copy_rows(const float* src, float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void split_rows(const float* src, bf16_t* hi, bf16_t* lo, int n) {
    for (int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += 2 * gridDim.x * blockDim.x)
        split_store2(hi, lo, i, src[i], src[i + 1]);
}
