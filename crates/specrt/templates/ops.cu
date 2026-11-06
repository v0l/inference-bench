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
constexpr int LINEAR_WARPS = 8;
constexpr int ATTN_THREADS = 512;
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float bf16_to_f32(bf16_t v) {
    return __uint_as_float(((unsigned)v) << 16);
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

template <bool IS_Q8, int K>
__device__ __forceinline__ float row_dot(const void* w, const float* scales, int row, const float* x, int lane) {
    float acc = 0.0f;
    if constexpr (IS_Q8) {
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

template <bool IS_Q8, int N, bool BIAS>
__device__ void linear_norm(const void* w, const float* scales, const float* bias, const float* x,
                            const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    float acc = row_dot<IS_Q8, HIDDEN>(w, scales, row, xs, lane);
    if (lane == 0) out[row] = BIAS ? acc + bias[row] : acc;
}

__device__ __forceinline__ float activation(float g) {
    return ACT_KIND == 1
        ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
        : g / (1.0f + expf(-g));
}

template <bool IS_Q8>
__device__ void gate_up_act_norm(const void* w, const float* scales, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= INTER) return;
    float g = row_dot<IS_Q8, HIDDEN>(w, scales, 2 * row, xs, lane);
    float u = row_dot<IS_Q8, HIDDEN>(w, scales, 2 * row + 1, xs, lane);
    if (lane == 0) out[row] = activation(g) * u;
}

constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 16;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;

template <int N, int K, bool BIAS, bool RESIDUAL, bool Q8>
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
                if (Q8) {
                    wv = (float)((const i8_t*)w)[(size_t)n * K + k0 + kk] *
                         scales[(size_t)n * (K / Q8_GROUP) + (k0 + kk) / Q8_GROUP];
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
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

__device__ __forceinline__ unsigned q8_pair_bf16(unsigned two_bytes) {
    int q0 = (signed char)(two_bytes & 0xff);
    int q1 = (signed char)((two_bytes >> 8) & 0xff);
    return (__float_as_uint((float)q0) >> 16) | (__float_as_uint((float)q1) & 0xffff0000u);
}

constexpr int EPI_STORE = 0;
constexpr int EPI_RESIDUAL = 1;
constexpr int EPI_ACT = 2;

template <int N, int K, int BM, int BN, int WM, int WN, int EPI, bool BIAS, bool Q8, bool NORM>
__device__ __forceinline__ void gemm_tc(const void* w, const float* scales, const float* bias, const float* x,
                                        const float* row_inv, const float* nw, float* out, int tokens) {
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int BK = 32;
    constexpr int AS = BK + 8;
    constexpr int WROW = Q8 ? BK + 16 : (BK + 8) * 2;
    constexpr int KT = K / BK;
    constexpr int A_ITERS = BM * BK / 4 / THREADS;
    constexpr int W_CHUNKS = Q8 ? 2 : 4;
    constexpr int W_ITERS = BN * W_CHUNKS / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of 32");
    static_assert(A_ITERS * THREADS * 4 == BM * BK, "A tile split");
    static_assert(W_ITERS * THREADS == BN * W_CHUNKS, "W tile split");
    static_assert(NJ % 2 == 0, "warp n tile");
    __shared__ __align__(16) bf16_t a_hi[BM * AS];
    __shared__ __align__(16) bf16_t a_lo[BM * AS];
    __shared__ __align__(16) unsigned char ws[2][BN * WROW];
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
    float4 xr[A_ITERS];
    auto load_x = [&](int kt) {
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 4);
            int c = (idx % (BK / 4)) * 4;
            int m = m0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m < tokens) {
                v = *(const float4*)(x + (size_t)m * K + kt * BK + c);
                if (NORM) {
                    float sc = row_inv[m];
                    float4 nv = *(const float4*)(nw + kt * BK + c);
                    v = make_float4(nv.x * (v.x * sc), nv.y * (v.y * sc), nv.z * (v.z * sc), nv.w * (v.w * sc));
                }
            }
            xr[i] = v;
        }
    };
    auto store_x = [&]() {
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 4);
            int c = (idx % (BK / 4)) * 4;
            unsigned h0, l0, h1, l1, h2, l2, h3, l3;
            split_hi_lo(xr[i].x, h0, l0);
            split_hi_lo(xr[i].y, h1, l1);
            split_hi_lo(xr[i].z, h2, l2);
            split_hi_lo(xr[i].w, h3, l3);
            *(uint2*)&a_hi[r * AS + c] = make_uint2(h0 | (h1 << 16), h2 | (h3 << 16));
            *(uint2*)&a_lo[r * AS + c] = make_uint2(l0 | (l1 << 16), l2 | (l3 << 16));
        }
    };
    auto load_w = [&](int buf, int kt) {
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            int r = idx / W_CHUNKS;
            int ck = idx % W_CHUNKS;
            int n = min(n0 + r, N - 1);
            const char* src = Q8 ? (const char*)w + (size_t)n * K + kt * BK + ck * 16
                                 : (const char*)w + ((size_t)n * K + kt * BK) * 2 + ck * 16;
            cp_async16(&ws[buf][r * WROW + ck * 16], src);
        }
    };
    load_w(0, 0);
    cp_async_commit();
    load_x(0);
    for (int kt = 0; kt < KT; kt++) {
        int cur = kt & 1;
        __syncthreads();
        store_x();
        if (kt + 1 < KT) {
            load_w(cur ^ 1, kt + 1);
            cp_async_commit();
            load_x(kt + 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        float part[MI][NJ][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) part[i][j][e] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned bw[NJ][2];
            if (Q8) {
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    const unsigned char* p = &ws[cur][(wn * TN + j * 8 + g) * WROW + ks * 16 + 2 * t];
                    bw[j][0] = q8_pair_bf16(*(const unsigned short*)p);
                    bw[j][1] = q8_pair_bf16(*(const unsigned short*)(p + 8));
                }
            } else {
#pragma unroll
                for (int jp = 0; jp < NJ / 2; jp++) {
                    unsigned r[4];
                    int row = wn * TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                    ldsm_x4(r, &ws[cur][row * WROW + (ks * 16 + ((lane >> 3) & 1) * 8) * 2]);
                    bw[2 * jp][0] = r[0];
                    bw[2 * jp][1] = r[1];
                    bw[2 * jp + 1][0] = r[2];
                    bw[2 * jp + 1][1] = r[3];
                }
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                unsigned ah[4], al[4];
                int row = wm * TM + i * 16 + (lane & 15);
                int col = ks * 16 + (lane >> 4) * 8;
                ldsm_x4(ah, &a_hi[row * AS + col]);
                ldsm_x4(al, &a_lo[row * AS + col]);
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    mma_bf16(part[i][j], al[0], al[1], al[2], al[3], bw[j][0], bw[j][1]);
                    mma_bf16(part[i][j], ah[0], ah[1], ah[2], ah[3], bw[j][0], bw[j][1]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            float s0 = 1.0f, s1 = 1.0f;
            if (Q8) {
                int col = n0 + wn * TN + j * 8 + 2 * t;
                s0 = scales[(size_t)min(col, N - 1) * (K / Q8_GROUP) + kt];
                s1 = scales[(size_t)min(col + 1, N - 1) * (K / Q8_GROUP) + kt];
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                acc[i][j][0] += part[i][j][0] * s0;
                acc[i][j][1] += part[i][j][1] * s1;
                acc[i][j][2] += part[i][j][2] * s0;
                acc[i][j][3] += part[i][j][3] * s1;
            }
        }
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
                if (EPI == EPI_ACT) {
                    out[(size_t)m * (N / 2) + col / 2] = activation(v0) * v1;
                } else {
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

extern "C" __global__ void row_rms(const float* x, float* inv) {
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * HIDDEN;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ss += x[i] * x[i];
    float r = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    if (threadIdx.x == 0) inv[blockIdx.x] = r;
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

extern "C" __global__ void qk_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        float* k_cache, float* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer) {
    __shared__ float scratch[32];
    __shared__ float vec[HEAD_DIM];
    int head = blockIdx.x;
    int d = threadIdx.x;
    int pos = pos_ptr[0] + blockIdx.y;
    bool is_q = head < HEADS;
    bool is_k = !is_q && head < HEADS + KV_HEADS;
    float* src = qkv + (size_t)blockIdx.y * (Q_DIM + 2 * KV_DIM) + head * HEAD_DIM;
    float v = d < HEAD_DIM ? src[d] : 0.0f;
    if (QK_NORM && (is_q || is_k)) {
        float inv = rsqrtf(block_reduce<false>(v * v, scratch) / HEAD_DIM + EPS);
        const float* w = is_q ? q_norm : k_norm;
        if (d < HEAD_DIM) v = w[d] * (v * inv);
    }
    if (d < HEAD_DIM) vec[d] = v;
    __syncthreads();
    if ((is_q || is_k) && d < HEAD_DIM) {
        int half = HEAD_DIM / 2;
        int i = d < half ? d : d - half;
        float inv_freq = 1.0f / powf(THETA, (float)(2 * i) / (float)HEAD_DIM);
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
        float angle = (float)pos * inv_freq;
        float c = cosf(angle);
        float s = sinf(angle);
        v = d < half ? vec[d] * c - vec[d + half] * s : vec[d] * c + vec[d - half] * s;
    }
    if (d >= HEAD_DIM) return;
    if (is_q) {
        src[d] = v;
        return;
    }
    int kv_head = is_k ? head - HEADS : head - HEADS - KV_HEADS;
    int page = block_table[pos / PAGE];
    size_t slot = (((size_t)layer * PAGES + page) * KV_HEADS + kv_head) * PAGE + pos % PAGE;
    (is_k ? k_cache : v_cache)[slot * HEAD_DIM + d] = v;
}

extern "C" __global__ void attention(const float* qkv, const float* k_cache, const float* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* out, float* scores_global) {
    __shared__ float scratch[32];
    __shared__ float scores_shared[MAX_CTX > 4096 ? 1 : MAX_CTX];
    int head = blockIdx.x;
    float* scores = MAX_CTX > 4096 ? scores_global + ((size_t)blockIdx.y * HEADS + head) * MAX_CTX : scores_shared;
    int kv_head = head / GROUP;
    int len = pos_ptr[0] + blockIdx.y + 1;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const float* q = qkv + (size_t)blockIdx.y * (Q_DIM + 2 * KV_DIM) + head * HEAD_DIM;
    out += (size_t)blockIdx.y * Q_DIM;
    float scale = rsqrtf((float)HEAD_DIM);
    for (int t = warp; t < len; t += ATTN_THREADS / 32) {
        size_t slot = (((size_t)layer * PAGES + block_table[t / PAGE]) * KV_HEADS + kv_head) * PAGE + t % PAGE;
        const float* k = k_cache + slot * HEAD_DIM;
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) dot += q[d] * k[d];
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
        size_t slot = (((size_t)layer * PAGES + block_table[t / PAGE]) * KV_HEADS + kv_head) * PAGE + t % PAGE;
        const float* v = v_cache + slot * HEAD_DIM;
        float p = scores[t];
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            if (d < HEAD_DIM) acc[i] += p * v[d];
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
        out[head * HEAD_DIM + d] = total / sum;
    }
}

constexpr int FA_BQ = 16;
constexpr int FA_BK = 32;
constexpr int FA_THREADS = 128;
constexpr int FA_STRIDE = HEAD_DIM + 1;
constexpr int FA_OUT = HEAD_DIM / 8;

extern "C" __global__ void flash_attention(const float* qkv, const float* k_cache, const float* v_cache,
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
        qs[r][d] = q < tokens ? qkv[(size_t)q * (Q_DIM + 2 * KV_DIM) + head * HEAD_DIM + d] : 0.0f;
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
                size_t slot = (((size_t)layer * PAGES + block_table[t / PAGE]) * KV_HEADS + kv_head) * PAGE + t % PAGE;
                kv = k_cache[slot * HEAD_DIM + d];
                vv = v_cache[slot * HEAD_DIM + d];
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
        for (int i = 0; i < FA_OUT; i++) dst[sub + 8 * i] = o[i] * inv;
    }
}

constexpr int FM_BQ = 64;
constexpr int FM_BK = 32;
constexpr int FM_KS = HEAD_DIM + 8;
constexpr int FM_VS = FM_BK + 8;
constexpr int FM_KB = HEAD_DIM / 16;
constexpr int FM_NB = HEAD_DIM / 8;

__device__ __forceinline__ unsigned pack_hi(float a, float b) {
    return bf16_bits_rn(a) | (bf16_bits_rn(b) << 16);
}

__device__ __forceinline__ unsigned pack_lo(float a, float b) {
    float ha = __uint_as_float(bf16_bits_rn(a) << 16);
    float hb = __uint_as_float(bf16_bits_rn(b) << 16);
    return bf16_bits_rn(a - ha) | (bf16_bits_rn(b - hb) << 16);
}

__device__ __forceinline__ void mma3(float* c, const unsigned* ah, const unsigned* al,
                                     unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma_bf16(c, al[0], al[1], al[2], al[3], bl0, bl1);
    mma_bf16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

extern "C" __global__ void flash_mma(const float* qkv, const float* k_cache, const float* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* out, int tokens) {
    __shared__ __align__(16) bf16_t k_hi[FM_BK][FM_KS];
    __shared__ __align__(16) bf16_t k_lo[FM_BK][FM_KS];
    __shared__ __align__(16) bf16_t v_hi[HEAD_DIM][FM_VS];
    __shared__ __align__(16) bf16_t v_lo[HEAD_DIM][FM_VS];
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = blockIdx.x * FM_BQ;
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
    float scale = rsqrtf((float)HEAD_DIM);
    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
    constexpr int QKV_ROW = Q_DIM + 2 * KV_DIM;
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 v[4];
        const float* p0 = qkv + (size_t)row0 * QKV_ROW + head * HEAD_DIM + d;
        const float* p1 = qkv + (size_t)row1 * QKV_ROW + head * HEAD_DIM + d;
        v[0] = row0 < tokens ? *(const float2*)p0 : make_float2(0.f, 0.f);
        v[1] = row1 < tokens ? *(const float2*)p1 : make_float2(0.f, 0.f);
        v[2] = row0 < tokens ? *(const float2*)(p0 + 8) : make_float2(0.f, 0.f);
        v[3] = row1 < tokens ? *(const float2*)(p1 + 8) : make_float2(0.f, 0.f);
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
    for (int k0 = 0; k0 < key_end; k0 += FM_BK) {
        __syncthreads();
        for (int i = tid; i < FM_BK * HEAD_DIM / 4; i += 128) {
            int r = i / (HEAD_DIM / 4);
            int d = (i % (HEAD_DIM / 4)) * 4;
            int tk = k0 + r;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv = kv;
            if (tk < key_end) {
                size_t slot = (((size_t)layer * PAGES + block_table[tk / PAGE]) * KV_HEADS + kv_head) * PAGE + tk % PAGE;
                kv = *(const float4*)(k_cache + slot * HEAD_DIM + d);
                vv = *(const float4*)(v_cache + slot * HEAD_DIM + d);
            }
            *(uint2*)&k_hi[r][d] = make_uint2(pack_hi(kv.x, kv.y), pack_hi(kv.z, kv.w));
            *(uint2*)&k_lo[r][d] = make_uint2(pack_lo(kv.x, kv.y), pack_lo(kv.z, kv.w));
            float vf[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                unsigned hb = bf16_bits_rn(vf[j]);
                v_hi[d + j][r] = (bf16_t)hb;
                v_lo[d + j][r] = (bf16_t)bf16_bits_rn(vf[j] - __uint_as_float(hb << 16));
            }
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int kb = 0; kb < FM_KB; kb++) {
                int kr = nb * 8 + g;
                int d = kb * 16 + 2 * t;
                mma3(s[nb], qh[kb], ql[kb],
                     *(const unsigned*)&k_hi[kr][d], *(const unsigned*)&k_hi[kr][d + 8],
                     *(const unsigned*)&k_lo[kr][d], *(const unsigned*)&k_lo[kr][d + 8]);
            }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int pos = e < 2 ? pos0 : pos1;
                float v = (key <= pos && key < key_end) ? s[nb][e] * scale : NEG_INF;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG_INF ? 0.0f : expf(m0 - n0);
        float a1 = m1 == NEG_INF ? 0.0f : expf(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG_INF ? 0.0f : expf(s[nb][e] - nm);
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
        for (int kb = 0; kb < 2; kb++) {
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
            for (int nb = 0; nb < FM_NB; nb++) {
                int dr = nb * 8 + g;
                int kc = kb * 16 + 2 * t;
                mma3(o[nb], ph, pl,
                     *(const unsigned*)&v_hi[dr][kc], *(const unsigned*)&v_hi[dr][kc + 8],
                     *(const unsigned*)&v_lo[dr][kc], *(const unsigned*)&v_lo[dr][kc + 8]);
            }
        }
    }
    float i0 = 1.0f / l0;
    float i1 = 1.0f / l1;
#pragma unroll
    for (int nb = 0; nb < FM_NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (row0 < tokens)
            *(float2*)(out + (size_t)row0 * Q_DIM + head * HEAD_DIM + d) = make_float2(o[nb][0] * i0, o[nb][1] * i0);
        if (row1 < tokens)
            *(float2*)(out + (size_t)row1 * Q_DIM + head * HEAD_DIM + d) = make_float2(o[nb][2] * i1, o[nb][3] * i1);
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
