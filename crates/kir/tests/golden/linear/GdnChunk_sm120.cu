// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// linear GdnChunk threads=256 smem=50176 chunk=64 ksplit=4 cols=32
//   LinearShape { key_heads: 16, value_heads: 48, head_dim: 128, conv_kernel: 4, tiled_value_heads: false }
constexpr int LA_KH = 16;
constexpr int LA_VH = 48;
constexpr int LA_D = 128;
constexpr int LA_K = 4;
constexpr int LA_KEY_DIM = LA_KH * LA_D;
constexpr int LA_VALUE_DIM = LA_VH * LA_D;
constexpr int LA_CONV_DIM = 2 * LA_KEY_DIM + LA_VALUE_DIM;
constexpr int LA_IN_ROWS = LA_CONV_DIM + LA_VALUE_DIM;
constexpr int LA_BA_ROWS = 2 * LA_VH;
constexpr int LA_RATIO = LA_VH / LA_KH;
constexpr int LA_V_OFF = 2 * LA_KEY_DIM;
constexpr int LA_Z_OFF = LA_CONV_DIM;
constexpr int LA_B_OFF = 0;
constexpr int LA_A_OFF = LA_VH;
constexpr int LA_STATE = LA_D * LA_D;
constexpr bool LA_TILED = false;
__device__ __forceinline__ int la_kh(int h) {
    return LA_TILED ? h % LA_KH : h / LA_RATIO;
}
constexpr int LA_CHUNK = 64;
constexpr int LA_KSPLIT = 4;
constexpr int LA_KS = LA_D + 1;
constexpr int LA_AS = LA_CHUNK + 1;
constexpr int LA_COLS = 32;
constexpr bool LA_LO = true;
constexpr int LA_FS = 2;
constexpr float LA_L2_EPS = 1e-6f;
static_assert(LA_D % LA_COLS == 0 && LA_D <= 128, "linear head dim");
static_assert(LA_VH % LA_KH == 0, "linear head groups");

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ void gdn_gate(const float* ba_row, const float* neg_a, const float* dt_bias, int h,
                                         float& g, float& beta) {
    g = neg_a[h] * softplus(ba_row[LA_A_OFF + h] + dt_bias[h]);
    beta = 1.0f / (1.0f + expf(-ba_row[LA_B_OFF + h]));
}

__device__ __forceinline__ float conv_tap(const float* w, const float* window) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < LA_K; j++) acc += w[j] * window[j];
    return silu(acc);
}

__device__ __forceinline__ float l2_q(float v, float ss) {
    return v * rsqrtf(ss + LA_L2_EPS) * rsqrtf((float)LA_D);
}

__device__ __forceinline__ float l2_k(float v, float ss) {
    return v * rsqrtf(ss + LA_L2_EPS);
}

__device__ __forceinline__ float* conv_state_row(float* state, int slot, int c) {
    return state + ((size_t)slot * LA_CONV_DIM + c) * (LA_K - 1);
}

template <int KSTEPS, bool LO = true, typename FA, typename FB>
__device__ __forceinline__ void mma_split_tile(float* c, int m0, int n0, int lane, FA a_at, FB b_at) {
    int g = lane >> 2;
    int t = lane & 3;
#pragma unroll 2
    for (int kk = 0; kk < KSTEPS; kk++) {
        int k = kk * 16 + 2 * t;
        float av[8] = {a_at(m0 + g, k),     a_at(m0 + g, k + 1),     a_at(m0 + g + 8, k),     a_at(m0 + g + 8, k + 1),
                       a_at(m0 + g, k + 8), a_at(m0 + g, k + 9), a_at(m0 + g + 8, k + 8), a_at(m0 + g + 8, k + 9)};
        float bv[4] = {b_at(k, n0 + g), b_at(k + 1, n0 + g), b_at(k + 8, n0 + g), b_at(k + 9, n0 + g)};
        unsigned ah[4], al[4], bh[2], bl[2];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned h0, l0, h1, l1;
            split_hi_lo(av[2 * i], h0, l0);
            split_hi_lo(av[2 * i + 1], h1, l1);
            ah[i] = h0 | (h1 << 16);
            al[i] = l0 | (l1 << 16);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned h0, l0, h1, l1;
            split_hi_lo(bv[2 * i], h0, l0);
            split_hi_lo(bv[2 * i + 1], h1, l1);
            bh[i] = h0 | (h1 << 16);
            bl[i] = l0 | (l1 << 16);
        }
        if (LO) {
            mma_bf16(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
            mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
        }
        mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
    }
}

__device__ __forceinline__ void gdn_pack2(float a, float b, unsigned& hi, unsigned& lo) {
    unsigned h0, l0, h1, l1;
    split_hi_lo(a, h0, l0);
    split_hi_lo(b, h1, l1);
    hi = h0 | (h1 << 16);
    lo = l0 | (l1 << 16);
}

template <int KSTEPS, bool LO = true>
__device__ __forceinline__ void gdn_frag_put(unsigned* base, int m, int k, float v0, float v1) {
    unsigned hi, lo;
    gdn_pack2(v0, v1, hi, lo);
    int rec = ((m >> 4) * KSTEPS + (k >> 4)) * 32 + (m & 7) * 4 + ((k & 7) >> 1);
    int reg = ((m >> 3) & 1) + 2 * ((k >> 3) & 1);
    base[rec * (LO ? 8 : 4) + reg] = hi;
    if (LO) base[rec * 8 + 4 + reg] = lo;
}

extern "C" __global__ void __launch_bounds__(256, 2) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
                                                                 const float* g, const float* beta, float* w,
                                                                 float* u, float* aqk, float* gc, float* qe, float* kt, int tokens) {
    constexpr int C = LA_CHUNK;
    extern __shared__ __align__(16) unsigned char la_smem[];
    float* ks = (float*)la_smem;
    float* A = ks + C * LA_KS;
    float* gs = A + C * LA_AS;
    float* bs = gs + C;
    int h = blockIdx.x;
    int kh = la_kh(h);
    int c = blockIdx.y;
    int t0 = c * C;
    int tid = threadIdx.x;
    if (tid < C) {
        int t = t0 + tid;
        gs[tid] = t < tokens ? g[(size_t)t * LA_VH + h] : 0.0f;
        bs[tid] = t < tokens ? beta[(size_t)t * LA_VH + h] : 0.0f;
    }
    for (int i = tid; i < C * LA_D; i += blockDim.x) {
        int r = i / LA_D;
        int d = i % LA_D;
        int t = t0 + r;
        ks[r * LA_KS + d] = t < tokens ? kn[((size_t)t * LA_KH + kh) * LA_D + d] : 0.0f;
    }
    __syncthreads();
    if (tid == 0)
        for (int i = 1; i < C; i++) gs[i] += gs[i - 1];
    __syncthreads();
    constexpr int WARPS = 8;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g8 = lane >> 2;
    int t4 = lane & 3;
    auto k_row = [&](int m, int k) { return ks[m * LA_KS + k]; };
    auto k_col = [&](int k, int n) { return ks[n * LA_KS + k]; };
    for (int idx = warp; idx < (C / 16) * (C / 16 + 1); idx += WARPS) {
        int mb = 0;
        while ((mb + 1) * (mb + 2) <= idx) mb++;
        int m0 = mb * 16;
        int n0 = (idx - mb * (mb + 1)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_split_tile<LA_D / 16, LA_LO>(acc, m0, n0, lane, k_row, k_col);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4 + (e & 1);
            A[i * LA_AS + j] = j < i ? -(bs[i] * acc[e]) * expf(gs[i] - gs[j]) : 0.0f;
        }
    }
    for (int idx = warp; idx < (C / 16) * (C / 8) - (C / 16) * (C / 16 + 1); idx += WARPS) {
        int mb = 0;
        while ((mb + 1) * (C / 8) - (mb + 1) * (mb + 2) <= idx) mb++;
        int m0 = mb * 16;
        int n0 = (2 * mb + 2 + idx - (mb * (C / 8) - mb * (mb + 1))) * 8;
#pragma unroll
        for (int e = 0; e < 4; e++) A[(m0 + g8 + (e >> 1) * 8) * LA_AS + n0 + 2 * t4 + (e & 1)] = 0.0f;
    }
    const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
    float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
    int live = min(C, tokens - t0);
    for (int idx = warp; idx < (C / 16) * (C / 16 + 1); idx += WARPS) {
        int mb = 0;
        while ((mb + 1) * (mb + 2) <= idx) mb++;
        int m0 = mb * 16;
        int n0 = (idx - mb * (mb + 1)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_split_tile<LA_D / 16, LA_LO>(
            acc, m0, n0, lane, [&](int m, int k) { return m < live ? qc[(size_t)m * LA_KH * LA_D + k] : 0.0f; }, k_col);
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4;
            float v0 = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
            float v1 = j + 1 <= i && i < live ? acc[e + 1] * expf(gs[i] - gs[j + 1]) : 0.0f;
            gdn_frag_put<C / 16, LA_LO>((unsigned*)ac, i, j, v0, v1);
        }
    }
    for (int idx = warp; idx < (C / 16) * (C / 8) - (C / 16) * (C / 16 + 1); idx += WARPS) {
        int mb = 0;
        while ((mb + 1) * (C / 8) - (mb + 1) * (mb + 2) <= idx) mb++;
        int m0 = mb * 16;
        int n0 = (2 * mb + 2 + idx - (mb * (C / 8) - mb * (mb + 1))) * 8;
#pragma unroll
        for (int e = 0; e < 4; e += 2) gdn_frag_put<C / 16, LA_LO>((unsigned*)ac, m0 + g8 + (e >> 1) * 8, n0 + 2 * t4, 0.0f, 0.0f);
    }
    __syncthreads();
    constexpr int LEAF = 16;
    for (int i = 1; i < LEAF; i++) {
        int b = (tid / LEAF) * LEAF;
        int j = tid % LEAF;
        float next = 0.0f;
        if (tid < C && j < i) {
            next = A[(b + i) * LA_AS + b + j];
            for (int m = j + 1; m < i; m++) next += A[(b + i) * LA_AS + b + m] * A[(b + m) * LA_AS + b + j];
        }
        __syncthreads();
        if (tid < C && j < i) A[(b + i) * LA_AS + b + j] = next;
        __syncthreads();
    }
    for (int sz = LEAF; sz < C; sz *= 2) {
        constexpr int PER = C * C / 4 / 256;
        int per = sz * sz * (C / (2 * sz));
        float acc[PER];
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            acc[p] = 0.0f;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                float y = A[(base + sz + r) * LA_AS + base + c2];
                for (int m = c2 + 1; m < sz; m++)
                    y += A[(base + sz + r) * LA_AS + base + m] * A[(base + m) * LA_AS + base + c2];
                acc[p] = y;
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                A[(base + sz + r) * LA_AS + base + c2] = acc[p];
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                float x = A[(base + sz + r) * LA_AS + base + c2];
                for (int m = 0; m < r; m++)
                    x += A[(base + sz + r) * LA_AS + base + sz + m] * A[(base + sz + m) * LA_AS + base + c2];
                acc[p] = x;
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < PER; p++) {
            int e = tid + p * 256;
            if (e < per) {
                int blk = e / (sz * sz);
                int r = (e / sz) % sz;
                int c2 = e % sz;
                int base = blk * 2 * sz;
                A[(base + sz + r) * LA_AS + base + c2] = acc[p];
            }
        }
        __syncthreads();
    }
    if (tid < C) A[tid * LA_AS + tid] = 1.0f;
    __syncthreads();
    const float* vc = mix + (size_t)t0 * LA_CONV_DIM + LA_V_OFF + h * LA_D;
    auto t_at = [&](int m, int k) { return k <= m ? A[m * LA_AS + k] : 0.0f; };
    {
        constexpr int MB = C / 16;
        unsigned* wp = (unsigned*)w + (size_t)(c * LA_VH + h) * C * LA_D;
        for (int nt = warp; nt < LA_D / 8; nt += WARPS) {
            int n = nt * 8 + g8;
            float uu[MB][4], ww[MB][4];
#pragma unroll
            for (int mb = 0; mb < MB; mb++)
#pragma unroll
                for (int e = 0; e < 4; e++) uu[mb][e] = ww[mb][e] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < MB; kk++) {
                int k = kk * 16 + 2 * t4;
                auto vb = [&](int r) { return r < live ? vc[(size_t)r * LA_CONV_DIM + n] * bs[r] : 0.0f; };
                auto kb = [&](int r) { return ks[r * LA_KS + n] * bs[r] * expf(gs[r]); };
                unsigned uh0, ul0, uh1, ul1, wh0, wl0, wh1, wl1;
                gdn_pack2(vb(k), vb(k + 1), uh0, ul0);
                gdn_pack2(vb(k + 8), vb(k + 9), uh1, ul1);
                gdn_pack2(kb(k), kb(k + 1), wh0, wl0);
                gdn_pack2(kb(k + 8), kb(k + 9), wh1, wl1);
#pragma unroll
                for (int mb = kk; mb < MB; mb++) {
                    int m0 = mb * 16;
                    unsigned th[4], tl[4];
                    gdn_pack2(t_at(m0 + g8, k), t_at(m0 + g8, k + 1), th[0], tl[0]);
                    gdn_pack2(t_at(m0 + g8 + 8, k), t_at(m0 + g8 + 8, k + 1), th[1], tl[1]);
                    gdn_pack2(t_at(m0 + g8, k + 8), t_at(m0 + g8, k + 9), th[2], tl[2]);
                    gdn_pack2(t_at(m0 + g8 + 8, k + 8), t_at(m0 + g8 + 8, k + 9), th[3], tl[3]);
                    if (LA_LO) {
                        mma_bf16(uu[mb], tl[0], tl[1], tl[2], tl[3], uh0, uh1);
                        mma_bf16(uu[mb], th[0], th[1], th[2], th[3], ul0, ul1);
                    }
                    mma_bf16(uu[mb], th[0], th[1], th[2], th[3], uh0, uh1);
                    if (LA_LO) {
                        mma_bf16(ww[mb], tl[0], tl[1], tl[2], tl[3], wh0, wh1);
                        mma_bf16(ww[mb], th[0], th[1], th[2], th[3], wl0, wl1);
                    }
                    mma_bf16(ww[mb], th[0], th[1], th[2], th[3], wh0, wh1);
                }
            }
#pragma unroll
            for (int mb = 0; mb < MB; mb++)
#pragma unroll
                for (int e = 0; e < 4; e += 2) {
                    int i = mb * 16 + g8 + (e >> 1) * 8;
                    int d = nt * 8 + 2 * t4;
                    size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
                    u[at] = uu[mb][e];
                    u[at + 1] = uu[mb][e + 1];
                    gdn_frag_put<LA_D / 16, LA_LO>(wp, i, d, ww[mb][e], ww[mb][e + 1]);
                }
        }
    }
    if (tid < C) gc[(size_t)(c * LA_VH + h) * C + tid] = gs[tid];
    float glast = gs[C - 1];
    unsigned* qp = (unsigned*)qe + (size_t)(c * LA_VH + h) * C * LA_D;
    unsigned* kp = (unsigned*)kt + (size_t)(c * LA_VH + h) * C * LA_D;
    for (int p = tid; p < C * LA_D / 2; p += blockDim.x) {
        int m = p / (LA_D / 2);
        int k = (p % (LA_D / 2)) * 2;
        float2 q = t0 + m < tokens ? *(const float2*)(qn + ((size_t)(t0 + m) * LA_KH + kh) * LA_D + k) : make_float2(0.0f, 0.0f);
        float eg = t0 + m < tokens ? expf(gs[m]) : 0.0f;
        gdn_frag_put<LA_D / 16, LA_LO>(qp, m, k, q.x * eg, q.y * eg);
    }
    for (int p = tid; p < C * LA_D / 2; p += blockDim.x) {
        int d = p % LA_D;
        int k = (p / LA_D) * 2;
        float v0 = ks[k * LA_KS + d] * expf(glast - gs[k]);
        float v1 = ks[(k + 1) * LA_KS + d] * expf(glast - gs[k + 1]);
        if (t0 + k >= tokens) v0 = 0.0f;
        if (t0 + k + 1 >= tokens) v1 = 0.0f;
        gdn_frag_put<C / 16, LA_LO>(kp, d, k, v0, v1);
    }
}


template <int KS, int NT, bool LO = true>
__device__ __forceinline__ void gdn_mma_planes(float (*acc)[4], const uint4* ap, const unsigned short* bh,
                                               const unsigned short* bl, int pitch, int g, int t) {
#pragma unroll
    for (int kk = 0; kk < KS; kk++) {
        uint4 ah = ap[kk * (LO ? 64 : 32)];
        uint4 al = LO ? ap[kk * 64 + 1] : ah;
#pragma unroll
        for (int n = 0; n < NT; n++) {
            const unsigned* rh = (const unsigned*)(bh + (n * 8 + g) * pitch);
            unsigned bh0 = rh[kk * 8 + t], bh1 = rh[kk * 8 + t + 4];
            if (LO) {
                const unsigned* rl = (const unsigned*)(bl + (n * 8 + g) * pitch);
                unsigned bl0 = rl[kk * 8 + t], bl1 = rl[kk * 8 + t + 4];
                mma_bf16(acc[n], al.x, al.y, al.z, al.w, bh0, bh1);
                mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bl0, bl1);
            }
            mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bh0, bh1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(LA_CHUNK * 2) gdn_chunk_scan(const float* w, const float* qe, const float* kt,
                                                                           const float* u, const float* aqk, const float* gc,
                                                                           float* state, float* o, int slot, int tokens) {
    constexpr int C = LA_CHUNK;
    constexpr int WARPS = C / 16;
    constexpr int NT = LA_COLS / 8;
    constexpr int KD = LA_D / 16;
    constexpr int KC = C / 16;
    constexpr int DPW = (KD + WARPS - 1) / WARPS;
    constexpr int SP = LA_D + 8;
    constexpr int VP = C + 8;
    __shared__ unsigned short sh[LA_COLS * SP];
    __shared__ unsigned short sl[LA_COLS * SP];
    __shared__ unsigned short vh[LA_COLS * VP];
    __shared__ unsigned short vl[LA_COLS * VP];
    int h = blockIdx.x;
    int c0 = blockIdx.y * LA_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t4 = lane & 3;
    float* st = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    float sr[DPW][NT][4];
#pragma unroll
    for (int j = 0; j < DPW; j++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int d = (warp + j * WARPS) * 16 + g + (e >> 1) * 8;
                if (warp + j * WARPS >= KD) continue;
                sr[j][n][e] = st[(size_t)d * LA_D + c0 + n * 8 + 2 * t4 + (e & 1)];
            }
    auto store_state_planes = [&]() {
#pragma unroll
        for (int j = 0; j < DPW; j++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int d = (warp + j * WARPS) * 16 + g + (e >> 1) * 8;
                    if (warp + j * WARPS >= KD) continue;
                    int col = n * 8 + 2 * t4 + (e & 1);
                    unsigned hi, lo;
                    split_hi_lo(sr[j][n][e], hi, lo);
                    sh[col * SP + d] = (unsigned short)hi;
                    if (LA_LO) sl[col * SP + d] = (unsigned short)lo;
                }
    };
    store_state_planes();
    __syncthreads();
    int chunks = (tokens + C - 1) / C;
    for (int c = 0; c < chunks; c++) {
        int t0 = c * C;
        size_t ch = (size_t)(c * LA_VH + h);
        const uint4* wp = (const uint4*)w + ch * (C * LA_D / 4) + LA_FS * (warp * KD * 32 + lane);
        const uint4* qp = (const uint4*)qe + ch * (C * LA_D / 4) + LA_FS * (warp * KD * 32 + lane);
        const uint4* ap = (const uint4*)aqk + ch * (C * C / 4) + LA_FS * (warp * KC * 32 + lane);
        const uint4* kp = (const uint4*)kt;
        float decay = expf(gc[ch * C + C - 1]);
        float acc[NT][4];
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[n][e] = 0.0f;
        gdn_mma_planes<KD, NT, LA_LO>(acc, wp, sh, sl, SP, g, t4);
        const float* uc = u + ((size_t)t0 * LA_VH + h) * LA_D + c0;
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int m = warp * 16 + g + (e >> 1) * 8;
                int col = n * 8 + 2 * t4 + (e & 1);
                float v = uc[(size_t)m * LA_VH * LA_D + col] - acc[n][e];
                unsigned hi, lo;
                split_hi_lo(v, hi, lo);
                vh[col * VP + m] = (unsigned short)hi;
                if (LA_LO) vl[col * VP + m] = (unsigned short)lo;
            }
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[n][e] = 0.0f;
        gdn_mma_planes<KD, NT, LA_LO>(acc, qp, sh, sl, SP, g, t4);
        __syncthreads();
        gdn_mma_planes<KC, NT, LA_LO>(acc, ap, vh, vl, VP, g, t4);
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int m = warp * 16 + g + (e >> 1) * 8;
                int col = n * 8 + 2 * t4 + (e & 1);
                if (t0 + m < tokens) o[((size_t)(t0 + m) * LA_VH + h) * LA_D + c0 + col] = acc[n][e];
            }
#pragma unroll
        for (int j = 0; j < DPW; j++) {
            if (warp + j * WARPS >= KD) continue;
            const uint4* kj = kp + ch * (C * LA_D / 4) + LA_FS * ((warp + j * WARPS) * KC * 32 + lane);
            float a2[NT][4];
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) a2[n][e] = sr[j][n][e] * decay;
            gdn_mma_planes<KC, NT, LA_LO>(a2, kj, vh, vl, VP, g, t4);
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) sr[j][n][e] = a2[n][e];
        }
        __syncthreads();
        store_state_planes();
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < DPW; j++)
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int d = (warp + j * WARPS) * 16 + g + (e >> 1) * 8;
                if (warp + j * WARPS >= KD) continue;
                st[(size_t)d * LA_D + c0 + n * 8 + 2 * t4 + (e & 1)] = sr[j][n][e];
            }
}


extern "C" __global__ void gdn_gated_norm(const float* o, const float* lin, const float* norm_w, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = 2 * threadIdx.x;
    const float* src = o + ((size_t)t * LA_VH + h) * LA_D;
    float a = src[d];
    float b = src[d + 1];
    float inv = rsqrtf(block_reduce<false>(a * a + b * b, scratch) / LA_D + EPS);
    const float* z = lin + (size_t)t * LA_IN_ROWS + LA_Z_OFF + h * LA_D;
    split_store2(hi, lo, (size_t)t * LA_VALUE_DIM + h * LA_D + d, norm_w[d] * (a * inv) * silu(z[d]),
                 norm_w[d + 1] * (b * inv) * silu(z[d + 1]));
}
