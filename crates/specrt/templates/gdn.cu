constexpr int LA_KH = {{LA_KH}};
constexpr int LA_VH = {{LA_VH}};
constexpr int LA_D = {{LA_D}};
constexpr int LA_K = {{LA_CONV_K}};
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
constexpr int LA_CHUNK = {{LA_CHUNK}};
constexpr int LA_KSPLIT = {{LA_KSPLIT}};
constexpr int LA_KS = LA_D + 1;
constexpr int LA_AS = LA_CHUNK + 1;
constexpr int LA_COLS = {{LA_COLS}};
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

extern "C" __global__ void gdn_conv_step(const float* lin, const float* w, float* state, float* mix, int slot, int seq_slots,
                                         const int* slots) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (c >= LA_CONV_DIM) return;
    lin += (size_t)seq * LA_IN_ROWS;
    mix += (size_t)seq * LA_CONV_DIM;
    float* s = conv_state_row(state, slot + slots[seq] * seq_slots, c);
    float window[LA_K];
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) window[j] = s[j];
    window[LA_K - 1] = lin[c];
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) s[j] = window[j + 1];
    mix[c] = conv_tap(w + (size_t)c * LA_K, window);
}

extern "C" __global__ void gdn_conv_prefill(const float* lin, const float* w, const float* state, float* mix, int slot) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= LA_CONV_DIM) return;
    const float* s = conv_state_row((float*)state, slot, c);
    float window[LA_K];
#pragma unroll
    for (int j = 0; j < LA_K; j++) {
        int src = t - (LA_K - 1) + j;
        window[j] = src >= 0 ? lin[(size_t)src * LA_IN_ROWS + c] : s[LA_K - 1 + src];
    }
    mix[(size_t)t * LA_CONV_DIM + c] = conv_tap(w + (size_t)c * LA_K, window);
}

extern "C" __global__ void gdn_conv_keep(const float* lin, float* state, int slot, int tokens) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= LA_CONV_DIM) return;
    float* s = conv_state_row(state, slot, c);
    float keep[LA_K - 1];
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) {
        int src = tokens - (LA_K - 1) + j;
        keep[j] = src >= 0 ? lin[(size_t)src * LA_IN_ROWS + c] : s[LA_K - 1 + src];
    }
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) s[j] = keep[j];
}

extern "C" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_step(const float* mix, const float* lin,
                                                                        const float* ba,
                                                                        const float* neg_a, const float* dt_bias,
                                                                        const float* norm_w, float* state,
                                                                        float* out, int slot, int seq_slots,
                                                                        const int* slots) {
    constexpr int KS = LA_D / LA_KSPLIT;
    __shared__ float scratch[32];
    __shared__ float q_s[LA_D];
    __shared__ float k_s[LA_D];
    __shared__ float part[LA_KSPLIT][LA_D];
    __shared__ float delta_s[LA_D];
    int h = blockIdx.x;
    int seq = blockIdx.y;
    mix += (size_t)seq * LA_CONV_DIM;
    lin += (size_t)seq * LA_IN_ROWS;
    ba += (size_t)seq * LA_BA_ROWS;
    out += (size_t)seq * LA_VALUE_DIM;
    slot += slots[seq] * seq_slots;
    int kh = h / LA_RATIO;
    int tid = threadIdx.x;
    int v = tid % LA_D;
    int ks = tid / LA_D;
    float qv = tid < LA_D ? mix[kh * LA_D + tid] : 0.0f;
    float kv = tid < LA_D ? mix[LA_KEY_DIM + kh * LA_D + tid] : 0.0f;
    float qss = block_reduce<false>(qv * qv, scratch);
    float kss = block_reduce<false>(kv * kv, scratch);
    if (tid < LA_D) {
        q_s[tid] = l2_q(qv, qss);
        k_s[tid] = l2_k(kv, kss);
    }
    float g, beta;
    gdn_gate(ba, neg_a, dt_bias, h, g, beta);
    float decay = expf(g);
    __syncthreads();
    float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    float s[KS];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < KS; i++) {
        int k = ks * KS + i;
        s[i] = __fmul_rn(S[k * LA_D + v], decay);
        acc = __fmaf_rn(s[i], k_s[k], acc);
    }
    part[ks][v] = acc;
    __syncthreads();
    if (ks == 0) {
        float mem = 0.0f;
#pragma unroll
        for (int j = 0; j < LA_KSPLIT; j++) mem += part[j][v];
        delta_s[v] = (mix[LA_V_OFF + h * LA_D + v] - mem) * beta;
    }
    __syncthreads();
    float d = delta_s[v];
    acc = 0.0f;
#pragma unroll
    for (int i = 0; i < KS; i++) {
        int k = ks * KS + i;
        s[i] = __fmaf_rn(k_s[k], d, s[i]);
        S[k * LA_D + v] = s[i];
        acc = __fmaf_rn(s[i], q_s[k], acc);
    }
    __syncthreads();
    part[ks][v] = acc;
    __syncthreads();
    float o = 0.0f;
    if (ks == 0) {
#pragma unroll
        for (int j = 0; j < LA_KSPLIT; j++) o += part[j][v];
    }
    float inv = rsqrtf(block_reduce<false>(o * o, scratch) / LA_D + EPS);
    if (ks == 0) out[h * LA_D + v] = norm_w[v] * (o * inv) * silu(lin[LA_Z_OFF + h * LA_D + v]);
}

extern "C" __global__ void gdn_qk_prep(const float* mix, float* qn, float* kn) {
    __shared__ float scratch[32];
    int kh = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* row = mix + (size_t)t * LA_CONV_DIM;
    float q = row[kh * LA_D + d];
    float k = row[LA_KEY_DIM + kh * LA_D + d];
    float qss = block_reduce<false>(q * q, scratch);
    float kss = block_reduce<false>(k * k, scratch);
    size_t at = ((size_t)t * LA_KH + kh) * LA_D + d;
    qn[at] = l2_q(q, qss);
    kn[at] = l2_k(k, kss);
}

extern "C" __global__ void gdn_gates(const float* ba, const float* neg_a, const float* dt_bias, float* g, float* beta) {
    int t = blockIdx.x;
    for (int h = threadIdx.x; h < LA_VH; h += blockDim.x) {
        size_t at = (size_t)t * LA_VH + h;
        gdn_gate(ba + (size_t)t * LA_BA_ROWS, neg_a, dt_bias, h, g[at], beta[at]);
    }
}

template <int KSTEPS, typename FA, typename FB>
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
        mma_bf16(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
        mma_bf16(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
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

template <int KSTEPS>
__device__ __forceinline__ void gdn_frag_put(unsigned* base, int m, int k, float v0, float v1) {
    unsigned hi, lo;
    gdn_pack2(v0, v1, hi, lo);
    int rec = ((m >> 4) * KSTEPS + (k >> 4)) * 32 + (m & 7) * 4 + ((k & 7) >> 1);
    int reg = ((m >> 3) & 1) + 2 * ((k >> 3) & 1);
    base[rec * 8 + reg] = hi;
    base[rec * 8 + 4 + reg] = lo;
}

extern "C" __global__ void __launch_bounds__(256) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
                                                                 const float* g, const float* beta, float* w,
                                                                 float* u, float* aqk, float* gc, float* qe, float* kt, int tokens) {
    constexpr int C = LA_CHUNK;
    extern __shared__ __align__(16) unsigned char la_smem[];
    float* ks = (float*)la_smem;
    float* A = ks + C * LA_KS;
    float* gs = A + C * LA_AS;
    float* bs = gs + C;
    int h = blockIdx.x;
    int kh = h / LA_RATIO;
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
    for (int tile = warp; tile < (C / 16) * (C / 8); tile += WARPS) {
        int m0 = (tile / (C / 8)) * 16;
        int n0 = (tile % (C / 8)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (n0 <= m0 + 15) mma_split_tile<LA_D / 16>(acc, m0, n0, lane, k_row, k_col);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4 + (e & 1);
            A[i * LA_AS + j] = j < i ? -(bs[i] * acc[e]) * expf(gs[i] - gs[j]) : 0.0f;
        }
    }
    const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
    float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
    int live = min(C, tokens - t0);
    for (int tile = warp; tile < (C / 16) * (C / 8); tile += WARPS) {
        int m0 = (tile / (C / 8)) * 16;
        int n0 = (tile % (C / 8)) * 8;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (n0 <= m0 + 15)
            mma_split_tile<LA_D / 16>(
                acc, m0, n0, lane, [&](int m, int k) { return m < live ? qc[(size_t)m * LA_KH * LA_D + k] : 0.0f; },
                k_col);
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4;
            float v0 = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
            float v1 = j + 1 <= i && i < live ? acc[e + 1] * expf(gs[i] - gs[j + 1]) : 0.0f;
            gdn_frag_put<C / 16>((unsigned*)ac, i, j, v0, v1);
        }
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
    for (int tile = warp; tile < (C / 16) * (LA_D / 8); tile += WARPS) {
        int m0 = (tile / (LA_D / 8)) * 16;
        int n0 = (tile % (LA_D / 8)) * 8;
        float uu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ww[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_split_tile<C / 16>(uu, m0, n0, lane, t_at,
                               [&](int k, int n) { return k < live ? vc[(size_t)k * LA_CONV_DIM + n] * bs[k] : 0.0f; });
        mma_split_tile<C / 16>(ww, m0, n0, lane, t_at,
                               [&](int k, int n) { return ks[k * LA_KS + n] * bs[k] * expf(gs[k]); });
        unsigned* wp = (unsigned*)w + (size_t)(c * LA_VH + h) * C * LA_D;
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
            int i = m0 + g8 + (e >> 1) * 8;
            int d = n0 + 2 * t4;
            size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
            u[at] = uu[e];
            u[at + 1] = uu[e + 1];
            gdn_frag_put<LA_D / 16>(wp, i, d, ww[e], ww[e + 1]);
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
        gdn_frag_put<LA_D / 16>(qp, m, k, q.x * eg, q.y * eg);
    }
    for (int p = tid; p < C * LA_D / 2; p += blockDim.x) {
        int d = p % LA_D;
        int k = (p / LA_D) * 2;
        float v0 = 0.0f, v1 = 0.0f;
        if (t0 + k < tokens) v0 = kn[((size_t)(t0 + k) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k]);
        if (t0 + k + 1 < tokens) v1 = kn[((size_t)(t0 + k + 1) * LA_KH + kh) * LA_D + d] * expf(glast - gs[k + 1]);
        gdn_frag_put<C / 16>(kp, d, k, v0, v1);
    }
}


template <int KS, int NT>
__device__ __forceinline__ void gdn_mma_planes(float (*acc)[4], const uint4* ap, const unsigned short* bh,
                                               const unsigned short* bl, int pitch, int g, int t) {
#pragma unroll
    for (int kk = 0; kk < KS; kk++) {
        uint4 ah = ap[kk * 64];
        uint4 al = ap[kk * 64 + 1];
#pragma unroll
        for (int n = 0; n < NT; n++) {
            const unsigned* rh = (const unsigned*)(bh + (n * 8 + g) * pitch);
            const unsigned* rl = (const unsigned*)(bl + (n * 8 + g) * pitch);
            unsigned bh0 = rh[kk * 8 + t], bh1 = rh[kk * 8 + t + 4];
            unsigned bl0 = rl[kk * 8 + t], bl1 = rl[kk * 8 + t + 4];
            mma_bf16(acc[n], al.x, al.y, al.z, al.w, bh0, bh1);
            mma_bf16(acc[n], ah.x, ah.y, ah.z, ah.w, bl0, bl1);
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
    constexpr int DPW = KD / WARPS;
    constexpr int SP = LA_D + 8;
    constexpr int VP = C + 8;
    static_assert(KD % WARPS == 0, "state tiles per warp");
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
                    int col = n * 8 + 2 * t4 + (e & 1);
                    unsigned hi, lo;
                    split_hi_lo(sr[j][n][e], hi, lo);
                    sh[col * SP + d] = (unsigned short)hi;
                    sl[col * SP + d] = (unsigned short)lo;
                }
    };
    store_state_planes();
    __syncthreads();
    int chunks = (tokens + C - 1) / C;
    for (int c = 0; c < chunks; c++) {
        int t0 = c * C;
        size_t ch = (size_t)(c * LA_VH + h);
        const uint4* wp = (const uint4*)w + ch * (C * LA_D / 4) + 2 * (warp * KD * 32 + lane);
        const uint4* qp = (const uint4*)qe + ch * (C * LA_D / 4) + 2 * (warp * KD * 32 + lane);
        const uint4* ap = (const uint4*)aqk + ch * (C * C / 4) + 2 * (warp * KC * 32 + lane);
        const uint4* kp = (const uint4*)kt;
        float decay = expf(gc[ch * C + C - 1]);
        float acc[NT][4];
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[n][e] = 0.0f;
        gdn_mma_planes<KD, NT>(acc, wp, sh, sl, SP, g, t4);
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
                vl[col * VP + m] = (unsigned short)lo;
            }
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[n][e] = 0.0f;
        gdn_mma_planes<KD, NT>(acc, qp, sh, sl, SP, g, t4);
        __syncthreads();
        gdn_mma_planes<KC, NT>(acc, ap, vh, vl, VP, g, t4);
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
            const uint4* kj = kp + ch * (C * LA_D / 4) + 2 * ((warp + j * WARPS) * KC * 32 + lane);
            float a2[NT][4];
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) a2[n][e] = sr[j][n][e] * decay;
            gdn_mma_planes<KC, NT>(a2, kj, vh, vl, VP, g, t4);
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

extern "C" __global__ void lin_in_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<{{WF}}, LA_IN_ROWS, false>(w, s, b, x, nw, out);
}

extern "C" __global__ void lin_ba_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {
    linear_norm<DecBf16, LA_BA_ROWS, false>(w, s, b, x, nw, out);
}

constexpr int BA_ROWS_MAX_TOKENS = 32;
constexpr int BA_ROWS_THREADS = 256;

extern "C" __global__ void lin_ba_rows(const bf16_t* w, const float* s, const float* b, const bf16_t* xh, const bf16_t* xl,
                                       float* out, const bf16_t* dummy_hi, const bf16_t* dummy_lo, int tokens) {
    constexpr int K = HIDDEN;
    static_assert(K % (8 * BA_ROWS_THREADS / 4) == 0 || K % 8 == 0, "rows are read 8 bf16 at a time");
    __shared__ float part[BA_ROWS_THREADS / 32][BA_ROWS_MAX_TOKENS];
    int row = blockIdx.x;
    int m0 = blockIdx.y * BA_ROWS_MAX_TOKENS;
    int chunk = min(tokens - m0, BA_ROWS_MAX_TOKENS);
    int tid = threadIdx.x;
    float acc[BA_ROWS_MAX_TOKENS];
#pragma unroll
    for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) acc[m] = 0.0f;
    const uint4* wr = (const uint4*)(w + (size_t)row * K);
    for (int i = tid; i < K / 8; i += BA_ROWS_THREADS) {
        uint4 q = wr[i];
        float wf[8];
        unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wf[2 * j] = __uint_as_float(words[j] << 16);
            wf[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
        }
#pragma unroll
        for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) {
            if (m < chunk) {
                uint4 h = ((const uint4*)(xh + (size_t)(m0 + m) * K))[i];
                uint4 l = ((const uint4*)(xl + (size_t)(m0 + m) * K))[i];
                unsigned hw[4] = {h.x, h.y, h.z, h.w};
                unsigned lw[4] = {l.x, l.y, l.z, l.w};
                float a = acc[m];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    a += wf[2 * j] * (__uint_as_float(hw[j] << 16) + __uint_as_float(lw[j] << 16));
                    a += wf[2 * j + 1] * (__uint_as_float(hw[j] & 0xffff0000u) + __uint_as_float(lw[j] & 0xffff0000u));
                }
                acc[m] = a;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < BA_ROWS_MAX_TOKENS; m++) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if ((tid & 31) == 0) part[tid >> 5][m] = v;
    }
    __syncthreads();
    if (tid < chunk) {
        float v = 0.0f;
#pragma unroll
        for (int wp = 0; wp < BA_ROWS_THREADS / 32; wp++) v += part[wp][tid];
        out[(size_t)(m0 + tid) * LA_BA_ROWS + row] = v;
    }
}

extern "C" __global__ void lin_out_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    linear_w<{{WF}}, HIDDEN, LA_VALUE_DIM, false, true>(w, s, b, x, out);
}
