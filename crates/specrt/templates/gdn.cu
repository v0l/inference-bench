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

extern "C" __global__ void gdn_conv_step(const float* lin, const float* w, float* state, float* mix, int slot, int seq_slots) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (c >= LA_CONV_DIM) return;
    lin += (size_t)seq * LA_IN_ROWS;
    mix += (size_t)seq * LA_CONV_DIM;
    float* s = conv_state_row(state, slot + seq * seq_slots, c);
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
                                                                        float* out, int slot, int seq_slots) {
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
    slot += seq * seq_slots;
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
        s[i] = S[k * LA_D + v] * decay;
        acc += s[i] * k_s[k];
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
        s[i] += k_s[k] * d;
        S[k * LA_D + v] = s[i];
        acc += s[i] * q_s[k];
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

extern "C" __global__ void __launch_bounds__(256) gdn_chunk_prep(const float* qn, const float* kn, const float* mix,
                                                                 const float* g, const float* beta, float* w,
                                                                 float* u, float* aqk, float* gc, int tokens) {
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
        for (int e = 0; e < 4; e++) {
            int i = m0 + g8 + (e >> 1) * 8;
            int j = n0 + 2 * t4 + (e & 1);
            ac[i * C + j] = j <= i && i < live ? acc[e] * expf(gs[i] - gs[j]) : 0.0f;
        }
    }
    __syncthreads();
    for (int i = 1; i < C; i++) {
        int j = tid;
        float next = 0.0f;
        if (j < i) {
            next = A[i * LA_AS + j];
            for (int m = j + 1; m < i; m++) next += A[i * LA_AS + m] * A[m * LA_AS + j];
        }
        __syncthreads();
        if (j < i) A[i * LA_AS + j] = next;
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
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int i = m0 + g8 + (e >> 1) * 8;
            int d = n0 + 2 * t4 + (e & 1);
            size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
            u[at] = uu[e];
            w[at] = ww[e];
        }
    }
    if (tid < C) gc[(size_t)(c * LA_VH + h) * C + tid] = gs[tid];
}


extern "C" __global__ void __launch_bounds__(256) gdn_chunk_scan(const float* qn, const float* kn, const float* w,
                                                                 const float* u, const float* aqk, const float* gc,
                                                                 float* state, float* o, int slot, int tokens) {
    constexpr int C = LA_CHUNK;
    constexpr int WARPS = 8;
    constexpr int NT = LA_COLS / 8;
    constexpr int CT = (C / 16) * NT;
    constexpr int ST = (LA_D / 16) * NT;
    static_assert(CT % WARPS == 0 && ST % WARPS == 0, "warp tiling");
    __shared__ float S[LA_D][LA_COLS + 1];
    __shared__ float vn[C][LA_COLS + 1];
    __shared__ float gsh[C];
    __shared__ float eg[C];
    __shared__ float tail[C];
    int h = blockIdx.x;
    int kh = h / LA_RATIO;
    int c0 = blockIdx.y * LA_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t4 = lane & 3;
    float* st = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    for (int i = threadIdx.x; i < LA_D * LA_COLS; i += blockDim.x)
        S[i / LA_COLS][i % LA_COLS] = st[(i / LA_COLS) * LA_D + c0 + i % LA_COLS];
    int chunks = (tokens + C - 1) / C;
    auto s_at = [&](int k, int n) { return S[k][n]; };
    auto vn_at = [&](int k, int n) { return vn[k][n]; };
    for (int c = 0; c < chunks; c++) {
        int t0 = c * C;
        __syncthreads();
        if (threadIdx.x < C) gsh[threadIdx.x] = gc[(size_t)(c * LA_VH + h) * C + threadIdx.x];
        __syncthreads();
        float glast = gsh[C - 1];
        if (threadIdx.x < C) {
            int i = threadIdx.x;
            bool live = t0 + i < tokens;
            tail[i] = live ? expf(glast - gsh[i]) : 0.0f;
            eg[i] = live ? expf(gsh[i]) : 0.0f;
        }
        const float* wc = w + ((size_t)t0 * LA_VH + h) * LA_D;
        const float* uc = u + ((size_t)t0 * LA_VH + h) * LA_D + c0;
        for (int tile = warp; tile < CT; tile += WARPS) {
            int m0 = (tile / NT) * 16;
            int n0 = (tile % NT) * 8;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma_split_tile<LA_D / 16>(acc, m0, n0, lane,
                                      [&](int m, int k) { return wc[(size_t)m * LA_VH * LA_D + k]; }, s_at);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int m = m0 + g + (e >> 1) * 8;
                int n = n0 + 2 * t4 + (e & 1);
                vn[m][n] = uc[(size_t)m * LA_VH * LA_D + n] - acc[e];
            }
        }
        __syncthreads();
        const float* qc = qn + ((size_t)t0 * LA_KH + kh) * LA_D;
        const float* ac = aqk + (size_t)(c * LA_VH + h) * C * C;
        for (int tile = warp; tile < CT; tile += WARPS) {
            int m0 = (tile / NT) * 16;
            int n0 = (tile % NT) * 8;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma_split_tile<LA_D / 16>(acc, m0, n0, lane,
                                      [&](int m, int k) { return qc[(size_t)m * LA_KH * LA_D + k] * eg[m]; }, s_at);
            mma_split_tile<C / 16>(acc, m0, n0, lane, [&](int m, int k) { return ac[m * C + k]; }, vn_at);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int m = m0 + g + (e >> 1) * 8;
                int n = n0 + 2 * t4 + (e & 1);
                if (t0 + m < tokens) o[((size_t)(t0 + m) * LA_VH + h) * LA_D + c0 + n] = acc[e];
            }
        }
        __syncthreads();
        float decay = expf(glast);
        const float* kc = kn + ((size_t)t0 * LA_KH + kh) * LA_D;
        for (int tile = warp; tile < ST; tile += WARPS) {
            int m0 = (tile / NT) * 16;
            int n0 = (tile % NT) * 8;
            float acc[4];
#pragma unroll
            for (int e = 0; e < 4; e++) acc[e] = S[m0 + g + (e >> 1) * 8][n0 + 2 * t4 + (e & 1)] * decay;
            mma_split_tile<C / 16>(acc, m0, n0, lane,
                                   [&](int m, int k) { return kc[(size_t)k * LA_KH * LA_D + m] * tail[k]; }, vn_at);
#pragma unroll
            for (int e = 0; e < 4; e++) S[m0 + g + (e >> 1) * 8][n0 + 2 * t4 + (e & 1)] = acc[e];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < LA_D * LA_COLS; i += blockDim.x)
        st[(i / LA_COLS) * LA_D + c0 + i % LA_COLS] = S[i / LA_COLS][i % LA_COLS];
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
    linear_norm<WF_BF16, LA_BA_ROWS, false>(w, s, b, x, nw, out);
}

extern "C" __global__ void lin_out_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<HIDDEN, LA_VALUE_DIM, false, true>(w, s, b, x, out);
}
