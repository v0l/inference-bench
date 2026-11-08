constexpr int LA_KH = {{LA_KH}};
constexpr int LA_VH = {{LA_VH}};
constexpr int LA_D = {{LA_D}};
constexpr int LA_K = {{LA_CONV_K}};
constexpr int LA_KEY_DIM = LA_KH * LA_D;
constexpr int LA_VALUE_DIM = LA_VH * LA_D;
constexpr int LA_CONV_DIM = 2 * LA_KEY_DIM + LA_VALUE_DIM;
constexpr int LA_IN_ROWS = LA_CONV_DIM + LA_VALUE_DIM + 2 * LA_VH;
constexpr int LA_RATIO = LA_VH / LA_KH;
constexpr int LA_V_OFF = 2 * LA_KEY_DIM;
constexpr int LA_Z_OFF = LA_CONV_DIM;
constexpr int LA_B_OFF = LA_CONV_DIM + LA_VALUE_DIM;
constexpr int LA_A_OFF = LA_B_OFF + LA_VH;
constexpr int LA_STATE = LA_D * LA_D;
constexpr int LA_CHUNK = {{LA_CHUNK}};
constexpr int LA_KSPLIT = {{LA_KSPLIT}};
constexpr int LA_KS = LA_D + 1;
constexpr int LA_AS = LA_CHUNK + 1;
constexpr int LA_COLS = 32;
constexpr float LA_L2_EPS = 1e-6f;
static_assert(LA_D % LA_COLS == 0 && LA_D <= 128, "linear head dim");
static_assert(LA_VH % LA_KH == 0, "linear head groups");

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ void gdn_gate(const float* lin_row, const float* neg_a, const float* dt_bias, int h,
                                         float& g, float& beta) {
    g = neg_a[h] * softplus(lin_row[LA_A_OFF + h] + dt_bias[h]);
    beta = 1.0f / (1.0f + expf(-lin_row[LA_B_OFF + h]));
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
    gdn_gate(lin, neg_a, dt_bias, h, g, beta);
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

extern "C" __global__ void gdn_gates(const float* lin, const float* neg_a, const float* dt_bias, float* g, float* beta) {
    int t = blockIdx.x;
    for (int h = threadIdx.x; h < LA_VH; h += blockDim.x) {
        size_t at = (size_t)t * LA_VH + h;
        gdn_gate(lin + (size_t)t * LA_IN_ROWS, neg_a, dt_bias, h, g[at], beta[at]);
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
    for (int e = tid; e < C * C; e += blockDim.x) {
        int i = e / C;
        int j = e % C;
        float v = 0.0f;
        if (j < i) {
            float dot = 0.0f;
            for (int d = 0; d < LA_D; d++) dot += ks[i * LA_KS + d] * ks[j * LA_KS + d];
            v = -(bs[i] * dot) * expf(gs[i] - gs[j]);
        }
        A[i * LA_AS + j] = v;
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
    for (int e = tid; e < C * LA_D; e += blockDim.x) {
        int i = e / LA_D;
        int d = e % LA_D;
        float uu = 0.0f, ww = 0.0f;
        for (int j = 0; j <= i; j++) {
            float tij = j == i ? 1.0f : A[i * LA_AS + j];
            int tj = t0 + j;
            float vj = tj < tokens ? mix[(size_t)tj * LA_CONV_DIM + LA_V_OFF + h * LA_D + d] : 0.0f;
            uu += tij * (vj * bs[j]);
            ww += tij * (ks[j * LA_KS + d] * bs[j] * expf(gs[j]));
        }
        size_t at = ((size_t)(t0 + i) * LA_VH + h) * LA_D + d;
        u[at] = uu;
        w[at] = ww;
    }
    for (int e = tid; e < C * C; e += blockDim.x) {
        int i = e / C;
        int j = e % C;
        int t = t0 + i;
        float v = 0.0f;
        if (j <= i && t < tokens) {
            const float* q = qn + ((size_t)t * LA_KH + kh) * LA_D;
            float dot = 0.0f;
            for (int d = 0; d < LA_D; d++) dot += q[d] * ks[j * LA_KS + d];
            v = dot * expf(gs[i] - gs[j]);
        }
        aqk[((size_t)(c * LA_VH + h) * C + i) * C + j] = v;
    }
    if (tid < C) gc[(size_t)(c * LA_VH + h) * C + tid] = gs[tid];
}

extern "C" __global__ void __launch_bounds__(256) gdn_chunk_scan(const float* qn, const float* kn, const float* w,
                                                                 const float* u, const float* aqk, const float* gc,
                                                                 float* state, float* o, int slot, int tokens) {
    constexpr int C = LA_CHUNK;
    constexpr int WARPS = 8;
    __shared__ float S[LA_D][LA_COLS];
    __shared__ float vn[C][LA_COLS];
    __shared__ float gsh[C];
    __shared__ float tail[C];
    int h = blockIdx.x;
    int kh = h / LA_RATIO;
    int c0 = blockIdx.y * LA_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float* st = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    for (int k = warp; k < LA_D; k += WARPS) S[k][lane] = st[k * LA_D + c0 + lane];
    int chunks = (tokens + C - 1) / C;
    for (int c = 0; c < chunks; c++) {
        __syncthreads();
        if (threadIdx.x < C) gsh[threadIdx.x] = gc[(size_t)(c * LA_VH + h) * C + threadIdx.x];
        __syncthreads();
        float glast = gsh[C - 1];
        if (threadIdx.x < C) tail[threadIdx.x] = expf(glast - gsh[threadIdx.x]);
        for (int i = warp; i < C; i += WARPS) {
            size_t row = ((size_t)(c * C + i) * LA_VH + h) * LA_D;
            float acc = 0.0f;
            for (int k = 0; k < LA_D; k++) acc += w[row + k] * S[k][lane];
            vn[i][lane] = u[row + c0 + lane] - acc;
        }
        __syncthreads();
        for (int i = warp; i < C; i += WARPS) {
            int t = c * C + i;
            if (t >= tokens) break;
            const float* q = qn + ((size_t)t * LA_KH + kh) * LA_D;
            float eg = expf(gsh[i]);
            float inter = 0.0f;
            for (int k = 0; k < LA_D; k++) inter += (q[k] * eg) * S[k][lane];
            const float* a = aqk + ((size_t)(c * LA_VH + h) * C + i) * C;
            float intra = 0.0f;
            for (int j = 0; j <= i; j++) intra += a[j] * vn[j][lane];
            o[((size_t)t * LA_VH + h) * LA_D + c0 + lane] = inter + intra;
        }
        __syncthreads();
        float decay = expf(glast);
        for (int k = warp; k < LA_D; k += WARPS) {
            float acc = S[k][lane] * decay;
            for (int i = 0; i < C; i++) {
                int t = c * C + i;
                float kd = t < tokens ? kn[((size_t)t * LA_KH + kh) * LA_D + k] * tail[i] : 0.0f;
                acc += kd * vn[i][lane];
            }
            S[k][lane] = acc;
        }
    }
    __syncthreads();
    for (int k = warp; k < LA_D; k += WARPS) st[k * LA_D + c0 + lane] = S[k][lane];
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
    linear_norm<{{Q8}}, LA_IN_ROWS, false>(w, s, b, x, nw, out);
}

extern "C" __global__ void lin_out_proj(const void* w, const float* s, const float* b, const float* x, float* out) {
    {{LINEAR}}<HIDDEN, LA_VALUE_DIM, false, true>(w, s, b, x, out);
}
