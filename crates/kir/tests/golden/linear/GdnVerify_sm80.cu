// target Cuda cc=8.0 sms=108 warp=32 threads=1024 smem=166912 graphs=true pdl=false features=[WarpShuffle,CpAsync,Mma16816,ShaderF16]
// linear GdnVerify threads=512 smem=0 chunk=64 ksplit=4 cols=32
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

extern "C" __global__ void gdn_conv_verify(const float* lin, const float* w, const float* state, float* mix, float* log,
                                           int slot, int seq_slots, const int* slots, int rows) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (c >= LA_CONV_DIM) return;
    const float* s = conv_state_row((float*)state, slot + slots[seq * rows] * seq_slots, c);
    float window[LA_K];
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) window[j] = s[j];
    for (int r = 0; r < rows; r++) {
        size_t row = (size_t)seq * rows + r;
        float x = lin[row * LA_IN_ROWS + c];
        window[LA_K - 1] = x;
        mix[row * LA_CONV_DIM + c] = conv_tap(w + (size_t)c * LA_K, window);
        log[row * LA_CONV_DIM + c] = x;
#pragma unroll
        for (int j = 0; j < LA_K - 1; j++) window[j] = window[j + 1];
    }
}

extern "C" __global__ void gdn_conv_commit(float* state, const float* log, int seq_slots, const int* slots, int rows,
                                           const int* accepted, int log_rows) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    int layer = blockIdx.z;
    if (c >= LA_CONV_DIM) return;
    float* s = conv_state_row(state, layer + slots[seq * rows] * seq_slots, c);
    log += (size_t)layer * log_rows * LA_CONV_DIM;
    float window[LA_K];
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) window[j] = s[j];
    for (int r = 0; r <= accepted[seq]; r++) {
        window[LA_K - 1] = log[((size_t)seq * rows + r) * LA_CONV_DIM + c];
#pragma unroll
        for (int j = 0; j < LA_K - 1; j++) window[j] = window[j + 1];
    }
#pragma unroll
    for (int j = 0; j < LA_K - 1; j++) s[j] = window[j];
}

extern "C" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_verify(const float* mix, const float* lin,
                                                                          const float* ba, const float* neg_a,
                                                                          const float* dt_bias, const float* norm_w,
                                                                          const float* state, float* out, float* klog,
                                                                          float* dlog, float* glog, int slot,
                                                                          int seq_slots, const int* slots, int rows) {
    constexpr int KS = LA_D / LA_KSPLIT;
    __shared__ float scratch[32];
    __shared__ float q_s[LA_D];
    __shared__ float k_s[LA_D];
    __shared__ float part[LA_KSPLIT][LA_D];
    __shared__ float delta_s[LA_D];
    int h = blockIdx.x;
    int seq = blockIdx.y;
    slot += slots[seq * rows] * seq_slots;
    int kh = la_kh(h);
    int tid = threadIdx.x;
    int v = tid % LA_D;
    int ks = tid / LA_D;
    const float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    float s[KS];
#pragma unroll
    for (int i = 0; i < KS; i++) s[i] = S[(ks * KS + i) * LA_D + v];
    for (int r = 0; r < rows; r++) {
        size_t row = (size_t)seq * rows + r;
        const float* m = mix + row * LA_CONV_DIM;
        const float* l = lin + row * LA_IN_ROWS;
        const float* b = ba + row * LA_BA_ROWS;
        float qv = tid < LA_D ? m[kh * LA_D + tid] : 0.0f;
        float kv = tid < LA_D ? m[LA_KEY_DIM + kh * LA_D + tid] : 0.0f;
        float qss = block_reduce<false>(qv * qv, scratch);
        float kss = block_reduce<false>(kv * kv, scratch);
        if (tid < LA_D) {
            q_s[tid] = l2_q(qv, qss);
            k_s[tid] = l2_k(kv, kss);
        }
        float g, beta;
        gdn_gate(b, neg_a, dt_bias, h, g, beta);
        float decay = expf(g);
        __syncthreads();
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < KS; i++) {
            int k = ks * KS + i;
            s[i] = __fmul_rn(s[i], decay);
            acc = __fmaf_rn(s[i], k_s[k], acc);
        }
        part[ks][v] = acc;
        __syncthreads();
        if (ks == 0) {
            float mem = 0.0f;
#pragma unroll
            for (int j = 0; j < LA_KSPLIT; j++) mem += part[j][v];
            delta_s[v] = (m[LA_V_OFF + h * LA_D + v] - mem) * beta;
        }
        __syncthreads();
        float d = delta_s[v];
        acc = 0.0f;
#pragma unroll
        for (int i = 0; i < KS; i++) {
            int k = ks * KS + i;
            s[i] = __fmaf_rn(k_s[k], d, s[i]);
            acc = __fmaf_rn(s[i], q_s[k], acc);
        }
        if (tid < LA_D) {
            size_t at = (row * LA_VH + h) * LA_D + tid;
            klog[at] = k_s[tid];
            dlog[at] = delta_s[tid];
        }
        if (tid == 0) glog[row * LA_VH + h] = decay;
        __syncthreads();
        part[ks][v] = acc;
        __syncthreads();
        float o = 0.0f;
        if (ks == 0) {
#pragma unroll
            for (int j = 0; j < LA_KSPLIT; j++) o += part[j][v];
        }
        float inv = rsqrtf(block_reduce<false>(o * o, scratch) / LA_D + EPS);
        if (ks == 0) out[row * LA_VALUE_DIM + h * LA_D + v] = norm_w[v] * (o * inv) * silu(l[LA_Z_OFF + h * LA_D + v]);
    }
}

extern "C" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_commit(float* state, const float* klog, const float* dlog,
                                                                          const float* glog, int seq_slots,
                                                                          const int* slots, int rows,
                                                                          const int* accepted, int log_rows) {
    constexpr int KS = LA_D / LA_KSPLIT;
    int h = blockIdx.x;
    int seq = blockIdx.y;
    int layer = blockIdx.z;
    int slot = layer + slots[seq * rows] * seq_slots;
    int v = threadIdx.x % LA_D;
    int ks = threadIdx.x / LA_D;
    size_t base = (size_t)layer * log_rows;
    float* S = state + ((size_t)slot * LA_VH + h) * LA_STATE;
    float s[KS];
#pragma unroll
    for (int i = 0; i < KS; i++) s[i] = S[(ks * KS + i) * LA_D + v];
    for (int r = 0; r <= accepted[seq]; r++) {
        size_t row = base + (size_t)seq * rows + r;
        const float* k_s = klog + (row * LA_VH + h) * LA_D;
        float decay = glog[row * LA_VH + h];
        float d = dlog[(row * LA_VH + h) * LA_D + v];
#pragma unroll
        for (int i = 0; i < KS; i++) {
            int k = ks * KS + i;
            s[i] = __fmul_rn(s[i], decay);
            s[i] = __fmaf_rn(k_s[k], d, s[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < KS; i++) S[(ks * KS + i) * LA_D + v] = s[i];
}
