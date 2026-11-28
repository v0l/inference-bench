// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// linear GdnConv threads=256 smem=0 chunk=64 ksplit=4 cols=32
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
