pub const STEP: &str = r##"extern "C" __global__ void __launch_bounds__(LA_D * LA_KSPLIT) gdn_step(const float* mix, const float* lin,
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
    int kh = la_kh(h);
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
"##;
