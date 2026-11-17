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
