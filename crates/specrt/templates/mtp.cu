extern "C" __global__ void mtp_fc_in(const bf16_t* table, const int* tokens, const float* hsrc, const int* hidx,
                                     const float* pre, const float* ne, const float* nh, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    int r = blockIdx.x;
    const bf16_t* e = table + (size_t)tokens[r] * HIDDEN;
    const float* h = hsrc + (size_t)hidx[r] * HIDDEN;
    float se = 0.0f, sx = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float a = bf16_to_f32(e[i]);
        se += a * a;
        sx += h[i] * h[i];
    }
    float ie = rsqrtf(block_reduce<false>(se, scratch) / HIDDEN + EPS);
    float ix = rsqrtf(block_reduce<false>(sx, scratch) / HIDDEN + EPS);
    float sh = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float g = pre[i] * (h[i] * ix);
        sh += g * g;
    }
    float ih = rsqrtf(block_reduce<false>(sh, scratch) / HIDDEN + EPS);
    size_t row = (size_t)r * 2 * HIDDEN;
    for (int i = 2 * threadIdx.x; i < HIDDEN; i += 2 * blockDim.x) {
        float g0 = pre[i] * (h[i] * ix), g1 = pre[i + 1] * (h[i + 1] * ix);
        split_store2(hi, lo, row + i, ne[i] * (bf16_to_f32(e[i]) * ie), ne[i + 1] * (bf16_to_f32(e[i + 1]) * ie));
        split_store2(hi, lo, row + HIDDEN + i, nh[i] * (g0 * ih), nh[i + 1] * (g1 * ih));
    }
}

extern "C" __global__ void mtp_draft(const float* logits, int* drafts, int col, int k, int* ctok) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    int s = blockIdx.x;
    const float* row = logits + (size_t)s * VOCAB;
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        if (row[i] > bv) {
            bv = row[i];
            bi = i;
        }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            float ov = best_v[threadIdx.x + w];
            int oi = best_i[threadIdx.x + w];
            if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                best_v[threadIdx.x] = ov;
                best_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        drafts[s * k + col] = best_i[0];
        ctok[s] = best_i[0];
    }
}

__device__ __forceinline__ float spec_uniform(unsigned seed, unsigned step) {
    unsigned x = seed * 747796405u + step * 2891336453u + 0x68e31da4u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    return ((float)(x >> 8) + 0.5f) / 16777216.0f;
}

extern "C" __global__ void spec_accept(const float* logits, const int* vtok, const int* params, const int* max_accept,
                                       int k, int* out, int* accepted) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    __shared__ float scratch[32];
    int s = blockIdx.x;
    const int* p = params + 3 * s;
    float inv_temp = __int_as_float(p[0]);
    unsigned seed = (unsigned)p[1];
    unsigned step = (unsigned)p[2];
    int a = 0;
    for (int j = 0; j <= k; j++) {
        const float* row = logits + (size_t)(s * (k + 1) + j) * VOCAB;
        int d = j < k ? vtok[s * (k + 1) + 1 + j] : -1;
        bool may = j < k && j < max_accept[s];
        bool sampled = inv_temp > 0.0f;
        bool take = false;
        if (sampled && may) {
            float m = NEG_INF;
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) m = fmaxf(m, row[i]);
            m = block_reduce<true>(m, scratch);
            float sum = 0.0f;
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) sum += expf((row[i] - m) * inv_temp);
            sum = block_reduce<false>(sum, scratch);
            take = spec_uniform(seed, step + j) < expf((row[d] - m) * inv_temp) / sum;
        }
        int tok = d;
        if (!take) {
            int excl = sampled && may ? d : -1;
            float bv = NEG_INF;
            int bi = 0;
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                if (i == excl) continue;
                float v = row[i];
                if (sampled) v = v * inv_temp + gumbel(seed, step + j, (unsigned)i);
                if (v > bv) {
                    bv = v;
                    bi = i;
                }
            }
            __syncthreads();
            best_v[threadIdx.x] = bv;
            best_i[threadIdx.x] = bi;
            __syncthreads();
            for (int w = blockDim.x / 2; w > 0; w >>= 1) {
                if (threadIdx.x < w) {
                    float ov = best_v[threadIdx.x + w];
                    int oi = best_i[threadIdx.x + w];
                    if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
                        best_v[threadIdx.x] = ov;
                        best_i[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            tok = best_i[0];
            take = !sampled && may && tok == d;
        }
        if (threadIdx.x == 0) out[s * (k + 1) + j] = tok;
        __syncthreads();
        if (!take) break;
        a++;
    }
    if (threadIdx.x == 0) accepted[s] = a;
}

extern "C" __global__ void mtp_fill(const int* drafts, int* vtok, int k) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    if (j < k) vtok[s * (k + 1) + 1 + j] = drafts[s * k + j];
}

extern "C" __global__ void mtp_gather(const float* src, const int* sidx, float* dst, const int* didx) {
    int r = blockIdx.y;
    const float* a = src + (size_t)sidx[r] * HIDDEN;
    float* b = dst + (size_t)didx[r] * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x) b[i] = a[i];
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
    int kh = h / LA_RATIO;
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
