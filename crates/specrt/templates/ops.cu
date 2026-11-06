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
constexpr int MAX_CTX = {{MAX_CTX}};
constexpr int PAGE = {{PAGE}};
constexpr int PAGES = (MAX_CTX + PAGE - 1) / PAGE;
constexpr int Q8_GROUP = {{Q8_GROUP}};
constexpr int LINEAR_WARPS = 8;
constexpr int ATTN_THREADS = 128;
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

template <int N, int K, bool BIAS, bool RESIDUAL>
__device__ void linear_bf16(const void* w, const float* scales, const float* bias, const float* x, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * LINEAR_WARPS + warp;
    if (row >= N) return;
    const bf16_t* wr = (const bf16_t*)w + (size_t)row * K;
    float acc = 0.0f;
#pragma unroll 4
    for (int k = lane * 8; k < K; k += 256) {
        uint4 packed = *(const uint4*)(wr + k);
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

extern "C" __global__ void embed(const bf16_t* table, const int* token, float* x) {
    const bf16_t* row = table + (size_t)token[0] * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x)
        x[i] = bf16_to_f32(row[i]);
}

extern "C" __global__ void rmsnorm(const float* x, const float* weight, float* out) {
    __shared__ float scratch[32];
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
    int pos = pos_ptr[0];
    bool is_q = head < HEADS;
    bool is_k = !is_q && head < HEADS + KV_HEADS;
    float* src = qkv + head * HEAD_DIM;
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
                                     const int* block_table, const int* pos_ptr, int layer, float* out) {
    __shared__ float scratch[32];
    __shared__ float scores[MAX_CTX];
    int head = blockIdx.x;
    int kv_head = head / GROUP;
    int len = pos_ptr[0] + 1;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const float* q = qkv + head * HEAD_DIM;
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
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; t++) {
            size_t slot = (((size_t)layer * PAGES + block_table[t / PAGE]) * KV_HEADS + kv_head) * PAGE + t % PAGE;
            acc += scores[t] * v_cache[slot * HEAD_DIM + d];
        }
        out[head * HEAD_DIM + d] = acc / sum;
    }
}

extern "C" __global__ void swiglu(const float* gate_up, float* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < INTER; i += gridDim.x * blockDim.x) {
        float g = gate_up[i];
        out[i] = g / (1.0f + expf(-g)) * gate_up[INTER + i];
    }
}

extern "C" __global__ void argmax_advance(const float* logits, int* token, int* pos, int* history, int* count) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
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
