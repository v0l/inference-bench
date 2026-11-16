// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// attention SplitDecode threads=128 smem=0
//   AttentionShape { heads: 24, kv_heads: 4, head_dim: 256, qkv_row: 14336, output_gate: true, qk_norm: true, eps: 1e-6, rope: RopeShape { rotary_dim: 64, theta: 10000000.0, kind: 0, factor: 1.0, low: 1.0, high: 1.0, original: 1.0, mscale: 1.0, mrope: None }, kv_layers: 16, kv_planes: 2, kv_fp8: false, kv_f16_layers: 0, query_scale: 0.0625, softcap: 0.0, local_theta: 10000.0, local_rope: [false, true, false, false], bidirectional: false, nope: [false, false, true, false] }
constexpr int ROT = 64;
constexpr int ROT_HALF = ROT / 2;
constexpr bool QK_NORM = true;
constexpr bool LOCAL_ROPE[] = { false, true, false, false };
constexpr bool NOPE[] = { false, false, true, false };
constexpr float LOCAL_THETA = 1e4f;
constexpr float QUERY_SCALE = 6.25e-2f;
constexpr bool BIDI = false;
constexpr float SOFTCAP = 0e0f;
constexpr float THETA = 1e7f;
constexpr int ROPE_KIND = 0;
constexpr bool MROPE = false;
constexpr bool MROPE_INTERLEAVED = false;
constexpr int MROPE_S0 = 0;
constexpr int MROPE_S1 = 0;
constexpr int MROPE_S2 = 0;
constexpr float ROPE_FACTOR = 1e0f;
constexpr float ROPE_LOW_FREQ = 1e0f;
constexpr float ROPE_HIGH_FREQ = 1e0f;
constexpr float ROPE_ORIG_POS = 1e0f;
constexpr float ROPE_MSCALE = 1e0f;
constexpr int PAGE = 16;
constexpr int ATTN_LAYERS = 16;
constexpr int KV_PLANES = 2;
constexpr bool KV_FP8 = false;
constexpr int KV_F16_LAYERS = 0;

__device__ __forceinline__ float rope_inv_freq(int i, int layer) {
    if (LOCAL_ROPE[layer]) return 1.0f / powf(LOCAL_THETA, (float)(2 * i) / (float)ROT);
    float inv_freq = 1.0f / powf(THETA, (float)(2 * i) / (float)ROT);
    if (ROPE_KIND == 1) {
        inv_freq /= ROPE_FACTOR;
    } else if (ROPE_KIND == 2) {
        float wavelen = 6.283185307179586f / inv_freq;
        float low_wavelen = ROPE_ORIG_POS / ROPE_LOW_FREQ;
        float high_wavelen = ROPE_ORIG_POS / ROPE_HIGH_FREQ;
        if (wavelen > low_wavelen) {
            inv_freq /= ROPE_FACTOR;
        } else if (wavelen >= high_wavelen) {
            float smooth = (ROPE_ORIG_POS / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
            inv_freq = (1.0f - smooth) * inv_freq / ROPE_FACTOR + smooth * inv_freq;
        }
    } else if (ROPE_KIND == 3) {
        float ramp = fminf(fmaxf(((float)i - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ), 0.0f), 1.0f);
        inv_freq = ramp * inv_freq / ROPE_FACTOR + (1.0f - ramp) * inv_freq;
    }
    return inv_freq;
}

__device__ __forceinline__ float cap_score(float s) {
    return SOFTCAP > 0.0f ? SOFTCAP * tanhf(s / SOFTCAP) : s;
}

__device__ __forceinline__ int window_lo(int len, int window) {
    return window > 0 ? max(0, len - window) : 0;
}

__device__ __forceinline__ unsigned f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float f16_to_f32(unsigned bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ unsigned attn_bits(float v) {
    return f16_bits(v);
}

__device__ __forceinline__ float attn_float(unsigned bits) {
    return f16_to_f32(bits & 0xffffu);
}

__device__ __forceinline__ void attn_split(float v, unsigned& hi, unsigned& lo) {
    hi = attn_bits(v);
    lo = attn_bits(v - attn_float(hi));
}

__device__ __forceinline__ size_t kv_slot(const int* block_table, int layer, int kv_head, int pos) {
    return (((size_t)block_table[pos / PAGE] * ATTN_LAYERS + layer) * KV_HEADS + kv_head) * PAGE + pos % PAGE;
}

constexpr size_t KV_ROWS_PER_LAYER = (size_t)KV_HEADS * PAGE;
constexpr size_t KV_RAW_LAYER_BYTES = KV_ROWS_PER_LAYER * HEAD_DIM * 2 * KV_PLANES;
constexpr size_t KV_Q_LAYER_BYTES = KV_ROWS_PER_LAYER * (HEAD_DIM + 4);
constexpr size_t KV_BLOCK_BYTES = KV_F16_LAYERS * KV_RAW_LAYER_BYTES + (ATTN_LAYERS - KV_F16_LAYERS) * KV_Q_LAYER_BYTES;

__device__ __forceinline__ bool kv_q(int layer) {
    return KV_FP8 && layer >= KV_F16_LAYERS;
}

__device__ __forceinline__ size_t kv_layer_base(const int* block_table, int layer, int pos) {
    size_t within = layer < KV_F16_LAYERS ? (size_t)layer * KV_RAW_LAYER_BYTES
                                          : KV_F16_LAYERS * KV_RAW_LAYER_BYTES + (size_t)(layer - KV_F16_LAYERS) * KV_Q_LAYER_BYTES;
    return (size_t)block_table[pos / PAGE] * KV_BLOCK_BYTES + within;
}

__device__ __forceinline__ bf16_t* kv_row(bf16_t* cache, const int* block_table, int layer, int kv_head, int pos) {
    if (KV_FP8) {
        size_t row = (size_t)kv_head * PAGE + pos % PAGE;
        size_t base = kv_layer_base(block_table, layer, pos);
        return (bf16_t*)((unsigned char*)cache + base + row * (kv_q(layer) ? HEAD_DIM : HEAD_DIM * 2 * KV_PLANES));
    }
    return cache + kv_slot(block_table, layer, kv_head, pos) * KV_PLANES * HEAD_DIM;
}

__device__ __forceinline__ const bf16_t* kv_row(const bf16_t* cache, const int* block_table, int layer, int kv_head,
                                                int pos) {
    return kv_row(const_cast<bf16_t*>(cache), block_table, layer, kv_head, pos);
}

__device__ __forceinline__ float* kv_scale_ptr(bf16_t* cache, const int* block_table, int layer, int kv_head, int pos) {
    size_t row = (size_t)kv_head * PAGE + pos % PAGE;
    return (float*)((unsigned char*)cache + kv_layer_base(block_table, layer, pos) + KV_ROWS_PER_LAYER * HEAD_DIM + row * 4);
}

__device__ __forceinline__ float kv_scale(const bf16_t* cache, const int* block_table, int layer, int kv_head, int pos) {
    if (!kv_q(layer)) return 1.0f;
    return *kv_scale_ptr(const_cast<bf16_t*>(cache), block_table, layer, kv_head, pos);
}

__device__ __forceinline__ const int* seq_table(const int* block_table, const int* slots, int row, int pages) {
    return block_table + (size_t)slots[row] * pages;
}

__device__ __forceinline__ float kv_at(const bf16_t* row, int d, float sc, bool q) {
    if (q) return e4m3_to_f32(((const unsigned char*)row)[d]) * sc;
    float v = attn_float(row[d]);
    return KV_PLANES == 2 ? v + attn_float(row[HEAD_DIM + d]) : v;
}

__device__ __forceinline__ float4 kv_at4(const bf16_t* row, int d, float sc, bool q) {
    if (q) {
        unsigned w = *(const unsigned*)((const unsigned char*)row + d);
        return make_float4(e4m3_to_f32(w & 0xffu) * sc, e4m3_to_f32((w >> 8) & 0xffu) * sc,
                           e4m3_to_f32((w >> 16) & 0xffu) * sc, e4m3_to_f32(w >> 24) * sc);
    }
    uint2 h = *(const uint2*)(row + d);
    float4 v = make_float4(attn_float(h.x), attn_float(h.x >> 16), attn_float(h.y), attn_float(h.y >> 16));
    if (KV_PLANES == 2) {
        uint2 l = *(const uint2*)(row + HEAD_DIM + d);
        v.x += attn_float(l.x);
        v.y += attn_float(l.x >> 16);
        v.z += attn_float(l.y);
        v.w += attn_float(l.y >> 16);
    }
    return v;
}

__device__ __forceinline__ int mrope_axis(int i) {
    if (MROPE_INTERLEAVED) {
        if (i % 3 == 1 && i < 3 * MROPE_S1) return 1;
        if (i % 3 == 2 && i < 3 * MROPE_S2) return 2;
        return 0;
    }
    if (i < MROPE_S0) return 0;
    if (i < MROPE_S0 + MROPE_S1) return 1;
    return 2;
}

__device__ __forceinline__ float rope_position(const int* mm_meta, const int* mm_table, int mm_stride, int slot, int idx, int i) {
    if (!MROPE) return (float)idx;
    if (idx < mm_meta[2 * slot]) return (float)mm_table[((size_t)slot * mm_stride + idx) * 3 + mrope_axis(i)];
    return (float)(idx + mm_meta[2 * slot + 1]);
}

__device__ __forceinline__ int span_end(const int* mm_meta, const int* bidi, int mm_stride, int slot, int idx) {
    if (!BIDI) return idx;
    return idx < mm_meta[2 * slot] ? max(idx, bidi[(size_t)slot * mm_stride + idx]) : idx;
}

__device__ __forceinline__ void norm_rope_head(float* qkv, const float* q_norm, const float* k_norm,
                                               bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                               int pos, int layer, int head, int row,
                                               const int* mm_meta, const int* mm_table, int mm_stride, int slot) {
    __shared__ float scratch[32];
    __shared__ float vec[HEAD_DIM];
    int d = threadIdx.x;
    bool is_q = head < HEADS;
    bool is_k = !is_q && head < HEADS + KV_HEADS;
    float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
    float v = d < HEAD_DIM ? src[d] : 0.0f;
    if (QK_NORM && (is_q || is_k)) {
        float inv = rsqrtf(block_reduce<false>(v * v, scratch) / HEAD_DIM + EPS);
        const float* w = is_q ? q_norm : k_norm;
        if (d < HEAD_DIM) v = w[d] * (v * inv);
    }
    if (d < HEAD_DIM) vec[d] = v;
    __syncthreads();
    if (!NOPE[layer] && (is_q || is_k) && d < ROT) {
        int fi = d < ROT_HALF ? d : d - ROT_HALF;
        float angle = rope_position(mm_meta, mm_table, mm_stride, slot, pos, fi) * rope_inv_freq(fi, layer);
        float c = cosf(angle) * ROPE_MSCALE;
        float s = sinf(angle) * ROPE_MSCALE;
        v = d < ROT_HALF ? vec[d] * c - vec[d + ROT_HALF] * s : vec[d] * c + vec[d - ROT_HALF] * s;
    }
    float row_scale = 1.0f;
    if (kv_q(layer) && !is_q) {
        float amax = block_reduce<true>(fabsf(v), scratch);
        row_scale = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    }
    if (d >= HEAD_DIM) return;
    if (is_q) {
        src[d] = v;
        return;
    }
    int kv_head = is_k ? head - HEADS : head - HEADS - KV_HEADS;
    bf16_t* dst = kv_row(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos);
    if (kv_q(layer)) {
        ((unsigned char*)dst)[d] = (unsigned char)(f32x2_to_e4m3x2(v / row_scale, 0.0f) & 0xffu);
        if (d == 0) *kv_scale_ptr(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos) = row_scale;
        return;
    }
    unsigned hi, lo;
    attn_split(v, hi, lo);
    dst[d] = (bf16_t)hi;
    if (KV_PLANES == 2) dst[HEAD_DIM + d] = (bf16_t)lo;
}


constexpr int ATTN_SPLITS = 64;
constexpr int ATTN_TILE = 16;
constexpr int ATTN_MIN_CHUNK = 16;
constexpr int ATTN_BLOCK = 128;
constexpr int ATTN_V4 = (HEAD_DIM / 4 + 31) / 32;
constexpr int ATTN_PART = HEAD_DIM + 2;

__device__ __forceinline__ int attn_chunk(int len) {
    return max(ATTN_MIN_CHUNK, (len + ATTN_SPLITS - 1) / ATTN_SPLITS);
}

constexpr int ATTN_WARPS = ATTN_BLOCK / 32;
constexpr int ATTN_UNROLL = 4;

extern "C" __global__ void __launch_bounds__(ATTN_BLOCK) attention_split(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                     const int* block_table, const int* pos_ptr, int layer, float* part, int window) {
    __shared__ float4 acc_s[ATTN_WARPS][HEAD_DIM / 4];
    __shared__ float m_s[ATTN_WARPS][GROUP];
    __shared__ float l_s[ATTN_WARPS][GROUP];
    int kv_head = blockIdx.x;
    int split = blockIdx.y;
    int len = pos_ptr[0] + 1;
    int lo = window_lo(len, window);
    int chunk = attn_chunk(len - lo);
    int start = lo + split * chunk;
    if (start >= len) return;
    int end = min(len, start + chunk);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    static_assert(HEAD_DIM % 4 == 0 && HEAD_DIM <= 256, "unsupported head_dim");
    float scale = QUERY_SCALE;
    float4 q[GROUP][ATTN_V4];
#pragma unroll
    for (int g = 0; g < GROUP; g++)
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++) {
            int c = lane + 32 * j;
            q[g][j] = c < HEAD_DIM / 4 ? *(const float4*)(qkv + (kv_head * GROUP + g) * HEAD_DIM + 4 * c)
                                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    float m[GROUP], l[GROUP];
    float4 acc[GROUP][ATTN_V4];
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        m[g] = NEG_INF;
        l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++) acc[g][j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int tb = start + warp; tb < end; tb += ATTN_WARPS * ATTN_UNROLL) {
        float4 kr[ATTN_UNROLL][ATTN_V4], vr[ATTN_UNROLL][ATTN_V4];
#pragma unroll
        for (int u = 0; u < ATTN_UNROLL; u++) {
            int t = min(tb + u * ATTN_WARPS, end - 1);
            const bf16_t* k = kv_row(k_cache, block_table, layer, kv_head, t);
            const bf16_t* v = kv_row(v_cache, block_table, layer, kv_head, t);
            float ksc = kv_scale(k_cache, block_table, layer, kv_head, t);
            float vsc = kv_scale(v_cache, block_table, layer, kv_head, t);
#pragma unroll
            for (int j = 0; j < ATTN_V4; j++) {
                int c = lane + 32 * j;
                bool on = c < HEAD_DIM / 4;
                kr[u][j] = on ? kv_at4(k, 4 * c, ksc, kv_q(layer)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vr[u][j] = on ? kv_at4(v, 4 * c, vsc, kv_q(layer)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNROLL; u++) {
            if (tb + u * ATTN_WARPS >= end) break;
#pragma unroll
            for (int g = 0; g < GROUP; g++) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < ATTN_V4; j++)
                    dot += q[g][j].x * kr[u][j].x + q[g][j].y * kr[u][j].y + q[g][j].z * kr[u][j].z + q[g][j].w * kr[u][j].w;
                dot = cap_score(warp_sum(dot) * scale);
                float nm = fmaxf(m[g], dot);
                float corr = expf(m[g] - nm);
                float p = expf(dot - nm);
                l[g] = l[g] * corr + p;
#pragma unroll
                for (int j = 0; j < ATTN_V4; j++) {
                    acc[g][j].x = acc[g][j].x * corr + p * vr[u][j].x;
                    acc[g][j].y = acc[g][j].y * corr + p * vr[u][j].y;
                    acc[g][j].z = acc[g][j].z * corr + p * vr[u][j].z;
                    acc[g][j].w = acc[g][j].w * corr + p * vr[u][j].w;
                }
                m[g] = nm;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        if (lane == 0) {
            m_s[warp][g] = m[g];
            l_s[warp][g] = l[g];
        }
#pragma unroll
        for (int j = 0; j < ATTN_V4; j++)
            if (lane + 32 * j < HEAD_DIM / 4) acc_s[warp][lane + 32 * j] = acc[g][j];
        __syncthreads();
        for (int d = threadIdx.x; d < HEAD_DIM; d += ATTN_BLOCK) {
            float M = NEG_INF;
            for (int w = 0; w < ATTN_WARPS; w++) M = fmaxf(M, m_s[w][g]);
            float L = 0.0f, A = 0.0f;
            for (int w = 0; w < ATTN_WARPS; w++) {
                float c = m_s[w][g] == NEG_INF ? 0.0f : expf(m_s[w][g] - M);
                L += l_s[w][g] * c;
                A += ((const float*)acc_s[w])[d] * c;
            }
            float* dst = part + ((size_t)(kv_head * GROUP + g) * ATTN_SPLITS + split) * ATTN_PART;
            dst[d] = A;
            if (d == 0) {
                dst[HEAD_DIM] = M;
                dst[HEAD_DIM + 1] = L;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void attention_reduce(const float* part, const int* pos_ptr, const float* qkv, float* out, int window) {
    int head = blockIdx.x;
    int len = pos_ptr[0] + 1 - window_lo(pos_ptr[0] + 1, window);
    int chunk = attn_chunk(len);
    int active = (len + chunk - 1) / chunk;
    const float* base = part + (size_t)head * ATTN_SPLITS * ATTN_PART;
    __shared__ float w_s[ATTN_SPLITS];
    int d = threadIdx.x;
    float ms = d < active ? base[d * ATTN_PART + HEAD_DIM] : NEG_INF;
    float ls = d < active ? base[d * ATTN_PART + HEAD_DIM + 1] : 0.0f;
    float m = warp_max(ms);
    if (ATTN_SPLITS > 32) {
        __shared__ float wm[ATTN_SPLITS / 32 + 1];
        if ((d & 31) == 0 && d < ATTN_SPLITS) wm[d >> 5] = m;
        __syncthreads();
        m = NEG_INF;
        for (int i = 0; i <= (ATTN_SPLITS - 1) / 32; i++) m = fmaxf(m, wm[i]);
    }
    float w = d < active ? expf(ms - m) : 0.0f;
    if (d < ATTN_SPLITS) w_s[d] = w;
    __syncthreads();
    float l = 0.0f;
    for (int i = 0; i < active; i++) l += w_s[i] * base[i * ATTN_PART + HEAD_DIM + 1];
    float total = 0.0f;
    if (d < HEAD_DIM) {
#pragma unroll 8
        for (int i = 0; i < active; i++) total += w_s[i] * base[i * ATTN_PART + d];
        out[head * HEAD_DIM + d] = total / l * out_gate(qkv, 0, head * HEAD_DIM + d);
    }
}
