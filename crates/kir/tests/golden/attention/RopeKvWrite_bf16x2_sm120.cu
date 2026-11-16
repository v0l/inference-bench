// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// attention RopeKvWrite threads=256 smem=0
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


extern "C" __global__ void qk_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos, const int* slots, int layer, int per_row, int pages, int mm_stride,
                                        const int* mm_meta, const int* mm_table, int slot_base) {
    int row = blockIdx.y;
    const int* table = per_row ? seq_table(block_table, slots, row, pages) : block_table;
    int p = per_row ? pos[row] : pos[0] + row;
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, table, p, layer, blockIdx.x, row, mm_meta, mm_table, mm_stride,
                   per_row ? slots[row] : slot_base);
}

extern "C" __global__ void kv_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer,
                                        const int* mm_meta, const int* mm_table, int mm_stride, int slot) {
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, block_table, pos_ptr[0] + blockIdx.y, layer,
                   HEADS + blockIdx.x, blockIdx.y, mm_meta, mm_table, mm_stride, slot);
}
