// target Cuda cc=12.0 sms=188 warp=32 threads=1024 smem=101376 graphs=true pdl=true features=[WarpShuffle,CpAsync,Mma16816,Tma,ShaderF16,Fp8Mma,BlockScaledFp4]
// attention GroupedDecodeMma threads=64 smem=32768
//   AttentionShape { heads: 24, kv_heads: 4, head_dim: 256, qkv_row: 14336, output_gate: true, qk_norm: true, eps: 1e-6, rope: RopeShape { rotary_dim: 64, theta: 10000000.0, kind: 0, factor: 1.0, low: 1.0, high: 1.0, original: 1.0, mscale: 1.0, mrope: Some(MropeShape { section: [11, 11, 10], interleaved: true }) }, kv_layers: 16, kv_planes: 1, kv_fp8: false, kv_f16_layers: 0, query_scale: 0.0625, softcap: 30.0, local_theta: 10000.0, local_rope: [false, true, false, false], bidirectional: false, nope: [false, false, true, false] }
constexpr int ROT = 64;
constexpr int ROT_HALF = ROT / 2;
constexpr bool QK_NORM = true;
constexpr bool LOCAL_ROPE[] = { false, true, false, false };
constexpr bool NOPE[] = { false, false, true, false };
constexpr float LOCAL_THETA = 1e4f;
constexpr float QUERY_SCALE = 6.25e-2f;
constexpr bool BIDI = false;
constexpr float SOFTCAP = 3e1f;
constexpr float THETA = 1e7f;
constexpr int ROPE_KIND = 0;
constexpr bool MROPE = true;
constexpr bool MROPE_INTERLEAVED = true;
constexpr int MROPE_S0 = 11;
constexpr int MROPE_S1 = 11;
constexpr int MROPE_S2 = 10;
constexpr float ROPE_FACTOR = 1e0f;
constexpr float ROPE_LOW_FREQ = 1e0f;
constexpr float ROPE_HIGH_FREQ = 1e0f;
constexpr float ROPE_ORIG_POS = 1e0f;
constexpr float ROPE_MSCALE = 1e0f;
constexpr int PAGE = 16;
constexpr int ATTN_LAYERS = 16;
constexpr int KV_PLANES = 1;
constexpr bool KV_FP8 = false;
constexpr int KV_F16_LAYERS = 0;
constexpr int DM_BK = 16;
constexpr int DM_STAGES = 2;
constexpr int DM_CHUNK = 256;
constexpr bool DM_SPLIT = true;
constexpr int DM_MAX_WARPS = 2;

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


__device__ __forceinline__ unsigned pack_hi(float a, float b) {
    return attn_bits(a) | (attn_bits(b) << 16);
}

__device__ __forceinline__ unsigned pack_lo(float a, float b) {
    return attn_bits(a - attn_float(attn_bits(a))) | (attn_bits(b - attn_float(attn_bits(b))) << 16);
}

__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma_hi_lo(float* c, const unsigned* ah, const unsigned* al,
                                          unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma_f16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

__device__ __forceinline__ void mma_kv(float* c, const unsigned* ah, const unsigned* al,
                                       unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    if (KV_PLANES == 2) {
        mma_hi_lo(c, ah, al, bh0, bh1, bl0, bl1);
        return;
    }
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

__device__ __forceinline__ int group_len(const int* pos, int group, int qlen) {
    int len = 0;
    for (int j = 0; j < qlen; j++) len = max(len, pos[group * qlen + j] + 1);
    return len;
}

__device__ __forceinline__ int decode_chunk(int len, int splits) {
    int c = DM_CHUNK;
    while (c * splits < len) c <<= 1;
    return c;
}

__device__ __forceinline__ void mma_dec(float* c, const unsigned* ah, const unsigned* al,
                                        unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    if (KV_PLANES == 2) {
        mma_hi_lo(c, ah, al, bh0, bh1, bl0, bl1);
        return;
    }
    if (DM_SPLIT) mma_f16(c, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_f16(c, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}


extern "C" __global__ void batch_decode_combine(const float* part_acc, const float* part_ml, const float* qkv, const int* pos, int qlen, int splits,
                                                bf16_t* out_hi, bf16_t* out_lo) {
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int d = 2 * threadIdx.x;
    int len = group_len(pos, seq / qlen, qlen);
    int active = (len + decode_chunk(len, splits) - 1) / decode_chunk(len, splits);
    size_t base = ((size_t)seq * HEADS + head) * splits;
    float M = NEG_INF;
    for (int s = 0; s < active; s++) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, A0 = 0.0f, A1 = 0.0f;
    for (int s = 0; s < active; s++) {
        float ms = part_ml[(base + s) * 2];
        float c = ms == NEG_INF ? 0.0f : exp2f(ms - M);
        L += part_ml[(base + s) * 2 + 1] * c;
        A0 += part_acc[(base + s) * HEAD_DIM + d] * c;
        A1 += part_acc[(base + s) * HEAD_DIM + d + 1] * c;
    }
    int at = head * HEAD_DIM + d;
    split_store2(out_hi, out_lo, (size_t)seq * Q_DIM + at, A0 / L * out_gate(qkv, seq, at), A1 / L * out_gate(qkv, seq, at + 1));
}

extern "C" __global__ void __launch_bounds__(32 * DM_MAX_WARPS) batch_decode_attention(
    const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache, const int* block_table, const int* pos, const int* slots,
    int layer, int qlen, int pages, int splits, float* part_acc, float* part_ml, int window) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = DM_BK * HEAD_DIM;
    constexpr int KB = HEAD_DIM / 16;
    constexpr int NB = HEAD_DIM / 8;
    constexpr int SB = DM_BK / 8;
    extern __shared__ __align__(16) unsigned char dm_smem[];
    bf16_t* kvs = (bf16_t*)dm_smem;
    int kv_head = blockIdx.x;
    int group = blockIdx.y;
    int z = blockIdx.z;
    int len = group_len(pos, group, qlen);
    int chunk = decode_chunk(len, splits);
    int key_begin = z * chunk;
    if (key_begin >= len) return;
    int key_end = min(len, key_begin + chunk);
    int mine = (key_end - key_begin + DM_BK - 1) / DM_BK;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int rows = GROUP * qlen;
    int r0 = warp * 16 + g;
    int r1 = r0 + 8;
    int row0 = group * qlen + min(r0, rows - 1) / GROUP;
    int row1 = group * qlen + min(r1, rows - 1) / GROUP;
    int head0 = kv_head * GROUP + r0 % GROUP;
    int head1 = kv_head * GROUP + r1 % GROUP;
    int pos0 = r0 < rows ? pos[row0] : -1;
    int pos1 = r1 < rows ? pos[row1] : -1;
    const int* table = seq_table(block_table, slots, group * qlen, pages);
    float scale_log2 = QUERY_SCALE * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;
    unsigned qh[KB][4];
    unsigned ql[KB][4];
    const float* q0 = qkv + (size_t)row0 * QKV_ROW + head0 * HEAD_DIM;
    const float* q1 = qkv + (size_t)row1 * QKV_ROW + head1 * HEAD_DIM;
#pragma unroll
    for (int kb = 0; kb < KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 a = pos0 >= 0 ? *(const float2*)&q0[d] : make_float2(0.0f, 0.0f);
        float2 b = pos1 >= 0 ? *(const float2*)&q1[d] : make_float2(0.0f, 0.0f);
        float2 c = pos0 >= 0 ? *(const float2*)&q0[d + 8] : make_float2(0.0f, 0.0f);
        float2 e = pos1 >= 0 ? *(const float2*)&q1[d + 8] : make_float2(0.0f, 0.0f);
        qh[kb][0] = pack_hi(a.x, a.y);
        qh[kb][1] = pack_hi(b.x, b.y);
        qh[kb][2] = pack_hi(c.x, c.y);
        qh[kb][3] = pack_hi(e.x, e.y);
        if (DM_SPLIT || KV_PLANES == 2) {
            ql[kb][0] = pack_lo(a.x, a.y);
            ql[kb][1] = pack_lo(b.x, b.y);
            ql[kb][2] = pack_lo(c.x, c.y);
            ql[kb][3] = pack_lo(e.x, e.y);
        }
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = kvs + buf * 2 * KV_PLANES * PLANE;
        constexpr int ROW_CHUNKS = KV_PLANES * CPR;
        constexpr int TOTAL = 2 * DM_BK * ROW_CHUNKS;
        for (int i = tid; i < TOTAL; i += blockDim.x) {
            int which = i / (DM_BK * ROW_CHUNKS);
            int rem = i % (DM_BK * ROW_CHUNKS);
            int r = rem / ROW_CHUNKS;
            int c = rem % ROW_CHUNKS;
            int tk = k0 + r;
            int at = min(tk, len - 1);
            const bf16_t* src = kv_row(which ? v_cache : k_cache, table, layer, kv_head, at);
            bf16_t* dst = st + (which * KV_PLANES + c / CPR) * PLANE + swz<CPR>(r, c % CPR) * 8;
            if (kv_q(layer)) {
                float sc = kv_scale(which ? v_cache : k_cache, table, layer, kv_head, at);
                uint2 raw = tk < len ? *(const uint2*)((const unsigned char*)src + c * 8) : make_uint2(0, 0);
                unsigned w[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    unsigned two = ((e < 2 ? raw.x : raw.y) >> (16 * (e & 1))) & 0xffffu;
                    w[e] = attn_bits(e4m3_to_f32(two & 0xffu) * sc) | (attn_bits(e4m3_to_f32(two >> 8) * sc) << 16);
                }
                *(uint4*)dst = make_uint4(w[0], w[1], w[2], w[3]);
                continue;
            }
            cp_async16_zfill(dst, src + c * 8, tk < len);
        }
    };
#pragma unroll
    for (int s = 0; s < DM_STAGES - 1; s++) {
        if (s < mine) load_kv(s, key_begin + s * DM_BK);
        cp_async_commit();
    }
    for (int i = 0; i < mine; i++) {
        int ahead = i + DM_STAGES - 1;
        if (ahead < mine) load_kv(ahead % DM_STAGES, key_begin + ahead * DM_BK);
        cp_async_commit();
        cp_async_wait<DM_STAGES - 1>();
        __syncthreads();
        int k0 = key_begin + i * DM_BK;
        const bf16_t* k_hi = kvs + (i % DM_STAGES) * 2 * KV_PLANES * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_hi + KV_PLANES * PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < KB; kb++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = swz<CPR>(kr, kb * 2 + ((lane >> 3) & 1)) * 8;
                ldsm_x4(bh, k_hi + at);
                if (KV_PLANES == 2) ldsm_x4(bl, k_lo + at);
                mma_dec(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                mma_dec(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
            }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int kpos = e < 2 ? pos0 : pos1;
                bool live = key < key_end && key <= kpos && (window <= 0 || key > kpos - window);
                float v = live ? (SOFTCAP > 0.0f ? cap_score(s[nb][e] * (scale_log2 / LOG2E)) * LOG2E : s[nb][e] * scale_log2) : NEG_INF;
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float n0 = fmaxf(m0, mx0);
        float n1 = fmaxf(m1, mx1);
        float a0 = m0 == NEG_INF ? 0.0f : fast_exp2(m0 - n0);
        float a1 = m1 == NEG_INF ? 0.0f : fast_exp2(m1 - n1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float nm = e < 2 ? n0 : n1;
                float p = s[nb][e] == NEG_INF ? 0.0f : fast_exp2(s[nb][e] - nm);
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            o[nb][0] *= a0;
            o[nb][1] *= a0;
            o[nb][2] *= a1;
            o[nb][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            if (DM_SPLIT || KV_PLANES == 2) {
                pl[0] = pack_lo(s[2 * kb][0], s[2 * kb][1]);
                pl[1] = pack_lo(s[2 * kb][2], s[2 * kb][3]);
                pl[2] = pack_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
                pl[3] = pack_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            }
#pragma unroll
            for (int np = 0; np < NB / 2; np++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = swz<CPR>(kr, np * 2 + (lane >> 4)) * 8;
                ldsm_x4_t(bh, v_hi + at);
                if (KV_PLANES == 2) ldsm_x4_t(bl, v_lo + at);
                mma_dec(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                mma_dec(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
            }
        }
        __syncthreads();
    }
    size_t base0 = ((size_t)row0 * HEADS + head0) * splits + z;
    size_t base1 = ((size_t)row1 * HEADS + head1) * splits + z;
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (pos0 >= 0) *(float2*)&part_acc[base0 * HEAD_DIM + d] = make_float2(o[nb][0], o[nb][1]);
        if (pos1 >= 0) *(float2*)&part_acc[base1 * HEAD_DIM + d] = make_float2(o[nb][2], o[nb][3]);
    }
    if (t == 0) {
        if (pos0 >= 0) *(float2*)&part_ml[base0 * 2] = make_float2(m0, l0);
        if (pos1 >= 0) *(float2*)&part_ml[base1 * 2] = make_float2(m1, l1);
    }
}
