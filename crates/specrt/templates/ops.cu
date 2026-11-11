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
constexpr int QKV_ROW = {{QKV_ROW}};
constexpr int GATE_OFF = Q_DIM + 2 * KV_DIM;
constexpr bool OUT_GATE = {{OUT_GATE}};
constexpr int ROT = {{ROT}};
constexpr int ROT_HALF = ROT / 2;
constexpr bool QK_NORM = {{QK_NORM}};
constexpr float EPS = {{EPS}};
constexpr float THETA = {{THETA}};
constexpr int ROPE_KIND = {{ROPE_KIND}};
constexpr bool MROPE = {{MROPE}};
constexpr bool MROPE_INTERLEAVED = {{MROPE_INTERLEAVED}};
constexpr int MROPE_S0 = {{MROPE_S0}};
constexpr int MROPE_S1 = {{MROPE_S1}};
constexpr int MROPE_S2 = {{MROPE_S2}};
constexpr float ROPE_FACTOR = {{ROPE_FACTOR}};
constexpr float ROPE_LOW_FREQ = {{ROPE_LOW_FREQ}};
constexpr float ROPE_HIGH_FREQ = {{ROPE_HIGH_FREQ}};
constexpr float ROPE_ORIG_POS = {{ROPE_ORIG_POS}};
constexpr int ACT_KIND = {{ACT_KIND}};
constexpr int MAX_CTX = {{MAX_CTX}};
constexpr int PAGE = {{PAGE}};
constexpr int PAGES = (MAX_CTX + PAGE - 1) / PAGE;
constexpr int ATTN_LAYERS = {{ATTN_LAYERS}};
constexpr int KV_PLANES = {{KV_PLANES}};
constexpr int Q8_GROUP = {{Q8_GROUP}};
constexpr int FP8_BLOCK = {{FP8_BLOCK}};
constexpr int LINEAR_WARPS = 8;
constexpr int ATTN_THREADS = 512;
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float bf16_to_f32(bf16_t v) {
    return __uint_as_float(((unsigned)v) << 16);
}

__device__ __forceinline__ float e4m3_to_f32(unsigned b) {
    return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * __uint_as_float(0x7b800000u);
}

__device__ __forceinline__ unsigned e4m3_pair_bf16(unsigned two_bytes) {
    return (__float_as_uint(e4m3_to_f32(two_bytes & 0xffu)) >> 16) |
           (__float_as_uint(e4m3_to_f32((two_bytes >> 8) & 0xffu)) & 0xffff0000u);
}

__device__ __forceinline__ float e4m3_dot4(unsigned word, float4 x) {
    return e4m3_to_f32(word & 0xffu) * x.x + e4m3_to_f32((word >> 8) & 0xffu) * x.y +
           e4m3_to_f32((word >> 16) & 0xffu) * x.z + e4m3_to_f32(word >> 24) * x.w;
}

__device__ __forceinline__ float out_gate(const float* qkv, size_t row, int at) {
    if (!OUT_GATE) return 1.0f;
    return 1.0f / (1.0f + expf(-qkv[row * QKV_ROW + GATE_OFF + at]));
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

__device__ __forceinline__ uint4 load_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ uint2 load_stream8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}

template <int K>
__device__ __forceinline__ float dot_bf16(const void* w, int row, const float* x, int lane) {
    const bf16_t* wr = (const bf16_t*)w + (size_t)row * K;
    float acc = 0.0f;
#pragma unroll 8
    for (int k = lane * 8; k < K; k += 256) {
        uint4 packed = load_stream(wr + k);
        float4 xa = *(const float4*)(x + k);
        float4 xb = *(const float4*)(x + k + 4);
        acc += __uint_as_float(packed.x << 16) * xa.x + __uint_as_float(packed.x & 0xffff0000u) * xa.y;
        acc += __uint_as_float(packed.y << 16) * xa.z + __uint_as_float(packed.y & 0xffff0000u) * xa.w;
        acc += __uint_as_float(packed.z << 16) * xb.x + __uint_as_float(packed.z & 0xffff0000u) * xb.y;
        acc += __uint_as_float(packed.w << 16) * xb.z + __uint_as_float(packed.w & 0xffff0000u) * xb.w;
    }
    return acc;
}

template <int K>
__device__ __forceinline__ float dot_q8(const void* w, const float* scales, int row, const float* x, int lane) {
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
    return acc;
}

__device__ __forceinline__ const float* fp8_row_scales(const float* scales, int row) {
    return scales + __float_as_uint(__ldg(scales + row));
}

constexpr int GG_Q4K = 0;
constexpr int GG_Q5K = 1;
constexpr int GG_Q6K = 2;
constexpr int GG_Q8 = 3;
constexpr int GG_BF16 = 4;
constexpr int GG_SB = 256;

__device__ __forceinline__ float gg_f16(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ unsigned gg_u16(const unsigned char* p) {
    return __ldg((const unsigned short*)p);
}

__device__ __forceinline__ unsigned gg_u32(const unsigned char* p) {
    return __ldg((const unsigned*)p);
}

__device__ __forceinline__ int gg_bytes(int type) {
    switch (type) {
        case GG_Q4K: return 144;
        case GG_Q5K: return 176;
        case GG_Q6K: return 224;
        case GG_Q8: return 272;
        default: return 512;
    }
}

__device__ __forceinline__ unsigned gg_byte(const unsigned* s, int i) {
    return (s[i >> 2] >> (8 * (i & 3))) & 0xffu;
}

__device__ __forceinline__ void gg_scale_min(const unsigned* s, int j, float& sc, float& m) {
    unsigned a, b;
    if (j < 4) {
        a = gg_byte(s, j) & 63u;
        b = gg_byte(s, j + 4) & 63u;
    } else {
        a = (gg_byte(s, j + 4) & 0xfu) | ((gg_byte(s, j - 4) >> 6) << 4);
        b = (gg_byte(s, j + 4) >> 4) | ((gg_byte(s, j) >> 6) << 4);
    }
    sc = (float)a;
    m = (float)b;
}

__device__ __forceinline__ void gg_dequant8(const unsigned char* sb, int type, int c, float* w) {
    if (type == GG_Q4K || type == GG_Q5K) {
        float d = gg_f16(gg_u16(sb));
        float dmin = gg_f16(gg_u16(sb + 2));
        unsigned s[3] = {gg_u32(sb + 4), gg_u32(sb + 8), gg_u32(sb + 12)};
        int j = c >> 3;
        int half = (c >> 2) & 1;
        int l = (c & 3) * 8;
        float sc, m;
        gg_scale_min(s, 2 * j + half, sc, m);
        float dl = __fmul_rn(d, sc), ml = __fmul_rn(dmin, m);
        const unsigned char* qp = sb + (type == GG_Q4K ? 16 : 48) + 32 * j + l;
        unsigned q[2] = {gg_u32(qp), gg_u32(qp + 4)};
        unsigned qh[2] = {0u, 0u};
        if (type == GG_Q5K) {
            qh[0] = gg_u32(sb + 16 + l);
            qh[1] = gg_u32(sb + 20 + l);
        }
        int bit = 2 * j + half;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned byte = (q[i >> 2] >> (8 * (i & 3))) & 0xffu;
            unsigned nib = half ? byte >> 4 : byte & 15u;
            unsigned hb = ((qh[i >> 2] >> (8 * (i & 3))) >> bit) & 1u;
            w[i] = __fsub_rn(__fmul_rn(dl, (float)(nib + 16u * hb)), ml);
        }
    } else if (type == GG_Q6K) {
        int h = c >> 4;
        int k = (c >> 2) & 3;
        int l = (c & 3) * 8;
        float d = gg_f16(gg_u16(sb + 208));
        float ds = __fmul_rn(d, (float)(signed char)__ldg(sb + 192 + 8 * h + l / 16 + 2 * k));
        const unsigned char* qlp = sb + 64 * h + (k & 1) * 32 + l;
        const unsigned char* qhp = sb + 128 + 32 * h + l;
        unsigned ql[2] = {gg_u32(qlp), gg_u32(qlp + 4)};
        unsigned qh[2] = {gg_u32(qhp), gg_u32(qhp + 4)};
        int ls = (k >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned lo = ((ql[i >> 2] >> (8 * (i & 3) + ls)) & 15u);
            unsigned hi = ((qh[i >> 2] >> (8 * (i & 3) + 2 * k)) & 3u);
            w[i] = __fmul_rn(ds, (float)((int)(lo | (hi << 4)) - 32));
        }
    } else if (type == GG_Q8) {
        const unsigned char* bp = sb + 34 * (c >> 2);
        float d = gg_f16(gg_u16(bp));
        int e = (c & 3) * 8;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned two = gg_u16(bp + 2 + e + 2 * k);
            w[2 * k] = __fmul_rn((float)(signed char)(two & 0xffu), d);
            w[2 * k + 1] = __fmul_rn((float)(signed char)(two >> 8), d);
        }
    } else {
        uint4 v = __ldg((const uint4*)(sb + 16 * c));
        unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (type == GG_BF16) {
                w[2 * k] = __uint_as_float(words[k] << 16);
                w[2 * k + 1] = __uint_as_float(words[k] & 0xffff0000u);
            } else {
                w[2 * k] = gg_f16((unsigned short)(words[k] & 0xffffu));
                w[2 * k + 1] = gg_f16((unsigned short)(words[k] >> 16));
            }
        }
    }
}

__device__ __forceinline__ const unsigned char* gg_row(const void* w, const unsigned* table, int row, int& type) {
    unsigned e = __ldg(table + row);
    type = (int)(e & 7u);
    return (const unsigned char*)w + (size_t)(e >> 3) * 16;
}

__device__ __forceinline__ void gg_load8(const void* w, const unsigned* table, int row, int k, float* out) {
    int type;
    const unsigned char* base = gg_row(w, table, row, type);
    gg_dequant8(base + (size_t)(k / GG_SB) * gg_bytes(type), type, (k % GG_SB) / 8, out);
}

template <int K, int R>
__device__ __forceinline__ void gg_q4k_rows(const unsigned char* const* base, const float* x, int lane, float* out) {
    constexpr int NSB = K / GG_SB;
    int j = (lane & 7) >> 1;
    int l0 = (lane & 1) * 16;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
#pragma unroll 2
    for (int s0 = 0; s0 < NSB; s0 += 4) {
        int s = s0 + (lane >> 3);
        bool valid = s < NSB;
        int sl = valid ? s : NSB - 1;
        uint4 head[R], q[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned char* sb = base[r] + (size_t)sl * 144;
            head[r] = load_stream(sb);
            q[r] = load_stream(sb + 16 + 32 * j + l0);
        }
        const float* xs = x + sl * GG_SB + 64 * j + l0;
        float4 a[4], b[4];
        float slo = 0.0f, shi = 0.0f;
#pragma unroll
        for (int w4 = 0; w4 < 4; w4++) {
            a[w4] = *(const float4*)(xs + 4 * w4);
            b[w4] = *(const float4*)(xs + 32 + 4 * w4);
            slo += a[w4].x + a[w4].y + a[w4].z + a[w4].w;
            shi += b[w4].x + b[w4].y + b[w4].z + b[w4].w;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            float d = gg_f16((unsigned short)(head[r].x & 0xffffu));
            float dmin = gg_f16((unsigned short)(head[r].x >> 16));
            unsigned scales[3] = {head[r].y, head[r].z, head[r].w};
            float sc1, m1, sc2, m2;
            gg_scale_min(scales, 2 * j, sc1, m1);
            gg_scale_min(scales, 2 * j + 1, sc2, m2);
            unsigned words[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            float lo = 0.0f, hi = 0.0f;
#pragma unroll
            for (int w4 = 0; w4 < 4; w4++) {
                unsigned v = words[w4];
                lo += (float)(v & 15u) * a[w4].x + (float)((v >> 8) & 15u) * a[w4].y + (float)((v >> 16) & 15u) * a[w4].z + (float)((v >> 24) & 15u) * a[w4].w;
                hi += (float)((v >> 4) & 15u) * b[w4].x + (float)((v >> 12) & 15u) * b[w4].y + (float)((v >> 20) & 15u) * b[w4].z + (float)(v >> 28) * b[w4].w;
            }
            float part = d * (sc1 * lo + sc2 * hi) - dmin * (m1 * slo + m2 * shi);
            acc[r] += valid ? part : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) out[r] = acc[r];
}

template <int K, int R>
__device__ __forceinline__ void gg_q6k_rows(const unsigned char* const* base, const float* x, int lane, float* out) {
    constexpr int NSB = K / GG_SB;
    int h = (lane & 7) >> 2;
    int l0 = (lane & 3) * 8;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
#pragma unroll 2
    for (int s0 = 0; s0 < NSB; s0 += 4) {
        int s = s0 + (lane >> 3);
        bool valid = s < NSB;
        int sl = valid ? s : NSB - 1;
        uint2 qa[R], qb[R], qc[R];
        unsigned sc[R][2];
        unsigned short dh[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const unsigned char* sb = base[r] + (size_t)sl * 224;
            qa[r] = load_stream8(sb + 64 * h + l0);
            qb[r] = load_stream8(sb + 64 * h + 32 + l0);
            qc[r] = load_stream8(sb + 128 + 32 * h + l0);
            unsigned w = __ldg((const unsigned*)(sb + 192 + 8 * h));
            unsigned w2 = __ldg((const unsigned*)(sb + 196 + 8 * h));
            sc[r][0] = w;
            sc[r][1] = w2;
            dh[r] = __ldg((const unsigned short*)(sb + 208));
        }
        const float* xs = x + sl * GG_SB + 128 * h + l0;
        float xv[4][8];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float4 u = *(const float4*)(xs + 32 * k);
            float4 v = *(const float4*)(xs + 32 * k + 4);
            xv[k][0] = u.x; xv[k][1] = u.y; xv[k][2] = u.z; xv[k][3] = u.w;
            xv[k][4] = v.x; xv[k][5] = v.y; xv[k][6] = v.z; xv[k][7] = v.w;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            unsigned a[2] = {qa[r].x, qa[r].y}, b[2] = {qb[r].x, qb[r].y}, c[2] = {qc[r].x, qc[r].y};
            float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned av = (a[i >> 2] >> (8 * (i & 3))) & 0xffu;
                unsigned bv = (b[i >> 2] >> (8 * (i & 3))) & 0xffu;
                unsigned cv = (c[i >> 2] >> (8 * (i & 3))) & 0xffu;
                part[0] += (float)((int)((av & 15u) | ((cv & 3u) << 4)) - 32) * xv[0][i];
                part[1] += (float)((int)((bv & 15u) | (((cv >> 2) & 3u) << 4)) - 32) * xv[1][i];
                part[2] += (float)((int)((av >> 4) | (((cv >> 4) & 3u) << 4)) - 32) * xv[2][i];
                part[3] += (float)((int)((bv >> 4) | (((cv >> 6) & 3u) << 4)) - 32) * xv[3][i];
            }
            int sh = 8 * (l0 / 16);
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned word = sc[r][k >> 1];
                sum += (float)(signed char)((word >> (16 * (k & 1) + sh)) & 0xffu) * part[k];
            }
            float d = gg_f16(dh[r]);
            acc[r] += valid ? d * sum : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) out[r] = acc[r];
}

template <int K>
__device__ __forceinline__ float gg_q4k_dot(const unsigned char* base, const float* x, int lane) {
    int j = (lane & 7) >> 1;
    int l0 = (lane & 1) * 16;
    float acc = 0.0f;
#pragma unroll 4
    for (int s0 = 0; s0 < K / GG_SB; s0 += 4) {
        int s = s0 + (lane >> 3);
        if (s >= K / GG_SB) break;
        const unsigned char* sb = base + (size_t)s * 144;
        uint4 head = __ldg((const uint4*)sb);
        uint4 q = __ldg((const uint4*)(sb + 16 + 32 * j + l0));
        float d = gg_f16((unsigned short)(head.x & 0xffffu));
        float dmin = gg_f16((unsigned short)(head.x >> 16));
        unsigned scales[3] = {head.y, head.z, head.w};
        float sc1, m1, sc2, m2;
        gg_scale_min(scales, 2 * j, sc1, m1);
        gg_scale_min(scales, 2 * j + 1, sc2, m2);
        const float* xs = x + s * GG_SB + 64 * j + l0;
        unsigned words[4] = {q.x, q.y, q.z, q.w};
        float lo = 0.0f, hi = 0.0f, slo = 0.0f, shi = 0.0f;
#pragma unroll
        for (int w4 = 0; w4 < 4; w4++) {
            float4 a = *(const float4*)(xs + 4 * w4);
            float4 b = *(const float4*)(xs + 32 + 4 * w4);
            unsigned v = words[w4];
            lo += (float)(v & 15u) * a.x + (float)((v >> 8) & 15u) * a.y + (float)((v >> 16) & 15u) * a.z + (float)((v >> 24) & 15u) * a.w;
            hi += (float)((v >> 4) & 15u) * b.x + (float)((v >> 12) & 15u) * b.y + (float)((v >> 20) & 15u) * b.z + (float)(v >> 28) * b.w;
            slo += a.x + a.y + a.z + a.w;
            shi += b.x + b.y + b.z + b.w;
        }
        acc += d * (sc1 * lo + sc2 * hi) - dmin * (m1 * slo + m2 * shi);
    }
    return acc;
}

template <int K>
__device__ __forceinline__ float gg_q6k_dot(const unsigned char* base, const float* x, int lane) {
    int h = (lane & 7) >> 2;
    int l0 = (lane & 3) * 8;
    float acc = 0.0f;
#pragma unroll 4
    for (int s0 = 0; s0 < K / GG_SB; s0 += 4) {
        int s = s0 + (lane >> 3);
        if (s >= K / GG_SB) break;
        const unsigned char* sb = base + (size_t)s * 224;
        uint2 qa = __ldg((const uint2*)(sb + 64 * h + l0));
        uint2 qb = __ldg((const uint2*)(sb + 64 * h + 32 + l0));
        uint2 qc = __ldg((const uint2*)(sb + 128 + 32 * h + l0));
        float d = gg_f16(gg_u16(sb + 208));
        const unsigned char* sc = sb + 192 + 8 * h + l0 / 16;
        const float* xs = x + s * GG_SB + 128 * h + l0;
        unsigned a[2] = {qa.x, qa.y}, b[2] = {qb.x, qb.y}, c[2] = {qc.x, qc.y};
        float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned av = (a[i >> 2] >> (8 * (i & 3))) & 0xffu;
            unsigned bv = (b[i >> 2] >> (8 * (i & 3))) & 0xffu;
            unsigned cv = (c[i >> 2] >> (8 * (i & 3))) & 0xffu;
            part[0] += (float)((int)((av & 15u) | ((cv & 3u) << 4)) - 32) * xs[i];
            part[1] += (float)((int)((bv & 15u) | (((cv >> 2) & 3u) << 4)) - 32) * xs[i + 32];
            part[2] += (float)((int)((av >> 4) | (((cv >> 4) & 3u) << 4)) - 32) * xs[i + 64];
            part[3] += (float)((int)((bv >> 4) | (((cv >> 6) & 3u) << 4)) - 32) * xs[i + 96];
        }
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; k++) sum += (float)(signed char)__ldg(sc + 2 * k) * part[k];
        acc += d * sum;
    }
    return acc;
}

template <int K>
__device__ __forceinline__ float gg_row_dot(const void* w, const unsigned* table, int row, const float* x, int lane) {
    int type;
    const unsigned char* base = gg_row(w, table, row, type);
    if (type == GG_Q4K) return gg_q4k_dot<K>(base, x, lane);
    if (type == GG_Q6K) return gg_q6k_dot<K>(base, x, lane);
    int bytes = gg_bytes(type);
    float acc = 0.0f;
    int c0 = (lane & 7) * 4;
#pragma unroll 4
    for (int s0 = 0; s0 < K / GG_SB; s0 += 4) {
        int s = s0 + (lane >> 3);
        if (s >= K / GG_SB) break;
        const unsigned char* sb = base + (size_t)s * bytes;
#pragma unroll
        for (int c = c0; c < c0 + 4; c++) {
            float wv[8];
            gg_dequant8(sb, type, c, wv);
            float4 xa = *(const float4*)(x + s * GG_SB + 8 * c);
            float4 xb = *(const float4*)(x + s * GG_SB + 8 * c + 4);
            acc += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w + wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
        }
    }
    return acc;
}

extern "C" __global__ void gg_dequant_rows(const void* w, const unsigned* table, float* out, int rows, int cols) {
    int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = cols / 8;
    if (chunk >= rows * per_row) return;
    int row = chunk / per_row;
    int k = (chunk % per_row) * 8;
    float v[8];
    gg_load8(w, table, row, k, v);
#pragma unroll
    for (int i = 0; i < 8; i++) out[(size_t)row * cols + k + i] = v[i];
}

template <int K>
__device__ __forceinline__ float dot_fp8(const void* w, const float* scales, int row, const float* x, int lane) {
    static_assert(K % 16 == 0, "fp8 rows load 16 bytes per lane");
    const unsigned char* wr = (const unsigned char*)w + (size_t)row * K;
    const float* sr = fp8_row_scales(scales, row);
    float acc = 0.0f;
#pragma unroll 2
    for (int k = lane * 16; k < K; k += 512) {
        uint4 packed = load_stream(wr + k);
        float part = e4m3_dot4(packed.x, *(const float4*)(x + k)) + e4m3_dot4(packed.y, *(const float4*)(x + k + 4)) +
                     e4m3_dot4(packed.z, *(const float4*)(x + k + 8)) + e4m3_dot4(packed.w, *(const float4*)(x + k + 12));
        acc += part * sr[k / FP8_BLOCK];
    }
    return acc;
}

__device__ __forceinline__ float activation(float g) {
    return ACT_KIND == 1
        ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
        : g / (1.0f + expf(-g));
}

template <typename D>
__device__ __forceinline__ void load_normed(const float* x, const float* nw, float* xs) {
    __shared__ float scratch[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[D::template x_at<HIDDEN>(i)] = nw[i] * (x[i] * inv);
    __syncthreads();
}

template <typename D, int N>
__host__ __device__ constexpr int gemv_rows() {
    return N % D::PAIR == 0 ? D::PAIR : 1;
}

template <typename D, int N, int K, bool BIAS, bool RESIDUAL, int ROWS = gemv_rows<D, N>()>
__device__ void linear_w(const void* w, const void* s, const float* bias, const float* x, float* out) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * D::WARPS + warp) * ROWS;
    do {
        if (row >= N) return;
        float acc[ROWS];
        D::template rows<K, ROWS>(w, s, row, 1, x, lane, acc);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float v = acc[r];
                if (BIAS) v += bias[row + r];
                out[row + r] = RESIDUAL ? out[row + r] + v : v;
            }
        }
        row += gridDim.x * D::WARPS * ROWS;
    } while (D::STRIDE);
}

template <typename D, int N, bool BIAS, int ROWS = gemv_rows<D, N>()>
__device__ void linear_norm(const void* w, const void* s, const float* bias, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed<D>(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * D::WARPS + warp) * ROWS;
    do {
        if (row >= N) return;
        float acc[ROWS];
        D::template rows<HIDDEN, ROWS>(w, s, row, 1, xs, lane, acc);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) out[row + r] = BIAS ? acc[r] + bias[row + r] : acc[r];
        }
        row += gridDim.x * D::WARPS * ROWS;
    } while (D::STRIDE);
}

template <typename D>
__device__ void gate_up_act_norm(const void* w, const void* s, const float* x, const float* nw, float* out) {
    __shared__ __align__(16) float xs[HIDDEN];
    load_normed<D>(x, nw, xs);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * D::WARPS + warp;
    do {
        if (row >= INTER) return;
        float gu[2];
        D::template rows<HIDDEN, 2>(w, s, 2 * row, 1, xs, lane, gu);
        if (lane == 0) out[D::template x_at<INTER>(row)] = activation(gu[0]) * gu[1];
        row += gridDim.x * D::WARPS;
    } while (D::STRIDE);
}

constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 16;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;

template <int N, int K, bool BIAS, bool RESIDUAL, typename D>
__device__ void gemm(const void* w, const void* scales, const float* bias, const float* x, float* out, int tokens) {
    __shared__ float xs[GEMM_BK][GEMM_BM];
    __shared__ float ws[GEMM_BK][GEMM_BN];
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    int m0 = blockIdx.y * GEMM_BM;
    int n0 = blockIdx.x * GEMM_BN;
    float acc[GEMM_TM][GEMM_TN];
#pragma unroll
    for (int i = 0; i < GEMM_TM; i++)
#pragma unroll
        for (int j = 0; j < GEMM_TN; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            int r = idx / GEMM_BK;
            int kk = idx % GEMM_BK;
            int m = m0 + r;
            xs[kk][r] = m < tokens ? x[(size_t)m * K + k0 + kk] : 0.0f;
            int n = n0 + r;
            float wv = 0.0f;
            if (n < N) wv = D::template at<K>(w, scales, n, k0 + kk);
            ws[kk][r] = wv;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GEMM_BK; kk++) {
            float a[GEMM_TM];
            float b[GEMM_TN];
#pragma unroll
            for (int i = 0; i < GEMM_TM; i++) a[i] = xs[kk][ty * GEMM_TM + i];
#pragma unroll
            for (int j = 0; j < GEMM_TN; j++) b[j] = ws[kk][tx * GEMM_TN + j];
#pragma unroll
            for (int i = 0; i < GEMM_TM; i++)
#pragma unroll
                for (int j = 0; j < GEMM_TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < GEMM_TM; i++) {
        int m = m0 + ty * GEMM_TM + i;
        if (m >= tokens) continue;
#pragma unroll
        for (int j = 0; j < GEMM_TN; j++) {
            int n = n0 + tx * GEMM_TN + j;
            if (n >= N) continue;
            float v = acc[i][j];
            if (BIAS) v += bias[n];
            float* o = out + (size_t)m * N + n;
            *o = RESIDUAL ? *o + v : v;
        }
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(n));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(PENDING));
}

__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ unsigned bf16_bits_rn(float v) {
    unsigned u = __float_as_uint(v);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

extern "C" __global__ void gg_dequant_bf16(const void* w, const unsigned* table, bf16_t* out, int rows, int cols) {
    int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = cols / 8;
    if (chunk >= rows * per_row) return;
    int row = chunk / per_row;
    int k = (chunk % per_row) * 8;
    float v[8];
    gg_load8(w, table, row, k, v);
    *(uint4*)(out + (size_t)row * cols + k) = make_uint4(bf16_bits_rn(v[0]) | (bf16_bits_rn(v[1]) << 16), bf16_bits_rn(v[2]) | (bf16_bits_rn(v[3]) << 16),
                                                     bf16_bits_rn(v[4]) | (bf16_bits_rn(v[5]) << 16), bf16_bits_rn(v[6]) | (bf16_bits_rn(v[7]) << 16));
}

__device__ __forceinline__ void split_hi_lo(float v, unsigned& hi, unsigned& lo) {
    hi = bf16_bits_rn(v);
    lo = bf16_bits_rn(v - __uint_as_float(hi << 16));
}

__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ unsigned q8_pair_bf16(unsigned two_bytes) {
    int q0 = (signed char)(two_bytes & 0xff);
    int q1 = (signed char)((two_bytes >> 8) & 0xff);
    return (__float_as_uint((float)q0) >> 16) | (__float_as_uint((float)q1) & 0xffff0000u);
}

constexpr int EPI_STORE = 0;
constexpr int EPI_RESIDUAL = 1;
constexpr int EPI_ACT = 2;

__device__ __forceinline__ void split_store2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha, la, hb, lb;
    split_hi_lo(a, ha, la);
    split_hi_lo(b, hb, lb);
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR >= 4, "rows of at least 64 bytes");
    constexpr int GROUP_ROWS = CPR >= 8 ? 1 : 8 / CPR;
    constexpr int MASK = CPR >= 8 ? 7 : CPR - 1;
    return r * CPR + (c ^ ((r / GROUP_ROWS) & MASK));
}

constexpr int ACC_PLAIN = 0;
constexpr int ACC_SCALED = 1;
constexpr int ACC_DIRECT = 2;

template <typename Self>
struct DecBase {
    static constexpr int ACC = ACC_PLAIN;
    static constexpr bool LDSM = true;
    static constexpr int WARPS = LINEAR_WARPS;
    static constexpr int PAIR = 1;
    static constexpr bool STRIDE = false;
    template <int BK> static constexpr int wrow = BK * 2;
    template <int BK> static constexpr int w_chunks = BK / 8;
    template <int K, int ROWS>
    static __device__ __forceinline__ void rows(const void* w, const void* s, int row0, int stride, const float* x, int lane, float* out) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) out[r] = warp_sum(Self::template dot<K>(w, s, row0 + r * stride, x, lane));
    }
    template <int W> static __device__ __forceinline__ int x_at(int i) { return i; }
    template <int K> static __device__ __forceinline__ unsigned col(const void*, int) { return 0; }
    template <int K, int BK> static __device__ __forceinline__ float scale(const void*, unsigned, int) { return 1.0f; }
    static __device__ __forceinline__ void frag(unsigned*, const unsigned char*, int, int) {}
    template <int MI, int NJ> static __device__ __forceinline__ void finish(float (&)[MI][NJ][4], const void*) {}
};

struct DecBf16 : DecBase<DecBf16> {
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void*, int r, int n, int kt, int ck) {
        cp_async16(ws + swz<CPR>(r, ck) * 16, (const char*)w + ((size_t)n * K + kt * BK) * 2 + ck * 16);
    }
    template <int K> static __device__ __forceinline__ float dot(const void* w, const void*, int row, const float* x, int lane) {
        return dot_bf16<K>(w, row, x, lane);
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void*, int n, int k) {
        return bf16_to_f32(((const bf16_t*)w)[(size_t)n * K + k]);
    }
};

template <typename Self, int GROUP_K>
struct DecBytes : DecBase<Self> {
    static constexpr int ACC = ACC_SCALED;
    static constexpr bool LDSM = false;
    template <int BK> static constexpr int wrow = BK + 16;
    template <int BK> static constexpr int w_chunks = BK / 16;
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void*, int r, int n, int kt, int ck) {
        static_assert(GROUP_K % BK == 0, "byte weight tiles stay inside one scale group");
        cp_async16(ws + r * WROW + ck * 16, (const char*)w + (size_t)n * K + kt * BK + ck * 16);
    }
    static __device__ __forceinline__ void frag(unsigned* bw, const unsigned char* row, int ks, int t) {
        const unsigned char* p = row + ks * 16 + 2 * t;
        bw[0] = Self::pair(*(const unsigned short*)p);
        bw[1] = Self::pair(*(const unsigned short*)(p + 8));
    }
    template <int K, int BK> static __device__ __forceinline__ float scale(const void* s, unsigned col, int kt) {
        return ((const float*)s)[col + kt * BK / GROUP_K];
    }
};

struct DecQ8 : DecBytes<DecQ8, Q8_GROUP> {
    static __device__ __forceinline__ unsigned pair(unsigned two) { return q8_pair_bf16(two); }
    template <int K> static __device__ __forceinline__ unsigned col(const void*, int c) { return (unsigned)c * (K / Q8_GROUP); }
    template <int K> static __device__ __forceinline__ float dot(const void* w, const void* s, int row, const float* x, int lane) {
        return dot_q8<K>(w, (const float*)s, row, x, lane);
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void* s, int n, int k) {
        return (float)((const i8_t*)w)[(size_t)n * K + k] * ((const float*)s)[(size_t)n * (K / Q8_GROUP) + k / Q8_GROUP];
    }
};

struct DecFp8 : DecBytes<DecFp8, FP8_BLOCK> {
    static __device__ __forceinline__ unsigned pair(unsigned two) { return e4m3_pair_bf16(two); }
    template <int K> static __device__ __forceinline__ unsigned col(const void* s, int c) { return __float_as_uint(__ldg((const float*)s + c)); }
    template <int K> static __device__ __forceinline__ float dot(const void* w, const void* s, int row, const float* x, int lane) {
        return dot_fp8<K>(w, (const float*)s, row, x, lane);
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void* s, int n, int k) {
        return e4m3_to_f32(((const unsigned char*)w)[(size_t)n * K + k]) * fp8_row_scales((const float*)s, n)[k / FP8_BLOCK];
    }
};

struct DecGgml : DecBase<DecGgml> {
    static constexpr int PAIR = 2;
    static constexpr bool STRIDE = true;
    template <int K, int BK, int CPR, int WROW>
    static __device__ __forceinline__ void load(unsigned char* ws, const void* w, const void* s, int r, int n, int kt, int ck) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, kt * BK + ck * 8, v);
        uint4 packed = make_uint4(bf16_bits_rn(v[0]) | (bf16_bits_rn(v[1]) << 16), bf16_bits_rn(v[2]) | (bf16_bits_rn(v[3]) << 16),
                                  bf16_bits_rn(v[4]) | (bf16_bits_rn(v[5]) << 16), bf16_bits_rn(v[6]) | (bf16_bits_rn(v[7]) << 16));
        *(uint4*)(ws + swz<CPR>(r, ck) * 16) = packed;
    }
    template <int K> static __device__ __forceinline__ float dot(const void* w, const void* s, int row, const float* x, int lane) {
        return gg_row_dot<K>(w, (const unsigned*)s, row, x, lane);
    }
    template <int K, int ROWS>
    static __device__ __forceinline__ void rows(const void* w, const void* s, int row0, int stride, const float* x, int lane, float* out) {
        const unsigned char* base[ROWS];
        int type[ROWS];
        bool same = true;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            base[r] = gg_row(w, (const unsigned*)s, row0 + r * stride, type[r]);            same = same && type[r] == type[0];
        }
        if (same && type[0] == GG_Q4K) {
            gg_q4k_rows<K, ROWS>(base, x, lane, out);
        } else if (same && type[0] == GG_Q6K) {
            gg_q6k_rows<K, ROWS>(base, x, lane, out);
        } else {
#pragma unroll
            for (int r = 0; r < ROWS; r++) out[r] = dot<K>(w, s, row0 + r * stride, x, lane);
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) out[r] = warp_sum(out[r]);
    }
    template <int K> static __device__ __forceinline__ float at(const void* w, const void* s, int n, int k) {
        float v[8];
        gg_load8(w, (const unsigned*)s, n, k & ~7, v);
        return v[k & 7];
    }
};

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS, bool MFAST = false>
__device__ __forceinline__ void gemm_epilogue(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                              bf16_t* out_hi, bf16_t* out_lo, int tokens);

template <int N, int K, int BM, int BN, int WM, int WN, int BK, int STAGES, int EPI, bool BIAS, typename D, bool SPLIT, bool MFAST = false>
__device__ __forceinline__ void gemm_tc(const void* w, const void* scales, const float* bias, const bf16_t* x_hi,
                                        const bf16_t* x_lo, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr bool SCALED = D::ACC == ACC_SCALED;
    constexpr bool DIRECT = D::ACC == ACC_DIRECT;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 16;
    constexpr int CPR = BK / 8;
    constexpr int PLANES = SPLIT ? 2 : 1;
    constexpr int A_TILE = BM * BK;
    constexpr int A_BYTES = A_TILE * 2 * PLANES;
    constexpr int WROW = D::template wrow<BK>;
    constexpr int STAGE_BYTES = A_BYTES + BN * WROW;
    constexpr int KT = K / BK;
    constexpr int A_CHUNKS = BK / 8;
    constexpr int A_TOTAL = BM * A_CHUNKS * PLANES;
    constexpr int A_ITERS = (A_TOTAL + THREADS - 1) / THREADS;
    constexpr int W_CHUNKS = D::template w_chunks<BK>;
    constexpr int W_ITERS = (BN * W_CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of BK");
    static_assert(NJ % 2 == 0, "warp n tile");
    static_assert(STAGES >= 2, "pipeline depth");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    int m0 = (MFAST ? blockIdx.x : blockIdx.y) * BM;
    int n0 = (MFAST ? blockIdx.y : blockIdx.x) * BN;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    int last_row = max(tokens - 1, 0);
    unsigned srow0[SCALED ? NJ : 1], srow1[SCALED ? NJ : 1];
    if (SCALED) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = n0 + wn * TN + j * 8 + 2 * t;
            srow0[j] = D::template col<K>(scales, min(col, N - 1));
            srow1[j] = D::template col<K>(scales, min(col + 1, N - 1));
        }
    }
    auto load_stage = [&](int buf, int kt) {
        unsigned char* base = gemm_smem + buf * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != A_TOTAL && idx >= A_TOTAL) break;
            int p = idx / (BM * A_CHUNKS);
            int rem = idx % (BM * A_CHUNKS);
            int r = rem / A_CHUNKS;
            int c = rem % A_CHUNKS;
            int m = m0 + r;
            const bf16_t* src = (p ? x_lo : x_hi) + (size_t)min(m, last_row) * K + kt * BK + c * 8;
            cp_async16_zfill(base + p * A_TILE * 2 + swz<CPR>(r, c) * 16, src, m < tokens);
        }
        unsigned char* ws = base + A_BYTES;
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * W_CHUNKS && idx >= BN * W_CHUNKS) break;
            int r = idx / W_CHUNKS;
            int ck = idx % W_CHUNKS;
            int n = min(n0 + r, N - 1);
            D::template load<K, BK, CPR, WROW>(ws, w, scales, r, n, kt, ck);
        }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < KT) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* base = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const bf16_t* a_hi = (const bf16_t*)base;
        const bf16_t* a_lo = a_hi + A_TILE;
        const unsigned char* ws = base + A_BYTES;
        float part[SPLIT ? MI : 1][SPLIT ? NJ : 1][4];
        if (SPLIT) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) part[i][j][e] = 0.0f;
        }
        float s0[NJ], s1[NJ];
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            s0[j] = 1.0f;
            s1[j] = 1.0f;
            if (SCALED) {
                s0[j] = D::template scale<K, BK>(scales, srow0[j], kt);
                s1[j] = D::template scale<K, BK>(scales, srow1[j], kt);
            }
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned bw[NJ][2];
            if constexpr (D::LDSM) {
#pragma unroll
                for (int jp = 0; jp < NJ / 2; jp++) {
                    unsigned r[4];
                    int row = wn * TN + jp * 16 + (lane >> 4) * 8 + (lane & 7);
                    ldsm_x4(r, ws + swz<CPR>(row, ks * 2 + ((lane >> 3) & 1)) * 16);
                    bw[2 * jp][0] = r[0];
                    bw[2 * jp][1] = r[1];
                    bw[2 * jp + 1][0] = r[2];
                    bw[2 * jp + 1][1] = r[3];
                }
            } else {
#pragma unroll
                for (int j = 0; j < NJ; j++) D::frag(bw[j], ws + (wn * TN + j * 8 + g) * WROW, ks, t);
            }
            unsigned ah[MI][4], al[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int row = wm * TM + i * 16 + (lane & 15);
                int at = swz<CPR>(row, ks * 2 + (lane >> 4)) * 8;
                ldsm_x4(ah[i], &a_hi[at]);
                if (SPLIT) ldsm_x4(al[i], &a_lo[at]);
            }
            if (DIRECT) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        if (SPLIT) mma_bf16(acc[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(acc[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                    }
            } else if (SPLIT && SCALED) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            } else if (SPLIT) {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        mma_bf16(part[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bw[j][0], bw[j][1]);
                        mma_bf16(part[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                    }
            } else {
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma_bf16(c, ah[i][0], ah[i][1], ah[i][2], ah[i][3], bw[j][0], bw[j][1]);
                        acc[i][j][0] += c[0] * s0[j];
                        acc[i][j][1] += c[1] * s1[j];
                        acc[i][j][2] += c[2] * s0[j];
                        acc[i][j][3] += c[3] * s1[j];
                    }
            }
        }
        if (SPLIT && !SCALED && !DIRECT) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] += part[i][j][e];
        }
    }
    cp_async_wait<0>();
    D::finish(acc, scales);
    gemm_epilogue<N, BM, BN, WM, WN, EPI, BIAS, MFAST>(acc, bias, out, out_hi, out_lo, tokens);
}

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS>
__device__ __forceinline__ void gemm_epilogue_at(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                                 bf16_t* out_hi, bf16_t* out_lo, int tokens, int m0, int n0);

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS, bool MFAST>
__device__ __forceinline__ void gemm_epilogue(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                              bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    int m0 = (MFAST ? blockIdx.x : blockIdx.y) * BM;
    int n0 = (MFAST ? blockIdx.y : blockIdx.x) * BN;
    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, BIAS>(acc, bias, out, out_hi, out_lo, tokens, m0, n0);
}

constexpr int RASTER_GROUP = 32;
constexpr int RASTER_MIN_COLS = 256;

template <int N, int BN>
__device__ __forceinline__ int2 grouped_tile() {
    constexpr int GROUP = (N + BN - 1) / BN >= RASTER_MIN_COLS ? RASTER_GROUP : 1;
    if (GROUP == 1) return make_int2(blockIdx.y, blockIdx.x);
    int num_m = gridDim.y;
    int num_n = gridDim.x;
    int pid = blockIdx.y * num_n + blockIdx.x;
    int per_group = GROUP * num_n;
    int first_m = pid / per_group * GROUP;
    int size = min(num_m - first_m, GROUP);
    int in_group = pid % per_group;
    return make_int2(first_m + in_group % size, in_group / size);
}

template <int N, int BM, int BN, int WM, int WN, int EPI, bool BIAS>
__device__ __forceinline__ void gemm_epilogue_at(float (&acc)[BM / WM / 16][BN / WN / 8][4], const float* bias, float* out,
                                                 bf16_t* out_hi, bf16_t* out_lo, int tokens, int m0, int n0) {
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    if (EPI == EPI_ACT) {
        static_assert(EPI != EPI_ACT || N % 4 == 0, "gate/up pairs");
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                int col = n0 + wn * TN + j * 8 + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int m = m0 + wm * TM + i * 16 + g + h * 8;
                    float a = activation(acc[i][j][h * 2]) * acc[i][j][h * 2 + 1];
                    float b = __shfl_xor_sync(0xffffffff, a, 1);
                    if ((t & 1) == 0 && col < N && m < tokens)
                        split_store2(out_hi, out_lo, (size_t)m * (N / 2) + col / 2, a, b);
                }
            }
        return;
    }
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            int col = n0 + wn * TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = 0.0f, b1 = 0.0f;
            if (BIAS) {
                b0 = bias[col];
                b1 = col + 1 < N ? bias[col + 1] : 0.0f;
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * TM + i * 16 + g + h * 8;
                if (m >= tokens) continue;
                float v0 = acc[i][j][h * 2] + b0;
                float v1 = acc[i][j][h * 2 + 1] + b1;
                {
                    float* o = out + (size_t)m * N + col;
                    if (EPI == EPI_RESIDUAL) {
                        v0 += o[0];
                        if (col + 1 < N) v1 += o[1];
                    }
                    o[0] = v0;
                    if (col + 1 < N) o[1] = v1;
                }
            }
        }
}

__device__ __forceinline__ void mma_e4m3(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ unsigned f32x2_to_e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

constexpr float FP8_MAX = 448.0f;

extern "C" __global__ void quant_rows_fp8(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {
    __shared__ float scratch[32];
    int row = blockIdx.y;
    int block = blockIdx.x;
    int c = block * FP8_BLOCK + threadIdx.x * 2;
    size_t at = (size_t)row * width + c;
    float a = bf16_to_f32(hi[at]) + bf16_to_f32(lo[at]);
    float b = bf16_to_f32(hi[at + 1]) + bf16_to_f32(lo[at + 1]);
    float amax = block_reduce<true>(fmaxf(fabsf(a), fabsf(b)), scratch);
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    *(unsigned short*)(q + at) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    if (threadIdx.x == 0) scale[(size_t)row * (width / FP8_BLOCK) + block] = s;
}

extern "C" __global__ void quant_rows_fp8_token(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {
    __shared__ float scratch[32];
    size_t row = (size_t)blockIdx.x * width;
    float amax = 0.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
    }
    amax = block_reduce<true>(amax, scratch);
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        *(unsigned short*)(q + row + c) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    }
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
}

template <int N, int K, int BM, int BN, int WM, int WN, int STAGES, int EPI, bool BIAS, bool PER_TOKEN = false>
__device__ __forceinline__ void gemm_fp8(const void* w, const float* wscale, const float* bias, const unsigned char* xq,
                                         const float* xscale, float* out, bf16_t* out_hi, bf16_t* out_lo, int tokens) {
    constexpr int BK = FP8_BLOCK;
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM;
    constexpr int TN = BN / WN;
    constexpr int MI = TM / 16;
    constexpr int NJ = TN / 8;
    constexpr int KS = BK / 32;
    constexpr int ROW = BK;
    constexpr int STAGE_BYTES = (BM + BN) * ROW;
    constexpr int KT = K / BK;
    constexpr int CHUNKS = BK / 16;
    constexpr int A_ITERS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_ITERS = (BN * CHUNKS + THREADS - 1) / THREADS;
    static_assert(K % BK == 0, "K must be a multiple of the FP8 block");
    extern __shared__ __align__(16) unsigned char gemm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / WN;
    int wn = warp % WN;
    int2 tile = grouped_tile<N, BN>();
    int m0 = tile.x * BM;
    int n0 = tile.y * BN;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    float c[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) c[i][j][e] = 0.0f;
    int last_row = max(tokens - 1, 0);
    unsigned srow0[NJ], srow1[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        int col = n0 + wn * TN + j * 8 + 2 * t;
        srow0[j] = __float_as_uint(__ldg(wscale + min(col, N - 1)));
        srow1[j] = __float_as_uint(__ldg(wscale + min(col + 1, N - 1)));
    }
    auto load_stage = [&](int buf, int kt) {
        unsigned char* as = gemm_smem + buf * STAGE_BYTES;
        unsigned char* ws = as + BM * ROW;
#pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (A_ITERS * THREADS != BM * CHUNKS && idx >= BM * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int m = m0 + r;
            cp_async16_zfill(as + r * ROW + ((c ^ (r & 7)) << 4), xq + (size_t)min(m, last_row) * K + kt * BK + c * 16, m < tokens);
        }
#pragma unroll
        for (int i = 0; i < W_ITERS; i++) {
            int idx = tid + i * THREADS;
            if (W_ITERS * THREADS != BN * CHUNKS && idx >= BN * CHUNKS) break;
            int r = idx / CHUNKS;
            int c = idx % CHUNKS;
            int n = min(n0 + r, N - 1);
            cp_async16(ws + r * ROW + ((c ^ (r & 7)) << 4), (const unsigned char*)w + (size_t)n * K + kt * BK + c * 16);
        }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < KT) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int next = kt + STAGES - 1;
        if (next < KT) load_stage(next % STAGES, next);
        cp_async_commit();
        const unsigned char* as = gemm_smem + (kt % STAGES) * STAGE_BYTES;
        const unsigned char* ws = as + BM * ROW;
        float sw0[NJ], sw1[NJ], sa[MI][2];
        if (!PER_TOKEN) {
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                sw0[j] = wscale[srow0[j] + kt];
                sw1[j] = wscale[srow1[j] + kt];
            }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int m = min(m0 + wm * TM + i * 16 + g + h * 8, last_row);
                    sa[i][h] = xscale[(size_t)m * KT + kt];
                }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) c[i][j][e] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                int r = wm * TM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int ch = ks * 2 + (lane >> 4);
                ldsm_x4(a[i], as + r * ROW + ((ch ^ (r & 7)) << 4));
            }
#pragma unroll
            for (int jj = 0; jj < NJ; jj += 2) {
                unsigned b[4];
                int r = wn * TN + jj * 8 + (lane & 7) + (lane >> 4) * 8;
                int ch = ks * 2 + ((lane >> 3) & 1);
                ldsm_x4(b, ws + r * ROW + ((ch ^ (r & 7)) << 4));
#pragma unroll
                for (int i = 0; i < MI; i++) {
                    mma_e4m3(c[i][jj], a[i], b[0], b[1]);
                    mma_e4m3(c[i][jj + 1], a[i], b[2], b[3]);
                }
            }
        }
        if (!PER_TOKEN) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    acc[i][j][0] += c[i][j][0] * sa[i][0] * sw0[j];
                    acc[i][j][1] += c[i][j][1] * sa[i][0] * sw1[j];
                    acc[i][j][2] += c[i][j][2] * sa[i][1] * sw0[j];
                    acc[i][j][3] += c[i][j][3] * sa[i][1] * sw1[j];
                }
        }
    }
    if (PER_TOKEN) {
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = min(m0 + wm * TM + i * 16 + g + h * 8, last_row);
                float sa = xscale[m];
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    acc[i][j][2 * h] = c[i][j][2 * h] * sa * wscale[srow0[j]];
                    acc[i][j][2 * h + 1] = c[i][j][2 * h + 1] * sa * wscale[srow1[j]];
                }
            }
    }
    cp_async_wait<0>();
    gemm_epilogue_at<N, BM, BN, WM, WN, EPI, BIAS>(acc, bias, out, out_hi, out_lo, tokens, m0, n0);
}

extern "C" __global__ void rms_split(const float* x, const float* weight, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    static_assert(HIDDEN % 4 == 0, "hidden size");
    size_t row = (size_t)blockIdx.x * HIDDEN;
    const float4* xv = (const float4*)(x + row);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN / 4; i += blockDim.x) {
        float4 v = xv[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN / 4; i += blockDim.x) {
        float4 v = xv[i];
        float4 nv = ((const float4*)weight)[i];
        split_store2(hi, lo, row + 4 * i, nv.x * (v.x * inv), nv.y * (v.y * inv));
        split_store2(hi, lo, row + 4 * i + 2, nv.z * (v.z * inv), nv.w * (v.w * inv));
    }
}

extern "C" __global__ void embed(const bf16_t* table, const int* token, float* x) {
    const bf16_t* row = table + (size_t)token[blockIdx.y] * HIDDEN;
    x += (size_t)blockIdx.y * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x)
        x[i] = bf16_to_f32(row[i]);
}

extern "C" __global__ void splice_rows(float* x, const int* rows, const float* embeds) {
    int row = rows[blockIdx.y];
    if (row < 0) return;
    float* dst = x + (size_t)blockIdx.y * HIDDEN;
    const float* src = embeds + (size_t)row * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void rmsnorm(const float* x, const float* weight, float* out) {
    __shared__ float scratch[32];
    x += (size_t)blockIdx.x * HIDDEN;
    out += (size_t)blockIdx.x * HIDDEN;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ss += x[i] * x[i];
    float inv = rsqrtf(block_reduce<false>(ss, scratch) / HIDDEN + EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) out[i] = weight[i] * (x[i] * inv);
}

__device__ __forceinline__ float rope_inv_freq(int i) {
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
    }
    return inv_freq;
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

__device__ __forceinline__ bf16_t* kv_row(bf16_t* cache, const int* block_table, int layer, int kv_head, int pos) {
    return cache + kv_slot(block_table, layer, kv_head, pos) * KV_PLANES * HEAD_DIM;
}

__device__ __forceinline__ const bf16_t* kv_row(const bf16_t* cache, const int* block_table, int layer, int kv_head,
                                                int pos) {
    return cache + kv_slot(block_table, layer, kv_head, pos) * KV_PLANES * HEAD_DIM;
}

__device__ __forceinline__ const int* seq_table(const int* block_table, const int* slots, int row) {
    return block_table + (size_t)slots[row] * PAGES;
}

__device__ __forceinline__ float kv_at(const bf16_t* row, int d) {
    float v = attn_float(row[d]);
    return KV_PLANES == 2 ? v + attn_float(row[HEAD_DIM + d]) : v;
}

__device__ __forceinline__ float4 kv_at4(const bf16_t* row, int d) {
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

__device__ __forceinline__ float rope_position(const int* mm_meta, const int* mm_table, int slot, int idx, int i) {
    if (!MROPE) return (float)idx;
    if (idx < mm_meta[2 * slot]) return (float)mm_table[((size_t)slot * MAX_CTX + idx) * 3 + mrope_axis(i)];
    return (float)(idx + mm_meta[2 * slot + 1]);
}

__device__ __forceinline__ void norm_rope_head(float* qkv, const float* q_norm, const float* k_norm,
                                               bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                               int pos, int layer, int head, int row,
                                               const int* mm_meta, const int* mm_table, int slot) {
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
    if ((is_q || is_k) && d < ROT) {
        int fi = d < ROT_HALF ? d : d - ROT_HALF;
        float angle = rope_position(mm_meta, mm_table, slot, pos, fi) * rope_inv_freq(fi);
        float c = cosf(angle);
        float s = sinf(angle);
        v = d < ROT_HALF ? vec[d] * c - vec[d + ROT_HALF] * s : vec[d] * c + vec[d - ROT_HALF] * s;
    }
    if (d >= HEAD_DIM) return;
    if (is_q) {
        src[d] = v;
        return;
    }
    int kv_head = is_k ? head - HEADS : head - HEADS - KV_HEADS;
    bf16_t* dst = kv_row(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos);
    unsigned hi, lo;
    attn_split(v, hi, lo);
    dst[d] = (bf16_t)hi;
    if (KV_PLANES == 2) dst[HEAD_DIM + d] = (bf16_t)lo;
}

extern "C" __global__ void qk_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos, const int* slots, int layer, int per_row,
                                        const int* mm_meta, const int* mm_table, int slot_base) {
    int row = blockIdx.y;
    const int* table = per_row ? seq_table(block_table, slots, row) : block_table;
    int p = per_row ? pos[row] : pos[0] + row;
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, table, p, layer, blockIdx.x, row, mm_meta, mm_table,
                   per_row ? slots[row] : slot_base);
}

extern "C" __global__ void kv_norm_rope(float* qkv, const float* q_norm, const float* k_norm,
                                        bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                        const int* pos_ptr, int layer,
                                        const int* mm_meta, const int* mm_table, int slot) {
    norm_rope_head(qkv, q_norm, k_norm, k_cache, v_cache, block_table, pos_ptr[0] + blockIdx.y, layer,
                   HEADS + blockIdx.x, blockIdx.y, mm_meta, mm_table, slot);
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
                                     const int* block_table, const int* pos_ptr, int layer, float* part) {
    __shared__ float4 acc_s[ATTN_WARPS][GROUP][HEAD_DIM / 4];
    __shared__ float m_s[ATTN_WARPS][GROUP];
    __shared__ float l_s[ATTN_WARPS][GROUP];
    int kv_head = blockIdx.x;
    int split = blockIdx.y;
    int len = pos_ptr[0] + 1;
    int chunk = attn_chunk(len);
    int start = split * chunk;
    if (start >= len) return;
    int end = min(len, start + chunk);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    static_assert(HEAD_DIM % 4 == 0 && HEAD_DIM <= 256, "unsupported head_dim");
    float scale = rsqrtf((float)HEAD_DIM);
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
#pragma unroll
            for (int j = 0; j < ATTN_V4; j++) {
                int c = lane + 32 * j;
                bool on = c < HEAD_DIM / 4;
                kr[u][j] = on ? kv_at4(k, 4 * c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vr[u][j] = on ? kv_at4(v, 4 * c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
                dot = warp_sum(dot) * scale;
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
            if (lane + 32 * j < HEAD_DIM / 4) acc_s[warp][g][lane + 32 * j] = acc[g][j];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < GROUP * HEAD_DIM; idx += ATTN_BLOCK) {
        int g = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        float M = NEG_INF;
        for (int w = 0; w < ATTN_WARPS; w++) M = fmaxf(M, m_s[w][g]);
        float L = 0.0f, A = 0.0f;
        for (int w = 0; w < ATTN_WARPS; w++) {
            float c = m_s[w][g] == NEG_INF ? 0.0f : expf(m_s[w][g] - M);
            L += l_s[w][g] * c;
            A += ((const float*)acc_s[w][g])[d] * c;
        }
        float* dst = part + ((size_t)(kv_head * GROUP + g) * ATTN_SPLITS + split) * ATTN_PART;
        dst[d] = A;
        if (d == 0) {
            dst[HEAD_DIM] = M;
            dst[HEAD_DIM + 1] = L;
        }
    }
}

extern "C" __global__ void attention_reduce(const float* part, const int* pos_ptr, const float* qkv, float* out) {
    int head = blockIdx.x;
    int len = pos_ptr[0] + 1;
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

constexpr int BD_WARPS = 8;
constexpr int BD_SPLITS = {{BD_SPLITS}};
constexpr int BD_UNROLL = 8;

extern "C" __global__ void batch_decode_attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                                  const int* block_table, const int* pos, const int* slots,
                                                  int layer, float* part_acc, float* part_ml) {
    constexpr int PER_LANE = (HEAD_DIM + 31) / 32;
    __shared__ float sm[BD_WARPS][GROUP];
    __shared__ float sl[BD_WARPS][GROUP];
    __shared__ float sacc[BD_WARPS][GROUP][HEAD_DIM];
    int kv_head = blockIdx.x;
    int seq = blockIdx.y;
    int split = blockIdx.z;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int len = pos[seq] + 1;
    int chunk = (len + BD_SPLITS - 1) / BD_SPLITS;
    int t0 = split * chunk;
    int t1 = min(len, t0 + chunk);
    const int* table = seq_table(block_table, slots, seq);
    const float* q = qkv + (size_t)seq * QKV_ROW + kv_head * GROUP * HEAD_DIM;
    float scale = rsqrtf((float)HEAD_DIM);
    float qr[GROUP][PER_LANE];
#pragma unroll
    for (int g = 0; g < GROUP; g++)
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            qr[g][i] = d < HEAD_DIM ? q[g * HEAD_DIM + d] * scale : 0.0f;
        }
    float m[GROUP], l[GROUP], acc[GROUP][PER_LANE];
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        m[g] = NEG_INF;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) acc[g][i] = 0.0f;
    }
    constexpr int U = BD_UNROLL;
    for (int tb = t0 + warp; tb < t1; tb += BD_WARPS * U) {
        float kr[U][PER_LANE], vr[U][PER_LANE];
#pragma unroll
        for (int u = 0; u < U; u++) {
            int t = tb + u * BD_WARPS;
            const bf16_t* k = kv_row(k_cache, table, layer, kv_head, min(t, t1 - 1));
            const bf16_t* v = kv_row(v_cache, table, layer, kv_head, min(t, t1 - 1));
#pragma unroll
            for (int i = 0; i < PER_LANE; i++) {
                int d = lane + i * 32;
                kr[u][i] = d < HEAD_DIM ? kv_at(k, d) : 0.0f;
                vr[u][i] = d < HEAD_DIM ? kv_at(v, d) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (tb + u * BD_WARPS >= t1) break;
#pragma unroll
            for (int g = 0; g < GROUP; g++) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < PER_LANE; i++) dot += qr[g][i] * kr[u][i];
                dot = warp_sum(dot);
                float nm = fmaxf(m[g], dot);
                float corr = expf(m[g] - nm);
                float p = expf(dot - nm);
                l[g] = l[g] * corr + p;
#pragma unroll
                for (int i = 0; i < PER_LANE; i++) acc[g][i] = acc[g][i] * corr + p * vr[u][i];
                m[g] = nm;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
        if (lane == 0) {
            sm[warp][g] = m[g];
            sl[warp][g] = l[g];
        }
#pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane + i * 32;
            if (d < HEAD_DIM) sacc[warp][g][d] = acc[g][i];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < GROUP * HEAD_DIM; idx += blockDim.x) {
        int g = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        float M = NEG_INF;
        for (int w = 0; w < BD_WARPS; w++) M = fmaxf(M, sm[w][g]);
        float L = 0.0f, A = 0.0f;
        if (M != NEG_INF) {
            for (int w = 0; w < BD_WARPS; w++) {
                float c = sm[w][g] == NEG_INF ? 0.0f : expf(sm[w][g] - M);
                L += sl[w][g] * c;
                A += sacc[w][g][d] * c;
            }
        }
        int head = kv_head * GROUP + g;
        size_t base = ((size_t)seq * HEADS + head) * BD_SPLITS + split;
        part_acc[base * HEAD_DIM + d] = A;
        if (d == 0) {
            part_ml[base * 2] = M;
            part_ml[base * 2 + 1] = L;
        }
    }
}

extern "C" __global__ void batch_decode_combine(const float* part_acc, const float* part_ml, const float* qkv, bf16_t* out_hi, bf16_t* out_lo) {
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int d = 2 * threadIdx.x;
    size_t base = ((size_t)seq * HEADS + head) * BD_SPLITS;
    float M = NEG_INF;
    for (int s = 0; s < BD_SPLITS; s++) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, A0 = 0.0f, A1 = 0.0f;
    for (int s = 0; s < BD_SPLITS; s++) {
        float ms = part_ml[(base + s) * 2];
        float c = ms == NEG_INF ? 0.0f : expf(ms - M);
        L += part_ml[(base + s) * 2 + 1] * c;
        A0 += part_acc[(base + s) * HEAD_DIM + d] * c;
        A1 += part_acc[(base + s) * HEAD_DIM + d + 1] * c;
    }
    int at = head * HEAD_DIM + d;
    split_store2(out_hi, out_lo, (size_t)seq * Q_DIM + at, A0 / L * out_gate(qkv, seq, at), A1 / L * out_gate(qkv, seq, at + 1));
}

constexpr int FA_BQ = 16;
constexpr int FA_BK = 32;
constexpr int FA_THREADS = 128;
constexpr int FA_STRIDE = HEAD_DIM + 1;
constexpr int FA_OUT = HEAD_DIM / 8;

extern "C" __global__ void flash_attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                           const int* block_table, const int* pos_ptr, int layer, float* out, int tokens) {
    __shared__ float qs[FA_BQ][FA_STRIDE];
    __shared__ float ks[FA_BK][FA_STRIDE];
    __shared__ float vs[FA_BK][HEAD_DIM];
    __shared__ float ps[FA_BQ][FA_BK + 1];
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = blockIdx.x * FA_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int qr = tid >> 3;
    int sub = tid & 7;
    int qi = q0 + qr;
    int pos = base + qi;
    float scale = rsqrtf((float)HEAD_DIM);
    for (int i = tid; i < FA_BQ * HEAD_DIM; i += FA_THREADS) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q = q0 + r;
        qs[r][d] = q < tokens ? qkv[(size_t)q * QKV_ROW + head * HEAD_DIM + d] : 0.0f;
    }
    float o[FA_OUT];
#pragma unroll
    for (int i = 0; i < FA_OUT; i++) o[i] = 0.0f;
    float m = NEG_INF;
    float l = 0.0f;
    int last = min(q0 + FA_BQ, tokens) - 1;
    int key_end = base + last + 1;
    for (int k0 = 0; k0 < key_end; k0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * HEAD_DIM; i += FA_THREADS) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int t = k0 + r;
            float kv = 0.0f, vv = 0.0f;
            if (t < key_end) {
                kv = kv_at(kv_row(k_cache, block_table, layer, kv_head, t), d);
                vv = kv_at(kv_row(v_cache, block_table, layer, kv_head, t), d);
            }
            ks[r][d] = kv;
            vs[r][d] = vv;
        }
        __syncthreads();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++) {
            float qv = qs[qr][d];
#pragma unroll
            for (int j = 0; j < 4; j++) s[j] += qv * ks[sub * 4 + j][d];
        }
        float tile_max = NEG_INF;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = k0 + sub * 4 + j;
            s[j] = (t <= pos && t < key_end) ? s[j] * scale : NEG_INF;
            tile_max = fmaxf(tile_max, s[j]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
        float m_new = fmaxf(m, tile_max);
        float alpha = m == NEG_INF ? 0.0f : expf(m - m_new);
        float row_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float p = s[j] == NEG_INF ? 0.0f : expf(s[j] - m_new);
            ps[qr][sub * 4 + j] = p;
            row_sum += p;
        }
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);
        l = l * alpha + row_sum;
        m = m_new;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) o[i] *= alpha;
        __syncwarp();
#pragma unroll 8
        for (int j = 0; j < FA_BK; j++) {
            float p = ps[qr][j];
#pragma unroll
            for (int i = 0; i < FA_OUT; i++) o[i] += p * vs[j][sub + 8 * i];
        }
    }
    if (qi < tokens) {
        float inv = 1.0f / l;
        float* dst = out + (size_t)qi * Q_DIM + head * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) dst[sub + 8 * i] = o[i] * inv * out_gate(qkv, qi, head * HEAD_DIM + sub + 8 * i);
    }
}

constexpr int FM_WARPS = {{FM_WARPS}};
constexpr int FM_BQ = 16 * FM_WARPS;
constexpr int FM_BK = {{FM_BK}};
constexpr int FM_STAGES = {{FM_STAGES}};
constexpr int FM_KB = HEAD_DIM / 16;
constexpr int FM_NB = HEAD_DIM / 8;

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

extern "C" __global__ void __launch_bounds__(32 * FM_WARPS) flash_mma(const float* qkv, const float* q_norm, const bf16_t* k_cache,
                                     const bf16_t* v_cache, const int* block_table, const int* pos_ptr, int layer,
                                     bf16_t* out_hi, bf16_t* out_lo, int tokens,
                                     const int* mm_meta, const int* mm_table, int mm_slot) {
    constexpr int CPR = HEAD_DIM / 8;
    constexpr int PLANE = FM_BK * HEAD_DIM;
    static_assert(FM_BQ * HEAD_DIM * 4 <= {{FM_SMEM}}, "Q staging fits");
    extern __shared__ __align__(16) unsigned char fm_smem[];
    bf16_t* kvs = (bf16_t*)fm_smem;
    float* qs = (float*)fm_smem;
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = (gridDim.x - 1 - blockIdx.x) * FM_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int pos0 = base + row0;
    int pos1 = base + row1;
    float scale_log2 = rsqrtf((float)HEAD_DIM) * 1.4426950408889634f;
    unsigned qh[FM_KB][4];
    unsigned ql[FM_KB][4];
    constexpr int PAIRS = (ROT_HALF + 31) / 32;
    float* wq = qs + warp * 16 * HEAD_DIM;
    float freq[PAIRS];
#pragma unroll
    for (int i = 0; i < PAIRS; i++) freq[i] = rope_inv_freq(min(lane + 32 * i, ROT_HALF - 1));
    for (int r = 0; r < 16; r++) {
        int row = q0 + warp * 16 + r;
        const float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
        float* dst = wq + r * HEAD_DIM;
        float ss = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            float v = row < tokens ? src[d] : 0.0f;
            dst[d] = v;
            ss += v * v;
        }
        float inv = QK_NORM ? rsqrtf(warp_sum(ss) / HEAD_DIM + EPS) : 1.0f;
        __syncwarp();
        if (QK_NORM)
            for (int d = lane; d < HEAD_DIM; d += 32) dst[d] = q_norm[d] * (dst[d] * inv);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < PAIRS; i++) {
            int d = lane + 32 * i;
            if (d < ROT_HALF) {
                float x = dst[d], y = dst[d + ROT_HALF];
                float sn, c;
                sincosf(rope_position(mm_meta, mm_table, mm_slot, base + row, d) * freq[i], &sn, &c);
                dst[d] = x * c - y * sn;
                dst[d + ROT_HALF] = y * c + x * sn;
            }
        }
        __syncwarp();
    }
    __syncwarp();
#pragma unroll
    for (int kb = 0; kb < FM_KB; kb++) {
        int d = kb * 16 + 2 * t;
        float2 v[4];
        v[0] = *(const float2*)&wq[g * HEAD_DIM + d];
        v[1] = *(const float2*)&wq[(g + 8) * HEAD_DIM + d];
        v[2] = *(const float2*)&wq[g * HEAD_DIM + d + 8];
        v[3] = *(const float2*)&wq[(g + 8) * HEAD_DIM + d + 8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            qh[kb][i] = pack_hi(v[i].x, v[i].y);
            if (KV_PLANES == 2) ql[kb][i] = pack_lo(v[i].x, v[i].y);
        }
    }
    float o[FM_NB][4];
#pragma unroll
    for (int i = 0; i < FM_NB; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.0f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    int last = min(q0 + FM_BQ, tokens) - 1;
    int key_end = base + last + 1;
    int warp_last = base + q0 + warp * 16 + 15;
    int tiles = (key_end + FM_BK - 1) / FM_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = kvs + buf * 4 * PLANE;
        constexpr int ROW_CHUNKS = KV_PLANES * CPR;
        constexpr int TOTAL = 2 * FM_BK * ROW_CHUNKS;
#pragma unroll
        for (int i = tid; i < TOTAL; i += 32 * FM_WARPS) {
            int which = i / (FM_BK * ROW_CHUNKS);
            int rem = i % (FM_BK * ROW_CHUNKS);
            int r = rem / ROW_CHUNKS;
            int c = rem % ROW_CHUNKS;
            int tk = k0 + r;
            const bf16_t* src = kv_row(which ? v_cache : k_cache, block_table, layer, kv_head, min(tk, key_end - 1));
            bf16_t* dst = st + (which * 2 + c / CPR) * PLANE + swz<CPR>(r, c % CPR) * 8;
            cp_async16_zfill(dst, src + c * 8, tk < key_end);
        }
    };
    __syncthreads();
    load_kv(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kt * FM_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) % FM_STAGES, k0 + FM_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* k_hi = kvs + (kt % FM_STAGES) * 4 * PLANE;
        const bf16_t* k_lo = k_hi + PLANE;
        const bf16_t* v_hi = k_lo + PLANE;
        const bf16_t* v_lo = v_hi + PLANE;
        if (k0 > warp_last) {
            __syncthreads();
            continue;
        }
        constexpr int SB = FM_BK / 8;
        float s[SB][4];
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[nb][e] = 0.0f;
#pragma unroll
        for (int np = 0; np < SB / 2; np++)
#pragma unroll
            for (int kb = 0; kb < FM_KB; kb++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
                int at = swz<CPR>(kr, kb * 2 + ((lane >> 3) & 1)) * 8;
                ldsm_x4(bh, k_hi + at);
                if (KV_PLANES == 2) ldsm_x4(bl, k_lo + at);
                mma_kv(s[2 * np], qh[kb], ql[kb], bh[0], bh[1], bl[0], bl[1]);
                mma_kv(s[2 * np + 1], qh[kb], ql[kb], bh[2], bh[3], bl[2], bl[3]);
            }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < SB; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = k0 + nb * 8 + 2 * t + (e & 1);
                int pos = e < 2 ? pos0 : pos1;
                float v = (key <= pos && key < key_end) ? s[nb][e] * scale_log2 : NEG_INF;
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
        for (int i = 0; i < FM_NB; i++) {
            o[i][0] *= a0;
            o[i][1] *= a0;
            o[i][2] *= a1;
            o[i][3] *= a1;
        }
#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
            ph[0] = pack_hi(s[2 * kb][0], s[2 * kb][1]);
            ph[1] = pack_hi(s[2 * kb][2], s[2 * kb][3]);
            ph[2] = pack_hi(s[2 * kb + 1][0], s[2 * kb + 1][1]);
            ph[3] = pack_hi(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            if (KV_PLANES == 2) {
                pl[0] = pack_lo(s[2 * kb][0], s[2 * kb][1]);
                pl[1] = pack_lo(s[2 * kb][2], s[2 * kb][3]);
                pl[2] = pack_lo(s[2 * kb + 1][0], s[2 * kb + 1][1]);
                pl[3] = pack_lo(s[2 * kb + 1][2], s[2 * kb + 1][3]);
            }
#pragma unroll
            for (int np = 0; np < FM_NB / 2; np++) {
                unsigned bh[4], bl[4] = {0, 0, 0, 0};
                int kr = kb * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int at = swz<CPR>(kr, np * 2 + (lane >> 4)) * 8;
                ldsm_x4_t(bh, v_hi + at);
                if (KV_PLANES == 2) ldsm_x4_t(bl, v_lo + at);
                mma_kv(o[2 * np], ph, pl, bh[0], bh[1], bl[0], bl[1]);
                mma_kv(o[2 * np + 1], ph, pl, bh[2], bh[3], bl[2], bl[3]);
            }
        }
        __syncthreads();
    }
    float i0 = 1.0f / l0;
    float i1 = 1.0f / l1;
#pragma unroll
    for (int nb = 0; nb < FM_NB; nb++) {
        int d = nb * 8 + 2 * t;
        int at = head * HEAD_DIM + d;
        if (row0 < tokens)
            split_store2(out_hi, out_lo, (size_t)row0 * Q_DIM + at, o[nb][0] * i0 * out_gate(qkv, row0, at),
                         o[nb][1] * i0 * out_gate(qkv, row0, at + 1));
        if (row1 < tokens)
            split_store2(out_hi, out_lo, (size_t)row1 * Q_DIM + at, o[nb][2] * i1 * out_gate(qkv, row1, at),
                         o[nb][3] * i1 * out_gate(qkv, row1, at + 1));
    }
}

extern "C" __global__ void swiglu(const float* gate_up, float* out) {
    gate_up += (size_t)blockIdx.y * 2 * INTER;
    out += (size_t)blockIdx.y * INTER;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < INTER; i += gridDim.x * blockDim.x) {
        float g = gate_up[2 * i];
        float a = ACT_KIND == 1
            ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))
            : g / (1.0f + expf(-g));
        out[i] = a * gate_up[2 * i + 1];
    }
}

__device__ float gumbel(unsigned seed, unsigned step, unsigned i) {
    unsigned x = seed * 747796405u + step * 2891336453u + i * 2246822519u + 1u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    float u = ((float)(x >> 8) + 0.5f) / 16777216.0f;
    return -logf(-logf(u));
}

extern "C" __global__ void argmax_advance(const float* logits, int* token, int* pos, int* history, int* count, float inv_temp, unsigned seed) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) v = v * inv_temp + gumbel(seed, (unsigned)count[0], (unsigned)i);
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

extern "C" __global__ void argmax_rows(const float* logits, int* tokens, int* history, const int* count, int stride) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    logits += (size_t)blockIdx.x * VOCAB;
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
        tokens[blockIdx.x] = best_i[0];
        history[(size_t)blockIdx.x * stride + count[0]] = best_i[0];
    }
}

__device__ __forceinline__ unsigned order_key(float v) {
    unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float order_value(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__device__ float sample_threshold(const float* row, float inv_temp, int top_k, float top_p, float min_p, float* row_max) {
    __shared__ unsigned hist[2048];
    __shared__ unsigned chunk[64];
    __shared__ float scratch[32];
    __shared__ unsigned sh_prefix;
    __shared__ int sh_need;
    float m = NEG_INF;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) m = fmaxf(m, row[i] * inv_temp);
    m = block_reduce<true>(m, scratch);
    *row_max = m;
    bool use_k = top_k > 0 && top_k < VOCAB;
    bool use_p = top_p < 1.0f;
    float thr = NEG_INF;
    if (use_k) {
        unsigned prefix = 0, mask = 0;
        int need = top_k;
        for (int pass = 0; pass < 3; pass++) {
            int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
            unsigned width_mask = pass == 2 ? 1023u : 2047u;
            for (int i = threadIdx.x; i < 2048; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                unsigned key = order_key(row[i] * inv_temp);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & width_mask], 1u);
            }
            __syncthreads();
            if (threadIdx.x < 64) {
                unsigned c = 0;
                for (int j = 0; j < 32; j++) c += hist[threadIdx.x * 32 + j];
                chunk[threadIdx.x] = c;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int c = need;
                int ch = 63;
                for (; ch > 0; ch--) {
                    if ((int)chunk[ch] >= c) break;
                    c -= (int)chunk[ch];
                }
                int b = ch * 32 + 31;
                for (; b > ch * 32; b--) {
                    if ((int)hist[b] >= c) break;
                    c -= (int)hist[b];
                }
                sh_need = c;
                sh_prefix = prefix | ((unsigned)b << shift);
            }
            __syncthreads();
            need = sh_need;
            prefix = sh_prefix;
            mask |= width_mask << shift;
        }
        thr = order_value(prefix);
    }
    if (use_p) {
        __shared__ float cand[1024];
        __shared__ int ncand;
        if (threadIdx.x == 0) ncand = 0;
        __syncthreads();
        if (use_k) {
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= thr) {
                    int at = atomicAdd(&ncand, 1);
                    if (at < 1024) cand[at] = v;
                }
            }
            __syncthreads();
        }
        bool cached = use_k && ncand <= 1024;
        int count = cached ? ncand : 0;
        float z = 0.0f;
        if (cached) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) z += expf(cand[i] - m);
        } else {
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= thr) z += expf(v - m);
            }
        }
        z = block_reduce<false>(z, scratch);
        float target = fmaxf(top_p, 1e-9f) * z;
        float lo = use_k ? thr : m - 88.0f;
        float hi = nextafterf(m, -NEG_INF);
        for (int it = 0; it < 48; it++) {
            float mid = 0.5f * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            float mass = 0.0f;
            if (cached) {
                for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= mid) mass += expf(cand[i] - m);
            } else {
                for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                    float v = row[i] * inv_temp;
                    if (v >= mid) mass += expf(v - m);
                }
            }
            mass = block_reduce<false>(mass, scratch);
            if (mass >= target) lo = mid; else hi = mid;
        }
        float low = NEG_INF;
        if (cached) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) if (cand[i] >= lo) low = fmaxf(low, -cand[i]);
        } else {
            for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
                float v = row[i] * inv_temp;
                if (v >= lo) low = fmaxf(low, -v);
            }
        }
        thr = fmaxf(thr, -block_reduce<true>(low, scratch));
    }
    if (min_p > 0.0f) thr = fmaxf(thr, m + logf(min_p));
    return thr;
}

extern "C" __global__ void sample_rows(const float* logits, int* tokens, const int* params) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    logits += (size_t)blockIdx.x * VOCAB;
    const int* p = params + 6 * blockIdx.x;
    float inv_temp = __int_as_float(p[0]);
    unsigned seed = (unsigned)p[1];
    unsigned step = (unsigned)p[2];
    float thr = NEG_INF;
    if (inv_temp > 0.0f) {
        float m;
        thr = sample_threshold(logits, inv_temp, p[3], __int_as_float(p[4]), __int_as_float(p[5]), &m);
    }
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) {
            v *= inv_temp;
            if (v < thr) continue;
            v += gumbel(seed, step, (unsigned)i);
        }
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
    if (threadIdx.x == 0) tokens[blockIdx.x] = best_i[0];
}

extern "C" __global__ void bump(int* pos, int* count) {
    pos[threadIdx.x] += 1;
    if (threadIdx.x == 0) count[0] += 1;
}

extern "C" __global__ void copy_rows(const float* src, float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void split_rows(const float* src, bf16_t* hi, bf16_t* lo, int n) {
    for (int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += 2 * gridDim.x * blockDim.x)
        split_store2(hi, lo, i, src[i], src[i + 1]);
}
