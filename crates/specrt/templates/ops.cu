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
constexpr float EPS = {{EPS}};
constexpr float EMBED_SCALE = {{EMBED_SCALE}};
constexpr int ACT_KIND = {{ACT_KIND}};
constexpr int Q8_GROUP = {{Q8_GROUP}};
constexpr int FP8_BLOCK = {{FP8_BLOCK}};
constexpr int LINEAR_WARPS = 8;
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

__device__ __forceinline__ void split_store2(bf16_t* hi, bf16_t* lo, size_t at, float a, float b) {
    unsigned ha, la, hb, lb;
    split_hi_lo(a, ha, la);
    split_hi_lo(b, hb, lb);
    *(unsigned*)(hi + at) = ha | (hb << 16);
    *(unsigned*)(lo + at) = la | (lb << 16);
}

template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR == 4 || CPR % 8 == 0 || CPR % 8 == 4, "head_dim of 32, or a multiple of 32 or 32 + 64k");
    constexpr int GROUP_ROWS = CPR >= 8 ? 1 : 8 / CPR;
    constexpr int MASK = CPR >= 8 ? 7 : CPR - 1;
    int m = (r / GROUP_ROWS) & MASK;
    if (CPR % 8 == 4 && CPR > 8 && c >= CPR - 4) m &= 3;
    return r * CPR + (c ^ m);
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

__device__ __forceinline__ unsigned f32x2_to_e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

constexpr float FP8_MAX = 448.0f;

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
        x[i] = bf16_to_f32(row[i]) * EMBED_SCALE;
}

extern "C" __global__ void splice_rows(float* x, const int* rows, const float* embeds) {
    int row = rows[blockIdx.y];
    if (row < 0) return;
    float* dst = x + (size_t)blockIdx.y * HIDDEN;
    const float* src = embeds + (size_t)row * HIDDEN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

__device__ float gumbel(unsigned seed, unsigned step, unsigned i) {
    unsigned x = seed * 747796405u + step * 2891336453u + i * 2246822519u + 1u;
    x ^= x >> 16; x *= 2246822507u; x ^= x >> 13; x *= 3266489909u; x ^= x >> 16;
    float u = ((float)(x >> 8) + 0.5f) / 16777216.0f;
    return -logf(-logf(u));
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

extern "C" __global__ void apply_mask(float* logits, const unsigned* mask, const int* flags, int words) {
    int row = blockIdx.x;
    if (!flags[row]) return;
    logits += (size_t)row * VOCAB;
    mask += (size_t)row * words;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < VOCAB; i += gridDim.y * blockDim.x) {
        if (!((mask[i >> 5] >> (i & 31)) & 1u)) logits[i] = NEG_INF;
    }
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

extern "C" __global__ void argmax_advance(const float* logits, int* token, int* pos, int* history, int* count, float inv_temp, unsigned seed, int top_k, float top_p, float min_p) {
    __shared__ float best_v[1024];
    __shared__ int best_i[1024];
    float thr = NEG_INF;
    if (inv_temp > 0.0f) {
        float m;
        thr = sample_threshold(logits, inv_temp, top_k, top_p, min_p, &m);
    }
    float bv = NEG_INF;
    int bi = 0;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
        float v = logits[i];
        if (inv_temp > 0.0f) {
            v *= inv_temp;
            if (v < thr) continue;
            v += gumbel(seed, (unsigned)count[0], (unsigned)i);
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
    if (threadIdx.x == 0) {
        token[0] = best_i[0];
        history[count[0]] = best_i[0];
        count[0] += 1;
        pos[0] += 1;
    }
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
