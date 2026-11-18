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
