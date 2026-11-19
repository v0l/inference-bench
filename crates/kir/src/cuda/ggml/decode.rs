use super::dequant;
use super::rows::{self, RowLoop};

pub fn body() -> String {
    [
        rows::format_ids(),
        r##"__device__ __forceinline__ float gg_f16(unsigned short h) {
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

"##.to_string(),
        rows::bytes_fn(),
        r##"__device__ __forceinline__ unsigned gg_byte(const unsigned* s, int i) {
    return (s[i >> 2] >> (8 * (i & 3))) & 0xffu;
}

"##.to_string(),
        dequant::scale_min(),
        dequant::dequant8(),
        r##"__device__ __forceinline__ const unsigned char* gg_row(const void* w, const unsigned* table, int row, int& type) {
    unsigned e = __ldg(table + row);
    type = (int)(e & 7u);
    return (const unsigned char*)w + (size_t)(e >> 3) * 16;
}

__device__ __forceinline__ void gg_load8(const void* w, const unsigned* table, int row, int k, float* out) {
    int type;
    const unsigned char* base = gg_row(w, table, row, type);
    gg_dequant8(base + (size_t)(k / GG_SB) * gg_bytes(type), type, (k % GG_SB) / 8, out);
}

"##.to_string(),
        rows::dequant_rows(&RowLoop {
            tag: "Q4K",
            setup: r##"    int j = (lane & 7) >> 1;
    int l0 = (lane & 1) * 16;
"##,
            decls: r##"        uint4 head[R], q[R];
"##,
            row_loads: r##"head[r] = load_stream(sb);
q[r] = load_stream(sb + 16 + 32 * j + l0);
"##,
            shared_loads: r##"        const float* xs = x + sl * GG_SB + 64 * j + l0;
        float4 a[4], b[4];
        float slo = 0.0f, shi = 0.0f;
#pragma unroll
        for (int w4 = 0; w4 < 4; w4++) {
            a[w4] = *(const float4*)(xs + 4 * w4);
            b[w4] = *(const float4*)(xs + 32 + 4 * w4);
            slo += a[w4].x + a[w4].y + a[w4].z + a[w4].w;
            shi += b[w4].x + b[w4].y + b[w4].z + b[w4].w;
        }
"##,
            row_math: r##"float d = gg_f16((unsigned short)(head[r].x & 0xffffu));
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
"##,
        }),
        rows::dequant_rows(&RowLoop {
            tag: "Q6K",
            setup: r##"    int h = (lane & 7) >> 2;
    int l0 = (lane & 3) * 8;
"##,
            decls: r##"        uint2 qa[R], qb[R], qc[R];
        unsigned sc[R][2];
        unsigned short dh[R];
"##,
            row_loads: r##"qa[r] = load_stream8(sb + 64 * h + l0);
qb[r] = load_stream8(sb + 64 * h + 32 + l0);
qc[r] = load_stream8(sb + 128 + 32 * h + l0);
unsigned w = __ldg((const unsigned*)(sb + 192 + 8 * h));
unsigned w2 = __ldg((const unsigned*)(sb + 196 + 8 * h));
sc[r][0] = w;
sc[r][1] = w2;
dh[r] = __ldg((const unsigned short*)(sb + 208));
"##,
            shared_loads: r##"        const float* xs = x + sl * GG_SB + 128 * h + l0;
        float xv[4][8];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float4 u = *(const float4*)(xs + 32 * k);
            float4 v = *(const float4*)(xs + 32 * k + 4);
            xv[k][0] = u.x; xv[k][1] = u.y; xv[k][2] = u.z; xv[k][3] = u.w;
            xv[k][4] = v.x; xv[k][5] = v.y; xv[k][6] = v.z; xv[k][7] = v.w;
        }
"##,
            row_math: r##"unsigned a[2] = {qa[r].x, qa[r].y}, b[2] = {qb[r].x, qb[r].y}, c[2] = {qc[r].x, qc[r].y};
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
"##,
        }),
        rows::dequant_dot("Q4K", r##"    int j = (lane & 7) >> 1;
    int l0 = (lane & 1) * 16;
"##, r##"        uint4 head = __ldg((const uint4*)sb);
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
"##),
        rows::dequant_dot("Q6K", r##"    int h = (lane & 7) >> 2;
    int l0 = (lane & 3) * 8;
"##, r##"        uint2 qa = __ldg((const uint2*)(sb + 64 * h + l0));
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
"##),
        dequant::row_dot(),
        dequant::dequant_rows_kernel(),
    ]
    .concat()
}
