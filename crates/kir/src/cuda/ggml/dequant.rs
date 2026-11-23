use super::rows::{FORMATS, Layout};
use crate::cuda::gemm::prim;

fn id(tag: &str) -> String {
    format!("GG_{tag}")
}

fn kquants() -> Vec<(&'static str, usize, Option<usize>)> {
    FORMATS
        .iter()
        .filter_map(|f| match f.layout {
            Layout::KQuant { qs, qh } => Some((f.tag, qs, qh)),
            _ => None,
        })
        .collect()
}

fn cond(tags: &[&str]) -> String {
    tags.iter().map(|t| format!("type == {}", id(t))).collect::<Vec<_>>().join(" || ")
}

fn select(items: &[(&str, usize)]) -> String {
    let (last, rest) = items.split_last().unwrap();
    if rest.is_empty() {
        return last.1.to_string();
    }
    let mut out = last.1.to_string();
    for (t, v) in rest.iter().rev() {
        out = format!("type == {} ? {} : {}", id(t), v, out);
    }
    out
}

fn kquant_branch() -> String {
    let ks = kquants();
    let tags: Vec<&str> = ks.iter().map(|k| k.0).collect();
    let qs = select(&ks.iter().map(|k| (k.0, k.1)).collect::<Vec<_>>());
    let qh: Vec<(&str, usize)> = ks.iter().filter_map(|k| k.2.map(|o| (k.0, o))).collect();
    let high = if qh.is_empty() {
        String::new()
    } else {
        let htags: Vec<&str> = qh.iter().map(|k| k.0).collect();
        let off = qh[0].1;
        let off4 = off + 4;
        format!(
            "        if ({}) {{\n            qh[0] = gg_u32(sb + {off} + l);\n            qh[1] = gg_u32(sb + {off4} + l);\n        }}\n",
            cond(&htags)
        )
    };
    let paren = if ks.len() > 1 { format!("({qs})") } else { qs };
    r##"    if (@COND@) {
        float d = gg_f16(gg_u16(sb));
        float dmin = gg_f16(gg_u16(sb + 2));
        unsigned s[3] = {gg_u32(sb + 4), gg_u32(sb + 8), gg_u32(sb + 12)};
        int j = c >> 3;
        int half = (c >> 2) & 1;
        int l = (c & 3) * 8;
        float sc, m;
        gg_scale_min(s, 2 * j + half, sc, m);
        float dl = __fmul_rn(d, sc), ml = __fmul_rn(dmin, m);
        const unsigned char* qp = sb + @QS@ + 32 * j + l;
        unsigned q[2] = {gg_u32(qp), gg_u32(qp + 4)};
        unsigned qh[2] = {0u, 0u};
@HIGH@        int bit = 2 * j + half;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned byte = (q[i >> 2] >> (8 * (i & 3))) & 0xffu;
            unsigned nib = half ? byte >> 4 : byte & 15u;
            unsigned hb = ((qh[i >> 2] >> (8 * (i & 3))) >> bit) & 1u;
            w[i] = __fsub_rn(__fmul_rn(dl, (float)(nib + 16u * hb)), ml);
        }
    }"##
        .replace("@COND@", &cond(&tags))
        .replace("@QS@", &paren)
        .replace("@HIGH@", &high)
}

fn q6k_branch(tag: &str, qh: usize, scales: usize, d: usize) -> String {
    r##" else if (type == @ID@) {
        int h = c >> 4;
        int k = (c >> 2) & 3;
        int l = (c & 3) * 8;
        float d = gg_f16(gg_u16(sb + @D@));
        float ds = __fmul_rn(d, (float)(signed char)__ldg(sb + @SC@ + 8 * h + l / 16 + 2 * k));
        const unsigned char* qlp = sb + 64 * h + (k & 1) * 32 + l;
        const unsigned char* qhp = sb + @QH@ + 32 * h + l;
        unsigned ql[2] = {gg_u32(qlp), gg_u32(qlp + 4)};
        unsigned qh[2] = {gg_u32(qhp), gg_u32(qhp + 4)};
        int ls = (k >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned lo = ((ql[i >> 2] >> (8 * (i & 3) + ls)) & 15u);
            unsigned hi = ((qh[i >> 2] >> (8 * (i & 3) + 2 * k)) & 3u);
            w[i] = __fmul_rn(ds, (float)((int)(lo | (hi << 4)) - 32));
        }
    }"##
        .replace("@ID@", &id(tag))
        .replace("@D@", &d.to_string())
        .replace("@SC@", &scales.to_string())
        .replace("@QH@", &qh.to_string())
}

fn blocks_branch(tag: &str, block: usize) -> String {
    r##" else if (type == @ID@) {
        const unsigned char* bp = sb + @B@ * (c >> 2);
        float d = gg_f16(gg_u16(bp));
        int e = (c & 3) * 8;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned two = gg_u16(bp + 2 + e + 2 * k);
            w[2 * k] = __fmul_rn((float)(signed char)(two & 0xffu), d);
            w[2 * k + 1] = __fmul_rn((float)(signed char)(two >> 8), d);
        }
    }"##
        .replace("@ID@", &id(tag))
        .replace("@B@", &block.to_string())
}

const HALF_BRANCH: &str = r##" else {
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
"##;

pub fn dequant8() -> String {
    let mut body = kquant_branch();
    for f in FORMATS.iter() {
        match f.layout {
            Layout::Q6K { qh, scales, d } => body.push_str(&q6k_branch(f.tag, qh, scales, d)),
            Layout::Blocks { block } => body.push_str(&blocks_branch(f.tag, block)),
            Layout::KQuant { .. } => {}
        }
    }
    body.push_str("    else if (type >= 8) {\n        gg_dequant8_iq(sb, type, c, w);\n    }\n");
    body.push_str(HALF_BRANCH);
    prim::device("void", "gg_dequant8", "const unsigned char* sb, int type, int c, float* w", &body)
}

pub fn scale_min() -> String {
    prim::device(
        "void",
        "gg_scale_min",
        "const unsigned* s, int j, float& sc, float& m",
        r##"    unsigned a, b;
    if (j < 4) {
        a = gg_byte(s, j) & 63u;
        b = gg_byte(s, j + 4) & 63u;
    } else {
        a = (gg_byte(s, j + 4) & 0xfu) | ((gg_byte(s, j - 4) >> 6) << 4);
        b = (gg_byte(s, j + 4) >> 4) | ((gg_byte(s, j) >> 6) << 4);
    }
    sc = (float)a;
    m = (float)b;
"##,
    )
}

pub fn fused_tags() -> Vec<&'static str> {
    FORMATS.iter().filter(|f| f.fused_rows).map(|f| f.tag).collect()
}

pub fn row_dot() -> String {
    let dispatch: String = fused_tags()
        .iter()
        .map(|t| format!("    if (type == {}) return gg_{}_dot<K>(base, x, lane);\n", id(t), t.to_lowercase()))
        .collect();
    let body = r##"    int type;
    const unsigned char* base = gg_row(w, table, row, type);
@DISPATCH@    int bytes = gg_bytes(type);
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
"##
    .replace("@DISPATCH@", &dispatch);
    prim::device_t("int K", "float", "gg_row_dot", "const void* w, const unsigned* table, int row, const float* x, int lane", &body)
}

pub fn dequant_rows_kernel() -> String {
    r##"extern "C" __global__ void gg_dequant_rows(const void* w, const unsigned* table, float* out, int rows, int cols) {
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
"##
    .to_string()
}

pub fn rows_dispatch() -> String {
    let tags = fused_tags();
    let mut out = String::new();
    for (i, t) in tags.iter().enumerate() {
        let lead = if i == 0 { "        if" } else { " else if" };
        out.push_str(&format!("{lead} (same && type[0] == {}) {{\n            gg_{}_rows<K, ROWS>(base, x, lane, out);\n        }}", id(t), t.to_lowercase()));
    }
    out.push_str(" else {\n");
    out
}
