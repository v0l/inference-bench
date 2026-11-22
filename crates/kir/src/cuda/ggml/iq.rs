use crate::cuda::gemm::prim;
use crate::iq_tables::*;

pub const EXT_CODE: usize = 7;
pub const FIRST_ID: usize = 8;
pub const HEADER_BYTES: usize = 16;
pub const FORMATS: [(&str, usize); 9] = [
    ("IQ2XXS", 66),
    ("IQ2XS", 74),
    ("IQ2S", 82),
    ("IQ3XXS", 98),
    ("IQ3S", 110),
    ("IQ1S", 50),
    ("IQ1M", 56),
    ("IQ4NL", 144),
    ("IQ4XS", 136),
];

pub fn table_row() -> String {
    prim::device(
        "const unsigned char*",
        "gg_row",
        "const void* w, const unsigned* table, int row, int& type",
        &format!(
            "    unsigned e = __ldg(table + row);\n    type = (int)(e & 7u);\n    const unsigned char* p = (const unsigned char*)w + (size_t)(e >> 3) * 16;\n    if (type == {EXT_CODE}) {{\n        type = *(const int*)p;\n        p += {HEADER_BYTES};\n    }}\n    return p;\n"
        ),
    )
}

pub fn ids() -> String {
    FORMATS.iter().enumerate().map(|(i, (tag, _))| format!("constexpr int GG_{tag} = {};\n", FIRST_ID + i)).collect()
}

pub fn bytes_cases() -> String {
    FORMATS.iter().map(|(tag, bytes)| format!("        case GG_{tag}: return {bytes};\n")).collect()
}

pub fn table_u64(name: &str, vals: &[u64]) -> String {
    let body: Vec<String> = vals.iter().map(|v| format!("0x{v:x}ull")).collect();
    format!("static const __device__ unsigned long long {name}[{}] = {{{}}};\n", vals.len(), body.join(","))
}

fn table_u32(name: &str, vals: &[u32]) -> String {
    let body: Vec<String> = vals.iter().map(|v| format!("0x{v:x}u")).collect();
    format!("static const __device__ unsigned {name}[{}] = {{{}}};\n", vals.len(), body.join(","))
}

fn table_u8(name: &str, vals: &[u8]) -> String {
    let body: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    format!("static const __device__ unsigned char {name}[{}] = {{{}}};\n", vals.len(), body.join(","))
}

fn table_i8(name: &str, vals: &[i8]) -> String {
    let body: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    format!("static const __device__ signed char {name}[{}] = {{{}}};\n", vals.len(), body.join(","))
}

pub fn tables() -> String {
    [
        table_u8("gg_ksigns", &KSIGNS_IQ2XS),
        table_u64("gg_iq2xxs_grid", &IQ2XXS_GRID),
        table_u64("gg_iq2xs_grid", &IQ2XS_GRID),
        table_u64("gg_iq2s_grid", &IQ2S_GRID),
        table_u32("gg_iq3xxs_grid", &IQ3XXS_GRID),
        table_u32("gg_iq3s_grid", &IQ3S_GRID),
        table_u64("gg_iq1s_grid", &IQ1S_GRID),
        table_i8("gg_kvalues_iq4nl", &KVALUES_IQ4NL),
        "\n".to_string(),
    ]
    .concat()
}

fn branch(tag: &str, body: &str) -> String {
    format!("    if (type == GG_{tag}) {{\n{body}        return;\n    }}\n")
}

pub fn dequant8_iq() -> String {
    let mut body = String::new();
    body.push_str("    int ib = c >> 2;\n    int l = c & 3;\n");
    body.push_str(&branch(
        "IQ2XXS",
        r##"        float d = gg_f16(gg_u16(sb));
        const unsigned char* p = sb + 2 + 8 * ib;
        unsigned aux = gg_u16(p + 4) | (gg_u16(p + 6) << 16);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        unsigned long long g = gg_iq2xxs_grid[p[l]];
        unsigned signs = gg_ksigns[(aux >> (7 * l)) & 127u];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sign(signs, j, db * gg_gb(g, j));
"##,
    ));
    body.push_str(&branch(
        "IQ2XS",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned sc = sb[66 + ib];
        float db = d * (0.5f + (float)((l >> 1) ? sc >> 4 : sc & 15u)) * 0.25f;
        unsigned q = gg_u16(sb + 2 + 2 * c);
        unsigned long long g = gg_iq2xs_grid[q & 511u];
        unsigned signs = gg_ksigns[q >> 9];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sign(signs, j, db * gg_gb(g, j));
"##,
    ));
    body.push_str(&branch(
        "IQ2S",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned sc = sb[74 + ib];
        float db = d * (0.5f + (float)((l >> 1) ? sc >> 4 : sc & 15u)) * 0.25f;
        unsigned qh = sb[66 + ib];
        unsigned long long g = gg_iq2s_grid[sb[2 + c] | ((qh << (8 - 2 * l)) & 0x300u)];
        unsigned signs = sb[34 + c];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = gg_sign(signs, j, db * gg_gb(g, j));
"##,
    ));
    body.push_str(&branch(
        "IQ3XXS",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned aux = gg_u16(sb + 66 + 4 * ib) | (gg_u16(sb + 68 + 4 * ib) << 16);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        unsigned signs = gg_ksigns[(aux >> (7 * l)) & 127u];
        unsigned g1 = gg_iq3xxs_grid[sb[2 + 2 * c]];
        unsigned g2 = gg_iq3xxs_grid[sb[3 + 2 * c]];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            w[j] = gg_sign(signs, j, db * gg_gb(g1, j));
            w[j + 4] = gg_sign(signs, j + 4, db * gg_gb(g2, j));
        }
"##,
    ));
    body.push_str(&branch(
        "IQ3S",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned sc = sb[106 + (ib >> 1)];
        float db = d * (float)(1 + 2 * (int)((ib & 1) ? sc >> 4 : sc & 15u));
        unsigned qh = sb[66 + ib];
        unsigned g1 = gg_iq3s_grid[sb[2 + 2 * c] | ((qh << (8 - 2 * l)) & 256u)];
        unsigned g2 = gg_iq3s_grid[sb[3 + 2 * c] | ((qh << (7 - 2 * l)) & 256u)];
        unsigned signs = sb[74 + c];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            w[j] = gg_sign(signs, j, db * gg_gb(g1, j));
            w[j + 4] = gg_sign(signs, j + 4, db * gg_gb(g2, j));
        }
"##,
    ));
    body.push_str(&branch(
        "IQ1S",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned qh = gg_u16(sb + 34 + 2 * ib);
        float dl = d * (float)(2 * (int)((qh >> 12) & 7u) + 1);
        float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
        unsigned long long g = gg_iq1s_grid[sb[2 + c] | (((qh >> (3 * l)) & 7u) << 8)];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = dl * (gg_gbs(g, j) + delta);
"##,
    ));
    body.push_str(&branch(
        "IQ1M",
        r##"        unsigned s0 = gg_u16(sb + 48), s1 = gg_u16(sb + 50), s2 = gg_u16(sb + 52), s3 = gg_u16(sb + 54);
        float d = gg_f16((unsigned short)((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u)));
        unsigned sc = gg_u16(sb + 48 + 2 * (ib >> 1));
        int shift = 6 * (ib & 1) + ((l >> 1) ? 3 : 0);
        float dl = d * (float)(2 * (int)((sc >> shift) & 7u) + 1);
        unsigned qh = sb[32 + (c >> 1)];
        bool odd = (l & 1) != 0;
        unsigned idx = sb[c] | ((qh << (odd ? 4 : 8)) & 0x700u);
        float delta = (qh & (odd ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        unsigned long long g = gg_iq1s_grid[idx];
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = dl * (gg_gbs(g, j) + delta);
"##,
    ));
    body.push_str(&branch(
        "IQ4NL",
        r##"        const unsigned char* b = sb + 18 * ib;
        float d = gg_f16(gg_u16(b));
        const unsigned char* q = b + 2 + 8 * (l & 1);
        bool hi = (l >> 1) != 0;
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = d * (float)gg_kvalues_iq4nl[hi ? q[j] >> 4 : q[j] & 15u];
"##,
    ));
    body.push_str(&branch(
        "IQ4XS",
        r##"        float d = gg_f16(gg_u16(sb));
        unsigned sh = gg_u16(sb + 2);
        int ls = (int)((sb[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15u) | (int)(((sh >> (2 * ib)) & 3u) << 4);
        float dl = d * (float)(ls - 32);
        const unsigned char* q = sb + 8 + 16 * ib + 8 * (l & 1);
        bool hi = (l >> 1) != 0;
#pragma unroll
        for (int j = 0; j < 8; j++) w[j] = dl * (float)gg_kvalues_iq4nl[hi ? q[j] >> 4 : q[j] & 15u];
"##,
    ));
    [
        prim::device("float", "gg_gb", "unsigned long long g, int j", "    return (float)((g >> (8 * j)) & 255ull);\n"),
        prim::device("float", "gg_gbs", "unsigned long long g, int j", "    return (float)(signed char)((g >> (8 * j)) & 255ull);\n"),
        prim::device("float", "gg_sign", "unsigned s, int j, float v", "    return ((s >> j) & 1u) ? -v : v;\n"),
        prim::device("void", "gg_dequant8_iq", "const unsigned char* sb, int type, int c, float* w", &body),
    ]
    .concat()
}
