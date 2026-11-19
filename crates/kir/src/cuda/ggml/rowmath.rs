use crate::cuda::gemm::prim;
use super::rows::{self, Format, Layout, RowLoop};

const KQ_LANE_SETUP: &str = "    int j = (lane & 7) >> 1;\n    int l0 = (lane & 1) * 16;\n";
const Q6_LANE_SETUP: &str = "    int h = (lane & 7) >> 2;\n    int l0 = (lane & 3) * 8;\n";

fn ind(text: &str, n: usize) -> String {
    let pad = " ".repeat(n);
    text.lines().map(|l| if l.starts_with('#') { format!("{l}\n") } else { format!("{pad}{l}\n") }).collect()
}

fn lanes(v: &str) -> [String; 4] {
    ["x", "y", "z", "w"].map(|c| format!("{v}.{c}"))
}

fn sum4(v: &str) -> String {
    lanes(v).join(" + ")
}

fn byte_at(arr: &str, i: &str) -> String {
    format!("({arr}[{i} >> 2] >> (8 * ({i} & 3))) & 0xffu")
}

fn nibble(word: &str, shift: usize) -> String {
    match shift {
        0 => format!("(float)({word} & 15u)"),
        28 => format!("(float)({word} >> 28)"),
        s => format!("(float)(({word} >> {s}) & 15u)"),
    }
}

fn nibble_dot(word: &str, first_shift: usize, x: &str) -> String {
    lanes(x).iter().enumerate().map(|(i, c)| format!("{} * {c}", nibble(word, first_shift + 8 * i))).collect::<Vec<_>>().join(" + ")
}

fn scale_min_pair(head: &str) -> String {
    format!(
        "unsigned scales[3] = {{{h}.y, {h}.z, {h}.w}};\nfloat sc1, m1, sc2, m2;\ngg_scale_min(scales, 2 * j, sc1, m1);\ngg_scale_min(scales, 2 * j + 1, sc2, m2);\n",
        h = head
    )
}

fn half_pair(head: &str) -> String {
    format!("float d = gg_f16((unsigned short)({head}.x & 0xffffu));\nfloat dmin = gg_f16((unsigned short)({head}.x >> 16));\n")
}

fn kq_combine() -> &'static str {
    "d * (sc1 * lo + sc2 * hi) - dmin * (m1 * slo + m2 * shi)"
}

fn kq_x_loads(rows_form: bool) -> String {
    let sums = |a: &str, b: &str| format!("slo += {};\nshi += {};\n", sum4(a), sum4(b));
    if rows_form {
        format!(
            "        const float* xs = x + sl * GG_SB + 64 * j + l0;\n        float4 a[4], b[4];\n        float slo = 0.0f, shi = 0.0f;\n#pragma unroll\n        for (int w4 = 0; w4 < 4; w4++) {{\n            a[w4] = *(const float4*)(xs + 4 * w4);\n            b[w4] = *(const float4*)(xs + 32 + 4 * w4);\n{}        }}\n",
            ind(&sums("a[w4]", "b[w4]"), 12)
        )
    } else {
        String::new()
    }
}

fn kquant_rows(f: &Format, qs: usize) -> String {
    let math = format!(
        "{half}{scales}unsigned words[4] = {{q[r].x, q[r].y, q[r].z, q[r].w}};\nfloat lo = 0.0f, hi = 0.0f;\n#pragma unroll\nfor (int w4 = 0; w4 < 4; w4++) {{\n    unsigned v = words[w4];\n    lo += {lo};\n    hi += {hi};\n}}\nfloat part = {comb};\nacc[r] += valid ? part : 0.0f;\n",
        half = half_pair("head[r]"),
        scales = scale_min_pair("head[r]"),
        lo = nibble_dot("v", 0, "a[w4]"),
        hi = nibble_dot("v", 4, "b[w4]"),
        comb = kq_combine(),
    );
    rows::dequant_rows(&RowLoop {
        tag: f.tag,
        setup: KQ_LANE_SETUP,
        decls: "        uint4 head[R], q[R];\n",
        row_loads: &format!("head[r] = load_stream(sb);\nq[r] = load_stream(sb + {qs} + 32 * j + l0);\n"),
        shared_loads: &kq_x_loads(true),
        row_math: &math,
    })
}

fn kquant_dot(f: &Format, qs: usize) -> String {
    let body = format!(
        "uint4 head = __ldg((const uint4*)sb);\nuint4 q = __ldg((const uint4*)(sb + {qs} + 32 * j + l0));\n{half}{scales}const float* xs = x + s * GG_SB + 64 * j + l0;\nunsigned words[4] = {{q.x, q.y, q.z, q.w}};\nfloat lo = 0.0f, hi = 0.0f, slo = 0.0f, shi = 0.0f;\n#pragma unroll\nfor (int w4 = 0; w4 < 4; w4++) {{\n    float4 a = *(const float4*)(xs + 4 * w4);\n    float4 b = *(const float4*)(xs + 32 + 4 * w4);\n    unsigned v = words[w4];\n    lo += {lo};\n    hi += {hi};\n    slo += {sa};\n    shi += {sb};\n}}\nacc += {comb};\n",
        half = half_pair("head"),
        scales = scale_min_pair("head"),
        lo = nibble_dot("v", 0, "a"),
        hi = nibble_dot("v", 4, "b"),
        sa = sum4("a"),
        sb = sum4("b"),
        comb = kq_combine(),
    );
    rows::dequant_dot(f.tag, KQ_LANE_SETUP, &ind(&body, 8))
}

fn q6_value(k: usize, qh: &str) -> String {
    let plane = if k & 1 == 0 { "av" } else { "bv" };
    let low = if k < 2 { format!("({plane} & 15u)") } else { format!("({plane} >> 4)") };
    let high = match k {
        0 => format!("(({qh} & 3u) << 4)"),
        _ => format!("((({qh} >> {}) & 3u) << 4)", 2 * k),
    };
    format!("(float)((int)({low} | {high}) - 32)")
}

fn q6_partials(x: &dyn Fn(usize) -> String, indent: usize) -> String {
    let mut s = String::from("float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n#pragma unroll\nfor (int i = 0; i < 8; i++) {\n");
    for p in ["a", "b", "c"] {
        s.push_str(&format!("    unsigned {p}v = {};\n", byte_at(p, "i")));
    }
    for k in 0..4 {
        s.push_str(&format!("    part[{k}] += {} * {};\n", q6_value(k, "cv"), x(k)));
    }
    s.push_str("}\n");
    ind(&s, indent)
}

fn q6_words(suffix: &str) -> String {
    let w = |n: &str, q: &str| format!("{n}[2] = {{{q}{suffix}.x, {q}{suffix}.y}}");
    format!("unsigned {}, {}, {};\n", w("a", "qa"), w("b", "qb"), w("c", "qc"))
}

fn q6k_rows(f: &Format, qh: usize, scales: usize, d: usize) -> String {
    let mut math = q6_words("[r]");
    math.push_str(&q6_partials(&|k| format!("xv[{k}][i]"), 0));
    math.push_str("int sh = 8 * (l0 / 16);\nfloat sum = 0.0f;\n");
    math.push_str(&prim::unroll_nest(0, &[("k", "4")], true, "unsigned word = sc[r][k >> 1];\nsum += (float)(signed char)((word >> (16 * (k & 1) + sh)) & 0xffu) * part[k];"));
    math.push_str("float d = gg_f16(dh[r]);\nacc[r] += valid ? d * sum : 0.0f;\n");
    let row_loads = format!(
        "qa[r] = load_stream8(sb + 64 * h + l0);\nqb[r] = load_stream8(sb + 64 * h + 32 + l0);\nqc[r] = load_stream8(sb + {qh} + 32 * h + l0);\nunsigned w = __ldg((const unsigned*)(sb + {scales} + 8 * h));\nunsigned w2 = __ldg((const unsigned*)(sb + {} + 8 * h));\nsc[r][0] = w;\nsc[r][1] = w2;\ndh[r] = __ldg((const unsigned short*)(sb + {d}));\n",
        scales + 4
    );
    let shared = "        const float* xs = x + sl * GG_SB + 128 * h + l0;\n        float xv[4][8];\n#pragma unroll\n        for (int k = 0; k < 4; k++) {\n            float4 u = *(const float4*)(xs + 32 * k);\n            float4 v = *(const float4*)(xs + 32 * k + 4);\n            xv[k][0] = u.x; xv[k][1] = u.y; xv[k][2] = u.z; xv[k][3] = u.w;\n            xv[k][4] = v.x; xv[k][5] = v.y; xv[k][6] = v.z; xv[k][7] = v.w;\n        }\n";
    rows::dequant_rows(&RowLoop {
        tag: f.tag,
        setup: Q6_LANE_SETUP,
        decls: "        uint2 qa[R], qb[R], qc[R];\n        unsigned sc[R][2];\n        unsigned short dh[R];\n",
        row_loads: &row_loads,
        shared_loads: shared,
        row_math: &math,
    })
}

fn q6k_dot(f: &Format, qh: usize, scales: usize, d: usize) -> String {
    let mut body = format!(
        "uint2 qa = __ldg((const uint2*)(sb + 64 * h + l0));\nuint2 qb = __ldg((const uint2*)(sb + 64 * h + 32 + l0));\nuint2 qc = __ldg((const uint2*)(sb + {qh} + 32 * h + l0));\nfloat d = gg_f16(gg_u16(sb + {d}));\nconst unsigned char* sc = sb + {scales} + 8 * h + l0 / 16;\nconst float* xs = x + s * GG_SB + 128 * h + l0;\n"
    );
    body.push_str(&q6_words(""));
    body.push_str(&q6_partials(&|k| if k == 0 { "xs[i]".to_string() } else { format!("xs[i + {}]", 32 * k) }, 0));
    body.push_str("float sum = 0.0f;\n");
    body.push_str(&prim::unroll_nest(0, &[("k", "4")], false, "sum += (float)(signed char)__ldg(sc + 2 * k) * part[k];"));
    body.push_str("acc += d * sum;\n");
    rows::dequant_dot(f.tag, Q6_LANE_SETUP, &ind(&body, 8))
}

pub fn rows_kernel(f: &Format) -> String {
    match f.layout {
        Layout::KQuant { qs, qh: None } => kquant_rows(f, qs),
        Layout::Q6K { qh, scales, d } => q6k_rows(f, qh, scales, d),
        _ => panic!("{} has no fused rows kernel", f.tag),
    }
}

pub fn dot_kernel(f: &Format) -> String {
    match f.layout {
        Layout::KQuant { qs, qh: None } => kquant_dot(f, qs),
        Layout::Q6K { qh, scales, d } => q6k_dot(f, qh, scales, d),
        _ => panic!("{} has no fused dot", f.tag),
    }
}

pub fn fused_rows() -> String {
    rows::FORMATS.iter().filter(|f| f.fused_rows).map(rows_kernel).collect()
}

pub fn fused_dots() -> String {
    rows::FORMATS.iter().filter(|f| f.fused_rows).map(dot_kernel).collect()
}
