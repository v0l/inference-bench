use crate::cuda::gemm::prim;

pub enum Layout {
    KQuant { qs: usize, qh: Option<usize> },
    Q6K { qh: usize, scales: usize, d: usize },
    Blocks { block: usize },
}

pub struct Format {
    pub tag: &'static str,
    pub bytes: usize,
    pub layout: Layout,
    pub fused_rows: bool,
}

pub const FORMATS: [Format; 4] = [
    Format { tag: "Q4K", bytes: 144, layout: Layout::KQuant { qs: 16, qh: None }, fused_rows: true },
    Format { tag: "Q5K", bytes: 176, layout: Layout::KQuant { qs: 48, qh: Some(16) }, fused_rows: false },
    Format { tag: "Q6K", bytes: 224, layout: Layout::Q6K { qh: 128, scales: 192, d: 208 }, fused_rows: true },
    Format { tag: "Q8", bytes: 272, layout: Layout::Blocks { block: 34 }, fused_rows: false },
];
pub const DEFAULT_BYTES: usize = 512;

pub fn format_ids() -> String {
    let mut out = String::new();
    for (i, f) in FORMATS.iter().enumerate() {
        out.push_str(&format!("constexpr int GG_{} = {i};\n", f.tag));
    }
    out.push_str(&format!("constexpr int GG_BF16 = {};\nconstexpr int GG_SB = 256;\n", FORMATS.len()));
    out.push_str(&super::iq::ids());
    out.push('\n');
    out
}

pub fn bytes_fn() -> String {
    let mut body = String::from("    switch (type) {\n");
    for f in FORMATS.iter() {
        body.push_str(&format!("        case GG_{}: return {};\n", f.tag, f.bytes));
    }
    body.push_str(&super::iq::bytes_cases());
    body.push_str(&format!("        default: return {DEFAULT_BYTES};\n    }}\n"));
    prim::device("int", "gg_bytes", "int type", &body)
}

pub fn bytes_of(tag: &str) -> usize {
    FORMATS.iter().find(|f| f.tag == tag).map(|f| f.bytes).unwrap()
}

pub struct RowLoop<'a> {
    pub tag: &'a str,
    pub setup: &'a str,
    pub decls: &'a str,
    pub row_loads: &'a str,
    pub shared_loads: &'a str,
    pub row_math: &'a str,
}

pub fn dequant_rows(f: &RowLoop) -> String {
    let bytes = bytes_of(f.tag);
    let name = f.tag.to_lowercase();
    let acc_zero = prim::unroll_nest(4, &[("r", "R")], false, "acc[r] = 0.0f;");
    let row_loads = prim::unroll_nest(8, &[("r", "R")], true, &format!("const unsigned char* sb = base[r] + (size_t)sl * {bytes};\n{}", f.row_loads));
    let math = prim::unroll_nest(8, &[("r", "R")], true, f.row_math);
    let out = prim::unroll_nest(4, &[("r", "R")], false, "out[r] = acc[r];");
    prim::device_t(
        "int K, int R",
        "void",
        &format!("gg_{name}_rows"),
        "const unsigned char* const* base, const float* x, int lane, float* out",
        &format!(
            "    constexpr int NSB = K / GG_SB;\n{setup}    float acc[R];\n{acc_zero}#pragma unroll 2\n    for (int s0 = 0; s0 < NSB; s0 += 4) {{\n        int s = s0 + (lane >> 3);\n        bool valid = s < NSB;\n        int sl = valid ? s : NSB - 1;\n{decls}{row_loads}{shared}{math}    }}\n{out}",
            setup = f.setup,
            decls = f.decls,
            shared = f.shared_loads,
        ),
    )
}

pub fn dequant_dot(tag: &str, setup: &str, body: &str) -> String {
    let bytes = bytes_of(tag);
    let name = tag.to_lowercase();
    prim::device_t(
        "int K",
        "float",
        &format!("gg_{name}_dot"),
        "const unsigned char* base, const float* x, int lane",
        &format!(
            "{setup}    float acc = 0.0f;\n#pragma unroll 4\n    for (int s0 = 0; s0 < K / GG_SB; s0 += 4) {{\n        int s = s0 + (lane >> 3);\n        if (s >= K / GG_SB) break;\n        const unsigned char* sb = base + (size_t)s * {bytes};\n{body}    }}\n    return acc;\n"
        ),
    )
}
