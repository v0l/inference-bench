use crate::{Language, Target, fnv};

pub const SUPER_BLOCK: usize = 256;
pub const BLOCK_BYTES: [(&str, usize); 4] = [("Q4_K", 144), ("Q5_K", 176), ("Q6_K", 224), ("Q8_0", 272)];
pub const TABLE_KERNELS: &[&str] = &["gg_dequant_rows", "gg_dequant_bf16"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlKernel {
    pub target: Target,
}

impl GgmlKernel {
    pub fn new(t: &Target) -> GgmlKernel {
        GgmlKernel { target: *t }
    }

    pub fn describe(&self) -> String {
        format!("{}\nggml decoders Q4_K Q5_K Q6_K Q8_0 bf16 f16", self.target.describe())
    }

    pub fn hash(&self) -> String {
        fnv(self.describe().bytes())
    }

    pub fn emit(&self) -> Result<String, String> {
        match self.target.language {
            Language::Cuda => Ok(crate::cuda::emit_ggml()),
            Language::Wgsl => Ok(wgsl_decoder()),
        }
    }
}

const WGSL_BYTES: &str = "fn wbyte(i: u32) -> u32 {
    return (wdata[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn wi8(i: u32) -> f32 {
    return f32((i32(wbyte(i)) ^ 128) - 128);
}

fn whalf(i: u32) -> u32 {
    let w = wdata[i >> 2u];
    return select(w & 0xffffu, w >> 16u, (i & 2u) != 0u);
}

fn wf16(i: u32) -> f32 {
    return unpack2x16float(whalf(i)).x;
}

fn scale_min_k4(j: u32, sb: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(wbyte(sb + 4u + j) & 63u, wbyte(sb + 8u + j) & 63u);
    }
    let a = wbyte(sb + 4u + j + 4u);
    let lo = wbyte(sb + 4u + j - 4u);
    let hi = wbyte(sb + 4u + j);
    return vec2<u32>((a & 0xfu) | ((lo >> 6u) << 4u), (a >> 4u) | ((hi >> 6u) << 4u));
}
";

fn k_quant(qs: usize, high_bits: bool) -> String {
    let (nib, high) = if high_bits {
        (
            "\n            let bit = select(1u << (2u * j), 2u << (2u * j), hi);\n            let high = select(0u, 16u, (wbyte(sb + 16u + l) & bit) != 0u);",
            "nib + high",
        )
    } else {
        ("", "nib")
    };
    format!(
        "            let d = wf16(sb);
            let dmin = wf16(sb + 2u);
            let j = i / 64u;
            let within = i % 64u;
            let l = within % 32u;
            let hi = within >= 32u;
            let sm = scale_min_k4(2u * j + select(0u, 1u, hi), sb);
            let q = wbyte(sb + {qs}u + 32u * j + l);
            let nib = select(q & 0xfu, q >> 4u, hi);{nib}
            let d1 = d * f32(sm.x);
            let mm = dmin * f32(sm.y);
            return d1 * f32({high}) - mm;
"
    )
}

const WGSL_Q6K: &str = "            let d = wf16(sb + 208u);
            let hr = i / 128u;
            let r = i % 128u;
            let seg = r / 32u;
            let l = r % 32u;
            let ql = sb + 64u * hr;
            let qh = sb + 128u + 32u * hr;
            let sc = sb + 192u + 8u * hr;
            let isc = l / 16u;
            let lo_byte = wbyte(ql + l + select(0u, 32u, (seg & 1u) == 1u));
            let hi_byte = wbyte(qh + l);
            let nib = select(lo_byte & 0xfu, lo_byte >> 4u, seg >= 2u);
            let top = (hi_byte >> (2u * seg)) & 3u;
            let q = i32(nib | (top << 4u)) - 32;
            return d * wi8(sc + isc + 2u * seg) * f32(q);
";

const WGSL_Q8_0: &str = "            let b = sb + (i / 32u) * 34u;
            let d = wf16(b);
            return d * wi8(b + 2u + (i % 32u));
";

pub fn wgsl_decoder() -> String {
    let bodies = [k_quant(16, false), k_quant(48, true), WGSL_Q6K.to_string(), WGSL_Q8_0.to_string(), "            return bitcast<f32>(whalf(sb + 2u * i) << 16u);\n".to_string()];
    let mut s = String::from(WGSL_BYTES);
    s.push_str("\nfn block_bytes(code: u32) -> u32 {\n    switch (code) {\n");
    for (code, (_, bytes)) in BLOCK_BYTES.iter().enumerate() {
        s.push_str(&format!("        case {code}u: {{ return {bytes}u; }}\n"));
    }
    s.push_str(&format!("        default: {{ return {}u; }}\n    }}\n}}\n\n", 2 * SUPER_BLOCK));
    s.push_str(&format!("fn dequant_at(row_base: u32, code: u32, e: u32) -> f32 {{\n    let sb = row_base + (e / {SUPER_BLOCK}u) * block_bytes(code);\n    let i = e % {SUPER_BLOCK}u;\n    switch (code) {{\n"));
    for (code, body) in bodies.iter().enumerate() {
        s.push_str(&format!("        case {code}u: {{\n{body}        }}\n"));
    }
    s.push_str("        default: {\n            return wf16(sb + 2u * i);\n        }\n    }\n}\n");
    s
}
