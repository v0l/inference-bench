use super::gemm::prim::{self, device};
use crate::dequant::{BlockDequant, Elem, Scale};

fn types() -> &'static str {
    "typedef unsigned short u16;\ntypedef unsigned char u8;\ntypedef unsigned int u32;\ntypedef unsigned long long u64;\n\n"
}

fn bf16_rn() -> String {
    device("u16", "bf16_rn", "float v", "    u32 u = __float_as_uint(v);\n    return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);\n")
}

fn scale_f32(scale: Scale) -> String {
    let body = match scale {
        Scale::E8m0 => "    return b == 0u ? __uint_as_float(0x00400000u) : b == 255u ? __uint_as_float(0x7fc00000u) : __uint_as_float(b << 23);\n",
        Scale::E4m3 => "    return e4m3_f32(b);\n",
    };
    device("float", "scale_f32", "u32 b", body)
}

fn store(elem: Elem) -> &'static str {
    match elem {
        Elem::E4m3 => "        dst[i] = bf16_rn(e4m3_f32(b) * sc);\n",
        Elem::E2m1 => "        dst[2 * i] = bf16_rn(e2m1_f32(b & 15u) * sc);\n        dst[2 * i + 1] = bf16_rn(e2m1_f32(b >> 4) * sc);\n",
    }
}

fn kernel(k: &BlockDequant) -> String {
    format!(
        "extern \"C\" __global__ void __launch_bounds__({threads}) {name}(const u8* w, const u8* s, float global, u64 groups, u16* out) {{
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float sc = scale_f32(s[g]) * global;
    const u8* src = w + g * {bytes};
    u16* dst = out + g * {group};
#pragma unroll
    for (int i = 0; i < {bytes}; i++) {{
        u32 b = src[i];
{store}    }}
}}
",
        threads = k.threads,
        name = k.name(),
        bytes = k.group_bytes(),
        group = k.group,
        store = store(k.elem),
    )
}

pub fn emit_block_dequant(k: &BlockDequant) -> Result<crate::KirSource, String> {
    k.validate()?;
    let mut out = String::from(types());
    out.push_str(&bf16_rn());
    out.push_str(&super::weight_quant::e4m3_f32());
    if k.elem == Elem::E2m1 {
        out.push_str(&prim::e2m1_to_f32("e2m1_f32"));
    }
    out.push_str(&scale_f32(k.scale));
    out.push_str(&kernel(k));
    Ok(crate::KirSource::emitted(out))
}
