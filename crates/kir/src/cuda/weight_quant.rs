use super::gemm::prim::device;
use crate::weight_quant::{AMAX, GROUP, QUANTIZE, SCALE_HEADER, WeightQuant};

fn types() -> &'static str {
    "typedef unsigned short u16;\ntypedef unsigned char u8;\ntypedef unsigned int u32;\ntypedef unsigned long long u64;\n\n"
}

fn bf16_f32() -> String {
    device("float", "bf16_f32", "u16 v", "    return __uint_as_float(((u32)v) << 16);\n")
}

fn f32_e4m3(max: u32) -> String {
    device(
        "u8",
        "f32_e4m3",
        "float v",
        &format!(
            "    float a = fabsf(v);
    u32 code;
    if (a < 0.015625f) {{
        code = (u32)rintf(a * 512.0f);
    }} else {{
        u32 bits = min(__float_as_uint(a), {bits:#x}u);
        bits = bits + 0x7ffffu + ((bits >> 20) & 1u);
        code = (((u32)((int)(bits >> 23) - 120)) << 3) | ((bits >> 20) & 7u);
    }}
    code = min(code, 0x7eu);
    return (u8)(code | ((__float_as_uint(v) >> 31) ? 0x80u : 0u));
",
            bits = (max as f32 + 32.0).to_bits()
        ),
    )
}

fn e4m3_f32() -> String {
    device(
        "float",
        "e4m3_f32",
        "u8 b",
        "    u32 mag = b & 0x7f;
    int exp = (int)(mag >> 3);
    float man = (float)(mag & 7);
    float v = exp == 0 ? man * ldexpf(1.0f, -9) : (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
    return (b & 0x80) ? -v : v;
",
    )
}

fn f32_e2m1() -> String {
    let steps = [(0.25, "<="), (0.75, "<"), (1.25, "<="), (1.75, "<"), (2.5, "<="), (3.5, "<"), (5.0, "<=")];
    let mut body = String::from("    float a = fabsf(v);\n    u32 index;\n");
    for (i, (edge, op)) in steps.iter().enumerate() {
        let kw = if i == 0 { "if" } else { "else if" };
        body += &format!("    {kw} (a {op} {edge:?}f) index = {i};\n");
    }
    body += &format!("    else index = {};\n", steps.len());
    body += "    return index | (((__float_as_uint(v) >> 31) && index != 0) ? 8u : 0u);\n";
    device("u32", "f32_e2m1", "float v", &body)
}

fn global_scale(k: &WeightQuant) -> String {
    device("float", "global_scale", "const u32* amax_bits", &format!("    float amax = __uint_as_float(*amax_bits);\n    return amax == 0.0f ? 1.0f : {}.0f / amax;\n", k.fp4_max * k.e4m3_max))
}

fn amax_kernel(k: &WeightQuant) -> String {
    format!(
        "extern \"C\" __global__ void {AMAX}(const u16* w, u64 n, u32* amax_bits) {{
    u32 local = 0;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {{
        local = max(local, ((u32)w[i] & 0x7fffu) << 16);
    }}
    for (int o = {half}; o > 0; o >>= 1) local = max(local, __shfl_xor_sync(0xffffffffu, local, o));
    if ((threadIdx.x & {lane}) == 0) atomicMax(amax_bits, local);
}}

",
        half = k.target.warp_size / 2,
        lane = k.target.warp_size - 1,
    )
}

fn quantize_kernel(k: &WeightQuant) -> String {
    format!(
        "extern \"C\" __global__ void {QUANTIZE}(const u16* w, u64 groups, const u32* amax_bits, u8* packed, u8* scales) {{
    float global = global_scale(amax_bits);
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g == 0) {{
        float* header = (float*)scales;
        header[0] = 1.0f / global;
    }}
    if (g >= groups) return;
    const u16* src = w + g * {GROUP};
    float v[{GROUP}];
    float amax = 0.0f;
    for (int i = 0; i < {GROUP}; i++) {{
        v[i] = bf16_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }}
    u8 code = f32_e4m3(fminf(amax / {fp4}.0f * global, {e4m3}.0f));
    scales[{SCALE_HEADER} + g] = code;
    float s = e4m3_f32(code) / global;
    u64 out = 0;
    for (int i = 0; i < {GROUP}; i++) {{
        u32 q = s > 0.0f ? f32_e2m1(v[i] / s) : 0u;
        out |= (u64)q << (4 * i);
    }}
    ((u64*)packed)[g] = out;
}}
",
        fp4 = k.fp4_max,
        e4m3 = k.e4m3_max,
    )
}

pub fn emit_weight_quant(k: &WeightQuant) -> Result<crate::KirSource, String> {
    k.validate()?;
    let mut out = String::from(types());
    for part in [bf16_f32(), f32_e4m3(k.e4m3_max), e4m3_f32(), f32_e2m1(), global_scale(k), amax_kernel(k), quantize_kernel(k)] {
        out.push_str(&part);
    }
    Ok(crate::KirSource::emitted(out))
}
