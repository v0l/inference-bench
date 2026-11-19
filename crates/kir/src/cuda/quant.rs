use super::attention::prim::{Reduce, block_reduce};
use crate::gemm::Fp8Scale;

pub fn quant_fp4(name: &str) -> String {
    format!(
        r#"extern "C" __global__ void {name}(const bf16_t* hi, const bf16_t* lo, const unsigned char* wscales, unsigned char* q,
                                          unsigned char* qs, int width) {{
    int groups = width / FP4_GROUP;
    int grp = blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= groups) return;
    size_t row = blockIdx.y;
    size_t at = row * width + (size_t)grp * FP4_GROUP;
    float igs = __ldg((const float*)wscales + 1);
    float v[FP4_GROUP];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i++) {{
        v[i] = bf16_to_f32(hi[at + i]) + bf16_to_f32(lo[at + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }}
    unsigned sbits = f32_to_e4m3_bits(fminf(igs * (amax * (1.0f / 6.0f)), 448.0f));
    float sx = e4m3_to_f32(sbits);
    float mul = sx > 0.0f ? igs / sx : 0.0f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int i = 0; i < FP4_GROUP; i += 2) {{
        unsigned b = f32x2_to_e2m1x2(v[i] * mul, v[i + 1] * mul);
        if (i < 8) w0 |= b << (4 * i); else w1 |= b << (4 * (i - 8));
    }}
    *(uint2*)(q + at / 2) = make_uint2(w0, w1);
    qs[row * groups + grp] = (unsigned char)sbits;
}}
"#
    )
}

pub fn quant_fp8(name: &str, scale: Fp8Scale) -> String {
    match scale {
        Fp8Scale::Block => format!(
            r#"extern "C" __global__ void {name}(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {{
    __shared__ float scratch[32];
    int row = blockIdx.y;
    int block = blockIdx.x;
    int c = block * FP8_BLOCK + threadIdx.x * 2;
    size_t at = (size_t)row * width + c;
    float a = bf16_to_f32(hi[at]) + bf16_to_f32(lo[at]);
    float b = bf16_to_f32(hi[at + 1]) + bf16_to_f32(lo[at + 1]);
    float amax = {reduce};
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    *(unsigned short*)(q + at) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    if (threadIdx.x == 0) scale[(size_t)row * (width / FP8_BLOCK) + block] = s;
}}
"#,
            reduce = block_reduce(Reduce::Max, "fmaxf(fabsf(a), fabsf(b))", "scratch")
        ),
        Fp8Scale::Channel => format!(
            r#"extern "C" __global__ void {name}(const bf16_t* hi, const bf16_t* lo, unsigned char* q, float* scale, int width) {{
    __shared__ float scratch[32];
    size_t row = (size_t)blockIdx.x * width;
    float amax = 0.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {{
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
    }}
    amax = {reduce};
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    for (int c = threadIdx.x * 2; c < width; c += blockDim.x * 2) {{
        float a = bf16_to_f32(hi[row + c]) + bf16_to_f32(lo[row + c]);
        float b = bf16_to_f32(hi[row + c + 1]) + bf16_to_f32(lo[row + c + 1]);
        *(unsigned short*)(q + row + c) = (unsigned short)f32x2_to_e4m3x2(a / s, b / s);
    }}
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
}}
"#,
            reduce = block_reduce(Reduce::Max, "amax", "scratch")
        ),
    }
}

pub fn quant_fp8_f32_groups(name: &str, threads: u32) -> String {
    format!(
        r##"extern "C" __global__ void __launch_bounds__({threads}) {name}(const float* x, unsigned char* q, float* scale, int width) {{
    size_t row = blockIdx.x;
    int blk = blockIdx.y;
    size_t at = row * width + (size_t)blk * FP8_BLOCK + threadIdx.x;
    float v = x[at];
    float amax = {reduce};
    float s = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    q[at] = (unsigned char)(f32x2_to_e4m3x2(v / s, 0.0f) & 0xffu);
    if ((threadIdx.x & 31) == 0) scale[row * (width / 32) + blk * 4 + (threadIdx.x >> 5)] = s;
}}

"##,
        reduce = super::attention::prim::warp_reduce(Reduce::Max, "fabsf(v)")
    )
}

pub fn dequant_bf16(name: &str) -> String {
    format!(
        r#"extern "C" __global__ void {name}(const void* w, const unsigned* table, bf16_t* out, int rows, int cols) {{
    int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    int per_row = cols / 8;
    if (chunk >= rows * per_row) return;
    int row = chunk / per_row;
    int k = (chunk % per_row) * 8;
    float v[8];
    gg_load8(w, table, row, k, v);
    *(uint4*)(out + (size_t)row * cols + k) = {pack};
}}
"#,
        pack = super::gemm::prim::pack_bf16x8(53)
    )
}
