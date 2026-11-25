use super::{chunk_lib, chunk_prep, chunk_scan};

pub fn body(gate: &str) -> String {
    let mut o = chunk_lib::library();
    o.push('\n');
    o.push_str(&chunk_prep::body());
    o.push_str("\n\n");
    o.push_str(&chunk_lib::planes());
    o.push('\n');
    o.push_str(&chunk_scan::body());
    o.push_str("\n\n");
    o.push_str(&gated_norm(gate));
    o
}

fn gated_norm(gate: &str) -> String {
    format!(
        r##"extern "C" __global__ void gdn_gated_norm(const float* o, const float* lin, const float* norm_w, bf16_t* hi, bf16_t* lo) {{
    __shared__ float scratch[32];
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = 2 * threadIdx.x;
    bool live = d < LA_D;
    const float* src = o + ((size_t)t * LA_VH + h) * LA_D;
    float a = live ? src[d] : 0.0f;
    float b = live ? src[d + 1] : 0.0f;
    float inv = rsqrtf(block_reduce<false>(a * a + b * b, scratch) / LA_D + EPS);
    if (!live) return;
    const float* z = lin + (size_t)t * LA_IN_ROWS + LA_Z_OFF + h * LA_D;
    split_store2(hi, lo, (size_t)t * LA_VALUE_DIM + h * LA_D + d, norm_w[d] * (a * inv) * {gate}(z[d]),
                 norm_w[d + 1] * (b * inv) * {gate}(z[d + 1]));
}}
"##
    )
}
