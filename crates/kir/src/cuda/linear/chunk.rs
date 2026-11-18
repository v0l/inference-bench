use super::{chunk_lib, chunk_prep, chunk_scan};

pub fn body() -> String {
    let mut o = String::from(chunk_lib::LIBRARY);
    o.push('\n');
    o.push_str(&chunk_prep::body());
    o.push_str("\n\n");
    o.push_str(chunk_lib::PLANES);
    o.push('\n');
    o.push_str(&chunk_scan::body());
    o.push_str("\n\n");
    o.push_str(&gated_norm());
    o
}

fn gated_norm() -> String {
    r##"extern "C" __global__ void gdn_gated_norm(const float* o, const float* lin, const float* norm_w, bf16_t* hi, bf16_t* lo) {
    __shared__ float scratch[32];
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = 2 * threadIdx.x;
    const float* src = o + ((size_t)t * LA_VH + h) * LA_D;
    float a = src[d];
    float b = src[d + 1];
    float inv = rsqrtf(block_reduce<false>(a * a + b * b, scratch) / LA_D + EPS);
    const float* z = lin + (size_t)t * LA_IN_ROWS + LA_Z_OFF + h * LA_D;
    split_store2(hi, lo, (size_t)t * LA_VALUE_DIM + h * LA_D + d, norm_w[d] * (a * inv) * silu(z[d]),
                 norm_w[d + 1] * (b * inv) * silu(z[d + 1]));
}
"##
    .to_string()
}
