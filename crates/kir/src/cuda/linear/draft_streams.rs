use super::prim::{for_stride, kernel};

pub fn body() -> String {
    let mut o = String::new();
    o.push_str(r##"extern "C" __global__ void mtp_streams_in(const bf16_t* table, const int* tokens, const float* xsrc, const float* psrc,
                                          const int* hidx, const float* ne, const float* nh, bf16_t* ehi, bf16_t* elo,
                                          bf16_t* hhi, bf16_t* hlo, int width) {
    __shared__ float scratch[32];
    int r = blockIdx.x;
    const bf16_t* e = table + (size_t)tokens[r] * HIDDEN;
    int at = hidx[r];
    const float* h = at >= 0 ? xsrc + (size_t)at * width : psrc + (size_t)(-1 - at) * width;
    float se = 0.0f, sx = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float a = bf16_to_f32(e[i]);
        se += a * a;
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) sx += h[i] * h[i];
    float ie = rsqrtf(block_reduce<false>(se, scratch) / HIDDEN + EPS);
    float ix = rsqrtf(block_reduce<false>(sx, scratch) / width + EPS);
    size_t er = (size_t)r * HIDDEN, hr = (size_t)r * width;
    for (int i = 2 * threadIdx.x; i < HIDDEN; i += 2 * blockDim.x)
        split_store2(ehi, elo, er + i, ne[i] * (bf16_to_f32(e[i]) * ie), ne[i + 1] * (bf16_to_f32(e[i + 1]) * ie));
    for (int i = 2 * threadIdx.x; i < width; i += 2 * blockDim.x)
        split_store2(hhi, hlo, hr + i, nh[i] * (h[i] * ix), nh[i + 1] * (h[i + 1] * ix));
}

"##);
    let stride = |body: &str| for_stride("i", "blockIdx.x * blockDim.x + threadIdx.x", "width", "gridDim.x * blockDim.x", body);
    o += &kernel(
        "extern \"C\" __global__ void mtp_copy_w(const float* src, const int* sidx, float* dst, const int* didx, int width)",
        &format!("int s = blockIdx.y;\nconst float* a = src + (size_t)sidx[s] * width;\nfloat* b = dst + (size_t)didx[s] * width;\n{}", stride("b[i] = a[i];")),
    );
    o.push('\n');
    o += &kernel(
        "extern \"C\" __global__ void mtp_state_save(const float* state, float* saved, const int* vslots, int k1, int width)",
        &format!("int s = blockIdx.y;\nconst float* a = state + (size_t)vslots[s * k1] * width;\nfloat* b = saved + (size_t)s * width;\n{}", stride("b[i] = a[i];")),
    );
    o.push('\n');
    o += &kernel(
        "extern \"C\" __global__ void mtp_state_load(float* state, const float* saved, const int* vslots, int k1, int width)",
        &format!("int s = blockIdx.y;\nfloat* a = state + (size_t)vslots[s * k1] * width;\nconst float* b = saved + (size_t)s * width;\n{}", stride("a[i] = b[i];")),
    );
    o.push('\n');
    o += &kernel(
        "extern \"C\" __global__ void mtp_replay_rows(const float* src, float* dst, int k1, int j, int width)",
        &format!("int s = blockIdx.y;\nconst float* a = src + ((size_t)s * k1 + j) * width;\nfloat* b = dst + (size_t)s * width;\n{}", stride("b[i] = a[i];")),
    );
    o.push('\n');
    o += &kernel(
        "extern \"C\" __global__ void mtp_replay_seqs(const int* vtok, const int* vslots, const int* accepted, int k1, int j, int pad,\n                                           int n, int* tokens, int* slots)",
        "int s = blockIdx.x * blockDim.x + threadIdx.x;
if (s < n) {
    tokens[s] = vtok[s * k1 + j];
    slots[s] = j <= accepted[s] ? vslots[s * k1] : pad;
}
",
    );
    o.push('\n');
    o
}
