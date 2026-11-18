use super::prim::{self, Reduce};

pub fn body() -> String {
    [
        r##"constexpr int FA_BQ = 16;
constexpr int FA_BK = 32;
constexpr int FA_THREADS = 128;
constexpr int FA_STRIDE = HEAD_DIM + 1;
constexpr int FA_OUT = HEAD_DIM / 8;

extern "C" __global__ void flash_attention(const float* qkv, const bf16_t* k_cache, const bf16_t* v_cache,
                                           const int* block_table, const int* pos_ptr, int layer, float* out, int tokens, int window) {
    extern __shared__ __align__(16) float fa_smem[];
    float (*qs)[FA_STRIDE] = (float (*)[FA_STRIDE])fa_smem;
    float (*ks)[FA_STRIDE] = (float (*)[FA_STRIDE])(fa_smem + FA_BQ * FA_STRIDE);
    float (*vs)[HEAD_DIM] = (float (*)[HEAD_DIM])(fa_smem + (FA_BQ + FA_BK) * FA_STRIDE);
    float (*ps)[FA_BK + 1] = (float (*)[FA_BK + 1])(fa_smem + (FA_BQ + FA_BK) * FA_STRIDE + FA_BK * HEAD_DIM);
    int head = blockIdx.y;
    int kv_head = head / GROUP;
    int q0 = blockIdx.x * FA_BQ;
    int base = pos_ptr[0];
    int tid = threadIdx.x;
    int qr = tid >> 3;
    int sub = tid & 7;
    int qi = q0 + qr;
    int pos = base + qi;
    float scale = QUERY_SCALE;
    for (int i = tid; i < FA_BQ * HEAD_DIM; i += FA_THREADS) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q = q0 + r;
        qs[r][d] = q < tokens ? qkv[(size_t)q * QKV_ROW + head * HEAD_DIM + d] : 0.0f;
    }
    float o[FA_OUT];
#pragma unroll
    for (int i = 0; i < FA_OUT; i++) o[i] = 0.0f;
    float m = NEG_INF;
    float l = 0.0f;
    int last = min(q0 + FA_BQ, tokens) - 1;
    int key_end = base + last + 1;
    for (int k0 = 0; k0 < key_end; k0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * HEAD_DIM; i += FA_THREADS) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            int t = k0 + r;
            float kv = 0.0f, vv = 0.0f;
            if (t < key_end) {
                kv = kv_at(kv_row(k_cache, block_table, layer, kv_head, t), d, kv_scale(k_cache, block_table, layer, kv_head, t), kv_q(layer));
                vv = kv_at(kv_row(v_cache, block_table, layer, kv_head, t), d, kv_scale(v_cache, block_table, layer, kv_head, t), kv_q(layer));
            }
            ks[r][d] = kv;
            vs[r][d] = vv;
        }
        __syncthreads();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++) {
            float qv = qs[qr][d];
#pragma unroll
            for (int j = 0; j < 4; j++) s[j] += qv * ks[sub * 4 + j][d];
        }
        float tile_max = NEG_INF;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = k0 + sub * 4 + j;
            s[j] = (t <= pos && t < key_end && (window <= 0 || t > pos - window)) ? cap_score(s[j] * scale) : NEG_INF;
            tile_max = fmaxf(tile_max, s[j]);
        }
"##.to_string(),
        prim::xor_reduce(Reduce::Max, "tile_max", &[1, 2, 4]),
        r##"        float m_new = fmaxf(m, tile_max);
        float alpha = m == NEG_INF ? 0.0f : expf(m - m_new);
        float row_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float p = s[j] == NEG_INF ? 0.0f : expf(s[j] - m_new);
            ps[qr][sub * 4 + j] = p;
            row_sum += p;
        }
"##.to_string(),
        prim::xor_reduce(Reduce::Sum, "row_sum", &[1, 2, 4]),
        r##"        l = l * alpha + row_sum;
        m = m_new;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) o[i] *= alpha;
        __syncwarp();
#pragma unroll 8
        for (int j = 0; j < FA_BK; j++) {
            float p = ps[qr][j];
#pragma unroll
            for (int i = 0; i < FA_OUT; i++) o[i] += p * vs[j][sub + 8 * i];
        }
    }
    if (qi < tokens) {
        float inv = 1.0f / l;
        float* dst = out + (size_t)qi * Q_DIM + head * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < FA_OUT; i++) dst[sub + 8 * i] = o[i] * inv * out_gate(qkv, qi, head * HEAD_DIM + sub + 8 * i);
    }
}
"##.to_string(),
    ]
    .concat()
}
