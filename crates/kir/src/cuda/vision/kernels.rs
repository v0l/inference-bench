use super::prim::{self, fill};
use crate::cuda::attention::prim::{self as attn, Scores};
use crate::vision::VisionKernel;

fn slots(k: &VisionKernel) -> Vec<(&'static str, String)> {
    let scores = Scores {
        neg: "NEG",
        binds: "                bool ok = e < 2 ? (key >= lo0 && key < hi0) : (key >= lo1 && key < hi1);\n",
        live: "ok",
        score: "s[nb][e] * scale_log2",
    };
    vec![
        ("LOAD_STAGE", prim::gemm_load_stage()),
        ("MAINLOOP", prim::gemm_mainloop()),
        ("GM_CONSTS", prim::gemm_consts()),
        ("AT_CONSTS", prim::attention_consts(k.precise)),
        ("ZERO_ACC", prim::zero_tiles("    ", "acc", &["GM_MI", "GM_NJ", "4"])),
        ("ZERO_O", prim::zero_tiles("    ", "o", &["AT_NB", "4"])),
        ("ZERO_S", attn::zero_scores().replace("        float s[SB][4];\n", "")),
        ("QK", prim::qk_scores()),
        ("SOFTMAX", attn::online_softmax_with(&scores)),
        ("RESCALE", attn::rescale_acc("i", "AT_NB")),
        ("PACK_PROBS", format!("{}{}", attn::pack_rows("            ", "ph", "pack_f16_hi"), attn::pack_rows("            ", "pl", "pack_f16_lo"))),
        ("PV", prim::pv_accumulate()),
        ("INV_L", prim::inv_l()),
        ("TAP_ROTATE", prim::tap_rotate("half", "quarter", "ph", "pw")),
    ]
}

pub fn gemm_kernel(k: &VisionKernel) -> String {
    fill(
        r##"@GM_CONSTS@
__device__ __forceinline__ int gswz(int r, int c) {
    if (GM_CPR == 8) return r * 8 + (c ^ (r & 7));
    return r * 4 + (c ^ ((r >> 1) & 3));
}

extern "C" __global__ void __launch_bounds__(GM_THREADS) vis_mma(const bf16_t* a_hi, const bf16_t* a_lo, const bf16_t* w,
                                                                const bf16_t* bias, const float* resid, float* out,
                                                                bf16_t* out_hi, bf16_t* out_lo, int M, int N, int K, int act) {
    extern __shared__ __align__(16) unsigned char gm_smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    int wm = warp / GM_WN;
    int wn = warp % GM_WN;
    int m0 = blockIdx.y * GM_BM;
    int n0 = blockIdx.x * GM_BN;
    int kt_count = (K + GM_BK - 1) / GM_BK;
    float acc[GM_MI][GM_NJ][4];
@ZERO_ACC@@LOAD_STAGE@#pragma unroll
    for (int st = 0; st < GM_STAGES - 1; st++) {
        if (st < kt_count) load_stage(st, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < kt_count; kt++) {
        cp_async_wait<GM_STAGES - 2>();
        __syncthreads();
        int next = kt + GM_STAGES - 1;
        if (next < kt_count) load_stage(next % GM_STAGES, next);
        cp_async_commit();
        const unsigned char* base = gm_smem + (kt % GM_STAGES) * GM_STAGE_BYTES;
        const bf16_t* sa_hi = (const bf16_t*)base;
        const bf16_t* sa_lo = sa_hi + GM_A_TILE;
        const unsigned char* ws = base + GM_A_BYTES;
@MAINLOOP@        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int i = 0; i < GM_MI; i++)
#pragma unroll
        for (int j = 0; j < GM_NJ; j++) {
            int col = n0 + wn * GM_TN + j * 8 + 2 * t;
            if (col >= N) continue;
            float b0 = bias ? bf(bias[col]) : 0.0f;
            float b1 = bias ? bf(bias[col + 1]) : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int m = m0 + wm * GM_TM + i * 16 + g + h * 8;
                if (m >= M) continue;
                float v0 = activate(acc[i][j][h * 2] + b0, act);
                float v1 = activate(acc[i][j][h * 2 + 1] + b1, act);
                size_t at = (size_t)m * N + col;
                if (out_hi) {
                    store_act2(out_hi, out_lo, at, v0, v1);
                } else {
                    if (resid) {
                        v0 += resid[at];
                        v1 += resid[at + 1];
                    }
                    *(float2*)(out + at) = make_float2(v0, v1);
                }
            }
        }
}
"##,
        &slots(k),
    )
}

pub fn norm_kernels() -> String {
    format!(
            r#"
extern "C" __global__ void vis_split(const float* x, bf16_t* hi, bf16_t* lo, int n) {{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;
    float2 v = *(const float2*)(x + i);
    store_act2(hi, lo, i, v.x, v.y);
}}

extern "C" __global__ void vis_layernorm(const float* x, const bf16_t* w, const bf16_t* b, bf16_t* hi, bf16_t* lo, int width, float eps) {{
    __shared__ float red[2][32];
    const float* row = x + (size_t)blockIdx.x * width;
{moments}{warp}{share}    float mean = s / width;
    float var = s2 / width - mean * mean;
    float inv = rsqrtf(fmaxf(var, 0.0f) + eps);
    size_t base = (size_t)blockIdx.x * width;
{apply}}}

"#,
            moments = prim::row_moments("s", "row", "width", "blockDim.x"),
            warp = prim::warp_pair("s"),
            share = prim::lane_share("s", "red"),
            apply = prim::layernorm_apply("row", "width", "blockDim.x", "mean", "inv"),
    )
}

pub fn pos_kernel() -> String {
    format!(
            r#"extern "C" __global__ void vis_pos(const float* table, const int* taps, const float* weights, float* out, int hidden) {{
    int tok = blockIdx.x;
    const int* ti = taps + tok * 4;
    const float* tw = weights + tok * 4;
{gather}        out[(size_t)tok * hidden + d] = v;
    }}
}}

"#,
            gather = prim::gather4("v", "table", "ti", "tw", "hidden", "blockDim.x"),
    )
}

pub fn attention_kernels(k: &VisionKernel) -> String {
    fill(
        r##"@AT_CONSTS@
extern "C" __global__ void vis_rope_pack(const float* qkv, const int* hw, bf16_t* planes, int tokens, int heads, int head_dim, float theta) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int half = head_dim / 2;
    int quarter = head_dim / 4;
    size_t width = (size_t)3 * heads * head_dim;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    size_t dst = ((size_t)h * tokens + tok) * AT_HDP;
    int ph = hw[2 * tok], pw = hw[2 * tok + 1];
    for (int part = 0; part < 3; part++) {
        const float* src = qkv + tok * width + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        bf16_t* hi = planes + (2 * part) * plane + dst;
        bf16_t* lo = planes + (2 * part + 1) * plane + dst;
        for (int d = threadIdx.x; d < AT_HDP; d += blockDim.x) {
            float v = 0.0f;
            if (d < head_dim) {
                v = src[d];
                if (part < 2 && theta > 0.0f) {
@TAP_ROTATE@                }
            }
            unsigned b = f16_bits(v);
            hi[d] = (bf16_t)b;
            if (ATT_SPLIT) lo[d] = (bf16_t)f16_bits(v - f16_float(b));
        }
    }
}

extern "C" __global__ void __launch_bounds__(32 * AT_WARPS) vis_flash(const bf16_t* planes, const int* seg_lo, const int* seg_hi,
                                                                     bf16_t* out_hi, bf16_t* out_lo, int tokens, int heads,
                                                                     int head_dim, float scale_log2) {
    extern __shared__ __align__(16) unsigned char at_smem[];
    bf16_t* sm = (bf16_t*)at_smem;
    int head = blockIdx.y;
    int q0 = blockIdx.x * AT_BQ;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int t = lane & 3;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    const bf16_t* head_base = planes + (size_t)head * tokens * AT_HDP;
    for (int i = tid; i < 2 * AT_BQ * AT_CHUNKS; i += 32 * AT_WARPS) {
        int p = i / (AT_BQ * AT_CHUNKS);
        int rem = i % (AT_BQ * AT_CHUNKS);
        int r = rem / AT_CHUNKS;
        int c = rem % AT_CHUNKS;
        int q = q0 + r;
        const bf16_t* src = head_base + p * plane + (size_t)min(q, tokens - 1) * AT_HDP + c * 8;
        cp_async16_zfill(sm + (p * AT_BQ + r) * AT_STRIDE + c * 8, src, q < tokens);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned qh[AT_KB][4], ql[AT_KB][4];
#pragma unroll
    for (int kb = 0; kb < AT_KB; kb++) {
        int row = warp * 16 + (lane & 15);
        int col = kb * 16 + (lane >> 4) * 8;
        ldsm_x4(qh[kb], sm + row * AT_STRIDE + col);
        if (ATT_SPLIT) ldsm_x4(ql[kb], sm + (AT_BQ + row) * AT_STRIDE + col);
    }
    __syncthreads();
    int row0 = q0 + warp * 16 + g;
    int row1 = row0 + 8;
    int lo0 = row0 < tokens ? seg_lo[row0] : 0, hi0 = row0 < tokens ? seg_hi[row0] : 0;
    int lo1 = row1 < tokens ? seg_lo[row1] : 0, hi1 = row1 < tokens ? seg_hi[row1] : 0;
    int kstart = seg_lo[q0];
    int kend = seg_hi[min(q0 + AT_BQ, tokens) - 1];
    int tiles = (kend - kstart + AT_BK - 1) / AT_BK;
    auto load_kv = [&](int buf, int k0) {
        bf16_t* st = sm + buf * AT_STAGE;
        constexpr int KV_LOADS = ATT_SPLIT ? 4 : 2;
        for (int i = tid; i < KV_LOADS * AT_BK * AT_CHUNKS; i += 32 * AT_WARPS) {
            int slot = i / (AT_BK * AT_CHUNKS);
            int p = ATT_SPLIT ? slot : slot * 2;
            int rem = i % (AT_BK * AT_CHUNKS);
            int r = rem / AT_CHUNKS;
            int c = rem % AT_CHUNKS;
            int key = k0 + r;
            const bf16_t* src = head_base + (size_t)(2 + p) * plane + (size_t)min(key, tokens - 1) * AT_HDP + c * 8;
            cp_async16_zfill(st + slot * AT_TILE + r * AT_STRIDE + c * 8, src, key < kend);
        }
    };
    float o[AT_NB][4];
@ZERO_O@    const float NEG = -3.0e38f;
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    if (tiles > 0) load_kv(0, kstart);
    cp_async_commit();
    for (int kt = 0; kt < tiles; kt++) {
        int k0 = kstart + kt * AT_BK;
        if (kt + 1 < tiles) load_kv((kt + 1) & 1, k0 + AT_BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const bf16_t* kh = sm + (kt & 1) * AT_STAGE;
        const bf16_t* kl = kh + AT_TILE;
        const bf16_t* vh = ATT_SPLIT ? kl + AT_TILE : kh + AT_TILE;
        const bf16_t* vl = vh + AT_TILE;
        constexpr int SB = AT_BK / 8;
        float s[SB][4];
@ZERO_S@@QK@@SOFTMAX@@RESCALE@#pragma unroll
        for (int kb = 0; kb < SB / 2; kb++) {
            unsigned ph[4], pl[4];
@PACK_PROBS@@PV@        }
        __syncthreads();
    }
@INV_L@    size_t width = (size_t)heads * head_dim;
#pragma unroll
    for (int nb = 0; nb < AT_NB; nb++) {
        int d = nb * 8 + 2 * t;
        if (d >= head_dim) continue;
        size_t col = (size_t)head * head_dim + d;
        if (row0 < tokens) store_act2(out_hi, out_lo, row0 * width + col, o[nb][0] * i0, o[nb][1] * i0);
        if (row1 < tokens) store_act2(out_hi, out_lo, row1 * width + col, o[nb][2] * i1, o[nb][3] * i1);
    }
}
"##,
        &slots(k),
    )
}

pub fn pool_kernel() -> String {
    format!(
            r#"extern "C" __global__ void vis_pool(const float* x, const bf16_t* w, const bf16_t* b, const float* soft, bf16_t* hi, bf16_t* lo,
                                    int hidden, int side, int kernel, int out_side, float eps, float soft_eps) {{
    extern __shared__ float acc[];
    __shared__ float red[32];
    int o = blockIdx.x;
    int per_tile = out_side * out_side;
    int base = (o / per_tile) * side * side;
    int local = o % per_tile;
    int oy = local / out_side, ox = local % out_side;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();
    for (int dy = 0; dy < kernel; dy++) {{
        for (int dx = 0; dx < kernel; dx++) {{
            const float* row = x + (size_t)(base + (oy * kernel + dy) * side + ox * kernel + dx) * hidden;
{stats}            for (int i = threadIdx.x; i < hidden; i += blockDim.x) acc[i] += (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
        }}
    }}
    float scale = 1.0f / (kernel * kernel);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {{
        acc[i] *= scale;
        ss += acc[i] * acc[i];
    }}
    float rinv = rsqrtf(block_sum(ss, red) / hidden + soft_eps);
    __syncthreads();
    size_t at = (size_t)o * hidden;
    for (int i = threadIdx.x * 2; i < hidden; i += blockDim.x * 2) store_act2(hi, lo, at + i, acc[i] * rinv * soft[i], acc[i + 1] * rinv * soft[i + 1]);
}}
"#,
            stats = prim::pool_stats("s", "row", "hidden", "blockDim.x", "red", "mean", "inv"),
    )
}

pub fn joint_kernels() -> String {
    format!(
        r#"extern "C" __global__ void vis_qk_pack(const float* src, int row_stride, const float* n0, const float* n1, const float* n2,
                                       const float* cosv, const float* sinv, bf16_t* out, int tokens, int heads, int head_dim,
                                       int parts, float eps, int rope) {{
    __shared__ float red[32];
    int tok = blockIdx.x;
    int hidden = heads * head_dim;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    for (int part = 0; part < parts; part++) {{
        const float* s = src + (size_t)tok * row_stride + (size_t)part * hidden;
        const float* w = part == 0 ? n0 : (part == 1 ? n1 : n2);
        float ss = 0.0f;
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) ss += s[c] * s[c];
        float inv = rsqrtf(block_sum(ss, red) / hidden + eps);
        for (int i = threadIdx.x; i < heads * AT_HDP; i += blockDim.x) {{
            int h = i / AT_HDP, d = i % AT_HDP;
            float v = 0.0f;
            if (d < head_dim) {{
                int c = h * head_dim + d;
                v = w ? s[c] * inv * w[c] : s[c];
                if (rope && part < 2) {{
                    int c2 = c ^ 1;
                    float o = w ? s[c2] * inv * w[c2] : s[c2];
                    float cs = cosv[(size_t)tok * head_dim + d];
                    float sn = sinv[(size_t)tok * head_dim + d];
                    v = (d & 1) ? v * cs + o * sn : v * cs - o * sn;
                }}
            }}
            size_t at = (size_t)(2 * part) * plane + ((size_t)h * tokens + tok) * AT_HDP + d;
            unsigned b = f16_bits(v);
            out[at] = (bf16_t)b;
            if (ATT_SPLIT) out[at + plane] = (bf16_t)f16_bits(v - f16_float(b));
        }}
    }}
}}

extern "C" __global__ void vis_axis_rope(const float* qkv, const float* qw_text, const float* kw_text, const float* qw_image, const float* kw_image,
                                         const float* cosv, const float* sinv, bf16_t* planes, int tokens, int heads, int head_dim,
                                         int row_stride, int text_rows, float eps, int seq_rows) {{
    __shared__ float red[32];
    int tok = blockIdx.x;
    int h = blockIdx.y;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    size_t dst = ((size_t)h * tokens + tok) * AT_HDP;
    bool text = tok % seq_rows < text_rows;
    for (int part = 0; part < 3; part++) {{
        const float* src = qkv + (size_t)tok * row_stride + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        const float* w = part == 0 ? (text ? qw_text : qw_image) : (text ? kw_text : kw_image);
{norm}        bf16_t* hi = planes + (2 * part) * plane + dst;
        bf16_t* lo = planes + (2 * part + 1) * plane + dst;
        for (int d = threadIdx.x; d < AT_HDP; d += blockDim.x) {{
            float v = 0.0f;
            if (d < head_dim) {{
                v = src[d];
                if (part < 2) {{
{rotate}                }}
            }}
            unsigned b = f16_bits(v);
            hi[d] = (bf16_t)b;
            if (ATT_SPLIT) lo[d] = (bf16_t)f16_bits(v - f16_float(b));
        }}
    }}
}}

"#,
        norm = prim::head_rms_inv("ss", "src", "head_dim", "red", "inv", "eps"),
        rotate = prim::table_rotate("src", "inv", "w", "cosv", "sinv", "tok", "head_dim"),
    )
}

pub fn cross_attention_kernel(k: &VisionKernel) -> Result<String, String> {
    let all = attention_kernels(k);
    let start = all.find("extern \"C\" __global__ void __launch_bounds__(32 * AT_WARPS) vis_flash(").ok_or("no vis_flash entry")?;
    let mut text = all[start..].to_string();
    let swaps = [
        ("vis_flash(const bf16_t* planes, const int* seg_lo, const int* seg_hi,", "vis_flash_kv(const bf16_t* planes, const bf16_t* kv, int kv_tokens, int q_seq, int k_seq,"),
        ("    size_t plane = (size_t)heads * tokens * AT_HDP;\n", "    size_t plane = (size_t)heads * tokens * AT_HDP;\n    size_t kv_plane = (size_t)heads * kv_tokens * AT_HDP;\n"),
        ("int lo0 = row0 < tokens ? seg_lo[row0] : 0, hi0 = row0 < tokens ? seg_hi[row0] : 0;", "int lo0 = row0 < tokens ? row0 / q_seq * k_seq : 0, hi0 = row0 < tokens ? lo0 + k_seq : 0;"),
        ("int lo1 = row1 < tokens ? seg_lo[row1] : 0, hi1 = row1 < tokens ? seg_hi[row1] : 0;", "int lo1 = row1 < tokens ? row1 / q_seq * k_seq : 0, hi1 = row1 < tokens ? lo1 + k_seq : 0;"),
        ("int kstart = seg_lo[q0];", "int kstart = q0 / q_seq * k_seq;"),
        ("int kend = seg_hi[min(q0 + AT_BQ, tokens) - 1];", "int kend = (min(q0 + AT_BQ, tokens) - 1) / q_seq * k_seq + k_seq;"),
        ("const bf16_t* src = head_base + (size_t)(2 + p) * plane + (size_t)min(key, tokens - 1) * AT_HDP + c * 8;", "const bf16_t* src = kv + (size_t)head * kv_tokens * AT_HDP + (size_t)p * kv_plane + (size_t)min(key, kv_tokens - 1) * AT_HDP + c * 8;"),
    ];
    for (from, to) in swaps {
        if !text.contains(from) {
            return Err(format!("the flash kernel has no {from:?}"));
        }
        text = text.replacen(from, to, 1);
    }
    Ok(text)
}

pub fn cross_attention_rows_kernel() -> String {
    r#"
__device__ __forceinline__ float rows_block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < blockDim.x / 32; i++) t = fmaxf(t, red[i]);
    return t;
}

extern "C" __global__ void vis_flash_kv_rows(const float* q, const float* kv, float* out, int kv_rows, int heads, int head_dim,
                                           long long seq_stride, float scale) {
    extern __shared__ float kr_smem[];
    float* p = kr_smem;
    float* red = kr_smem + kv_rows;
    int head = blockIdx.x;
    int seq = blockIdx.y;
    size_t width = (size_t)heads * head_dim;
    const float* qr = q + (size_t)seq * width + (size_t)head * head_dim;
    const float* base = kv + (size_t)seq * (size_t)seq_stride;
    float mx = -3.0e38f;
    for (int k = threadIdx.x; k < kv_rows; k += blockDim.x) {
        const float* kr = base + (size_t)k * 2 * width + (size_t)head * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += qr[d] * kr[d];
        s *= scale;
        p[k] = s;
        mx = fmaxf(mx, s);
    }
    mx = rows_block_max(mx, red);
    float z = 0.0f;
    for (int k = threadIdx.x; k < kv_rows; k += blockDim.x) {
        float e = expf(p[k] - mx);
        p[k] = e;
        z += e;
    }
    z = block_sum(z, red);
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kv_rows; k++) acc += p[k] * base[(size_t)k * 2 * width + width + (size_t)head * head_dim + d];
        out[(size_t)seq * width + (size_t)head * head_dim + d] = acc / z;
    }
}
"#
    .to_string()
}

pub fn tiled_kernels() -> String {
    format!(
        r#"
extern "C" __global__ void vis_assemble(const float* patch, const float* cls, const float* pos, float* x, int n, int hidden) {{
    int tok = blockIdx.x;
    int tile = tok / (n + 1), local = tok % (n + 1);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {{
        float v = local < n ? patch[((size_t)tile * n + local) * hidden + d] : cls[d];
        x[(size_t)tok * hidden + d] = v + pos[(size_t)local * hidden + d];
    }}
}}

extern "C" __global__ void vis_rope_pairs(const float* qkv, bf16_t* planes, int tokens, int heads, int head_dim, float theta, int side, int per_tile) {{
    int tok = blockIdx.x;
    int h = blockIdx.y;
    int quarter = head_dim / 4;
    size_t width = (size_t)3 * heads * head_dim;
    size_t plane = (size_t)heads * tokens * AT_HDP;
    size_t dst = ((size_t)h * tokens + tok) * AT_HDP;
    int local = tok % per_tile;
    bool rotate = local < per_tile - 1;
    int px = local % side, py = local / side;
    for (int part = 0; part < 3; part++) {{
        const float* src = qkv + tok * width + (size_t)part * heads * head_dim + (size_t)h * head_dim;
        bf16_t* hi = planes + (2 * part) * plane + dst;
        bf16_t* lo = planes + (2 * part + 1) * plane + dst;
        for (int d = threadIdx.x; d < AT_HDP; d += blockDim.x) {{
            float v = 0.0f;
            if (d < head_dim) {{
                v = src[d];
                if (part < 2 && rotate) {{
                    int p = d >> 1;
                    int j = p % quarter;
                    float coord = (float)((p < quarter ? px : py) + 1);
                    float angle = coord * exp2f(-((float)j / (float)quarter) * log2f(theta));
                    float s, c;
                    sincosf(angle, &s, &c);
                    float other = src[d ^ 1];
                    v = (d & 1) ? other * s + v * c : v * c - other * s;
                }}
            }}
            unsigned b = f16_bits(v);
            hi[d] = (bf16_t)b;
            if (ATT_SPLIT) lo[d] = (bf16_t)f16_bits(v - f16_float(b));
        }}
    }}
}}

extern "C" __global__ void vis_layernorm_f32(const float* x, const bf16_t* w, const bf16_t* b, float* out, int hidden, float eps) {{
    __shared__ float red[32];
    const float* row = x + (size_t)blockIdx.x * hidden;
{stats}    for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[(size_t)blockIdx.x * hidden + i] = (row[i] - mean) * inv * bf(w[i]) + bf(b[i]);
}}

extern "C" __global__ void vis_shuffle_norm(const float* x, const bf16_t* w, const bf16_t* b, bf16_t* hi, bf16_t* lo,
                                            int hidden, int side, int per_tile, float eps) {{
    __shared__ float red[32];
    int o = blockIdx.x;
    int half = side / 2;
    int q = half * half;
    int tile = o / q, local = o % q;
    int oy = local / half, ox = local % half;
    for (int slot = 0; slot < 4; slot++) {{
        int tok = tile * per_tile + (oy * 2 + slot / 2) * side + ox * 2 + (slot & 1);
        const float* row = x + (size_t)tok * hidden;
{stats}        size_t at = ((size_t)o * 4 + slot) * hidden;
        for (int i = threadIdx.x * 2; i < hidden; i += blockDim.x * 2) {{
            store_act2(hi, lo, at + i, (row[i] - mean) * inv * bf(w[i]) + bf(b[i]), (row[i + 1] - mean) * inv * bf(w[i + 1]) + bf(b[i + 1]));
        }}
        __syncthreads();
    }}
}}
"#,
        stats = prim::pool_stats("s", "row", "hidden", "blockDim.x", "red", "mean", "inv"),
    )
}

