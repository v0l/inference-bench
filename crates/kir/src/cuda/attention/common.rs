use super::prim::{self, Reduce};
use crate::attention::AttentionShape;
use crate::rope;

pub fn body(s: &AttentionShape) -> String {
    let kv = prim::KvShape {
        kv_heads: s.kv_heads,
        page: crate::attention::PAGE,
        head_dim: s.head_dim,
        planes: s.kv_planes,
        layers: s.kv_layers.max(1),
        f16_layers: if s.kv_fp8 { s.kv_f16_layers.min(s.kv_layers) } else { 0 },
        fp8: s.kv_fp8,
    };
    let m = s.rope.mrope;
    [
        r##"__device__ __forceinline__ float cap_score(float s) {
    return SOFTCAP > 0.0f ? SOFTCAP * tanhf(s / SOFTCAP) : s;
}

__device__ __forceinline__ int window_lo(int len, int window) {
    return window > 0 ? max(0, len - window) : 0;
}

__device__ __forceinline__ unsigned f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ float f16_to_f32(unsigned bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ unsigned attn_bits(float v) {
    return f16_bits(v);
}

__device__ __forceinline__ float attn_float(unsigned bits) {
    return f16_to_f32(bits & 0xffffu);
}

__device__ __forceinline__ void attn_split(float v, unsigned& hi, unsigned& lo) {
    hi = attn_bits(v);
    lo = attn_bits(v - attn_float(hi));
}

"##.to_string(),
        prim::kv_addr(&kv),
        rope::mrope_axis_fn(),
        rope::position_fn(),
        r##"__device__ __forceinline__ int span_end(const int* mm_meta, const int* bidi, int mm_stride, int slot, int idx) {
    if (!BIDI) return idx;
    return idx < mm_meta[2 * slot] ? max(idx, bidi[(size_t)slot * mm_stride + idx]) : idx;
}

__device__ __forceinline__ void norm_rope_head(float* qkv, const float* q_norm, const float* k_norm,
                                               bf16_t* k_cache, bf16_t* v_cache, const int* block_table,
                                               int pos, int layer, int head, int row,
                                               const int* mm_meta, const int* mm_table, int mm_stride, int slot) {
    __shared__ float scratch[32];
    __shared__ float vec[HEAD_DIM];
    int d = threadIdx.x;
    bool is_q = head < HEADS;
    bool is_k = !is_q && head < HEADS + KV_HEADS;
    float* src = qkv + (size_t)row * QKV_ROW + head * HEAD_DIM;
    float v = d < HEAD_DIM ? src[d] : 0.0f;
    if (QK_NORM && (is_q || is_k)) {
        float inv = rsqrtf("##.to_string(),
        prim::block_reduce(Reduce::Sum, "v * v", "scratch"),
        r##" / HEAD_DIM + EPS);
        const float* w = is_q ? q_norm : k_norm;
        if (d < HEAD_DIM) v = w[d] * (v * inv);
    }
    if (d < HEAD_DIM) vec[d] = v;
    __syncthreads();
    if (!NOPE[layer] && (is_q || is_k) && d < ROT) {
"##.to_string(),
        rope::rotate_head(),
        r##"    }
    float row_scale = 1.0f;
    if (kv_q(layer) && !is_q) {
        float amax = "##.to_string(),
        prim::block_reduce(Reduce::Max, "fabsf(v)", "scratch"),
        r##";
        row_scale = amax > 0.0f ? amax / FP8_MAX : 1.0f;
    }
    if (d >= HEAD_DIM) return;
    if (is_q) {
        src[d] = v;
        return;
    }
    int kv_head = is_k ? head - HEADS : head - HEADS - KV_HEADS;
    bf16_t* dst = kv_row(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos);
    if (kv_q(layer)) {
        ((unsigned char*)dst)[d] = (unsigned char)(f32x2_to_e4m3x2(v / row_scale, 0.0f) & 0xffu);
        if (d == 0) *kv_scale_ptr(is_k ? k_cache : v_cache, block_table, layer, kv_head, pos) = row_scale;
        return;
    }
    unsigned hi, lo;
    attn_split(v, hi, lo);
    dst[d] = (bf16_t)hi;
    if (KV_PLANES == 2) dst[HEAD_DIM + d] = (bf16_t)lo;
}

"##.to_string(),
    ]
    .concat()
}
