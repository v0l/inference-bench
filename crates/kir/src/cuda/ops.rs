use crate::ops::{MoeShape, OpsShape, f32_literal};

const OPS: &str = include_str!("ops/ops.cu");
const LIN: &str = include_str!("ops/lin.cu");
const MOE: &str = include_str!("ops/moe.cu");
const MOE_MMA: &str = include_str!("ops/moe_mma.cu");

pub fn emit_ops(s: &OpsShape) -> String {
    let group = s.heads / s.kv_heads;
    let act = if s.gated_gelu { 1 } else { 0 };
    format!(
        "typedef unsigned short bf16_t;
typedef signed char i8_t;

constexpr int HIDDEN = {hidden};
constexpr int INTER = {inter};
constexpr int VOCAB = {vocab};
constexpr int HEADS = {heads};
constexpr int KV_HEADS = {kv_heads};
constexpr int HEAD_DIM = {head_dim};
constexpr int GROUP = {group};
constexpr int Q_DIM = HEADS * HEAD_DIM;
constexpr int KV_DIM = KV_HEADS * HEAD_DIM;
constexpr int QKV_ROW = {qkv_row};
constexpr int GATE_OFF = Q_DIM + 2 * KV_DIM;
constexpr bool OUT_GATE = {out_gate};
constexpr float EPS = {eps};
constexpr float EMBED_SCALE = {embed_scale};
constexpr int ACT_KIND = {act};
constexpr int Q8_GROUP = {q8_group};
constexpr int FP8_BLOCK = {fp8_block};
{OPS}",
        hidden = s.hidden,
        inter = s.inter,
        vocab = s.vocab,
        heads = s.heads,
        kv_heads = s.kv_heads,
        head_dim = s.head_dim,
        qkv_row = s.qkv_row,
        out_gate = s.output_gate,
        eps = f32_literal(s.eps),
        embed_scale = f32_literal(s.embed_scale),
        q8_group = s.q8_group,
        fp8_block = s.fp8_block,
    )
}

pub fn emit_model(s: &OpsShape) -> String {
    let (wf, hwf) = (&s.weight_decoder, &s.head_decoder);
    let (n, h, i, v, q) = (s.qkv_row, s.hidden, s.inter, s.vocab, s.q_dim);
    let bias = s.qkv_bias;
    let gu = 2 * s.inter;
    let mut o = String::new();
    let mut gemv = |name: &str, dec: &str, rows: usize, cols: usize, bias: bool, accumulate: bool| {
        o.push_str(&format!("extern \"C\" __global__ void {name}(const void* w, const float* s, const float* b, const float* x, float* out) {{\n    linear_w<{dec}, {rows}, {cols}, {bias}, {accumulate}>(w, s, b, x, out);\n}}\n\n"));
    };
    gemv("qkv_proj", wf, n, h, bias, false);
    gemv("o_proj", wf, h, q, false, true);
    gemv("gate_up_proj", wf, gu, h, false, false);
    gemv("down_proj", wf, h, i, false, true);
    gemv("lm_head", hwf, v, h, false, false);
    let mut gemm = |name: &str, dec: &str, rows: usize, cols: usize, bias: bool, accumulate: bool| {
        o.push_str(&format!("extern \"C\" __global__ void {name}(const void* w, const float* s, const float* b, const float* x, float* out, int tokens) {{\n    gemm<{rows}, {cols}, {bias}, {accumulate}, {dec}>(w, s, b, x, out, tokens);\n}}\n\n"));
    };
    gemm("qkv_gemm", wf, n, h, bias, false);
    gemm("o_gemm", wf, h, q, false, true);
    gemm("gate_up_gemm", wf, gu, h, false, false);
    gemm("down_gemm", wf, h, i, false, true);
    gemm("lm_head_gemm", hwf, v, h, false, false);
    o.push_str(&format!("extern \"C\" __global__ void qkv_proj_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {{\n    linear_norm<{wf}, {n}, {bias}>(w, s, b, x, nw, out);\n}}\n\n"));
    o.push_str(&format!("extern \"C\" __global__ void gate_up_act(const void* w, const float* s, const float* x, const float* nw, float* out) {{\n    gate_up_act_norm<{wf}>(w, s, x, nw, out);\n}}\n\n"));
    o.push_str(&format!("extern \"C\" __global__ void lm_head_norm(const void* w, const float* s, const float* b, const float* x, const float* nw, float* out) {{\n    linear_norm<{hwf}, {v}, false>(w, s, b, x, nw, out);\n}}\n"));
    o
}

pub fn emit_lin(s: &OpsShape) -> String {
    LIN.replace("LIN_WF", &s.weight_decoder)
}

pub fn emit_moe(m: &MoeShape) -> String {
    format!(
        "constexpr int MOE_E = {};
constexpr int MOE_K = {};
constexpr int MOE_PAIRS = {};
constexpr int MOE_INTER = {};
constexpr bool MOE_SHARED = {};
constexpr bool MOE_NORM_TOPK = {};
constexpr bool MOE_SHARED_GATE = {};
constexpr bool MOE_SIGMOID = {};
{MOE}",
        m.experts, m.top_k, m.pairs, m.inter, m.shared, m.norm_topk, m.shared_gate, m.sigmoid
    )
}

pub fn emit_moe_mma() -> &'static str {
    MOE_MMA
}
