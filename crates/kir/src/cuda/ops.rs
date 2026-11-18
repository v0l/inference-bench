use crate::ops::{OpsShape, f32_literal};

const PRELUDE: &str = include_str!("prelude.cu");

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
{PRELUDE}",
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
