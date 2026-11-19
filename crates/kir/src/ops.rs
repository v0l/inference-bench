#[derive(Clone, Debug, PartialEq)]
pub struct OpsShape {
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qkv_row: usize,
    pub q_dim: usize,
    pub output_gate: bool,
    pub qkv_bias: bool,
    pub eps: f64,
    pub embed_scale: f64,
    pub gated_gelu: bool,
    pub q8_group: usize,
    pub fp8_block: usize,
    pub weight_decoder: String,
    pub head_decoder: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeShape {
    pub experts: usize,
    pub local: usize,
    pub top_k: usize,
    pub pairs: usize,
    pub inter: usize,
    pub shared: usize,
    pub norm_topk: bool,
    pub routed_scale: f64,
    pub shared_gate: bool,
    pub sigmoid: bool,
}

pub(crate) fn f32_literal(v: f64) -> String {
    format!("{v:e}f")
}

#[derive(Clone, Debug, PartialEq)]
pub struct WgslShape {
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qk_norm: bool,
    pub eps: f64,
    pub theta: f64,
    pub rope_kind: u32,
    pub rope_factor: f64,
    pub rope_low: f64,
    pub rope_high: f64,
    pub rope_original: f64,
    pub rope_mscale: f64,
    pub gated_gelu: bool,
    pub max_ctx: usize,
    pub q_dim: usize,
    pub qkv_row: usize,
    pub rotary_dim: usize,
}

pub(crate) fn wgsl_float(v: f64) -> String {
    format!("{v:e}")
}
