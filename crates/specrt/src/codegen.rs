use crate::spec::{Activation, Quant, RopeScalingKind, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const PAGE: usize = 16;
pub const Q8_GROUP: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub wm: usize,
    pub wn: usize,
}

impl Tile {
    pub const SMALL: Tile = Tile { bm: 64, bn: 64, wm: 2, wn: 2 };
    pub const NARROW: Tile = Tile { bm: 64, bn: 32, wm: 2, wn: 2 };

    pub fn tiny() -> Tile {
        let wn = std::env::var("SPECRT_TINY_WN").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        Tile { bm: 16, bn: 16 * wn, wm: 1, wn }
    }

    pub fn threads(&self) -> u32 {
        (self.wm * self.wn * 32) as u32
    }

    pub fn big() -> Tile {
        match std::env::var("SPECRT_BIG_TILE").ok().and_then(|v| {
            let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 4).then(|| Tile { bm: p[0], bn: p[1], wm: p[2], wn: p[3] })
        }) {
            Some(t) => t,
            None => Tile { bm: 64, bn: 128, wm: 2, wn: 4 },
        }
    }
}

pub const KERNELS: &[&str] = &[
    "embed",
    "rmsnorm",
    "qkv_proj",
    "qk_norm_rope",
    "attention",
    "attention_split",
    "attention_reduce",
    "o_proj",
    "gate_up_proj",
    "swiglu",
    "down_proj",
    "lm_head",
    "argmax_advance",
    "qkv_gemm",
    "o_gemm",
    "gate_up_gemm",
    "down_gemm",
    "lm_head_gemm",
    "qkv_proj_norm",
    "gate_up_act",
    "lm_head_norm",
    "qkv_mma",
    "o_mma",
    "gate_up_mma",
    "down_mma",
    "lm_head_mma",
    "qkv_mma_big",
    "o_mma_big",
    "gate_up_mma_big",
    "down_mma_big",
    "lm_head_mma_big",
    "row_rms",
    "qkv_mma_tiny",
    "o_mma_tiny",
    "gate_up_mma_tiny",
    "down_mma_tiny",
    "lm_head_mma_tiny",
    "kv_norm_rope",
    "argmax_rows",
    "bump",
    "qkv_mma_narrow",
    "o_mma_narrow",
    "gate_up_mma_narrow",
    "down_mma_narrow",
    "lm_head_mma_narrow",
    "flash_attention",
    "flash_mma",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub source: String,
    pub kernels: Vec<&'static str>,
}

pub fn render(template: &str, vars: &BTreeMap<&str, String>) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            bail!("unterminated slot");
        };
        let key = &after[..end];
        match vars.get(key) {
            Some(v) => out.push_str(v),
            None => bail!("template slot {key} has no value"),
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

fn float_literal(v: f64) -> String {
    format!("{v:e}f")
}

impl Module {
    pub fn generate(spec: &Spec, max_ctx: usize) -> Result<Self> {
        let a = &spec.attention;
        let d = &spec.dims;
        let linear = match spec.quant {
            Quant::Bf16 => "linear_bf16",
            Quant::Q8 => "linear_q8",
        };
        let scaling = spec.rope.scaling.as_ref();
        let rope_kind = match scaling.map(|s| s.kind) {
            None => 0,
            Some(RopeScalingKind::Linear) => 1,
            Some(RopeScalingKind::Llama3) => 2,
        };
        let opt = |f: fn(&crate::spec::RopeScaling) -> Option<f64>| {
            float_literal(scaling.and_then(f).unwrap_or(1.0))
        };
        let big = Tile::big();
        let unsplit = std::env::var("SPECRT_UNSPLIT").unwrap_or_default();
        let split_default = spec.quant == Quant::Bf16;
        let split =
            |name: &str| (split_default && !unsplit.split(',').any(|n| n == name)).to_string();
        let act = match spec.mlp.activation {
            Activation::Swiglu => 0,
            Activation::Geglu => 1,
        };
        let vars: BTreeMap<&str, String> = [
            ("HIDDEN", d.hidden.to_string()),
            ("INTER", d.intermediate.to_string()),
            ("VOCAB", d.vocab.to_string()),
            ("HEADS", a.heads.to_string()),
            ("KV_HEADS", a.kv_heads.to_string()),
            ("HEAD_DIM", a.head_dim.to_string()),
            ("GROUP", (a.heads / a.kv_heads).to_string()),
            ("QK_NORM", a.qk_norm.to_string()),
            ("QKV_BIAS", a.qkv_bias.to_string()),
            ("EPS", float_literal(spec.norm.eps)),
            ("THETA", float_literal(spec.rope.theta)),
            ("ROPE_KIND", rope_kind.to_string()),
            ("ROPE_FACTOR", float_literal(scaling.map_or(1.0, |s| s.factor))),
            ("ROPE_LOW_FREQ", opt(|s| s.low_freq_factor)),
            ("ROPE_HIGH_FREQ", opt(|s| s.high_freq_factor)),
            ("ROPE_ORIG_POS", opt(|s| s.original_max_positions)),
            ("ACT_KIND", act.to_string()),
            ("MAX_CTX", max_ctx.to_string()),
            ("PAGE", PAGE.to_string()),
            ("Q8_GROUP", Q8_GROUP.to_string()),
            ("Q_DIM", spec.q_dim().to_string()),
            ("QKV_N", (spec.q_dim() + 2 * spec.kv_dim()).to_string()),
            ("GATE_UP_N", (2 * d.intermediate).to_string()),
            ("LINEAR", linear.to_string()),
            ("Q8", (spec.quant == Quant::Q8).to_string()),
            ("SPLIT_QKV", split("qkv")),
            ("SPLIT_O", split("o")),
            ("SPLIT_GATE_UP", split("gate_up")),
            ("SPLIT_DOWN", split("down")),
            ("SPLIT_LM_HEAD", split("lm_head")),
            ("GEMM_STAGES", std::env::var("SPECRT_STAGES").unwrap_or_else(|_| "2".into())),
            ("TINY_BN", Tile::tiny().bn.to_string()),
            ("TINY_WN", Tile::tiny().wn.to_string()),
            ("BIG_BM", big.bm.to_string()),
            ("BIG_BN", big.bn.to_string()),
            ("BIG_WM", big.wm.to_string()),
            ("BIG_WN", big.wn.to_string()),
        ]
        .into_iter()
        .collect();
        let source =
            format!("{}\n{}", render(OPS_TEMPLATE, &vars)?, render(MODEL_TEMPLATE, &vars)?);
        Ok(Module { source, kernels: KERNELS.to_vec() })
    }
}
