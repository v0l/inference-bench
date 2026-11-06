use crate::spec::{Activation, Quant, RopeScalingKind, Spec};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

pub const OPS_TEMPLATE: &str = include_str!("../templates/ops.cu");
pub const MODEL_TEMPLATE: &str = include_str!("../templates/model.cu");
pub const PAGE: usize = 16;
pub const Q8_GROUP: usize = 32;

pub const KERNELS: &[&str] = &[
    "embed",
    "rmsnorm",
    "qkv_proj",
    "qk_norm_rope",
    "attention",
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
    "flash_attention",
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
        ]
        .into_iter()
        .collect();
        let source =
            format!("{}\n{}", render(OPS_TEMPLATE, &vars)?, render(MODEL_TEMPLATE, &vars)?);
        Ok(Module { source, kernels: KERNELS.to_vec() })
    }
}
