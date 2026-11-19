use anyhow::{Context, Result, bail, ensure};
use kir;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Flag {
    Fixed(bool),
    Config { config: String, #[serde(default)] default: bool },
}

impl Flag {
    pub fn read(&self, config: &serde_json::Value) -> bool {
        match self {
            Flag::Fixed(b) => *b,
            Flag::Config { config: key, default } => config.get(key).and_then(|v| v.as_bool()).unwrap_or(*default),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamSource {
    Number(f64),
    Config { config: String, #[serde(default)] default: Option<f64> },
    Expr { expr: String },
}

pub fn read_params(params: &std::collections::BTreeMap<String, ParamSource>, config: &serde_json::Value) -> Result<std::collections::BTreeMap<String, f64>> {
    let mut out = std::collections::BTreeMap::new();
    for _ in 0..=params.len() {
        for (n, p) in params {
            if out.contains_key(n) {
                continue;
            }
            let v = match p {
                ParamSource::Expr { expr } => {
                    let e = kir::expr::Expr::parse(expr).map_err(anyhow::Error::msg).with_context(|| format!("attention param {n}"))?;
                    let env = |k: &str| out.get(k).copied().or_else(|| config.get(k).and_then(|v| v.as_f64().or_else(|| v.as_bool().map(f64::from))));
                    match e.eval(&env) {
                        Some(v) => v,
                        None => continue,
                    }
                }
                other => other.read(n, config)?,
            };
            out.insert(n.clone(), v);
        }
    }
    if let Some((n, p)) = params.iter().find(|(n, _)| !out.contains_key(*n)) {
        bail!("attention param {n} = {p:?} reads a config.json key or param that is missing");
    }
    Ok(out)
}

impl ParamSource {
    pub fn read(&self, name: &str, config: &serde_json::Value) -> Result<f64> {
        match self {
            ParamSource::Number(v) => Ok(*v),
            ParamSource::Expr { .. } => read_params(&[(name.to_string(), self.clone())].into_iter().collect(), config).map(|m| m[name]),
            ParamSource::Config { config: key, default } => key
                .split('.')
                .try_fold(config, |v, k| v.get(k))
                .and_then(|v| v.as_f64().or_else(|| v.as_bool().map(f64::from)))
                .or(*default)
                .with_context(|| format!("attention param {name} reads config.json key {key}, which is missing")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    #[default]
    Dense,
    Hybrid,
    FromConfig,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Match {
    pub model_types: Vec<String>,
    pub architectures: Vec<String>,
    pub gguf: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyAttention {
    pub node: Vec<crate::graph::Node>,
    pub params: std::collections::BTreeMap<String, ParamSource>,
    pub projection: Vec<crate::spec::Projection>,
    pub o_cols: Option<String>,
    pub scratch: std::collections::BTreeMap<String, String>,
    pub int_scratch: std::collections::BTreeMap<String, String>,
    pub qkv_bias: Flag,
    pub o_bias: Flag,
    pub qk_norm: Flag,
    pub output_gate: Flag,
    pub qk_norm_full: bool,
    pub fused_qkv: bool,
    pub o_tensor: Option<String>,
}

impl Default for FamilyAttention {
    fn default() -> Self {
        FamilyAttention {
            node: Vec::new(),
            params: Default::default(),
            projection: Vec::new(),
            o_cols: None,
            scratch: Default::default(),
            int_scratch: Default::default(),
            qkv_bias: Flag::Config { config: "attention_bias".into(), default: false },
            o_bias: Flag::Fixed(false),
            qk_norm: Flag::Fixed(false),
            output_gate: Flag::Config { config: "attn_output_gate".into(), default: false },
            qk_norm_full: false,
            fused_qkv: false,
            o_tensor: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyNorm {
    pub unit_offset: Option<bool>,
    pub sandwich: bool,
    pub pre: bool,
}

impl Default for FamilyNorm {
    fn default() -> Self {
        FamilyNorm { unit_offset: None, sandwich: false, pre: true }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyMultipliers {
    pub embedding: Option<String>,
    pub attention: Option<String>,
    pub residual: Option<String>,
    pub logits_divisor: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyMlp {
    pub fused_gate_up: bool,
    pub gated: Option<FamilyGated>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyGated {
    pub expr: String,
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, ParamSource>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyMoe {
    pub shared_module: Option<String>,
    pub shared_gate: Option<bool>,
    pub router: Option<crate::spec::RouterAct>,
    pub norm_topk: Option<bool>,
    pub score_bias: bool,
    pub expert_block: Option<crate::dequant::BlockFormat>,
    pub act_quant: Option<crate::spec::ActQuant>,
    pub gate_up_interleaved: bool,
    pub router_bias: bool,
    pub expert_bias: bool,
    pub router_module: Option<String>,
    pub fused_experts: bool,
    pub hash_table: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyRuntime {
    pub split_activations: bool,
}

impl Default for FamilyRuntime {
    fn default() -> Self {
        FamilyRuntime { split_activations: true }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyEmbedding {
    pub scale: Option<String>,
    pub tie: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VisionTokenKeys {
    pub image: String,
    pub video: String,
    pub start: String,
    pub end: String,
}

impl Default for VisionTokenKeys {
    fn default() -> Self {
        VisionTokenKeys { image: "image_token_id".into(), video: "video_token_id".into(), start: "vision_start_token_id".into(), end: "vision_end_token_id".into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[derive(Default)]
pub struct FamilyVision {
    pub pooled: bool,
    pub bidirectional: bool,
    pub soft_tokens_key: String,
    pub placeholder: Option<String>,
    pub wrap: String,
    pub token_keys: VisionTokenKeys,
}


#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyBlocks {
    pub kind: BlockKind,
    pub layer_types: std::collections::BTreeMap<String, crate::spec::Block>,
    pub other_layer_type: Option<crate::spec::Block>,
    pub from_layer_types: bool,
    pub sliding_period_key: Option<String>,
    pub attention_indices_key: Option<String>,
    pub types_key: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyMixer {
    pub module: String,
    pub projection: Vec<crate::spec::Projection>,
    pub out_tensor: String,
    pub out_cols: String,
    pub params: std::collections::BTreeMap<String, ParamSource>,
    pub scratch: std::collections::BTreeMap<String, String>,
    pub node: Vec<crate::graph::Node>,
}

impl FamilyMixer {
    pub fn read(&self, config: &serde_json::Value) -> Result<Option<crate::spec::Mixer>> {
        if self.node.is_empty() {
            return Ok(None);
        }
        Ok(Some(crate::spec::Mixer {
            module: self.module.clone(),
            projection: self.projection.clone(),
            out_tensor: self.out_tensor.clone(),
            out_cols: self.out_cols.clone(),
            params: read_params(&self.params, config)?,
            scratch: self.scratch.clone(),
            node: self.node.clone(),
        }))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyActivation {
    pub map: std::collections::BTreeMap<String, crate::spec::Activation>,
    pub lenient_default: Option<crate::spec::Activation>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyDtypes {
    pub keys: Vec<String>,
    pub accepted: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyRope {
    pub type_keys: Vec<String>,
    pub unscaled: Vec<String>,
    pub types: std::collections::BTreeMap<String, crate::spec::RopeScalingKind>,
    pub keys: std::collections::BTreeMap<String, String>,
    pub local_theta_key: Option<String>,
    pub nope_flags: Option<String>,
    pub theta_key: Option<String>,
    pub scaling_path: Option<String>,
    pub rotary_dim_key: Option<String>,
    pub yarn_attention_factor: Option<f64>,
    pub nope_key: Option<String>,
    pub nope_value: Option<String>,
