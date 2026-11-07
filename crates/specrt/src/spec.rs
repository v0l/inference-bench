use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const SPEC_FORMAT: u32 = 1;

pub const EMBEDDED: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/embedded_specs.rs"));

#[derive(Clone, Debug, PartialEq)]
pub enum Checked {
    Ready(Box<Spec>),
    NeedsNewer { name: String, min_engine: u32 },
}

pub fn needs_newer_note(min_engine: u32) -> String {
    format!("needs a newer inference-bench (spec format {min_engine}, this build reads {SPEC_FORMAT})")
}

pub fn check(text: &str) -> Result<Checked> {
    let value: toml::Table = toml::from_str(text)?;
    let min_engine = match value.get("min_engine") {
        None => 0,
        Some(v) => v.as_integer().and_then(|n| u32::try_from(n).ok()).context("min_engine must be a whole number")?,
    };
    if min_engine > SPEC_FORMAT {
        let name = value.get("name").and_then(|n| n.as_str()).context("spec has no name")?;
        return Ok(Checked::NeedsNewer { name: name.to_string(), min_engine });
    }
    let spec: Spec = toml::from_str(text)?;
    spec.validate()?;
    Ok(Checked::Ready(Box::new(spec)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quant {
    Bf16,
    Q8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Block {
    Dense,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NormKind {
    Rmsnorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionKind {
    Gqa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeStyle {
    Neox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Swiglu,
    Geglu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingKind {
    Linear,
    Llama3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeScaling {
    pub kind: RopeScalingKind,
    pub factor: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_freq_factor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_freq_factor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_max_positions: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dims {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub tie_embeddings: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layers {
    pub count: usize,
    pub pattern: Vec<Block>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Norm {
    pub kind: NormKind,
    pub eps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attention {
    pub kind: AttentionKind,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qkv_bias: bool,
    pub qk_norm: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub style: RopeStyle,
    pub theta: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling: Option<RopeScaling>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mlp {
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub name: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub min_engine: u32,
    pub repo: String,
    pub quant: Quant,
    pub dims: Dims,
    pub layers: Layers,
    pub norm: Norm,
    pub attention: Attention,
    pub rope: Rope,
    pub mlp: Mlp,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    attention_bias: Option<bool>,
    rope_scaling: Option<serde_json::Value>,
    torch_dtype: Option<String>,
}

fn parse_rope_scaling(value: Option<&serde_json::Value>) -> Result<Option<RopeScaling>> {
    let Some(v) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let kind = v
        .get("rope_type")
        .or_else(|| v.get("type"))
        .and_then(|k| k.as_str())
        .context("rope_scaling has no type")?;
    let num = |key: &str| v.get(key).and_then(|x| x.as_f64());
    let factor = num("factor").context("rope_scaling has no factor")?;
    ensure!(factor > 0.0, "rope_scaling factor must be positive");
    match kind {
        "default" => Ok(None),
        "linear" => Ok(Some(RopeScaling {
            kind: RopeScalingKind::Linear,
            factor,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_positions: None,
        })),
        "llama3" => Ok(Some(RopeScaling {
            kind: RopeScalingKind::Llama3,
            factor,
            low_freq_factor: Some(num("low_freq_factor").context("no low_freq_factor")?),
            high_freq_factor: Some(num("high_freq_factor").context("no high_freq_factor")?),
            original_max_positions: Some(
                num("original_max_position_embeddings")
                    .context("no original_max_position_embeddings")?,
            ),
        })),
        other => bail!("rope_scaling {other} is not supported"),
    }
}

impl Spec {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        Self::parse(&text).with_context(|| path.display().to_string())
    }

    pub fn parse(text: &str) -> Result<Self> {
        match check(text)? {
            Checked::Ready(spec) => Ok(*spec),
            Checked::NeedsNewer { name, min_engine } => bail!("spec {name} {}", needs_newer_note(min_engine)),
        }
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string(self)?)
    }

    pub fn from_hf_config(json: &str) -> Result<Self> {
        let hf: HfConfig = serde_json::from_str(json)?;
        let scaling = parse_rope_scaling(hf.rope_scaling.as_ref())?;
        let activation = match hf.hidden_act.as_str() {
            "silu" => Activation::Swiglu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Activation::Geglu,
            other => bail!("hidden_act {other} is not supported"),
        };
        if let Some(dtype) = hf.torch_dtype.as_deref()
            && !matches!(dtype, "bfloat16" | "float16" | "float32")
        {
            bail!("torch_dtype {dtype} is not supported");
        }
        let (qkv_bias, qk_norm) = match hf.model_type.as_str() {
            "qwen2" => (true, false),
            "qwen3" => (hf.attention_bias.unwrap_or(false), true),
            "llama" | "mistral" => (hf.attention_bias.unwrap_or(false), false),
            other => bail!("model_type {other} is not supported"),
        };
        let spec = Spec {
            name: hf.model_type.clone(),
            min_engine: 0,
            repo: String::new(),
            quant: Quant::Bf16,
            dims: Dims {
                hidden: hf.hidden_size,
                intermediate: hf.intermediate_size,
                vocab: hf.vocab_size,
                tie_embeddings: hf.tie_word_embeddings,
            },
            layers: Layers { count: hf.num_hidden_layers, pattern: vec![Block::Dense] },
            norm: Norm { kind: NormKind::Rmsnorm, eps: hf.rms_norm_eps },
            attention: Attention {
                kind: AttentionKind::Gqa,
                heads: hf.num_attention_heads,
                kv_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
                head_dim: hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads),
                qkv_bias,
                qk_norm,
            },
            rope: Rope { style: RopeStyle::Neox, theta: hf.rope_theta, scaling },
            mlp: Mlp { activation },
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        let a = &self.attention;
        ensure!(
            a.kv_heads > 0 && a.heads.is_multiple_of(a.kv_heads),
            "heads must be a multiple of kv_heads"
        );
        ensure!(
            a.head_dim.is_multiple_of(2) && a.head_dim <= 256,
            "head_dim must be even and at most 256"
        );
        ensure!(self.dims.hidden.is_multiple_of(8), "hidden must be a multiple of 8");
        ensure!(self.dims.intermediate.is_multiple_of(8), "intermediate must be a multiple of 8");
        ensure!(!self.layers.pattern.is_empty(), "layer pattern is empty");
        ensure!(self.layers.count > 0, "no layers");
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.attention.heads * self.attention.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.attention.kv_heads * self.attention.head_dim
    }
}
