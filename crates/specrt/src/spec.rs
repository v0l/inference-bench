use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const SPEC_FORMAT: u32 = 3;

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
#[serde(rename_all = "snake_case")]
pub enum Quant {
    Bf16,
    Q8,
    #[serde(rename = "fp8_block128")]
    Fp8Block128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Block {
    Dense,
    Full,
    Linear,
}

impl Block {
    pub fn is_linear(self) -> bool {
        self == Block::Linear
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearKind {
    GatedDeltaNet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearAttention {
    pub kind: LinearKind,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl LinearAttention {
    pub fn conv_dim(&self) -> usize {
        2 * self.key_heads * self.key_head_dim + self.value_dim()
    }

    pub fn value_dim(&self) -> usize {
        self.value_heads * self.value_head_dim
    }

    pub fn in_proj_rows(&self) -> usize {
        self.conv_dim() + self.value_dim()
    }

    pub fn ba_rows(&self) -> usize {
        2 * self.value_heads
    }
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
    #[serde(default, skip_serializing_if = "is_false")]
    pub unit_offset: bool,
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
    #[serde(default, skip_serializing_if = "is_false")]
    pub output_gate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotary_dim: Option<usize>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_attention: Option<LinearAttention>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

fn is_false(b: &bool) -> bool {
    !*b
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
    rope_theta: Option<f64>,
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    attention_bias: Option<bool>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    torch_dtype: Option<String>,
    dtype: Option<String>,
    #[serde(default)]
    attn_output_gate: bool,
    partial_rotary_factor: Option<f64>,
    layer_types: Option<Vec<String>>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
}

fn layer_pattern(types: &[String]) -> Result<Vec<Block>> {
    let blocks = types
        .iter()
        .map(|t| match t.as_str() {
            "linear_attention" => Ok(Block::Linear),
            "full_attention" => Ok(Block::Full),
            other => bail!("layer type {other} is not supported"),
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!blocks.is_empty(), "layer_types is empty");
    let period = (1..=blocks.len())
        .find(|&p| blocks.len().is_multiple_of(p) && blocks.iter().enumerate().all(|(i, b)| *b == blocks[i % p]))
        .unwrap_or(blocks.len());
    Ok(blocks[..period].to_vec())
}

fn parse_quantization(value: Option<&serde_json::Value>) -> Result<Quant> {
    let Some(q) = value.filter(|v| !v.is_null()) else {
        return Ok(Quant::Bf16);
    };
    let method = q.get("quant_method").and_then(|m| m.as_str()).unwrap_or("");
    let fmt = q.get("fmt").and_then(|f| f.as_str()).unwrap_or("e4m3");
    let block: Option<Vec<u64>> = q
        .get("weight_block_size")
        .and_then(|b| b.as_array())
        .map(|b| b.iter().filter_map(|x| x.as_u64()).collect());
    match (method, fmt, block.as_deref()) {
        ("fp8", "e4m3", Some([128, 128])) => Ok(Quant::Fp8Block128),
        _ => bail!("quantization {q} is not supported (fp8 e4m3 with 128x128 blocks is)"),
    }
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
    if kind == "default" {
        return Ok(None);
    }
    let num = |key: &str| v.get(key).and_then(|x| x.as_f64());
    let factor = num("factor").context("rope_scaling has no factor")?;
    ensure!(factor > 0.0, "rope_scaling factor must be positive");
    match kind {
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
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        let quant = parse_quantization(value.get("quantization_config"))?;
        let outer_type = value.get("model_type").cloned();
        if let Some(text) = value.get_mut("text_config").map(serde_json::Value::take) {
            value = text;
            if let Some(t) = outer_type {
                value["model_type"] = t;
            }
        }
        let hf: HfConfig = serde_json::from_value(value)?;
        let params = hf.rope_parameters.as_ref();
        let param = |key: &str| params.and_then(|p| p.get(key)).and_then(|v| v.as_f64());
        let scaling = parse_rope_scaling(hf.rope_scaling.as_ref().or(params))?;
        let theta = hf.rope_theta.or(param("rope_theta")).context("no rope_theta")?;
        let activation = match hf.hidden_act.as_str() {
            "silu" => Activation::Swiglu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Activation::Geglu,
            other => bail!("hidden_act {other} is not supported"),
        };
        if let Some(dtype) = hf.torch_dtype.as_deref().or(hf.dtype.as_deref())
            && !matches!(dtype, "bfloat16" | "float16" | "float32")
        {
            bail!("torch_dtype {dtype} is not supported");
        }
        let hybrid = matches!(hf.model_type.as_str(), "qwen3_5" | "qwen3_5_text");
        let (qkv_bias, qk_norm) = match hf.model_type.as_str() {
            "qwen2" => (true, false),
            "qwen3" | "qwen3_5" | "qwen3_5_text" => (hf.attention_bias.unwrap_or(false), true),
            "llama" | "mistral" => (hf.attention_bias.unwrap_or(false), false),
            other => bail!("model_type {other} is not supported"),
        };
        let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads);
        let rotary_dim = hf
            .partial_rotary_factor
            .or(param("partial_rotary_factor"))
            .filter(|f| *f != 1.0)
            .map(|f| (head_dim as f64 * f).round() as usize);
        let pattern = match &hf.layer_types {
            Some(types) if hybrid => layer_pattern(types)?,
            _ => vec![Block::Dense],
        };
        let linear_attention = if pattern.iter().any(|b| b.is_linear()) {
            let need = |v: Option<usize>, key: &str| v.with_context(|| format!("no {key}"));
            Some(LinearAttention {
                kind: LinearKind::GatedDeltaNet,
                key_heads: need(hf.linear_num_key_heads, "linear_num_key_heads")?,
                value_heads: need(hf.linear_num_value_heads, "linear_num_value_heads")?,
                key_head_dim: need(hf.linear_key_head_dim, "linear_key_head_dim")?,
                value_head_dim: need(hf.linear_value_head_dim, "linear_value_head_dim")?,
                conv_kernel: need(hf.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
            })
        } else {
            None
        };
        let spec = Spec {
            name: hf.model_type.clone(),
            min_engine: 0,
            repo: String::new(),
            quant,
            dims: Dims {
                hidden: hf.hidden_size,
                intermediate: hf.intermediate_size,
                vocab: hf.vocab_size,
                tie_embeddings: hf.tie_word_embeddings,
            },
            layers: Layers { count: hf.num_hidden_layers, pattern },
            norm: Norm { kind: NormKind::Rmsnorm, eps: hf.rms_norm_eps, unit_offset: hybrid },
            attention: Attention {
                kind: AttentionKind::Gqa,
                heads: hf.num_attention_heads,
                kv_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
                head_dim,
                qkv_bias,
                qk_norm,
                output_gate: hf.attn_output_gate,
                rotary_dim,
            },
            rope: Rope { style: RopeStyle::Neox, theta, scaling },
            mlp: Mlp { activation },
            linear_attention,
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
        ensure!(
            self.layers.count.is_multiple_of(self.layers.pattern.len()),
            "layer count must be a multiple of the pattern length"
        );
        let rot = self.rotary_dim();
        if self.quant == Quant::Fp8Block128 {
            let block = 128;
            let la_ok = self.linear_attention.as_ref().is_none_or(|l| l.value_dim().is_multiple_of(block) && l.conv_dim().is_multiple_of(block));
            ensure!(
                self.dims.hidden.is_multiple_of(block)
                    && self.dims.intermediate.is_multiple_of(block)
                    && self.attention.head_dim.is_multiple_of(block)
                    && self.q_dim().is_multiple_of(block)
                    && la_ok,
                "fp8_block128 needs hidden, intermediate, head_dim and the linear value dim to be multiples of 128"
            );
        }
        ensure!(rot > 0 && rot.is_multiple_of(2) && rot <= a.head_dim, "rotary_dim must be even and at most head_dim");
        let has_linear = self.layers.pattern.iter().any(|b| b.is_linear());
        match (&self.linear_attention, has_linear) {
            (None, true) => bail!("a linear layer in the pattern needs a [linear_attention] section"),
            (Some(_), false) => bail!("[linear_attention] is set but the pattern has no linear layer"),
            (None, false) => {}
            (Some(la), true) => {
                ensure!(
                    la.key_heads > 0 && la.value_heads.is_multiple_of(la.key_heads),
                    "linear value_heads must be a multiple of key_heads"
                );
                ensure!(
                    la.key_head_dim == la.value_head_dim
                        && la.key_head_dim.is_multiple_of(32)
                        && la.key_head_dim <= 128,
                    "linear key and value head dims must be equal, a multiple of 32 and at most 128"
                );
                ensure!((2..=8).contains(&la.conv_kernel), "conv_kernel must be 2 to 8");
                ensure!(!self.layers.pattern.contains(&Block::Dense), "use full, not dense, beside linear layers");
            }
        }
        Ok(())
    }

    pub fn block(&self, layer: usize) -> Block {
        self.layers.pattern[layer % self.layers.pattern.len()]
    }

    pub fn linear_layers(&self) -> usize {
        (0..self.layers.count).filter(|&i| self.block(i).is_linear()).count()
    }

    pub fn attention_layers(&self) -> usize {
        self.layers.count - self.linear_layers()
    }

    pub fn rotary_dim(&self) -> usize {
        self.attention.rotary_dim.unwrap_or(self.attention.head_dim)
    }

    pub fn qkv_rows(&self) -> usize {
        let gate = if self.attention.output_gate { self.q_dim() } else { 0 };
        self.q_dim() + 2 * self.kv_dim() + gate
    }

    pub fn q_dim(&self) -> usize {
        self.attention.heads * self.attention.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.attention.kv_heads * self.attention.head_dim
    }
}
