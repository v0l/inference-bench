use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SPEC_FORMAT: u32 = 7;

pub const EMBEDDED: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/embedded_specs.rs"));

#[derive(Clone, Debug, PartialEq)]
pub enum Checked {
    Ready(Box<Spec>),
    NeedsNewer { name: String, min_engine: u32 },
}

pub fn needs_newer_note(min_engine: u32) -> String {
    format!("needs a newer inference-bench (spec format {min_engine}, this build reads {SPEC_FORMAT})")
}

const MOE_UNSUPPORTED: &str = "mixture-of-experts models are not supported by the specrt engine";

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
    ensure!(spec.moe.is_none(), "{MOE_UNSUPPORTED}");
    Ok(Checked::Ready(Box::new(spec)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quant {
    Bf16,
    Q8,
    #[serde(rename = "fp8_block128")]
    Fp8Block128,
    #[serde(rename = "fp8_channel")]
    Fp8Channel,
    #[serde(rename = "nvfp4_g16")]
    Nvfp4G16,
}

impl Quant {
    pub fn is_fp8(self) -> bool {
        matches!(self, Quant::Fp8Block128 | Quant::Fp8Channel)
    }

    pub fn quantised_checkpoint(self) -> bool {
        matches!(self, Quant::Fp8Block128 | Quant::Fp8Channel | Quant::Nvfp4G16)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActQuant {
    #[default]
    None,
    Fp8,
    Fp4,
}

impl ActQuant {
    fn is_none(&self) -> bool {
        *self == ActQuant::None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TileChoice {
    #[default]
    Auto,
    Small,
    Big,
    Narrow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeOptions {
    pub graphs: bool,
    pub fuse: bool,
    pub tile: TileChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub big_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_stages: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unsplit: Vec<String>,
    pub gemm_fp32: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "PrefixCacheOptions::is_default")]
    pub prefix_cache: PrefixCacheOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrefixCacheOptions {
    pub enabled: bool,
    pub snapshot_interval: usize,
    pub max_snapshots: usize,
    pub max_blocks: usize,
}

impl Default for PrefixCacheOptions {
    fn default() -> Self {
        PrefixCacheOptions { enabled: false, snapshot_interval: 2048, max_snapshots: 4, max_blocks: 0 }
    }
}

impl PrefixCacheOptions {
    pub fn is_default(&self) -> bool {
        *self == PrefixCacheOptions::default()
    }

    pub fn validate(&self) -> Result<()> {
        let page = crate::codegen::PAGE;
        ensure!(
            self.snapshot_interval > 0 && self.snapshot_interval.is_multiple_of(page),
            "[runtime.prefix_cache] snapshot_interval {} must be a positive multiple of the {page} token block",
            self.snapshot_interval
        );
        Ok(())
    }
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions { graphs: true, fuse: true, tile: TileChoice::Auto, big_tile: None, fp4_tile: None, fp4_stages: None, unsplit: Vec::new(), gemm_fp32: false, cache_dir: None, prefix_cache: PrefixCacheOptions::default() }
    }
}

impl RuntimeOptions {
    pub fn is_default(&self) -> bool {
        *self == RuntimeOptions::default()
    }

    fn validate(&self) -> Result<()> {
        self.prefix_cache.validate()?;
        ensure!(self.fp4_stages.is_none_or(|n| n >= 2), "[runtime] fp4_stages needs at least 2");
        for (key, tile) in [("big_tile", self.big_tile), ("fp4_tile", self.fp4_tile)] {
            if let Some([bm, bn, wm, wn]) = tile {
                ensure!(
                    wm > 0 && wn > 0 && bm.is_multiple_of(16 * wm) && bn.is_multiple_of(8 * wn),
                    "[runtime] {key} [{bm}, {bn}, {wm}, {wn}] needs bm a multiple of 16 * wm and bn a multiple of 8 * wn"
                );
            }
        }
        for name in &self.unsplit {
            ensure!(
                crate::codegen::MMA_LINEARS.iter().chain(&crate::codegen::GDN_LINEARS).any(|l| l == name),
                "[runtime] unsplit names {name}, which is not a linear"
            );
        }
        Ok(())
    }
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moe {
    pub experts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<usize>,
    pub expert_intermediate: usize,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub shared_intermediate: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dense_layers: Vec<usize>,
    #[serde(default = "default_sparse_step", skip_serializing_if = "is_one")]
    pub sparse_step: usize,
}

fn default_sparse_step() -> usize {
    1
}

fn is_zero_usize(n: &usize) -> bool {
    *n == 0
}

fn is_one(n: &usize) -> bool {
    *n == 1
}

impl Moe {
    pub fn is_sparse(&self, layer: usize) -> bool {
        !self.dense_layers.contains(&layer) && (layer + 1).is_multiple_of(self.sparse_step.max(1))
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<Quant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_until: Option<usize>,
    #[serde(default, skip_serializing_if = "ActQuant::is_none")]
    pub act_quant: ActQuant,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub name: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub min_engine: u32,
    pub repo: String,
    pub quant: Quant,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_quant: Option<Quant>,
    #[serde(default, skip_serializing_if = "ActQuant::is_none")]
    pub act_quant: ActQuant,
    pub dims: Dims,
    pub layers: Layers,
    pub norm: Norm,
    pub attention: Attention,
    pub rope: Rope,
    pub mlp: Mlp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<Moe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_attention: Option<LinearAttention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<Mtp>,
    #[serde(default, skip_serializing_if = "RuntimeOptions::is_default")]
    pub runtime: RuntimeOptions,
}

pub const MAX_DRAFT_TOKENS: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mtp {
    pub layers: usize,
    #[serde(default = "default_draft_tokens")]
    pub draft_tokens: usize,
    #[serde(default = "default_true")]
    pub speculate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub graph: bool,
}

fn default_draft_tokens() -> usize {
    2
}

fn default_true() -> bool {
    true
}

impl Mtp {
    pub fn new(layers: usize) -> Mtp {
        Mtp { layers, draft_tokens: default_draft_tokens(), speculate: true, max_batch: None, graph: false }
    }
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
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    decoder_sparse_step: Option<usize>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: Option<f64>,
    #[serde(alias = "hidden_activation")]
    hidden_act: Option<String>,
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
    full_attention_interval: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
}

fn layer_pattern(types: &[String], strict: bool) -> Result<Vec<Block>> {
    let blocks = types
        .iter()
        .map(|t| match t.as_str() {
            "linear_attention" => Ok(Block::Linear),
            "full_attention" => Ok(Block::Full),
            _ if !strict => Ok(Block::Full),
            other => bail!("layer type {other} is not supported"),
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!blocks.is_empty(), "layer_types is empty");
    let period = (1..=blocks.len())
        .find(|&p| blocks.len().is_multiple_of(p) && blocks.iter().enumerate().all(|(i, b)| *b == blocks[i % p]))
        .unwrap_or(blocks.len());
    Ok(blocks[..period].to_vec())
}

#[derive(Debug, PartialEq)]
pub struct Quantization {
    pub quant: Quant,
    pub mlp: Option<(Quant, Option<usize>)>,
}

fn compressed_group(name: &str, g: &serde_json::Value) -> Result<Quant> {
    let w = &g["weights"];
    let field = |v: &serde_json::Value, k: &str| v.get(k).cloned().unwrap_or(serde_json::Value::Null);
    let text = |v: &serde_json::Value, k: &str| field(v, k).as_str().map(str::to_string);
    let int = |v: &serde_json::Value, k: &str| field(v, k).as_u64();
    let symmetric = field(w, "symmetric").as_bool() == Some(true);
    match (text(g, "format").as_deref(), text(w, "type").as_deref(), int(w, "num_bits"), text(w, "strategy").as_deref()) {
        (Some("float-quantized"), Some("float"), Some(8), Some("channel")) if symmetric => Ok(Quant::Fp8Channel),
        (Some("nvfp4-pack-quantized"), Some("float"), Some(4), Some("tensor_group"))
            if symmetric && int(w, "group_size") == Some(16) =>
        {
            Ok(Quant::Nvfp4G16)
        }
        _ => bail!("compressed-tensors group {name} is not supported (fp8 per-channel and nvfp4 group 16 are)"),
    }
}

fn target_slot(target: &str) -> Result<(usize, bool)> {
    let rest = target
        .split_once("layers.")
        .map(|(_, r)| r)
        .with_context(|| format!("compressed-tensors target {target} is not a layer projection"))?;
    let (layer, module) = rest.split_once('.').context("layer target without a module")?;
    let layer: usize = layer.parse().with_context(|| format!("target {target}"))?;
    Ok((layer, module.starts_with("mlp.")))
}

fn parse_compressed(q: &serde_json::Value, layers: usize) -> Result<Quantization> {
    let groups = q["config_groups"].as_object().context("compressed-tensors config has no config_groups")?;
    let mut projection: Option<Quant> = None;
    let mut mlp: Vec<Option<Quant>> = vec![None; layers];
    if groups.len() == 1 {
        let (name, g) = groups.iter().next().unwrap();
        if g["targets"].as_array().is_some_and(|t| t.len() == 1 && t[0].as_str() == Some("Linear")) {
            return Ok(Quantization { quant: compressed_group(name, g)?, mlp: None });
        }
    }
    for (name, g) in groups {
        let quant = compressed_group(name, g)?;
        for t in g["targets"].as_array().context("group without targets")? {
            let t = t.as_str().context("target is not a string")?;
            let (layer, is_mlp) = target_slot(t)?;
            ensure!(layer < layers, "target {t} is past the last layer");
            if is_mlp {
                ensure!(mlp[layer].is_none_or(|m| m == quant), "{t} is in two groups");
                mlp[layer] = Some(quant);
            } else {
                ensure!(projection.is_none_or(|p| p == quant), "attention projections mix {projection:?} and {quant:?}");
                projection = Some(quant);
            }
        }
    }
    let quant = projection.context("compressed-tensors config quantises no attention projection")?;
    ensure!(quant == Quant::Fp8Channel, "attention projections in {quant:?} are not supported");
    let mlp: Vec<Quant> = mlp
        .into_iter()
        .enumerate()
        .map(|(i, m)| m.with_context(|| format!("the MLP of layer {i} is not quantised")))
        .collect::<Result<_>>()?;
    let first = mlp[0];
    let until = mlp.iter().position(|&m| m != first).unwrap_or(layers);
    ensure!(
        mlp[until..].iter().all(|&m| m == quant),
        "MLP quantisation must be one kind for the first layers and the projection kind after them"
    );
    let mlp = (first != quant).then_some((first, (until < layers).then_some(until)));
    Ok(Quantization { quant, mlp })
}

fn parse_quantization(value: Option<&serde_json::Value>, layers: usize) -> Result<Quantization> {
    let Some(q) = value.filter(|v| !v.is_null()) else {
        return Ok(Quantization { quant: Quant::Bf16, mlp: None });
    };
    let method = q.get("quant_method").and_then(|m| m.as_str()).unwrap_or("");
    if method == "compressed-tensors" {
        return parse_compressed(q, layers);
    }
    let fmt = q.get("fmt").and_then(|f| f.as_str()).unwrap_or("e4m3");
    let block: Option<Vec<u64>> = q
        .get("weight_block_size")
        .and_then(|b| b.as_array())
        .map(|b| b.iter().filter_map(|x| x.as_u64()).collect());
    match (method, fmt, block.as_deref()) {
        ("fp8", "e4m3", Some([128, 128])) => Ok(Quantization { quant: Quant::Fp8Block128, mlp: None }),
        _ => bail!("quantization {q} is not supported (fp8 e4m3 with 128x128 blocks and compressed-tensors fp8/nvfp4 are)"),
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
        let spec = Self::hf_config(json, true)?;
        spec.validate()?;
        ensure!(spec.moe.is_none(), "{MOE_UNSUPPORTED}");
        ensure!(spec.dims.intermediate > 0, "config has no intermediate_size");
        Ok(spec)
    }

    pub fn describe_hf_config(json: &str) -> Result<Self> {
        Self::hf_config(json, false)
    }

    pub fn apply_structure(&mut self, structure: &crate::structure::Structure) {
        let Some(stack) = structure.main() else { return };
        let any = |group: &str, leaf: &str| structure.any_leaf(group, leaf);
        self.attention.qk_norm = any("q_norm", "weight") || any("k_norm", "weight");
        self.attention.qkv_bias = any("q_proj", "bias") || any("k_proj", "bias") || any("v_proj", "bias");
        let gated = stack.layers.iter().find_map(|l| {
            let q = l.group("self_attn.q_proj")?.weight()?;
            let rows = *q.shape.first()?;
            Some(rows == 2 * self.attention.heads * self.attention.head_dim)
        });
        if let Some(gated) = gated {
            self.attention.output_gate = gated;
        }
    }

    fn hf_config(json: &str, strict: bool) -> Result<Self> {
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        let quant_config = value.get("quantization_config").cloned();
        let outer_type = value.get("model_type").cloned();
        if let Some(text) = value.get_mut("text_config").map(serde_json::Value::take) {
            value = text;
            if let Some(t) = outer_type {
                value["model_type"] = t;
            }
        }
        if strict {
            let model_type = value.get("model_type").and_then(|t| t.as_str()).unwrap_or("");
            ensure!(
                matches!(
                    model_type,
                    "qwen2" | "qwen3" | "qwen3_moe" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next" | "llama" | "mistral"
                ),
                "model_type {model_type} is not supported"
            );
        }
        let hf: HfConfig = serde_json::from_value(value).map_err(|e| {
            let text = e.to_string();
            match text.strip_prefix("missing field `").and_then(|r| r.split_once('`')) {
                Some((field, _)) => anyhow::anyhow!("config.json has no {field}"),
                None => anyhow::Error::from(e),
            }
        })?;
        let quantization = parse_quantization(quant_config.as_ref(), hf.num_hidden_layers);
        let Quantization { quant, mlp: mlp_quant } = match (quantization, strict) {
            (Ok(q), _) => q,
            (Err(e), true) => return Err(e),
            (Err(_), false) => Quantization { quant: Quant::Bf16, mlp: None },
        };
        let nested = hf
            .rope_parameters
            .as_ref()
            .and_then(|p| p.as_object())
            .filter(|_| !strict)
            .and_then(|o| o.values().find(|v| v.get("rope_theta").is_some()));
        let params = hf.rope_parameters.as_ref();
        let param = |key: &str| params.and_then(|p| p.get(key)).or(nested.and_then(|n| n.get(key))).and_then(|v| v.as_f64());
        let scaling = match parse_rope_scaling(hf.rope_scaling.as_ref().or(params)) {
            Ok(s) => s,
            Err(e) if strict => return Err(e),
            Err(_) => None,
        };
        let theta = hf.rope_theta.or(param("rope_theta")).context("no rope_theta")?;
        let activation = match (hf.hidden_act.as_deref(), strict) {
            (Some("silu"), _) => Activation::Swiglu,
            (Some("gelu_pytorch_tanh" | "gelu_new" | "gelu_fast"), _) => Activation::Geglu,
            (_, false) => Activation::Swiglu,
            (Some(other), true) => bail!("hidden_act {other} is not supported"),
            (None, true) => bail!("config.json has no hidden_act"),
        };
        if strict
            && let Some(dtype) = hf.torch_dtype.as_deref().or(hf.dtype.as_deref())
            && !matches!(dtype, "bfloat16" | "float16" | "float32")
        {
            bail!("torch_dtype {dtype} is not supported");
        }
        let family_hybrid = matches!(hf.model_type.as_str(), "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next");
        let config_hybrid = hf.layer_types.as_ref().is_some_and(|t| t.iter().any(|x| x == "linear_attention"))
            || hf.full_attention_interval.is_some_and(|n| n > 1)
            || hf.linear_num_key_heads.is_some();
        let hybrid = if strict { family_hybrid } else { config_hybrid };
        let (qkv_bias, qk_norm) = if strict {
            match hf.model_type.as_str() {
                "qwen2" => (true, false),
                "qwen3" | "qwen3_moe" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next" => {
                    (hf.attention_bias.unwrap_or(false), true)
                }
                "llama" | "mistral" => (hf.attention_bias.unwrap_or(false), false),
                other => bail!("model_type {other} is not supported"),
            }
        } else {
            (hf.attention_bias.unwrap_or(false), false)
        };
        let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads);
        let rotary_dim = hf
            .partial_rotary_factor
            .or(param("partial_rotary_factor"))
            .filter(|f| *f != 1.0)
            .map(|f| (head_dim as f64 * f).round() as usize);
        let pattern = match &hf.layer_types {
            Some(types) if hybrid => layer_pattern(types, strict)?,
            None if hybrid && hf.full_attention_interval.is_some_and(|n| n > 1) => {
                let n = hf.full_attention_interval.unwrap_or(4);
                let mut p = vec![Block::Linear; n - 1];
                p.push(Block::Full);
                p
            }
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
        let moe = match (hf.num_experts, hf.moe_intermediate_size) {
            (Some(experts), Some(expert_intermediate)) if experts > 0 => Some(Moe {
                experts,
                active: hf.num_experts_per_tok,
                expert_intermediate,
                shared_intermediate: hf.shared_expert_intermediate_size.unwrap_or(0),
                dense_layers: hf.mlp_only_layers.clone(),
                sparse_step: hf.decoder_sparse_step.unwrap_or(1),
            }),
            _ => None,
        };
        let spec = Spec {
            name: hf.model_type.clone(),
            min_engine: 0,
            repo: String::new(),
            quant,
            head_quant: None,
            act_quant: ActQuant::None,
            dims: Dims {
                hidden: hf.hidden_size,
                intermediate: hf.intermediate_size.unwrap_or(0),
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
                output_gate: hf.attn_output_gate || (strict && hf.model_type == "qwen3_next"),
                rotary_dim,
            },
            rope: Rope { style: RopeStyle::Neox, theta, scaling },
            mlp: Mlp {
                activation,
                quant: mlp_quant.map(|m| m.0),
                quant_until: mlp_quant.and_then(|m| m.1),
                act_quant: ActQuant::None,
            },
            moe,
            mtp: (hybrid && hf.mtp_num_hidden_layers > 0).then(|| Mtp::new(hf.mtp_num_hidden_layers)),
            linear_attention,
            runtime: RuntimeOptions::default(),
        };
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        self.runtime.validate()?;
        match self.head_quant {
            None | Some(Quant::Bf16) => {}
            Some(Quant::Fp8Channel) => {
                ensure!(self.dims.hidden.is_multiple_of(128), "head_quant = \"fp8_channel\" needs hidden to be a multiple of 128")
            }
            Some(Quant::Nvfp4G16) => {
                ensure!(self.dims.hidden.is_multiple_of(64), "head_quant = \"nvfp4_g16\" needs hidden to be a multiple of 64")
            }
            Some(h) => bail!("head_quant {h:?} is not supported (bf16, fp8_channel and nvfp4_g16 are)"),
        }
        if let Some(m) = &self.mtp {
            ensure!(m.layers == 1, "[mtp] layers = {} is not supported (one MTP layer is)", m.layers);
            ensure!(
                (1..=MAX_DRAFT_TOKENS).contains(&m.draft_tokens),
                "[mtp] draft_tokens must be 1 to {MAX_DRAFT_TOKENS}, not {}",
                m.draft_tokens
            );
            ensure!(m.max_batch != Some(0), "[mtp] max_batch must be at least 1");
            ensure!(self.linear_attention.is_some(), "[mtp] needs a hybrid spec with [linear_attention]");
            ensure!(self.attention.output_gate && self.attention.qk_norm, "[mtp] needs the Qwen3.5 attention (output_gate and qk_norm)");
        }
        match self.act_quant {
            ActQuant::None => {}
            ActQuant::Fp8 => ensure!(self.quant.is_fp8(), "act_quant = \"fp8\" needs fp8 weights (quant = \"fp8_block128\" or \"fp8_channel\")"),
            ActQuant::Fp4 => bail!("act_quant = \"fp4\" is for [mlp] only (act_quant at the top is \"none\" or \"fp8\")"),
        }
        match self.mlp.act_quant {
            ActQuant::None => {}
            ActQuant::Fp4 => ensure!(self.mlp.quant == Some(Quant::Nvfp4G16), "[mlp] act_quant = \"fp4\" needs [mlp] quant = \"nvfp4_g16\""),
            ActQuant::Fp8 => bail!("[mlp] act_quant is \"none\" or \"fp4\"; FP8 MLP inputs follow the top-level act_quant"),
        }
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
        ensure!(self.quant != Quant::Nvfp4G16, "nvfp4_g16 is supported for [mlp] quant only");
        match (self.mlp.quant, self.mlp.quant_until) {
            (None, Some(_)) => bail!("[mlp] quant_until needs [mlp] quant"),
            (Some(m), until) => {
                ensure!(m == Quant::Nvfp4G16 && self.quant == Quant::Fp8Channel, "[mlp] quant {m:?} beside {:?} is not supported (nvfp4_g16 beside fp8_channel is)", self.quant);
                ensure!(until.is_none_or(|u| u > 0 && u < self.layers.count), "[mlp] quant_until must be inside the layer count");
                ensure!(
                    self.dims.hidden.is_multiple_of(256) && self.dims.intermediate.is_multiple_of(256),
                    "nvfp4_g16 needs hidden and intermediate to be multiples of 256"
                );
            }
            (None, None) => {}
        }
        if self.quant.is_fp8() {
            let block = 128;
            let la_ok = self.linear_attention.as_ref().is_none_or(|l| l.value_dim().is_multiple_of(block) && l.conv_dim().is_multiple_of(block));
            ensure!(
                self.dims.hidden.is_multiple_of(block)
                    && self.dims.intermediate.is_multiple_of(block)
                    && self.attention.head_dim.is_multiple_of(block)
                    && self.q_dim().is_multiple_of(block)
                    && la_ok,
                "fp8 weights need hidden, intermediate, head_dim and the linear value dim to be multiples of 128"
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

    pub fn with_env_overrides(&self) -> Spec {
        let mut s = self.clone();
        let env = |name: &str| std::env::var(name).ok();
        let on = |name: &str| env(name).is_some_and(|v| v != "0");
        let mut changed = Vec::new();
        if on("SPECRT_FP8_ACT") && s.quant.is_fp8() && s.act_quant != ActQuant::Fp8 {
            s.act_quant = ActQuant::Fp8;
            changed.push("SPECRT_FP8_ACT");
        }
        if on("SPECRT_FP4_ACT") && s.mlp.quant == Some(Quant::Nvfp4G16) && s.mlp.act_quant != ActQuant::Fp4 {
            s.mlp.act_quant = ActQuant::Fp4;
            changed.push("SPECRT_FP4_ACT");
        }
        if env("SPECRT_NO_GRAPH").is_some() && s.runtime.graphs {
            s.runtime.graphs = false;
            changed.push("SPECRT_NO_GRAPH");
        }
        if env("SPECRT_NO_FUSE").is_some() && s.runtime.fuse {
            s.runtime.fuse = false;
            changed.push("SPECRT_NO_FUSE");
        }
        if env("SPECRT_GEMM_FP32").is_some() && !s.runtime.gemm_fp32 {
            s.runtime.gemm_fp32 = true;
            changed.push("SPECRT_GEMM_FP32");
        }
        let tile = match env("SPECRT_TILE").as_deref() {
            Some("small") => Some(TileChoice::Small),
            Some("big") => Some(TileChoice::Big),
            Some("narrow") => Some(TileChoice::Narrow),
            _ => None,
        };
        if let Some(t) = tile.filter(|t| *t != s.runtime.tile) {
            s.runtime.tile = t;
            changed.push("SPECRT_TILE");
        }
        let big = env("SPECRT_BIG_TILE").and_then(|v| {
            let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            <[usize; 4]>::try_from(p).ok()
        });
        if let Some(b) = big.filter(|b| s.runtime.big_tile != Some(*b)) {
            s.runtime.big_tile = Some(b);
            changed.push("SPECRT_BIG_TILE");
        }
        if let Some(u) = env("SPECRT_UNSPLIT") {
            let names: Vec<String> = u.split(',').filter(|n| !n.is_empty()).map(String::from).collect();
            if !names.is_empty() && names != s.runtime.unsplit {
                s.runtime.unsplit = names;
                changed.push("SPECRT_UNSPLIT");
            }
        }
        if let Some(dir) = std::env::var_os("SPECRT_CACHE").map(PathBuf::from)
            && s.runtime.cache_dir.as_ref() != Some(&dir)
        {
            s.runtime.cache_dir = Some(dir);
            changed.push("SPECRT_CACHE");
        }
        for var in changed {
            tracing::warn!(spec = %s.name, var, "{var} is a development override and is deprecated; set it in the spec (act_quant, [mlp] act_quant or [runtime]) or the server config");
        }
        s
    }

    pub fn lm_head_quant(&self) -> Quant {
        match self.head_quant {
            Some(q) => q,
            None if self.quant.quantised_checkpoint() => Quant::Bf16,
            None => self.quant,
        }
    }

    pub fn mlp_quant(&self, layer: usize) -> Quant {
        match self.mlp.quant {
            Some(q) if self.mlp.quant_until.is_none_or(|u| layer < u) => q,
            _ => self.quant,
        }
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

    pub fn speculative(&self) -> Option<&Mtp> {
        self.mtp.as_ref().filter(|m| m.speculate)
    }

    pub fn kv_layers(&self) -> usize {
        self.attention_layers() + self.speculative().map_or(0, |m| m.layers)
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
