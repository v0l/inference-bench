use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SPEC_FORMAT: u32 = 7;

pub const EMBEDDED: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/embedded_specs.rs"));

#[derive(Clone, Debug, PartialEq)]
pub enum Checked {
    Ready(Box<Spec>),
    Family(String),
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
    if crate::family::is_family(&value) {
        let family: crate::family::Family = toml::from_str(text)?;
        return Ok(Checked::Family(family.name));
    }
    let spec: Spec = toml::from_str(text)?;
    spec.validate()?;
    spec.check_moe_runnable()?;
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
    Gguf,
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
pub enum KvDtype {
    #[default]
    F16,
    Fp8,
}

impl KvDtype {
    pub fn is_fp8(self) -> bool {
        self == KvDtype::Fp8
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    #[default]
    Auto,
    Cuda,
    Wgpu,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeOptions {
    pub backend: BackendChoice,
    pub graphs: bool,
    pub fuse: bool,
    pub tile: TileChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub big_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_stages: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_chunk: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unsplit: Vec<String>,
    pub gemm_fp32: bool,
    pub kv_dtype: KvDtype,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_f16_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_extension: Option<RopeExtension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "PrefixCacheOptions::is_default")]
    pub prefix_cache: PrefixCacheOptions,
    #[serde(skip_serializing_if = "crate::loader::LoaderOptions::is_default")]
    pub loader: crate::loader::LoaderOptions,
    #[serde(skip_serializing_if = "VisionOptions::is_default")]
    pub vision: VisionOptions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeExtensionKind {
    #[default]
    Yarn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RopeExtension {
    #[serde(default, rename = "type")]
    pub kind: RopeExtensionKind,
    #[serde(default = "RopeExtension::no_factor")]
    pub factor: u32,
}

impl RopeExtension {
    fn no_factor() -> u32 {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VisionOptions {
    pub precision: crate::vision_encoder::VisionPrecision,
    pub cache_mb: usize,
    pub max_batch: usize,
    pub pan_and_scan: bool,
}

impl Default for VisionOptions {
    fn default() -> Self {
        VisionOptions { precision: crate::vision_encoder::VisionPrecision::Fast, cache_mb: 1024, max_batch: 8, pan_and_scan: false }
    }
}

impl VisionOptions {
    pub fn is_default(&self) -> bool {
        *self == VisionOptions::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrefixCacheOptions {
    pub enabled: bool,
    pub snapshot_interval: usize,
    pub max_snapshots: usize,
    pub max_blocks: usize,
    pub async_snapshots: bool,
}

impl Default for PrefixCacheOptions {
    fn default() -> Self {
        PrefixCacheOptions { enabled: false, snapshot_interval: 2048, max_snapshots: 4, max_blocks: 0, async_snapshots: true }
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
        RuntimeOptions { backend: BackendChoice::Auto, graphs: true, fuse: true, tile: TileChoice::Auto, big_tile: None, fp4_tile: None, fp4_stages: None, prefill_chunk: None, unsplit: Vec::new(), gemm_fp32: false, kv_dtype: KvDtype::F16, kv_f16_layers: None, rope_extension: None, cache_dir: None, prefix_cache: PrefixCacheOptions::default(), loader: crate::loader::LoaderOptions::default(), vision: VisionOptions::default() }
    }
}

impl RuntimeOptions {
    pub fn is_default(&self) -> bool {
        *self == RuntimeOptions::default()
    }

    fn validate(&self) -> Result<()> {
        self.prefix_cache.validate()?;
        self.loader.validate()?;
        ensure!(self.vision.max_batch >= 1, "[runtime.vision] max_batch needs at least 1");
        ensure!(self.fp4_stages.is_none_or(|n| n >= 2), "[runtime] fp4_stages needs at least 2");
        ensure!(self.prefill_chunk.is_none_or(|n| n >= 64), "[runtime] prefill_chunk needs at least 64 tokens");
        ensure!(self.rope_extension.is_none_or(|e| e.factor >= 1), "[runtime] rope_extension factor must be at least 1");
        for (key, tile) in [("big_tile", self.big_tile), ("fp4_tile", self.fp4_tile)] {
            if let Some([bm, bn, wm, wn]) = tile {
                ensure!(
                    wm > 0 && wn > 0 && bm.is_multiple_of(16 * wm) && bn.is_multiple_of(8 * wn),
                    "[runtime] {key} [{bm}, {bn}, {wm}, {wn}] needs bm a multiple of 16 * wm and bn a multiple of 8 * wn"
                );
            }
        }
        if let Some([bm, bn, wm, wn]) = self.fp4_tile {
            ensure!(bn.is_multiple_of(16 * wn), "[runtime] fp4_tile [{bm}, {bn}, {wm}, {wn}] needs bn a multiple of 16 * wn");
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
    Sliding,
    Linear,
}

impl Block {
    pub fn is_linear(self) -> bool {
        self == Block::Linear
    }

    pub fn name(self) -> &'static str {
        match self {
            Block::Dense => "dense",
            Block::Full => "full",
            Block::Sliding => "sliding",
            Block::Linear => "linear",
        }
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
    Yarn,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Yarn {
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: f64,
    pub truncate: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeConstants {
    pub kind: u32,
    pub factor: f64,
    pub low: f64,
    pub high: f64,
    pub original: f64,
    pub mscale: f64,
}

pub fn yarn_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 { 1.0 } else { 0.1 * mscale * scale.ln() + 1.0 }
}

pub fn yarn_correction_range(theta: f64, rot: usize, original: f64, y: &Yarn) -> (f64, f64) {
    let dim = |rotations: f64| rot as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln());
    let (mut low, mut high) = (dim(y.beta_fast), dim(y.beta_slow));
    if y.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let (low, mut high) = (low.max(0.0), high.min(rot as f64 - 1.0));
    if low == high {
        high += 0.001;
    }
    (low, high)
}

pub fn yarn_inv_freq(theta: f64, rot: usize, s: &RopeScaling) -> Vec<f64> {
    let y = s.yarn.as_ref().expect("yarn scaling");
    let original = s.original_max_positions.unwrap_or(1.0);
    let (low, high) = yarn_correction_range(theta, rot, original, y);
    (0..rot / 2)
        .map(|i| {
            let pos_freq = theta.powf(2.0 * i as f64 / rot as f64);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            ramp / (s.factor * pos_freq) + (1.0 - ramp) / pos_freq
        })
        .collect()
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yarn: Option<Yarn>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dims {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub tie_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_softcap: Option<f64>,
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
    #[serde(default, skip_serializing_if = "RouterAct::is_softmax")]
    pub router: RouterAct,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub norm_topk: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub shared_gate: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterAct {
    #[default]
    Softmax,
    Sigmoid,
}

impl RouterAct {
    fn is_softmax(&self) -> bool {
        *self == RouterAct::Softmax
    }
}

fn is_true(b: &bool) -> bool {
    *b
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

    pub fn top_k(&self) -> usize {
        self.active.unwrap_or(1)
    }

    pub fn pairs(&self) -> usize {
        self.top_k() + usize::from(self.shared_intermediate > 0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layers {
    pub count: usize,
    pub pattern: Vec<Block>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Norm {
    pub kind: NormKind,
    pub eps: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub unit_offset: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub sandwich: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub softcap: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_scalar: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub style: RopeStyle,
    pub theta: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling: Option<RopeScaling>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mrope: Option<Mrope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_theta: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mrope {
    pub section: [usize; 3],
    #[serde(default)]
    pub interleaved: bool,
}

fn parse_mrope(params: Option<&serde_json::Value>) -> Option<Mrope> {
    let p = params?;
    let section = p.get("mrope_section")?.as_array()?;
    let mut it = section.iter().filter_map(|v| v.as_u64().map(|v| v as usize));
    let section = [it.next()?, it.next()?, it.next()?];
    let interleaved = p.get("mrope_interleaved").and_then(|v| v.as_bool()).unwrap_or(false);
    Some(Mrope { section, interleaved })
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<crate::vision::Vision>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_quant: Option<Quant>,
}

fn default_draft_tokens() -> usize {
    2
}

fn default_true() -> bool {
    true
}

impl Mtp {
    pub fn new(layers: usize) -> Mtp {
        Mtp { layers, draft_tokens: default_draft_tokens(), speculate: true, max_batch: None, graph: false, head_quant: None }
    }
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

fn is_false(b: &bool) -> bool {
    !*b
}

pub const HF_CONFIG_KEYS: &[&str] = &[
    "model_type", "hidden_size", "intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size", "num_experts",
    "num_experts_per_tok", "mlp_only_layers", "decoder_sparse_step", "norm_topk_prob", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "vocab_size", "rms_norm_eps", "rope_theta", "hidden_act", "hidden_activation", "tie_word_embeddings", "attention_bias",
    "rope_scaling", "rope_parameters", "torch_dtype", "dtype", "attn_output_gate", "partial_rotary_factor", "layer_types",
    "full_attention_interval", "sliding_window", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
    "linear_conv_kernel_dim", "mtp_num_hidden_layers", "attn_logit_softcapping", "final_logit_softcapping", "sliding_window_pattern", "rope_local_base_freq", "query_pre_attn_scalar", "text_config", "quantization_config",
];

pub fn unread_hf_keys(json: &str) -> Vec<(String, String)> {
    let Ok(serde_json::Value::Object(top)) = serde_json::from_str::<serde_json::Value>(json) else { return Vec::new() };
    let mut out = Vec::new();
    let nested = top.get("text_config").and_then(|t| t.as_object());
    for map in std::iter::once(&top).chain(nested) {
        for (k, v) in map {
            if !HF_CONFIG_KEYS.contains(&k.as_str()) && !out.iter().any(|(o, _): &(String, String)| o == k) {
                out.push((k.clone(), v.to_string()));
            }
        }
    }
    out
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
    norm_topk_prob: Option<bool>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: Option<f64>,
    hidden_act: Option<String>,
    tie_word_embeddings: Option<bool>,
    attention_bias: Option<bool>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    attn_output_gate: bool,
    partial_rotary_factor: Option<f64>,
    layer_types: Option<Vec<String>>,
    full_attention_interval: Option<usize>,
    sliding_window: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    attn_logit_softcapping: Option<f64>,
    final_logit_softcapping: Option<f64>,
    query_pre_attn_scalar: Option<f64>,
}

fn layer_pattern(blocks_spec: &crate::family::FamilyBlocks, types: &[String], strict: bool) -> Result<Vec<Block>> {
    let blocks = types
        .iter()
        .map(|t| match blocks_spec.layer_types.get(t) {
            Some(b) => Ok(*b),
            None if !strict => Ok(blocks_spec.other_layer_type.unwrap_or(Block::Full)),
            None => bail!("layer type {t} is not supported"),
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
    pub act: ActQuant,
    pub mlp_act: ActQuant,
}

impl Quantization {
    fn weights(quant: Quant) -> Self {
        Quantization { quant, mlp: None, act: ActQuant::None, mlp_act: ActQuant::None }
    }
}

fn compressed_act(rule: &crate::family::FamilyQuant, g: &serde_json::Value) -> ActQuant {
    crate::family::first_match(&rule.act_rules, g).unwrap_or(ActQuant::None)
}

fn compressed_group(rule: &crate::family::FamilyQuant, name: &str, g: &serde_json::Value) -> Result<Quant> {
    match crate::family::first_match(&rule.group_rules, g) {
        Some(q) => Ok(q),
        None => bail!("{}", rule.group_error.replace("{name}", name)),
    }
}

fn target_slot(rule: &crate::family::FamilyQuant, target: &str) -> Result<(usize, bool)> {
    let rest = target
        .split_once(rule.layer_marker.as_str())
        .map(|(_, r)| r)
        .with_context(|| format!("quantisation target {target} is not a layer projection"))?;
    let (layer, module) = rest.split_once('.').context("layer target without a module")?;
    let layer: usize = layer.parse().with_context(|| format!("target {target}"))?;
    Ok((layer, module.starts_with(rule.mlp_marker.as_str())))
}

fn parse_compressed(rule: &crate::family::FamilyQuant, q: &serde_json::Value, layers: usize) -> Result<Quantization> {
    let groups = q[rule.groups_key.as_str()].as_object().context("the quantisation config has no groups")?;
    let mut projection: Option<Quant> = None;
    let mut act = ActQuant::None;
    let mut mlp_act = ActQuant::None;
    let mut mlp: Vec<Option<Quant>> = vec![None; layers];
    if groups.len() == 1 {
        let (name, g) = groups.iter().next().unwrap();
        if g["targets"].as_array().is_some_and(|t| t.len() == 1 && t[0].as_str() == Some(rule.all_linear_target.as_str())) {
            let quant = compressed_group(rule, name, g)?;
            let act = if quant.is_fp8() { compressed_act(rule, g) } else { ActQuant::None };
            return Ok(Quantization { quant, mlp: None, act, mlp_act: ActQuant::None });
        }
    }
    for (name, g) in groups {
        let quant = compressed_group(rule, name, g)?;
        let group_act = compressed_act(rule, g);
        for t in g["targets"].as_array().context("group without targets")? {
            let t = t.as_str().context("target is not a string")?;
            let (layer, is_mlp) = target_slot(rule, t)?;
            ensure!(layer < layers, "target {t} is past the last layer");
            if is_mlp {
                ensure!(mlp[layer].is_none_or(|m| m == quant), "{t} is in two groups");
                mlp[layer] = Some(quant);
                if quant == Quant::Nvfp4G16 {
                    mlp_act = group_act;
                }
            } else {
                ensure!(projection.is_none_or(|p| p == quant), "attention projections mix {projection:?} and {quant:?}");
                projection = Some(quant);
                act = group_act;
            }
        }
    }
    let quant = projection.context("the quantisation config covers no attention projection")?;
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
    let act = if quant.is_fp8() { act } else { ActQuant::None };
    let mlp_act = if mlp.is_some_and(|m| m.0 == Quant::Nvfp4G16) && mlp_act == ActQuant::Fp4 { ActQuant::Fp4 } else { ActQuant::None };
    Ok(Quantization { quant, mlp, act, mlp_act })
}

fn parse_quantization(rule: &crate::family::FamilyQuant, value: Option<&serde_json::Value>, layers: usize) -> Result<Quantization> {
    let Some(q) = value.filter(|v| !v.is_null()) else {
        return Ok(Quantization::weights(Quant::Bf16));
    };
    let method = q.get(rule.method_key.as_str()).and_then(|m| m.as_str()).unwrap_or("");
    if rule.grouped_methods.iter().any(|m| m == method) {
        return parse_compressed(rule, q, layers);
    }
    match crate::family::first_match(&rule.flat_rules, q) {
        Some(quant) => Ok(Quantization::weights(quant)),
        None => bail!("{}", rule.flat_error.replace("{q}", &q.to_string())),
    }
}

fn parse_rope_scaling(rope: &crate::family::FamilyRope, value: Option<&serde_json::Value>) -> Result<Option<RopeScaling>> {
    let Some(v) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let kind = rope
        .type_keys
        .iter()
        .find_map(|k| v.get(k))
        .and_then(|k| k.as_str())
        .context("rope_scaling has no type")?;
    if rope.unscaled.iter().any(|u| u == kind) {
        return Ok(None);
    }
    let num = |key: &str| v.get(rope.key(key)).and_then(|x| x.as_f64());
    let factor = num("factor").context("rope_scaling has no factor")?;
    ensure!(factor > 0.0, "rope_scaling factor must be positive");
    let Some(&scaling_kind) = rope.types.get(kind) else { bail!("rope_scaling {kind} is not supported") };
    match scaling_kind {
        RopeScalingKind::Linear => Ok(Some(RopeScaling {
            kind: scaling_kind,
            factor,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_positions: None,
            yarn: None,
        })),
        RopeScalingKind::Llama3 => Ok(Some(RopeScaling {
            kind: scaling_kind,
            factor,
            low_freq_factor: Some(num("low_freq_factor").context("no low_freq_factor")?),
            high_freq_factor: Some(num("high_freq_factor").context("no high_freq_factor")?),
            original_max_positions: Some(
                num("original_max_positions")
                    .context("no original_max_position_embeddings")?,
            ),
            yarn: None,
        })),
        RopeScalingKind::Yarn => {
            let original = num("original_max_positions").context("yarn rope_scaling has no original_max_position_embeddings")?;
            ensure!(original > 0.0, "original_max_position_embeddings must be positive");
            let attention_factor = match (num("attention_factor"), num("mscale"), num("mscale_all_dim")) {
                (Some(a), _, _) => a,
                (None, Some(m), Some(d)) if m != 0.0 && d != 0.0 => yarn_mscale(factor, m) / yarn_mscale(factor, d),
                _ => yarn_mscale(factor, 1.0),
            };
            let beta_fast = num("beta_fast").filter(|b| *b != 0.0).unwrap_or(32.0);
            let beta_slow = num("beta_slow").filter(|b| *b != 0.0).unwrap_or(1.0);
            let truncate = v.get(rope.key("truncate")).and_then(|t| t.as_bool()).unwrap_or(true);
            Ok(Some(RopeScaling {
                kind: scaling_kind,
                factor,
                low_freq_factor: None,
                high_freq_factor: None,
                original_max_positions: Some(original),
                yarn: Some(Yarn { beta_fast, beta_slow, attention_factor, truncate }),
            }))
        }
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
            Checked::Family(name) => bail!("{name} is a family spec, not a model spec"),
            Checked::NeedsNewer { name, min_engine } => bail!("spec {name} {}", needs_newer_note(min_engine)),
        }
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string(self)?)
    }

    pub fn from_gguf_file(path: &Path) -> Result<Self> {
        let mut spec = crate::gguf::Gguf::open(path)?.spec()?;
        spec.repo = path.display().to_string();
        Ok(spec)
    }

    pub fn from_hf_config(json: &str) -> Result<Self> {
        let mut spec = Self::from_hf_config_serving(json)?;
        spec.act_quant = ActQuant::None;
        spec.mlp.act_quant = ActQuant::None;
        Ok(spec)
    }

    pub fn from_hf_config_serving(json: &str) -> Result<Self> {
        Self::from_hf_config_serving_in(&crate::family::active(), json)
    }

    pub fn from_hf_config_serving_in(registry: &crate::family::Registry, json: &str) -> Result<Self> {
        let spec = Self::hf_config(registry, json, true)?;
        spec.validate()?;
        spec.check_moe_runnable()?;
        ensure!(spec.dims.intermediate > 0 || spec.moe.is_some(), "config has no intermediate_size");
        Ok(spec)
    }

    pub fn runnable(&self) -> Result<()> {
        self.validate()?;
        self.check_moe_runnable()?;
        ensure!(self.dims.intermediate > 0 || self.moe.is_some(), "the config has no intermediate_size");
        Ok(())
    }

    pub fn check_moe_runnable(&self) -> Result<()> {
        let Some(m) = &self.moe else { return Ok(()) };
        ensure!(self.linear_attention.is_some() || self.attention_layers() > 0, "mixture-of-experts needs attention layers");
        let experts_quant = self.mlp.quant.unwrap_or(self.quant);
        ensure!(
            matches!(experts_quant, Quant::Bf16 | Quant::Nvfp4G16),
            "mixture-of-experts with {experts_quant:?} expert weights is not supported yet (bf16 and nvfp4_g16 are)"
        );
        ensure!(self.mtp.is_none(), "mixture-of-experts with an MTP layer is not supported yet");
        ensure!(self.quant != Quant::Gguf, "mixture-of-experts with GGUF weights is not supported yet");
        ensure!(m.active.is_some_and(|k| k >= 1 && k <= m.experts), "mixture-of-experts needs num_experts_per_tok between 1 and the expert count");
        ensure!(m.expert_intermediate.is_multiple_of(256) || experts_quant == Quant::Bf16, "nvfp4 experts need an intermediate size that is a multiple of 256");
        ensure!(m.expert_intermediate.is_multiple_of(16) && m.shared_intermediate.is_multiple_of(16), "expert intermediate sizes must be multiples of 16");
        ensure!(self.dims.hidden.is_multiple_of(256), "mixture-of-experts needs hidden to be a multiple of 256");
        Ok(())
    }

    pub fn complete_from_checkpoint_config(&mut self, json: &str) -> Result<()> {
        if self.vision.is_none() {
            self.vision = crate::vision::Vision::from_hf_config(json)?;
        }
        if self.rope.mrope.is_none() && self.vision.is_some() {
            let value: serde_json::Value = serde_json::from_str(json)?;
            let text = value.get("text_config").unwrap_or(&value);
            self.rope.mrope = parse_mrope(text.get("rope_parameters").or(text.get("rope_scaling")));
            self.validate()?;
        }
        Ok(())
    }

    pub fn describe_hf_config(json: &str) -> Result<Self> {
        Self::hf_config(&crate::family::active(), json, false)
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

    fn hf_config(registry: &crate::family::Registry, json: &str, strict: bool) -> Result<Self> {
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        let quant_config = value.get("quantization_config").cloned();
        let outer_type = value.get("model_type").cloned();
        if let Some(text) = value.get_mut("text_config").map(serde_json::Value::take) {
            value = text;
            if let Some(t) = outer_type {
                value["model_type"] = t;
            }
        }
        let family = if strict {
            registry.strict_for(value.get("model_type").and_then(|t| t.as_str()).unwrap_or(""))?
        } else {
            registry.describing()?
        };
        let attn_flags = (
            family.attention.qkv_bias.read(&value),
            family.attention.qk_norm.read(&value),
            family.attention.output_gate.read(&value),
        );
        family.normalise(&mut value);
        let dtype_values: std::collections::HashMap<&str, String> = family
            .dtypes
            .keys
            .iter()
            .filter_map(|k| Some((k.as_str(), value.get(k)?.as_str()?.to_string())))
            .collect();
        let hf: HfConfig = serde_json::from_value(value.clone()).map_err(|e| {
            let text = e.to_string();
            match text.strip_prefix("missing field `").and_then(|r| r.split_once('`')) {
                Some((field, _)) => anyhow::anyhow!("config.json has no {field}"),
                None => anyhow::Error::from(e),
            }
        })?;
        let quantization = parse_quantization(&family.quantization, quant_config.as_ref(), hf.num_hidden_layers);
        let Quantization { quant, mlp: mlp_quant, act, mlp_act } = match (quantization, strict) {
            (Ok(q), _) => q,
            (Err(e), true) => return Err(e),
            (Err(_), false) => Quantization::weights(Quant::Bf16),
        };
        let experts_nvfp4 = hf.num_experts.is_some_and(|e| e > 0) && quant == Quant::Nvfp4G16;
        let (quant, mlp_quant, act, mlp_act) =
            if experts_nvfp4 { (Quant::Bf16, Some((Quant::Nvfp4G16, None)), ActQuant::None, ActQuant::None) } else { (quant, mlp_quant, act, mlp_act) };
        let nested = hf
            .rope_parameters
            .as_ref()
            .and_then(|p| p.as_object())
            .filter(|_| !strict)
            .and_then(|o| o.values().find(|v| v.get("rope_theta").is_some()));
        let params = hf.rope_parameters.as_ref();
        let param = |key: &str| params.and_then(|p| p.get(key)).or(nested.and_then(|n| n.get(key))).and_then(|v| v.as_f64());
        let full_params = params.and_then(|p| p.get("full_attention")).filter(|_| family.blocks.layer_types.values().any(|b| *b == Block::Full));
        let scaling = match parse_rope_scaling(&family.rope, hf.rope_scaling.as_ref().or(full_params).or(params)) {
            Ok(s) => s,
            Err(e) if strict => return Err(e),
            Err(_) => None,
        };
        let by_block = |block: Block| {
            let o = params?.as_object()?;
            o.iter()
                .find(|(k, v)| family.blocks.layer_types.get(*k) == Some(&block) && v.get("rope_theta").is_some())
                .and_then(|(_, v)| v.get("rope_theta")?.as_f64())
        };
        let local_theta = by_block(Block::Sliding).or_else(|| family.rope.local_theta_key.as_deref().and_then(|k| value.get(k)?.as_f64()));
        let theta = hf.rope_theta.or(param("rope_theta")).or(by_block(Block::Full)).context("no rope_theta")?;
        let mapped = hf.hidden_act.as_deref().and_then(|a| family.activation.map.get(a).copied());
        let activation = match (mapped, hf.hidden_act.as_deref(), strict) {
            (Some(a), _, _) => a,
            (None, _, false) => family.activation.lenient_default.context("the family has no lenient_default activation")?,
            (None, Some(other), true) => bail!("hidden_act {other} is not supported"),
            (None, None, true) => bail!("config.json has no hidden_act"),
        };
        let stated_dtype = family.dtypes.keys.iter().find_map(|k| dtype_values.get(k.as_str())).map(String::as_str);
        if strict
            && let Some(dtype) = stated_dtype
            && !family.dtypes.accepted.iter().any(|d| d == dtype)
        {
            bail!("torch_dtype {dtype} is not supported");
        }
        let config_hybrid = hf.layer_types.as_ref().is_some_and(|t| t.iter().any(|x| family.blocks.layer_types.get(x).is_some_and(|b| b.is_linear())))
            || hf.full_attention_interval.is_some_and(|n| n > 1)
            || hf.linear_num_key_heads.is_some();
        let hybrid = match family.blocks.kind {
            crate::family::BlockKind::Hybrid => true,
            crate::family::BlockKind::Dense => false,
            crate::family::BlockKind::FromConfig => config_hybrid,
        };
        let (qkv_bias, qk_norm, output_gate) = attn_flags;
        let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads);
        let rotary_dim = hf
            .partial_rotary_factor
            .or(param("partial_rotary_factor"))
            .filter(|f| *f != 1.0)
            .map(|f| (head_dim as f64 * f).round() as usize);
        let pattern = match &hf.layer_types {
            Some(types) if hybrid || !strict || types.iter().any(|t| family.blocks.layer_types.get(t) == Some(&Block::Sliding)) => layer_pattern(&family.blocks, types, strict)?,
            None if family.blocks.sliding_period_key.as_deref().and_then(|k| value.get(k)?.as_u64()).is_some_and(|n| n > 1) && hf.sliding_window.is_some() => {
                let n = family.blocks.sliding_period_key.as_deref().and_then(|k| value.get(k)?.as_u64()).unwrap_or(2) as usize;
                let all: Vec<String> = (0..hf.num_hidden_layers).map(|i| if (i + 1) % n == 0 { "full_attention" } else { "sliding_attention" }.to_string()).collect();
                layer_pattern(&family.blocks, &all, strict)?
            }
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
                router: RouterAct::Softmax,
                norm_topk: hf.norm_topk_prob.unwrap_or(hybrid),
                shared_gate: hf.shared_expert_intermediate_size.unwrap_or(0) > 0,
            }),
            _ => None,
        };
        let has_moe = moe.is_some();
        let spec = Spec {
            name: hf.model_type.clone(),
            min_engine: 0,
            repo: String::new(),
            quant,
            head_quant: None,
            act_quant: act,
            dims: Dims {
                hidden: hf.hidden_size,
                intermediate: hf.intermediate_size.or_else(|| {
                    let m = moe.as_ref().filter(|_| strict)?;
                    Some(if m.shared_intermediate > 0 { m.shared_intermediate } else { m.expert_intermediate })
                }).unwrap_or(0),
                vocab: hf.vocab_size,
                tie_embeddings: hf.tie_word_embeddings.or(family.embedding.tie).unwrap_or(false),
                embed_scale: family.embedding.scale.as_deref().and_then(|s| (s == "sqrt_hidden").then(|| (hf.hidden_size as f64).sqrt())),
                logit_softcap: hf.final_logit_softcapping,
            },
            layers: Layers { count: hf.num_hidden_layers, window: hf.sliding_window.filter(|_| pattern.contains(&Block::Sliding)), pattern },
            norm: Norm { kind: NormKind::Rmsnorm, eps: hf.rms_norm_eps, unit_offset: family.norm.unit_offset.unwrap_or(hybrid), sandwich: family.norm.sandwich },
            attention: Attention {
                kind: AttentionKind::Gqa,
                heads: hf.num_attention_heads,
                kv_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
                head_dim,
                qkv_bias,
                qk_norm,
                output_gate,
                rotary_dim,
                softcap: hf.attn_logit_softcapping,
                query_scalar: hf.query_pre_attn_scalar,
            },
            rope: Rope { style: RopeStyle::Neox, theta, scaling, mrope: parse_mrope(params), local_theta },
            mlp: Mlp {
                activation,
                quant: mlp_quant.map(|m| m.0),
                quant_until: mlp_quant.and_then(|m| m.1),
                act_quant: mlp_act,
            },
            moe,
            mtp: (family.mtp && hybrid && hf.mtp_num_hidden_layers > 0 && !has_moe).then(|| Mtp::new(hf.mtp_num_hidden_layers)),
            linear_attention,
            vision: if strict { crate::vision::Vision::from_hf_config(json)? } else { None },
            runtime: RuntimeOptions::default(),
        };
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        self.runtime.validate()?;
        ensure!(!self.layers.pattern.contains(&Block::Sliding) || self.layers.window.is_some_and(|w| w > 0), "sliding layers need a sliding_window");
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
            match m.head_quant {
                None | Some(Quant::Bf16) => {}
                Some(Quant::Fp8Channel) => {
                    ensure!(self.dims.hidden.is_multiple_of(128), "[mtp] head_quant = \"fp8_channel\" needs hidden to be a multiple of 128")
                }
                Some(Quant::Nvfp4G16) => {
                    ensure!(self.dims.hidden.is_multiple_of(64), "[mtp] head_quant = \"nvfp4_g16\" needs hidden to be a multiple of 64")
                }
                Some(h) => bail!("[mtp] head_quant {h:?} is not supported (bf16, fp8_channel and nvfp4_g16 are)"),
            }
            ensure!(self.linear_attention.is_some(), "[mtp] needs a hybrid spec with [linear_attention]");
            ensure!(self.attention.output_gate && self.attention.qk_norm, "[mtp] needs output-gated, qk-normed attention (output_gate and qk_norm)");
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
        if let Some(m) = &self.rope.mrope {
            ensure!(
                m.section.iter().sum::<usize>() == self.rotary_dim() / 2,
                "mrope_section {:?} must add up to half the rotary dims ({})",
                m.section,
                self.rotary_dim() / 2
            );
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
        if self.quant == Quant::Gguf {
            ensure!(
                [self.dims.hidden, self.dims.intermediate, self.q_dim()].iter().all(|d| d.is_multiple_of(crate::gguf::DEVICE_SUPERBLOCK)),
                "quant = \"gguf\" needs hidden, intermediate and heads * head_dim to be multiples of 256"
            );
            ensure!(self.linear_attention.is_none(), "quant = \"gguf\" is for dense attention models");
        }
        match (self.mlp.quant, self.mlp.quant_until) {
            (None, Some(_)) => bail!("[mlp] quant_until needs [mlp] quant"),
            (Some(m), until) => {
                let beside_fp8 = self.quant == Quant::Fp8Channel;
                let experts = self.quant == Quant::Bf16 && self.moe.is_some();
                ensure!(m == Quant::Nvfp4G16 && (beside_fp8 || experts), "[mlp] quant {m:?} beside {:?} is not supported (nvfp4_g16 beside fp8_channel is, and beside bf16 for mixture-of-experts)", self.quant);
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
            None if self.quant == Quant::Gguf => Quant::Gguf,
            None if self.quant.quantised_checkpoint() => Quant::Bf16,
            None => self.quant,
        }
    }

    pub fn draft_head_quant(&self) -> Option<Quant> {
        let fallback = if self.runtime.gemm_fp32 { Quant::Bf16 } else { Quant::Nvfp4G16 };
        let q = self.speculative()?.head_quant.unwrap_or(fallback);
        (q != self.lm_head_quant()).then_some(q)
    }

    pub fn mlp_quant(&self, layer: usize) -> Quant {
        match self.mlp.quant {
            Some(q) if self.mlp.quant_until.is_none_or(|u| layer < u) => q,
            _ => self.quant,
        }
    }

    pub fn stage(&self, start: usize, end: usize) -> Spec {
        let mut s = self.clone();
        s.name = format!("{}[{start}..{end}]", self.name);
        s.layers.pattern = (start..end).map(|i| self.block(i)).collect();
        s.layers.count = end - start;
        s.mlp.quant_until = self.mlp.quant_until.map(|u| u.saturating_sub(start));
        if !s.layers.pattern.iter().any(|b| b.is_linear()) {
            s.linear_attention = None;
        }
        s.mtp = None;
        s.vision = None;
        s
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

    pub fn kv_f16_layers(&self) -> usize {
        if !self.runtime.kv_dtype.is_fp8() {
            return 0;
        }
        self.runtime.kv_f16_layers.unwrap_or(self.attention_layers() / 2).min(self.kv_layers())
    }

    pub fn kv_layers(&self) -> usize {
        self.attention_layers() + self.speculative().map_or(0, |m| m.layers)
    }

    pub fn rope_constants(&self) -> RopeConstants {
        let Some(s) = self.rope.scaling.as_ref() else {
            return RopeConstants { kind: 0, factor: 1.0, low: 1.0, high: 1.0, original: 1.0, mscale: 1.0 };
        };
        let original = s.original_max_positions.unwrap_or(1.0);
        match (s.kind, s.yarn.as_ref()) {
            (RopeScalingKind::Linear, _) => RopeConstants { kind: 1, factor: s.factor, low: 1.0, high: 1.0, original, mscale: 1.0 },
            (RopeScalingKind::Llama3, _) => RopeConstants {
                kind: 2,
                factor: s.factor,
                low: s.low_freq_factor.unwrap_or(1.0),
                high: s.high_freq_factor.unwrap_or(1.0),
                original,
                mscale: 1.0,
            },
            (RopeScalingKind::Yarn, Some(y)) => {
                let (low, high) = yarn_correction_range(self.rope.theta, self.rotary_dim(), original, y);
                RopeConstants { kind: 3, factor: s.factor, low, high, original, mscale: y.attention_factor }
            }
            (RopeScalingKind::Yarn, None) => RopeConstants { kind: 0, factor: 1.0, low: 1.0, high: 1.0, original: 1.0, mscale: 1.0 },
        }
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
