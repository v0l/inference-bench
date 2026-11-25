use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use kir as kir;

pub const SPEC_FORMAT: u32 = 21;

pub fn modelopt_verified() -> bool {
    include_str!("../modelopt-verified.txt").trim() == "pass"
}

pub const EMBEDDED: &[(&str, &str)] = include!(concat!(env!("OUT_DIR"), "/embedded_specs.rs"));

#[derive(Clone, Debug, PartialEq)]
pub enum Checked {
    Ready(Box<Spec>),
    Family(String),
    Kernel(String),
    NeedsNewer { name: String, min_engine: u32 },
}

pub fn needs_newer_note(min_engine: u32) -> String {
    format!("needs a newer inference-bench (spec format {min_engine}, this build reads {SPEC_FORMAT})")
}


fn rename_old_ids(value: &toml::Table) -> Option<toml::Table> {
    let aliases: toml::Table = toml::from_str(include_str!("../spec-aliases.toml")).expect("spec-aliases.toml is valid");
    let old = value.get("rope")?.get("scaling")?.get("kind")?.as_str()?;
    let new = aliases.get("rope_scaling_kind")?.get(old)?.as_str()?;
    tracing::warn!(old, new, "this rope scaling kind is renamed, update the spec");
    let mut table = value.clone();
    *table.get_mut("rope")?.get_mut("scaling")?.get_mut("kind")? = toml::Value::String(new.to_string());
    Some(table)
}

pub fn check(text: &str) -> Result<Checked> {
    let value: toml::Table = toml::from_str(text)?;
    let renamed = rename_old_ids(&value);
    let renamed_text;
    let text = match renamed {
        Some(table) => {
            renamed_text = toml::to_string(&table)?;
            renamed_text.as_str()
        }
        None => text,
    };
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
    if kir::desc::KIND == value.get("kind").and_then(|k| k.as_str()).unwrap_or_default() {
        let d: kir::desc::Description = toml::from_str(text).map_err(|e| anyhow::anyhow!("kernel description: {e}"))?;
        return Ok(Checked::Kernel(d.name));
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

pub const FP8_KV_VERIFIED_CONTEXT: usize = 32768;
pub const FP8_KV_LONG_VERIFIED_CONTEXT: usize = 163840;

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
    #[serde(skip_serializing_if = "is_false")]
    pub pdl: bool,
    pub fuse: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub store_epilogue: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub pdl_hoist: bool,
    pub tile: TileChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub big_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp8_tile: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp4_stages: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_chunk: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unsplit: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_activations: Option<bool>,
    pub gemm_fp32: bool,
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub chunked_scan: bool,
    pub kv_dtype: KvDtype,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_f16_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_extension: Option<RopeExtension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_max_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compile_jobs: Option<usize>,
    #[serde(skip_serializing_if = "PrefixCacheOptions::is_default")]
    pub prefix_cache: PrefixCacheOptions,
    #[serde(skip_serializing_if = "crate::loader::LoaderOptions::is_default")]
    pub loader: crate::loader::LoaderOptions,
    #[serde(skip_serializing_if = "VisionOptions::is_default")]
    pub vision: VisionOptions,
    #[serde(skip_serializing_if = "crate::offload::OffloadSettings::is_default")]
    pub offload: crate::offload::OffloadSettings,
    #[serde(skip_serializing_if = "Requant::is_default")]
    pub requant: Requant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Requant {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lin_in: Option<Quant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lin_out: Option<Quant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qkv: Option<Quant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub o: Option<Quant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlp_tail: Option<Quant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm_head: Option<Quant>,
}

impl Requant {
    pub const CLASSES: [&'static str; 6] = ["lin_in", "lin_out", "qkv", "o", "mlp_tail", "lm_head"];

    pub fn is_default(&self) -> bool {
        *self == Requant::default()
    }

    pub fn class(&self, linear: &str) -> Option<Quant> {
        match linear {
            "lin_in" => self.lin_in,
            "lin_out" => self.lin_out,
            "qkv" => self.qkv,
            "o" => self.o,
            "gate_up" | "down" | "mlp_tail" => self.mlp_tail,
            "lm_head" => self.lm_head,
            _ => None,
        }
    }

    pub fn set(&mut self, class: &str, q: Option<Quant>) {
        match class {
            "lin_in" => self.lin_in = q,
            "lin_out" => self.lin_out = q,
            "qkv" => self.qkv = q,
            "o" => self.o = q,
            "mlp_tail" => self.mlp_tail = q,
            _ => self.lm_head = q,
        }
    }

    fn validate(&self) -> Result<()> {
        for class in Self::CLASSES {
            let Some(q) = self.class(class) else { continue };
            let allowed: &[Quant] = if class == "lm_head" { &[Quant::Bf16, Quant::Fp8Channel, Quant::Nvfp4G16] } else { &[Quant::Fp8Channel, Quant::Nvfp4G16] };
            ensure!(allowed.contains(&q), "[runtime.requant] {class} = {q:?} is not supported (use nvfp4_g16{})", if class == "lm_head" { ", fp8_channel or bf16" } else { " or fp8_channel" });
        }
        Ok(())
    }
}

fn yes() -> bool {
    true
}

fn is_yes(v: &bool) -> bool {
    *v
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
        PrefixCacheOptions { enabled: false, snapshot_interval: 2048, max_snapshots: 32, max_blocks: 0, async_snapshots: true }
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
        RuntimeOptions { backend: BackendChoice::Auto, graphs: true, pdl: false, fuse: true, store_epilogue: false, pdl_hoist: false, tile: TileChoice::Auto, big_tile: None, fp4_tile: None, fp8_tile: None, fp4_stages: None, prefill_chunk: None, unsplit: Vec::new(), split_activations: None, gemm_fp32: false, chunked_scan: true, kv_dtype: KvDtype::F16, kv_f16_layers: None, rope_extension: None, cache_dir: None, cache_max_bytes: None, compile_jobs: None, prefix_cache: PrefixCacheOptions::default(), loader: crate::loader::LoaderOptions::default(), vision: VisionOptions::default(), offload: crate::offload::OffloadSettings::default(), requant: Requant::default() }
    }
}

impl RuntimeOptions {
    pub fn is_default(&self) -> bool {
        *self == RuntimeOptions::default()
    }

    fn validate(&self) -> Result<()> {
        self.prefix_cache.validate()?;
        self.loader.validate()?;
        self.requant.validate()?;
        ensure!(self.vision.max_batch >= 1, "[runtime.vision] max_batch needs at least 1");
        ensure!(self.fp4_stages.is_none_or(|n| n >= 2), "[runtime] fp4_stages needs at least 2");
        ensure!(self.prefill_chunk.is_none_or(|n| n >= 64), "[runtime] prefill_chunk needs at least 64 tokens");
        ensure!(self.rope_extension.is_none_or(|e| e.factor >= 1), "[runtime] rope_extension factor must be at least 1");
        for (key, tile) in [("big_tile", self.big_tile), ("fp4_tile", self.fp4_tile), ("fp8_tile", self.fp8_tile)] {
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
    Shared,
    Compressed,
    Indexed,
    Mlp,
}

impl Block {
    pub fn is_linear(self) -> bool {
        self == Block::Linear
    }

    pub fn is_attention(self) -> bool {
        !matches!(self, Block::Linear | Block::Mlp)
    }

    pub fn windowed(self) -> bool {
        matches!(self, Block::Sliding | Block::Compressed | Block::Indexed)
    }

    pub fn parse(name: &str) -> Result<Block> {
        [Block::Dense, Block::Full, Block::Sliding, Block::Linear, Block::Shared, Block::Compressed, Block::Indexed, Block::Mlp].into_iter().find(|b| b.name() == name).with_context(|| format!("{name:?} is not a layer kind"))
    }

    pub fn name(self) -> &'static str {
        match self {
            Block::Dense => "dense",
            Block::Full => "full",
            Block::Sliding => "sliding",
            Block::Linear => "linear",
            Block::Shared => "shared",
            Block::Compressed => "compressed",
            Block::Indexed => "indexed",
            Block::Mlp => "mlp",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearKind {
    GatedDeltaNet,
    Kda,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VHeadOrder {
    #[default]
    Grouped,
    Tiled,
}

impl VHeadOrder {
    pub fn is_grouped(&self) -> bool {
        *self == VHeadOrder::Grouped
    }
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
    #[serde(default, skip_serializing_if = "VHeadOrder::is_grouped")]
    pub v_heads: VHeadOrder,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fused_interleaved: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gate_sigmoid: bool,
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
    Mla,
    Dsa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Indexer {
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    #[serde(default = "default_sparse_step", skip_serializing_if = "is_one")]
    pub kpool: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hyper {
    pub streams: usize,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub sinkhorn_iters: usize,
    #[serde(default)]
    pub eps: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sites: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatedResidual {
    pub streams: usize,
    pub module: String,
    pub sites: Vec<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub layer_params: std::collections::BTreeMap<String, Vec<f64>>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub final_norm: bool,
    pub node: Vec<crate::graph::Node>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prelude {
    pub module: String,
    pub layers: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub int_scratch: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub stored: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub words: std::collections::BTreeMap<String, String>,
    pub node: Vec<crate::graph::Node>,
}

impl Prelude {
    pub const BUFFERS: [&'static str; 3] = ["x", "streams", "tokens"];

    pub fn stored_as(&self, weight: &str) -> Option<(&str, &str)> {
        let (prefix, _) = self.stored.iter().find(|(_, w)| *w == weight)?;
        Some((prefix.as_str(), self.words.get(&format!("{prefix}_dtype")).map_or("f32", String::as_str)))
    }

    pub fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }

    pub fn weights(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for n in &self.node {
            for b in n.bind.values() {
                if let crate::graph::Bind::Weight { weight, len } = b
                    && !out.iter().any(|(w, _)| w == weight)
                {
                    out.push((weight.clone(), len.clone()));
                }
            }
        }
        out
    }

    pub fn symbols(&self, slot: usize) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = self.params.iter().map(|(n, v)| (n.clone(), *v)).collect();
        out.extend(self.index.iter().map(|n| (n.clone(), slot as f64)));
        out.extend(self.count.iter().map(|n| (n.clone(), self.layers.len() as f64)));
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockResidual {
    pub size: usize,
    pub sites: Vec<String>,
    pub output: String,
}

pub const MAX_RESIDUAL_BLOCKS: usize = 16;

impl BlockResidual {
    pub fn blocks(&self, layers: usize) -> usize {
        layers.div_ceil(self.size)
    }

    pub fn opens(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedStage {
    Expand,
    Pre,
    Post,
    Collapse,
}

impl GatedStage {
    pub fn op(self) -> &'static str {
        match self {
            GatedStage::Expand => "expand",
            GatedStage::Pre => "pre",
            GatedStage::Collapse => "collapse",
            GatedStage::Post => "post",
        }
    }
}

pub const GATED_ROOT: &str = ".";

impl GatedResidual {
    pub fn width(&self, hidden: usize) -> usize {
        self.streams * hidden
    }

    pub fn collapses(&self) -> bool {
        self.node.iter().any(|n| n.op == "collapse")
    }

    pub fn nodes(&self, stage: GatedStage) -> impl Iterator<Item = (usize, &crate::graph::Node)> {
        let own = self.collapses();
        self.node.iter().enumerate().filter(move |(_, n)| match stage {
            GatedStage::Collapse if !own => n.op == "pre" && n.when.as_deref() != Some("site"),
            _ => n.op == stage.op(),
        })
    }

    pub fn tagged(&self, tag: &str) -> impl Iterator<Item = (usize, &crate::graph::Node)> {
        let tag = tag.to_string();
        self.node.iter().enumerate().filter(move |(_, n)| n.op == tag)
    }

    pub fn tags(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for n in &self.node {
            if !matches!(n.op.as_str(), "expand" | "pre" | "post" | "collapse") && !out.contains(&n.op.as_str()) {
                out.push(n.op.as_str());
            }
        }
        out
    }

    pub fn module_path(module: &str, under: &str) -> String {
        if module == GATED_ROOT { under.to_string() } else { format!("{under}.{module}") }
    }

    pub fn layer_weights(&self) -> Vec<(String, String)> {
        self.collect_weights(self.node.iter().filter(|n| !matches!(n.op.as_str(), "expand" | "collapse")))
    }

    pub fn head_weights(&self) -> Vec<(String, String)> {
        if self.collapses() {
            self.collect_weights(self.node.iter().filter(|n| matches!(n.op.as_str(), "expand" | "collapse")))
        } else {
            self.weights(GatedStage::Collapse)
        }
    }

    fn collect_weights<'a>(&self, nodes: impl Iterator<Item = &'a crate::graph::Node>) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for n in nodes {
            for b in n.bind.values() {
                if let crate::graph::Bind::Weight { weight, len } = b
                    && !out.iter().any(|(w, _)| w == weight)
                {
                    out.push((weight.clone(), len.clone()));
                }
            }
        }
        out
    }

    pub fn weights(&self, stage: GatedStage) -> Vec<(String, String)> {
        self.collect_weights(self.nodes(stage).map(|(_, n)| n))
    }

    pub fn layer_value(&self, name: &str, layer: usize) -> Option<f64> {
        self.layer_params.get(name).and_then(|v| v.get(layer).copied())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerLayer {
    pub dim: usize,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    pub node: Vec<crate::graph::Node>,
}

impl PerLayer {
    pub fn nodes<'a>(&'a self, stage: &'a str) -> impl Iterator<Item = (usize, &'a crate::graph::Node)> {
        self.node.iter().enumerate().filter(move |(_, n)| n.op == stage)
    }

    pub fn weights(&self, stage: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for (_, n) in self.nodes(stage) {
            for b in n.bind.values() {
                if let crate::graph::Bind::Weight { weight, len } = b
                    && !out.iter().any(|(w, _)| w == weight)
                {
                    out.push((weight.clone(), len.clone()));
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputStream {
    pub name: String,
    pub width: usize,
    pub mask: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub streams: Vec<InputStream>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    pub node: Vec<crate::graph::Node>,
}

impl Inputs {
    pub fn buffers(&self) -> Vec<(String, usize)> {
        self.streams.iter().flat_map(|s| [(s.name.clone(), s.width), (s.mask.clone(), 1)]).collect()
    }

    pub fn width_of(&self, buffer: &str) -> Option<usize> {
        self.buffers().into_iter().find(|(n, _)| n == buffer).map(|(_, w)| w)
    }

    pub fn weights(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for n in &self.node {
            for b in n.bind.values() {
                if let crate::graph::Bind::Weight { weight, len } = b
                    && !out.iter().any(|(w, _)| w == weight)
                {
                    out.push((weight.clone(), len.clone()));
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeStyle {
    Neox,
    Pairs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Swiglu,
    Geglu,
    Mapped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingKind {
    Linear,
    Banded,
    Yarn,
    Longrope,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longrope: Option<Longrope>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Longrope {
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
    pub attention_factor: f64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_scale: Option<f64>,
    #[serde(skip)]
    pub head_rows: Option<std::ops::Range<usize>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    #[serde(default = "default_shared_module", skip_serializing_if = "is_default_shared_module")]
    pub shared_module: String,
    #[serde(default = "default_routed_scale", skip_serializing_if = "is_default_routed_scale")]
    pub routed_scale: f64,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub groups: usize,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub topk_group: usize,
    #[serde(default, skip_serializing_if = "is_false")]
    pub score_bias: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_block: Option<crate::dequant::BlockFormat>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub gate_up_interleaved: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub router_bias: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub expert_bias: bool,
    #[serde(default = "default_router_module", skip_serializing_if = "is_default_router_module")]
    pub router_module: String,
    #[serde(default = "default_routed_scale", skip_serializing_if = "is_default_routed_scale")]
    pub out_scale: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub fused_experts: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub input_scaled: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub experts_transposed: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub quantized_experts: bool,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub hash_layers: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_table: Option<String>,
    #[serde(default, skip_serializing_if = "TpSplit::is_auto")]
    pub tp_split: TpSplit,
    #[serde(default, skip_serializing_if = "is_false")]
    pub parallel_dense: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_norm: Option<String>,
    #[serde(default = "default_routed_scale", skip_serializing_if = "is_default_routed_scale")]
    pub router_norm_scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_norm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_scale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent: Option<LatentExperts>,
    #[serde(skip)]
    pub local: Option<ExpertRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentExperts {
    pub width: usize,
    pub down: String,
    pub up: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TpSplit {
    #[default]
    Auto,
    Experts,
    Columns,
}

impl TpSplit {
    fn is_auto(&self) -> bool {
        *self == TpSplit::Auto
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRange {
    pub first: usize,
    pub count: usize,
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

fn default_router_module() -> String {
    "gate".to_string()
}

fn is_default_router_module(s: &String) -> bool {
    s == "gate"
}

fn is_true(b: &bool) -> bool {
    *b
}

pub const DEFAULT_SHARED_MODULE: &str = "shared_expert";

fn default_shared_module() -> String {
    DEFAULT_SHARED_MODULE.to_string()
}

fn is_default_shared_module(s: &String) -> bool {
    s == DEFAULT_SHARED_MODULE
}

fn default_routed_scale() -> f64 {
    1.0
}

fn is_default_routed_scale(v: &f64) -> bool {
    *v == 1.0
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

    pub fn local_experts(&self) -> usize {
        self.local.map_or(self.experts, |r| r.count)
    }

    pub fn first_expert(&self) -> usize {
        self.local.map_or(0, |r| r.first)
    }

    pub fn is_hashed(&self, layer: usize) -> bool {
        self.hash_table.is_some() && layer < self.hash_layers
    }

    pub fn top_k(&self) -> usize {
        self.active.unwrap_or(1)
    }

    pub fn shared_slots(&self) -> usize {
        if self.shared_intermediate == 0 || self.latent.is_some() { 0 } else { (self.shared_intermediate / self.expert_intermediate.max(1)).max(1) }
    }

    pub fn pairs(&self) -> usize {
        self.top_k() + self.shared_slots()
    }

    pub fn width(&self, hidden: usize) -> usize {
        self.latent.as_ref().map_or(hidden, |l| l.width)
    }
}

impl Spec {
    pub fn expert_width(&self) -> usize {
        self.moe.as_ref().map_or(0, |m| m.width(self.dims.hidden))
    }

    pub fn shared_expert_rows(&self) -> usize {
        self.moe.as_ref().map_or(0, |m| m.shared_intermediate)
    }

    pub fn latent_tensors(&self, up: bool) -> Vec<String> {
        self.moe.iter().flat_map(|m| m.latent.as_ref()).map(|l| format!("mlp.{}.weight", if up { &l.up } else { &l.down })).collect()
    }

    pub fn latent_norm_tensors(&self) -> Vec<String> {
        self.moe.iter().flat_map(|m| m.latent.as_ref()).flat_map(|l| l.norm.as_ref()).map(|n| format!("mlp.{n}.weight")).collect()
    }

    pub fn shared_tensors(&self, down: bool) -> Vec<String> {
        let parts: &[&str] = if down { &["down_proj"] } else { &["gate_proj", "up_proj"] };
        self.moe.iter().flat_map(|m| parts.iter().map(move |p| format!("mlp.{}.{p}.weight", m.shared_module))).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layers {
    pub count: usize,
    pub pattern: Vec<Block>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub split: bool,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub shared_tail: usize,
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
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub pre: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_scale: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attention {
    pub kind: AttentionKind,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qkv_bias: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub o_bias: bool,
    pub qk_norm: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub qk_norm_unit: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub output_gate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotary_dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub softcap: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_scalar: Option<f64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub qk_norm_full: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub fused_qkv: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub node: Vec<crate::graph::Node>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projection: Vec<Projection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub o_cols: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub o_cols_by: Vec<OCols>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub int_scratch: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub o_tensor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexer: Option<Indexer>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Mixer {
    pub module: String,
    pub projection: Vec<Projection>,
    pub out_tensor: String,
    pub out_cols: String,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub scratch: std::collections::BTreeMap<String, String>,
    pub node: Vec<crate::graph::Node>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub tensor: String,
    pub rows: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub kv_owner: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub append: bool,
}

impl Projection {
    pub fn present_in(&self, block: Block) -> bool {
        self.layers.is_empty() || self.layers.iter().any(|l| l == block.name())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OCols {
    pub layers: Vec<String>,
    pub cols: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub style: RopeStyle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling: Option<RopeScaling>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mrope: Option<Mrope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_theta: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nope_layers: Vec<usize>,
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
    #[serde(default = "one_plane", skip_serializing_if = "is_one_plane")]
    pub up_act_planes: u8,
    #[serde(default = "one_plane", skip_serializing_if = "is_one_plane")]
    pub down_act_planes: u8,
    #[serde(default, skip_serializing_if = "is_false")]
    pub fused_gate_up: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gated: Option<GatedAct>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped: Option<GatedAct>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub wide_shared_tail: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stored_gate_up: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatedAct {
    pub expr: String,
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, f64>,
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
    pub mixer: Option<Mixer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hyper: Option<Hyper>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gated_residual: Option<GatedResidual>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prelude: Option<Prelude>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer: Option<PerLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_residual: Option<BlockResidual>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Inputs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<Mtp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<crate::vision::Vision>,
    #[serde(default = "crate::state::stock", skip_serializing_if = "crate::state::is_stock")]
    pub state: Vec<crate::state::State>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub tensors: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "RuntimeOptions::is_default")]
    pub runtime: RuntimeOptions,
    #[serde(default, skip_serializing_if = "is_false")]
    pub plain_activations: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub iq_quants: bool,
    #[serde(default, skip_serializing_if = "crate::quant_map::QuantMap::is_empty")]
    pub quant_map: crate::quant_map::QuantMap,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refused: Vec<Refusal>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusal {
    pub part: String,
    pub message: String,
}

impl Spec {
    pub fn refused_part(&self, part: &str) -> Option<&str> {
        self.refused.iter().find(|r| r.part == part).map(|r| r.message.as_str())
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<MtpLayout>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpLayout {
    pub layer: String,
    pub fc: String,
    pub norm_embed: String,
    pub norm_hidden: String,
    #[serde(default)]
    pub norm: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fc_hidden: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixer: Option<String>,
}

impl MtpLayout {
    pub fn stock() -> MtpLayout {
        MtpLayout {
            layer: "mtp.layers.0".into(),
            fc: "mtp.fc.weight".into(),
            norm_embed: "mtp.pre_fc_norm_embedding.weight".into(),
            norm_hidden: "mtp.pre_fc_norm_hidden.weight".into(),
            norm: "mtp.norm.weight".into(),
            head: None,
            fc_hidden: None,
            mixer: None,
        }
    }

    fn at(&self, layers: usize) -> MtpLayout {
        let sub = |t: &str| t.replace("{n}", &layers.to_string());
        MtpLayout {
            layer: sub(&self.layer),
            fc: sub(&self.fc),
            norm_embed: sub(&self.norm_embed),
            norm_hidden: sub(&self.norm_hidden),
            norm: sub(&self.norm),
            head: self.head.as_deref().map(sub),
            fc_hidden: self.fc_hidden.as_deref().map(sub),
            mixer: self.mixer.as_deref().map(sub),
        }
    }

    fn validate(&self) -> Result<()> {
        for (name, v) in [("layer", &self.layer), ("fc", &self.fc), ("norm_embed", &self.norm_embed), ("norm_hidden", &self.norm_hidden)] {
            ensure!(!v.is_empty(), "[mtp.layout] {name} is empty");
        }
        ensure!(!self.norm.is_empty() || self.mixer.is_some(), "[mtp.layout] norm is empty; only a layout with a mixer collapses its streams instead of a norm");
        for (name, v) in [("fc", &self.fc), ("norm_embed", &self.norm_embed), ("norm_hidden", &self.norm_hidden)] {
            ensure!(v.ends_with(".weight"), "[mtp.layout] {name} must name a .weight tensor, not {v:?}");
        }
        ensure!(self.norm.is_empty() || self.norm.ends_with(".weight"), "[mtp.layout] norm must name a .weight tensor, not {:?}", self.norm);
        if let Some(v) = &self.fc_hidden {
            ensure!(v.ends_with(".weight"), "[mtp.layout] fc_hidden must name a .weight tensor, not {v:?}");
        }
        ensure!(self.mixer.as_deref().is_none_or(|m| !m.is_empty() && !m.ends_with(".weight")), "[mtp.layout] mixer names a module, not a tensor");
        Ok(())
    }
}

fn default_draft_tokens() -> usize {
    2
}

fn default_true() -> bool {
    true
}

impl Mtp {
    pub fn new(layers: usize) -> Mtp {
        Mtp { layers, draft_tokens: default_draft_tokens(), speculate: true, max_batch: None, graph: false, head_quant: None, layout: None }
    }
}

fn one_plane() -> u8 {
    1
}

fn is_one_plane(n: &u8) -> bool {
    *n == 1
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
    "n_shared_experts", "routed_scaling_factor", "n_group", "topk_group", "num_hash_layers", "swiglu_alpha", "swiglu_limit", "first_k_dense_replace", "n_routed_experts", "num_nextn_predict_layers", "linear_attn_config",
    "rope_scaling", "rope_parameters", "torch_dtype", "dtype", "attn_output_gate", "partial_rotary_factor", "layer_types",
    "full_attention_interval", "sliding_window", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
    "linear_conv_kernel_dim", "mtp_num_hidden_layers", "attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar", "text_config", "quantization_config",
];

pub fn unread_hf_keys(json: &str) -> Vec<(String, String)> {
    let Ok(serde_json::Value::Object(top)) = serde_json::from_str::<serde_json::Value>(json) else { return Vec::new() };
    let mut known: Vec<String> = HF_CONFIG_KEYS.iter().map(|k| k.to_string()).collect();
    known.extend(crate::family::active().config_keys());
    let mut out = Vec::new();
    let nested = top.get("text_config").and_then(|t| t.as_object());
    for map in std::iter::once(&top).chain(nested) {
        for (k, v) in map {
            if !known.iter().any(|o| o == k) && !out.iter().any(|(o, _): &(String, String)| o == k) {
                out.push((k.clone(), v.to_string()));
            }
        }
    }
    out
}

fn uniform_size<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|n| n as usize),
        Some(serde_json::Value::Array(a)) => {
            let first = a.first().and_then(|x| x.as_u64());
            first.filter(|f| a.iter().all(|x| x.as_u64() == Some(*f))).map(|n| n as usize)
        }
        _ => None,
    })
}

#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    hidden_size: usize,
    #[serde(default, deserialize_with = "uniform_size")]
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    n_shared_experts: Option<usize>,
    first_k_dense_replace: Option<usize>,
    routed_scaling_factor: Option<f64>,
    n_group: Option<usize>,
    num_hash_layers: Option<usize>,
    topk_group: Option<usize>,
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
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
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

fn nope_layers(rope: &crate::family::FamilyRope, config: &serde_json::Value, layers: usize) -> Vec<usize> {
    if rope.nope_all {
        return (0..layers).collect();
    }
    if rope.nope_zero_key.as_deref().and_then(|k| config.get(k)).and_then(|v| v.as_u64()) == Some(0) {
        return (0..layers).collect();
    }
    if let (Some(key), Some(value)) = (rope.nope_key.as_deref(), rope.nope_value.as_deref())
        && config.get(key).and_then(|v| v.as_str()) == Some(value)
    {
        return (0..layers).collect();
    }
    let Some(flags) = rope.nope_flags.as_deref().and_then(|k| config.get(k)).and_then(|v| v.as_array()) else { return Vec::new() };
    flags
        .iter()
        .enumerate()
        .filter(|(_, f)| f.as_u64() == Some(0) || f.as_bool() == Some(false))
        .map(|(i, _)| i)
        .collect()
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

    fn per_tensor(map: &crate::quant_map::QuantMap, experts: bool, layers: usize) -> Self {
        let mlp = (experts && map.fp4_experts(layers)).then_some((Quant::Nvfp4G16, None));
        Quantization { quant: Quant::Bf16, mlp, act: ActQuant::None, mlp_act: ActQuant::None }
    }
}

fn compressed_act(rule: &crate::family::FamilyQuant, g: &serde_json::Value) -> ActQuant {
    crate::family::first_match(&rule.act_rules, g).unwrap_or(ActQuant::None)
}

fn compressed_group(rule: &crate::family::FamilyQuant, name: &str, g: &serde_json::Value, outer: &serde_json::Value) -> Result<Quant> {
    let mut g = g.clone();
    for key in ["format", "quant_method"] {
        if g.get(key).is_none()
            && let Some(f) = outer.get(key)
        {
            g[key] = f.clone();
        }
    }
    match crate::family::first_match(&rule.group_rules, &g) {
        Some(q) => Ok(q),
        None => bail!("{}", rule.group_error.replace("{name}", name)),
    }
}

struct Slot {
    layer: usize,
    module: String,
}

impl Slot {
    fn is_mlp(&self, rule: &crate::family::FamilyQuant) -> bool {
        self.module.starts_with(rule.mlp_marker.as_str())
    }
}

fn target_slot(rule: &crate::family::FamilyQuant, target: &str) -> Result<Slot> {
    let rest = target
        .split_once(rule.layer_marker.as_str())
        .map(|(_, r)| r)
        .with_context(|| format!("quantisation target {target} is not a layer projection"))?;
    let (layer, module) = rest.split_once('.').context("layer target without a module")?;
    let layer: usize = layer.parse().with_context(|| format!("target {target}"))?;
    Ok(Slot { layer, module: module.to_string() })
}

fn anchored(pattern: &str) -> Result<fancy_regex::Regex> {
    fancy_regex::Regex::new(&format!("^(?:{pattern})")).with_context(|| format!("quantisation target pattern {pattern}"))
}

fn pattern_slots(rule: &crate::family::FamilyQuant, pattern: &str, ignore: &[fancy_regex::Regex], layers: usize) -> Result<Vec<Slot>> {
    let re = anchored(pattern)?;
    let mut slots = Vec::new();
    for layer in 0..layers {
        for module in &rule.projections {
            let names: Vec<String> = rule.layer_prefixes.iter().map(|p| format!("{p}{layer}.{module}")).collect();
            let hit = |r: &fancy_regex::Regex| names.iter().any(|n| r.is_match(n).unwrap_or(false));
            if hit(&re) && !ignore.iter().any(hit) {
                slots.push(Slot { layer, module: module.clone() });
            }
        }
    }
    Ok(slots)
}

fn ignore_patterns(q: &serde_json::Value) -> Result<Vec<fancy_regex::Regex>> {
    let list = q["ignore"].as_array().map(Vec::as_slice).unwrap_or_default();
    list.iter().filter_map(|i| i.as_str()?.strip_prefix("re:")).map(anchored).collect()
}

fn quant_groups(rule: &crate::family::FamilyQuant, q: &serde_json::Value) -> Result<serde_json::Map<String, serde_json::Value>> {
    if !rule.layers_key.is_empty()
        && let Some(layers) = q[rule.layers_key.as_str()].as_object()
    {
        let mut groups = serde_json::Map::new();
        for (module, spec) in layers {
            let algo = spec["quant_algo"].as_str().with_context(|| format!("{module} has no quant_algo"))?;
            let mut key = algo.to_string();
            if let Some(size) = spec["group_size"].as_u64() {
                key.push_str(&format!("/g{size}"));
            }
            let group = groups.entry(key).or_insert_with(|| {
                let mut g = spec.clone();
                g["targets"] = serde_json::json!([]);
                g["quant_method"] = q[rule.method_key.as_str()].clone();
                g
            });
            group["targets"].as_array_mut().unwrap().push(module.clone().into());
        }
        return Ok(groups);
    }
    Ok(q[rule.groups_key.as_str()].as_object().context("the quantisation config has no groups")?.clone())
}

fn group_slots(rule: &crate::family::FamilyQuant, q: &serde_json::Value, g: &serde_json::Value, layers: usize) -> Result<Vec<(String, Slot)>> {
    let ignore = ignore_patterns(q)?;
    let mut out = Vec::new();
    for t in g["targets"].as_array().context("group without targets")? {
        let t = t.as_str().context("target is not a string")?;
        if rule.skip_targets.iter().any(|s| s == t) {
            continue;
        }
        if let Some(pattern) = t.strip_prefix("re:") {
            out.extend(pattern_slots(rule, pattern, &ignore, layers)?.into_iter().map(|s| (t.to_string(), s)));
        } else {
            out.push((t.to_string(), target_slot(rule, t)?));
        }
    }
    Ok(out)
}

pub fn module_quants(json: &str) -> Result<Vec<(usize, String, Quant)>> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let layers = value["num_hidden_layers"].as_u64().or_else(|| value["text_config"]["num_hidden_layers"].as_u64()).context("config.json has no num_hidden_layers")? as usize;
    let q = value.get("quantization_config").context("config.json has no quantization_config")?;
    let family = crate::family::active();
    let rule = &family.describing()?.quantization;
    let mut out = Vec::new();
    for (name, g) in quant_groups(rule, q)? {
        let quant = compressed_group(rule, &name, &g, q)?;
        for (_, slot) in group_slots(rule, q, &g, layers)? {
            out.push((slot.layer, slot.module, quant));
        }
    }
    out.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    Ok(out)
}

fn parse_compressed(rule: &crate::family::FamilyQuant, q: &serde_json::Value, layers: usize) -> Result<Quantization> {
    let groups = quant_groups(rule, q)?;
    let mut projection: Option<Quant> = None;
    let mut act = ActQuant::None;
    let mut mlp_act = ActQuant::None;
    let mut mlp: Vec<Option<Quant>> = vec![None; layers];
    if groups.len() == 1 {
        let (name, g) = groups.iter().next().unwrap();
        if g["targets"].as_array().is_some_and(|t| t.len() == 1 && t[0].as_str() == Some(rule.all_linear_target.as_str())) {
            let quant = compressed_group(rule, name, g, q)?;
            let act = if quant.is_fp8() { compressed_act(rule, g) } else { ActQuant::None };
            return Ok(Quantization { quant, mlp: None, act, mlp_act: ActQuant::None });
        }
    }
    for (name, g) in &groups {
        let quant = compressed_group(rule, name, g, q)?;
        let group_act = compressed_act(rule, g);
        for (t, slot) in group_slots(rule, q, g, layers)? {
            ensure!(slot.layer < layers, "target {t} is past the last layer");
            let layer = slot.layer;
            if slot.is_mlp(rule) {
                ensure!(mlp[layer].is_none_or(|m| m == quant), "layer {layer} {} is in two groups", slot.module);
                mlp[layer] = Some(quant);
                if quant == Quant::Nvfp4G16 {
                    mlp_act = group_act;
                }
            } else {
                ensure!(
                    projection.is_none_or(|p| p == quant),
                    "layer {layer} {} is {quant:?} but other attention projections are {projection:?}",
                    slot.module
                );
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

fn moe_act(rule: &crate::family::FamilyQuant, q: Option<&serde_json::Value>) -> ActQuant {
    let groups = q.and_then(|q| q["config_groups"].as_object());
    let Some(g) = groups.and_then(|g| g.values().next()) else { return ActQuant::None };
    let mut g = g.clone();
    for key in ["format", "quant_method", "quant_algo"] {
        if g.get(key).is_none()
            && let Some(v) = q.and_then(|q| q.get(key))
        {
            g[key] = v.clone();
        }
    }
    compressed_act(rule, &g)
}

fn parse_quantization(rule: &crate::family::FamilyQuant, value: Option<&serde_json::Value>, layers: usize) -> Result<Quantization> {
    let Some(q) = value.filter(|v| !v.is_null()) else {
        return Ok(Quantization::weights(Quant::Bf16));
    };
    let method = q.get(rule.method_key.as_str()).and_then(|m| m.as_str()).unwrap_or("");
    let has_groups = q.get(rule.groups_key.as_str()).is_some_and(|g| g.is_object()) || q.get(rule.layers_key.as_str()).is_some_and(|g| g.is_object());
    if rule.grouped_methods.iter().any(|m| m == method) && (has_groups || crate::family::first_match(&rule.flat_rules, q).is_none()) {
        return parse_compressed(rule, q, layers);
    }
    match crate::family::first_match(&rule.flat_rules, q) {
        Some(quant) => Ok(Quantization::weights(quant)),
        None => bail!("{}", rule.flat_error.replace("{q}", &q.to_string())),
    }
}

fn parse_rope_scaling(rope: &crate::family::FamilyRope, value: Option<&serde_json::Value>, config: &serde_json::Value) -> Result<Option<RopeScaling>> {
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
    if rope.types.get(kind) == Some(&RopeScalingKind::Longrope) {
        let list = |key: &str| -> Result<Vec<f64>> {
            let items = v.get(key).and_then(|x| x.as_array()).with_context(|| format!("longrope has no {key}"))?;
            items.iter().map(|x| x.as_f64().with_context(|| format!("longrope {key} has a non-number"))).collect()
        };
        let (short_factor, long_factor) = (list("short_factor")?, list("long_factor")?);
        ensure!(short_factor.len() == long_factor.len() && !short_factor.is_empty(), "longrope short_factor and long_factor must be the same non-zero length");
        let original = config.get(rope.key("original_max_positions")).and_then(|x| x.as_f64()).context("longrope needs original_max_position_embeddings")?;
        let max = config.get("max_position_embeddings").and_then(|x| x.as_f64()).unwrap_or(original);
        let attention_factor = num("attention_factor").unwrap_or_else(|| if max > original { (1.0 + (max / original).ln() / original.ln()).sqrt() } else { 1.0 });
        return Ok(Some(RopeScaling {
            kind: RopeScalingKind::Longrope,
            factor: max / original,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_positions: Some(original),
            yarn: None,
            longrope: Some(Longrope { short_factor, long_factor, attention_factor }),
        }));
    }
    let factor = num("factor").context("rope_scaling has no factor")?;
    ensure!(factor > 0.0, "rope_scaling factor must be positive");
    let Some(&scaling_kind) = rope.types.get(kind) else { bail!("rope_scaling {kind} is not supported") };
    match scaling_kind {
        RopeScalingKind::Longrope => unreachable!("handled above"),
        RopeScalingKind::Linear => Ok(Some(RopeScaling {
            kind: scaling_kind,
            factor,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_positions: None,
            yarn: None,
            longrope: None,
        })),
        RopeScalingKind::Banded => Ok(Some(RopeScaling {
            kind: scaling_kind,
            factor,
            low_freq_factor: Some(num("low_freq_factor").context("no low_freq_factor")?),
            high_freq_factor: Some(num("high_freq_factor").context("no high_freq_factor")?),
            original_max_positions: Some(
                num("original_max_positions")
                    .context("no original_max_position_embeddings")?,
            ),
            yarn: None,
            longrope: None,
        })),
        RopeScalingKind::Yarn => {
            let original = num("original_max_positions").context("yarn rope_scaling has no original_max_position_embeddings")?;
            ensure!(original > 0.0, "original_max_position_embeddings must be positive");
            let attention_factor = match (num("attention_factor"), num("mscale"), num("mscale_all_dim")) {
                (Some(a), _, _) => a,
                (None, None, None) if rope.yarn_attention_factor.is_some() => rope.yarn_attention_factor.unwrap_or(1.0),
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
                longrope: None,
            }))
        }
    }
}

impl Spec {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let mut spec = Self::parse(&text).with_context(|| path.display().to_string())?;
        let beside = path.with_extension("hotlist");
        if spec.runtime.offload.hotlist.is_none() && beside.exists() {
            spec.runtime.offload.hotlist = Some(beside.display().to_string());
        }
        Ok(spec)
    }

    pub fn parse(text: &str) -> Result<Self> {
        match check(text)? {
            Checked::Ready(spec) => Ok(*spec),
            Checked::Family(name) => bail!("{name} is a family spec, not a model spec"),
            Checked::Kernel(name) => bail!("{name} is a kernel description, not a model spec"),
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
        spec.mlp.up_act_planes = 1;
        spec.mlp.down_act_planes = 1;
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

    pub fn expert_tensors(&self) -> Vec<String> {
        let shared = self.moe.as_ref().map_or(DEFAULT_SHARED_MODULE, |m| &m.shared_module);
        vec!["mlp.experts".into(), format!("mlp.{shared}")]
    }

    pub fn check_moe_runnable(&self) -> Result<()> {
        let Some(m) = &self.moe else { return Ok(()) };
        ensure!(self.linear_attention.is_some() || self.attention_layers() > 0, "mixture-of-experts needs attention layers");
        let experts_quant = self.mlp.quant.unwrap_or(self.quant);
        ensure!(self.mlp.mapped.is_none() || experts_quant == Quant::Bf16, "non-gated ([mlp.mapped]) experts run on bf16 weights only");
        ensure!(
            matches!(experts_quant, Quant::Bf16 | Quant::Nvfp4G16 | Quant::Fp8Block128 | Quant::Gguf | Quant::Q8),
            "mixture-of-experts with {experts_quant:?} expert weights is not supported yet (bf16, nvfp4_g16 and fp8_block128 are)"
        );
        ensure!(m.active.is_some_and(|k| k >= 1 && k <= m.experts), "mixture-of-experts needs num_experts_per_tok between 1 and the expert count");
        ensure!(!m.quantized_experts || matches!(experts_quant, Quant::Gguf | Quant::Q8), "[moe] quantized_experts keeps GGUF expert blocks and needs GGUF weights, not {experts_quant:?}");
        ensure!(m.expert_intermediate.is_multiple_of(128) || experts_quant == Quant::Bf16, "quantised experts need an intermediate size that is a multiple of 128");
        ensure!(m.expert_intermediate.is_multiple_of(16) && m.shared_intermediate.is_multiple_of(16), "expert intermediate sizes must be multiples of 16");
        ensure!(
            m.parallel_dense || (m.router_norm.is_none() && m.input_norm.is_none() && m.expert_scale.is_none() && m.router_norm_scale == 1.0),
            "[moe] router_norm, router_norm_scale, input_norm and expert_scale take parallel_dense = true"
        );
        ensure!(
            !m.parallel_dense || (self.norm.sandwich && self.dims.intermediate > 0 && m.dense_layers.is_empty() && m.sparse_step <= 1),
            "[moe] parallel_dense needs sandwich norms, a dense intermediate size and experts on every layer"
        );
        ensure!(
            m.shared_intermediate == 0 || m.shared_intermediate.is_multiple_of(m.expert_intermediate),
            "the shared experts ({} wide) must be a whole number of routed expert widths ({})",
            m.shared_intermediate,
            m.expert_intermediate
        );
        ensure!(m.shared_slots() <= 1 || experts_quant == Quant::Bf16, "shared experts wider than the routed experts are only supported with bf16 weights");
        ensure!(
            self.dims.hidden.is_multiple_of(256) || (m.expert_block.is_some_and(|b| b.device_packed()) && self.dims.hidden.is_multiple_of(64)) || (experts_quant == Quant::Bf16 && self.dims.hidden.is_multiple_of(32)),
            "mixture-of-experts needs hidden to be a multiple of 256 (64 with block scaled experts, 32 with bf16 experts)"
        );
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
        let outer = value.clone();
        let quant_config = value.get("quantization_config").cloned();
        let outer_type = value.get("model_type").cloned();
        let text_key = outer_type.as_ref().and_then(|t| t.as_str()).and_then(|t| registry.for_model_type(t)).and_then(|f| f.text_config.clone()).unwrap_or_else(|| "text_config".into());
        if let Some(text) = value.get_mut(&text_key).map(serde_json::Value::take) {
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
            family.attention.o_bias.read(&value),
            family.attention.qk_norm.read(&value),
            family.attention.output_gate.read(&value),
        );
        family.normalise(&mut value);
        let mut refused = Vec::new();
        if strict {
            for r in &family.refuse {
                match r.part.as_deref() {
                    None => ensure!(!r.hits(&value)?, "{}", r.message),
                    Some("mtp") => {
                        if r.hits(&value)? {
                            refused.push(Refusal { part: "mtp".into(), message: r.message.clone() });
                        }
                    }
                    Some(other) => bail!("[[refuse]] part {other:?} is not mtp"),
                }
            }
        }
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
        let quant_map = match (quant_config.as_ref().map(|q| crate::quant_map::QuantMap::from_config(&family.quantization, q)), strict) {
            (None, _) => Default::default(),
            (Some(Ok(m)), _) => m,
            (Some(Err(e)), true) => return Err(e),
            (Some(Err(_)), false) => Default::default(),
        };
        let quantization = match quantization {
            Err(_) if (family.quantization.per_tensor || !strict) && !quant_map.is_empty() => Ok(Quantization::per_tensor(&quant_map, hf.num_experts.is_some_and(|e| e > 0), hf.num_hidden_layers)),
            other => other,
        };
        let Quantization { quant, mlp: mlp_quant, act, mlp_act } = match (quantization, strict) {
            (Ok(q), _) => q,
            (Err(e), true) => return Err(e),
            (Err(_), false) => Quantization::weights(Quant::Bf16),
        };
        let block_act = family.moe.expert_block.is_some_and(|b| b.native_mma()) && family.moe.act_quant == Some(ActQuant::Fp4);
        let mlp_act = if block_act { ActQuant::Fp4 } else { mlp_act };
        let experts_nvfp4 = hf.num_experts.is_some_and(|e| e > 0) && quant == Quant::Nvfp4G16;
        let (quant, mlp_quant, act, mlp_act) =
            if experts_nvfp4 { (Quant::Bf16, Some((Quant::Nvfp4G16, None)), ActQuant::None, moe_act(&family.quantization, quant_config.as_ref())) } else { (quant, mlp_quant, act, mlp_act) };
        let mlp_act = if strict && hf.num_experts.is_some_and(|e| e > 0) && mlp_quant.map_or(quant, |m| m.0) == Quant::Fp8Block128 { ActQuant::Fp8 } else { mlp_act };
        let nested = hf
            .rope_parameters
            .as_ref()
            .and_then(|p| p.as_object())
            .filter(|_| !strict)
            .and_then(|o| o.values().find(|v| v.get("rope_theta").is_some()));
        let params = hf.rope_parameters.as_ref();
        let param = |key: &str| params.and_then(|p| p.get(key)).or(nested.and_then(|n| n.get(key))).and_then(|v| v.as_f64());
        let full_params = params.and_then(|p| p.get("full_attention")).filter(|_| family.blocks.layer_types.values().any(|b| *b == Block::Full));
        let scaling_source = family.rope.scaling_path.as_deref().and_then(|p| p.split('.').try_fold(&value, |v, k| v.get(k)));
        let scaling = match parse_rope_scaling(&family.rope, hf.rope_scaling.as_ref().or(scaling_source).or(full_params).or(params), &value) {
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
        let local_theta = by_block(Block::Sliding).or_else(|| family.rope.local_theta_key.as_deref().and_then(|k| k.split('.').try_fold(&value, |v, p| v.get(p)).and_then(|v| v.as_f64())));
        let dotted = |path: &str| path.split('.').try_fold(&value, |v, k| v.get(k)).and_then(|v| v.as_f64());
        let theta = family.rope.theta_key.as_deref().and_then(dotted).or(hf.rope_theta).or(param("rope_theta")).or(by_block(Block::Full));
        let theta = theta.or(family.rope.nope_all.then_some(10_000.0));
        let theta = if strict { Some(theta.context("no rope_theta")?) } else { theta };
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
        let (qkv_bias, o_bias, qk_norm, output_gate) = attn_flags;
        let multiplier = |key: &Option<String>| key.as_deref().and_then(|k| value.get(k)).and_then(|v| v.as_f64());
        let head_dim = match family.attention.head_dim.as_deref() {
            Some(expr) => {
                let e = kir::expr::Expr::parse(expr).map_err(anyhow::Error::msg).with_context(|| format!("[attention] head_dim {expr:?}"))?;
                let env = |k: &str| k.split('.').try_fold(&value, |v, p| v.get(p)).and_then(|v| v.as_f64());
                e.eval(&env).with_context(|| format!("[attention] head_dim {expr:?} reads a missing config key"))? as usize
            }
            None => hf.head_dim.filter(|d| *d > 0).unwrap_or(hf.hidden_size / hf.num_attention_heads),
        };
        let keyed_rotary = family.rope.rotary_dim_key.as_deref().and_then(|k| value.get(k)?.as_u64()).map(|d| d as usize);
        let rotary_dim = keyed_rotary.or_else(|| {
            hf.partial_rotary_factor.or(param("partial_rotary_factor")).filter(|f| *f != 1.0).map(|f| (head_dim as f64 * f).round() as usize)
        });
        let indexed = family.blocks.attention_indices_key.as_deref().and_then(|k| k.split('.').try_fold(&value, |v, p| v.get(p))).and_then(|v| v.as_array()).map(|a| {
            let base = family.blocks.attention_indices_base as u64;
            let full: Vec<u64> = a.iter().filter_map(|v| v.as_u64()?.checked_sub(base)).collect();
            (0..hf.num_hidden_layers).map(|i| if full.contains(&(i as u64)) { "full_attention" } else { "linear_attention" }.to_string()).collect::<Vec<_>>()
        });
        let keyed: Option<Vec<String>> = family.blocks.types_key.as_deref().and_then(|k| value.get(k)).and_then(|v| match v {
            serde_json::Value::String(s) => Some(s.chars().map(String::from).collect()),
            serde_json::Value::Array(a) => Some(a.iter().filter_map(|v| v.as_str().map(String::from).or_else(|| v.as_u64().map(|n| n.to_string()))).collect()),
            _ => None,
        });
        if strict && let Some(types) = keyed.as_ref() {
            ensure!(types.len() == hf.num_hidden_layers, "{} names {} layers but num_hidden_layers is {}", family.blocks.types_key.as_deref().unwrap_or_default(), types.len(), hf.num_hidden_layers);
        }
        let dense_types: Vec<usize> = match keyed.as_ref() {
            Some(types) if !family.blocks.sparse_types.is_empty() => types.iter().enumerate().filter(|(_, t)| !family.blocks.sparse_types.contains(t)).map(|(i, _)| i).collect(),
            _ => Vec::new(),
        };
        let mixer = family.mixer.read(&value)?;
        let pattern = match keyed.as_ref().or(indexed.as_ref()).or(hf.layer_types.as_ref()) {
            Some(types) if keyed.is_some() || hybrid || !strict || family.blocks.from_layer_types || types.iter().any(|t| family.blocks.layer_types.get(t) == Some(&Block::Sliding)) => layer_pattern(&family.blocks, types, strict)?,
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
        let linear_attention = if mixer.is_none() && pattern.iter().any(|b| b.is_linear()) {
            let need = |v: Option<usize>, key: &str| v.with_context(|| format!("no {key}"));
            let dims: Result<LinearAttention> = (|| {
                Ok(LinearAttention {
                    kind: crate::family::first_kind(&family.linear.kinds, &value).unwrap_or(LinearKind::GatedDeltaNet),
                    key_heads: need(hf.linear_num_key_heads, "linear_num_key_heads")?,
                    value_heads: need(hf.linear_num_value_heads, "linear_num_value_heads")?,
                    key_head_dim: need(hf.linear_key_head_dim, "linear_key_head_dim")?,
                    value_head_dim: need(hf.linear_value_head_dim, "linear_value_head_dim")?,
                    conv_kernel: need(hf.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
                    v_heads: VHeadOrder::Grouped,
                    fused_interleaved: family.linear.fused_interleaved,
                    gate_sigmoid: family.linear.gate_key.as_deref().and_then(|k| value.get(k).filter(|v| !v.is_null()).or(value.get("hidden_act"))).and_then(|v| v.as_str()) == Some("sigmoid"),
                })
            })();
            if strict { Some(dims?) } else { dims.ok() }
        } else {
            None
        };
        let moe = match (hf.num_experts, hf.moe_intermediate_size) {
            (Some(experts), Some(expert_intermediate)) if experts > 0 => {
                let shared = hf
                    .shared_expert_intermediate_size
                    .or_else(|| hf.n_shared_experts.filter(|n| *n > 0).map(|n| n * expert_intermediate))
                    .unwrap_or(0);
                let mut dense = hf.mlp_only_layers.clone();
                dense.extend(dense_types.iter().copied());
                dense.extend(0..hf.first_k_dense_replace.unwrap_or(0));
                if let Some(sparse) = family.blocks.sparse_layers_key.as_deref().and_then(|k| value.get(k)).and_then(|v| v.as_array()) {
                    let sparse: Vec<usize> = sparse.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect();
                    dense.extend((0..hf.num_hidden_layers).filter(|l| !sparse.contains(l)));
                }
                dense.sort_unstable();
                dense.dedup();
                Some(Moe {
                    experts,
                    active: hf.num_experts_per_tok,
                    expert_intermediate,
                    shared_intermediate: shared,
                    dense_layers: dense,
                    sparse_step: hf.decoder_sparse_step.unwrap_or(1),
                    router: family.moe.router.unwrap_or_default(),
                    norm_topk: hf.norm_topk_prob.or(family.moe.norm_topk).unwrap_or(hybrid),
                    shared_gate: family.moe.shared_gate.unwrap_or_else(|| match family.moe.shared_gate_key.as_deref() {
                        Some(k) => value.get(k).is_some_and(|v| !v.is_null()),
                        None => hf.n_shared_experts.is_some_and(|n| n > 0) || shared > 0,
                    }),
                    shared_module: family.moe.shared_module.clone().unwrap_or_else(default_shared_module),
                    routed_scale: hf.routed_scaling_factor.unwrap_or(1.0),
                    groups: hf.n_group.filter(|g| *g > 1).unwrap_or(0),
                    topk_group: if hf.n_group.is_some_and(|g| g > 1) { hf.topk_group.unwrap_or(1) } else { 0 },
                    score_bias: family.moe.score_bias,
                    expert_block: family.moe.expert_block,
                    gate_up_interleaved: family.moe.gate_up_interleaved,
                    router_bias: family.moe.router_bias,
                    expert_bias: family.moe.expert_bias,
                    router_module: family.moe.router_module.clone().unwrap_or_else(default_router_module),
                    out_scale: multiplier(&family.multipliers.residual).unwrap_or(1.0),
                    fused_experts: family.moe.fused_experts,
                    input_scaled: family.moe.input_scaled,
                    experts_transposed: family.moe.experts_transposed,
                    quantized_experts: false,
                    hash_layers: if family.moe.hash_table.is_some() { hf.num_hash_layers.unwrap_or(0) } else { 0 },
                    hash_table: family.moe.hash_table.clone(),
                    tp_split: TpSplit::Auto,
                    parallel_dense: family.moe.parallel_dense,
                    router_norm: family.moe.router_norm.clone(),
                    router_norm_scale: match &family.moe.router_norm_scale {
                        Some(expr) => crate::family::ParamSource::Expr { expr: expr.clone() }.read("router_norm_scale", &value)?,
                        None => 1.0,
                    },
                    input_norm: family.moe.input_norm.clone(),
                    expert_scale: family.moe.expert_scale.clone(),
                    latent: family.moe.latent.as_ref().and_then(|l| l.read(&value)),
                    local: None,
                })
            }
            _ => None,
        };
        let spec = Spec {
            name: hf.model_type.clone(),
            min_engine: 0,
            repo: String::new(),
            quant,
            head_quant: None,
            act_quant: act,
            dims: Dims {
                hidden: hf.hidden_size,
                intermediate: family.mlp.dense_intermediate_key.as_deref().and_then(|k| value.get(k)?.as_u64()).map(|n| n as usize).or(hf.intermediate_size).or_else(|| {
                    let m = moe.as_ref().filter(|_| strict)?;
                    Some(if m.shared_intermediate > 0 { m.shared_intermediate } else { m.expert_intermediate })
                }).unwrap_or(0),
                vocab: hf.vocab_size,
                tie_embeddings: hf.tie_word_embeddings.or(family.embedding.tie).unwrap_or(false),
                embed_scale: family
                    .embedding
                    .scale
                    .as_deref()
                    .and_then(|s| (s == "sqrt_hidden").then(|| (hf.hidden_size as f64).sqrt()))
                    .or(multiplier(&family.multipliers.embedding)),
                logit_softcap: hf.final_logit_softcapping,
                logit_scale: multiplier(&family.multipliers.logits_divisor).filter(|d| *d > 0.0).map(|d| 1.0 / d),
                head_rows: None,
            },
            layers: Layers { count: hf.num_hidden_layers, window: hf.sliding_window.filter(|_| pattern.iter().any(|b| b.windowed())), pattern, split: family.blocks.split, shared_tail: family.blocks.shared_tail_key.as_deref().and_then(|k| value.get(k)?.as_u64()).unwrap_or(0) as usize },
            norm: Norm {
                kind: NormKind::Rmsnorm,
                eps: hf.rms_norm_eps,
                unit_offset: family.norm.unit_offset.unwrap_or(hybrid && !family.norm.no_offset_keys.iter().any(|k| k.split('.').try_fold(&value, |v, p| v.get(p)).is_some())),
                sandwich: family.norm.sandwich,
                pre: family.norm.pre,
                residual_scale: multiplier(&family.multipliers.residual),
                layer_scale: family.norm.layer_scale.clone().filter(|t| !t.is_empty()),
            },
            attention: Attention {
                kind: crate::family::first_kind(&family.attention.kinds, &value).unwrap_or(AttentionKind::Gqa),
                heads: hf.num_attention_heads,
                kv_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
                head_dim,
                qkv_bias,
                o_bias,
                qk_norm,
                qk_norm_unit: qk_norm && family.attention.qk_norm_unit,
                output_gate,
                rotary_dim,
                softcap: hf.attn_logit_softcapping,
                query_scalar: hf.query_pre_attn_scalar.or(multiplier(&family.multipliers.attention).filter(|m| *m > 0.0).map(|m| 1.0 / (m * m))),
                qk_norm_full: qk_norm && family.attention.qk_norm_full,
                fused_qkv: family.attention.fused_qkv,
                node: family.attention.node.clone(),
                params: crate::family::read_params(&family.attention.params, &value)?,
                projection: family.attention.projection.clone(),
                o_cols: family.attention.o_cols.clone(),
                o_cols_by: family.attention.o_cols_by.clone(),
                scratch: family.attention.scratch.clone(),
                int_scratch: family.attention.int_scratch.clone(),
                o_tensor: family.attention.o_tensor.clone(),
                indexer: family.attention.indexer.as_ref().and_then(|i| i.read(&value)),
            },
            rope: Rope { style: if family.rope.interleaved { RopeStyle::Pairs } else { RopeStyle::Neox }, theta, scaling, mrope: parse_mrope(params), local_theta, nope_layers: nope_layers(&family.rope, &value, hf.num_hidden_layers) },
            mlp: Mlp {
                activation,
                quant: mlp_quant.map(|m| m.0),
                quant_until: mlp_quant.and_then(|m| m.1),
                act_quant: mlp_act,
                up_act_planes: if block_act { family.moe.act_planes.unwrap_or(1) } else { 1 },
                down_act_planes: if block_act { family.moe.act_planes.unwrap_or(1) } else { 1 },
                fused_gate_up: family.mlp.fused_gate_up,
                gated: match &family.mlp.gated {
                    Some(g) if g.applies(hf.hidden_act.as_deref()) => Some(GatedAct { expr: g.expr.clone(), params: crate::family::read_params(&g.params, &value)? }),
                    _ => None,
                },
                mapped: match &family.mlp.mapped {
                    Some(g) if activation == Activation::Mapped => Some(GatedAct { expr: g.expr.clone(), params: crate::family::read_params(&g.params, &value)? }),
                    _ => None,
                },
                wide_shared_tail: family.mlp.shared_tail_wide_key.as_deref().and_then(|k| value.get(k)?.as_bool()).unwrap_or(false),
                stored_gate_up: family.mlp.stored_gate_up,
            },
            moe,
            mtp: (family.mtp && (hybrid || family.mtp_layout.is_some()) && hf.mtp_num_hidden_layers > 0 && !refused.iter().any(|r| r.part == "mtp"))
                .then(|| Mtp { layout: family.mtp_layout.clone(), ..Mtp::new(hf.mtp_num_hidden_layers) }),
            linear_attention,
            mixer,
            hyper: family.hyper.read(&value),
            gated_residual: family.gated_residual.read(&value)?,
            prelude: family.prelude.read(&value)?,
            per_layer: family.per_layer.read(&value)?,
            block_residual: family.block_residual.read(&value),
            inputs: family.inputs.read(&outer)?,
            tensors: family.tensors.clone(),
            vision: if strict { crate::vision::Vision::from_hf_config(json)? } else { None },
            state: if family.state.is_empty() { crate::state::stock() } else { family.state.clone() },
            runtime: RuntimeOptions::default(),
            plain_activations: !family.runtime.split_activations,
            iq_quants: false,
            quant_map,
            refused,
        };
        Ok(spec)
    }

    pub fn split_activations(&self) -> bool {
        self.runtime.split_activations.unwrap_or(!self.plain_activations)
    }

    pub fn validate(&self) -> Result<()> {
        self.runtime.validate()?;
        ensure!(!self.layers.pattern.iter().any(|b| b.windowed()) || self.layers.window.is_some_and(|w| w > 0), "sliding layers need a sliding_window");
        ensure!(self.layers.split || !self.layers.pattern.contains(&Block::Mlp), "mlp layers need [layers] split = true");
        ensure!(self.norm.layer_scale.is_none() || (self.uses_branch() && !self.layers.split), "[norm] layer_scale needs sandwich or scaled residual norms");
        ensure!((self.mlp.activation == Activation::Mapped) == self.mlp.mapped.is_some(), "activation = \"mapped\" and [mlp.mapped] go together");
        ensure!(self.mlp.mapped.is_none() || self.mlp.gated.is_none(), "[mlp] cannot be both gated and mapped");
        ensure!(self.mlp.mapped.is_none() || self.mlp.quant.is_none(), "a non-gated ([mlp.mapped]) mlp runs on bf16, fp8 and q8 weights only");
        self.validate_attention_nodes()?;
        self.validate_mixer()?;
        self.validate_hyper()?;
        self.validate_prelude()?;
        self.validate_per_layer()?;
        self.validate_block_residual()?;
        self.validate_inputs()?;
        crate::state::validate(self)?;
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
            if let Some(l) = &m.layout {
                l.validate()?;
            }
            if let Some(h) = self.gated_residual() {
                let l = self.mtp_layout();
                ensure!(
                    l.fc_hidden.is_some() && l.mixer.is_some(),
                    "[mtp] on a [gated_residual] spec takes the {} stream hidden: its [mtp.layout] needs fc_hidden and mixer",
                    h.streams
                );
                ensure!(self.moe.is_some(), "[mtp] on a [gated_residual] spec runs the gated MoE block, so it needs [moe]");
            } else {
                let l = self.mtp_layout();
                ensure!(l.fc_hidden.is_none() && l.mixer.is_none(), "[mtp.layout] fc_hidden and mixer take the streams of a [gated_residual] spec");
            }
            ensure!(
                self.linear_attention.is_some() || m.layout.is_some(),
                "[mtp] on a spec without [linear_attention] needs a [mtp.layout] naming its tensors"
            );
        }
        match self.act_quant {
            ActQuant::None => {}
            ActQuant::Fp8 => ensure!(self.quant.is_fp8(), "act_quant = \"fp8\" needs fp8 weights (quant = \"fp8_block128\" or \"fp8_channel\")"),
            ActQuant::Fp4 => bail!("act_quant = \"fp4\" is for [mlp] only (act_quant at the top is \"none\" or \"fp8\")"),
        }
        match self.mlp.act_quant {
            ActQuant::None => {}
            ActQuant::Fp4 => ensure!(
                self.mlp.quant == Some(Quant::Nvfp4G16) || self.moe.as_ref().and_then(|m| m.expert_block).is_some_and(|b| b.native_mma()),
                "[mlp] act_quant = \"fp4\" needs [mlp] quant = \"nvfp4_g16\" or a 32 wide e2m1 e8m0 [moe] expert_block"
            ),
            ActQuant::Fp8 => ensure!(
                self.moe.is_some() && self.mlp.quant.unwrap_or(self.quant) == Quant::Fp8Block128,
                "[mlp] act_quant = \"fp8\" is for mixture-of-experts with fp8_block128 experts; dense FP8 MLP inputs follow the top-level act_quant"
            ),
        }
        ensure!(
            (1..=2).contains(&self.mlp.up_act_planes) && (1..=2).contains(&self.mlp.down_act_planes),
            "[mlp] up_act_planes and down_act_planes are 1 or 2"
        );
        ensure!(
            (self.mlp.up_act_planes == 1 && self.mlp.down_act_planes == 1) || self.moe.as_ref().and_then(|m| m.expert_block).is_some_and(|b| b.native_mma()),
            "[mlp] up_act_planes and down_act_planes = 2 need a 32 wide e2m1 e8m0 [moe] expert_block"
        );
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
            let value_dim = self.linear_attention.as_ref().map_or(0, |l| l.value_dim());
            ensure!(value_dim.is_multiple_of(crate::gguf::DEVICE_SUPERBLOCK), "quant = \"gguf\" needs the linear value dim to be a multiple of 256");
        }
        match (self.mlp.quant, self.mlp.quant_until) {
            (None, Some(_)) => bail!("[mlp] quant_until needs [mlp] quant"),
            (Some(m), until) => {
                let beside_fp8 = self.quant == Quant::Fp8Channel;
                let experts = self.quant == Quant::Bf16 && self.moe.is_some();
                ensure!(m == Quant::Nvfp4G16 && (beside_fp8 || experts), "[mlp] quant {m:?} beside {:?} is not supported (nvfp4_g16 beside fp8_channel is, and beside bf16 for mixture-of-experts)", self.quant);
                ensure!(until.is_none_or(|u| u > 0 && u < self.layers.count), "[mlp] quant_until must be inside the layer count");
                ensure!(
                    self.dense_fp4_fits() || (self.moe.as_ref().is_some_and(|m| !m.parallel_dense) && self.dims.hidden.is_multiple_of(256)),
                    "nvfp4_g16 needs hidden to be a multiple of 256 and intermediate a multiple of 256"
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
        ensure!(self.rope.theta.is_some(), "rope theta is required to run (theta is unknown)");
        let has_linear = self.layers.pattern.iter().any(|b| b.is_linear());
        match (&self.linear_attention, has_linear) {
            (None, true) if self.mixer.is_some() => {}
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

    pub fn head_rows(&self) -> std::ops::Range<usize> {
        self.dims.head_rows.clone().unwrap_or(0..self.dims.vocab)
    }

    pub fn tensor_quant(&self, tensor: &str) -> Option<crate::quant_map::TensorQuant> {
        self.quant_map.tensor(tensor)
    }

    pub fn lm_head_quant(&self) -> Quant {
        match self.runtime.requant.lm_head.or(self.head_quant) {
            Some(q) => q,
            None if self.quant == Quant::Gguf => Quant::Gguf,
            None if self.quant.quantised_checkpoint() => Quant::Bf16,
            None => self.quant,
        }
    }

    pub fn draft_head_quant(&self) -> Option<Quant> {
        let fallback = if self.runtime.gemm_fp32 { Quant::Bf16 } else { Quant::Nvfp4G16 };
        let q = self.speculative()?.head_quant.unwrap_or(fallback);
        let own_head = self.mtp.as_ref().is_some_and(|m| m.layout.as_ref().is_some_and(|l| l.head.is_some()));
        (own_head || q != self.lm_head_quant()).then_some(q)
    }

    pub fn mlp_quant(&self, layer: usize) -> Quant {
        let q = match self.mlp.quant {
            Some(Quant::Nvfp4G16) if !self.quant_map.is_empty() && self.quant_map.mlp_fp4(layer) != Some(true) => self.quant,
            Some(q) if self.mlp.quant_until.is_none_or(|u| layer < u) => q,
            _ => self.quant,
        };
        let dense = self.moe.as_ref().is_some_and(|m| !m.is_sparse(layer));
        if q == Quant::Nvfp4G16 && dense && !self.dense_fp4_fits() { self.quant } else { q }
    }

    pub fn dense_fp4_fits(&self) -> bool {
        let width = if self.moe.is_some() { 128 } else { 256 };
        self.dims.hidden.is_multiple_of(256) && self.dims.intermediate.is_multiple_of(width)
    }

    pub fn requanted(&self, linear: &str, stored: Quant) -> Quant {
        match self.runtime.requant.class(linear) {
            Some(Quant::Nvfp4G16) if stored.is_fp8() => Quant::Nvfp4G16,
            _ => stored,
        }
    }

    pub fn served_mlp_quant(&self, layer: usize) -> Quant {
        self.requanted("mlp_tail", self.mlp_quant(layer))
    }

    pub fn requanted_linears(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = ["qkv", "o"].into_iter().filter(|l| self.requanted(l, self.quant) != self.quant).collect();
        if self.linear_attention.is_some() {
            out.extend(["lin_in", "lin_out"].into_iter().filter(|l| self.requanted(l, self.quant) != self.quant));
        }
        out
    }

    pub fn serves_fp4_mlp(&self) -> bool {
        (0..self.layers.count).any(|i| self.served_mlp_quant(i) == Quant::Nvfp4G16)
    }

    pub fn stage(&self, start: usize, end: usize) -> Spec {
        let mut s = self.clone();
        s.name = format!("{}[{start}..{end}]", self.name);
        s.layers.pattern = (start..end).map(|i| self.block(i)).collect();
        s.layers.count = end - start;
        s.mlp.quant_until = self.mlp.quant_until.map(|u| u.saturating_sub(start));
        s.quant_map = self.quant_map.stage(start, end);
        s.rope.nope_layers = (start..end).filter(|&l| self.nope(l)).map(|l| l - start).collect();
        if let (Some(m), Some(whole)) = (s.moe.as_mut(), self.moe.as_ref()) {
            m.dense_layers = (start..end).filter(|&l| !whole.is_sparse(l)).map(|l| l - start).collect();
            m.sparse_step = 1;
        }
        if let Some(p) = s.prelude.as_mut() {
            p.layers = p.layers.iter().filter(|&&l| (start..end).contains(&l)).map(|l| l - start).collect();
        }
        if s.prelude.as_ref().is_some_and(|p| p.layers.is_empty()) {
            s.prelude = None;
        }
        if !s.layers.pattern.iter().any(|b| b.is_linear()) {
            s.linear_attention = None;
        }
        if !s.layers.pattern.iter().any(|b| b.is_attention()) {
            s.attention.node.clear();
            s.attention.projection.clear();
            s.attention.o_cols = None;
        }
        s.mtp = None;
        if self.runtime.kv_dtype.is_fp8() {
            let attention = |range: std::ops::Range<usize>| range.filter(|&i| self.block(i).is_attention()).count();
            s.runtime.kv_f16_layers = Some(self.kv_f16_layers().saturating_sub(attention(0..start)).min(attention(start..end)));
        }
        s
    }

    pub fn uses_branch(&self) -> bool {
        self.norm.sandwich || self.norm.residual_scale.is_some()
    }

    pub fn gated_residual(&self) -> Option<&GatedResidual> {
        self.gated_residual.as_ref()
    }

    pub fn per_layer(&self) -> Option<&PerLayer> {
        self.per_layer.as_ref()
    }

    fn validate_inputs(&self) -> Result<()> {
        let Some(p) = &self.inputs else { return Ok(()) };
        ensure!(!p.streams.is_empty() && !p.node.is_empty(), "[inputs] needs streams and nodes");
        ensure!(p.node.len() <= 64, "[[inputs.node]] has more than 64 nodes");
        let builtin = |n: &str| matches!(n, "x" | "tokens");
        for (n, w) in p.buffers() {
            ensure!(w > 0 && !builtin(&n) && !p.scratch.contains_key(&n), "[[inputs.streams]] buffer {n} needs a positive width and a name of its own");
        }
        for (n, w) in &p.scratch {
            ensure!(!builtin(n), "[inputs.scratch] {n} shadows a built in buffer");
            crate::state::dim(self, w).with_context(|| format!("[inputs.scratch] {n}"))?;
        }
        for (i, n) in p.node.iter().enumerate() {
            ensure!(n.op == "input" && n.kernel.is_some(), "[[inputs.node]] {i} must be an input node with `kernel`");
            ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[inputs.node]] {i} needs a grid of one to three symbols");
            for g in n.grid.iter().filter(|g| *g != "rows") {
                crate::state::dim(self, g).with_context(|| format!("[[inputs.node]] {i} grid symbol {g:?}"))?;
            }
            for b in n.bind.values() {
                match b {
                    crate::graph::Bind::Weight { len, .. } => {
                        self.weight_len(len)?;
                    }
                    crate::graph::Bind::Buffer { buffer } => ensure!(
                        builtin(buffer) || p.width_of(buffer).is_some() || p.scratch.contains_key(buffer),
                        "[[inputs.node]] {i} binds buffer {buffer:?}, which is not x, tokens, a declared stream or mask, or an [inputs.scratch] entry"
                    ),
                    crate::graph::Bind::Number(_) => {}
                    crate::graph::Bind::Name(n) => bail!("[[inputs.node]] {i} binds the name {n:?}; input nodes take numbers"),
                    crate::graph::Bind::State { .. } => bail!("[[inputs.node]] {i} binds a state, which input nodes do not take"),
                }
            }
        }
        Ok(())
    }

    fn validate_per_layer(&self) -> Result<()> {
        let Some(p) = self.per_layer() else { return Ok(()) };
        ensure!(p.dim > 0, "[per_layer] needs a positive dim");
        ensure!(self.hyper.is_none() && self.mixer.is_none(), "[per_layer] cannot be combined with [hyper] or [mixer] yet");
        ensure!(self.speculative().is_none(), "[per_layer] cannot be combined with [mtp] yet");
        ensure!(p.node.len() <= 64, "[[per_layer.node]] has more than 64 nodes");
        for stage in ["head", "layer"] {
            ensure!(p.nodes(stage).next().is_some(), "[[per_layer.node]] has no {stage} node");
        }
        for (n, w) in &p.scratch {
            ensure!(!matches!(n.as_str(), "x" | "tokens"), "[per_layer.scratch] {n} shadows a built in buffer");
            crate::state::dim(self, w).with_context(|| format!("[per_layer.scratch] {n}"))?;
        }
        for (i, n) in p.node.iter().enumerate() {
            ensure!(matches!(n.op.as_str(), "head" | "layer") && n.kernel.is_some(), "[[per_layer.node]] {i} must be a head or layer node with `kernel`");
            ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[per_layer.node]] {i} needs a grid of one to three symbols");
            for g in n.grid.iter().filter(|g| *g != "rows") {
                crate::state::dim(self, g).with_context(|| format!("[[per_layer.node]] {i} grid symbol {g:?}"))?;
            }
            for b in n.bind.values() {
                match b {
                    crate::graph::Bind::Weight { len, .. } => {
                        self.weight_len(len)?;
                    }
                    crate::graph::Bind::Buffer { buffer } => ensure!(
                        matches!(buffer.as_str(), "x" | "tokens") || (buffer == "streams" && self.gated_residual().is_some()) || p.scratch.contains_key(buffer),
                        "[[per_layer.node]] {i} binds buffer {buffer:?}, which is not x, tokens, the gated streams or a [per_layer.scratch] entry"
                    ),
                    crate::graph::Bind::Name(n) => ensure!(n == "layer", "[[per_layer.node]] {i} binds the name {n:?}, which is not layer"),
                    crate::graph::Bind::Number(_) => {}
                    crate::graph::Bind::State { .. } => bail!("[[per_layer.node]] {i} binds a state, which per layer nodes do not take"),
                }
            }
        }
        for (w, _how) in &p.scratch {
            let _ = w;
        }
        Ok(())
    }

    fn validate_hyper(&self) -> Result<()> {
        let Some(h) = self.gated_residual() else { return Ok(()) };
        ensure!(h.streams > 1, "[gated_residual] needs more than one stream");
        ensure!(self.hyper.is_none(), "[gated_residual] and [hyper] are two different residuals, a spec takes one");
        ensure!(!h.module.is_empty() && !h.sites.is_empty() && h.sites.len() <= 2, "[gated_residual] needs a module and one or two sites");
        ensure!(!self.layers.split, "[gated_residual] takes no split layers");
        let tags = h.tags();
        if tags.is_empty() {
            ensure!(!self.uses_branch() && !self.norm.pre, "[gated_residual] replaces the block norms and the residual, so it takes no pre, sandwich, scaled or split layers");
        } else {
            ensure!(
                self.norm.sandwich && self.norm.pre && self.norm.residual_scale.is_none() && self.moe.is_none() && !self.attention.node.is_empty(),
                "a [gated_residual] with tagged nodes runs in the sandwich block: it needs sandwich and pre norms, described attention, no residual scale and no experts"
            );
            ensure!(h.collapses(), "a [gated_residual] with tagged nodes needs its own collapse nodes");
        }
        ensure!(!self.mlp.stored_gate_up || tags.contains(&"act"), "[mlp] stored_gate_up needs a [[gated_residual.node]] tagged act that turns gate_up into the activation planes");
        ensure!(tags.is_empty() || self.speculative().is_none(), "a [gated_residual] with tagged nodes cannot be combined with [mtp] yet");
        ensure!(self.mixer.is_none(), "[gated_residual] cannot be combined with [mixer] yet");
        ensure!(h.node.len() <= 64, "[[gated_residual.node]] has more than 64 nodes");
        let stages: &[GatedStage] = if tags.is_empty() { &[GatedStage::Expand, GatedStage::Pre, GatedStage::Post, GatedStage::Collapse] } else { &[GatedStage::Expand, GatedStage::Collapse] };
        for &stage in stages {
            ensure!(h.nodes(stage).next().is_some(), "[[gated_residual.node]] has no {:?} node", stage);
        }
        for (n, w) in &h.scratch {
            ensure!(!matches!(n.as_str(), "x" | "streams" | "gate_up" | "act_hi" | "act_lo"), "[gated_residual.scratch] {n} shadows a built in buffer");
            crate::state::dim(self, w).with_context(|| format!("[gated_residual.scratch] {n}"))?;
        }
        for (n, v) in &h.layer_params {
            ensure!(v.len() == self.layers.count, "[gated_residual.layer_params] {n} has {} values for {} layers", v.len(), self.layers.count);
        }
        for (i, n) in h.node.iter().enumerate() {
            ensure!(n.kernel.is_some(), "[[gated_residual.node]] {i} needs `kernel`");
            ensure!(
                matches!(n.op.as_str(), "expand" | "pre" | "post") || (!tags.is_empty() && (n.op == "collapse" || tags.contains(&n.op.as_str()))),
                "[[gated_residual.node]] {i} must be an expand, pre or post node, or a tag beside a collapse node"
            );
            ensure!(n.when.as_deref().is_none_or(|w| w == "site"), "[[gated_residual.node]] {i} when {:?} is not site", n.when);
            ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[gated_residual.node]] {i} needs a grid of one to three symbols");
            for g in n.grid.iter().filter(|g| *g != "rows") {
                crate::state::dim(self, g).with_context(|| format!("[[gated_residual.node]] {i} grid symbol {g:?}"))?;
            }
            for b in n.bind.values() {
                match b {
                    crate::graph::Bind::Weight { len, .. } => {
                        ensure!(n.op != "expand" || h.collapses(), "[[gated_residual.node]] {i} is an expand node, which reads no weight");
                        self.weight_len(len)?;
                    }
                    crate::graph::Bind::Buffer { buffer } => ensure!(
                        matches!(buffer.as_str(), "x" | "streams") || (!tags.is_empty() && matches!(buffer.as_str(), "gate_up" | "act_hi" | "act_lo")) || h.scratch.contains_key(buffer),
                        "[[gated_residual.node]] {i} binds buffer {buffer:?}, which is not x, streams or a [gated_residual.scratch] entry"
                    ),
                    crate::graph::Bind::Name(name) => ensure!(h.layer_params.contains_key(name), "[[gated_residual.node]] {i} binds {name:?}, which is not a [gated_residual.layer_params] entry"),
                    crate::graph::Bind::Number(_) => {}
                    crate::graph::Bind::State { .. } => bail!("[[gated_residual.node]] {i} binds a state, which gated nodes do not take"),
                }
            }
        }
        Ok(())
    }

    fn validate_prelude(&self) -> Result<()> {
        let Some(p) = &self.prelude else { return Ok(()) };
        ensure!(self.gated_residual().is_some() && self.mixer.is_none(), "[prelude] runs inside the [gated_residual] blocks and cannot be combined with [mixer] yet");
        ensure!(!p.module.is_empty() && !p.node.is_empty() && p.node.len() <= 64, "[prelude] needs a module and one to 64 nodes");
        ensure!(p.layers.len() == 1, "[prelude] runs on {} layers; its kernels fold the layer index in, so more than one layer needs one build per layer, which is not supported yet", p.layers.len());
        ensure!(p.layers.iter().all(|&l| l < self.layers.count), "[prelude] layers {:?} are not inside the model's {} layers", p.layers, self.layers.count);
        for (n, w) in p.scratch.iter().chain(&p.int_scratch) {
            ensure!(!Prelude::BUFFERS.contains(&n.as_str()), "[prelude] scratch {n} shadows a built in buffer");
            crate::state::dim(self, w).with_context(|| format!("[prelude] scratch {n}"))?;
        }
        ensure!(p.int_scratch.keys().all(|n| !p.scratch.contains_key(n)), "[prelude.int_scratch] repeats a [prelude.scratch] name");
        let weights = p.weights();
        for (prefix, w) in &p.stored {
            ensure!(weights.iter().any(|(n, _)| n == w), "[prelude.stored] {prefix} names {w:?}, which no [[prelude.node]] binds as a weight");
        }
        for (k, v) in &p.words {
            let prefix = k.strip_suffix("_dtype").filter(|pre| p.stored.contains_key(*pre));
            ensure!(prefix.is_some() && matches!(v.as_str(), "f32" | "bf16" | "fp8"), "[prelude.words] {k} = {v:?} is not the f32, bf16 or fp8 dtype of a [prelude.stored] weight");
        }
        for (i, n) in p.node.iter().enumerate() {
            ensure!(n.op == "kernel" && n.kernel.is_some(), "[[prelude.node]] {i} must be a kernel node with `kernel`");
            ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[prelude.node]] {i} needs a grid of one to three symbols");
            for g in n.grid.iter().filter(|g| !matches!(g.as_str(), "rows" | "seqs")) {
                crate::state::dim(self, g).with_context(|| format!("[[prelude.node]] {i} grid symbol {g:?}"))?;
            }
            for b in n.bind.values() {
                match b {
                    crate::graph::Bind::Weight { len, .. } => {
                        self.weight_len(len)?;
                    }
                    crate::graph::Bind::Buffer { buffer } => ensure!(
                        Prelude::BUFFERS.contains(&buffer.as_str()) || p.scratch.contains_key(buffer) || p.int_scratch.contains_key(buffer),
                        "[[prelude.node]] {i} binds buffer {buffer:?}, which is not x, streams, tokens or a [prelude] scratch entry"
                    ),
                    crate::graph::Bind::State { state } => ensure!(
                        crate::state::named(self, state).is_some_and(|s| s.per == crate::state::Per::Sequence && s.layers == crate::state::Layers::Prelude && s.dtype == crate::state::Dtype::F32),
                        "[[prelude.node]] {i} binds state {state:?}, which is not a per sequence f32 [[state]] on the prelude layers"
                    ),
                    crate::graph::Bind::Number(_) => {}
                    crate::graph::Bind::Name(name) => ensure!(p.params.contains_key(name), "[[prelude.node]] {i} binds {name:?}, which is not a [prelude.params] entry"),
                }
            }
        }
        Ok(())
    }

    pub fn prelude_slot(&self, layer: usize) -> Option<usize> {
        self.prelude.as_ref().and_then(|p| p.slot(layer))
    }

    fn validate_block_residual(&self) -> Result<()> {
        let Some(r) = &self.block_residual else { return Ok(()) };
        ensure!(r.size > 0 && r.sites.len() == 2 && !r.output.is_empty(), "[block_residual] needs a block size, two sites (before the mixer, before the mlp) and an output module");
        ensure!(r.blocks(self.layers.count) <= MAX_RESIDUAL_BLOCKS, "[block_residual] keeps {} blocks, more than the {MAX_RESIDUAL_BLOCKS} the kernel holds", r.blocks(self.layers.count));
        ensure!(self.hyper.is_none() && self.gated_residual.is_none() && !self.uses_branch() && !self.layers.split && self.norm.pre, "[block_residual] replaces the plain pre-norm residual and takes no hyper, gated, sandwich, scaled or split layers");
        ensure!(self.speculative().is_none(), "[block_residual] cannot be combined with [mtp] yet");
        ensure!(self.moe.as_ref().is_none_or(|m| m.latent.is_some() && !m.parallel_dense && !m.input_scaled && m.shared_gate == false), "[block_residual] runs mixture-of-experts layers with [moe] latent experts and an ungated shared expert only");
        Ok(())
    }

    pub fn parallel_moe(&self) -> bool {
        self.moe.as_ref().is_some_and(|m| m.parallel_dense)
    }

    pub fn weight_len(&self, symbol: &str) -> Result<usize> {
        crate::state::dim(self, symbol).with_context(|| format!("weight length {symbol:?}"))
    }

    fn validate_mixer(&self) -> Result<()> {
        let Some(m) = &self.mixer else { return Ok(()) };
        ensure!(self.linear_attention.is_none(), "[mixer] and [linear_attention] cannot both describe the linear layers");
        ensure!(self.linear_layers() > 0, "[mixer] is set but the pattern has no linear layer");
        ensure!(!m.module.is_empty() && !m.projection.is_empty() && !m.out_tensor.is_empty(), "[mixer] needs module, projection and out_tensor");
        for p in &m.projection {
            crate::state::dim(self, &p.rows).with_context(|| format!("[[mixer.projection]] {}", p.tensor))?;
        }
        ensure!(crate::state::dim(self, &m.out_cols).context("[mixer] out_cols")? > 0, "[mixer] out_cols is zero");
        for (n, w) in &m.scratch {
            ensure!(!matches!(n.as_str(), "mix" | "x"), "[mixer.scratch] {n} shadows a built in buffer");
            crate::state::dim(self, w).with_context(|| format!("[mixer.scratch] {n}"))?;
        }
        ensure!(!m.node.is_empty() && m.node.len() <= 256, "[[mixer.node]] needs one to 256 nodes");
        for (i, n) in m.node.iter().enumerate() {
            ensure!(n.op == "kernel" && n.kernel.is_some(), "[[mixer.node]] {i} must be a kernel node with `kernel`");
            ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[mixer.node]] {i} needs a grid of one to three symbols");
            for g in &n.grid {
                ensure!(matches!(g.as_str(), "rows" | "seqs") || g.parse::<u32>().is_ok() || m.params.contains_key(g), "[[mixer.node]] {i} grid symbol {g:?} is not rows, seqs, a number or a [mixer.params] entry");
            }
            for b in n.bind.values() {
                match b {
                    crate::graph::Bind::Weight { len, .. } => {
                        self.weight_len(len)?;
                    }
                    crate::graph::Bind::Buffer { buffer } => {
                        ensure!(buffer == "mix" || m.scratch.contains_key(buffer), "[[mixer.node]] {i} binds buffer {buffer:?}, which is not mix or a [mixer.scratch] entry")
                    }
                    crate::graph::Bind::State { state } => ensure!(
                        crate::state::named(self, state).is_some_and(|s| s.per == crate::state::Per::Sequence && s.layers == crate::state::Layers::Linear && s.dtype == crate::state::Dtype::F32),
                        "[[mixer.node]] {i} binds state {state:?}, which is not a per sequence f32 [[state]] on the linear layers"
                    ),
                    crate::graph::Bind::Number(_) => {}
                    crate::graph::Bind::Name(n) => ensure!(m.params.contains_key(n), "[[mixer.node]] {i} binds {n:?}, which is not a [mixer.params] entry"),
                }
            }
        }
        ensure!(self.speculative().is_none(), "[mixer] cannot be combined with [mtp] yet");
        ensure!(!self.norm.sandwich, "[mixer] layers with sandwich norms are not supported yet");
        Ok(())
    }

    fn validate_attention_nodes(&self) -> Result<()> {
        let nodes = &self.attention.node;
        let appended = self.attention.projection.iter().filter(|p| p.append).count();
        ensure!(appended == 0 || appended == self.attention.projection.len(), "[[attention.projection]] mixes appended projections with ones that replace q, k and v");
        ensure!(appended == 0 || !self.attention.fused_qkv, "[[attention.projection]] append needs separate q, k and v projections");
        ensure!(appended == 0 || !self.attention.qkv_bias, "[[attention.projection]] append does not cover a qkv bias");
        for p in &self.attention.projection {
            let rows = crate::state::dim(self, &p.rows).with_context(|| format!("[[attention.projection]] {}", p.tensor))?;
            for k in &p.layers {
                Block::parse(k).with_context(|| format!("[[attention.projection]] {}", p.tensor))?;
            }
            if let Some(s) = &p.slot {
                let width = self.attention.params.get(s).with_context(|| format!("[[attention.projection]] {} slot {s:?} is not an [attention.params] entry", p.tensor))?;
                ensure!(rows as f64 <= *width, "[[attention.projection]] {} has {rows} rows, more than its slot {s} ({width})", p.tensor);
            }
            if let Some(w) = &p.when {
                let name = w.strip_prefix('!').unwrap_or(w);
                ensure!(self.attention.params.contains_key(name), "[[attention.projection]] {} when {w:?} is not an [attention.params] entry", p.tensor);
            }
        }
        for o in &self.attention.o_cols_by {
            let cols = crate::state::dim(self, &o.cols).context("[[attention.o_cols_by]] cols")?;
            ensure!(cols <= self.o_dim(), "[[attention.o_cols_by]] {cols} columns are wider than o_cols ({})", self.o_dim());
            for k in &o.layers {
                Block::parse(k).context("[[attention.o_cols_by]] layers")?;
            }
        }
        if let Some(c) = &self.attention.o_cols {
            crate::state::dim(self, c).context("[attention] o_cols")?;
        }
        for (n, w) in self.attention.scratch.iter().chain(&self.attention.int_scratch) {
            crate::state::dim_at(self, w, crate::engine::DEFAULT_CTX).with_context(|| format!("[attention.scratch] {n}"))?;
        }
        ensure!(self.attention.int_scratch.keys().all(|n| !self.attention.scratch.contains_key(n)), "[attention.int_scratch] repeats an [attention.scratch] name");
        if nodes.is_empty() {
            ensure!(self.attention.projection.is_empty() && self.attention.o_cols.is_none(), "[[attention.projection]] and o_cols need [[attention.node]]");
            return Ok(());
        }
        ensure!(nodes.len() <= 256, "[[attention.node]] has more than 256 nodes");
        let mut launches = 0;
        for (i, n) in nodes.iter().enumerate() {
            match n.op.as_str() {
                "qk_rope" | "decode_rope" => {}
                "kernel" => {
                    launches += 1;
                    for l in &n.layers {
                        Block::parse(l).with_context(|| format!("[[attention.node]] {i} layers"))?;
                    }
                    ensure!(n.kernel.is_some(), "[[attention.node]] {i} is a kernel node without `kernel`");
                    ensure!(!n.grid.is_empty() && n.grid.len() <= 3, "[[attention.node]] {i} needs a grid of one to three symbols");
                    for g in &n.grid {
                        ensure!(
                            matches!(g.as_str(), "heads" | "kv_heads" | "rows") || g.parse::<u32>().is_ok() || self.attention.params.contains_key(g),
                            "[[attention.node]] {i} grid symbol {g:?} is not heads, kv_heads, rows, a number or an [attention.params] entry"
                        );
                    }
                    for b in n.bind.values() {
                        match b {
                            crate::graph::Bind::Weight { len, .. } => {
                                self.weight_len(len)?;
                            }
                            crate::graph::Bind::Buffer { buffer } => ensure!(
                                matches!(buffer.as_str(), "qkv" | "attn") || self.attention.scratch.contains_key(buffer) || self.attention.int_scratch.contains_key(buffer),
                                "[[attention.node]] {i} binds buffer {buffer:?}, which is not qkv, attn or an [attention.scratch] or [attention.int_scratch] entry"
                            ),
                            crate::graph::Bind::State { state } => ensure!(
                                crate::state::named(self, state).is_some_and(|s| s.positional()),
                                "[[attention.node]] {i} binds state {state:?}, which is not a per token [[state]]"
                            ),
                            crate::graph::Bind::Number(_) | crate::graph::Bind::Name(_) => {}
                        }
                    }
                }
                other => bail!("[[attention.node]] {i} has op {other:?}, not qk_rope, decode_rope or kernel"),
            }
        }
        ensure!(launches > 0, "[[attention.node]] declares no kernel node");
        Ok(())
    }

    pub fn nope(&self, layer: usize) -> bool {
        self.rope.nope_layers.contains(&layer)
    }

    pub fn theta(&self) -> f64 {
        self.rope.theta.unwrap_or(10_000.0)
    }

    pub fn block(&self, layer: usize) -> Block {
        self.layers.pattern[layer % self.layers.pattern.len()]
    }

    pub fn linear_layers(&self) -> usize {
        (0..self.layers.count).filter(|&i| self.block(i).is_linear()).count()
    }

    pub fn attention_layers(&self) -> usize {
        (0..self.layers.count).filter(|&i| self.owns_kv(i)).count()
    }

    pub fn attention_before(&self, layer: usize) -> usize {
        (0..layer).filter(|&i| self.owns_kv(i)).count()
    }

    pub fn owns_kv(&self, layer: usize) -> bool {
        self.block(layer).is_attention() && !self.shares_kv(layer)
    }

    pub fn kv_index(&self, layer: usize) -> usize {
        self.attention_before(self.kv_source(layer))
    }

    pub fn kind_index(&self, layer: usize) -> usize {
        self.kind_before(self.kv_source(layer))
    }

    pub fn max_intermediate(&self) -> usize {
        if self.mlp.wide_shared_tail && self.layers.shared_tail > 0 { 2 * self.dims.intermediate } else { self.dims.intermediate }
    }

    pub fn gate_up_rows_wide(&self) -> usize {
        if self.mlp.mapped.is_some() { 2 * self.dims.intermediate } else { 4 * self.dims.intermediate }
    }

    pub fn wide_layer(&self, layer: usize) -> bool {
        self.mlp.wide_shared_tail && self.shares_kv(layer)
    }

    pub fn shares_kv(&self, layer: usize) -> bool {
        layer >= self.layers.count.saturating_sub(self.layers.shared_tail)
    }

    pub fn kv_source(&self, layer: usize) -> usize {
        if !self.shares_kv(layer) {
            return layer;
        }
        let first = self.layers.count - self.layers.shared_tail;
        (0..first).rev().find(|&j| self.block(j) == self.block(layer)).unwrap_or(layer)
    }

    pub fn layer_intermediate(&self, layer: usize) -> usize {
        if self.mlp.wide_shared_tail && self.shares_kv(layer) { 2 * self.dims.intermediate } else { self.dims.intermediate }
    }

    pub fn kind_before(&self, layer: usize) -> usize {
        (0..layer).filter(|&i| self.block(i) == self.block(layer)).count()
    }

    pub fn gate_up_rows(&self) -> usize {
        if self.mlp.mapped.is_some() { self.dims.intermediate } else { 2 * self.dims.intermediate }
    }

    pub fn gate_up_out(&self, rows: usize) -> usize {
        if self.mlp.mapped.is_some() { rows } else { rows / 2 }
    }

    pub fn has_mlp(&self, layer: usize) -> bool {
        !self.layers.split || self.block(layer) == Block::Mlp
    }

    pub fn has_mixer(&self, layer: usize) -> bool {
        !self.layers.split || self.block(layer) != Block::Mlp
    }

    pub fn mtp_layout(&self) -> MtpLayout {
        self.mtp.as_ref().and_then(|m| m.layout.as_ref()).map_or_else(MtpLayout::stock, |l| l.at(self.layers.count))
    }

    pub fn speculative(&self) -> Option<&Mtp> {
        self.mtp.as_ref().filter(|m| m.speculate)
    }

    pub fn streamed_mtp(&self) -> bool {
        self.gated_residual().is_some() && self.mtp.as_ref().is_some_and(|m| m.layout.as_ref().is_some_and(|l| l.fc_hidden.is_some()))
    }

    pub fn draft_width(&self) -> usize {
        match self.gated_residual() {
            Some(h) if self.streamed_mtp() => h.width(self.dims.hidden),
            _ => self.dims.hidden,
        }
    }

    pub fn kv_f16_layers(&self) -> usize {
        if !self.runtime.kv_dtype.is_fp8() {
            return 0;
        }
        self.runtime.kv_f16_layers.unwrap_or(self.attention_layers() / 2).min(self.kv_layers())
    }

    pub fn default_kv_f16_layers(&self, context: usize) -> usize {
        if context > FP8_KV_VERIFIED_CONTEXT { (self.attention_layers() * 3).div_ceil(4) } else { self.attention_layers() / 2 }
    }

    pub fn for_kv_context(&self, context: usize) -> Spec {
        let mut s = self.clone();
        if s.runtime.kv_dtype.is_fp8() && s.runtime.kv_f16_layers.is_none() {
            s.runtime.kv_f16_layers = Some(s.default_kv_f16_layers(context));
        }
        s
    }

    pub fn kv_context_note(&self, context: usize) -> Option<(bool, String)> {
        if !self.runtime.kv_dtype.is_fp8() {
            return None;
        }
        let resolved = self.for_kv_context(context);
        let (raw, all) = (resolved.kv_f16_layers().min(self.attention_layers()), self.attention_layers());
        let size = crate::runtime::kv_block_bytes(&resolved, 1) as f64 / crate::runtime::kv_block_bytes(&Spec { runtime: RuntimeOptions { kv_dtype: KvDtype::F16, ..self.runtime.clone() }, ..self.clone() }, 1) as f64;
        let split = format!("{raw} of {all} attention layers keep f16 KV, {:.2}x the size of f16", size);
        if context <= FP8_KV_VERIFIED_CONTEXT {
            return Some((true, format!("fp8 KV: {split}.")));
        }
        let safe = self.default_kv_f16_layers(context);
        if raw < safe {
            return Some((false, format!("fp8 KV: {split}. Past {FP8_KV_VERIFIED_CONTEXT} tokens fewer than {safe} f16 layers drift from f16 and can derail long generations; leave f16 kv layers on auto or use f16 (docs/specrt-kv-quant.md).")));
        }
        let note = if context > FP8_KV_LONG_VERIFIED_CONTEXT { format!(", verified to {FP8_KV_LONG_VERIFIED_CONTEXT} tokens") } else { String::new() };
        Some((true, format!("fp8 KV past {FP8_KV_VERIFIED_CONTEXT} tokens: {split}{note}.")))
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
            (RopeScalingKind::Banded, _) => RopeConstants {
                kind: 2,
                factor: s.factor,
                low: s.low_freq_factor.unwrap_or(1.0),
                high: s.high_freq_factor.unwrap_or(1.0),
                original,
                mscale: 1.0,
            },
            (RopeScalingKind::Yarn, Some(y)) => {
                let (low, high) = yarn_correction_range(self.theta(), self.rotary_dim(), original, y);
                RopeConstants { kind: 3, factor: s.factor, low, high, original, mscale: y.attention_factor }
            }
            (RopeScalingKind::Longrope, _) => RopeConstants { kind: 4, factor: s.factor, low: 1.0, high: 1.0, original, mscale: s.longrope.as_ref().map_or(1.0, |l| l.attention_factor) },
            (RopeScalingKind::Yarn, None) => RopeConstants { kind: 0, factor: 1.0, low: 1.0, high: 1.0, original: 1.0, mscale: 1.0 },
        }
    }

    pub fn rope_freq_factors(&self, max_ctx: usize) -> Vec<f64> {
        let Some(l) = self.rope.scaling.as_ref().and_then(|s| s.longrope.as_ref().map(|l| (s, l))) else { return Vec::new() };
        let original = l.0.original_max_positions.unwrap_or(f64::MAX);
        if max_ctx as f64 > original { l.1.long_factor.clone() } else { l.1.short_factor.clone() }
    }

    pub fn rotary_dim(&self) -> usize {
        self.attention.rotary_dim.unwrap_or(self.attention.head_dim)
    }

    pub fn mix_in_rows(&self) -> usize {
        self.mixer.as_ref().map_or(0, |m| m.projection.iter().map(|p| crate::state::dim(self, &p.rows).unwrap_or(0)).sum())
    }

    pub fn mix_out_cols(&self) -> usize {
        self.mixer.as_ref().map_or(0, |m| crate::state::dim(self, &m.out_cols).unwrap_or(0))
    }

    pub fn qkv_rows(&self) -> usize {
        if self.attention.projection.iter().any(|p| !p.append) {
            return self.projection_groups().iter().map(|g| g.1).sum();
        }
        let gate = if self.attention.output_gate { self.q_dim() } else { 0 };
        self.q_dim() + 2 * self.kv_dim() + gate + self.appended_rows()
    }

    pub fn appended_rows(&self) -> usize {
        self.projection_groups().iter().filter(|g| g.0[0].append).map(|g| g.1).sum()
    }

    pub fn appended_offset(&self) -> usize {
        let gate = if self.attention.output_gate { self.q_dim() } else { 0 };
        self.q_dim() + 2 * self.kv_dim() + gate
    }

    pub fn projection_groups(&self) -> Vec<(Vec<&Projection>, usize)> {
        let mut out: Vec<(Vec<&Projection>, usize)> = Vec::new();
        for p in &self.attention.projection {
            match p.slot.as_deref() {
                Some(s) => match out.iter_mut().find(|(g, _)| g[0].slot.as_deref() == Some(s)) {
                    Some((g, _)) => g.push(p),
                    None => out.push((vec![p], self.attention.params.get(s).map_or(0, |v| *v as usize))),
                },
                None => out.push((vec![p], crate::state::dim(self, &p.rows).unwrap_or(0))),
            }
        }
        out
    }

    pub fn projection_live(&self, p: &Projection, block: Block, shared: bool) -> bool {
        p.present_in(block)
            && !(p.kv_owner && shared)
            && p.when.as_deref().is_none_or(|w| {
                let (name, want) = match w.strip_prefix('!') {
                    Some(n) => (n, false),
                    None => (w, true),
                };
                (self.attention.params.get(name).copied().unwrap_or(0.0) != 0.0) == want
            })
    }

    pub fn o_cols_in(&self, block: Block) -> usize {
        match self.attention.o_cols_by.iter().find(|o| o.layers.iter().any(|l| l == block.name())) {
            Some(o) => crate::state::dim(self, &o.cols).unwrap_or(0),
            None => self.o_dim(),
        }
    }


    pub fn o_dim(&self) -> usize {
        self.attention.o_cols.as_deref().map_or_else(|| self.q_dim(), |c| crate::state::dim(self, c).unwrap_or(0))
    }

    pub fn q_dim(&self) -> usize {
        self.attention.heads * self.attention.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.attention.kv_heads * self.attention.head_dim
    }
}
