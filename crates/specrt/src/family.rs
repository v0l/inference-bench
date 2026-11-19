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
}

impl ParamSource {
    pub fn read(&self, name: &str, config: &serde_json::Value) -> Result<f64> {
        match self {
            ParamSource::Number(v) => Ok(*v),
            ParamSource::Config { config: key, default } => config
                .get(key)
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
    pub qkv_bias: Flag,
    pub qk_norm: Flag,
    pub output_gate: Flag,
    pub qk_norm_full: bool,
    pub fused_qkv: bool,
}

impl Default for FamilyAttention {
    fn default() -> Self {
        FamilyAttention {
            node: Vec::new(),
            params: Default::default(),
            qkv_bias: Flag::Config { config: "attention_bias".into(), default: false },
            qk_norm: Flag::Fixed(false),
            output_gate: Flag::Config { config: "attn_output_gate".into(), default: false },
            qk_norm_full: false,
            fused_qkv: false,
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
    pub sliding_period_key: Option<String>,
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
}

impl FamilyRope {
    pub fn key<'a>(&'a self, canonical: &'a str) -> &'a str {
        self.keys.get(canonical).map(String::as_str).unwrap_or(canonical)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule<T> {
    #[serde(alias = "quant", alias = "act")]
    pub gives: T,
    #[serde(default)]
    pub when: toml::Table,
    #[serde(default)]
    pub defaults: toml::Table,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyQuant {
    pub method_key: String,
    pub grouped_methods: Vec<String>,
    pub groups_key: String,
    pub all_linear_target: String,
    pub layer_marker: String,
    pub mlp_marker: String,
    pub layers_key: String,
    pub layer_prefixes: Vec<String>,
    pub projections: Vec<String>,
    pub skip_targets: Vec<String>,
    pub group_rules: Vec<Rule<crate::spec::Quant>>,
    pub flat_rules: Vec<Rule<crate::spec::Quant>>,
    pub act_rules: Vec<Rule<crate::spec::ActQuant>>,
    pub group_error: String,
    pub flat_error: String,
}

fn json_at<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    path.split('.').try_fold(v, |cur, key| cur.get(key))
}

fn value_matches(want: &toml::Value, got: &serde_json::Value) -> bool {
    match (want, got) {
        (toml::Value::Table(t), got) => t.get("any").and_then(|a| a.as_array()).is_some_and(|alts| alts.iter().any(|a| value_matches(a, got))),
        (toml::Value::String(a), serde_json::Value::String(b)) => a == b,
        (toml::Value::Boolean(a), serde_json::Value::Bool(b)) => a == b,
        (toml::Value::Integer(a), serde_json::Value::Number(b)) => b.as_i64() == Some(*a),
        (toml::Value::Float(a), serde_json::Value::Number(b)) => b.as_f64() == Some(*a),
        (toml::Value::Array(a), serde_json::Value::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_matches(x, y)),
        _ => false,
    }
}

impl<T: Copy> Rule<T> {
    pub fn matches(&self, json: &serde_json::Value) -> bool {
        self.when.iter().all(|(path, want)| match json_at(json, path) {
            Some(got) => value_matches(want, got),
            None => self.defaults.get(path).is_some_and(|d| value_matches(want, &toml_to_json(d))),
        })
    }
}

fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
}

pub fn first_match<T: Copy>(rules: &[Rule<T>], json: &serde_json::Value) -> Option<T> {
    rules.iter().find(|r| r.matches(json)).map(|r| r.gives)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FamilyGguf {
    pub keys: std::collections::BTreeMap<String, String>,
    pub tensors: std::collections::BTreeMap<String, String>,
    pub rope_scaling: std::collections::BTreeMap<String, crate::spec::RopeScalingKind>,
    pub unscaled: Vec<String>,
    pub pretokenizers: std::collections::BTreeMap<String, String>,
    pub names: std::collections::BTreeMap<String, String>,
    pub layer: std::collections::BTreeMap<String, String>,
    pub mtp: std::collections::BTreeMap<String, String>,
    pub permuted_qk: bool,
    pub v_heads: crate::spec::VHeadOrder,
    pub output_gate: bool,
    pub mrope_interleaved: bool,
    pub a_stored_negated: bool,
}

impl FamilyGguf {
    pub fn key(&self, canonical: &str, arch: &str) -> Result<String> {
        let template = self.keys.get(canonical).with_context(|| format!("the family has no GGUF key for {canonical}"))?;
        Ok(template.replace("{arch}", arch))
    }

    pub fn key_if_declared(&self, canonical: &str, arch: &str) -> Option<String> {
        self.keys.get(canonical).map(|t| t.replace("{arch}", arch))
    }

    pub fn with_default_names(&self, defaults: &FamilyGguf) -> FamilyGguf {
        let mut out = self.clone();
        for (k, v) in &defaults.names {
            out.names.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &defaults.layer {
            out.layer.entry(k.clone()).or_insert_with(|| v.clone());
        }
        out
    }

    pub fn hf_to_gguf(&self, hf: &str, main_layers: usize) -> Option<String> {
        if let Some(name) = self.names.get(hf) {
            return Some(name.clone());
        }
        if let Some(rest) = hf.strip_prefix("model.layers.") {
            let (layer, tail) = rest.split_once('.')?;
            return Some(format!("blk.{layer}.{}", self.layer.get(tail)?));
        }
        if let Some(rest) = hf.strip_prefix("mtp.layers.") {
            let (index, tail) = rest.split_once('.')?;
            let layer = main_layers + index.parse::<usize>().ok()?;
            return Some(format!("blk.{layer}.{}", self.layer.get(tail)?));
        }
        let tail = hf.strip_prefix("mtp.")?;
        Some(format!("blk.{main_layers}.{}", self.mtp.get(tail)?))
    }

    pub fn tensor(&self, canonical: &str) -> Result<&str> {
        self.tensors.get(canonical).map(String::as_str).with_context(|| format!("the family has no GGUF tensor name for {canonical}"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub min_engine: u32,
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default, rename = "match")]
    pub matches: Match,
    #[serde(default)]
    pub attention: FamilyAttention,
    #[serde(default)]
    pub blocks: FamilyBlocks,
    #[serde(default)]
    pub norm: FamilyNorm,
    #[serde(default)]
    pub multipliers: FamilyMultipliers,
    #[serde(default)]
    pub mlp: FamilyMlp,
    #[serde(default)]
    pub runtime: FamilyRuntime,
    #[serde(default)]
    pub embedding: FamilyEmbedding,
    #[serde(default)]
    pub vision: FamilyVision,
    #[serde(default)]
    pub mtp: bool,
    #[serde(default)]
    pub activation: FamilyActivation,
    #[serde(default)]
    pub dtypes: FamilyDtypes,
    #[serde(default)]
    pub rope: FamilyRope,
    #[serde(default)]
    pub quantization: FamilyQuant,
    #[serde(default)]
    pub gguf: FamilyGguf,
    #[serde(default)]
    pub fields: std::collections::BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub state: Vec<crate::state::State>,
}

impl Family {
    pub fn normalise(&self, config: &mut serde_json::Value) {
        for (canonical, alternatives) in &self.fields {
            if dotted(config, canonical).is_some() {
                continue;
            }
            if let Some(v) = alternatives.iter().find_map(|alt| dotted(config, alt)).cloned()
                && let Some(map) = config.as_object_mut()
            {
                insert_dotted(map, canonical, v);
            }
        }
    }
}

fn dotted<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    path.split('.').try_fold(value, |v, k| v.get(k))
}

fn insert_dotted(map: &mut serde_json::Map<String, serde_json::Value>, path: &str, value: serde_json::Value) {
    let (last, parents) = match path.rsplit_once('.') {
        Some((rest, last)) => (last, rest.split('.').collect::<Vec<_>>()),
        None => (path, Vec::new()),
    };
    let mut at = map;
    for parent in parents {
        at = at.entry(parent.to_string()).or_insert_with(|| serde_json::json!({})).as_object_mut().expect("a dotted field alias overwrote a non-object");
    }
    at.insert(last.to_string(), value);
}

static ACTIVE: std::sync::RwLock<Option<Registry>> = std::sync::RwLock::new(None);

pub fn install(registry: Registry) {
    *ACTIVE.write().unwrap() = Some(registry);
}

pub fn active() -> Registry {
    ACTIVE.read().unwrap().clone().unwrap_or_else(|| Registry::embedded().clone())
}

pub const FAMILY_KIND: &str = "family";

pub fn is_family(table: &toml::Table) -> bool {
    table.get("kind").and_then(|k| k.as_str()) == Some(FAMILY_KIND)
}

pub fn is_kernel(table: &toml::Table) -> bool {
    table.get("kind").and_then(|k| k.as_str()) == Some(kir::desc::KIND)
}

fn merge(base: &mut toml::Table, over: toml::Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registry {
    families: Vec<Family>,
}

impl Registry {
    pub fn from_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Result<Registry> {
        let mut tables: Vec<toml::Table> = Vec::new();
        for text in texts {
            let table: toml::Table = toml::from_str(text)?;
            if is_family(&table) {
                tables.push(table);
            }
        }
        let mut families = Vec::new();
        for table in &tables {
            let resolved = resolve(table, &tables, 0)?;
            families.push(toml::Value::Table(resolved).try_into::<Family>()?);
        }
        Ok(Registry { families })
    }

    pub fn embedded() -> &'static Registry {
        static EMBEDDED_FAMILIES: OnceLock<Registry> = OnceLock::new();
        EMBEDDED_FAMILIES.get_or_init(|| {
            Registry::from_texts(crate::spec::EMBEDDED.iter().map(|(_, t)| *t)).expect("the embedded family specs parse")
        })
    }

    pub fn families(&self) -> &[Family] {
        &self.families
    }

    pub fn for_model_type(&self, model_type: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.matches.model_types.iter().any(|m| m == model_type))
    }

    pub fn for_gguf(&self, arch: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.matches.gguf.iter().any(|m| m == arch))
    }

    pub fn named(&self, name: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.name == name)
    }

    pub fn gguf_architectures(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for arch in self.families.iter().flat_map(|f| f.matches.gguf.iter().map(String::as_str)) {
            if !out.contains(&arch) {
                out.push(arch);
            }
        }
        out
    }

    pub fn strict_for(&self, model_type: &str) -> Result<&Family> {
        match self.for_model_type(model_type) {
            Some(f) => Ok(f),
            None => bail!("model_type {model_type} is not supported"),
        }
    }

    pub fn describing(&self) -> Result<&Family> {
        self.named("generic").context("no generic family spec")
    }
}

fn resolve(table: &toml::Table, all: &[toml::Table], depth: usize) -> Result<toml::Table> {
    ensure!(depth < 8, "family specs extend each other in a loop");
    let Some(parent) = table.get("extends").and_then(|e| e.as_str()) else { return Ok(table.clone()) };
    let base = all
        .iter()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(parent))
        .with_context(|| format!("family extends {parent}, which does not exist"))?;
    let mut merged = resolve(base, all, depth + 1)?;
    if let Some(toml::Value::Table(m)) = merged.get_mut("match") {
        m.remove("gguf");
    }
    merge(&mut merged, table.clone());
    merged.remove("extends");
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "name = \"a\"\nkind = \"family\"\n[match]\nmodel_types = [\"a\", \"a_text\"]\n[attention]\nqk_norm = true\n";
    const CHILD: &str = "name = \"b\"\nkind = \"family\"\nextends = \"a\"\n[match]\nmodel_types = [\"b\"]\n[attention]\nqkv_bias = true\n";

    #[test]
    fn a_family_parses_with_defaults() {
        let r = Registry::from_texts([BASE]).unwrap();
        let f = r.for_model_type("a_text").unwrap();
        assert_eq!(f.attention.qk_norm, Flag::Fixed(true));
        assert_eq!(f.attention.qkv_bias, Flag::Config { config: "attention_bias".into(), default: false });
        assert_eq!(f.blocks.kind, BlockKind::Dense);
        assert!(r.for_model_type("zzz").is_none());
    }

    #[test]
    fn a_child_family_overrides_only_what_it_states() {
        let r = Registry::from_texts([BASE, CHILD]).unwrap();
        let b = r.for_model_type("b").unwrap();
        assert_eq!((b.attention.qk_norm.clone(), b.attention.qkv_bias.clone()), (Flag::Fixed(true), Flag::Fixed(true)));
        assert!(r.for_model_type("a").is_some());
    }

    #[test]
    fn a_missing_parent_is_an_error() {
        assert!(Registry::from_texts([CHILD]).is_err());
    }

    #[test]
    fn plain_model_specs_are_not_families() {
        let r = Registry::from_texts(["name = \"x\"\nrepo = \"y\"\n", BASE]).unwrap();
        assert_eq!(r.families().len(), 1);
    }

    #[test]
    fn flags_read_the_config_or_the_default() {
        let f = Flag::Config { config: "k".into(), default: true };
        assert!(f.read(&serde_json::json!({})));
        assert!(!f.read(&serde_json::json!({"k": false})));
        assert!(Flag::Fixed(true).read(&serde_json::json!({"k": false})));
    }
}
