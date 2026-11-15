use crate::loader::LoaderOptions;
use crate::serve::{ServeOptions, SpeculativeOptions};
use crate::spec::{BackendChoice, PrefixCacheOptions, Quant, RuntimeOptions, TileChoice};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::time::Duration;

pub const DEFAULT_MAX_NUM_SEQS: usize = 8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    #[default]
    MaxPerformance,
    LowMemory,
}

impl Profile {
    pub const ALL: [Profile; 2] = [Profile::MaxPerformance, Profile::LowMemory];

    pub fn label(self) -> &'static str {
        match self {
            Profile::MaxPerformance => "max performance",
            Profile::LowMemory => "low memory",
        }
    }

    pub fn values(self) -> BTreeMap<String, Value> {
        let mut v = BTreeMap::new();
        match self {
            Profile::MaxPerformance => {
                v.insert("runtime.prefix_cache.enabled".into(), json!(true));
            }
            Profile::LowMemory => {
                v.insert("speculative.enabled".into(), json!(false));
                v.insert("runtime.prefix_cache.enabled".into(), json!(false));
                v.insert("runtime.prefix_cache.max_snapshots".into(), json!(0));
                v.insert("runtime.loader.host_budget_mb".into(), json!(512));
            }
        }
        v
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineChoice {
    pub profile: Profile,
    pub overrides: BTreeMap<String, Value>,
}

impl EngineChoice {
    pub fn is_custom(&self) -> bool {
        !self.overrides.is_empty()
    }

    pub fn select(&mut self, profile: Profile) {
        self.profile = profile;
        self.overrides.clear();
    }

    pub fn value(&self, key: &str) -> Value {
        if let Some(v) = self.overrides.get(key) {
            return v.clone();
        }
        if let Some(v) = self.profile.values().get(key) {
            return v.clone();
        }
        default_of(key).unwrap_or(Value::Null)
    }

    pub fn base(&self, key: &str) -> Value {
        self.profile.values().get(key).cloned().or_else(|| default_of(key)).unwrap_or(Value::Null)
    }

    pub fn is_default(&self, key: &str) -> bool {
        self.base(key) == self.value(key)
    }

    pub fn set(&mut self, key: &str, value: Value) {
        if self.base(key) == value {
            self.overrides.remove(key);
        } else {
            self.overrides.insert(key.to_string(), value);
        }
    }

    pub fn reset(&mut self, key: &str) {
        self.overrides.remove(key);
    }

    pub fn values(&self) -> BTreeMap<String, Value> {
        schema().iter().map(|f| (f.key.clone(), self.value(&f.key))).collect()
    }

    fn validate(&self, opts: &ServeOptions) -> Result<()> {
        let s = &opts.speculative;
        if let Some(k) = s.draft_tokens {
            anyhow::ensure!((1..=crate::spec::MAX_DRAFT_TOKENS).contains(&k), "speculative.draft_tokens (draft tokens) must be 1 to {}, not {k}", crate::spec::MAX_DRAFT_TOKENS);
        }
        if let Some(b) = s.max_batch {
            anyhow::ensure!(b >= 1, "speculative.max_batch (mtp batch limit) must be at least 1");
            anyhow::ensure!(b <= opts.max_num_seqs, "speculative.max_batch (mtp batch limit) {b} is above parallel requests {}", opts.max_num_seqs);
        }
        if let Some(r) = &opts.runtime {
            r.prefix_cache.validate()?;
            r.loader.validate()?;
        }
        anyhow::ensure!(opts.max_batched_tokens >= 64, "serve.max_batched_tokens (batched tokens) must be at least 64");
        Ok(())
    }

    fn sparse(&self, prefix: &str) -> Map<String, Value> {
        let mut out = Map::new();
        for (k, v) in self.profile.values().into_iter().chain(self.overrides.clone()) {
            if let Some(rest) = k.strip_prefix(prefix) {
                out.insert(rest.to_string(), v);
            }
        }
        out
    }

    pub fn apply(&self, mut opts: ServeOptions) -> Result<ServeOptions> {
        let values = self.values();
        if let Some(v) = values.get("serve.max_batched_tokens").and_then(Value::as_u64) {
            opts.max_batched_tokens = v as usize;
        }
        if let Some(v) = values.get("serve.itl_target_ms").and_then(Value::as_u64) {
            opts.itl_target = Duration::from_millis(v);
        }
        if let Some(v) = values.get("serve.whole_prompt_tokens").and_then(Value::as_u64) {
            opts.whole_prompt_tokens = v as usize;
        }
        let mut runtime = serde_json::to_value(RuntimeOptions::default())?;
        for (key, value) in &values {
            if let Some(path) = key.strip_prefix("runtime.").filter(|_| !value.is_null()) {
                set_path(&mut runtime, path, value.clone());
            }
        }
        opts.runtime = Some(serde_json::from_value(runtime).context("runtime options")?);
        let spec = Value::Object(self.sparse("speculative."));
        opts.speculative = serde_json::from_value(spec).context("speculative options")?;
        self.validate(&opts)?;
        Ok(opts)
    }
}

fn set_path(root: &mut Value, path: &str, value: Value) {
    let mut node = root;
    let parts: Vec<&str> = path.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        let Some(map) = node.as_object_mut() else { return };
        if i + 1 == parts.len() {
            map.insert(part.to_string(), value);
            return;
        }
        node = map.entry(part.to_string()).or_insert_with(|| Value::Object(Map::new()));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Basic,
    Parallel,
    Mtp,
    PrefixCache,
    Serving,
    Runtime,
    Loader,
}

impl Group {
    pub const ALL: [Group; 7] = [Group::Basic, Group::Parallel, Group::Mtp, Group::PrefixCache, Group::Serving, Group::Runtime, Group::Loader];

    pub fn label(self) -> &'static str {
        match self {
            Group::Basic => "basic",
            Group::Parallel => "parallel requests",
            Group::Mtp => "mtp",
            Group::PrefixCache => "prefix cache",
            Group::Serving => "serving",
            Group::Runtime => "runtime",
            Group::Loader => "loader",
        }
    }

    pub fn advanced(self) -> bool {
        self != Group::Basic
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Bool,
    Int { min: u64, max: u64 },
    Choice(&'static [&'static str]),
    Text,
    Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub key: String,
    pub label: &'static str,
    pub help: &'static str,
    pub group: Group,
    pub kind: Kind,
    pub optional: bool,
    pub default: Value,
}

impl Field {
    pub fn special(&self) -> bool {
        matches!(self.key.as_str(), "serve.max_model_len" | "serve.max_num_seqs" | "serve.kv_cache_bytes")
    }
}

struct Meta {
    key: &'static str,
    label: &'static str,
    group: Group,
    help: &'static str,
    choices: &'static [&'static str],
    range: (u64, u64),
}

const QUANTS: &[&str] = &["bf16", "fp8_channel", "nvfp4_g16"];

const META: &[Meta] = &[
    Meta { key: "serve.max_model_len", label: "context length", group: Group::Basic, help: "Tokens one conversation can hold. Bigger contexts keep more and cost more memory.", choices: &[], range: (1, u64::MAX) },
    Meta { key: "serve.kv_cache_bytes", label: "memory", group: Group::Basic, help: "GPU memory to use. What is left after the weights becomes the KV cache that all requests share.", choices: &[], range: (0, u64::MAX) },
    Meta { key: "serve.max_num_seqs", label: "parallel requests", group: Group::Parallel, help: "Requests decoded together. They share one KV cache, and a single request still runs at full speed.", choices: &[], range: (1, 64) },
    Meta { key: "serve.max_batched_tokens", label: "batched tokens", group: Group::Serving, help: "Token budget of one scheduler step. Larger prefills faster, smaller keeps decode steps short.", choices: &[], range: (64, 1 << 20) },
    Meta { key: "serve.itl_target_ms", label: "inter-token target", group: Group::Serving, help: "Milliseconds a decode stream may wait between tokens while prompts are being read.", choices: &[], range: (1, 10_000) },
    Meta { key: "serve.whole_prompt_tokens", label: "whole prompt limit", group: Group::Serving, help: "A new prompt up to this many tokens is read in a single step even while others decode. 0 keeps the inter-token target in charge.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "speculative.enabled", label: "mtp", group: Group::Mtp, help: "Draft tokens with the model's own MTP head and verify them in one step. Output is unchanged.", choices: &[], range: (0, 0) },
    Meta { key: "speculative.draft_tokens", label: "draft tokens", group: Group::Mtp, help: "Tokens drafted per step. Auto uses the model spec, 2 on the 27B.", choices: &[], range: (1, crate::spec::MAX_DRAFT_TOKENS as u64) },
    Meta { key: "speculative.max_batch", label: "mtp batch limit", group: Group::Mtp, help: "Largest batch that still drafts. Auto uses the model spec.", choices: &[], range: (1, 64) },
    Meta { key: "speculative.head_quant", label: "draft head", group: Group::Mtp, help: "Precision of the extra lm_head copy used for drafts. Auto is nvfp4_g16, bf16 makes no copy.", choices: QUANTS, range: (0, 0) },
    Meta { key: "runtime.prefix_cache.enabled", label: "prefix cache", group: Group::PrefixCache, help: "Reuse the KV of a prompt prefix seen before, so a long shared prompt is read once.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.prefix_cache.snapshot_interval", label: "snapshot interval", group: Group::PrefixCache, help: "Tokens between recurrent state snapshots. A multiple of the 16 token block.", choices: &[], range: (16, 1 << 20) },
    Meta { key: "runtime.prefix_cache.max_snapshots", label: "max snapshots", group: Group::PrefixCache, help: "Recurrent state snapshots kept in host memory. 0 keeps none.", choices: &[], range: (0, 4096) },
    Meta { key: "runtime.prefix_cache.max_blocks", label: "max blocks", group: Group::PrefixCache, help: "KV blocks the cache may hold. 0 means no limit beyond the pool.", choices: &[], range: (0, 1 << 24) },
    Meta { key: "runtime.prefix_cache.async_snapshots", label: "async snapshots", group: Group::PrefixCache, help: "Copy snapshots to the host on a side stream instead of stalling the step.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.backend", label: "backend", group: Group::Runtime, help: "Which compute backend runs the model. Auto picks CUDA when there is a card.", choices: &["auto", "cuda", "wgpu"], range: (0, 0) },
    Meta { key: "runtime.graphs", label: "cuda graphs", group: Group::Runtime, help: "Capture decode steps as CUDA graphs to cut launch overhead.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fuse", label: "fused kernels", group: Group::Runtime, help: "Fuse neighbouring operations into single kernels.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.tile", label: "gemm tile", group: Group::Runtime, help: "Tile shape for matrix multiplies. Auto picks per shape.", choices: &["auto", "small", "big", "narrow"], range: (0, 0) },
    Meta { key: "runtime.big_tile", label: "big tile", group: Group::Runtime, help: "Explicit tile [m, n, k, stages] for the big choice. Auto uses the built in one.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fp4_tile", label: "fp4 tile", group: Group::Runtime, help: "Explicit tile [m, n, k, stages] for the NVFP4 matrix multiply. Auto uses the built in one.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fp4_stages", label: "fp4 stages", group: Group::Runtime, help: "Pipeline stages of the NVFP4 matrix multiply. Auto uses the built in count.", choices: &[], range: (1, 16) },
    Meta { key: "runtime.prefill_chunk", label: "prefill chunk", group: Group::Runtime, help: "Tokens read per prefill pass. Auto sizes it from memory.", choices: &[], range: (16, 1 << 20) },
    Meta { key: "runtime.unsplit", label: "unsplit ops", group: Group::Runtime, help: "Names of fused operations to keep whole instead of splitting across the batch.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.gemm_fp32", label: "fp32 accumulate", group: Group::Runtime, help: "Accumulate matrix multiplies in fp32 on the plain path. Slower, for debugging.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.cache_dir", label: "kernel cache", group: Group::Runtime, help: "Where compiled kernels are kept. Auto uses the user cache folder.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.loader.enabled", label: "fast loader", group: Group::Loader, help: "Read weights with parallel readers straight to the card.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.loader.direct", label: "direct reads", group: Group::Loader, help: "Bypass the page cache with O_DIRECT reads.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.loader.chunk_mb", label: "chunk size", group: Group::Loader, help: "Megabytes per read.", choices: &[], range: (1, 1024) },
    Meta { key: "runtime.loader.readers", label: "readers", group: Group::Loader, help: "Parallel read threads.", choices: &[], range: (1, 64) },
    Meta { key: "runtime.loader.ahead", label: "read ahead", group: Group::Loader, help: "Chunks queued ahead of the copy.", choices: &[], range: (1, 64) },
    Meta { key: "runtime.loader.workers", label: "workers", group: Group::Loader, help: "Threads that convert and upload tensors.", choices: &[], range: (1, 64) },
    Meta { key: "runtime.loader.lanes", label: "lanes", group: Group::Loader, help: "Parallel upload streams.", choices: &[], range: (1, 16) },
    Meta { key: "runtime.loader.skip_cached", label: "skip cached files", group: Group::Loader, help: "Read files already in the page cache the plain way.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.loader.pinned", label: "pinned buffers", group: Group::Loader, help: "Use pinned host memory for staging.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.loader.host_budget_mb", label: "host budget", group: Group::Loader, help: "Megabytes of host memory the loader may use for staging.", choices: &[], range: (64, 1 << 20) },
];

#[derive(Serialize)]
struct ServeKnobs {
    max_model_len: Option<usize>,
    max_num_seqs: usize,
    kv_cache_bytes: Option<u64>,
    max_batched_tokens: usize,
    itl_target_ms: u64,
    whole_prompt_tokens: usize,
}

fn knobs(o: &ServeOptions) -> ServeKnobs {
    let ServeOptions { max_model_len, max_num_seqs, kv_cache_bytes, kv_cache_tokens: _, max_batched_tokens, itl_target, whole_prompt_tokens, runtime: _, speculative: _, placement: _, progress: _ } = o;
    ServeKnobs { max_model_len: *max_model_len, max_num_seqs: *max_num_seqs, kv_cache_bytes: *kv_cache_bytes, max_batched_tokens: *max_batched_tokens, itl_target_ms: itl_target.as_millis() as u64, whole_prompt_tokens: *whole_prompt_tokens }
}

fn populated_knobs() -> ServeKnobs {
    ServeKnobs { max_model_len: Some(1), max_num_seqs: 1, kv_cache_bytes: Some(1), max_batched_tokens: 1, itl_target_ms: 1, whole_prompt_tokens: 1 }
}

fn populated_runtime() -> RuntimeOptions {
    RuntimeOptions {
        backend: BackendChoice::Cuda,
        graphs: true,
        fuse: true,
        tile: TileChoice::Big,
        big_tile: Some([1, 1, 1, 1]),
        fp4_tile: Some([1, 1, 1, 1]),
        fp4_stages: Some(1),
        prefill_chunk: Some(1),
        unsplit: vec!["x".into()],
        gemm_fp32: true,
        cache_dir: Some("x".into()),
        prefix_cache: PrefixCacheOptions { enabled: true, snapshot_interval: 16, max_snapshots: 1, max_blocks: 1, async_snapshots: true },
        loader: LoaderOptions { enabled: true, direct: true, chunk_mb: 1, readers: 1, ahead: 1, workers: 1, lanes: 1, skip_cached: true, pinned: true, host_budget_mb: 64 },
    }
}

fn populated_speculative() -> SpeculativeOptions {
    SpeculativeOptions { enabled: Some(true), draft_tokens: Some(1), max_batch: Some(1), head_quant: Some(Quant::Bf16) }
}

fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
    match v {
        Value::Object(m) => {
            for (k, v) in m {
                flatten(&format!("{prefix}.{k}"), v, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn flat(prefix: &str, v: impl Serialize) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    flatten(prefix, &serde_json::to_value(v).expect("options serialise"), &mut out);
    out
}

fn defaults() -> BTreeMap<String, Value> {
    let mut d = flat("serve", knobs(&ServeOptions::default()));
    d.extend(flat("speculative", SpeculativeOptions::default()));
    d.extend(flat("runtime", RuntimeOptions::default()));
    d.extend(flat("runtime.prefix_cache", PrefixCacheOptions::default()));
    d.extend(flat("runtime.loader", LoaderOptions::default()));
    d.insert("speculative.enabled".into(), json!(true));
    d
}

fn populated() -> BTreeMap<String, Value> {
    let mut d = flat("serve", populated_knobs());
    d.extend(flat("speculative", populated_speculative()));
    d.extend(flat("runtime", populated_runtime()));
    d
}

pub fn default_of(key: &str) -> Option<Value> {
    schema().into_iter().find(|f| f.key == key).map(|f| f.default)
}

pub fn declared_keys() -> Vec<String> {
    populated().into_keys().collect()
}

pub fn schema() -> Vec<Field> {
    let defaults = defaults();
    let sample = populated();
    let mut fields = Vec::new();
    for (key, shown) in &sample {
        let default = defaults.get(key).cloned().unwrap_or_else(|| if key.ends_with(".unsplit") { json!([]) } else { Value::Null });
        let meta = META.iter().find(|m| m.key == key);
        let kind = match (meta, shown) {
            (Some(m), _) if !m.choices.is_empty() => Kind::Choice(m.choices),
            (_, Value::Bool(_)) => Kind::Bool,
            (Some(m), Value::Number(_)) => Kind::Int { min: m.range.0, max: m.range.1 },
            (None, Value::Number(_)) => Kind::Int { min: 0, max: u64::MAX },
            (_, Value::Array(a)) if a.iter().all(Value::is_number) => Kind::Json,
            (_, Value::Array(_)) => Kind::Json,
            _ => Kind::Text,
        };
        fields.push(Field {
            key: key.clone(),
            label: meta.map_or("", |m| m.label),
            help: meta.map_or("", |m| m.help),
            group: meta.map_or(Group::Runtime, |m| m.group),
            kind,
            optional: default.is_null(),
            default,
        });
    }
    fields.sort_by_key(|f| (Group::ALL.iter().position(|g| *g == f.group), META.iter().position(|m| m.key == f.key)));
    fields
}

pub fn undocumented() -> Vec<String> {
    let keys = declared_keys();
    let mut out: Vec<String> = keys.iter().filter(|k| !META.iter().any(|m| m.key == k.as_str())).cloned().collect();
    out.extend(META.iter().filter(|m| !keys.iter().any(|k| k == m.key)).map(|m| format!("{} (no such option)", m.key)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_option_has_a_label_a_help_line_and_a_group() {
        assert_eq!(undocumented(), Vec::<String>::new());
        for f in schema() {
            assert!(!f.label.is_empty() && !f.help.is_empty(), "{}", f.key);
        }
    }

    #[test]
    fn schema_defaults_equal_the_engine_defaults() {
        let d = ServeOptions::default();
        let v = EngineChoice::default();
        assert_eq!(default_of("serve.max_batched_tokens"), Some(json!(d.max_batched_tokens)));
        assert_eq!(default_of("serve.itl_target_ms"), Some(json!(d.itl_target.as_millis() as u64)));
        assert_eq!(default_of("serve.whole_prompt_tokens"), Some(json!(d.whole_prompt_tokens)));
        assert_eq!(default_of("serve.max_num_seqs"), Some(json!(d.max_num_seqs)));
        let rt = RuntimeOptions::default();
        assert_eq!(default_of("runtime.graphs"), Some(json!(rt.graphs)));
        assert_eq!(default_of("runtime.loader.chunk_mb"), Some(json!(rt.loader.chunk_mb)));
        assert_eq!(default_of("runtime.prefix_cache.snapshot_interval"), Some(json!(rt.prefix_cache.snapshot_interval)));
        assert_eq!(default_of("runtime.big_tile"), Some(Value::Null));
        assert_eq!(default_of("speculative.draft_tokens"), Some(Value::Null));
        assert!(v.is_default("runtime.fuse"));
    }

    #[test]
    fn max_performance_turns_everything_on() {
        let o = EngineChoice::default().apply(ServeOptions::default()).unwrap();
        let rt = o.runtime.unwrap();
        assert!(rt.graphs && rt.fuse && rt.loader.enabled && rt.prefix_cache.enabled);
        assert_eq!(o.speculative, SpeculativeOptions::default());
    }

    #[test]
    fn low_memory_drops_the_draft_head_the_snapshots_and_the_loader_budget() {
        let mut c = EngineChoice::default();
        c.select(Profile::LowMemory);
        assert!(!c.is_custom());
        let o = c.apply(ServeOptions::default()).unwrap();
        let rt = o.runtime.unwrap();
        assert_eq!(o.speculative.enabled, Some(false));
        assert!(!rt.prefix_cache.enabled);
        assert_eq!(rt.prefix_cache.max_snapshots, 0);
        assert_eq!(rt.loader.host_budget_mb, 512);
        assert!(rt.graphs && rt.fuse);
    }

    #[test]
    fn a_changed_field_makes_the_choice_custom_and_reset_restores_the_default() {
        let mut c = EngineChoice::default();
        c.set("runtime.graphs", json!(false));
        assert!(c.is_custom() && !c.is_default("runtime.graphs"));
        c.reset("runtime.graphs");
        assert!(!c.is_custom() && c.is_default("runtime.graphs"));
        c.set("speculative.draft_tokens", json!(3));
        let o = c.apply(ServeOptions::default()).unwrap();
        assert_eq!(o.speculative.draft_tokens, Some(3));
    }

    #[test]
    fn overrides_reach_the_runtime_options() {
        let mut c = EngineChoice::default();
        c.set("runtime.loader.readers", json!(3));
        c.set("runtime.prefill_chunk", json!(512));
        c.set("runtime.tile", json!("big"));
        c.set("runtime.big_tile", json!([64, 64, 32, 4]));
        c.set("serve.max_batched_tokens", json!(2048));
        c.set("speculative.head_quant", json!("bf16"));
        let o = c.apply(ServeOptions::default()).unwrap();
        let rt = o.runtime.unwrap();
        assert_eq!((rt.loader.readers, rt.prefill_chunk, rt.tile, rt.big_tile), (3, Some(512), TileChoice::Big, Some([64, 64, 32, 4])));
        assert_eq!(o.max_batched_tokens, 2048);
        assert_eq!(o.speculative.head_quant, Some(Quant::Bf16));
    }

    #[test]
    fn an_invalid_value_is_refused_naming_the_option() {
        for (key, value, named) in [
            ("speculative.draft_tokens", json!(0), "draft_tokens"),
            ("speculative.draft_tokens", json!(9), "draft_tokens"),
            ("speculative.max_batch", json!(99), "max_batch"),
            ("runtime.prefix_cache.snapshot_interval", json!(100), "snapshot_interval"),
            ("runtime.loader.readers", json!(0), "readers"),
            ("serve.max_batched_tokens", json!(1), "max_batched_tokens"),
        ] {
            let mut c = EngineChoice::default();
            c.set(key, value);
            let e = c.apply(ServeOptions { max_num_seqs: 8, ..ServeOptions::default() }).unwrap_err();
            assert!(format!("{e:#}").contains(named), "{key}: {e:#}");
        }
    }

    #[test]
    fn a_bad_value_is_refused() {
        let mut c = EngineChoice::default();
        c.set("runtime.tile", json!("huge"));
        assert!(c.apply(ServeOptions::default()).is_err());
    }
}
