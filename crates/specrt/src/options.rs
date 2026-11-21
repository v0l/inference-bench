use crate::loader::LoaderOptions;
use crate::serve::{ServeOptions, SpeculativeOptions};
use crate::spec::{BackendChoice, KvDtype, PrefixCacheOptions, Quant, RopeExtension, RuntimeOptions, TileChoice};
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
                v.insert("runtime.kv_dtype".into(), json!("fp8"));
                v.insert("serve.max_num_seqs".into(), json!(2));
                v.insert("serve.max_batched_tokens".into(), json!(2048));
                v.insert("runtime.prefill_chunk".into(), json!(512));
                v.insert("runtime.prefix_cache.enabled".into(), json!(false));
                v.insert("runtime.prefix_cache.max_snapshots".into(), json!(0));
                v.insert("runtime.vision.cache_mb".into(), json!(64));
                v.insert("runtime.vision.max_batch".into(), json!(1));
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
    #[serde(skip)]
    pub solved: Option<Box<crate::solve::Solved>>,
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
        if let Some(v) = self.solved.as_ref().and_then(|s| s.options.get(key)) {
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

    pub fn sparse_values(&self) -> BTreeMap<String, Value> {
        self.profile.values().into_iter().chain(self.overrides.clone()).collect()
    }

    pub fn apply(&self, opts: ServeOptions) -> Result<ServeOptions> {
        apply_sparse(&self.sparse_values(), opts)
    }

    pub fn apply_overrides(&self, opts: ServeOptions) -> Result<ServeOptions> {
        apply_sparse(&self.overrides, opts)
    }
}

fn validate(opts: &ServeOptions) -> Result<()> {
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
    anyhow::ensure!(opts.max_long_in_flight >= 1, "serve.max_long_in_flight (long prompts at once) must be at least 1");
    Ok(())
}

pub fn apply_sparse(sparse: &BTreeMap<String, Value>, opts: ServeOptions) -> Result<ServeOptions> {
    let mut values: BTreeMap<String, Value> = schema().iter().map(|f| (f.key.clone(), f.default.clone())).collect();
    values.extend(sparse.clone());
    apply_values(&values, sparse, opts)
}

fn apply_values(values: &BTreeMap<String, Value>, sparse: &BTreeMap<String, Value>, mut opts: ServeOptions) -> Result<ServeOptions> {
    {
        if let Some(v) = values.get("serve.max_batched_tokens").and_then(Value::as_u64) {
            opts.max_batched_tokens = v as usize;
        }
        if let Some(v) = values.get("serve.itl_target_ms").and_then(Value::as_u64) {
            opts.itl_target = Duration::from_millis(v);
        }
        if let Some(v) = values.get("serve.whole_prompt_tokens").and_then(Value::as_u64) {
            opts.whole_prompt_tokens = v as usize;
        }
        if let Some(v) = values.get("serve.min_prefill_tokens").and_then(Value::as_u64) {
            opts.min_prefill_tokens = v as usize;
        }
        if let Some(v) = values.get("serve.swap_space_gb").and_then(Value::as_f64) {
            opts.swap_space_gb = v;
        }
        if let Some(v) = values.get("serve.admit_lookahead").and_then(Value::as_u64) {
            opts.admit_lookahead = v as usize;
        }
        if let Some(v) = values.get("serve.prefill_burst_tokens").and_then(Value::as_u64) {
            opts.prefill_burst_tokens = v as usize;
        }
        if let Some(v) = values.get("serve.max_long_in_flight").and_then(Value::as_u64) {
            opts.max_long_in_flight = v as usize;
        }
        if let Some(v) = values.get("serve.tp").and_then(Value::as_u64) {
            opts.tp = v as usize;
        }
        let mut runtime = serde_json::to_value(RuntimeOptions::default())?;
        for (key, value) in values {
            if let Some(path) = key.strip_prefix("runtime.").filter(|_| !value.is_null()) {
                set_path(&mut runtime, path, value.clone());
            }
        }
        opts.runtime = Some(serde_json::from_value(runtime).context("runtime options")?);
        let mut spec = Map::new();
        for (k, v) in sparse {
            if let Some(rest) = k.strip_prefix("speculative.") {
                spec.insert(rest.to_string(), v.clone());
            }
        }
        opts.speculative = serde_json::from_value(Value::Object(spec)).context("speculative options")?;
        validate(&opts)?;
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
    Meta { key: "serve.tp", label: "tensor parallel", group: Group::Parallel, help: "GPUs one model is split across, by attention head and MLP column, with an allreduce after each half layer. One stream decodes faster when the GPUs have peer access. 1 keeps the model on one GPU.", choices: &[], range: (1, 8) },
    Meta { key: "serve.max_batched_tokens", label: "batched tokens", group: Group::Serving, help: "Token budget of one scheduler step. Larger prefills faster, smaller keeps decode steps short.", choices: &[], range: (64, 1 << 20) },
    Meta { key: "serve.itl_target_ms", label: "inter-token target", group: Group::Serving, help: "Milliseconds a decode stream may wait between tokens while prompts are being read.", choices: &[], range: (1, 10_000) },
    Meta { key: "serve.whole_prompt_tokens", label: "whole prompt limit", group: Group::Serving, help: "A new prompt up to this many tokens is read in a single step even while others decode. 0 keeps the inter-token target in charge.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "serve.min_prefill_tokens", label: "guaranteed prefill per step", group: Group::Serving, help: "Prompt tokens read every step even when decoders already fill the inter-token target. Keeps time to first token bounded under load. 0 leaves prefill to the target alone.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "serve.swap_space_gb", label: "swap space", group: Group::Serving, help: "Gigabytes of pinned host memory where a preempted request parks its KV cache and state, instead of recomputing its prompt. 0 always recomputes.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "serve.admit_lookahead", label: "admission lookahead", group: Group::Serving, help: "Tokens of generation a new request and each running one reserve KV room for before another request is admitted. Fewer preemptions, later admission. 0 reserves the prompt only.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "serve.prefill_burst_tokens", label: "long prompt burst", group: Group::Serving, help: "Prompt tokens read per step while a prompt over 2048 tokens is waiting or being read, even when decoders fill the inter-token target. Long prompts start sooner, running streams wait longer between tokens. 0 leaves long prompts to the guaranteed prefill.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "serve.max_long_in_flight", label: "long prompts at once", group: Group::Serving, help: "Prompts over 2048 tokens read at the same time. More starts long prompts together instead of one after another.", choices: &[], range: (1, 64) },
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
    Meta { key: "runtime.pdl", label: "programmatic launch", group: Group::Runtime, help: "Let each kernel of a decode graph start launching while the one before it finishes (Hopper and newer).", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fuse", label: "fused kernels", group: Group::Runtime, help: "Fuse neighbouring operations into single kernels.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.tile", label: "gemm tile", group: Group::Runtime, help: "Tile shape for matrix multiplies. Auto picks per shape.", choices: &["auto", "small", "big", "narrow"], range: (0, 0) },
    Meta { key: "runtime.big_tile", label: "big tile", group: Group::Runtime, help: "Explicit tile [m, n, k, stages] for the big choice. Auto uses the built in one.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fp4_tile", label: "fp4 tile", group: Group::Runtime, help: "Explicit tile [m, n, k, stages] for the NVFP4 matrix multiply. Auto uses the built in one.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fp8_tile", label: "fp8 tile", group: Group::Runtime, help: "Explicit tile [m, n, wm, wn] for the large FP8 prefill matrix multiply. Auto uses the built in one.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.fp4_stages", label: "fp4 stages", group: Group::Runtime, help: "Pipeline stages of the NVFP4 matrix multiply. Auto uses the built in count.", choices: &[], range: (1, 16) },
    Meta { key: "runtime.kv_f16_layers", label: "f16 kv layers", group: Group::Runtime, help: "With fp8 KV, how many of the first attention layers keep f16 KV. The early layers are the most sensitive. Auto keeps the first half up to 32k tokens of context and three quarters above it, where fewer f16 layers drift from f16.", choices: &[], range: (0, 256) },
    Meta { key: "runtime.prefill_chunk", label: "prefill chunk", group: Group::Runtime, help: "Tokens read per prefill pass. Auto sizes it from memory.", choices: &[], range: (16, 1 << 20) },
    Meta { key: "runtime.unsplit", label: "unsplit ops", group: Group::Runtime, help: "Names of fused operations to keep whole instead of splitting across the batch.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.split_activations", label: "split activations", group: Group::Runtime, help: "Run each bf16 matrix multiply on a high and a low half of the activations, which matches an fp32 reference to 1e-3 at the cost of prefill speed. Off runs plain bf16 activations like other engines. Auto follows the model family.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.chunked_scan", label: "chunked scan", group: Group::Runtime, help: "Read long prompts through recurrent mixers in tensor core chunks instead of one row at a time. Same results, faster prefill.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.gemm_fp32", label: "fp32 accumulate", group: Group::Runtime, help: "Accumulate matrix multiplies in fp32 on the plain path. Slower, for debugging.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.vision.precision", label: "vision precision", group: Group::Runtime, help: "Arithmetic of the image encoder. fast runs f16 tensor cores; exact splits every value into two halves and matches transformers in f32 at about twice the time.", choices: &["fast", "exact"], range: (0, 0) },
    Meta { key: "runtime.vision.cache_mb", label: "vision cache", group: Group::Runtime, help: "Host memory for encoded images, keyed by the image bytes, so a repeated image is not encoded again. 0 turns it off.", choices: &[], range: (0, 1 << 20) },
    Meta { key: "runtime.vision.max_batch", label: "vision batch", group: Group::Runtime, help: "Images encoded together in one pass of the image encoder.", choices: &[], range: (1, 64) },
    Meta { key: "runtime.vision.pan_and_scan", label: "pan and scan", group: Group::Runtime, help: "For tiled image models, also read crops of wide or tall images so small details survive the downscale. Costs one more tile of tokens per crop.", choices: &[], range: (0, 0) },
    Meta { key: "runtime.kv_dtype", label: "kv cache type", group: Group::Runtime, help: "Precision of the KV cache. fp8 stores one byte per value with a scale per head and token, halving KV memory and attention reads against f16.", choices: &["f16", "fp8"], range: (0, 0) },
    Meta { key: "runtime.rope_extension.type", label: "context extension", group: Group::Runtime, help: "How the context is stretched past what the model was trained on. YaRN changes short context numerics slightly, so leave it off unless you need the length.", choices: &["yarn"], range: (0, 0) },
    Meta { key: "runtime.rope_extension.factor", label: "extension factor", group: Group::Runtime, help: "Multiplies the model's own maximum context, so 4 takes 262144 tokens to 1048576. 1 or auto keeps the trained length.", choices: &[], range: (1, 16) },
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
    min_prefill_tokens: usize,
    swap_space_gb: f64,
    admit_lookahead: usize,
    prefill_burst_tokens: usize,
    max_long_in_flight: usize,
    tp: usize,
}

fn knobs(o: &ServeOptions) -> ServeKnobs {
    let ServeOptions { max_model_len, max_num_seqs, kv_cache_bytes, kv_cache_tokens: _, max_batched_tokens, itl_target, whole_prompt_tokens, min_prefill_tokens, swap_space_gb, admit_lookahead, prefill_burst_tokens, max_long_in_flight, runtime: _, speculative: _, placement: _, progress: _, watchdog: _, sessions: _, tp, grammar_cache: _, solved: _ } = o;
    ServeKnobs { max_model_len: *max_model_len, max_num_seqs: *max_num_seqs, kv_cache_bytes: *kv_cache_bytes, max_batched_tokens: *max_batched_tokens, itl_target_ms: itl_target.as_millis() as u64, whole_prompt_tokens: *whole_prompt_tokens, min_prefill_tokens: *min_prefill_tokens, swap_space_gb: *swap_space_gb, admit_lookahead: *admit_lookahead, prefill_burst_tokens: *prefill_burst_tokens, max_long_in_flight: *max_long_in_flight, tp: *tp }
}

fn populated_knobs() -> ServeKnobs {
    ServeKnobs { max_model_len: Some(1), max_num_seqs: 1, kv_cache_bytes: Some(1), max_batched_tokens: 1, itl_target_ms: 1, whole_prompt_tokens: 1, min_prefill_tokens: 1, swap_space_gb: 1.0, admit_lookahead: 1, prefill_burst_tokens: 1, max_long_in_flight: 1, tp: 1 }
}

fn populated_runtime() -> RuntimeOptions {
    RuntimeOptions {
        backend: BackendChoice::Cuda,
        graphs: true,
        pdl: true,
        fuse: true,
        tile: TileChoice::Big,
        big_tile: Some([1, 1, 1, 1]),
        fp4_tile: Some([1, 1, 1, 1]),
        fp8_tile: Some([1, 1, 1, 1]),
        fp4_stages: Some(1),
        prefill_chunk: Some(1),
        unsplit: vec!["x".into()],
        split_activations: Some(false),
        gemm_fp32: true,
        chunked_scan: false,
        kv_dtype: KvDtype::Fp8,
        kv_f16_layers: Some(1),
        rope_extension: Some(RopeExtension { kind: crate::spec::RopeExtensionKind::Yarn, factor: 4 }),
        cache_dir: Some("x".into()),
        prefix_cache: PrefixCacheOptions { enabled: true, snapshot_interval: 16, max_snapshots: 1, max_blocks: 1, async_snapshots: true },
        loader: LoaderOptions { enabled: true, direct: true, chunk_mb: 1, readers: 1, ahead: 1, workers: 1, lanes: 1, skip_cached: true, pinned: true, host_budget_mb: 64 },
        vision: crate::spec::VisionOptions { precision: crate::vision_encoder::VisionPrecision::Exact, cache_mb: 1, max_batch: 1, pan_and_scan: false },
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
    d.insert("runtime.pdl".into(), json!(false));
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
        assert_eq!(default_of("serve.min_prefill_tokens"), Some(json!(d.min_prefill_tokens)));
        assert_eq!(default_of("serve.swap_space_gb"), Some(json!(d.swap_space_gb)));
        assert_eq!(default_of("serve.admit_lookahead"), Some(json!(d.admit_lookahead)));
        assert_eq!(default_of("serve.prefill_burst_tokens"), Some(json!(0)));
        assert_eq!(default_of("serve.max_long_in_flight"), Some(json!(1)));
        let burst = EngineChoice { overrides: [("serve.prefill_burst_tokens".to_string(), json!(16384)), ("serve.max_long_in_flight".to_string(), json!(4))].into(), ..Default::default() };
        let o = burst.apply(ServeOptions::default()).unwrap();
        assert_eq!((o.prefill_burst_tokens, o.max_long_in_flight), (16384, 4));
        assert_eq!(default_of("serve.max_num_seqs"), Some(json!(d.max_num_seqs)));
        assert_eq!(default_of("serve.tp"), Some(json!(1)));
        let two = EngineChoice { overrides: [("serve.tp".to_string(), json!(2))].into(), ..Default::default() };
        assert_eq!(two.apply(ServeOptions::default()).unwrap().tp, 2);
        let rt = RuntimeOptions::default();
        assert_eq!(default_of("runtime.graphs"), Some(json!(rt.graphs)));
        assert_eq!(default_of("runtime.pdl"), Some(json!(rt.pdl)));
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
    fn low_memory_halves_the_kv_cache_shrinks_batches_and_drops_the_draft_head_and_snapshots() {
        let mut c = EngineChoice::default();
        c.select(Profile::LowMemory);
        assert!(!c.is_custom());
        let o = c.apply(ServeOptions::default()).unwrap();
        let rt = o.runtime.unwrap();
        assert_eq!(o.speculative.enabled, Some(false));
        assert!(!rt.prefix_cache.enabled);
        assert_eq!(rt.prefix_cache.max_snapshots, 0);
        assert_eq!(rt.loader.host_budget_mb, 512);
        assert_eq!(rt.kv_dtype, KvDtype::Fp8);
        assert_eq!(rt.prefill_chunk, Some(512));
        assert_eq!((rt.vision.cache_mb, rt.vision.max_batch), (64, 1));
        assert_eq!(o.max_batched_tokens, 2048);
        assert_eq!(c.value("serve.max_num_seqs"), json!(2));
        assert!(rt.graphs && rt.fuse);
        let max = EngineChoice::default();
        let changed = schema().iter().filter(|f| max.value(&f.key) != c.value(&f.key)).count();
        assert!(changed >= 9, "low memory changes only {changed} options");
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
