use crate::fit;
use crate::hf::{self, Hub, RepoFile};
use crate::library::{self, Format};
use serde::{Deserialize, Serialize};
use specrt::anatomy::ModelAnatomy;
use specrt::placement::{CostModel, DeviceConfig};
use specrt::spec::{KvDtype, Spec};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub label: String,
    pub format: Format,
    pub files: Vec<RepoFile>,
    pub expected_parts: usize,
    pub size: u64,
    pub bits: f32,
}

impl Variant {
    pub fn complete(&self) -> bool {
        self.files.len() >= self.expected_parts.max(1)
    }

    pub fn missing_parts(&self) -> usize {
        self.expected_parts.saturating_sub(self.files.len())
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Grouped {
    pub variants: Vec<Variant>,
    pub companions: Vec<RepoFile>,
    pub mmproj: Option<RepoFile>,
}

fn shard_of(stem: &str) -> (usize, usize) {
    let Some(i) = stem.rfind("-of-") else { return (1, 1) };
    let Some(j) = stem[..i].rfind('-') else { return (1, 1) };
    match (stem[j + 1..i].parse::<usize>(), stem[i + 4..].parse::<usize>()) {
        (Ok(n), Ok(m)) => (n, m),
        _ => (1, 1),
    }
}

fn is_quant_token(t: &str) -> bool {
    let u = t.to_uppercase();
    let digits_after = |p: &str| u.strip_prefix(p).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
    digits_after("Q") || digits_after("IQ") || matches!(u.as_str(), "BF16" | "F16" | "F32" | "FP16") || u.starts_with("MXFP4")
}

pub fn quant_of_name(stem: &str) -> String {
    let base = match stem.rfind("-of-") {
        Some(i) => stem[..i].rsplit_once('-').map_or(stem, |(b, _)| b),
        None => stem,
    };
    base.split(['-', '.']).rev().find(|t| is_quant_token(t)).map(|t| t.to_uppercase()).unwrap_or_else(|| library::gguf_quant(None, base))
}

pub fn quant_bits(label: &str) -> f32 {
    let u = label.to_uppercase();
    match u.as_str() {
        "BF16" | "F16" | "FP16" => return 16.0,
        "F32" => return 32.0,
        _ => {}
    }
    if u.starts_with("MXFP4") {
        return 4.25;
    }
    let digits: String = u.trim_start_matches("UD-").trim_start_matches('I').trim_start_matches('Q').chars().take_while(|c| c.is_ascii_digit()).collect();
    let base: f32 = digits.parse().unwrap_or(0.0);
    let suffix = u.rsplit('_').next().unwrap_or("");
    base + match suffix {
        "XXS" => 0.0,
        "XS" => 0.1,
        "S" => 0.2,
        "M" | "NL" => 0.3,
        "L" => 0.4,
        "XL" => 0.5,
        _ => 0.25,
    }
}

fn safetensors_label(config: Option<&str>) -> String {
    let Some(v) = config.and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok()) else { return "safetensors".into() };
    let text = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    if let Some(q) = v.get("quantization_config").filter(|q| !q.is_null()) {
        let method = q["quant_method"].as_str().unwrap_or("quantised").to_uppercase();
        return match q["bits"].as_u64() {
            Some(b) => format!("{method} {b}-bit"),
            None => method,
        };
    }
    match text["torch_dtype"].as_str().or(text["dtype"].as_str()) {
        Some("bfloat16") => "BF16".into(),
        Some("float16") => "F16".into(),
        Some("float32") => "F32".into(),
        _ => "safetensors".into(),
    }
}

pub fn group(files: &[RepoFile], config: Option<&str>) -> Grouped {
    let mut out = Grouped { companions: hf::companion_files(files).into_iter().cloned().collect(), ..Grouped::default() };
    let mut gguf: BTreeMap<(String, String), Vec<(usize, usize, RepoFile)>> = BTreeMap::new();
    let mut safetensors = Vec::new();
    for f in files {
        let name = f.path.rsplit('/').next().unwrap_or(&f.path);
        if name.ends_with(".gguf") {
            if name.starts_with("mmproj") {
                if out.mmproj.as_ref().is_none_or(|m| f.size < m.size) {
                    out.mmproj = Some(f.clone());
                }
                continue;
            }
            let stem = name.trim_end_matches(".gguf");
            let (n, m) = shard_of(stem);
            let dir = f.path.rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default();
            gguf.entry((dir, library::shard_key(name).0)).or_default().push((n, m, f.clone()));
        } else if name.ends_with(".safetensors") {
            safetensors.push(f.clone());
        }
    }
    for ((_, stem), mut parts) in gguf {
        parts.sort_by_key(|p| p.0);
        let expected = parts.iter().map(|p| p.1).max().unwrap_or(1);
        let label = quant_of_name(&stem);
        let files: Vec<RepoFile> = parts.into_iter().map(|p| p.2).collect();
        out.variants.push(Variant { size: files.iter().map(|f| f.size).sum(), bits: quant_bits(&label), label, format: Format::Gguf, files, expected_parts: expected });
    }
    if out.variants.is_empty() && !safetensors.is_empty() {
        let expected = safetensors.len();
        out.variants.push(Variant { label: safetensors_label(config), format: Format::Safetensors, size: safetensors.iter().map(|f| f.size).sum(), bits: 0.0, files: safetensors, expected_parts: expected });
    }
    out.variants.sort_by(|a, b| a.size.cmp(&b.size).then(a.label.cmp(&b.label)));
    out
}

pub fn download_list(g: &Grouped, v: &Variant) -> Vec<RepoFile> {
    let mut all = g.companions.clone();
    if v.format == Format::Gguf {
        all.extend(g.mmproj.clone());
    }
    all.extend(v.files.clone());
    all
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kv {
    F16,
    MixedFp8,
    Fp8,
}

impl Kv {
    pub const ALL: [Kv; 3] = [Kv::F16, Kv::MixedFp8, Kv::Fp8];

    pub fn label(self) -> &'static str {
        match self {
            Kv::F16 => "f16 KV",
            Kv::MixedFp8 => "mixed fp8 KV",
            Kv::Fp8 => "fp8 KV",
        }
    }

    pub fn dtype(self) -> KvDtype {
        match self {
            Kv::F16 => KvDtype::F16,
            _ => KvDtype::Fp8,
        }
    }

    pub fn f16_layers(self, spec: &Spec) -> Option<usize> {
        match self {
            Kv::F16 => None,
            Kv::MixedFp8 => Some(spec.attention_layers() / 2),
            Kv::Fp8 => Some(0),
        }
    }

    pub fn bytes_per_token(self, spec: &Spec) -> u64 {
        let mut s = spec.clone();
        s.runtime.kv_dtype = self.dtype();
        s.runtime.kv_f16_layers = self.f16_layers(spec);
        (specrt::runtime::kv_block_bytes(&s, 1) / specrt::codegen::PAGE) as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    Tight,
    DoesNotFit,
    Unknown,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Fits => "fits",
            Verdict::Tight => "tight",
            Verdict::DoesNotFit => "does not fit",
            Verdict::Unknown => "fit unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvFit {
    pub kv: Kv,
    pub bytes_per_token: u64,
    pub context: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assessment {
    pub weights: u64,
    pub native: usize,
    pub kvs: Vec<KvFit>,
    pub chosen: KvFit,
    pub tier: u8,
    pub verdict: Verdict,
    pub need: u64,
    pub free: Option<u64>,
    pub gpus: usize,
}

impl Assessment {
    pub fn full(&self) -> bool {
        self.chosen.context >= self.native && self.verdict != Verdict::Unknown
    }
}

pub fn pool_devices(gpus: &[u64]) -> Vec<DeviceConfig> {
    gpus.iter()
        .enumerate()
        .map(|(i, free)| DeviceConfig {
            name: format!("gpu{i}"),
            memory_gib: (*free as f64 * fit::GPU_UTILIZATION) / fit::GB,
            bandwidth_gbs: 1000.0,
            tflops: 100.0,
            ordinal: Some(i),
            address: None,
            reserve_gib: Some(0.0),
            link: Default::default(),
            offload: None,
            host: None,
            peer: None,
            wgpu: false,
        })
        .collect()
}

pub fn place_on(spec: &Spec, weights: u64, kv_bytes_per_token: u64, context: usize, seqs: usize, devices: &[DeviceConfig]) -> anyhow::Result<specrt::placement::Placement> {
    let mut model = CostModel::new(spec, context, seqs.max(1));
    model.max_tp = 1;
    model.rescale_weights(weights);
    model.set_kv(kv_bytes_per_token, context);
    let devs: Vec<specrt::placement::Device> = devices.iter().map(specrt::placement::Device::from_config).collect();
    specrt::placement::plan(&model, &devs, specrt::placement::Objective::Latency)
}

fn pooled_context(spec: &Spec, anatomy: &ModelAnatomy, weights: u64, gpus: &[u64], seqs: usize, kv: Kv, native: usize) -> usize {
    let devices = pool_devices(gpus);
    let per_token = kv.bytes_per_token(spec);
    let fits = |ctx: usize| {
        let _ = anatomy;
        place_on(spec, weights, per_token, ctx, seqs, &devices).is_ok()
    };
    if !fits(fit::MIN_CTX.min(native)) {
        return 0;
    }
    let (mut lo, mut hi) = (fit::MIN_CTX.min(native), native);
    while lo < hi {
        let mid = (lo + (hi - lo).div_ceil(2)).min(hi);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

pub fn assess(spec: &Spec, base: &ModelAnatomy, weights: u64, gpus: &[u64], seqs: usize) -> Assessment {
    let mut a = base.clone();
    a.total_bytes = weights;
    let native = fit::max_context(&a);
    let free: Option<u64> = (!gpus.is_empty()).then(|| gpus.iter().sum());
    let usable = free.map(|f| (f as f64 * fit::GPU_UTILIZATION) as u64);
    let room = fit::room_bytes(&a, usable, seqs);
    let kvs: Vec<KvFit> = Kv::ALL
        .iter()
        .map(|&kv| {
            let bytes_per_token = kv.bytes_per_token(spec);
            a.kv_bytes_per_token = bytes_per_token;
            let context = match gpus.len() {
                0 => 0,
                1 => room.map_or(0, |r| fit::pool_tokens(&a, r).min(native)),
                _ => pooled_context(spec, base, weights, gpus, seqs, kv, native),
            };
            KvFit { kv, bytes_per_token, context }
        })
        .collect();
    let f16 = kvs[0].clone();
    let need = weights + a.state_bytes_per_seq * seqs.max(1) as u64 + kvs[0].bytes_per_token * fit::TYPICAL_REQUEST as u64;
    let (chosen, tier, verdict) = match free {
        None => (f16, 3, Verdict::Unknown),
        Some(_) => match kvs.iter().find(|k| k.context >= native) {
            Some(k) if k.kv == Kv::F16 => (k.clone(), 0, Verdict::Fits),
            Some(k) => (k.clone(), 1, Verdict::Fits),
            None => {
                let widest = kvs.iter().fold(kvs[0].clone(), |best, k| if k.context > best.context { k.clone() } else { best });
                let verdict = if widest.context >= fit::TYPICAL_REQUEST.min(native) { Verdict::Tight } else { Verdict::DoesNotFit };
                let tier = if verdict == Verdict::Tight { 2 } else { 3 };
                (widest, tier, verdict)
            }
        },
    };
    Assessment { weights, native, kvs, chosen, tier, verdict, need, free, gpus: gpus.len() }
}

pub fn context_label(tokens: usize) -> String {
    if tokens >= 1024 && tokens.is_multiple_of(1024) {
        format!("{}k", tokens / 1024)
    } else if tokens >= 1000 {
        format!("{:.0}k", tokens as f64 / 1024.0)
    } else {
        tokens.to_string()
    }
}

#[derive(Clone)]
pub struct Shape {
    pub spec: Spec,
    pub anatomy: ModelAnatomy,
}

#[derive(Clone)]
pub struct RepoInfo {
    pub repo: String,
    pub grouped: Grouped,
    pub shape: Result<Shape, String>,
    pub draft: Option<String>,
}

pub fn short_reason(text: &str) -> String {
    const MAX: usize = 190;
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= MAX {
        return text.to_string();
    }
    let tail: String = chars[chars.len() - 100..].iter().collect();
    let head: String = chars[..50].iter().collect();
    format!("{head} ... {tail}")
}

pub fn draft_of(config: &serde_json::Value) -> Option<String> {
    let text = config.get("text_config").filter(|t| t.is_object()).unwrap_or(config);
    let names_drafter = config["architectures"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str())
        .any(|a| ["Draft", "DFlash", "DSpark", "Eagle", "Medusa"].iter().any(|m| a.contains(m)));
    let target_layers = text["num_target_layers"].as_u64().or(config["num_target_layers"].as_u64());
    let has_target_ids = config.as_object().is_some_and(|o| o.values().any(|v| v.get("target_layer_ids").is_some())) || config.get("target_layer_ids").is_some();
    let named = ["target_model", "target_model_name_or_path"].iter().find_map(|k| config[*k].as_str().map(String::from));
    if !(names_drafter || target_layers.is_some() || has_target_ids || named.is_some()) {
        return None;
    }
    Some(match (named, target_layers) {
        (Some(n), _) => n,
        (None, Some(l)) => format!("a larger model with {l} layers"),
        (None, None) => "a larger model".into(),
    })
}

#[derive(Clone, Serialize, Deserialize)]
struct StoredGguf {
    spec: Spec,
    context: Option<usize>,
}

fn gguf_probe(hub: &Hub, repo: &str, g: &Grouped) -> Result<StoredGguf, String> {
    let mut probes: Vec<&Variant> = g.variants.iter().filter(|v| v.format == Format::Gguf && !v.files.is_empty()).collect();
    probes.sort_by_key(|v| !v.complete());
    if probes.is_empty() {
        return Err("no weights in this repo".into());
    }
    let mut last = String::new();
    for v in probes.into_iter().take(3) {
        let attempt = (|| {
            let header = hub.gguf_header(repo, &v.files[0].path)?;
            let mut spec = header.spec().map_err(|e| format!("{:#}", e.root_cause()))?;
            spec.name = repo.to_string();
            let arch = header.architecture().map_err(|e| e.to_string())?.to_string();
            Ok(StoredGguf { spec, context: header.count(&format!("{arch}.context_length")) })
        })();
        match attempt {
            Ok(stored) => return Ok(stored),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn gguf_shape(stored: StoredGguf) -> Shape {
    let anatomy = ModelAnatomy::new(&stored.spec, 0, stored.context);
    Shape { spec: stored.spec, anatomy }
}

pub fn config_shape_of(repo: &str, text: &str) -> (Result<Shape, String>, Option<String>) {
    let draft = serde_json::from_str::<serde_json::Value>(text).ok().and_then(|v| draft_of(&v));
    if let Some(target) = &draft {
        return (Err(format!("draft model for {target}, it only speeds up its target and cannot run alone")), draft);
    }
    let shape = (|| {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let support = library::specrt_support(&v);
        if !support.supported {
            if v["quantization_config"]["quant_method"].as_str() == Some("modelopt") {
                return Err("modelopt NVFP4 not supported yet".into());
            }
            return Err(short_reason(&support.reason));
        }
        let d = fit::describe_config_json(repo, text).map_err(|e| short_reason(&e))?;
        Ok(Shape { spec: d.spec, anatomy: d.anatomy })
    })();
    (shape, None)
}

const ENGINE: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Serialize, Deserialize)]
struct StoredInfo {
    engine: String,
    fetched: u64,
    sha: String,
    files: Vec<RepoFile>,
    config: Option<String>,
    gguf: Option<StoredGguf>,
}

fn info_from(repo: &str, s: &StoredInfo, gguf_err: Option<String>) -> Result<RepoInfo, String> {
    let has_gguf = s.files.iter().any(|f| f.path.ends_with(".gguf") && !f.path.contains("mmproj"));
    if has_gguf {
        let grouped = group(&s.files, None);
        let shape = match (&s.gguf, gguf_err) {
            (Some(g), _) => Ok(gguf_shape(g.clone())),
            (None, Some(e)) => Err(e),
            (None, None) => Err("no weights in this repo".into()),
        };
        return Ok(RepoInfo { repo: repo.to_string(), grouped, shape, draft: None });
    }
    if !s.files.iter().any(|f| f.path.ends_with(".safetensors")) {
        return Err("no GGUF or safetensors files".into());
    }
    let (shape, draft) = match &s.config {
        Some(text) => config_shape_of(repo, text),
        None => (Err("no config.json".into()), None),
    };
    Ok(RepoInfo { repo: repo.to_string(), grouped: group(&s.files, s.config.as_deref()), shape, draft })
}

fn fetch_info(hub: &Hub, repo: &str, sha: String, now: u64) -> (Result<RepoInfo, String>, Option<StoredInfo>) {
    let files = match hub.files(repo) {
        Ok(f) => f,
        Err(e) => return (Err(e), None),
    };
    let mut stored = StoredInfo { engine: ENGINE.into(), fetched: now, sha, files, config: None, gguf: None };
    let has_gguf = stored.files.iter().any(|f| f.path.ends_with(".gguf") && !f.path.contains("mmproj"));
    let mut transient = None;
    if has_gguf {
        match gguf_probe(hub, repo, &group(&stored.files, None)) {
            Ok(g) => stored.gguf = Some(g),
            Err(e) => transient = Some(e),
        }
    } else if stored.files.iter().any(|f| f.path.ends_with(".safetensors")) {
        match hub.text(repo, "config.json") {
            Ok(t) => stored.config = Some(t),
            Err(e) => transient = Some(e),
        }
    }
    let info = match (&transient, has_gguf) {
        (Some(e), false) => {
            let grouped = group(&stored.files, None);
            Ok(RepoInfo { repo: repo.to_string(), grouped, shape: Err(e.clone()), draft: None })
        }
        _ => info_from(repo, &stored, transient.clone()),
    };
    let keep = info.is_ok() && transient.is_none();
    (info, keep.then_some(stored))
}

pub fn inspect(hub: &Hub, repo: &str) -> Result<RepoInfo, String> {
    fetch_info(hub, repo, String::new(), 0).0
}

pub const INFO_TTL_SECS: u64 = 3600;

fn info_path(dir: &Path, repo: &str) -> PathBuf {
    dir.join(format!("{}.json", repo.replace('/', "__")))
}

fn save_info(path: &Path, stored: &StoredInfo) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, serde_json::to_string(stored).unwrap_or_default());
}

pub fn inspect_cached(hub: &Hub, repo: &str, dir: &Path, ttl: u64, now: u64) -> Result<RepoInfo, String> {
    let path = info_path(dir, repo);
    let stored: Option<StoredInfo> = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).filter(|s: &StoredInfo| s.engine == ENGINE);
    if let Some(mut s) = stored {
        if now.saturating_sub(s.fetched) < ttl {
            return info_from(repo, &s, None);
        }
        match hub.sha(repo) {
            Ok(sha) if !s.sha.is_empty() && sha == s.sha => {
                s.fetched = now;
                save_info(&path, &s);
                return info_from(repo, &s, None);
            }
            Err(_) if !s.sha.is_empty() => return info_from(repo, &s, None),
            _ => {}
        }
    }
    let sha = hub.sha(repo).unwrap_or_default();
    let (info, stored) = fetch_info(hub, repo, sha, now);
    if let Some(s) = stored {
        save_info(&path, &s);
    }
    info
}

pub fn inspect_each(hub: &Hub, repos: &[String], dir: &Path, on_result: &(dyn Fn(String, Result<RepoInfo, String>) + Sync)) {
    let now = hf::now_secs();
    std::thread::scope(|s| {
        for r in repos {
            s.spawn(move || on_result(r.clone(), inspect_cached(hub, r, dir, INFO_TTL_SECS, now)));
        }
    });
}

pub fn inspect_all(hub: &Hub, repos: &[String], dir: &Path) -> Vec<(String, Result<RepoInfo, String>)> {
    let out = std::sync::Mutex::new(Vec::new());
    inspect_each(hub, repos, dir, &|repo, info| out.lock().unwrap().push((repo, info)));
    out.into_inner().unwrap()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub variant: Variant,
    pub runnable: Result<(), String>,
    pub fit: Option<Assessment>,
}

impl Scored {
    pub fn key(&self) -> (u8, u8, std::cmp::Reverse<u32>, u64) {
        let tier = self.fit.as_ref().map_or(3, |f| f.tier);
        let rank = match (&self.runnable, self.variant.complete()) {
            (Ok(()), true) => 0,
            (Ok(()), false) => 1,
            (Err(_), _) => 2,
        };
        (rank, tier, std::cmp::Reverse((self.variant.bits * 100.0) as u32), self.variant.size)
    }

    pub fn downloadable(&self) -> bool {
        self.runnable.is_ok() && self.variant.complete()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepoPlan {
    pub repo: String,
    pub draft: Option<String>,
    pub unsupported: Option<String>,
    pub ranked: Vec<Scored>,
}

impl RepoPlan {
    pub fn best(&self) -> Option<&Scored> {
        self.ranked.first()
    }

    pub fn key(&self) -> (u8, u8) {
        match self.best() {
            Some(b) if self.unsupported.is_none() && self.draft.is_none() => {
                let (rank, tier, _, _) = b.key();
                (rank, tier)
            }
            _ => (3, 3),
        }
    }
}

pub fn plan(info: &RepoInfo, gpus: &[u64], seqs: usize) -> RepoPlan {
    let mut ranked: Vec<Scored> = info
        .grouped
        .variants
        .iter()
        .map(|v| {
            let quant_ok = v.format != Format::Gguf || specrt::gguf::quant_label_supported(&v.label);
            let runnable = match (&info.shape, quant_ok) {
                (Err(e), _) => Err(e.clone()),
                (Ok(_), false) => Err(format!("{} weights are not supported by specrt", v.label)),
                (Ok(_), true) => Ok(()),
            };
            let (fit, bits) = match &info.shape {
                Ok(s) => {
                    let bits = if v.format == Format::Gguf { v.bits } else { (v.size as f64 * 8.0 / s.anatomy.total_params.max(1) as f64) as f32 };
                    (Some(assess(&s.spec, &s.anatomy, v.size, gpus, seqs)), bits)
                }
                Err(_) => (None, v.bits),
            };
            Scored { variant: Variant { bits, ..v.clone() }, runnable, fit }
        })
        .collect();
    ranked.sort_by_key(|s| s.key());
    RepoPlan { repo: info.repo.clone(), draft: info.draft.clone(), unsupported: info.shape.as_ref().err().cloned().filter(|_| info.draft.is_none()), ranked }
}

pub fn rank_repos(mut plans: Vec<RepoPlan>) -> Vec<RepoPlan> {
    plans.sort_by_key(|p| p.key());
    plans
}

pub fn model_name(repo: &str, v: &Variant, first_file: &std::path::Path) -> String {
    match v.format {
        Format::Gguf => {
            let stem = first_file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let h = library::read_gguf_header(first_file);
            format!("{repo} {}", library::gguf_quant(h.and_then(|h| h.file_type), &library::shard_key(&stem).0))
        }
        _ => repo.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use specrt::gguf::writer::{Kv as W, build};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    const TREE_1017: &str = include_str!("../tests/fixtures/hf/tree-1017.json");
    const GB: u64 = 1 << 30;

    fn files_1017() -> Vec<RepoFile> {
        let all: Vec<RepoFile> = serde_json::from_str(TREE_1017).unwrap();
        all.into_iter().filter(|f| f.kind == "file").collect()
    }

    fn qwen3_header(context: u32) -> Vec<u8> {
        build(
            3,
            &[
                ("general.architecture", W::Str("qwen3")),
                ("qwen3.block_count", W::U32(36)),
                ("qwen3.embedding_length", W::U32(4096)),
                ("qwen3.feed_forward_length", W::U32(12288)),
                ("qwen3.attention.head_count", W::U32(32)),
                ("qwen3.attention.head_count_kv", W::U32(8)),
                ("qwen3.attention.key_length", W::U32(128)),
                ("qwen3.attention.layer_norm_rms_epsilon", W::F32(1e-6)),
                ("qwen3.context_length", W::U32(context)),
                ("tokenizer.ggml.tokens", W::Strs(&["a", "b", "c"])),
            ],
            &[],
        )
    }

    fn serve(routes: Vec<(String, Vec<u8>)>) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = log.clone();
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let routes = routes.clone();
                let seen = seen.clone();
                std::thread::spawn(move || {
                    let mut r = BufReader::new(s.try_clone().unwrap());
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_string();
                    let mut range: Option<(usize, Option<usize>)> = None;
                    loop {
                        line.clear();
                        r.read_line(&mut line).unwrap();
                        if line.trim().is_empty() {
                            break;
                        }
                        if let Some(v) = line.to_lowercase().strip_prefix("range: bytes=") {
                            let (a, b) = v.trim().split_once('-').unwrap();
                            range = Some((a.parse().unwrap(), b.parse().ok()));
                        }
                    }
                    seen.lock().unwrap().push(path.clone());
                    let found = routes.iter().find(|(p, _)| path.starts_with(p.as_str()));
                    let (head, body) = match (found, range) {
                        (None, _) => ("404 Not Found", Vec::new()),
                        (Some((_, b)), Some((from, to))) if from < b.len() => ("206 Partial Content", b[from..to.map_or(b.len(), |t| (t + 1).min(b.len()))].to_vec()),
                        (Some((_, b)), Some(_)) => ("416 Range Not Satisfiable", b[..0].to_vec()),
                        (Some((_, b)), None) => ("200 OK", b.clone()),
                    };
                    let _ = write!(s, "HTTP/1.1 {head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = s.write_all(&body);
                });
            }
        });
        (base, log)
    }

    #[test]
    fn the_1017_repo_groups_into_four_variants_with_merged_shards_and_a_vision_companion() {
        let g = group(&files_1017(), None);
        let shape: Vec<(&str, usize, usize)> = g.variants.iter().map(|v| (v.label.as_str(), v.files.len(), v.expected_parts)).collect();
        assert_eq!(shape.len(), 4, "{shape:?}");
        for want in [("IQ1_M", 5, 6), ("IQ2_XS", 5, 6), ("IQ2_XXS", 5, 6), ("IQ4_NL", 1, 1)] {
            assert!(shape.contains(&want), "{want:?} not in {shape:?}");
        }
        let iq1 = g.variants.iter().find(|v| v.label == "IQ1_M").unwrap();
        assert_eq!(iq1.size, 5780704 + 39340012896 + 38698805888 + 39707055008 + 2969539840);
        assert!(!iq1.complete() && iq1.missing_parts() == 1);
        assert_eq!(g.mmproj.as_ref().unwrap().path, "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf");
        assert!(g.companions.iter().any(|f| f.path == "chat_template.jinja"));
        assert!(g.variants.iter().all(|v| v.files.iter().all(|f| !f.path.starts_with("mmproj"))));
    }

    #[test]
    fn variant_labels_and_bit_order() {
        assert_eq!(quant_of_name("Qwen3-8B-Q4_K_M"), "Q4_K_M");
        assert_eq!(quant_of_name("model-UD-Q5_K_XL-00001-of-00003"), "Q5_K_XL");
        assert_eq!(quant_of_name("Foo-BF16"), "BF16");
        assert_eq!(quant_of_name("DeepSeek-V4.1-Flash-Engram-IQ4_NL"), "IQ4_NL");
        let order = ["IQ2_XXS", "IQ2_XS", "Q2_K", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"];
        assert!(order.windows(2).all(|w| quant_bits(w[0]) < quant_bits(w[1])), "{:?}", order.map(quant_bits));
    }

    #[test]
    fn a_safetensors_repo_is_one_variant_labelled_from_its_quantization_config() {
        let files = vec![RepoFile::new("model-00001-of-00002.safetensors", 10), RepoFile::new("model-00002-of-00002.safetensors", 5), RepoFile::new("config.json", 3)];
        let g = group(&files, Some(r#"{"quantization_config":{"quant_method":"fp8"}}"#));
        assert_eq!(g.variants.len(), 1);
        assert_eq!((g.variants[0].label.as_str(), g.variants[0].size, g.variants[0].format), ("FP8", 15, Format::Safetensors));
        let plain = group(&files, Some(r#"{"torch_dtype":"bfloat16"}"#));
        assert_eq!(plain.variants[0].label, "BF16");
    }

    #[test]
    fn the_gguf_header_is_read_with_range_requests_that_grow_until_it_parses() {
        let header = qwen3_header(131072);
        let (base, log) = serve(vec![("/acme/m/resolve/main/m-Q4_K_M.gguf".into(), header.clone())]);
        let hub = Hub::new(&base, "");
        let g = hub.gguf_header_from("acme/m", "m-Q4_K_M.gguf", 64, 1 << 20).unwrap();
        assert!(log.lock().unwrap().len() >= 2, "a 64 byte window cannot hold the header");
        assert_eq!(g.count("qwen3.context_length"), Some(131072));
        let spec = g.spec().unwrap();
        assert_eq!((spec.layers.count, spec.attention.kv_heads, spec.attention.head_dim, spec.dims.vocab), (36, 8, 128, 3));
        let e = hub.gguf_header_from("acme/m", "missing.gguf", 64, 1024).err().unwrap();
        assert!(e.contains("404"), "{e}");
    }

    fn cached_repo_routes(sha: &str) -> Vec<(String, Vec<u8>)> {
        vec![
            ("/api/models/acme/m/tree".into(), br#"[{"type":"file","path":"m-Q4_K_M.gguf","size":5000000000}]"#.to_vec()),
            ("/api/models/acme/m".into(), format!(r#"{{"sha":"{sha}"}}"#).into_bytes()),
            ("/acme/m/resolve/main/m-Q4_K_M.gguf".into(), qwen3_header(131072)),
        ]
    }

    #[test]
    fn a_cached_repo_is_not_refetched_while_fresh_or_while_its_revision_is_unchanged() {
        let dir = std::env::temp_dir().join(format!("ib-info-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (base, log) = serve(cached_repo_routes("rev1"));
        let hub = Hub::new(&base, "");
        let first = inspect_cached(&hub, "acme/m", &dir, 100, 1000).unwrap();
        assert!(first.shape.is_ok());
        let after_first = log.lock().unwrap().len();
        assert!(after_first >= 3);
        let fresh = inspect_cached(&hub, "acme/m", &dir, 100, 1050).unwrap();
        assert_eq!(log.lock().unwrap().len(), after_first, "fresh entry makes no request");
        assert_eq!(fresh.shape.unwrap().spec, first.shape.unwrap().spec);
        inspect_cached(&hub, "acme/m", &dir, 100, 1500).unwrap();
        assert_eq!(log.lock().unwrap().len(), after_first + 1, "stale entry only checks the revision");
        inspect_cached(&hub, "acme/m", &dir, 100, 1550).unwrap();
        assert_eq!(log.lock().unwrap().len(), after_first + 1, "the revision check renews the entry");
    }

    #[test]
    fn a_changed_revision_refetches_the_repo() {
        let dir = std::env::temp_dir().join(format!("ib-info-rev-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (base, log) = serve(cached_repo_routes("rev1"));
        inspect_cached(&Hub::new(&base, ""), "acme/m", &dir, 100, 1000).unwrap();
        let after_first = log.lock().unwrap().len();
        let (base2, log2) = serve(cached_repo_routes("rev2"));
        inspect_cached(&Hub::new(&base2, ""), "acme/m", &dir, 100, 2000).unwrap();
        let paths = log2.lock().unwrap().clone();
        assert!(paths.iter().any(|p| p.contains("/tree/")) && paths.iter().any(|p| p.contains("/resolve/")), "{paths:?} after {after_first}");
    }

    fn shape() -> Shape {
        let g = specrt::gguf::Gguf::from_header(qwen3_header(131072)).unwrap();
        let spec = g.spec().unwrap();
        let anatomy = ModelAnatomy::new(&spec, 0, Some(131072));
        Shape { spec, anatomy }
    }

    #[test]
    fn kv_bytes_per_token_follow_the_runtime_for_each_dtype() {
        let s = shape();
        let f16 = Kv::F16.bytes_per_token(&s.spec);
        assert_eq!(f16, 2 * 36 * 8 * 128 * 2);
        assert_eq!(f16, fit::kv_cost_at(&s.spec, KvDtype::F16, 1));
        let mixed = Kv::MixedFp8.bytes_per_token(&s.spec);
        let fp8 = Kv::Fp8.bytes_per_token(&s.spec);
        assert_eq!(mixed, fit::kv_cost_at(&s.spec, KvDtype::Fp8, 1));
        assert!(fp8 < mixed && mixed < f16, "{fp8} {mixed} {f16}");
    }

    #[test]
    fn context_per_dtype_is_what_the_free_memory_holds_after_weights_and_state() {
        let s = shape();
        let weights = 5 * GB;
        let free = 12 * GB;
        let a = assess(&s.spec, &s.anatomy, weights, &[free], 1);
        let mut base = s.anatomy.clone();
        base.total_bytes = weights;
        let room = fit::room_bytes(&base, Some((free as f64 * fit::GPU_UTILIZATION) as u64), 1).unwrap();
        for k in &a.kvs {
            let per = k.kv.bytes_per_token(&s.spec);
            assert_eq!(k.context, ((room / per) as usize).min(131072), "{:?}", k.kv);
        }
        assert!(a.kvs[0].context < 131072 && a.kvs[2].context > a.kvs[0].context);
        assert_eq!((a.verdict, a.tier), (Verdict::Tight, 2));
        assert_eq!(a.chosen.kv, Kv::Fp8);
    }

    #[test]
    fn plenty_of_room_gives_full_context_with_f16_and_less_room_falls_back_to_quantized_kv() {
        let s = shape();
        let roomy = assess(&s.spec, &s.anatomy, 5 * GB, &[80 * GB], 1);
        assert_eq!((roomy.chosen.kv, roomy.tier, roomy.verdict), (Kv::F16, 0, Verdict::Fits));
        assert!(roomy.full());
        let f16_need = Kv::F16.bytes_per_token(&s.spec) * 131072;
        let mixed_need = Kv::MixedFp8.bytes_per_token(&s.spec) * 131072;
        let free = (((5 * GB + mixed_need + GB) as f64) / fit::GPU_UTILIZATION) as u64;
        assert!(f16_need > mixed_need + GB);
        let tight = assess(&s.spec, &s.anatomy, 5 * GB, &[free], 1);
        assert_eq!((tight.chosen.kv, tight.tier, tight.verdict), (Kv::MixedFp8, 1, Verdict::Fits));
        assert!(tight.full());
    }

    #[test]
    fn place_on_agrees_with_assess_for_a_5080_and_two_3090s() {
        let s = shape();
        let cards = [16 * GB, 24 * GB, 24 * GB];
        let weights = 27 * GB;
        assert_eq!(assess(&s.spec, &s.anatomy, weights, &cards[..1], 1).verdict, Verdict::DoesNotFit);
        let pooled = assess(&s.spec, &s.anatomy, weights, &cards, 1);
        assert_ne!(pooled.verdict, Verdict::DoesNotFit, "{pooled:?}");
        let devices = pool_devices(&cards);
        let per_token = pooled.chosen.kv.bytes_per_token(&s.spec);
        let placed = place_on(&s.spec, weights, per_token, pooled.chosen.context, 1, &devices).unwrap();
        assert!(placed.stages.len() >= 2);
        assert!(place_on(&s.spec, weights, per_token, pooled.chosen.context, 1, &devices[..1]).is_err());
    }

    #[test]
    fn a_model_that_fits_only_across_two_gpus_is_planned_across_both() {
        let s = shape();
        let one = assess(&s.spec, &s.anatomy, 30 * GB, &[24 * GB], 1);
        assert_eq!((one.verdict, one.chosen.context), (Verdict::DoesNotFit, 0));
        let two = assess(&s.spec, &s.anatomy, 30 * GB, &[24 * GB, 24 * GB], 1);
        assert_eq!(two.gpus, 2);
        assert!(two.chosen.context >= fit::TYPICAL_REQUEST && two.verdict != Verdict::DoesNotFit, "{two:?}");
        assert_eq!(two.free, Some(48 * GB));
        let three = assess(&s.spec, &s.anatomy, 30 * GB, &[24 * GB, 24 * GB, 24 * GB], 1);
        assert!(three.kvs[0].context >= two.kvs[0].context);
        let roomy = assess(&s.spec, &s.anatomy, 5 * GB, &[40 * GB, 40 * GB], 1);
        assert_eq!((roomy.chosen.kv, roomy.tier), (Kv::F16, 0));
    }

    #[test]
    fn mixed_fp8_is_its_own_choice_between_f16_and_fp8() {
        let s = shape();
        let f16 = Kv::F16.bytes_per_token(&s.spec);
        let mixed = Kv::MixedFp8.bytes_per_token(&s.spec);
        let fp8 = Kv::Fp8.bytes_per_token(&s.spec);
        assert!(fp8 < mixed && mixed < f16);
        assert_eq!(Kv::MixedFp8.f16_layers(&s.spec), Some(18));
        assert_eq!(Kv::Fp8.f16_layers(&s.spec), Some(0));
        let need = |per: u64| 5 * GB + per * 131072 + GB / 2;
        let free = (need(mixed) as f64 / fit::GPU_UTILIZATION) as u64;
        let a = assess(&s.spec, &s.anatomy, 5 * GB, &[free], 1);
        assert_eq!((a.chosen.kv, a.chosen.kv.label(), a.tier), (Kv::MixedFp8, "mixed fp8 KV", 1));
    }

    #[test]
    fn nothing_fits_is_reported_with_what_the_smallest_variant_needs() {
        let s = shape();
        let a = assess(&s.spec, &s.anatomy, 30 * GB, &[8 * GB], 1);
        assert_eq!((a.verdict, a.tier, a.chosen.context), (Verdict::DoesNotFit, 3, 0));
        assert!(a.need > 30 * GB && a.free == Some(8 * GB));
        let unknown = assess(&s.spec, &s.anatomy, 5 * GB, &[], 1);
        assert_eq!(unknown.verdict, Verdict::Unknown);
    }

    fn info(variants: &[(&str, u64)]) -> RepoInfo {
        let files: Vec<RepoFile> = variants.iter().map(|(l, size)| RepoFile::new(&format!("m-{l}.gguf"), *size)).collect();
        RepoInfo { repo: "acme/m".into(), grouped: group(&files, None), shape: Ok(shape()), draft: None }
    }

    #[test]
    fn ranking_prefers_full_f16_then_full_quantized_then_widest_and_takes_the_highest_bits_within_a_tier() {
        let all = [("Q4_K_M", 5 * GB), ("Q5_K_M", 6 * GB), ("Q6_K", 7 * GB), ("Q8_0", 9 * GB)];
        let plan_big = plan(&info(&all), &[80 * GB], 1);
        assert_eq!(plan_big.best().unwrap().variant.label, "Q8_0");
        assert_eq!(plan_big.ranked.iter().map(|s| s.variant.label.as_str()).collect::<Vec<_>>(), ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]);
        let s = shape();
        let mixed_need = Kv::Fp8.bytes_per_token(&s.spec) * 131072;
        let free = (((9 * GB + mixed_need + GB / 4) as f64) / fit::GPU_UTILIZATION) as u64;
        let squeezed = plan(&info(&all), &[free], 1);
        let best = squeezed.best().unwrap();
        assert_eq!((best.variant.label.as_str(), best.fit.as_ref().unwrap().chosen.kv, best.fit.as_ref().unwrap().tier), ("Q8_0", Kv::Fp8, 1));
        let free = (((7 * GB + mixed_need + GB / 4) as f64) / fit::GPU_UTILIZATION) as u64;
        let mid = plan(&info(&all), &[free], 1);
        assert_eq!(mid.best().unwrap().variant.label, "Q6_K");
        let tiers: Vec<u8> = mid.ranked.iter().map(|s| s.fit.as_ref().unwrap().tier).collect();
        assert!(tiers.windows(2).all(|w| w[0] <= w[1]), "{tiers:?}");
        let tiny = plan(&info(&all), &[7 * GB + GB / 2], 1);
        let best = tiny.best().unwrap();
        assert_eq!(best.variant.label, "Q5_K_M", "only the best reduced-context tier that fits at all: {:?}", tiny.ranked.iter().map(|s| (&s.variant.label, s.fit.as_ref().unwrap().tier)).collect::<Vec<_>>());
        assert_eq!(best.fit.as_ref().unwrap().tier, 2);
    }

    #[test]
    fn unsupported_quant_types_and_incomplete_splits_are_never_the_best() {
        let files = files_1017();
        let info = RepoInfo { repo: "x/y".into(), grouped: group(&files, None), shape: Ok(shape()), draft: None };
        let p = plan(&info, &[500 * GB], 1);
        let best = p.best().unwrap();
        assert!(best.variant.label == "IQ4_NL" || best.runnable.is_err() || !best.variant.complete());
        assert!(p.ranked.iter().all(|s| s.runnable.is_err()), "IQ quants are not supported by specrt");
        let unsupported = RepoInfo { shape: Err("model_type deepseek_v4 is not supported".into()), ..info };
        let p = plan(&unsupported, &[500 * GB], 1);
        assert_eq!(p.unsupported.as_deref(), Some("model_type deepseek_v4 is not supported"));
        assert!(p.best().is_some_and(|b| !b.downloadable()));
    }

    #[test]
    fn repos_rank_by_their_best_variants_fit_and_unsupported_ones_go_last() {
        let mk = |name: &str, free: u64, size: u64| {
            let mut i = info(&[("Q4_K_M", size)]);
            i.repo = name.into();
            plan(&i, &[free], 1)
        };
        let unsupported = {
            let mut i = info(&[("Q4_K_M", GB)]);
            i.repo = "z/unsupported".into();
            i.shape = Err("nope".into());
            plan(&i, &[80 * GB], 1)
        };
        let ranked = rank_repos(vec![unsupported, mk("a/too-big", 80 * GB, 90 * GB), mk("b/ok", 80 * GB, 5 * GB)]);
        assert_eq!(ranked.iter().map(|p| p.repo.as_str()).collect::<Vec<_>>(), ["b/ok", "a/too-big", "z/unsupported"]);
    }

    #[test]
    fn repos_that_fit_equally_keep_the_search_order_whatever_their_size() {
        let mk = |name: &str, size: u64| {
            let mut i = info(&[("Q4_K_M", size)]);
            i.repo = name.into();
            plan(&i, &[80 * GB], 1)
        };
        let ranked = rank_repos(vec![mk("popular/big", 20 * GB), mk("second/small", 2 * GB), mk("third/mid", 8 * GB)]);
        assert_eq!(ranked.iter().map(|p| p.repo.as_str()).collect::<Vec<_>>(), ["popular/big", "second/small", "third/mid"]);
    }

    #[test]
    fn a_variant_downloads_every_shard_and_companions_into_the_hub_cache_and_resumes_after_an_interruption() {
        let shard = |n: u8, len: usize| (0..len).map(|i| (i as u8).wrapping_add(n)).collect::<Vec<u8>>();
        let (s1, s2) = (shard(1, 100), shard(2, 250));
        let routes = vec![
            ("/acme/m/resolve/main/m-Q4_K_M-00001-of-00002.gguf".to_string(), s1.clone()),
            ("/acme/m/resolve/main/m-Q4_K_M-00002-of-00002.gguf".to_string(), s2.clone()),
            ("/acme/m/resolve/main/config.json".to_string(), b"{}".to_vec()),
            ("/acme/m/resolve/main/mmproj-m-BF16.gguf".to_string(), shard(3, 40)),
        ];
        let (base, log) = serve(routes);
        let files = vec![
            RepoFile::new("config.json", 2),
            RepoFile::new("mmproj-m-BF16.gguf", 40),
            RepoFile::new("m-Q4_K_M-00001-of-00002.gguf", 100),
            RepoFile::new("m-Q4_K_M-00002-of-00002.gguf", 250),
        ];
        let hub = Hub::new(&base, "");
        let dir = std::env::temp_dir().join(format!("ib-variant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let second = hf::snapshot_path(&dir, "acme/m", "m-Q4_K_M-00002-of-00002.gguf");
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(format!("{}.part", second.display()), &s2[..60]).unwrap();
        let first = hf::snapshot_path(&dir, "acme/m", "m-Q4_K_M-00001-of-00002.gguf");
        std::fs::write(&first, &s1).unwrap();
        let progress = hf::Progress::default();
        let paths = hub.download_variant("acme/m", &files, &dir, &progress).unwrap();
        assert_eq!(paths.len(), 4);
        assert_eq!(std::fs::read(&second).unwrap(), s2);
        assert_eq!(std::fs::read(&first).unwrap(), s1);
        assert_eq!(progress.fraction(), Some(1.0));
        assert_eq!(progress.total.load(Ordering::Relaxed), 392);
        let seen = log.lock().unwrap().clone();
        assert!(!seen.iter().any(|p| p.contains("00001-of-00002")), "a finished shard is not fetched again: {seen:?}");
        assert!(dir.join("models--acme--m/snapshots/main/config.json").exists());
        assert!(!PathBuf::from(format!("{}.part", second.display())).exists());
    }

    #[test]
    fn a_shard_of_the_wrong_size_fails_the_download() {
        let (base, _) = serve(vec![("/acme/m/resolve/main/a.gguf".into(), vec![1u8; 10])]);
        let dir = std::env::temp_dir().join(format!("ib-variant-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let e = Hub::new(&base, "").download_variant("acme/m", &[RepoFile::new("a.gguf", 99)], &dir, &hf::Progress::default()).unwrap_err();
        assert!(e.contains("expected 99"), "{e}");
    }

    #[test]
    fn the_download_list_has_companions_then_mmproj_then_the_shards() {
        let g = group(&files_1017(), None);
        let v = g.variants.iter().find(|v| v.label == "IQ4_NL").unwrap();
        let names: Vec<String> = download_list(&g, v).into_iter().map(|f| f.path).collect();
        assert_eq!(names, ["chat_template.jinja", "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf", "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"]);
    }

    #[test]
    fn a_dflash_draft_repo_is_a_draft_model_not_a_runnable_one_and_ranks_below_real_models() {
        let text = include_str!("../tests/fixtures/hf/dflash2-draft-config.json");
        let (shape, draft) = config_shape_of("z-lab/Qwen3.8-27B-DFlash2", text);
        assert_eq!(draft.as_deref(), Some("a larger model with 64 layers"));
        assert!(shape.as_ref().err().is_some_and(|e| e.starts_with("draft model for")), "{:?}", shape.as_ref().err());
        let files = vec![RepoFile::new("model.safetensors", 3 * GB), RepoFile::new("config.json", 100)];
        let drafter = RepoInfo { repo: "z-lab/Qwen3.8-27B-DFlash2".into(), grouped: group(&files, Some(text)), shape, draft };
        let real = info(&[("Q4_K_M", 5 * GB)]);
        let plans = rank_repos(vec![plan(&drafter, &[90 * GB], 1), plan(&real, &[90 * GB], 1)]);
        assert_eq!(plans[0].repo, "acme/m");
        assert!(plans[1].draft.is_some() && plans[1].unsupported.is_none());
        assert!(!plans[1].best().unwrap().downloadable());
    }

    #[test]
    fn target_and_drafter_markers_are_read_from_the_config_and_plain_models_are_not_drafts() {
        let v = |t: &str| serde_json::from_str::<serde_json::Value>(t).unwrap();
        assert!(draft_of(&v(r#"{"architectures":["LlamaForCausalLMEagle"]}"#)).is_some());
        assert_eq!(draft_of(&v(r#"{"target_model":"org/Big"}"#)).as_deref(), Some("org/Big"));
        assert!(draft_of(&v(r#"{"architectures":["Qwen3ForCausalLM"],"num_hidden_layers":36}"#)).is_none());
        assert!(draft_of(&v(r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"mtp_num_hidden_layers":1}}"#)).is_none());
    }

    #[test]
    fn modelopt_nvfp4_is_flagged_with_one_short_reason() {
        let text = include_str!("../tests/fixtures/hf/nvfp4-qwen3.8-27b-config.json");
        let (shape, draft) = config_shape_of("nvidia/Qwen3.8-27B-NVFP4", text);
        assert!(draft.is_none());
        let e = shape.err().expect("specrt does not read modelopt mixed precision yet");
        assert_eq!(e, "modelopt NVFP4 not supported yet");
    }

    #[test]
    fn the_users_compressed_tensors_nvfp4_27b_is_supported_and_sized_for_the_gpu() {
        let text = include_str!("../tests/fixtures/hf/orcarouter-nvfp4-config.json");
        let (shape, draft) = config_shape_of("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", text);
        assert!(draft.is_none());
        let shape = shape.unwrap_or_else(|e| panic!("{e}"));
        assert!(shape.anatomy.total_bytes > 10 * GB && shape.anatomy.total_bytes < 30 * GB, "{}", shape.anatomy.total_bytes);
        let a = assess(&shape.spec, &shape.anatomy, 20 * GB, &[90 * GB], 1);
        assert_ne!(a.verdict, Verdict::DoesNotFit);
    }

    #[test]
    fn context_labels() {
        assert_eq!((context_label(131072).as_str(), context_label(49152).as_str(), context_label(4096).as_str(), context_label(512).as_str()), ("128k", "48k", "4k", "512"));
    }
}
