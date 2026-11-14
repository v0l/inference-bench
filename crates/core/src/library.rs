use crate::specs::{self, SpecDir, SpecOrigin};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const BACKENDS: [&str; 4] = ["specrt", "specrt (wgpu)", "llama.cpp", "vllm"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Safetensors,
    Gguf,
    Spec,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Spec => "spec",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Support {
    pub backend: &'static str,
    pub supported: bool,
    pub reason: String,
}

impl Support {
    fn yes(backend: &'static str, reason: &str) -> Self {
        Self { backend, supported: true, reason: reason.into() }
    }

    fn no(backend: &'static str, reason: impl Into<String>) -> Self {
        Self { backend, supported: false, reason: reason.into() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalModel {
    pub name: String,
    pub repo: String,
    pub path: PathBuf,
    pub root: PathBuf,
    pub format: Format,
    pub size_bytes: u64,
    pub architecture: String,
    pub quant: String,
    pub context_length: Option<u64>,
    pub files: Vec<PathBuf>,
    pub support: Vec<Support>,
    pub origin: Option<SpecOrigin>,
    pub vision: bool,
}

pub fn config_has_vision(config: &Value) -> bool {
    specrt::vision::Vision::from_hf_config(&config.to_string()).is_ok_and(|v| v.is_some())
}

impl LocalModel {
    pub fn support_for(&self, backend: &str) -> Option<&Support> {
        self.support.iter().find(|s| s.backend == backend)
    }

    pub fn model_id(&self, backend: &str) -> String {
        match (backend, self.format) {
            ("specrt", Format::Spec) => self.path.display().to_string(),
            ("specrt" | "llama.cpp", Format::Gguf) => self.path.display().to_string(),
            ("specrt" | "vllm", _) if !self.repo.is_empty() => self.repo.clone(),
            _ => self.name.clone(),
        }
    }

    pub fn reveal_dir(&self) -> PathBuf {
        match self.path.is_dir() {
            true => self.path.clone(),
            false => self.path.parent().map(Path::to_path_buf).unwrap_or_default(),
        }
    }
}

pub fn hub_dir() -> PathBuf {
    if let Some(h) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(h);
    }
    if let Some(h) = std::env::var_os("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    dirs::home_dir().unwrap_or_default().join(".cache/huggingface/hub")
}

const VLLM_ARCHS: [&str; 22] = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Phi3ForCausalLM",
    "GraniteForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoXForCausalLM",
    "FalconForCausalLM",
    "DeepseekV3ForCausalLM",
    "OlmoForCausalLM",
];

const TEXT_GENERATION_HINTS: [&str; 10] = [
    "CausalLM",
    "LMHeadModel",
    "ConditionalGeneration",
    "ForCausalLM",
    "Llama",
    "Qwen",
    "Mistral",
    "Gemma",
    "Phi",
    "Granite",
];

fn is_text_generation(arch: &str) -> bool {
    let non_text = ["Whisper", "Vit", "ViT", "Bert", "Clip", "Wav2Vec", "TTS", "Tts", "Speech", "ForImageClassification", "ASR"];
    !non_text.iter().any(|n| arch.contains(n)) && TEXT_GENERATION_HINTS.iter().any(|h| arch.contains(h))
}

pub fn specrt_support(config: &Value) -> Support {
    match specrt::spec::Spec::from_hf_config(&config.to_string()) {
        Ok(spec) => Support {
            backend: "specrt",
            supported: true,
            reason: format!("{} on CUDA", describe_spec(&spec)),
        },
        Err(e) => Support::no("specrt", e.root_cause().to_string()),
    }
}

pub fn specrt_wgpu_support(config: &Value) -> Support {
    match specrt::spec::Spec::from_hf_config(&config.to_string()) {
        Ok(spec) => match specrt::wgsl::unsupported(&spec) {
            None => Support { backend: "specrt (wgpu)", supported: true, reason: "dense decoder on Vulkan, Metal or DX12".into() },
            Some(why) => Support::no("specrt (wgpu)", why),
        },
        Err(e) => Support::no("specrt (wgpu)", e.root_cause().to_string()),
    }
}

fn text_config(config: &Value) -> &Value {
    match config.get("text_config") {
        Some(t) if t.is_object() => t,
        _ => config,
    }
}

fn describe_spec(spec: &specrt::spec::Spec) -> String {
    let layout = if spec.linear_attention.is_some() { "hybrid delta-net decoder" } else { "dense decoder" };
    let quant = serde_json::to_value(spec.quant).ok().and_then(|q| q.as_str().map(String::from)).unwrap_or_default();
    format!("{layout}, {quant}")
}

fn support_for_safetensors(config: &Value, arch: &str) -> Vec<Support> {
    let text = is_text_generation(arch);
    let none = || "not a text generation model".to_string();
    let vllm = match (text, VLLM_ARCHS.contains(&arch)) {
        (false, _) => Support::no("vllm", none()),
        (true, true) => Support::yes("vllm", "architecture is registered"),
        (true, false) => Support::no("vllm", format!("Model architectures ['{arch}'] are not supported")),
    };
    let specrt = specrt_support(config);
    let wgpu = specrt_wgpu_support(config);
    vec![
        specrt,
        wgpu,
        Support::no("llama.cpp", "needs a GGUF conversion"),
        vllm,
    ]
}

const LLAMA_CPP_ARCHS: [&str; 22] = [
    "llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next", "gemma",
    "gemma2", "gemma3", "phi3", "mistral3", "gpt2", "gptneox", "falcon", "granite", "olmo", "deepseek2",
    "mixtral", "starcoder2", "command-r",
];

fn support_for_gguf(arch: &str, files: &[PathBuf]) -> Vec<Support> {
    let known = LLAMA_CPP_ARCHS.contains(&arch);
    vec![
        specrt_gguf_support(files),
        specrt_wgpu_gguf_support(files),
        if known {
            Support::yes("llama.cpp", "architecture is registered")
        } else {
            Support::no("llama.cpp", format!("unknown model architecture: '{arch}'"))
        },
        if known {
            Support::yes("vllm", "experimental GGUF loader")
        } else {
            Support::no("vllm", format!("GGUF architecture '{arch}' is not supported"))
        },
    ]
}

fn specrt_gguf_support(files: &[PathBuf]) -> Support {
    let [file] = files else { return Support::no("specrt", "split GGUF files are not read by specrt") };
    match specrt::gguf::Gguf::open(file).and_then(|g| g.spec().map(|_| g.weight_type_bytes())) {
        Ok(types) => {
            let names: Vec<String> = types.iter().map(|(k, _)| k.name()).collect();
            Support { backend: "specrt", supported: true, reason: format!("GGUF {}", names.join(" + ")) }
        }
        Err(e) => Support::no("specrt", e.root_cause().to_string()),
    }
}

fn specrt_wgpu_gguf_support(files: &[PathBuf]) -> Support {
    let [file] = files else { return Support::no("specrt (wgpu)", "split GGUF files are not read by specrt") };
    match specrt::gguf::Gguf::open(file).and_then(|g| g.spec()) {
        Ok(spec) => match specrt::wgsl::unsupported(&spec) {
            None => Support { backend: "specrt (wgpu)", supported: true, reason: "dense decoder on Vulkan, Metal or DX12".into() },
            Some(why) => Support::no("specrt (wgpu)", why),
        },
        Err(e) => Support::no("specrt (wgpu)", e.root_cause().to_string()),
    }
}

fn dir_size(files: &[PathBuf]) -> u64 {
    files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum()
}

fn gguf_quant(file_type: Option<u32>, file_name: &str) -> String {
    let named = match file_type {
        Some(0) => "F32",
        Some(1) => "F16",
        Some(2) => "Q4_0",
        Some(3) => "Q4_1",
        Some(7) => "Q8_0",
        Some(8) => "Q5_0",
        Some(9) => "Q5_1",
        Some(10) => "Q2_K",
        Some(11) => "Q3_K_S",
        Some(12) => "Q3_K_M",
        Some(13) => "Q3_K_L",
        Some(14) => "Q4_K_S",
        Some(15) => "Q4_K_M",
        Some(16) => "Q5_K_S",
        Some(17) => "Q5_K_M",
        Some(18) => "Q6_K",
        Some(32) => "BF16",
        _ => "",
    };
    if !named.is_empty() {
        return named.to_string();
    }
    let upper = file_name.to_uppercase();
    for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q2_K", "BF16", "F16", "F32", "MXFP4"] {
        if upper.contains(q) {
            return q.to_string();
        }
    }
    "gguf".into()
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GgufHeader {
    pub architecture: String,
    pub name: String,
    pub file_type: Option<u32>,
    pub context_length: Option<u64>,
}

fn read_u32(r: &mut impl Read) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> Option<String> {
    let len = read_u64(r)? as usize;
    if len > 1 << 20 {
        return None;
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn skip(r: &mut impl Read, n: u64) -> Option<()> {
    std::io::copy(&mut r.take(n), &mut std::io::sink()).ok().filter(|c| *c == n).map(|_| ())
}

#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Num(u64),
    Text(String),
    Other,
}

fn read_value(r: &mut impl Read, kind: u32) -> Option<Scalar> {
    let fixed = |r: &mut dyn Read, n: usize| -> Option<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b[..n]).ok()?;
        Some(u64::from_le_bytes(b))
    };
    Some(match kind {
        0 | 1 | 7 => Scalar::Num(fixed(r, 1)?),
        2 | 3 => Scalar::Num(fixed(r, 2)?),
        4 | 5 | 6 => Scalar::Num(fixed(r, 4)?),
        10 | 11 | 12 => Scalar::Num(fixed(r, 8)?),
        8 => Scalar::Text(read_string(r)?),
        9 => {
            let inner = read_u32(r)?;
            let count = read_u64(r)?;
            if inner == 8 {
                for _ in 0..count {
                    read_string(r)?;
                }
            } else {
                let width = match inner {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    _ => return None,
                };
                skip(r, count * width)?;
            }
            Scalar::Other
        }
        _ => return None,
    })
}

pub fn read_gguf_header(path: &Path) -> Option<GgufHeader> {
    let file = std::fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut r)?;
    if version < 2 {
        return None;
    }
    let _tensors = read_u64(&mut r)?;
    let kvs = read_u64(&mut r)?;
    let mut header = GgufHeader::default();
    let mut context: BTreeMap<String, u64> = BTreeMap::new();
    for _ in 0..kvs {
        let key = read_string(&mut r)?;
        let kind = read_u32(&mut r)?;
        let value = read_value(&mut r, kind)?;
        match (key.as_str(), value) {
            ("general.architecture", Scalar::Text(t)) => header.architecture = t,
            ("general.name", Scalar::Text(t)) => header.name = t,
            ("general.file_type", Scalar::Num(n)) => header.file_type = Some(n as u32),
            (k, Scalar::Num(n)) if k.ends_with(".context_length") => {
                context.insert(k.to_string(), n);
            }
            _ => {}
        }
    }
    header.context_length = context
        .get(&format!("{}.context_length", header.architecture))
        .or_else(|| context.values().next())
        .copied();
    Some(header)
}

fn shard_key(file: &str) -> (String, bool) {
    let stem = file.trim_end_matches(".gguf");
    if let Some(i) = stem.rfind("-of-")
        && let Some(j) = stem[..i].rfind('-')
        && stem[j + 1..i].chars().all(|c| c.is_ascii_digit())
        && stem[i + 4..].chars().all(|c| c.is_ascii_digit())
    {
        return (stem[..j].to_string(), &stem[j + 1..i] == "00001");
    }
    (stem.to_string(), true)
}

fn gguf_models(dir: &Path, repo: &str, root: &Path, found: &mut Vec<LocalModel>) {
    let mut files = Vec::new();
    collect(dir, 4, &mut |p| p.extension().is_some_and(|e| e == "gguf"), &mut files);
    files.sort();
    let mut groups: BTreeMap<(PathBuf, String), Vec<PathBuf>> = BTreeMap::new();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("mmproj") || name.contains("-vocab") {
            continue;
        }
        let key = (f.parent().unwrap().to_path_buf(), shard_key(&name).0);
        groups.entry(key).or_default().push(f);
    }
    for ((_, stem), shards) in groups {
        let first = shards[0].clone();
        let Some(h) = read_gguf_header(&first) else { continue };
        let name = match repo.is_empty() {
            true => stem.clone(),
            false => format!("{} {}", repo.rsplit('/').next().unwrap_or(repo), gguf_quant(h.file_type, &stem)),
        };
        found.push(LocalModel {
            name,
            repo: repo.to_string(),
            path: first,
            root: root.to_path_buf(),
            format: Format::Gguf,
            size_bytes: dir_size(&shards),
            quant: gguf_quant(h.file_type, &stem),
            context_length: h.context_length,
            support: support_for_gguf(&h.architecture, &shards),
            architecture: h.architecture,
            files: shards,
            origin: None,
            vision: false,
        });
    }
}

fn collect(dir: &Path, depth: usize, keep: &mut dyn FnMut(&Path) -> bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if depth > 0 {
                collect(&p, depth - 1, keep, out);
            }
        } else if keep(&p) {
            out.push(p);
        }
    }
}

fn safetensors_model(dir: &Path, repo: &str, root: &Path) -> Option<LocalModel> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let config: Value = serde_json::from_str(&text).ok()?;
    let mut files = Vec::new();
    collect(dir, 0, &mut |p| p.extension().is_some_and(|e| e == "safetensors"), &mut files);
    if files.is_empty() {
        return None;
    }
    files.sort();
    let arch = config["architectures"][0].as_str().unwrap_or("unknown").to_string();
    let cfg = text_config(&config);
    let quant = match config.get("quantization_config").filter(|q| !q.is_null()) {
        Some(q) => q["quant_method"].as_str().unwrap_or("quantised").to_uppercase(),
        None => {
            let dtype = cfg["torch_dtype"].as_str().or(cfg["dtype"].as_str()).unwrap_or("");
            match dtype {
                "bfloat16" => "BF16",
                "float16" => "F16",
                "float32" => "F32",
                _ => "-",
            }
            .into()
        }
    };
    let name = match repo.is_empty() {
        true => dir.file_name()?.to_string_lossy().into_owned(),
        false => repo.rsplit('/').next().unwrap_or(repo).to_string(),
    };
    Some(LocalModel {
        name,
        repo: repo.to_string(),
        path: dir.to_path_buf(),
        root: root.to_path_buf(),
        format: Format::Safetensors,
        size_bytes: dir_size(&files),
        support: support_for_safetensors(&config, &arch),
        architecture: arch,
        quant,
        context_length: cfg["max_position_embeddings"].as_u64(),
        files,
        origin: None,
        vision: config_has_vision(&config),
    })
}

fn newest_snapshot(snapshots: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshots) else { return Vec::new() };
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| (e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), e.path()))
        .collect();
    dirs.sort();
    dirs.into_iter().rev().map(|(_, p)| p).collect()
}

pub fn scan_hub(hub: &Path) -> Vec<LocalModel> {
    let Ok(entries) = std::fs::read_dir(hub) else { return Vec::new() };
    let mut found = Vec::new();
    let mut repos: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    repos.sort();
    for dir in repos {
        let Some(name) = dir.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(slug) = name.strip_prefix("models--") else { continue };
        let repo = slug.replacen("--", "/", 1);
        for snap in newest_snapshot(&dir.join("snapshots")) {
            let before = found.len();
            if let Some(m) = safetensors_model(&snap, &repo, &dir) {
                found.push(m);
            }
            gguf_models(&snap, &repo, &dir, &mut found);
            if found.len() > before {
                break;
            }
        }
    }
    found
}

pub fn scan_folder(folder: &Path) -> Vec<LocalModel> {
    let mut found = Vec::new();
    gguf_models(folder, "", folder, &mut found);
    let mut configs = Vec::new();
    collect(folder, 3, &mut |p| p.file_name().is_some_and(|n| n == "config.json"), &mut configs);
    for c in configs {
        if let Some(m) = c.parent().and_then(|d| safetensors_model(d, "", folder)) {
            found.push(m);
        }
    }
    found
}

fn spec_model(listed: &specs::Listed, hub: &Path) -> LocalModel {
    let spec = &listed.table;
    let repo = spec.get("repo").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let quant = spec.get("quant").and_then(|v| v.as_str()).unwrap_or("bf16").to_uppercase();
    let weights = hub.join(format!("models--{}", repo.replace('/', "--")));
    let mut files = Vec::new();
    for snap in newest_snapshot(&weights.join("snapshots")) {
        collect(&snap, 0, &mut |p| p.extension().is_some_and(|e| e == "safetensors"), &mut files);
        if !files.is_empty() {
            break;
        }
    }
    let present = !files.is_empty();
    let layers = spec.get("layers").and_then(|l| l.get("count")).and_then(|c| c.as_integer());
    let specrt = match (&listed.problem, present) {
        (Some(problem), _) => Support::no("specrt", problem.clone()),
        (None, true) => Support::yes("specrt", "spec file, weights in the hub cache"),
        (None, false) => Support::no("specrt", "weights are not in the hub cache"),
    };
    LocalModel {
        name: listed.name.clone(),
        repo,
        path: listed.path.clone(),
        root: listed.dir.clone(),
        format: Format::Spec,
        size_bytes: dir_size(&files),
        architecture: format!("spec, {} layers", layers.unwrap_or(0)),
        quant,
        context_length: None,
        support: vec![
            specrt,
            Support::no("llama.cpp", "spec files are specrt only"),
            Support::no("vllm", "spec files are specrt only"),
        ],
        vision: files.first().and_then(|f| f.parent()).and_then(|d| std::fs::read_to_string(d.join("config.json")).ok()).and_then(|t| serde_json::from_str::<Value>(&t).ok()).is_some_and(|c| config_has_vision(&c)),
        files,
        origin: Some(listed.origin.clone()),
    }
}

pub fn spec_models(layers: &[SpecDir], hub: &Path) -> Vec<LocalModel> {
    specs::list(layers).iter().map(|l| spec_model(l, hub)).collect()
}

pub fn scan_specs(dir: &Path, hub: &Path) -> Vec<LocalModel> {
    spec_models(&[SpecDir { path: dir.to_path_buf(), origin: SpecOrigin::Folder }], hub)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sources {
    pub hub: PathBuf,
    pub folders: Vec<PathBuf>,
    pub specs: Vec<SpecDir>,
}

impl Sources {
    pub fn standard(data: &Path, folders: &[PathBuf], spec_folders: &[PathBuf]) -> Self {
        Self { hub: hub_dir(), folders: folders.to_vec(), specs: specs::standard_layers(data, spec_folders) }
    }
}

pub fn scan(sources: &Sources) -> Vec<LocalModel> {
    let mut all = scan_hub(&sources.hub);
    all.extend(spec_models(&sources.specs, &sources.hub));
    for folder in &sources.folders {
        all.extend(scan_folder(folder));
    }
    let mut seen = std::collections::HashSet::new();
    all.retain(|m| seen.insert(m.path.clone()));
    all
}

pub fn deletable(model: &LocalModel) -> bool {
    model.format != Format::Spec || model.origin == Some(SpecOrigin::Folder)
}

pub fn delete(model: &LocalModel) -> std::io::Result<()> {
    if !deletable(model) {
        return Err(std::io::Error::other("this spec is managed by inference-bench: turn off spec updates or add your own folder instead"));
    }
    match model.format {
        Format::Spec => std::fs::remove_file(&model.path),
        _ if model.root.join("snapshots").is_dir() || model.root.file_name().is_some_and(|n| n.to_string_lossy().starts_with("models--")) => {
            std::fs::remove_dir_all(&model.root)
        }
        _ => {
            for f in &model.files {
                std::fs::remove_file(f)?;
            }
            Ok(())
        }
    }
}

pub fn human_size(bytes: u64) -> String {
    match bytes {
        0 => "-".into(),
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-lib-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    pub fn gguf_bytes(arch: &str, file_type: u32, context: u32) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(5u64.to_le_bytes());
        put_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, arch);
        put_str(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        put_str(&mut b, "a");
        put_str(&mut b, "b");
        put_str(&mut b, "general.file_type");
        b.extend(4u32.to_le_bytes());
        b.extend(file_type.to_le_bytes());
        put_str(&mut b, "general.name");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, "Test Model");
        put_str(&mut b, &format!("{arch}.context_length"));
        b.extend(4u32.to_le_bytes());
        b.extend(context.to_le_bytes());
        b
    }

    fn fixture(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures").join(name);
        std::fs::read_to_string(path).unwrap()
    }

    fn hf_model(hub: &Path, repo: &str, config: &str) -> PathBuf {
        let snap = hub.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), config).unwrap();
        std::fs::write(snap.join("model.safetensors"), vec![0u8; 2048]).unwrap();
        snap
    }

    #[test]
    fn a_gguf_header_gives_architecture_quant_and_context() {
        let dir = temp("gguf");
        let f = dir.join("m-Q4_K_M.gguf");
        std::fs::write(&f, gguf_bytes("qwen35", 15, 262144)).unwrap();
        let h = read_gguf_header(&f).unwrap();
        assert_eq!((h.architecture.as_str(), h.file_type, h.context_length), ("qwen35", Some(15), Some(262144)));
        assert_eq!(h.name, "Test Model");
        assert!(read_gguf_header(&dir).is_none());
    }

    #[test]
    fn the_hub_lists_a_supported_and_an_unsupported_architecture() {
        let hub = temp("hub");
        hf_model(&hub, "Qwen/Qwen3-0.6B", &fixture("qwen3-0.6b.config.json"));
        hf_model(
            &hub,
            "acme/Weird-7B",
            r#"{"architectures":["WeirdForCausalLM"],"model_type":"weird","hidden_act":"silu","max_position_embeddings":4096,"torch_dtype":"bfloat16"}"#,
        );
        let models = scan_hub(&hub);
        assert_eq!(models.len(), 2);
        let qwen = models.iter().find(|m| m.name == "Qwen3-0.6B").unwrap();
        assert_eq!((qwen.quant.as_str(), qwen.context_length, qwen.size_bytes), ("BF16", Some(40960), 2048));
        assert!(qwen.support_for("specrt").unwrap().supported);
        assert!(qwen.support_for("specrt (wgpu)").unwrap().supported);
        assert!(qwen.support_for("vllm").unwrap().supported);
        assert!(!qwen.support_for("llama.cpp").unwrap().supported);
        let weird = models.iter().find(|m| m.name == "Weird-7B").unwrap();
        let s = weird.support_for("specrt").unwrap();
        assert!(!s.supported && s.reason.contains("model_type weird is not supported"), "{}", s.reason);
        assert!(weird.support_for("vllm").unwrap().reason.contains("WeirdForCausalLM"));
        assert_eq!(qwen.model_id("specrt"), "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn a_speech_model_is_not_a_text_generation_model() {
        let hub = temp("speech");
        hf_model(
            &hub,
            "openai/whisper-tiny.en",
            r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper"}"#,
        );
        let m = &scan_hub(&hub)[0];
        assert!(m.support.iter().all(|s| !s.supported));
        assert!(m.support_for("vllm").unwrap().reason.contains("not a text generation"));
    }

    #[test]
    fn the_real_qwen3_8_27b_fp8_config_is_supported_by_specrt() {
        let hub = temp("qwen27");
        hf_model(&hub, "Qwen/Qwen3.8-27B-FP8", &fixture("qwen3.8-27b.config.json"));
        let m = &scan_hub(&hub)[0];
        assert_eq!(m.quant, "FP8");
        let s = m.support_for("specrt").unwrap();
        assert!(s.supported, "{}", s.reason);
        assert!(s.reason.contains("fp8_block128"), "{}", s.reason);
        let wgpu = m.support_for("specrt (wgpu)").unwrap();
        assert!(!wgpu.supported && wgpu.reason.contains("delta-net"), "{}", wgpu.reason);
    }

    #[test]
    fn the_real_qwen3_8_27b_nvfp4_config_is_supported_by_specrt() {
        let hub = temp("qwen27-nvfp4");
        hf_model(&hub, "orcarouter/Qwen3.8-27B-Uncensored-NVFP4", &fixture("qwen3.8-27b-nvfp4.config.json"));
        let m = &scan_hub(&hub)[0];
        let s = m.support_for("specrt").unwrap();
        assert!(s.supported, "{}", s.reason);
        assert!(s.reason.contains("fp8_channel"), "{}", s.reason);
        let wgpu = m.support_for("specrt (wgpu)").unwrap();
        assert!(!wgpu.supported && wgpu.reason.contains("delta-net"), "{}", wgpu.reason);
    }

    #[test]
    fn an_unsupported_compressed_tensors_group_is_refused_with_the_spec_error() {
        let mut config: Value = serde_json::from_str(&fixture("qwen3.8-27b-nvfp4.config.json")).unwrap();
        config["quantization_config"]["config_groups"]["group_1"]["weights"]["group_size"] = serde_json::json!(32);
        let s = specrt_support(&config);
        assert!(!s.supported && s.reason.contains("group_1"), "{}", s.reason);
    }

    fn engine_config_fixtures() -> Vec<(String, Value)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures");
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            let (name, file) = match path.is_dir() {
                true => (path.file_name().unwrap().to_string_lossy().into_owned(), path.join("config.json")),
                false => (path.file_name().unwrap().to_string_lossy().into_owned(), path.clone()),
            };
            let is_config = match path.is_dir() {
                true => file.exists(),
                false => name.ends_with(".config.json"),
            };
            if is_config {
                found.push((name, serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap()));
            }
        }
        found.sort_by(|a, b| a.0.cmp(&b.0));
        found
    }

    #[test]
    fn every_config_the_engine_tests_load_is_supported_by_the_library() {
        let engine_refuses = ["qwen3.6-35b-a3b-fp8.config.json", "gemma4-31b.config.json"];
        let fixtures = engine_config_fixtures();
        assert!(fixtures.len() >= 12, "{} fixtures", fixtures.len());
        for (name, config) in &fixtures {
            let engine = specrt::spec::Spec::from_hf_config(&config.to_string());
            let s = specrt_support(config);
            assert_eq!(s.supported, engine.is_ok(), "{name}: library says {:?}, engine says {:?}", s, engine.err());
            assert_eq!(s.supported, !engine_refuses.contains(&name.as_str()), "{name}: {}", s.reason);
        }
    }

    #[test]
    fn an_unsupported_quant_scheme_is_refused_with_the_spec_error() {
        let mut config: Value = serde_json::from_str(&fixture("qwen3.8-27b.config.json")).unwrap();
        config["quantization_config"] = serde_json::json!({"quant_method": "gptq", "bits": 4});
        let s = specrt_support(&config);
        assert!(!s.supported && s.reason.contains("quantization") && s.reason.contains("gptq"), "{}", s.reason);
    }

    #[test]
    fn an_unknown_model_type_is_refused_with_the_spec_error() {
        let mut config: Value = serde_json::from_str(&fixture("qwen3-0.6b.config.json")).unwrap();
        config["model_type"] = "weird".into();
        let s = specrt_support(&config);
        assert!(!s.supported && s.reason == "model_type weird is not supported", "{}", s.reason);
    }

    #[test]
    fn a_dense_bf16_config_is_supported_on_cuda_and_wgpu() {
        let hub = temp("dense");
        hf_model(&hub, "Qwen/Qwen3-0.6B", &fixture("qwen3-0.6b.config.json"));
        let m = &scan_hub(&hub)[0];
        assert!(m.support_for("specrt").unwrap().supported);
        assert!(m.support_for("specrt (wgpu)").unwrap().supported);
    }

    #[test]
    fn specrt_supports_a_dense_qwen3_gguf_and_names_an_unsupported_type() {
        use specrt::gguf::writer::{Kv, build};
        let hub = temp("specrt-gguf");
        let snap = hub.join("models--unsloth--Tiny-GGUF/snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        let file = |weight_type: u32, weight: &[u8]| {
            let embd = vec![0u8; 2 * 256 * 4];
            build(
                3,
                &[
                    ("general.architecture", Kv::Str("qwen3")),
                    ("qwen3.block_count", Kv::U32(1)),
                    ("qwen3.embedding_length", Kv::U32(256)),
                    ("qwen3.feed_forward_length", Kv::U32(256)),
                    ("qwen3.attention.head_count", Kv::U32(2)),
                    ("qwen3.attention.head_count_kv", Kv::U32(1)),
                    ("qwen3.attention.key_length", Kv::U32(128)),
                    ("qwen3.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
                    ("general.file_type", Kv::U32(15)),
                ],
                &[("token_embd.weight", &[256, 4], 1, &embd[..]), ("blk.0.ffn_down.weight", &[256, 256], weight_type, weight)],
            )
        };
        std::fs::write(snap.join("Tiny-Q4_K_M.gguf"), file(12, &vec![0u8; 256 * 144])).unwrap();
        std::fs::write(snap.join("Tiny-Q4_0.gguf"), file(2, &vec![0u8; 256 * 8 * 18])).unwrap();
        let models = scan_hub(&hub);
        let find = |q: &str| models.iter().find(|m| m.path.to_string_lossy().contains(q)).unwrap();
        let good = find("Q4_K_M");
        let s = good.support_for("specrt").unwrap();
        assert!(s.supported && s.reason.contains("Q4_K"), "{}", s.reason);
        assert_eq!(good.model_id("specrt"), good.path.display().to_string());
        let bad = find("Q4_0").support_for("specrt").unwrap();
        assert!(!bad.supported && bad.reason.contains("Q4_0") && bad.reason.contains("not supported"), "{}", bad.reason);
    }

    #[test]
    fn gguf_shards_are_one_model_and_unknown_architectures_are_flagged() {
        let hub = temp("shards");
        let snap = hub.join("models--unsloth--Big-GGUF/snapshots/abc/Q4_K_M");
        std::fs::create_dir_all(&snap).unwrap();
        for n in 1..=2 {
            std::fs::write(snap.join(format!("Big-Q4_K_M-0000{n}-of-00002.gguf")), gguf_bytes("brandnew", 15, 8192)).unwrap();
        }
        std::fs::write(snap.join("mmproj-BF16.gguf"), gguf_bytes("clip", 32, 1)).unwrap();
        let models = scan_hub(&hub);
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!((m.files.len(), m.quant.as_str(), m.format), (2, "Q4_K_M", Format::Gguf));
        assert!(m.size_bytes > 0);
        let s = m.support_for("llama.cpp").unwrap();
        assert!(!s.supported && s.reason.contains("unknown model architecture"), "{}", s.reason);
    }

    #[test]
    fn spec_files_show_whether_their_weights_are_cached() {
        let hub = temp("spechub");
        hf_model(&hub, "Qwen/Qwen3-0.6B", r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#);
        let specs = temp("specs");
        let real = specrt::spec::EMBEDDED.iter().find(|(n, _)| *n == "qwen3-0.6b.toml").unwrap().1;
        std::fs::write(specs.join("a.toml"), real).unwrap();
        std::fs::write(specs.join("b.toml"), real.replace("qwen3-0.6b", "missing").replace("Qwen/Qwen3-0.6B", "x/y")).unwrap();
        std::fs::write(specs.join("c.toml"), "name = \"half\"\nrepo = \"Qwen/Qwen3-0.6B\"\n").unwrap();
        let found = scan_specs(&specs, &hub);
        assert_eq!(found.len(), 3);
        assert!(found[0].support_for("specrt").unwrap().supported);
        assert!(!found[1].support_for("specrt").unwrap().supported);
        assert!(found[2].support_for("specrt").unwrap().reason.starts_with("spec does not parse"));
        assert_eq!(found[0].architecture, "spec, 28 layers");
        assert_eq!(found[0].model_id("specrt"), specs.join("a.toml").display().to_string());
        assert_eq!(found[0].origin, Some(SpecOrigin::Folder));
    }

    #[test]
    fn a_spec_for_a_newer_engine_says_so_and_managed_specs_cannot_be_deleted() {
        let hub = temp("newerhub");
        hf_model(&hub, "Qwen/Qwen3-0.6B", r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#);
        let data = temp("newerdata");
        let mine = data.join("mine");
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::write(mine.join("f.toml"), "min_engine = 50\nname = \"future-moe\"\nrepo = \"Qwen/Qwen3-0.6B\"\n[moe]\nexperts = 8\n").unwrap();
        let sources = Sources { hub: hub.clone(), folders: vec![], specs: specs::layers(&specs::SpecStore::new(&data), &[mine], &[]) };
        let all = scan(&sources);
        let future = all.iter().find(|m| m.name == "future-moe").unwrap();
        let s = future.support_for("specrt").unwrap();
        assert!(!s.supported && s.reason.starts_with("needs a newer inference-bench"), "{}", s.reason);
        assert!(deletable(future));
        let embedded = all.iter().find(|m| m.name == "qwen3-0.6b").unwrap();
        assert_eq!(embedded.origin, Some(SpecOrigin::Embedded));
        assert!(embedded.support_for("specrt").unwrap().supported);
        assert!(!deletable(embedded));
        assert!(delete(embedded).is_err());
        assert!(embedded.path.exists());
    }

    #[test]
    fn deleting_a_hub_model_removes_its_repo_folder() {
        let hub = temp("del");
        hf_model(&hub, "a/b", r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#);
        let m = scan_hub(&hub).remove(0);
        delete(&m).unwrap();
        assert!(scan_hub(&hub).is_empty());
    }

    #[test]
    fn sizes_are_readable() {
        assert_eq!(human_size(3 << 30), "3.0 GB");
        assert_eq!(human_size(5 << 20), "5 MB");
    }
}
