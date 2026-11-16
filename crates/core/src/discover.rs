use crate::fit;
use crate::hf::{self, Hub, RepoFile};
use crate::library::{self, Format};
use specrt::anatomy::ModelAnatomy;
use specrt::spec::{KvDtype, Spec};
use std::collections::BTreeMap;

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

    pub fn f16_layers(self) -> Option<usize> {
        (self == Kv::Fp8).then_some(0)
    }

    pub fn bytes_per_token(self, spec: &Spec) -> u64 {
        let mut s = spec.clone();
        s.runtime.kv_dtype = self.dtype();
        s.runtime.kv_f16_layers = self.f16_layers();
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
}

impl Assessment {
    pub fn full(&self) -> bool {
        self.chosen.context >= self.native && self.verdict != Verdict::Unknown
    }
}

pub fn assess(spec: &Spec, base: &ModelAnatomy, weights: u64, free: Option<u64>, seqs: usize) -> Assessment {
    let mut a = base.clone();
    a.total_bytes = weights;
    let native = fit::max_context(&a);
    let usable = free.map(|f| (f as f64 * fit::GPU_UTILIZATION) as u64);
    let room = fit::room_bytes(&a, usable, seqs);
    let kvs: Vec<KvFit> = Kv::ALL
        .iter()
        .map(|&kv| {
            let bytes_per_token = kv.bytes_per_token(spec);
            a.kv_bytes_per_token = bytes_per_token;
            let context = room.map_or(0, |r| fit::pool_tokens(&a, r).min(native));
            KvFit { kv, bytes_per_token, context }
        })
        .collect();
    let f16 = kvs[0].clone();
    let need = weights + a.state_bytes_per_seq * seqs.max(1) as u64 + kvs[0].bytes_per_token * fit::TYPICAL_REQUEST as u64;
    let (chosen, tier, verdict) = match room {
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
    Assessment { weights, native, kvs, chosen, tier, verdict, need, free }
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
}

fn gguf_shape(hub: &Hub, repo: &str, g: &Grouped) -> Result<Shape, String> {
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
            let anatomy = ModelAnatomy::new(&spec, 0, header.count(&format!("{arch}.context_length")));
            Ok(Shape { spec, anatomy })
        })();
        match attempt {
            Ok(shape) => return Ok(shape),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn config_shape(hub: &Hub, repo: &str) -> (Result<Shape, String>, Option<String>) {
    let text = match hub.text(repo, "config.json") {
        Ok(t) => t,
        Err(e) => return (Err(e), None),
    };
    let shape = (|| {
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let support = library::specrt_support(&v);
        if !support.supported {
            return Err(support.reason);
        }
        let d = fit::describe_config_json(repo, &text)?;
        Ok(Shape { spec: d.spec, anatomy: d.anatomy })
    })();
    (shape, Some(text))
}

pub fn inspect(hub: &Hub, repo: &str) -> Result<RepoInfo, String> {
    let files = hub.files(repo)?;
    let has_gguf = files.iter().any(|f| f.path.ends_with(".gguf") && !f.path.contains("mmproj"));
    if has_gguf {
        let grouped = group(&files, None);
        let shape = gguf_shape(hub, repo, &grouped);
        return Ok(RepoInfo { repo: repo.to_string(), grouped, shape });
    }
    if !files.iter().any(|f| f.path.ends_with(".safetensors")) {
        return Err("no GGUF or safetensors files".into());
    }
    let (shape, text) = config_shape(hub, repo);
    Ok(RepoInfo { repo: repo.to_string(), grouped: group(&files, text.as_deref()), shape })
}

pub fn inspect_all(hub: &Hub, repos: &[String]) -> Vec<(String, Result<RepoInfo, String>)> {
    std::thread::scope(|s| {
        let jobs: Vec<_> = repos.iter().map(|r| s.spawn(move || (r.clone(), inspect(hub, r)))).collect();
        jobs.into_iter().filter_map(|j| j.join().ok()).collect()
    })
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
    pub unsupported: Option<String>,
    pub ranked: Vec<Scored>,
}

impl RepoPlan {
    pub fn best(&self) -> Option<&Scored> {
        self.ranked.first()
    }

    pub fn key(&self) -> (u8, u8, std::cmp::Reverse<u32>, u64) {
        match self.best() {
            Some(b) if self.unsupported.is_none() => b.key(),
            _ => (3, 3, std::cmp::Reverse(0), u64::MAX),
        }
    }
}

pub fn plan(info: &RepoInfo, free: Option<u64>, seqs: usize) -> RepoPlan {
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
                    (Some(assess(&s.spec, &s.anatomy, v.size, free, seqs)), bits)
                }
                Err(_) => (None, v.bits),
            };
            Scored { variant: Variant { bits, ..v.clone() }, runnable, fit }
        })
        .collect();
    ranked.sort_by_key(|s| s.key());
    RepoPlan { repo: info.repo.clone(), unsupported: info.shape.as_ref().err().cloned(), ranked }
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
        let a = assess(&s.spec, &s.anatomy, weights, Some(free), 1);
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
        let roomy = assess(&s.spec, &s.anatomy, 5 * GB, Some(80 * GB), 1);
        assert_eq!((roomy.chosen.kv, roomy.tier, roomy.verdict), (Kv::F16, 0, Verdict::Fits));
        assert!(roomy.full());
        let f16_need = Kv::F16.bytes_per_token(&s.spec) * 131072;
        let mixed_need = Kv::MixedFp8.bytes_per_token(&s.spec) * 131072;
        let free = (((5 * GB + mixed_need + GB) as f64) / fit::GPU_UTILIZATION) as u64;
        assert!(f16_need > mixed_need + GB);
        let tight = assess(&s.spec, &s.anatomy, 5 * GB, Some(free), 1);
        assert_eq!((tight.chosen.kv, tight.tier, tight.verdict), (Kv::MixedFp8, 1, Verdict::Fits));
        assert!(tight.full());
    }

    #[test]
    fn nothing_fits_is_reported_with_what_the_smallest_variant_needs() {
        let s = shape();
        let a = assess(&s.spec, &s.anatomy, 30 * GB, Some(8 * GB), 1);
        assert_eq!((a.verdict, a.tier, a.chosen.context), (Verdict::DoesNotFit, 3, 0));
        assert!(a.need > 30 * GB && a.free == Some(8 * GB));
        let unknown = assess(&s.spec, &s.anatomy, 5 * GB, None, 1);
        assert_eq!(unknown.verdict, Verdict::Unknown);
    }

    fn info(variants: &[(&str, u64)]) -> RepoInfo {
        let files: Vec<RepoFile> = variants.iter().map(|(l, size)| RepoFile::new(&format!("m-{l}.gguf"), *size)).collect();
        RepoInfo { repo: "acme/m".into(), grouped: group(&files, None), shape: Ok(shape()) }
    }

    #[test]
    fn ranking_prefers_full_f16_then_full_quantized_then_widest_and_takes_the_highest_bits_within_a_tier() {
        let all = [("Q4_K_M", 5 * GB), ("Q5_K_M", 6 * GB), ("Q6_K", 7 * GB), ("Q8_0", 9 * GB)];
        let plan_big = plan(&info(&all), Some(80 * GB), 1);
        assert_eq!(plan_big.best().unwrap().variant.label, "Q8_0");
        assert_eq!(plan_big.ranked.iter().map(|s| s.variant.label.as_str()).collect::<Vec<_>>(), ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]);
        let s = shape();
        let mixed_need = Kv::Fp8.bytes_per_token(&s.spec) * 131072;
        let free = (((9 * GB + mixed_need + GB / 4) as f64) / fit::GPU_UTILIZATION) as u64;
        let squeezed = plan(&info(&all), Some(free), 1);
        let best = squeezed.best().unwrap();
        assert_eq!((best.variant.label.as_str(), best.fit.as_ref().unwrap().chosen.kv, best.fit.as_ref().unwrap().tier), ("Q8_0", Kv::Fp8, 1));
        let free = (((7 * GB + mixed_need + GB / 4) as f64) / fit::GPU_UTILIZATION) as u64;
        let mid = plan(&info(&all), Some(free), 1);
        assert_eq!(mid.best().unwrap().variant.label, "Q6_K");
        let tiers: Vec<u8> = mid.ranked.iter().map(|s| s.fit.as_ref().unwrap().tier).collect();
        assert!(tiers.windows(2).all(|w| w[0] <= w[1]), "{tiers:?}");
        let tiny = plan(&info(&all), Some(7 * GB + GB / 2), 1);
        let best = tiny.best().unwrap();
        assert_eq!(best.variant.label, "Q5_K_M", "only the best reduced-context tier that fits at all: {:?}", tiny.ranked.iter().map(|s| (&s.variant.label, s.fit.as_ref().unwrap().tier)).collect::<Vec<_>>());
        assert_eq!(best.fit.as_ref().unwrap().tier, 2);
    }

    #[test]
    fn unsupported_quant_types_and_incomplete_splits_are_never_the_best() {
        let files = files_1017();
        let info = RepoInfo { repo: "x/y".into(), grouped: group(&files, None), shape: Ok(shape()) };
        let p = plan(&info, Some(500 * GB), 1);
        let best = p.best().unwrap();
        assert!(best.variant.label == "IQ4_NL" || best.runnable.is_err() || !best.variant.complete());
        assert!(p.ranked.iter().all(|s| s.runnable.is_err()), "IQ quants are not supported by specrt");
        let unsupported = RepoInfo { shape: Err("model_type deepseek_v4 is not supported".into()), ..info };
        let p = plan(&unsupported, Some(500 * GB), 1);
        assert_eq!(p.unsupported.as_deref(), Some("model_type deepseek_v4 is not supported"));
        assert!(p.best().is_some_and(|b| !b.downloadable()));
    }

    #[test]
    fn repos_rank_by_their_best_variants_fit_and_unsupported_ones_go_last() {
        let mk = |name: &str, free: u64, size: u64| {
            let mut i = info(&[("Q4_K_M", size)]);
            i.repo = name.into();
            plan(&i, Some(free), 1)
        };
        let unsupported = {
            let mut i = info(&[("Q4_K_M", GB)]);
            i.repo = "z/unsupported".into();
            i.shape = Err("nope".into());
            plan(&i, Some(80 * GB), 1)
        };
        let ranked = rank_repos(vec![unsupported, mk("a/too-big", 80 * GB, 90 * GB), mk("b/ok", 80 * GB, 5 * GB)]);
        assert_eq!(ranked.iter().map(|p| p.repo.as_str()).collect::<Vec<_>>(), ["b/ok", "a/too-big", "z/unsupported"]);
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
    fn context_labels() {
        assert_eq!((context_label(131072).as_str(), context_label(49152).as_str(), context_label(4096).as_str(), context_label(512).as_str()), ("128k", "48k", "4k", "512"));
    }
}
