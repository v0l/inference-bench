use serde_json::Value;
use specrt::chat::ChatTemplate;
use specrt::spec::Spec;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

const MEDIA_KEYS: [&str; 2] = ["vision_config", "audio_config"];
const CARD_SHARE: f64 = 0.9;
const FILE_THRESHOLD: u64 = 1_000_000;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn popular() -> PathBuf {
    root().join("tests/fixtures/popular")
}

fn cache() -> PathBuf {
    std::env::var_os("POPULAR_CACHE").map(PathBuf::from).unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/specrt-popular"))
}

fn slug(repo: &str) -> String {
    repo.replace('/', "__")
}

fn committed(repo: &str, name: &str) -> Option<String> {
    std::fs::read_to_string(popular().join(slug(repo)).join(name)).ok()
}

fn fetched(repo: &str, name: &str) -> Option<String> {
    committed(repo, name).or_else(|| std::fs::read_to_string(cache().join("files").join(slug(repo)).join(name)).ok())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Loads,
    TextOnly,
    Offload,
    NoFit,
    UnknownQuant,
    Refused,
    Unchecked,
}

impl Class {
    const ALL: [Class; 7] = [Class::Loads, Class::TextOnly, Class::Offload, Class::NoFit, Class::UnknownQuant, Class::Refused, Class::Unchecked];

    fn label(self) -> &'static str {
        match self {
            Class::Loads => "LOADS",
            Class::TextOnly => "LOADS-TEXT-ONLY",
            Class::Offload => "SUPPORTED-WITH-OFFLOAD",
            Class::NoFit => "DOES-NOT-FIT",
            Class::UnknownQuant => "UNKNOWN-QUANT",
            Class::Refused => "REFUSED",
            Class::Unchecked => "UNCHECKED",
        }
    }

    fn arch_ok(self) -> bool {
        matches!(self, Class::Loads | Class::TextOnly | Class::Offload | Class::NoFit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Family,
    Variant,
    Quant,
    Chat,
    Media,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Family => "new family spec",
            Kind::Variant => "attention, block or MoE variant",
            Kind::Quant => "quant format",
            Kind::Chat => "tokenizer or chat template",
            Kind::Media => "vision or audio input (text loads)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Gap {
    kind: Kind,
    key: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Verdict {
    class: Class,
    why: String,
    gap: Option<Gap>,
    offload: Option<Offload>,
}

impl Verdict {
    fn new(class: Class, why: impl Into<String>, gap: Option<Gap>) -> Self {
        Verdict { class, why: why.into(), gap, offload: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Offload {
    cards: u64,
    host_gb: f64,
    nvme_gb: f64,
    short_gb: f64,
    placement: Option<bool>,
}

impl Offload {
    fn tiers(&self) -> String {
        format!("{} cards plus {:.0} GB host plus {:.0} GB NVMe", self.cards, self.host_gb, self.nvme_gb)
    }

    fn placement_text(&self) -> &'static str {
        match self.placement {
            Some(true) => "planner accepts today",
            Some(false) if self.nvme_gb > 0.0 => "planner models cards and host only, the NVMe tier is the expert cache and layer spill path",
            Some(false) => "planner rejects today",
            None => "planner not run until the family exists",
        }
    }

    fn text(&self) -> String {
        if self.short_gb > 0.0 {
            format!("{}; does not fit this machine, short by {:.0} GB of NVMe", self.tiers(), self.short_gb)
        } else {
            format!("{}; {}", self.tiers(), self.placement_text())
        }
    }
}

struct Lab {
    key: String,
    name: String,
    orgs: Vec<String>,
    absent: Vec<(String, String)>,
    note: String,
}

fn labs() -> Vec<Lab> {
    let text = std::fs::read_to_string(popular().join("labs.toml")).unwrap();
    let t: toml::Table = toml::from_str(&text).unwrap();
    let strings = |v: &toml::Value| v.as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect::<Vec<_>>()).unwrap_or_default();
    t["lab"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| Lab {
            key: l["key"].as_str().unwrap().into(),
            name: l["name"].as_str().unwrap().into(),
            orgs: strings(&l["orgs"]),
            absent: l.get("absent").and_then(|a| a.as_array()).map(|a| a.iter().map(|x| (x["name"].as_str().unwrap().to_string(), x["note"].as_str().unwrap().to_string())).collect()).unwrap_or_default(),
            note: l.get("note").and_then(|n| n.as_str()).unwrap_or("").to_string(),
        })
        .collect()
}

fn excluded() -> BTreeSet<String> {
    let Ok(text) = std::fs::read_to_string(popular().join("gaps.toml")) else { return BTreeSet::new() };
    let t: toml::Table = toml::from_str(&text).unwrap();
    t.get("excluded").and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn is_excluded(row: &Row, out: &BTreeSet<String>) -> bool {
    out.contains(row.s("base")) || out.contains(row.repo())
}

struct Row<'a> {
    m: &'a Value,
}

impl<'a> Row<'a> {
    fn s(&self, k: &str) -> &'a str {
        self.m[k].as_str().unwrap_or("")
    }
    fn repo(&self) -> &'a str {
        self.s("repo")
    }
    fn downloads(&self) -> u64 {
        self.m["downloads"].as_u64().unwrap_or(0)
    }
    fn size(&self) -> Option<f64> {
        self.m["size_gb"].as_f64()
    }
    fn is_gguf(&self) -> bool {
        self.s("format") == "GGUF"
    }
    fn has_file(&self, name: &str) -> bool {
        self.m["files"].get(name).is_some()
    }
    fn file_committed(&self, name: &str) -> bool {
        self.m["files"][name]["committed"].as_bool() == Some(true)
    }
}

fn error_text(e: anyhow::Error) -> String {
    format!("{e:#}").replace('\n', " ")
}

fn serving(v: &Value) -> Result<Spec, String> {
    Spec::from_hf_config_serving(&v.to_string()).map_err(error_text)
}

fn without(v: &Value, keys: &[&str]) -> Value {
    let mut out = v.clone();
    if let Some(o) = out.as_object_mut() {
        for k in keys {
            o.remove(*k);
        }
    }
    out
}

fn quant_key(label: &str) -> String {
    let re = fancy_regex::Regex::new(r"[- ]g\d+|-None| gNone").unwrap();
    format!("{} weights", re.replace_all(label, ""))
}

fn quant_gap(msg: &str, label: &str) -> String {
    let core = msg.rsplit(": ").next().unwrap_or(msg);
    if core.starts_with("quantization {") || core.contains("compressed-tensors group") {
        return quant_key(label);
    }
    let digits = fancy_regex::Regex::new(r"\b\d+\b").unwrap();
    digits.replace_all(core, "N").to_string()
}

fn norm_arch(name: &str) -> String {
    let lower: String = name.to_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let v = fancy_regex::Regex::new(r"v(\d)").unwrap();
    v.replace_all(&lower, "$1").to_string()
}

fn gguf_family_by_alias(arch: &str) -> Option<String> {
    let want = norm_arch(arch);
    specrt::family::active().families().iter().find(|f| f.matches.model_types.iter().any(|m| {
        let n = norm_arch(m);
        want == n || want == format!("{n}moe")
    })).map(|f| f.name.clone())
}

fn refusal_gap(msg: &str) -> Gap {
    let family = fancy_regex::Regex::new(r"model_type (\S+) is not supported").unwrap();
    if let Ok(Some(c)) = family.captures(msg) {
        return Gap { kind: Kind::Family, key: format!("model_type {}", &c[1]) };
    }
    let core = msg.rsplit(": ").next().unwrap_or(msg).to_string();
    if msg.contains("quantization") || msg.contains("compressed-tensors") {
        return Gap { kind: Kind::Quant, key: core };
    }
    Gap { kind: Kind::Variant, key: core }
}

fn describe(spec: &Spec) -> String {
    let quant = serde_json::to_value(spec.mlp.quant.unwrap_or(spec.quant)).ok().and_then(|q| q.as_str().map(String::from)).unwrap_or_default();
    let mut kinds: BTreeSet<String> = BTreeSet::new();
    for i in 0..spec.layers.count {
        kinds.insert(serde_json::to_value(spec.block(i)).ok().and_then(|b| b.as_str().map(String::from)).unwrap_or_default());
    }
    let mut out = format!("{} layers ({}), hidden {}", spec.layers.count, kinds.into_iter().collect::<Vec<_>>().join("+"), spec.dims.hidden);
    if let Some(m) = &spec.moe {
        write!(out, ", {} experts top {}", m.experts, m.active.unwrap_or(0)).unwrap();
    }
    if spec.mixer.is_some() {
        out.push_str(", ssm mixer");
    }
    if spec.mtp.is_some() {
        out.push_str(", mtp");
    }
    if spec.vision.is_some() {
        out.push_str(", vision");
    }
    write!(out, ", {quant}").unwrap();
    out
}

fn classify_config(json: &str, row: &Row) -> Verdict {
    let Ok(v) = serde_json::from_str::<Value>(json) else { return Verdict::new(Class::Unchecked, "config.json does not parse", None) };
    let media: Vec<&str> = MEDIA_KEYS.iter().copied().filter(|k| v.get(*k).is_some_and(|x| !x.is_null())).collect();
    let quantised = v.get("quantization_config").is_some_and(|q| !q.is_null());
    let hf_quant_only = !quantised && row.has_file("hf_quant_config.json");
    let label = row.s("quant");
    let core = without(&v, &[&media[..], &["quantization_config"][..]].concat());
    match serving(&v) {
        Ok(spec) if hf_quant_only => Verdict::new(
            Class::UnknownQuant,
            format!("weights are {label} but config.json has no quantization_config, and specrt does not read hf_quant_config.json ({})", describe(&spec)),
            Some(Gap { kind: Kind::Quant, key: "quantisation declared only in hf_quant_config.json".into() }),
        ),
        Ok(spec) => {
            let unread: Vec<&str> = media.iter().copied().filter(|k| *k != "vision_config" || spec.vision.is_none()).collect();
            if unread.is_empty() {
                Verdict::new(Class::Loads, describe(&spec), None)
            } else {
                let key = format!("{} not read", unread.join(" and "));
                Verdict::new(Class::TextOnly, format!("{key}; {}", describe(&spec)), Some(Gap { kind: Kind::Media, key }))
            }
        }
        Err(full) => {
            let text = (!media.is_empty()).then(|| without(&v, &media));
            if let Some(Ok(spec)) = text.as_ref().map(serving) {
                let key = refusal_gap(&full).key;
                return Verdict::new(Class::TextOnly, format!("{} refused: {full}; {}", media.join(" and "), describe(&spec)), Some(Gap { kind: Kind::Media, key }));
            }
            match serving(&core) {
                Ok(spec) if quantised => Verdict::new(Class::UnknownQuant, format!("{full} ({})", describe(&spec)), Some(Gap { kind: Kind::Quant, key: quant_gap(&full, label) })),
                Ok(spec) => Verdict::new(Class::Refused, format!("{full} ({})", describe(&spec)), Some(refusal_gap(&full))),
                Err(arch) => match inner_text_model(&v) {
                    Some((outer, inner, spec)) => Verdict::new(
                        Class::Refused,
                        format!("{arch}; its text model ({inner}) passes on its own ({})", describe(&spec)),
                        Some(Gap { kind: Kind::Family, key: format!("model_type {outer} (wraps {inner}, which specrt runs)") }),
                    ),
                    None => Verdict::new(Class::Refused, arch.clone(), Some(refusal_gap(&arch))),
                },
            }
        }
    }
}

fn inner_text_model(v: &Value) -> Option<(String, String, Spec)> {
    let outer = v["model_type"].as_str()?.to_string();
    let text = v.get("text_config").filter(|t| t.is_object())?;
    let inner = text["model_type"].as_str()?.to_string();
    if inner == outer || specrt::family::active().for_model_type(&outer).is_some() {
        return None;
    }
    serving(text).ok().map(|spec| (outer, inner, spec))
}

fn gguf_header(arch: &str) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    let key = "general.architecture";
    b.extend((key.len() as u64).to_le_bytes());
    b.extend(key.as_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend((arch.len() as u64).to_le_bytes());
    b.extend(arch.as_bytes());
    b
}

fn readable_ggml() -> BTreeSet<String> {
    (0..64).map(specrt::gguf::GgmlType::from_id).filter(|t| t.block().is_some()).map(|t| t.name()).collect()
}

fn ggml_main_type(label: &str) -> String {
    let k = fancy_regex::Regex::new(r"^(Q\d_K)(_.*)?$").unwrap();
    let iq = fancy_regex::Regex::new(r"^(IQ\d)_M$").unwrap();
    if let Ok(Some(c)) = k.captures(label) {
        return c[1].to_string();
    }
    if let Ok(Some(c)) = iq.captures(label) {
        return format!("{}_S", &c[1]);
    }
    label.split("_MOE").next().unwrap_or(label).to_string()
}

fn classify_gguf(row: &Row) -> Verdict {
    let arch = row.m["gguf_arch"].as_str().or(row.m["model_type"].as_str()).unwrap_or("");
    if arch.is_empty() {
        return Verdict::new(Class::Unchecked, "the hub lists no GGUF architecture", None);
    }
    if arch == "clip" {
        return Verdict::new(Class::Unchecked, "the hub reports the vision projector's architecture (clip), not the model's", None);
    }
    let checked = specrt::gguf::Gguf::from_header(gguf_header(arch)).and_then(|g| g.spec());
    if specrt::family::active().for_gguf(arch).is_none() {
        let msg = checked.err().map(error_text).unwrap_or_default();
        let gap = match gguf_family_by_alias(arch) {
            Some(_) => Gap { kind: Kind::Variant, key: "GGUF tensor and metadata names for families that already load from safetensors".into() },
            None => Gap { kind: Kind::Family, key: format!("GGUF architecture {arch}") },
        };
        return Verdict::new(Class::Refused, msg, Some(gap));
    }
    let readable = readable_ggml();
    let labels: Vec<String> = row.m["gguf_labels"].as_array().map(|a| a.iter().filter_map(|l| l.as_str().map(String::from)).collect()).unwrap_or_default();
    let (read, unread): (Vec<&String>, Vec<&String>) = labels.iter().partition(|l| readable.contains(&ggml_main_type(l)));
    let join = |v: &[&String]| v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
    if labels.is_empty() {
        return Verdict::new(Class::Unchecked, format!("GGUF {arch}: no file name names a GGUF type, so the type is unknown without reading a header"), None);
    }
    if read.is_empty() {
        return Verdict::new(Class::UnknownQuant, format!("GGUF {arch}: no file in a type specrt reads ({})", join(&unread)), Some(Gap { kind: Kind::Quant, key: "GGUF types".into() }));
    }
    let mut why = format!("GGUF {arch}; reads {}", join(&read));
    if !unread.is_empty() {
        write!(why, "; not {}", join(&unread)).unwrap();
    }
    Verdict::new(Class::Loads, why, None)
}

struct Machine {
    card_gb: f64,
    cards: u64,
    ram_available_gb: f64,
    host_share: f64,
    host_bandwidth_gbs: f64,
    nvme_free_gb: f64,
}

fn machine(man: &Value) -> Machine {
    let m = &man["machine"];
    Machine {
        card_gb: man["card_gb"].as_f64().unwrap(),
        cards: m["cards"].as_u64().unwrap(),
        ram_available_gb: m["ram_available_gb"].as_f64().unwrap(),
        host_share: m["host_share"].as_f64().unwrap(),
        host_bandwidth_gbs: m["host_bandwidth_gbs"].as_f64().unwrap(),
        nvme_free_gb: m["nvme_free_gb"].as_f64().unwrap(),
    }
}

fn offload_need(gb: f64, m: &Machine, spec: Option<&Spec>) -> Offload {
    let usable = m.card_gb * CARD_SHARE;
    let cards = ((gb / usable).ceil() as u64).clamp(1, m.cards);
    let rest = (gb - cards as f64 * usable).max(0.0);
    let host_budget = m.ram_available_gb * m.host_share;
    let host_gb = rest.min(host_budget);
    let nvme_gb = rest - host_gb;
    let short_gb = (nvme_gb - m.nvme_free_gb).max(0.0);
    let placement = spec.map(|s| place_on_desktop(s, gb, m, host_budget));
    Offload { cards, host_gb, nvme_gb, short_gb, placement }
}

fn place_on_desktop(spec: &Spec, gb: f64, m: &Machine, host_budget_gb: f64) -> bool {
    use specrt::placement::{DeviceConfig, LinkConfig, OffloadConfig, PlacementConfig};
    let gib = (1u64 << 30) as f64;
    let devices = (0..m.cards as usize)
        .map(|i| DeviceConfig {
            name: format!("desktop/gpu{i}"),
            memory_gib: m.card_gb * 1e9 / gib,
            bandwidth_gbs: 1792.0,
            tflops: 500.0,
            ordinal: Some(i),
            address: None,
            reserve_gib: None,
            link: LinkConfig::default(),
            offload: Some(OffloadConfig { host_gib: host_budget_gb * 1e9 / gib / m.cards as f64, bandwidth_gbs: m.host_bandwidth_gbs }),
            host: Some("desktop".into()),
            peer: Some(LinkConfig::default()),
            wgpu: false,
        })
        .collect();
    let config = PlacementConfig { devices, weight_bytes: Some((gb * 1e9) as u64), ..PlacementConfig::default() };
    config.place(spec).is_ok()
}

fn sized(v: Verdict, row: &Row, m: &Machine, config: Option<&str>) -> Verdict {
    let Some(gb) = row.size().filter(|gb| *gb > m.card_gb * CARD_SHARE) else { return v };
    match v.class {
        Class::Loads | Class::TextOnly => {
            let spec = config.and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|j| serving(&j).ok());
            let need = offload_need(gb, m, spec.as_ref());
            let class = if need.short_gb > 0.0 { Class::NoFit } else { Class::Offload };
            let why = format!("{gb} GB of weights, {}; {}", need.text(), v.why);
            Verdict { class, why, gap: v.gap, offload: Some(need) }
        }
        Class::Refused | Class::UnknownQuant => {
            let need = offload_need(gb, m, None);
            let why = format!("{}; {gb} GB of weights, then offload: {}", v.why, need.text());
            Verdict { why, offload: Some(need), ..v }
        }
        _ => v,
    }
}

fn classify(row: &Row, m: &Machine, committed_only: bool) -> Option<Verdict> {
    let mut config_text = None;
    let v = if row.is_gguf() {
        classify_gguf(row)
    } else {
        let config = if committed_only { committed(row.repo(), "config.json") } else { fetched(row.repo(), "config.json") };
        match config {
            Some(c) => {
                let v = classify_config(&c, row);
                config_text = Some(c);
                v
            }
            None if committed_only && row.has_file("config.json") => return None,
            None => Verdict::new(Class::Unchecked, row.m["config_from"].as_str().unwrap_or("no config.json"), None),
        }
    };
    Some(sized(v, row, m, config_text.as_deref()))
}

fn chat(row: &Row) -> (String, Option<Gap>) {
    if row.is_gguf() {
        return ("in the GGUF".into(), None);
    }
    if row.m["tokenizer_json"].as_bool() == Some(false) {
        return ("no tokenizer.json".into(), Some(Gap { kind: Kind::Chat, key: "no tokenizer.json (specrt reads only tokenizer.json)".into() }));
    }
    let Some(tc) = fetched(row.repo(), "tokenizer_config.json") else { return ("no tokenizer_config.json".into(), None) };
    let jinja = fetched(row.repo(), "chat_template.jinja");
    match ChatTemplate::from_files(&tc, jinja.as_deref()) {
        Ok(Some(t)) => match t.render_user("hi") {
            Ok(_) => (format!("ok, tools {:?}", t.tool_format()), None),
            Err(e) => {
                let msg = error_text(e);
                if msg.contains("invalid operation") && t.source().contains("raise_exception") {
                    return (format!("the template refuses a plain user turn by design: {}", short(&msg, 60)), None);
                }
                let key = msg.split(" (in ").next().unwrap_or(&msg).trim_start_matches("chat template: ").to_string();
                (format!("fails: {msg}"), Some(Gap { kind: Kind::Chat, key: format!("chat template: {key}") }))
            }
        },
        Ok(None) => ("no template".into(), None),
        Err(e) => (format!("fails: {}", error_text(e)), Some(Gap { kind: Kind::Chat, key: "tokenizer_config.json does not parse".into() })),
    }
}

fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(popular().join("manifest.json")).unwrap()).unwrap()
}

fn issues() -> BTreeMap<String, u64> {
    let Ok(text) = std::fs::read_to_string(popular().join("gaps.toml")) else { return BTreeMap::new() };
    let t: toml::Table = toml::from_str(&text).unwrap();
    t.get("issues").and_then(|i| i.as_table()).map(|i| i.iter().filter_map(|(k, v)| Some((k.clone(), v.as_integer()? as u64))).collect()).unwrap_or_default()
}

fn family_of(model_type: &str) -> Option<String> {
    let reg = specrt::family::active();
    reg.for_model_type(model_type).or_else(|| reg.for_gguf(model_type)).map(|f| f.name.clone())
}

fn fixture_families() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for e in std::fs::read_dir(root().join("tests/fixtures")).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("tiny-") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(e.path().join("config.json")) else { continue };
        let v: Value = serde_json::from_str(&text).unwrap();
        if let Some(f) = v["model_type"].as_str().and_then(family_of) {
            out.insert(f);
        }
    }
    out
}

fn weights(rows: &[Row]) -> Vec<u64> {
    let mut others: BTreeMap<&str, u64> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.s("role") != "base") {
        *others.entry(r.s("base")).or_default() += r.downloads();
    }
    rows.iter()
        .map(|r| match r.s("role") {
            "base" => r.m["group_downloads"].as_u64().unwrap_or(0).saturating_sub(others.get(r.s("base")).copied().unwrap_or(0)).max(r.downloads()),
            _ => r.downloads(),
        })
        .collect()
}

fn unread_of<'a>(mut rows: impl Iterator<Item = &'a Row<'a>>) -> Vec<String> {
    rows.find_map(|r| fetched(r.repo(), "config.json")).map(|c| specrt::spec::unread_hf_keys(&c).into_iter().map(|(k, _)| k).collect()).unwrap_or_default()
}

fn lab_class_cell(v: &Verdict, issue: &str) -> String {
    match (&v.offload, v.class) {
        (Some(o), Class::Refused | Class::UnknownQuant) => {
            let blocker = if issue.is_empty() { " (no issue yet)".to_string() } else { format!(" {issue}") };
            let what = if v.class == Class::Refused { "family" } else { "quantisation support" };
            format!("needs {what}{blocker}, then offload: {}", o.text())
        }
        (Some(o), _) => format!("{}: {}", v.class.label(), o.text()),
        _ => v.class.label().to_string(),
    }
}

fn own_row(lab: &Lab, r: &Row, skip: &BTreeSet<String>) -> bool {
    r.s("lab") == lab.key && !is_excluded(r, skip) && lab.orgs.iter().any(|o| r.repo().split('/').next().is_some_and(|x| x.eq_ignore_ascii_case(o)))
}

fn main_labs(rows: &[Row], verdicts: &[Verdict], filed: &BTreeMap<String, u64>, fold: &dyn Fn(&Gap) -> Gap, mach: &Machine, man: &Value, skip: &BTreeSet<String>) -> String {
    let mut out = String::new();
    let since = man["labs_since"].as_str().unwrap_or("");
    writeln!(out, "## Main labs\n").unwrap();
    writeln!(out, "Source: `tests/fixtures/popular/labs.toml`. Per lab: every model its hub orgs released since {since} (9 months) plus every family the lab still ships, one row per repo of the lab itself with its release date. Quantised copies by other orgs sit under the source lab and are counted, not listed. A model over one card is never refused for size: it reads SUPPORTED-WITH-OFFLOAD, or `needs family #N, then offload: ...` when the architecture is also missing. Machine read for the offload numbers: {:.0} GB free host RAM ({:.0}% usable by the host tier), {:.0} GB free NVMe, {} cards of {} GB.\n", mach.ram_available_gb, mach.host_share * 100.0, mach.nvme_free_gb, mach.cards, mach.card_gb).unwrap();
    for lab in labs() {
        let own: Vec<usize> = (0..rows.len()).filter(|&i| own_row(&lab, &rows[i], skip)).collect();
        let copies: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].s("lab") == lab.key && !own.contains(&i)).collect();
        let loading = |i: usize| matches!(verdicts[i].class, Class::Loads | Class::TextOnly | Class::Offload);
        let status = if own.is_empty() {
            "no public text model".to_string()
        } else if own.iter().all(|&i| loading(i)) {
            "GREEN: every model loads".to_string()
        } else {
            let blocked = own.iter().filter(|&&i| !loading(i)).count();
            format!("{blocked} of {} models are blocked, the rest load", own.len())
        };
        writeln!(out, "### {} ({})\n", lab.name, status).unwrap();
        writeln!(out, "| model | released | size GB | arch | class | gap | issue |").unwrap();
        writeln!(out, "|---|---|---|---|---|---|---|").unwrap();
        if own.is_empty() {
            writeln!(out, "| none | | | | no public text model | | |").unwrap();
        }
        let mut sorted = own.clone();
        sorted.sort_by(|&a, &b| rows[b].s("created").cmp(rows[a].s("created")).then(rows[a].repo().cmp(rows[b].repo())));
        for i in sorted {
            let r = &rows[i];
            let v = &verdicts[i];
            let gap = v.gap.as_ref().map(fold);
            let issue = gap.as_ref().and_then(|g| filed.get(&g.key)).map(|n| format!("#{n}")).unwrap_or_default();
            let key = gap.as_ref().map(|g| g.key.clone()).unwrap_or_default();
            let class = lab_class_cell(v, &issue);
            writeln!(out, "| {} | {} | {} | {} | {} | {} | {} |", r.repo(), r.s("created"), r.size().map(|s| s.to_string()).unwrap_or_default(), r.s("model_type"), cell(&short(&class, 220)), cell(&short(&key, 90)), issue).unwrap();
        }
        for (name, note) in &lab.absent {
            writeln!(out, "| {name} | | | | no public repo | {} | |", cell(note)).unwrap();
        }
        let loads = copies.iter().filter(|&&i| loading(i)).count();
        writeln!(out, "\nQuantised copies under this lab: {}, of which {} load or load text only.\n", copies.len(), loads).unwrap();
        if !lab.note.is_empty() {
            writeln!(out, "{}\n", lab.note).unwrap();
        }
    }
    out
}

fn lab_gaps(rows: &[Row], verdicts: &[Verdict], filed: &BTreeMap<String, u64>, fold: &dyn Fn(&Gap) -> Gap, skip: &BTreeSet<String>) -> String {
    let mut out = String::new();
    writeln!(out, "## Gaps by lab priority\n").unwrap();
    writeln!(out, "Labs in the order of `labs.toml`, then by how many of that lab's own models a gap blocks. Downloads do not enter the order.\n").unwrap();
    writeln!(out, "| lab | gap | lab models blocked | models | issue |").unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    for lab in labs() {
        let mut by_gap: BTreeMap<String, Vec<&str>> = BTreeMap::new();
        for i in (0..rows.len()).filter(|&i| own_row(&lab, &rows[i], skip)) {
            if let Some(g) = &verdicts[i].gap {
                by_gap.entry(fold(g).key).or_default().push(rows[i].repo());
            }
        }
        let mut list: Vec<(String, Vec<&str>)> = by_gap.into_iter().collect();
        list.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        for (key, models) in list {
            let issue = filed.get(&key).map(|n| format!("#{n}")).unwrap_or_default();
            let shown = models.iter().take(4).copied().collect::<Vec<_>>().join(", ");
            let more = if models.len() > 4 { format!(" and {} more", models.len() - 4) } else { String::new() };
            writeln!(out, "| {} | {} | {} | {}{} | {} |", lab.name, cell(&short(&key, 100)), models.len(), cell(&shown), more, issue).unwrap();
        }
    }
    out.push('\n');
    out
}

fn traffic_section(rows: &[Row], verdicts: &[Verdict], skip: &BTreeSet<String>) -> String {
    let mut out = String::new();
    writeln!(out, "## Test and tutorial traffic\n").unwrap();
    writeln!(out, "Listed in `gaps.toml` under `excluded`. These repos are downloaded by CI, tutorials and smoke tests, not by people running a model, so they stay out of the ranked tables. Counts are the hub's last-30-day downloads.\n").unwrap();
    writeln!(out, "| model | rows | downloads | class | gap |").unwrap();
    writeln!(out, "|---|---|---|---|---|").unwrap();
    for name in skip {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].s("base") == name || rows[i].repo() == name).collect();
        if idx.is_empty() {
            continue;
        }
        let downloads: u64 = idx.iter().map(|&i| rows[i].downloads()).sum();
        let base = idx.iter().copied().find(|&i| rows[i].repo() == name).unwrap_or(idx[0]);
        writeln!(out, "| {name} | {} | {} | {} | {} |", idx.len(), downloads, verdicts[base].class.label(), cell(&verdicts[base].gap.as_ref().map(|g| g.key.clone()).unwrap_or_default())).unwrap();
    }
    out.push('\n');
    out
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

fn millions(n: u64) -> String {
    format!("{:.2} M", n as f64 / 1e6)
}

struct Report {
    tsv: String,
    doc: String,
    gaps: Value,
}

fn report() -> Report {
    let man = manifest();
    let mach = machine(&man);
    let rows: Vec<Row> = man["models"].as_array().unwrap().iter().map(|m| Row { m }).collect();
    let weight = weights(&rows);
    let tiny = fixture_families();
    let filed = issues();
    let skip = excluded();
    let verdicts: Vec<Verdict> = rows.iter().map(|r| classify(r, &mach, false).unwrap()).collect();
    let chats: Vec<(String, Option<Gap>)> = rows.iter().map(chat).collect();

    let mut tsv = String::from("repo\tclass\tgap\twhy\n");
    for (r, v) in rows.iter().zip(&verdicts) {
        let gap = v.gap.as_ref().map(|g| g.key.as_str()).unwrap_or("");
        writeln!(tsv, "{}\t{}\t{}\t{}", r.repo(), v.class.label(), gap, v.why).unwrap();
    }

    let mut gaps: BTreeMap<Gap, (u64, Vec<usize>)> = BTreeMap::new();
    let family_keys: BTreeMap<String, String> = verdicts
        .iter()
        .filter_map(|v| v.gap.as_ref())
        .filter(|g| g.kind == Kind::Family)
        .filter_map(|g| g.key.strip_prefix("model_type ").map(|m| (norm_arch(m), g.key.clone())))
        .collect();
    let fold = |g: &Gap| -> Gap {
        let alias = g.key.strip_prefix("GGUF architecture ").and_then(|a| {
            let n = norm_arch(a);
            family_keys.get(&n).or_else(|| n.strip_suffix("moe").and_then(|b| family_keys.get(b)))
        });
        match alias {
            Some(k) => Gap { kind: Kind::Family, key: k.clone() },
            None => g.clone(),
        }
    };
    for (i, v) in verdicts.iter().enumerate() {
        if is_excluded(&rows[i], &skip) {
            continue;
        }
        if let Some(g) = &v.gap {
            let e = gaps.entry(fold(g)).or_default();
            e.0 += weight[i];
            e.1.push(i);
        }
        if let (Some(g), true) = (&chats[i].1, v.class.arch_ok()) {
            let e = gaps.entry(g.clone()).or_default();
            e.0 += weight[i];
            e.1.push(i);
        }
    }
    let mut ranked: Vec<(&Gap, &(u64, Vec<usize>))> = gaps.iter().collect();
    ranked.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(b.0)));
    let gaps_json = Value::Array(
        ranked
            .iter()
            .map(|(g, (sum, idx))| {
                let members: Vec<Value> = idx
                    .iter()
                    .map(|&i| {
                        let why = verdicts[i].gap.as_ref().filter(|v| fold(v) == **g).map(|_| verdicts[i].why.clone()).unwrap_or_else(|| chats[i].0.clone());
                        serde_json::json!({"repo": rows[i].repo(), "downloads": weight[i], "class": verdicts[i].class.label(), "size_gb": rows[i].size(), "why": why})
                    })
                    .collect();
                serde_json::json!({"key": g.key, "kind": g.kind.label(), "downloads": sum, "rows": members, "unread_keys": unread_of(idx.iter().map(|&i| &rows[i]))})
            })
            .collect(),
    );

    let mut doc = String::new();
    let fetched_on = man["fetched"].as_str().unwrap();
    writeln!(doc, "# specrt model coverage\n").unwrap();
    writeln!(doc, "Fetched {fetched_on}. Generated by `crates/specrt/scripts/model_coverage.sh`; do not edit by hand.\n").unwrap();
    doc.push_str(&main_labs(&rows, &verdicts, &filed, &fold, &mach, &man, &skip));
    doc.push_str(&lab_gaps(&rows, &verdicts, &filed, &fold, &skip));
    doc.push_str(&traffic_section(&rows, &verdicts, &skip));
    writeln!(doc, "\"Popular\" means popular on Hugging Face on {fetched_on}: the most downloaded and the trending repos for the text-generation, image-text-to-text and any-to-any pipeline tags, as the hub API ranked them that day. Downloads are the hub's last-30-day counts. Nothing here was measured by loading weights: every class comes from the model's `config.json` (and the hub's file list for GGUF repos) run through specrt's own checks, `Spec::from_hf_config_serving`, `Vision::from_hf_config` and `Gguf::spec`, with the family specs embedded in this build.\n").unwrap();
    writeln!(doc, "Source: {}.\n", man["source"].as_str().unwrap()).unwrap();
    writeln!(doc, "Selection: {}. Each group keeps the original lab's repo (`base`) and the most downloaded NVFP4, FP8, AWQ, GPTQ and GGUF copy (`nvfp4`, `fp8`, `awq`, `gptq`, `gguf`), plus any copy already in the local hub cache (`disk`).\n", man["selection"].as_str().unwrap()).unwrap();
    writeln!(doc, "## Classes\n").unwrap();
    writeln!(doc, "- LOADS: the serving check passes and every input tower the config declares is read.").unwrap();
    writeln!(doc, "- LOADS-TEXT-ONLY: the text model passes; a vision or audio tower in the config is refused or not read, so only text input works.").unwrap();
    writeln!(doc, "- SUPPORTED-WITH-OFFLOAD: the architecture passes and the weights are more than {:.0}% of one {} GB card. This is not a refusal: placement spreads layers over the cards, then host RAM, then NVMe. The why column gives the cards, host RAM and NVMe the tiers need on this machine (read from the manifest's `machine` block) and whether `PlacementConfig::place` accepts it today with the desktop's device list.", CARD_SHARE * 100.0, mach.card_gb).unwrap();
    writeln!(doc, "- DOES-NOT-FIT: the architecture passes but the weights exceed every tier together on this machine; the why column says by how much.").unwrap();
    writeln!(doc, "- UNKNOWN-QUANT: the architecture passes without the quantisation, the quantisation format is refused.").unwrap();
    writeln!(doc, "- REFUSED: the architecture itself is missing or refused; the why column is the exact refusal text. A refused model that is also over one card reads `needs family #N, then offload: ...` in the Main labs tables. Size never refuses a model.").unwrap();
    writeln!(doc, "- UNCHECKED: nothing specrt could check was found: gated with no open mirror, or a GGUF repo whose hub metadata names only the vision projector.\n").unwrap();

    writeln!(doc, "## Counts\n").unwrap();
    writeln!(doc, "| class | base models | all rows |").unwrap();
    writeln!(doc, "|---|---|---|").unwrap();
    for c in Class::ALL {
        let all = verdicts.iter().filter(|v| v.class == c).count();
        let base = rows.iter().zip(&verdicts).filter(|(r, v)| v.class == c && r.s("role") == "base").count();
        writeln!(doc, "| {} | {base} | {all} |", c.label()).unwrap();
    }
    let base_total = rows.iter().filter(|r| r.s("role") == "base").count();
    writeln!(doc, "| total | {base_total} | {} |\n", rows.len()).unwrap();

    writeln!(doc, "## Ranked gaps\n").unwrap();
    writeln!(doc, "The same gaps as above, ordered by the summed downloads of the rows they block. This is the older view and it is not the queue order: the lab priority at the top of this section is, because downloads reward CI and tutorial traffic. A base row counts its whole group less the quantised copies listed beside it. Chat and tokenizer gaps are counted only on rows whose architecture passes. The issue column maps a gap to its tracker issue through `crates/specrt/tests/fixtures/popular/gaps.toml`; every gap over {} downloads has one.\n", millions(FILE_THRESHOLD)).unwrap();
    let unfiled: Vec<&str> = ranked.iter().filter(|(g, (sum, _))| *sum >= FILE_THRESHOLD && !filed.contains_key(&g.key)).map(|(g, _)| g.key.as_str()).collect();
    if !unfiled.is_empty() {
        writeln!(doc, "Gaps over the threshold with no issue yet: {}.\n", unfiled.join("; ")).unwrap();
    }
    writeln!(doc, "| # | kind | gap | blocked downloads | rows | models | issue |").unwrap();
    writeln!(doc, "|---|---|---|---|---|---|---|").unwrap();
    for (n, (g, (sum, idx))) in ranked.iter().enumerate() {
        let mut names: Vec<&str> = idx.iter().map(|&i| rows[i].repo()).collect();
        names.dedup();
        let shown = names.iter().take(4).copied().collect::<Vec<_>>().join(", ");
        let more = if names.len() > 4 { format!(" and {} more", names.len() - 4) } else { String::new() };
        let issue = filed.get(&g.key).map(|i| format!("#{i}")).unwrap_or_default();
        writeln!(doc, "| {} | {} | {} | {} | {} | {}{} | {} |", n + 1, g.kind.label(), cell(&g.key), millions(*sum), idx.len(), cell(&shown), more, issue).unwrap();
    }

    writeln!(doc, "\n## Quantisation formats in the safetensors indexes\n").unwrap();
    writeln!(doc, "Modules per weight layout, counted from each repo's `model.safetensors.index.json` (tensor names only). Read today means at least one repo holding that layout classifies as LOADS, LOADS-TEXT-ONLY or SUPPORTED-WITH-OFFLOAD.\n").unwrap();
    let mut formats: BTreeMap<String, (u64, usize, bool)> = BTreeMap::new();
    for (r, v) in rows.iter().zip(&verdicts) {
        if let Some(t) = r.m["tensor_formats"].as_object() {
            for (k, n) in t {
                let e = formats.entry(k.clone()).or_default();
                e.0 += n.as_u64().unwrap_or(0);
                e.1 += 1;
                e.2 |= v.class.arch_ok() && (k == "plain weight" || r.s("quant") != "none");
            }
        }
    }
    writeln!(doc, "| layout | modules | repos | read today |").unwrap();
    writeln!(doc, "|---|---|---|---|").unwrap();
    for (k, (n, repos, read)) in &formats {
        writeln!(doc, "| {k} | {n} | {repos} | {} |", if *read { "yes" } else { "no" }).unwrap();
    }
    let readable = readable_ggml();
    writeln!(doc, "\nGGUF tensor types specrt reads: {}. A GGUF file label counts as read when its main type is; K-quant mixes also hold other types, which only a real load checks.\n", readable.iter().cloned().collect::<Vec<_>>().join(", ")).unwrap();
    let mut by_quant: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (r, v) in rows.iter().zip(&verdicts).filter(|(r, _)| !r.is_gguf()) {
        let e = by_quant.entry(quant_key(r.s("quant"))).or_default();
        e.0 += 1;
        if v.class != Class::UnknownQuant && v.class != Class::Unchecked && v.gap.as_ref().is_none_or(|g| g.kind != Kind::Quant) {
            e.1 += 1;
        }
    }
    writeln!(doc, "Quantisation configs, from each `config.json` (`quantization_config`, or `hf_quant_config.json` when that is the only place it is declared):\n").unwrap();
    writeln!(doc, "| quantization | repos | not refused for the quantisation |").unwrap();
    writeln!(doc, "|---|---|---|").unwrap();
    for (k, (n, ok)) in &by_quant {
        writeln!(doc, "| {} | {n} | {ok} |", cell(k)).unwrap();
    }

    writeln!(doc, "\n## Needs a real load\n").unwrap();
    writeln!(doc, "A config check cannot prove these; they stay unproven until a real load (phase 2):\n").unwrap();
    writeln!(doc, "- Every LOADS row without a tiny fixture in its family: the config passes, but no logits check covers the family.").unwrap();
    writeln!(doc, "- GGUF rows: classified from the hub's architecture field and file names, never from the file header (no range requests). A K-quant mix may hold a type specrt does not read.").unwrap();
    writeln!(doc, "- Rows whose weights the hub sizes above one card: the planner's offload and multi-card paths decide those.\n").unwrap();

    writeln!(doc, "## Models\n").unwrap();
    writeln!(doc, "| model | role | downloads | size GB | arch | quant | class | why | family | tiny fixture | chat template | issue |").unwrap();
    writeln!(doc, "|---|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| rows[a].s("base").cmp(rows[b].s("base")).then((rows[a].s("role") != "base").cmp(&(rows[b].s("role") != "base"))).then(rows[a].repo().cmp(rows[b].repo())));
    let mut group_weight: BTreeMap<&str, u64> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        *group_weight.entry(r.s("base")).or_default() += weight[i];
    }
    order.sort_by(|&a, &b| {
        group_weight[rows[b].s("base")].cmp(&group_weight[rows[a].s("base")]).then(rows[a].s("base").cmp(rows[b].s("base"))).then((rows[a].s("role") != "base").cmp(&(rows[b].s("role") != "base"))).then(rows[a].repo().cmp(rows[b].repo()))
    });
    for i in order {
        let r = &rows[i];
        let v = &verdicts[i];
        let mt = r.s("model_type");
        let fam = family_of(mt);
        let fam_cell = fam.as_ref().map(|f| format!("`family-{f}.toml`")).unwrap_or_default();
        let fixture = match (&fam, v.class.arch_ok()) {
            (Some(f), true) => if tiny.contains(f) { "yes" } else { "no" },
            _ => "",
        };
        let size = r.size().map(|s| format!("{s}")).unwrap_or_default();
        let issue = v.gap.as_ref().map(fold).or(chats[i].1.clone().filter(|_| v.class.arch_ok())).and_then(|g| filed.get(&g.key)).map(|n| format!("#{n}")).unwrap_or_default();
        let config_note = r.m["config_from"].as_str().map(|c| format!(" (config from {c})")).unwrap_or_default();
        writeln!(
            doc,
            "| {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.repo(),
            cell(&config_note),
            r.s("role"),
            r.downloads(),
            size,
            mt,
            cell(r.s("quant")),
            v.class.label(),
            cell(&short(&v.why, 220)),
            fam_cell,
            fixture,
            cell(&short(&chats[i].0, 80)),
            issue
        )
        .unwrap();
    }
    Report { tsv, doc, gaps: gaps_json }
}

fn expected() -> BTreeMap<String, (String, String, String)> {
    let text = std::fs::read_to_string(popular().join("classes.tsv")).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            let mut p = l.splitn(4, '\t');
            let repo = p.next().unwrap().to_string();
            let class = p.next().unwrap().to_string();
            let gap = p.next().unwrap_or("").to_string();
            let why = p.next().unwrap_or("").to_string();
            (repo, (class, gap, why))
        })
        .collect()
}

fn section<'a>(doc: &'a str, title: &str) -> &'a str {
    let start = doc.find(&format!("## {title}")).unwrap_or_else(|| panic!("no section {title}"));
    let rest = &doc[start + 3..];
    let end = rest.find("\n## ").map(|e| e + 3).unwrap_or(rest.len());
    &doc[start..start + end]
}

#[test]
fn every_lab_has_a_model_row_or_an_explicit_no_public_model_row() {
    let man = manifest();
    let rows: Vec<Row> = man["models"].as_array().unwrap().iter().map(|m| Row { m }).collect();
    let skip = excluded();
    let doc = report().doc;
    let main = section(&doc, "Main labs");
    for lab in labs() {
        let heading = format!("### {} (", lab.name);
        let at = main.find(&heading).unwrap_or_else(|| panic!("lab {} has no table", lab.name));
        let table = &main[at..main[at..].find("\nQuantised copies").map(|e| at + e).unwrap_or(main.len())];
        let own = rows.iter().filter(|r| own_row(&lab, r, &skip)).count();
        if own == 0 {
            assert!(table.contains("no public text model"), "lab {} has no rows and no explicit row", lab.name);
        } else {
            assert!(table.lines().filter(|l| l.starts_with("| ") && !l.starts_with("| model")).count() >= own, "lab {} lost rows", lab.name);
        }
    }
    for key in ["zai", "deepseek", "kimi", "tencent", "xiaomi", "minimax", "alibaba", "mistral", "thinkingmachines", "meta", "nvidia", "google"] {
        assert!(labs().iter().any(|l| l.key == key), "labs.toml lacks {key}");
    }
}

#[test]
fn an_excluded_model_never_appears_in_the_ranked_tables() {
    let skip = excluded();
    assert!(skip.contains("openai-community/gpt2") && skip.contains("facebook/opt-125m"));
    let doc = report().doc;
    for title in ["Main labs", "Gaps by lab priority", "Ranked gaps"] {
        let body = section(&doc, title);
        for name in &skip {
            assert!(!body.contains(&format!("| {name} ")) && !body.contains(&format!(", {name},")) && !body.contains(&format!(", {name} ")), "{name} is in {title}");
        }
    }
    let traffic = section(&doc, "Test and tutorial traffic");
    for name in &skip {
        assert!(traffic.contains(&format!("| {name} |")), "{name} missing from the traffic section");
    }
}

#[test]
fn a_model_larger_than_one_card_is_refused_only_for_a_missing_architecture() {
    let man = manifest();
    let mach = machine(&man);
    let big = mach.card_gb * CARD_SHARE;
    let mut oversize = 0;
    for m in man["models"].as_array().unwrap() {
        let row = Row { m };
        if row.size().is_none_or(|gb| gb <= big) || (!row.is_gguf() && !row.file_committed("config.json")) {
            continue;
        }
        let Some(v) = classify(&row, &mach, true) else { continue };
        oversize += 1;
        if v.class == Class::Refused {
            let missing = v.gap.as_ref().is_some_and(|g| g.kind == Kind::Family || g.kind == Kind::Variant);
            assert!(missing, "{} is REFUSED over one card without a missing architecture: {}", row.repo(), v.why);
            assert!(v.why.contains("then offload"), "{}: {}", row.repo(), v.why);
        }
        assert!(!v.why.contains("refused for size"), "{}", row.repo());
    }
    assert!(oversize > 20, "only {oversize} oversize rows checked");
}

#[test]
fn the_offload_numbers_follow_the_machine_block() {
    let man = manifest();
    let mach = machine(&man);
    let huge = offload_need(mach.card_gb * CARD_SHARE * 2.0 + 100.0, &mach, None);
    assert_eq!(huge.cards, 2);
    assert!((huge.host_gb - 100.0f64.min(mach.ram_available_gb * mach.host_share)).abs() < 1e-6);
    let beyond = offload_need((mach.cards as f64) * mach.card_gb * CARD_SHARE + mach.ram_available_gb * mach.host_share + mach.nvme_free_gb + 500.0, &mach, None);
    assert!((beyond.short_gb - 500.0).abs() < 1e-6, "{beyond:?}");
    let fits = offload_need(mach.card_gb * CARD_SHARE * 1.5, &mach, None);
    assert_eq!((fits.cards, fits.nvme_gb), (2, 0.0));
    let probe = serde_json::json!({"repo": "x/y", "size_gb": 5000.0});
    let refused = sized(Verdict::new(Class::Refused, "model_type zzz is not supported", Some(Gap { kind: Kind::Family, key: "model_type zzz".into() })), &Row { m: &probe }, &mach, None);
    assert_eq!(refused.class, Class::Refused);
    assert!(refused.why.contains("then offload"));
    let loads = sized(Verdict::new(Class::Loads, "ok", None), &Row { m: &probe }, &mach, None);
    assert_eq!(loads.class, Class::NoFit);
}

#[test]
fn kimi_k3_reads_needs_family_then_offload() {
    let doc = report().doc;
    let kimi = section(&doc, "Main labs");
    let line = kimi.lines().find(|l| l.starts_with("| moonshotai/Kimi-K3 |")).expect("Kimi-K3 row");
    assert!(line.contains("needs family #1214, then offload:"), "{line}");
    assert!(line.contains(" cards plus ") && line.contains(" GB host plus ") && line.contains(" GB NVMe"), "{line}");
    assert!(!line.contains("refused for size"));
}

#[test]
fn committed_popular_configs_still_classify_as_the_manifest_says() {
    let man = manifest();
    let mach = machine(&man);
    let want = expected();
    let mut checked = 0;
    let mut failures = Vec::new();
    for m in man["models"].as_array().unwrap() {
        let row = Row { m };
        if !row.is_gguf() && !row.file_committed("config.json") {
            continue;
        }
        let Some(v) = classify(&row, &mach, true) else { continue };
        let got = (v.class.label().to_string(), v.gap.map(|g| g.key).unwrap_or_default(), v.why);
        match want.get(row.repo()) {
            Some(w) if *w == got => {}
            Some(w) => failures.push(format!("{}: classes.tsv says {} ({}), now {} ({})", row.repo(), w.0, w.2, got.0, got.2)),
            None => failures.push(format!("{}: missing from classes.tsv", row.repo())),
        }
        checked += 1;
    }
    assert!(checked > 400, "only {checked} committed rows");
    assert!(failures.is_empty(), "{} rows changed class; rerun crates/specrt/scripts/model_coverage.sh and commit the new doc:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn the_committed_survey_has_a_row_per_class_it_reports() {
    let want = expected();
    let classes: BTreeSet<&str> = want.values().map(|v| v.0.as_str()).collect();
    for c in [Class::Loads, Class::Refused, Class::UnknownQuant] {
        assert!(classes.contains(c.label()), "no {} row", c.label());
    }
    for f in std::fs::read_dir(popular()).unwrap().flatten().filter(|e| e.path().is_dir()) {
        for file in std::fs::read_dir(f.path()).unwrap().flatten() {
            let len = file.metadata().unwrap().len();
            assert!(len <= 100 * 1024, "{} is {len} bytes, over the 100 KB commit limit", file.path().display());
        }
    }
}

#[test]
fn a_gguf_header_with_an_unknown_architecture_is_refused_by_the_real_check() {
    let err = specrt::gguf::Gguf::from_header(gguf_header("no_such_arch")).unwrap().spec().unwrap_err();
    assert!(format!("{err:#}").contains("GGUF architecture no_such_arch is not supported"), "{err:#}");
}

#[test]
#[ignore]
fn popular_model_coverage_report() {
    let r = report();
    if std::env::var_os("POPULAR_WRITE").is_some() {
        std::fs::write(popular().join("classes.tsv"), &r.tsv).unwrap();
        std::fs::write(root().join("../../docs/specrt-model-coverage.md"), &r.doc).unwrap();
        std::fs::write(cache().join("gaps.json"), serde_json::to_string_pretty(&r.gaps).unwrap()).unwrap();
    }
    print!("{}", r.doc);
}
