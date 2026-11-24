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
    TooBig,
    UnknownQuant,
    Refused,
    Unchecked,
}

impl Class {
    const ALL: [Class; 6] = [Class::Loads, Class::TextOnly, Class::TooBig, Class::UnknownQuant, Class::Refused, Class::Unchecked];

    fn label(self) -> &'static str {
        match self {
            Class::Loads => "LOADS",
            Class::TextOnly => "LOADS-TEXT-ONLY",
            Class::TooBig => "TOO-BIG",
            Class::UnknownQuant => "UNKNOWN-QUANT",
            Class::Refused => "REFUSED",
            Class::Unchecked => "UNCHECKED",
        }
    }

    fn arch_ok(self) -> bool {
        matches!(self, Class::Loads | Class::TextOnly | Class::TooBig)
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
}

impl Verdict {
    fn new(class: Class, why: impl Into<String>, gap: Option<Gap>) -> Self {
        Verdict { class, why: why.into(), gap }
    }
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
                Err(arch) => Verdict::new(Class::Refused, arch.clone(), Some(refusal_gap(&arch))),
            }
        }
    }
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
    if read.is_empty() {
        return Verdict::new(Class::UnknownQuant, format!("GGUF {arch}: no file in a type specrt reads ({})", join(&unread)), Some(Gap { kind: Kind::Quant, key: "GGUF types".into() }));
    }
    let mut why = format!("GGUF {arch}; reads {}", join(&read));
    if !unread.is_empty() {
        write!(why, "; not {}", join(&unread)).unwrap();
    }
    Verdict::new(Class::Loads, why, None)
}

fn sized(v: Verdict, row: &Row, card_gb: f64) -> Verdict {
    match row.size() {
        Some(gb) if v.class.arch_ok() && gb > card_gb * CARD_SHARE => {
            let cards = (gb / (card_gb * CARD_SHARE)).ceil() as u64;
            Verdict::new(Class::TooBig, format!("{gb} GB of weights needs {cards} cards of {card_gb} GB; {}", v.why), v.gap)
        }
        _ => v,
    }
}

fn classify(row: &Row, card_gb: f64, committed_only: bool) -> Option<Verdict> {
    let v = if row.is_gguf() {
        classify_gguf(row)
    } else {
        let config = if committed_only { committed(row.repo(), "config.json") } else { fetched(row.repo(), "config.json") };
        match config {
            Some(c) => classify_config(&c, row),
            None if committed_only && row.has_file("config.json") => return None,
            None => Verdict::new(Class::Unchecked, row.m["config_from"].as_str().unwrap_or("no config.json"), None),
        }
    };
    Some(sized(v, row, card_gb))
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
}

fn report() -> Report {
    let man = manifest();
    let card_gb = man["card_gb"].as_f64().unwrap();
    let rows: Vec<Row> = man["models"].as_array().unwrap().iter().map(|m| Row { m }).collect();
    let weight = weights(&rows);
    let tiny = fixture_families();
    let filed = issues();
    let verdicts: Vec<Verdict> = rows.iter().map(|r| classify(r, card_gb, false).unwrap()).collect();
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

    let mut doc = String::new();
    let fetched_on = man["fetched"].as_str().unwrap();
    writeln!(doc, "# specrt model coverage\n").unwrap();
    writeln!(doc, "Fetched {fetched_on}. Generated by `crates/specrt/scripts/model_coverage.sh`; do not edit by hand.\n").unwrap();
    writeln!(doc, "\"Popular\" means popular on Hugging Face on {fetched_on}: the most downloaded and the trending repos for the text-generation, image-text-to-text and any-to-any pipeline tags, as the hub API ranked them that day. Downloads are the hub's last-30-day counts. Nothing here was measured by loading weights: every class comes from the model's `config.json` (and the hub's file list for GGUF repos) run through specrt's own checks, `Spec::from_hf_config_serving`, `Vision::from_hf_config` and `Gguf::spec`, with the family specs embedded in this build.\n").unwrap();
    writeln!(doc, "Source: {}.\n", man["source"].as_str().unwrap()).unwrap();
    writeln!(doc, "Selection: {}. Each group keeps the original lab's repo (`base`) and the most downloaded NVFP4, FP8, AWQ, GPTQ and GGUF copy (`nvfp4`, `fp8`, `awq`, `gptq`, `gguf`), plus any copy already in the local hub cache (`disk`).\n", man["selection"].as_str().unwrap()).unwrap();
    writeln!(doc, "## Classes\n").unwrap();
    writeln!(doc, "- LOADS: the serving check passes and every input tower the config declares is read.").unwrap();
    writeln!(doc, "- LOADS-TEXT-ONLY: the text model passes; a vision or audio tower in the config is refused or not read, so only text input works.").unwrap();
    writeln!(doc, "- TOO-BIG: the architecture passes but the weights are more than {:.0}% of one {card_gb} GB card; the why column says how many cards.", CARD_SHARE * 100.0).unwrap();
    writeln!(doc, "- UNKNOWN-QUANT: the architecture passes without the quantisation, the quantisation format is refused.").unwrap();
    writeln!(doc, "- REFUSED: the architecture itself is refused; the why column is the exact refusal text.").unwrap();
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
    writeln!(doc, "Each gap is one missing piece, ranked by the summed downloads of the rows it blocks (a base row counts its whole group less the quantised copies listed beside it). Chat and tokenizer gaps are counted only on rows whose architecture passes. Gaps under {} downloads have no issue filed.\n", millions(FILE_THRESHOLD)).unwrap();
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
    writeln!(doc, "Modules per weight layout, counted from each repo's `model.safetensors.index.json` (tensor names only). Read today means at least one repo holding that layout classifies as LOADS, LOADS-TEXT-ONLY or TOO-BIG.\n").unwrap();
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
        let issue = v.gap.as_ref().map(&fold).or(chats[i].1.clone().filter(|_| v.class.arch_ok())).and_then(|g| filed.get(&g.key)).map(|n| format!("#{n}")).unwrap_or_default();
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
    Report { tsv, doc }
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

#[test]
fn committed_popular_configs_still_classify_as_the_manifest_says() {
    let man = manifest();
    let card_gb = man["card_gb"].as_f64().unwrap();
    let want = expected();
    let mut checked = 0;
    let mut failures = Vec::new();
    for m in man["models"].as_array().unwrap() {
        let row = Row { m };
        if !row.is_gguf() && !row.file_committed("config.json") {
            continue;
        }
        let Some(v) = classify(&row, card_gb, true) else { continue };
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
    }
    print!("{}", r.doc);
}
