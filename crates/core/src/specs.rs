use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use specrt::spec::{Checked, EMBEDDED, check, needs_newer_note};
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const GITHUB_API: &str = "https://api.github.com";
pub const DEFAULT_REPO: &str = "v0l/inference-bench";
pub const DEFAULT_BRANCH: &str = "master";
pub const DEFAULT_HOURS: u64 = 24;
pub const KEEP_GENERATIONS: usize = 3;
const SPECS_DIR: &str = "crates/specrt/specs";
const LISTING_LIMIT: u64 = 4 << 20;
const META: &str = ".meta.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecsConfig {
    pub auto_update: bool,
    pub repo: String,
    pub branch: String,
    pub tag: String,
    pub update_hours: u64,
    pub base: String,
    pub max_download: u64,
    pub max_spec: u64,
    pub timeout: Duration,
}

impl Default for SpecsConfig {
    fn default() -> Self {
        Self {
            auto_update: true,
            repo: DEFAULT_REPO.into(),
            branch: DEFAULT_BRANCH.into(),
            tag: String::new(),
            update_hours: DEFAULT_HOURS,
            base: std::env::var("INFERENCE_BENCH_SPECS_URL").unwrap_or_else(|_| GITHUB_API.into()),
            max_download: 8 << 20,
            max_spec: 256 << 10,
            timeout: Duration::from_secs(60),
        }
    }
}

impl SpecsConfig {
    pub fn git_ref(&self) -> String {
        match self.tag.trim() {
            "" => format!("refs/heads/{}", self.branch.trim()),
            tag => format!("refs/tags/{tag}"),
        }
    }

    pub fn source(&self) -> String {
        let at = if self.tag.trim().is_empty() { self.branch.trim() } else { self.tag.trim() };
        format!("{}@{at}", self.repo.trim())
    }

    pub fn url(&self) -> String {
        format!("{}/repos/{}/commits/{}", self.base.trim_end_matches('/'), self.repo.trim(), encode_path(&self.git_ref()))
    }

    fn contents_url(&self, commit: &str, file: Option<&str>) -> String {
        let tail = file.map(|f| format!("/{f}")).unwrap_or_default();
        format!("{}/repos/{}/contents/{SPECS_DIR}{tail}?ref={commit}", self.base.trim_end_matches('/'), self.repo.trim())
    }

    pub fn interval(&self) -> Option<Duration> {
        (self.auto_update && self.update_hours > 0).then(|| Duration::from_secs(self.update_hours * 3600))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    pub source: String,
    pub commit: Option<String>,
    pub fetched_at: DateTime<Utc>,
}

impl Generation {
    pub fn short_commit(&self) -> String {
        self.commit.as_deref().map(|c| c.chars().take(7).collect()).unwrap_or_else(|| "unknown commit".into())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateState {
    pub source: String,
    pub checked_at: Option<DateTime<Utc>>,
    pub last_success: Option<DateTime<Utc>>,
    pub last_ok: bool,
    pub last_result: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Changes {
    pub commit: Option<String>,
    pub added: Vec<String>,
    pub updated: Vec<String>,
    pub removed: Vec<String>,
    pub skipped: Vec<(String, String)>,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.updated.is_empty() && self.removed.is_empty()
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (label, names) in [("added", &self.added), ("updated", &self.updated), ("removed", &self.removed)] {
            if !names.is_empty() {
                parts.push(format!("{label} {}", names.join(", ")));
            }
        }
        if !self.skipped.is_empty() {
            parts.push(format!("skipped {}", self.skipped.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")));
        }
        if parts.is_empty() { "no changes".into() } else { parts.join("; ") }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Disabled,
    Unchanged,
    Updated(Changes),
}

impl Outcome {
    pub fn describe(&self) -> String {
        match self {
            Outcome::Disabled => "automatic spec updates are off".into(),
            Outcome::Unchanged => "specs are up to date".into(),
            Outcome::Updated(c) => format!("specs updated: {}", c.summary()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpecOrigin {
    Folder,
    Updated(Generation),
    Embedded,
    Checkout,
}

impl SpecOrigin {
    pub fn label(&self) -> String {
        match self {
            SpecOrigin::Folder => "your folder".into(),
            SpecOrigin::Updated(g) => format!(
                "updated from GitHub {} ({}) at {}",
                g.source,
                g.short_commit(),
                g.fetched_at.format("%Y-%m-%d %H:%M UTC")
            ),
            SpecOrigin::Embedded => format!("embedded in inference-bench {}", env!("CARGO_PKG_VERSION")),
            SpecOrigin::Checkout => "source checkout".into(),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            SpecOrigin::Folder => "folder",
            SpecOrigin::Updated(_) => "github",
            SpecOrigin::Embedded => "embedded",
            SpecOrigin::Checkout => "checkout",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecDir {
    pub path: PathBuf,
    pub origin: SpecOrigin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Listed {
    pub name: String,
    pub path: PathBuf,
    pub dir: PathBuf,
    pub origin: SpecOrigin,
    pub table: toml::Table,
    pub problem: Option<String>,
}

impl Listed {
    pub fn loadable(&self) -> bool {
        self.problem.is_none()
    }
}

fn toml_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> =
        entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "toml")).collect();
    paths.sort();
    paths
}

fn read_listed(path: &Path, dir: &SpecDir) -> Option<Listed> {
    let text = std::fs::read_to_string(path).ok()?;
    let table: toml::Table = toml::from_str(&text).ok()?;
    let name = table.get("name").and_then(|v| v.as_str())?.to_string();
    let problem = match check(&text) {
        Ok(Checked::Ready(_)) => None,
        Ok(Checked::NeedsNewer { min_engine, .. }) => Some(needs_newer_note(min_engine)),
        Err(e) => Some(format!("spec does not parse: {e}")),
    };
    Some(Listed { name, path: path.to_path_buf(), dir: dir.path.clone(), origin: dir.origin.clone(), table, problem })
}

pub fn list(layers: &[SpecDir]) -> Vec<Listed> {
    let mut all = Vec::new();
    for layer in layers {
        all.extend(toml_files(&layer.path).iter().filter_map(|p| read_listed(p, layer)));
    }
    let ready: HashSet<String> = all.iter().filter(|l| l.loadable()).map(|l| l.name.clone()).collect();
    let mut seen = HashSet::new();
    all.retain(|l| (l.loadable() || !ready.contains(&l.name)) && seen.insert(l.name.clone()));
    all
}

pub fn find(name: &str, layers: &[SpecDir]) -> Option<PathBuf> {
    list(layers)
        .into_iter()
        .find(|l| l.loadable() && (l.name == name || l.path.file_stem().is_some_and(|s| s == name)))
        .map(|l| l.path)
}

pub fn checkout_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs"))];
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(5) {
            dirs.push(ancestor.join("crates/specrt/specs"));
            dirs.push(ancestor.join("specs"));
        }
    }
    dirs.push(PathBuf::from("crates/specrt/specs"));
    dirs.into_iter().filter(|d| d.is_dir()).collect()
}

pub fn layers(store: &SpecStore, folders: &[PathBuf], checkout: &[PathBuf]) -> Vec<SpecDir> {
    let mut out: Vec<SpecDir> = folders.iter().map(|f| SpecDir { path: f.clone(), origin: SpecOrigin::Folder }).collect();
    if let Some(g) = store.generation() {
        out.push(SpecDir { path: store.current(), origin: SpecOrigin::Updated(g) });
    }
    match store.materialize_embedded() {
        Ok(dir) => out.push(SpecDir { path: dir, origin: SpecOrigin::Embedded }),
        Err(e) => tracing::warn!(error = %e, "could not write the embedded spec snapshot"),
    }
    out.extend(checkout.iter().map(|d| SpecDir { path: d.clone(), origin: SpecOrigin::Checkout }));
    let mut seen = HashSet::new();
    out.retain(|d| seen.insert(std::fs::canonicalize(&d.path).unwrap_or_else(|_| d.path.clone())));
    out
}

pub fn standard_layers(data: &Path, folders: &[PathBuf]) -> Vec<SpecDir> {
    layers(&SpecStore::new(data), folders, &checkout_dirs())
}

fn plain_toml(name: &str) -> bool {
    name.len() > 5
        && name.ends_with(".toml")
        && !name.starts_with('.')
        && !name.contains("..")
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

#[derive(Debug, Default)]
struct Extracted {
    commit: Option<String>,
    files: Vec<(String, Vec<u8>)>,
    ignored: Vec<String>,
    too_large: Vec<String>,
}

fn encode_path(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        match b {
            b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn get(agent: &ureq::Agent, url: &str, accept: &str) -> Result<ureq::http::Response<ureq::Body>> {
    let resp = agent.get(url).header("Accept", accept).header("User-Agent", "inference-bench").call().with_context(|| format!("fetching {url}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        bail!("{url} answered {status}");
    }
    Ok(resp)
}

fn download(cfg: &SpecsConfig, agent: &ureq::Agent, commit: &str) -> Result<Extracted> {
    let listing_url = cfg.contents_url(commit, None);
    let mut resp = get(agent, &listing_url, "application/vnd.github+json")?;
    let text = resp
        .body_mut()
        .with_config()
        .limit(LISTING_LIMIT)
        .read_to_string()
        .with_context(|| format!("{listing_url}: listing over {LISTING_LIMIT} bytes or broken"))?;
    let listing: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("{listing_url} is not a directory listing"))?;
    let entries = listing.as_array().with_context(|| format!("{listing_url} is not a directory listing"))?;
    let mut out = Extracted { commit: Some(commit.to_string()), ..Extracted::default() };
    let mut wanted = Vec::new();
    let mut total = 0u64;
    for entry in entries {
        let name = entry.get("name").and_then(|n| n.as_str()).unwrap_or_default();
        let is_file = entry.get("type").and_then(|t| t.as_str()) == Some("file");
        if !is_file || !plain_toml(name) {
            tracing::warn!(entry = %name, "ignoring a listing entry that is not a plain .toml spec");
            out.ignored.push(name.to_string());
            continue;
        }
        let size = entry.get("size").and_then(|n| n.as_u64()).unwrap_or(0);
        if size > cfg.max_spec {
            tracing::warn!(entry = %name, bytes = size, max_spec = cfg.max_spec, "ignoring a spec file over the size cap");
            out.too_large.push(name.to_string());
            continue;
        }
        total += size;
        wanted.push(name.to_string());
    }
    if total > cfg.max_download {
        bail!("the specs add up to {total} bytes, over the {} byte limit", cfg.max_download);
    }
    for name in wanted {
        let url = cfg.contents_url(commit, Some(&name));
        let mut resp = get(agent, &url, "application/vnd.github.raw+json")?;
        let mut buf = Vec::new();
        resp.body_mut().as_reader().take(cfg.max_spec + 1).read_to_end(&mut buf).with_context(|| format!("reading {url}"))?;
        if buf.len() as u64 > cfg.max_spec {
            out.too_large.push(name);
        } else {
            out.files.push((name, buf));
        }
    }
    Ok(out)
}

fn spec_name(file: &str, text: &str) -> String {
    toml::from_str::<toml::Table>(text)
        .ok()
        .and_then(|t| t.get("name").and_then(|n| n.as_str()).map(String::from))
        .unwrap_or_else(|| file.trim_end_matches(".toml").to_string())
}

fn read_dir_files(dir: &Path) -> BTreeMap<String, String> {
    toml_files(dir)
        .into_iter()
        .filter_map(|p| Some((p.file_name()?.to_string_lossy().into_owned(), std::fs::read_to_string(&p).ok()?)))
        .collect()
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_global(Some(timeout))
        .build()
        .into()
}

fn stamp(now: DateTime<Utc>) -> String {
    now.format("%Y%m%dT%H%M%S%3f").to_string()
}

pub fn resolve_model(data: &Path, folders: &[PathBuf], model: &str) -> String {
    if model.ends_with(".toml") || model.contains('/') || model.contains('\\') {
        return model.to_string();
    }
    match find(model, &standard_layers(data, folders)) {
        Some(path) => path.display().to_string(),
        None => model.to_string(),
    }
}

pub fn status_report(data: &Path, folders: &[PathBuf], cfg: &SpecsConfig) -> String {
    let store = SpecStore::new(data);
    let state = store.state();
    let mut out = String::new();
    let when = |t: Option<DateTime<Utc>>| t.map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string()).unwrap_or_else(|| "never".into());
    out.push_str(&format!("source       {}\n", cfg.source()));
    out.push_str(&format!(
        "auto update  {}\n",
        match cfg.interval() {
            Some(_) => format!("on, every {} h", cfg.update_hours),
            None if cfg.auto_update => "on at start only".into(),
            None => "off".into(),
        }
    ));
    out.push_str(&format!("last check   {}\n", when(state.checked_at)));
    out.push_str(&format!("last success {}\n", when(state.last_success)));
    if !state.last_result.is_empty() {
        out.push_str(&format!("last result  {}{}\n", if state.last_ok { "" } else { "failed: " }, state.last_result));
    }
    match store.generation() {
        Some(g) => out.push_str(&format!("cache        {} ({})\n", g.short_commit(), store.current().display())),
        None => out.push_str("cache        empty, using the embedded snapshot\n"),
    }
    out.push_str(&format!("history      {} earlier generation(s)\n\n", store.history().len()));
    for l in list(&layers(&store, folders, &checkout_dirs())) {
        out.push_str(&format!("{:<24} {:<9} {}", l.name, l.origin.kind(), l.origin.label()));
        if let Some(p) = &l.problem {
            out.push_str(&format!("  [{p}]"));
        }
        out.push('\n');
    }
    out
}

pub fn refresh_if_due(data: &Path, cfg: &SpecsConfig) -> Option<Result<Outcome>> {
    let store = SpecStore::new(data);
    store.due(cfg, Utc::now()).then(|| store.update(cfg))
}

pub struct SpecStore {
    root: PathBuf,
}

impl SpecStore {
    pub fn new(data: &Path) -> Self {
        Self { root: data.join("specs") }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn current(&self) -> PathBuf {
        self.root.join("current")
    }

    pub fn embedded_dir(&self) -> PathBuf {
        self.root.join("embedded")
    }

    fn history_dir(&self) -> PathBuf {
        self.root.join(".history")
    }

    fn etag_path(&self) -> PathBuf {
        self.root.join(".etag")
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    pub fn state(&self) -> UpdateState {
        std::fs::read_to_string(self.state_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    fn save_state(&self, state: &UpdateState) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::write(self.state_path(), serde_json::to_string_pretty(state)?)?;
        Ok(())
    }

    fn read_meta(dir: &Path) -> Option<Generation> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(META)).ok()?).ok()
    }

    pub fn generation(&self) -> Option<Generation> {
        Self::read_meta(&self.current())
    }

    pub fn history(&self) -> Vec<(PathBuf, Generation)> {
        let Ok(entries) = std::fs::read_dir(self.history_dir()) else { return Vec::new() };
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        dirs.reverse();
        dirs.into_iter().filter_map(|d| Some((d.clone(), Self::read_meta(&d)?))).collect()
    }

    pub fn materialize_embedded(&self) -> Result<PathBuf> {
        let dir = self.embedded_dir();
        std::fs::create_dir_all(&dir)?;
        let keep: HashSet<&str> = EMBEDDED.iter().map(|(n, _)| *n).collect();
        for (name, text) in EMBEDDED {
            let path = dir.join(name);
            if std::fs::read_to_string(&path).ok().as_deref() != Some(*text) {
                std::fs::write(&path, text)?;
            }
        }
        for path in toml_files(&dir) {
            if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| !keep.contains(n)) {
                std::fs::remove_file(path)?;
            }
        }
        Ok(dir)
    }

    pub fn due(&self, cfg: &SpecsConfig, now: DateTime<Utc>) -> bool {
        let Some(every) = cfg.interval() else { return false };
        match self.state().checked_at {
            Some(t) if self.state().source == cfg.source() => {
                now.signed_duration_since(t).to_std().unwrap_or_default() >= every
            }
            _ => true,
        }
    }

    pub fn auto_update(&self, cfg: &SpecsConfig) -> Result<Outcome> {
        if !cfg.auto_update {
            return Ok(Outcome::Disabled);
        }
        self.update(cfg)
    }

    pub fn update(&self, cfg: &SpecsConfig) -> Result<Outcome> {
        let now = Utc::now();
        let mut state = self.state();
        let same_source = state.source == cfg.source();
        state.source = cfg.source();
        state.checked_at = Some(now);
        let result = self.fetch_and_apply(cfg, same_source, now);
        match &result {
            Ok(outcome) => {
                state.last_ok = true;
                state.last_success = Some(now);
                state.last_result = outcome.describe();
            }
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), source = %cfg.source(), "spec update failed");
                state.last_ok = false;
                state.last_result = format!("{e:#}");
            }
        }
        self.save_state(&state)?;
        result
    }

    fn fetch_and_apply(&self, cfg: &SpecsConfig, same_source: bool, now: DateTime<Utc>) -> Result<Outcome> {
        let url = cfg.url();
        let etag = std::fs::read_to_string(self.etag_path()).ok().filter(|e| !e.trim().is_empty());
        let agent = agent(cfg.timeout);
        let mut req = agent.get(&url).header("Accept", "application/vnd.github.sha").header("User-Agent", "inference-bench");
        if let Some(tag) = etag.as_deref().filter(|_| same_source && self.current().is_dir()) {
            req = req.header("If-None-Match", tag.trim());
        }
        let mut resp = req.call().with_context(|| format!("fetching {url}"))?;
        let status = resp.status().as_u16();
        if status == 304 {
            tracing::info!(source = %cfg.source(), "specs are up to date");
            return Ok(Outcome::Unchanged);
        }
        if status != 200 {
            bail!("{url} answered {status}");
        }
        let new_etag = resp.headers().get("etag").and_then(|v| v.to_str().ok()).map(String::from);
        let commit = resp.body_mut().with_config().limit(256).read_to_string().with_context(|| format!("reading {url}"))?.trim().to_string();
        if commit.is_empty() || commit.len() > 64 || !commit.chars().all(|c| c.is_ascii_alphanumeric()) {
            bail!("{url} did not answer with a commit id");
        }
        let extracted = download(cfg, &agent, &commit)?;
        if extracted.files.is_empty() {
            bail!("{url} has no spec files under {SPECS_DIR}");
        }
        let changes = self.apply(cfg, extracted, now)?;
        std::fs::write(self.etag_path(), new_etag.unwrap_or_default())?;
        Ok(match changes.is_empty() {
            true => Outcome::Unchanged,
            false => Outcome::Updated(changes),
        })
    }

    fn apply(&self, cfg: &SpecsConfig, extracted: Extracted, now: DateTime<Utc>) -> Result<Changes> {
        std::fs::create_dir_all(&self.root)?;
        let old = read_dir_files(&self.current());
        let incoming = self.root.join(format!(".incoming-{}-{}", std::process::id(), now.timestamp_nanos_opt().unwrap_or(0)));
        let _ = std::fs::remove_dir_all(&incoming);
        let built = self.build_incoming(&incoming, &old, extracted, cfg, now);
        let changes = match built {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&incoming);
                return Err(e);
            }
        };
        if changes.is_empty() && self.current().is_dir() {
            std::fs::remove_dir_all(&incoming)?;
            std::fs::write(
                self.current().join(META),
                serde_json::to_string_pretty(&Generation { source: cfg.source(), commit: changes.commit.clone(), fetched_at: now })?,
            )?;
            return Ok(changes);
        }
        if let Err(e) = self.swap(&incoming, now) {
            let _ = std::fs::remove_dir_all(&incoming);
            return Err(e);
        }
        tracing::info!(
            source = %cfg.source(),
            commit = ?changes.commit,
            added = ?changes.added,
            updated = ?changes.updated,
            removed = ?changes.removed,
            skipped = ?changes.skipped,
            "specs updated"
        );
        Ok(changes)
    }

    fn build_incoming(
        &self,
        incoming: &Path,
        old: &BTreeMap<String, String>,
        extracted: Extracted,
        cfg: &SpecsConfig,
        now: DateTime<Utc>,
    ) -> Result<Changes> {
        std::fs::create_dir_all(incoming)?;
        let mut changes = Changes { commit: extracted.commit.clone(), ..Changes::default() };
        let mut written: BTreeMap<String, String> = BTreeMap::new();
        let keep_old = |file: &str, why: String, changes: &mut Changes, written: &mut BTreeMap<String, String>| -> Result<()> {
            tracing::warn!(file, reason = %why, "skipping a downloaded spec");
            changes.skipped.push((file.to_string(), why));
            if let Some(prev) = old.get(file) {
                std::fs::write(incoming.join(file), prev)?;
                written.insert(file.to_string(), prev.clone());
            }
            Ok(())
        };
        for (file, bytes) in extracted.files {
            let text = match String::from_utf8(bytes) {
                Ok(t) => t,
                Err(_) => {
                    keep_old(&file, "not UTF-8".into(), &mut changes, &mut written)?;
                    continue;
                }
            };
            if let Err(e) = check(&text) {
                keep_old(&file, format!("{e:#}"), &mut changes, &mut written)?;
                continue;
            }
            std::fs::write(incoming.join(&file), &text)?;
            written.insert(file, text);
        }
        for file in extracted.too_large {
            keep_old(&file, format!("larger than {} bytes", cfg.max_spec), &mut changes, &mut written)?;
        }
        if written.is_empty() {
            bail!("no downloaded spec parsed");
        }
        for (file, text) in &written {
            match old.get(file) {
                None => changes.added.push(spec_name(file, text)),
                Some(prev) if prev != text => changes.updated.push(spec_name(file, text)),
                Some(_) => {}
            }
        }
        for (file, text) in old {
            if !written.contains_key(file) {
                changes.removed.push(spec_name(file, text));
            }
        }
        std::fs::write(
            incoming.join(META),
            serde_json::to_string_pretty(&Generation { source: cfg.source(), commit: extracted.commit, fetched_at: now })?,
        )?;
        Ok(changes)
    }

    fn swap(&self, incoming: &Path, now: DateTime<Utc>) -> Result<()> {
        let current = self.current();
        let mut parked = None;
        if current.is_dir() {
            let history = self.history_dir();
            std::fs::create_dir_all(&history)?;
            let mut target = history.join(stamp(now));
            let mut n = 1;
            while target.exists() {
                target = history.join(format!("{}-{n}", stamp(now)));
                n += 1;
            }
            std::fs::rename(&current, &target).context("moving the current specs to history")?;
            parked = Some(target);
        }
        if let Err(e) = std::fs::rename(incoming, &current) {
            if let Some(p) = &parked {
                let _ = std::fs::rename(p, &current);
            }
            return Err(e).context("swapping in the new specs");
        }
        self.prune();
        Ok(())
    }

    fn prune(&self) {
        for (dir, _) in self.history().into_iter().skip(KEEP_GENERATIONS) {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    pub fn rollback(&self) -> Result<Generation> {
        let Some((previous, generation)) = self.history().into_iter().next() else {
            bail!("there is no earlier spec generation to roll back to");
        };
        let current = self.current();
        if current.is_dir() {
            let discard = self.root.join(format!(".discard-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&discard);
            std::fs::rename(&current, &discard)?;
            std::fs::rename(&previous, &current)?;
            std::fs::remove_dir_all(&discard)?;
        } else {
            std::fs::rename(&previous, &current)?;
        }
        let mut state = self.state();
        state.last_result = format!("rolled back to {} from {}", generation.short_commit(), generation.fetched_at.format("%Y-%m-%d %H:%M UTC"));
        self.save_state(&state)?;
        tracing::info!(commit = ?generation.commit, fetched_at = %generation.fetched_at, "specs rolled back");
        Ok(generation)
    }
}

#[cfg(test)]
mod tests;
