use anyhow::{Context, Result};
use chrono::{DateTime, Local, Utc};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    #[default]
    Human,
    Json,
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "human" | "text" | "pretty" => Ok(Self::Human),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown log format {other}, use human or json")),
        }
    }
}

#[derive(Debug, Clone, clap::Args, Default)]
pub struct LogArgs {
    #[arg(long, global = true, help = "Log filter such as info or specrt=debug,info (overrides RUST_LOG)")]
    pub log_level: Option<String>,
    #[arg(long, global = true, default_value = "human", help = "human or json")]
    pub log_format: LogFormat,
    #[arg(long, global = true, env = "INFERENCE_BENCH_LOG_DIR", help = "Write rolling log files here")]
    pub log_dir: Option<PathBuf>,
    #[arg(long, global = true, help = "Write rolling log files to the platform state directory")]
    pub log_file: bool,
    #[arg(long, global = true, help = "Log prompts at info level (otherwise only at trace)")]
    pub log_prompts: bool,
}

impl LogArgs {
    pub fn config(&self) -> LogConfig {
        let file = match (&self.log_dir, self.log_file) {
            (Some(dir), _) => Some(FileLog::in_dir(dir.clone())),
            (None, true) => Some(FileLog::in_dir(default_log_dir())),
            _ => None,
        };
        LogConfig {
            level: self.log_level.clone(),
            format: self.log_format,
            file,
            buffer: None,
            log_prompts: self.log_prompts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileLog {
    pub dir: PathBuf,
    pub prefix: String,
    pub max_bytes: u64,
    pub max_files: usize,
    pub max_age: Duration,
}

impl FileLog {
    pub fn in_dir(dir: PathBuf) -> Self {
        Self {
            dir,
            prefix: "inference-bench".into(),
            max_bytes: 20 * 1024 * 1024,
            max_files: 10,
            max_age: Duration::from_secs(14 * 24 * 3600),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LogConfig {
    pub level: Option<String>,
    pub format: LogFormat,
    pub file: Option<FileLog>,
    pub buffer: Option<LogBuffer>,
    pub log_prompts: bool,
}

pub fn default_log_dir() -> PathBuf {
    dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("inference-bench")
        .join("logs")
}

pub fn filter_spec(level: Option<&str>) -> String {
    match (level, std::env::var("RUST_LOG")) {
        (Some(l), _) if !l.is_empty() => l.to_string(),
        (_, Ok(env)) if !env.is_empty() => env,
        _ => "info".into(),
    }
}

fn env_filter(spec: &str) -> EnvFilter {
    EnvFilter::try_new(spec).unwrap_or_else(|_| EnvFilter::new("info"))
}

pub struct LogGuard {
    _file: Option<tracing_appender::non_blocking::WorkerGuard>,
}

type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync>;

pub fn layers(config: &LogConfig) -> Result<(Vec<BoxedLayer>, LogGuard)> {
    let spec = filter_spec(config.level.as_deref());
    let mut out: Vec<BoxedLayer> = Vec::new();
    let tty = std::io::stderr().is_terminal();
    out.push(match config.format {
        LogFormat::Human => fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(tty)
            .with_filter(env_filter(&spec))
            .boxed(),
        LogFormat::Json => fmt::layer()
            .json()
            .with_writer(std::io::stderr)
            .with_filter(env_filter(&spec))
            .boxed(),
    });
    let mut guard = None;
    if let Some(file) = &config.file {
        let writer = RotatingFile::open(file.clone())?;
        let (writer, g) = tracing_appender::non_blocking(writer);
        guard = Some(g);
        out.push(
            fmt::layer().json().with_ansi(false).with_writer(writer).with_filter(env_filter(&spec)).boxed(),
        );
    }
    if let Some(buffer) = &config.buffer {
        out.push(buffer.clone().with_filter(env_filter(&spec)).boxed());
    }
    Ok((out, LogGuard { _file: guard }))
}

pub fn init(config: &LogConfig) -> Result<LogGuard> {
    let (layers, guard) = layers(config)?;
    tracing_subscriber::registry()
        .with(layers)
        .try_init()
        .map_err(|e| anyhow::anyhow!("logging already initialised: {e}"))?;
    install_panic_hook();
    Ok(guard)
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map(|l| l.to_string()).unwrap_or_default();
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".into());
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        tracing::error!(%location, %thread, %backtrace, "panic: {message}");
    }));
}

pub fn redact_key(key: &str) -> String {
    match key.len() {
        0 => String::new(),
        1..=8 => "***".into(),
        n => format!("{}...{} ({n} chars)", &key[..2], &key[n - 2..]),
    }
}

pub fn preview(text: &str, max: usize) -> String {
    let mut cut = max.min(text.len());
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    match cut < text.len() {
        true => format!("{}... ({} bytes)", &text[..cut], text.len()),
        false => text.to_string(),
    }
}

pub struct RotatingFile {
    config: FileLog,
    file: File,
    size: u64,
    day: String,
}

impl RotatingFile {
    pub fn open(config: FileLog) -> Result<Self> {
        std::fs::create_dir_all(&config.dir).with_context(|| config.dir.display().to_string())?;
        let path = config.dir.join(format!("{}.log", config.prefix));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| path.display().to_string())?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let this = Self { config, file, size, day: today() };
        this.prune();
        Ok(this)
    }

    fn current(&self) -> PathBuf {
        self.config.dir.join(format!("{}.log", self.config.prefix))
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        let stamp = Local::now().format("%Y%m%dT%H%M%S%3f");
        let rotated = self.config.dir.join(format!("{}.{stamp}.log", self.config.prefix));
        std::fs::rename(self.current(), rotated)?;
        self.file = OpenOptions::new().create(true).append(true).open(self.current())?;
        self.size = 0;
        self.day = today();
        self.prune();
        Ok(())
    }

    fn rotated_files(&self) -> Vec<(PathBuf, SystemTime)> {
        let prefix = format!("{}.", self.config.prefix);
        let current = self.current();
        let mut files: Vec<_> = std::fs::read_dir(&self.config.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with(&prefix) && name.ends_with(".log") && e.path() != current
            })
            .filter_map(|e| Some((e.path(), e.metadata().ok()?.modified().ok()?)))
            .collect();
        files.sort_by_key(|(path, modified)| (*modified, path.clone()));
        files
    }

    fn prune(&self) {
        let files = self.rotated_files();
        let now = SystemTime::now();
        let excess = files.len().saturating_sub(self.config.max_files);
        for (i, (path, modified)) in files.iter().enumerate() {
            let old = now.duration_since(*modified).map(|d| d > self.config.max_age).unwrap_or(false);
            if i < excess || old {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn today() -> String {
    Local::now().format("%Y%m%d").to_string()
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let full = self.size > 0 && self.size + buf.len() as u64 > self.config.max_bytes;
        if full || (self.size > 0 && self.day != today()) {
            self.rotate()?;
        }
        let n = self.file.write(buf)?;
        self.size += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub time: DateTime<Utc>,
    pub level: Level,
    pub target: String,
    pub message: String,
    pub fields: Vec<(String, String)>,
    pub spans: Vec<(String, String)>,
}

impl LogEntry {
    pub fn line(&self) -> String {
        let mut s = format!(
            "{} {:>5} {}: {}",
            self.time.with_timezone(&Local).format("%H:%M:%S%.3f"),
            self.level,
            self.target,
            self.message
        );
        for (k, v) in self.spans.iter().chain(self.fields.iter()) {
            let _ = write!(s, " {k}={v}");
        }
        s
    }

    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields.iter().chain(self.spans.iter()).find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<VecDeque<LogEntry>>>,
    capacity: usize,
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { inner: Arc::default(), capacity: capacity.max(1) }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    pub fn push(&self, entry: LogEntry) {
        let mut q = self.inner.lock().unwrap();
        if q.len() == self.capacity {
            q.pop_front();
        }
        q.push_back(entry);
    }

    pub fn snapshot(&self, max_level: Level, search: &str) -> Vec<LogEntry> {
        let needle = search.to_lowercase();
        self.inner
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.level <= max_level)
            .filter(|e| needle.is_empty() || e.line().to_lowercase().contains(&needle))
            .cloned()
            .collect()
    }

    pub fn text(&self, max_level: Level, search: &str) -> String {
        self.snapshot(max_level, search).iter().map(LogEntry::line).collect::<Vec<_>>().join("\n")
    }

    pub fn save(&self, path: &Path, max_level: Level, search: &str) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        }
        std::fs::write(path, self.text(max_level, search)).with_context(|| path.display().to_string())
    }
}

#[derive(Default)]
struct Fields {
    message: String,
    fields: Vec<(String, String)>,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "message" => self.message = format!("{value:?}"),
            name => self.fields.push((name.to_string(), format!("{value:?}"))),
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "message" => self.message = value.to_string(),
            name => self.fields.push((name.to_string(), value.to_string())),
        }
    }
}

struct SpanFields(Vec<(String, String)>);

impl<S> Layer<S> for LogBuffer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: LayerContext<'_, S>,
    ) {
        let mut v = Fields::default();
        attrs.record(&mut v);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanFields(v.fields));
        }
    }

    fn on_record(&self, id: &tracing::span::Id, values: &tracing::span::Record<'_>, ctx: LayerContext<'_, S>) {
        let mut v = Fields::default();
        values.record(&mut v);
        if let Some(span) = ctx.span(id) {
            if let Some(f) = span.extensions_mut().get_mut::<SpanFields>() {
                for (k, val) in v.fields {
                    match f.0.iter_mut().find(|(name, _)| *name == k) {
                        Some(slot) => slot.1 = val,
                        None => f.0.push((k, val)),
                    }
                }
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: LayerContext<'_, S>) {
        let mut v = Fields::default();
        event.record(&mut v);
        let spans = ctx
            .event_scope(event)
            .map(|scope| {
                scope
                    .from_root()
                    .flat_map(|s| s.extensions().get::<SpanFields>().map(|f| f.0.clone()).unwrap_or_default())
                    .collect()
            })
            .unwrap_or_default();
        self.push(LogEntry {
            time: Utc::now(),
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            message: v.message,
            fields: v.fields,
            spans,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_never_shown_whole() {
        let shown = redact_key("sk-supersecretvalue");
        assert!(!shown.contains("supersecret"), "{shown}");
        assert_eq!(redact_key("abc"), "***");
    }

    #[test]
    fn previews_cut_on_a_character_boundary() {
        assert_eq!(preview("héllo wörld", 2), "h... (13 bytes)");
        assert_eq!(preview("short", 40), "short");
    }

    #[test]
    fn formats_parse() {
        assert_eq!("json".parse::<LogFormat>().unwrap(), LogFormat::Json);
        assert!("xml".parse::<LogFormat>().is_err());
    }

    #[test]
    fn rotation_by_size_keeps_a_bounded_number_of_files() {
        let dir = tempdir("rotate");
        let mut w = RotatingFile::open(FileLog {
            dir: dir.clone(),
            prefix: "t".into(),
            max_bytes: 100,
            max_files: 2,
            max_age: Duration::from_secs(3600),
        })
        .unwrap();
        for _ in 0..12 {
            w.write_all(&[b'x'; 60]).unwrap();
            std::thread::sleep(Duration::from_millis(3));
        }
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(names.len(), 3, "current plus two rotated");
        for e in names {
            assert!(e.metadata().unwrap().len() <= 120);
        }
    }

    #[test]
    fn buffer_filters_by_level_and_search_and_is_bounded() {
        let b = LogBuffer::new(3);
        let sub = tracing_subscriber::registry().with(b.clone());
        tracing::subscriber::with_default(sub, || {
            let span = tracing::info_span!("request", request_id = "req-1");
            let _e = span.enter();
            tracing::info!(model = "m", "first");
            tracing::debug!("second");
            tracing::warn!("third needle");
            tracing::error!("fourth");
        });
        assert_eq!(b.len(), 3);
        let warnings = b.snapshot(Level::WARN, "");
        assert_eq!(warnings.len(), 2);
        assert_eq!(warnings[0].field("request_id"), Some("req-1"));
        assert_eq!(b.snapshot(Level::TRACE, "NEEDLE").len(), 1);
        assert!(b.text(Level::TRACE, "").contains("fourth"));
    }

    fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }
}
