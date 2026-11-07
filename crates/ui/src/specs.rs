use crate::job::{self, Pending};
use inference_bench_core::engine::Wake;
use inference_bench_core::specs::{self, Outcome, SpecStore, SpecsConfig, UpdateState};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_secs(600);

#[derive(Default)]
pub struct SpecsUpdater {
    pub auto: bool,
    pub base: Option<String>,
    pub state: UpdateState,
    pub last: Option<(bool, String)>,
    check: Pending<Option<Result<Outcome, String>>>,
    polled: Option<Instant>,
}

impl SpecsUpdater {
    pub fn busy(&self) -> bool {
        self.check.busy()
    }

    fn config(&self, settings: &inference_bench_core::settings::Settings) -> SpecsConfig {
        let mut cfg = settings.specs();
        if let Some(b) = &self.base {
            cfg.base = b.clone();
        }
        cfg
    }

    pub fn load_state(&mut self, dir: &Path) {
        self.state = SpecStore::new(dir).state();
    }

    pub fn check_now(&mut self, wake: &Wake, dir: &Path, settings: &inference_bench_core::settings::Settings) {
        if self.busy() {
            return;
        }
        let (cfg, dir) = (self.config(settings), dir.to_path_buf());
        self.check.start(job::spawn(wake, move || Some(SpecStore::new(&dir).update(&cfg).map_err(|e| format!("{e:#}")))));
    }

    fn auto_check(&mut self, wake: &Wake, dir: &Path, settings: &inference_bench_core::settings::Settings) {
        let cfg = self.config(settings);
        if !self.auto || !cfg.auto_update || self.busy() || self.polled.is_some_and(|t| t.elapsed() < POLL) {
            return;
        }
        self.polled = Some(Instant::now());
        let dir: PathBuf = dir.to_path_buf();
        self.check.start(job::spawn(wake, move || {
            let store = SpecStore::new(&dir);
            let first = store.state().checked_at.is_none() || store.generation().is_none();
            match first || store.due(&cfg, chrono::Utc::now()) {
                true => Some(store.update(&cfg).map_err(|e| format!("{e:#}"))),
                false => None,
            }
        }));
    }

    pub fn pump(
        &mut self,
        wake: &Wake,
        dir: &Path,
        settings: &inference_bench_core::settings::Settings,
        notice: &mut Option<(bool, String)>,
    ) -> bool {
        self.auto_check(wake, dir, settings);
        let Some(done) = self.check.poll() else { return false };
        self.load_state(dir);
        let Some(result) = done else { return false };
        let (ok, text, landed) = match result {
            Ok(o) => (true, o.describe(), matches!(o, Outcome::Updated(_))),
            Err(e) => (false, format!("spec update failed: {e}"), false),
        };
        self.last = Some((ok, text.clone()));
        if landed || !ok {
            *notice = Some((ok, text));
        }
        landed
    }
}

pub fn when(t: Option<chrono::DateTime<chrono::Utc>>) -> String {
    t.map(|t| t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string()).unwrap_or_else(|| "never".into())
}

pub fn rollback(dir: &Path) -> Result<String, String> {
    specs::SpecStore::new(dir)
        .rollback()
        .map(|g| format!("rolled back to {} from {}", g.short_commit(), when(Some(g.fetched_at))))
        .map_err(|e| e.to_string())
}
