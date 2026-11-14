mod chat;
mod breakdown;
mod table;
mod block;
mod detail;
mod engine;
mod strip;
mod job;
mod load;
mod models;
mod logs;
mod server;
mod settings;
mod shell;
mod specs;

use eframe::egui;
use inference_bench_core::engine::{Command, Engine, EngineEvent, Wake};
use inference_bench_core::gpu::GpuStats;
use inference_bench_core::library::{LocalModel, Sources};
use inference_bench_core::settings::Settings;
use inference_bench_core::store;
use job::Pending;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Models,
    Server,
    Logs,
    Settings,
}

pub struct App {
    pub dir: PathBuf,
    pub tab: Tab,
    pub settings: Settings,
    pub library: Vec<LocalModel>,
    pub scan: Pending<Vec<LocalModel>>,
    pub engine: Engine,
    pub wake: Wake,
    pub loading: Option<String>,
    pub gpu: Vec<GpuStats>,
    gpu_poll: Pending<Vec<GpuStats>>,
    pub gpu_at: Option<Instant>,
    pub notice: Option<(bool, String)>,
    pub chat: chat::ChatState,
    pub models: models::ModelsState,
    pub server: server::ServerState,
    pub logs: inference_bench_core::logging::LogBuffer,
    pub logs_ui: logs::LogsState,
    pub settings_ui: settings::SettingsState,
    pub specs: specs::SpecsUpdater,
    pub load: Option<load::LoadDialog>,
}

impl App {
    pub fn new(dir: PathBuf, wake: Wake) -> Self {
        let mut app = Self::bare(dir, wake);
        app.specs.auto = true;
        app.rescan();
        app
    }

    pub fn bare(dir: PathBuf, wake: Wake) -> Self {
        let settings = Settings::load(&dir);
        Self {
            tab: Tab::Chat,
            library: Vec::new(),
            scan: Pending::default(),
            engine: Engine::spawn(wake.clone()),
            loading: None,
            gpu: Vec::new(),
            gpu_poll: Pending::default(),
            gpu_at: None,
            notice: None,
            chat: chat::ChatState::load(&dir, &settings),
            models: models::ModelsState::default(),
            server: server::ServerState::default(),
            logs: inference_bench_core::logging::LogBuffer::new(5000),
            logs_ui: logs::LogsState::default(),
            settings_ui: settings::SettingsState::default(),
            specs: {
                let mut s = specs::SpecsUpdater::default();
                s.load_state(&dir);
                s
            },
            load: None,
            settings,
            wake,
            dir,
        }
    }

    pub fn rescan(&mut self) {
        let sources = Sources::standard(&self.dir, &self.settings.model_folders, &self.settings.spec_folders);
        self.scan.start(job::spawn(&self.wake, move || inference_bench_core::library::scan(&sources)));
    }

    pub fn loaded(&self) -> Option<inference_bench_core::engine::Loaded> {
        self.engine.loaded()
    }

    pub fn free_gpu_bytes(&self) -> Option<u64> {
        self.gpu.first().map(|g| g.total_bytes.saturating_sub(g.used_bytes))
    }

    pub fn open_load_dialog(&mut self, model: &LocalModel) {
        self.load = Some(load::LoadDialog::open(self, model));
    }

    pub fn load_model(&mut self, model: &LocalModel) {
        let ctx = self.settings.ctx_for(&model.name);
        self.load_model_with(model, ctx);
    }

    pub fn load_model_with(&mut self, model: &LocalModel, ctx: usize) {
        self.engine.send(Command::Load {
            model_id: model.model_id(inference_bench_core::target::NATIVE),
            display: model.name.clone(),
            max_ctx: Some(ctx),
        });
    }

    pub fn unload(&mut self) {
        self.engine.send(Command::Unload);
    }

    fn pump(&mut self) {
        if let Some(found) = self.scan.poll() {
            self.library = found;
        }
        while let Ok(event) = self.engine.events.try_recv() {
            match event {
                EngineEvent::Loading(name) => self.loading = Some(name),
                EngineEvent::Loaded(l) => {
                    self.loading = None;
                    self.chat.load_error = None;
                    self.notice = Some((true, format!("{} loaded in {:.1} s", l.display, l.load_secs)));
                }
                EngineEvent::LoadFailed(e) => {
                    self.loading = None;
                    self.chat.load_error = Some(e.to_string());
                    self.notice = Some((false, e.to_string()));
                }
                EngineEvent::Unloaded => self.notice = Some((true, "model unloaded".into())),
            }
        }
        if let Some(g) = self.gpu_poll.poll() {
            self.gpu = g;
        }
        let due = self.gpu_at.is_none_or(|t| t.elapsed() > Duration::from_secs(2));
        if due && !self.gpu_poll.busy() {
            self.gpu_at = Some(Instant::now());
            self.gpu_poll.start(job::spawn(&self.wake, inference_bench_core::gpu::sample));
        }
        self.chat.pump(&self.dir);
        self.models.pump(&mut self.notice);
        if self.specs.pump(&self.wake, &self.dir, &self.settings, &mut self.notice) {
            self.rescan();
        }
    }

    pub fn save_settings(&mut self) {
        self.notice = Some(match self.settings.save(&self.dir) {
            Ok(()) => (true, "settings saved".into()),
            Err(e) => (false, format!("settings: {e}")),
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.pump();
        if self.models.take_rescan() {
            self.rescan();
        }
        let animating = self.chat.streaming()
            || self.loading.is_some()
            || self.models.downloading()
            || self.specs.busy()
            || self.scan.busy();
        root.ctx().request_repaint_after(Duration::from_millis(if animating { 100 } else { 2000 }));
        shell::draw(self, root);
    }
}

fn wake_for(ctx: &egui::Context) -> Wake {
    let ctx = ctx.clone();
    Arc::new(move || ctx.request_repaint())
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().map(PathBuf::from).unwrap_or_else(store::data_dir);
    let buffer = inference_bench_core::logging::LogBuffer::new(5000);
    let mut log = inference_bench_core::logging::LogConfig::default();
    log.buffer = Some(buffer.clone());
    log.file = Some(inference_bench_core::logging::FileLog::in_dir(inference_bench_core::logging::default_log_dir()));
    let _log_guard = inference_bench_core::logging::init(&log).ok();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 560.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon-256.png"))
                    .unwrap_or_default(),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "inference-bench",
        options,
        Box::new(|cc| {
            egui_bench::install(&cc.egui_ctx);
            let mut app = App::new(dir, wake_for(&cc.egui_ctx));
            app.logs = buffer;
            Ok(Box::new(app))
        }),
    )
}

#[cfg(test)]
mod tests;
