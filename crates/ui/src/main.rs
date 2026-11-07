mod bench;
mod chart;
mod chat;
mod job;
mod models;
mod logs;
mod server;
mod settings;
mod shell;

use eframe::egui;
use egui_bench::prelude::*;
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
    Bench,
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
    gpu_at: Option<Instant>,
    pub notice: Option<(bool, String)>,
    pub chat: chat::ChatState,
    pub models: models::ModelsState,
    pub bench: bench::BenchState,
    pub server: server::ServerState,
    pub logs: inference_bench_core::logging::LogBuffer,
    pub logs_ui: logs::LogsState,
    pub settings_ui: settings::SettingsState,
}

impl App {
    pub fn new(dir: PathBuf, wake: Wake) -> Self {
        let mut app = Self::bare(dir, wake);
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
            bench: bench::BenchState::load(&dir),
            server: server::ServerState::default(),
            logs: inference_bench_core::logging::LogBuffer::new(5000),
            logs_ui: logs::LogsState::default(),
            settings_ui: settings::SettingsState::default(),
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

    pub fn load_model(&mut self, model: &LocalModel) {
        self.engine.send(Command::Load {
            model_id: model.model_id(inference_bench_core::target::NATIVE),
            display: model.name.clone(),
            max_ctx: Some(self.settings.specrt_ctx),
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
        self.bench.pump(&self.dir, &mut self.notice);
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
            || self.bench.running()
            || self.models.downloading()
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
