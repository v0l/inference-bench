#![allow(deprecated, clippy::too_many_arguments)]

mod working;
mod chat;
mod breakdown;
mod table;
mod block;
mod detail;
mod engine;
mod fmt;
mod hardware;
mod strip;
mod job;
mod load;
mod loadedlist;
mod running;
mod modelbar;
mod models;
mod logs;
mod server;
mod serving;
mod sessions;
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
    Hardware,
    Settings,
}

#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub stage: String,
    pub done: u64,
    pub total: u64,
    pub weights_started: Option<Instant>,
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
    pub load_progress: Option<LoadProgress>,
    pub gpu: Vec<GpuStats>,
    gpu_poll: Pending<Vec<GpuStats>>,
    pub gpu_at: Option<Instant>,
    pub notice: Option<(bool, String)>,
    pub chat: chat::ChatState,
    pub models: models::ModelsState,
    pub server: server::ServerState,
    pub hardware: hardware::HardwareState,
    pub logs: inference_bench_core::logging::LogBuffer,
    pub logs_ui: logs::LogsState,
    pub settings_ui: settings::SettingsState,
    pub specs: specs::SpecsUpdater,
    pub load: Option<load::LoadDialog>,
    pub picker_query: String,
    pub chat_model: Option<String>,
}

impl App {
    pub fn new(dir: PathBuf, wake: Wake, ctx: &egui::Context) -> Self {
        let mut app = Self::bare(dir, wake, ctx);
        app.specs.auto = true;
        app.hardware.auto = true;
        app.rescan();
        app
    }

    pub fn bare(dir: PathBuf, wake: Wake, ctx: &egui::Context) -> Self {
        let settings = Settings::load(&dir);
        let hardware_slot = inference_bench_core::cluster::ClusterSlot::default();
        Self {
            tab: Tab::Chat,
            library: Vec::new(),
            scan: Pending::default(),
            engine: engine_for(&hardware_slot, wake.clone(), &dir),
            loading: None,
            load_progress: None,
            gpu: Vec::new(),
            gpu_poll: Pending::default(),
            gpu_at: None,
            notice: None,
            chat: chat::ChatState::new(ctx),
            models: models::ModelsState::with_info_dir(dir.join("repo-info")),
            server: server::ServerState::default(),
            hardware: hardware::HardwareState::with_slot(hardware_slot.clone()),
            logs: inference_bench_core::logging::LogBuffer::new(5000),
            logs_ui: logs::LogsState::default(),
            settings_ui: settings::SettingsState::default(),
            specs: {
                let mut s = specs::SpecsUpdater::default();
                s.load_state(&dir);
                s
            },
            load: None,
            picker_query: String::new(),
            chat_model: None,
            settings,
            wake,
            dir,
        }
    }

    pub fn rescan(&mut self) {
        let sources = Sources::standard(&self.dir, &self.settings.model_folders, &self.settings.spec_folders);
        self.scan.start(job::spawn(&self.wake, move || inference_bench_core::library::scan(&sources)));
    }

    pub fn landing(&self, l: &inference_bench_core::engine::Loaded) -> Option<inference_bench_core::inventory::Landing> {
        let planned = l.engine.solved.as_deref().map(inference_bench_core::inventory::planned_names).unwrap_or_default();
        self.inventory().where_landed(&l.landed, &planned)
    }

    pub fn loaded(&self) -> Option<inference_bench_core::engine::Loaded> {
        let all = self.engine.loaded_all();
        let chosen = self.chat_model.as_ref().and_then(|m| all.iter().find(|l| &l.display == m).cloned());
        chosen.or_else(|| all.into_iter().next())
    }

    pub fn loaded_all(&self) -> Vec<inference_bench_core::engine::Loaded> {
        self.engine.loaded_all()
    }

    pub fn max_loaded(&self) -> usize {
        self.settings.server.max_loaded_models.max(1)
    }

    pub fn alongside_blocked(&self) -> Option<String> {
        let n = self.engine.loaded_all().len();
        (n >= self.max_loaded()).then(|| format!("{n} of {} allowed models are loaded: unload one, or raise the limit on the Server tab", self.max_loaded()))
    }

    pub fn inventory(&self) -> inference_bench_core::inventory::Inventory {
        let peers: Vec<specrt::cluster::Peer> = self.hardware.peers.values().cloned().collect();
        inference_bench_core::inventory::inventory(self.hardware.machine.as_ref(), &self.gpu, &peers, &self.settings.cluster)
    }

    pub fn free_gpu_list(&self) -> Vec<u64> {
        self.inventory().free_local()
    }

    pub fn free_gpu_bytes(&self) -> Option<u64> {
        let list = self.free_gpu_list();
        (!list.is_empty()).then(|| list.iter().sum())
    }

    pub fn open_load_dialog(&mut self, model: &LocalModel) {
        self.load = Some(load::LoadDialog::open(self, model));
        self.models.preselect.retain(|p| p.model != model.name);
    }

    pub fn load_model(&mut self, model: &LocalModel) {
        let ctx = self.settings.ctx_for(&model.name);
        let seqs = self.settings.seqs_for(&model.name);
        let kv = self.settings.kv_gb_for(&model.name).map(|g| (g * (1u64 << 30) as f64) as u64);
        let served_names = self.settings.served_names_for(&model.name);
        let engine = self.settings.engine_for(&model.name);
        self.load_model_named(model, ctx, seqs, kv, served_names, engine);
    }

    pub fn load_model_with(&mut self, model: &LocalModel, ctx: usize, seqs: usize) {
        let served_names = self.settings.served_names_for(&model.name);
        let engine = self.settings.engine_for(&model.name);
        self.load_model_named(model, ctx, seqs, None, served_names, engine);
    }

    pub fn load_model_named(&mut self, model: &LocalModel, ctx: usize, seqs: usize, kv_cache_bytes: Option<u64>, served_names: Vec<String>, engine: specrt::options::EngineChoice) {
        self.load_model_placed(model, ctx, seqs, kv_cache_bytes, served_names, engine, false);
    }

    pub fn load_model_placed(&mut self, model: &LocalModel, ctx: usize, seqs: usize, kv_cache_bytes: Option<u64>, served_names: Vec<String>, engine: specrt::options::EngineChoice, alongside: bool) {
        self.hardware.release();
        self.chat_model = Some(model.name.clone());
        self.engine.set_max_loaded(self.max_loaded());
        self.engine.send(Command::Load {
            alongside,
            model_id: model.model_id(inference_bench_core::target::NATIVE),
            display: model.name.clone(),
            max_ctx: Some(ctx),
            max_num_seqs: Some(seqs),
            kv_cache_bytes,
            served_names,
            engine,
        });
    }

    pub fn unload(&mut self) {
        self.hardware.release();
        self.engine.send(Command::Unload { model: None });
    }

    pub fn unload_model(&mut self, display: &str) {
        self.hardware.release();
        self.engine.send(Command::Unload { model: Some(display.to_string()) });
    }

    fn pump(&mut self) {
        if let Some(found) = self.scan.poll() {
            self.library = found;
        }
        while let Ok(event) = self.engine.events.try_recv() {
            match event {
                EngineEvent::Loading(name) => {
                    self.chat.attempted = name.clone();
                    self.loading = Some(name);
                    self.load_progress = None;
                }
                EngineEvent::LoadProgress { stage, done, total } => {
                    let now = Instant::now();
                    let p = self.load_progress.get_or_insert_with(|| LoadProgress { stage: String::new(), done: 0, total: 0, weights_started: None });
                    if p.stage != stage {
                        p.weights_started = (stage == "reading weights").then_some(now);
                    }
                    p.stage = stage;
                    p.done = done;
                    p.total = total;
                }
                EngineEvent::Loaded(l) => {
                    self.loading = None;
                    self.load_progress = None;
                    self.chat.load_error = None;
                    let landing = self.landing(&l);
                    self.notice = Some(match landing {
                        Some(w) if !w.matches_plan => (false, format!("{} loaded in {:.1} s but is {}, not where the load dialog planned it", l.display, l.load_secs, w.text)),
                        Some(w) => (true, format!("{} loaded in {:.1} s, {}", l.display, l.load_secs, w.text)),
                        None => (true, format!("{} loaded in {:.1} s", l.display, l.load_secs)),
                    });
                }
                EngineEvent::LoadFailed(e) => {
                    self.loading = None;
                    self.load_progress = None;
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
        self.chat.agent.poll();
        self.models.pump(&self.wake, &self.settings.hf_token, &mut self.notice);
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
        self.engine.watch(self.tab == Tab::Server && self.loaded().is_some());
        root.ctx().request_repaint_after(Duration::from_millis(if animating { 100 } else { 2000 }));
        agent_ui::serve(self, &root.ctx().clone());
        shell::draw(self, root);
    }
}

fn engine_for(slot: &inference_bench_core::cluster::ClusterSlot, wake: Wake, dir: &std::path::Path) -> Engine {
    let sessions = inference_bench_server::ServerConfig::load(&server::config_path(dir)).ok().map(|c| c.sessions).filter(|s| s.enabled);
    let inner = inference_bench_core::cluster::serving_factory(slot.clone());
    let serving: inference_bench_core::engine::ServingFactory = Arc::new(move |mut opts| {
        opts.sessions = sessions.clone();
        inner(opts)
    });
    Engine::with_factories(wake, inference_bench_core::native::specrt_factory(), serving)
}

fn wake_for(ctx: &egui::Context) -> Wake {
    let ctx = ctx.clone();
    Arc::new(move || ctx.request_repaint())
}

fn main() -> eframe::Result<()> {
    if let Some(code) = agent_core::sandbox::run_helper() {
        std::process::exit(code);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().map(PathBuf::from).unwrap_or_else(store::data_dir);
    let buffer = inference_bench_core::logging::LogBuffer::new(5000);
    let log = inference_bench_core::logging::LogConfig {
        buffer: Some(buffer.clone()),
        file: Some(inference_bench_core::logging::FileLog::in_dir(inference_bench_core::logging::default_log_dir())),
        ..Default::default()
    };
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
            let mut app = App::new(dir, wake_for(&cc.egui_ctx), &cc.egui_ctx);
            app.logs = buffer;
            Ok(Box::new(app))
        }),
    )
}

#[cfg(test)]
mod tests;
