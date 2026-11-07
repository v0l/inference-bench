use crate::job::{self, Pending};
use crate::shell::page;
use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::hf::{self, Hub, Progress, Repo, RepoFile};
use inference_bench_core::library::{self, BACKENDS, LocalModel, human_size};
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    Library,
    Discover,
}

pub struct Download {
    pub repo: String,
    pub file: String,
    pub progress: Progress,
    pub result: Option<Result<(), String>>,
    rx: Receiver<Result<(), String>>,
}

pub struct ModelsState {
    pub pane: Pane,
    pub query: String,
    pub results: Vec<Repo>,
    search: Pending<Result<Vec<Repo>, String>>,
    pub picked: Option<String>,
    pub files: Vec<RepoFile>,
    listing: Pending<Result<Vec<RepoFile>, String>>,
    pub downloads: Vec<Download>,
    pub confirm_delete: Option<usize>,
    pub show_all: bool,
    pub hub_base: String,
    rescan: bool,
    pub message: Option<(bool, String)>,
}

impl Default for ModelsState {
    fn default() -> Self {
        Self {
            pane: Pane::Library,
            query: String::new(),
            results: Vec::new(),
            search: Pending::default(),
            picked: None,
            files: Vec::new(),
            listing: Pending::default(),
            downloads: Vec::new(),
            confirm_delete: None,
            show_all: false,
            hub_base: hf::HF_BASE.to_string(),
            rescan: false,
            message: None,
        }
    }
}

impl ModelsState {
    pub fn downloading(&self) -> bool {
        self.downloads.iter().any(|d| d.result.is_none()) || self.search.busy() || self.listing.busy()
    }

    pub fn take_rescan(&mut self) -> bool {
        std::mem::take(&mut self.rescan)
    }

    pub fn pump(&mut self, notice: &mut Option<(bool, String)>) {
        if let Some(r) = self.search.poll() {
            match r {
                Ok(list) => {
                    self.message = Some((true, format!("{} repos", list.len())));
                    self.results = list;
                }
                Err(e) => self.message = Some((false, e)),
            }
        }
        if let Some(r) = self.listing.poll() {
            match r {
                Ok(files) => self.files = files,
                Err(e) => self.message = Some((false, e)),
            }
        }
        let mut finished = false;
        for d in &mut self.downloads {
            if d.result.is_none()
                && let Ok(r) = d.rx.try_recv()
            {
                if r.is_ok() {
                    finished = true;
                    *notice = Some((true, format!("downloaded {}", d.file)));
                } else if let Err(e) = &r
                    && e != "paused"
                {
                    *notice = Some((false, format!("{}: {e}", d.file)));
                }
                d.result = Some(r);
            }
        }
        if finished {
            self.rescan = true;
        }
    }

    fn start_search(&mut self, app_wake: &inference_bench_core::engine::Wake, token: &str) {
        let (base, token, q) = (self.hub_base.clone(), token.to_string(), self.query.trim().to_string());
        self.search.start(job::spawn(app_wake, move || Hub::new(&base, &token).search(&q, 25)));
    }

    fn open_repo(&mut self, wake: &inference_bench_core::engine::Wake, token: &str, repo: &str) {
        self.picked = Some(repo.to_string());
        self.files.clear();
        let (base, token, repo) = (self.hub_base.clone(), token.to_string(), repo.to_string());
        self.listing.start(job::spawn(wake, move || Hub::new(&base, &token).files(&repo)));
    }

    pub fn start_download(
        &mut self,
        wake: &inference_bench_core::engine::Wake,
        token: &str,
        repo: &str,
        files: Vec<String>,
    ) {
        let progress = Progress::default();
        let (tx, rx) = channel();
        let (base, token, hub_dir) = (self.hub_base.clone(), token.to_string(), library::hub_dir());
        let (r, p, w) = (repo.to_string(), progress.clone(), wake.clone());
        let label = files.first().cloned().unwrap_or_default();
        std::thread::spawn(move || {
            let hub = Hub::new(&base, &token);
            let mut outcome = Ok(());
            for f in &files {
                if let Err(e) = hub.download(&r, f, &hub_dir, &p) {
                    outcome = Err(e);
                    break;
                }
                w();
            }
            let _ = tx.send(outcome);
            w();
        });
        self.downloads.push(Download { repo: repo.to_string(), file: label, progress, result: None, rx });
    }
}

pub fn reveal(model: &LocalModel) {
    let dir = model.reveal_dir();
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(opener).arg(dir).spawn();
}

pub fn backend_chip(ui: &mut egui::Ui, model: &LocalModel, backend: &str) {
    let Some(s) = model.support_for(backend) else { return };
    let (tint, lit) = if s.supported { (OK, true) } else { (WARN, false) };
    let text = if s.supported { backend.to_string() } else { format!("{backend}: no") };
    let resp = chip(ui, &text, tint, lit);
    let label = format!("{backend} {}", if s.supported { "supported" } else { "unsupported architecture" });
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, label.clone()));
    resp.on_hover_text(if s.supported {
        format!("supported: {}", s.reason)
    } else {
        format!("unsupported architecture ({})", s.reason)
    });
}

fn model_card(app: &mut App, ui: &mut egui::Ui, i: usize) {
    let m = app.library[i].clone();
    let loaded = app.loaded().is_some_and(|l| l.display == m.name);
    card(
        ui,
        if loaded { Some(OK) } else { None },
        |ui| {
            ui.label(egui::RichText::new(&m.name).font(legend_font(13.5)).color(VALUE));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                chip(ui, m.format.label(), LEGEND, false);
                if loaded {
                    chip(ui, "loaded", OK, true);
                }
            });
        },
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                Line::new().legend("size").value(human_size(m.size_bytes)).show(ui);
                Line::new().legend("arch").value(m.architecture.clone()).show(ui);
                Line::new().legend("quant").value(m.quant.clone()).show(ui);
                Line::new()
                    .legend("context")
                    .value(m.context_length.map(|c| format!("{c}")).unwrap_or("-".into()))
                    .show(ui);
            });
            ui.horizontal_wrapped(|ui| {
                for b in BACKENDS {
                    backend_chip(ui, &m, b);
                }
            });
            if let Some(why) = m.support.iter().find(|s| !s.supported && s.backend == "specrt" && m.format != library::Format::Gguf) {
                Line::new().note(format!("specrt: {}", why.reason)).size(10.5).elided(ui);
            }
            ui.horizontal(|ui| {
                if m.support_for("specrt").is_some_and(|s| s.supported) {
                    let label = if loaded { "reload" } else { "load" };
                    if ui.add(Key::new(label).solid(READOUT).id_salt(("load", i))).clicked() {
                        app.load_model(&m);
                        app.tab = crate::Tab::Chat;
                        app.chat.pick(&m.name);
                    }
                }
                if ui.add(Key::new("reveal").outline(VALUE).id_salt(("reveal", i))).clicked() {
                    reveal(&m);
                }
                if ui.add(Key::new("delete").outline(FAULT).id_salt(("del", i))).clicked() {
                    app.models.confirm_delete = Some(i);
                }
            });
        },
    );
}

fn library_pane(app: &mut App, ui: &mut egui::Ui) {
    if app.scan.busy() && app.library.is_empty() {
        note(ui, "scanning the hub cache and model folders", LEGEND);
    }
    let n = app.library.len();
    let usable_at = |m: &LocalModel| m.support.iter().any(|s| s.supported);
    let usable = app.library.iter().filter(|m| usable_at(m)).count();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| !usable_at(&app.library[i]));
    ui.horizontal(|ui| {
        Line::new().legend("models").value(n.to_string()).gap(14.0).legend("usable").value(usable.to_string()).show(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut all = app.models.show_all;
            if switch(ui, "show unusable", &mut all, "", "") {
                app.models.show_all = all;
            }
        });
    });
    for i in order {
        if app.models.show_all || usable_at(&app.library[i]) {
            model_card(app, ui, i);
        }
    }
    if n == 0 && !app.scan.busy() {
        note(ui, "no models found: download one from discover, or add a folder in settings", LEGEND);
    }
}

fn discover_pane(app: &mut App, ui: &mut egui::Ui) {
    let mut go = false;
    let token = app.settings.hf_token.clone();
    let busy = app.models.search.busy();
    field_then(ui, &mut app.models.query, "search Hugging Face, e.g. qwen3 gguf", 110.0, |ui| {
        go = ui.add(Key::new(if busy { "searching" } else { "search" }).solid(READOUT).enabled(!busy)).clicked();
    });
    if go {
        app.models.start_search(&app.wake, &token);
    }
    if let Some((ok, text)) = &app.models.message {
        status(ui, *ok, text);
    }
    let results = app.models.results.clone();
    let picked = app.models.picked.clone();
    for r in &results {
        let on = picked.as_deref() == Some(r.id.as_str());
        card(
            ui,
            if on { Some(READOUT) } else { None },
            |ui| {
                ui.label(egui::RichText::new(&r.id).font(legend_font(13.0)).color(VALUE));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    Line::new().legend("downloads").measured(r.downloads.to_string()).gap(10.0).legend("likes").measured(r.likes.to_string()).size(11.0).show(ui);
                });
            },
            |ui| {
                if !on {
                    if ui.add(Key::new("files").outline(VALUE).id_salt(("files", &r.id))).clicked() {
                        let wake = app.wake.clone();
                        app.models.open_repo(&wake, &token, &r.id);
                    }
                    return;
                }
                if app.models.listing.busy() {
                    note(ui, "listing files", LEGEND);
                }
                let files: Vec<RepoFile> = app.models.files.iter().filter(|f| hf::is_weight_file(&f.path)).cloned().collect();
                let companions: Vec<String> = hf::companion_files(&app.models.files).iter().map(|f| f.path.clone()).collect();
                for f in &files {
                    ui.horizontal(|ui| {
                        Line::new().value(f.path.clone()).gap(10.0).measured(human_size(f.size)).elided(ui);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let running = app.models.downloads.iter().any(|d| d.file == f.path && d.result.is_none());
                            let key = Key::new(if running { "downloading" } else { "download" }).solid(READOUT).enabled(!running).id_salt(("dl", &f.path));
                            if ui.add(key).clicked() {
                                let mut all = companions.clone();
                                all.push(f.path.clone());
                                let wake = app.wake.clone();
                                app.models.start_download(&wake, &token, &r.id, all);
                            }
                        });
                    });
                }
                if files.is_empty() && !app.models.listing.busy() {
                    note(ui, "no GGUF or safetensors files in this repo", LEGEND);
                }
            },
        );
    }
    downloads_section(app, ui);
}

fn downloads_section(app: &mut App, ui: &mut egui::Ui) {
    if app.models.downloads.is_empty() {
        return;
    }
    section(ui, "downloads", "resumes from the .part file if interrupted", |ui| {
        let mut retry = None;
        for (i, d) in app.models.downloads.iter().enumerate() {
            let done = d.progress.done.load(std::sync::atomic::Ordering::Relaxed) as f32;
            let total = d.progress.total.load(std::sync::atomic::Ordering::Relaxed) as f32;
            Line::new().legend(&d.repo).gap(8.0).value(d.file.clone()).elided(ui);
            match &d.result {
                None => progress(ui, "file", done, Some(total), &format!("{} / {}", human_size(done as u64), human_size(total as u64))),
                Some(Ok(())) => status(ui, true, "done"),
                Some(Err(e)) => {
                    status(ui, false, e);
                    if ui.add(Key::new("resume").outline(READOUT).id_salt(("resume", i))).clicked() {
                        retry = Some(i);
                    }
                }
            }
            if d.result.is_none() && ui.add(Key::new("pause").outline(VALUE).id_salt(("pause", i))).clicked() {
                d.progress.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if let Some(i) = retry {
            let d = app.models.downloads.remove(i);
            let companions: Vec<String> = hf::companion_files(&app.models.files).iter().map(|f| f.path.clone()).collect();
            let mut all = companions;
            all.push(d.file);
            let (wake, token) = (app.wake.clone(), app.settings.hf_token.clone());
            app.models.start_download(&wake, &token, &d.repo, all);
        }
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "models", |ui| {
        let mut pane = app.models.pane;
        segments(ui, &mut pane, &[(Pane::Library, "my models", Some(app.library.len())), (Pane::Discover, "discover", None)]);
        app.models.pane = pane;
        match pane {
            Pane::Library => library_pane(app, ui),
            Pane::Discover => discover_pane(app, ui),
        }
    });
}

pub fn modals(app: &mut App, ctx: &egui::Context) {
    let Some(i) = app.models.confirm_delete else { return };
    let Some(m) = app.library.get(i).cloned() else {
        app.models.confirm_delete = None;
        return;
    };
    let mut remove = false;
    let mut close = false;
    let r = egui::Modal::new(egui::Id::new("delete-model"))
        .backdrop_color(egui::Color32::from_black_alpha(150))
        .show(ctx, |ui| {
            ui.set_width(400.0);
            modal_title(ui, &format!("Delete {}?", m.name));
            note(ui, format!("{} of files are removed from disk. This cannot be undone.", human_size(m.size_bytes)), FAULT);
            footer(ui, |ui| {
                if ui.add(Key::new("delete").solid(FAULT).min_width(100.0)).clicked() {
                    remove = true;
                }
                if ui.add(Key::new("keep it").outline(VALUE).min_width(100.0)).clicked() {
                    close = true;
                }
            });
        });
    if r.should_close() {
        close = true;
    }
    if remove {
        match library::delete(&m) {
            Ok(()) => app.notice = Some((true, format!("deleted {}", m.name))),
            Err(e) => app.notice = Some((false, format!("delete failed: {e}"))),
        }
        app.rescan();
    }
    if remove || close {
        app.models.confirm_delete = None;
    }
}
