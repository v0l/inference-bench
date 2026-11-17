use crate::job::{self, Pending};
use crate::shell::page;
use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::discover::{self, Kv, RepoInfo, RepoPlan, Scored, Verdict};
use inference_bench_core::hf::{self, Hub, Progress, Repo, RepoFile};
use inference_bench_core::library::{self, LocalModel, Relation, human_size};
use inference_bench_core::target::NATIVE;
use std::collections::BTreeMap;
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
    files: Vec<RepoFile>,
    fitted: Option<(usize, Kv)>,
    rx: Receiver<Result<String, String>>,
}

pub struct Preselect {
    pub model: String,
    pub context: usize,
    pub kv: Kv,
}

pub struct PlanRow {
    pub repo: Repo,
    pub plan: Option<RepoPlan>,
    pub height: Option<f32>,
}

#[derive(PartialEq)]
struct PlanKey {
    generation: u64,
    results: usize,
    infos: usize,
    seqs: usize,
    gpus: Vec<u64>,
}

const BUDGET_BUCKET: u64 = 256 << 20;

type Inspected = Vec<(String, Result<RepoInfo, String>)>;

pub struct ModelsState {
    pub pane: Pane,
    pub query: String,
    pub library_query: String,
    pub results: Vec<Repo>,
    search: Pending<Result<Vec<Repo>, String>>,
    pub infos: BTreeMap<String, Result<RepoInfo, String>>,
    inspecting: Pending<Inspected>,
    pub picks: BTreeMap<String, String>,
    pub preselect: Vec<Preselect>,
    pub downloads: Vec<Download>,
    pub confirm_delete: Option<usize>,
    pub show_all: bool,
    pub hub_base: String,
    rescan: bool,
    pub message: Option<(bool, String)>,
    pub detail: Option<crate::detail::Detail>,
    pub relation: Option<(Relation, String)>,
    pub info_dir: std::path::PathBuf,
    generation: u64,
    plan_key: Option<PlanKey>,
    rows: Vec<PlanRow>,
    pub plan_runs: usize,
}

impl Default for ModelsState {
    fn default() -> Self {
        Self {
            pane: Pane::Library,
            query: String::new(),
            library_query: String::new(),
            results: Vec::new(),
            search: Pending::default(),
            infos: BTreeMap::new(),
            inspecting: Pending::default(),
            picks: BTreeMap::new(),
            preselect: Vec::new(),
            downloads: Vec::new(),
            confirm_delete: None,
            show_all: false,
            hub_base: hf::HF_BASE.to_string(),
            rescan: false,
            message: None,
            detail: None,
            relation: None,
            info_dir: std::path::PathBuf::new(),
            generation: 0,
            plan_key: None,
            rows: Vec::new(),
            plan_runs: 0,
        }
    }
}

impl ModelsState {
    pub fn with_info_dir(info_dir: std::path::PathBuf) -> Self {
        Self { info_dir, ..Self::default() }
    }

    pub fn downloading(&self) -> bool {
        self.downloads.iter().any(|d| d.result.is_none()) || self.search.busy() || self.inspecting.busy()
    }

    pub fn take_rescan(&mut self) -> bool {
        std::mem::take(&mut self.rescan)
    }

    pub fn pump(&mut self, wake: &inference_bench_core::engine::Wake, token: &str, notice: &mut Option<(bool, String)>) {
        if let Some(r) = self.search.poll() {
            match r {
                Ok(list) => {
                    self.message = Some((true, format!("{} repos", list.len())));
                    self.results = list;
                    self.generation += 1;
                    self.infos.clear();
                    self.picks.clear();
                    self.inspect_results(wake, token);
                }
                Err(e) => self.message = Some((false, e)),
            }
        }
        if let Some(found) = self.inspecting.poll() {
            self.generation += 1;
            self.infos.extend(found);
        }
        let mut finished = false;
        for d in &mut self.downloads {
            if d.result.is_none()
                && let Ok(r) = d.rx.try_recv()
            {
                match &r {
                    Ok(model) => {
                        finished = true;
                        *notice = Some((true, format!("downloaded {}", d.file)));
                        if let Some((context, kv)) = d.fitted {
                            self.preselect.push(Preselect { model: model.clone(), context, kv });
                        }
                    }
                    Err(e) if e != "paused" => *notice = Some((false, format!("{}: {e}", d.file))),
                    Err(_) => {}
                }
                d.result = Some(r.map(|_| ()));
            }
        }
        if finished {
            self.rescan = true;
        }
    }

    fn start_search(&mut self, app_wake: &inference_bench_core::engine::Wake, token: &str) {
        let (base, token, q) = (self.hub_base.clone(), token.to_string(), self.query.trim().to_string());
        match self.relation.clone() {
            Some((relation, of)) => self.search.start(job::spawn(app_wake, move || Hub::new(&base, &token).children(&of, relation, 25))),
            None => self.search.start(job::spawn(app_wake, move || Hub::new(&base, &token).search(&q, 25))),
        }
    }

    pub fn discover_relation(&mut self, wake: &inference_bench_core::engine::Wake, token: &str, relation: Relation, of: &str) {
        self.relation = Some((relation, of.to_string()));
        self.pane = Pane::Discover;
        self.results.clear();
        self.start_search(wake, token);
    }

    fn inspect_results(&mut self, wake: &inference_bench_core::engine::Wake, token: &str) {
        let repos: Vec<String> = self.results.iter().map(|r| r.id.clone()).collect();
        let (base, token, dir) = (self.hub_base.clone(), token.to_string(), self.info_dir.clone());
        if repos.is_empty() {
            return;
        }
        self.inspecting.start(job::spawn(wake, move || discover::inspect_all(&Hub::new(&base, &token), &repos, &dir)));
    }

    pub fn plans(&self, gpus: &[u64], seqs: usize) -> Vec<RepoPlan> {
        let planned: Vec<RepoPlan> = self.infos.values().filter_map(|i| i.as_ref().ok()).map(|i| discover::plan(i, gpus, seqs)).collect();
        let ranked = discover::rank_repos(planned);
        let mut out: Vec<RepoPlan> = Vec::new();
        for r in &self.results {
            if let Some(p) = ranked.iter().find(|p| p.repo == r.id) {
                out.push(p.clone());
            }
        }
        out.sort_by_key(|p| p.key());
        out
    }

    pub fn refresh_rows(&mut self, gpus: &[u64], seqs: usize) {
        let key = PlanKey {
            generation: self.generation,
            results: self.results.len(),
            infos: self.infos.len(),
            seqs,
            gpus: gpus.iter().map(|g| g / BUDGET_BUCKET).collect(),
        };
        if self.plan_key.as_ref() == Some(&key) {
            return;
        }
        self.plan_runs += 1;
        let plans = self.plans(gpus, seqs);
        let mut by_repo: BTreeMap<String, RepoPlan> = plans.iter().map(|p| (p.repo.clone(), p.clone())).collect();
        let mut planned = Vec::new();
        for p in &plans {
            if let Some(r) = self.results.iter().find(|r| r.id == p.repo) {
                planned.push(PlanRow { repo: r.clone(), plan: by_repo.remove(&p.repo), height: None });
            }
        }
        let pending = self.results.iter().filter(|r| !plans.iter().any(|p| p.repo == r.id)).map(|r| PlanRow { repo: r.clone(), plan: None, height: None });
        planned.extend(pending);
        self.rows = planned;
        self.plan_key = Some(key);
    }

    pub fn start_download(&mut self, wake: &inference_bench_core::engine::Wake, token: &str, repo: &str, label: &str, files: Vec<RepoFile>, fitted: Option<(usize, Kv)>, first_weight: String, format: library::Format) {
        let progress = Progress::default();
        let (tx, rx) = channel();
        let (base, token, hub_dir) = (self.hub_base.clone(), token.to_string(), library::hub_dir());
        let (r, p, w, list) = (repo.to_string(), progress.clone(), wake.clone(), files.clone());
        std::thread::spawn(move || {
            let hub = Hub::new(&base, &token);
            let outcome = hub.download_variant(&r, &list, &hub_dir, &p).map(|_| {
                let first = hf::snapshot_path(&hub_dir, &r, &first_weight);
                let v = discover::Variant { label: String::new(), format, files: Vec::new(), expected_parts: 1, size: 0, bits: 0.0 };
                discover::model_name(&r, &v, &first)
            });
            let _ = tx.send(outcome);
            w();
        });
        self.downloads.push(Download { repo: repo.to_string(), file: label.to_string(), progress, result: None, files, fitted, rx });
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

pub fn matches(model: &LocalModel, query: &str) -> bool {
    let base = model.lineage.as_ref().map(|l| l.bases.join(" ")).unwrap_or_default();
    let haystack = format!("{} {} {} {} {} {}", model.name, model.repo, base, model.architecture, model.quant, model.format.label()).to_lowercase();
    query.split_whitespace().all(|word| haystack.contains(&word.to_lowercase()))
}

pub fn runs_here(model: &LocalModel) -> bool {
    model.support_for(NATIVE).is_some_and(|s| s.supported)
}

pub fn support_chip(ui: &mut egui::Ui, model: &LocalModel) {
    let Some(s) = model.support_for(NATIVE) else { return };
    let (tint, text) = if s.supported { (OK, "runs here") } else { (WARN, "not supported") };
    let resp = chip(ui, text, tint, s.supported);
    let label = if s.supported { "runs here" } else { "unsupported architecture" };
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, label));
    resp.on_hover_text(if s.supported { s.reason.clone() } else { format!("unsupported architecture ({})", s.reason) });
}

pub fn name_job(name: &str, size: f32, dim: bool) -> egui::text::LayoutJob {
    let (org, rest) = match name.split_once('/') {
        Some((org, rest)) => (format!("{org}/"), rest),
        None => (String::new(), name),
    };
    let mut job = egui::text::LayoutJob::default();
    let format = |color| egui::TextFormat { font_id: legend_font(size), color, ..Default::default() };
    job.append(&org, 0.0, format(LEGEND));
    job.append(rest, 0.0, format(if dim { LEGEND } else { VALUE }));
    job
}

pub fn is_loaded(app: &App, m: &LocalModel) -> bool {
    app.loaded().is_some_and(|l| l.model_id == m.model_id(NATIVE))
}

fn model_card(app: &mut App, ui: &mut egui::Ui, i: usize) {
    let m = app.library[i].clone();
    let loaded = is_loaded(app, &m);
    card(
        ui,
        if loaded { Some(OK) } else { None },
        |ui| {
            ui.label(name_job(&m.name, 13.5, false));
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
                support_chip(ui, &m);
                let tint = if m.vision { OK } else { LEGEND };
                let chip_text = if m.vision { "vision: yes" } else { "vision: no" };
                let r = chip(ui, chip_text, tint, m.vision);
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, chip_text));
            });
            if let Some(why) = m.support_for(NATIVE).filter(|s| !s.supported) {
                Line::new().note(why.reason.clone()).size(10.5).elided(ui);
            }
            if let Some(source) = &m.spec_source {
                Line::new().legend("spec").value(source.clone()).size(10.5).elided(ui);
            }
            if let Some(origin) = &m.origin {
                Line::new().legend("spec from").value(origin.label()).size(10.5).elided(ui);
            }
            ui.horizontal(|ui| {
                if runs_here(&m) {
                    let label = if loaded { "reload" } else { "load" };
                    if ui.add(Key::new(label).solid(READOUT).id_salt(("load", i))).clicked() {
                        app.open_load_dialog(&m);
                    }
                }
                if ui.add(Key::new("details").outline(READOUT).id_salt(("details", i))).clicked() {
                    app.models.detail = Some(crate::detail::Detail::open(&m));
                }
                if ui.add(Key::new("reveal").outline(VALUE).id_salt(("reveal", i))).clicked() {
                    reveal(&m);
                }
                if library::deletable(&m) && ui.add(Key::new("delete").outline(FAULT).id_salt(("del", i))).clicked() {
                    app.models.confirm_delete = Some(i);
                }
            });
        },
    );
}

fn specs_bar(app: &mut App, ui: &mut egui::Ui) {
    let state = app.specs.state.clone();
    let busy = app.specs.busy();
    let (ok, detail) = match (&app.specs.last, state.last_result.is_empty()) {
        (Some((ok, text)), _) => (*ok, text.clone()),
        (None, false) => (state.last_ok, state.last_result.clone()),
        (None, true) => (true, String::new()),
    };
    let label = if busy { "checking" } else { "check specs now" };
    ui.horizontal(|ui| {
        let reserve = Key::new("check specs now").width(ui);
        if ui.add(Key::new(label).outline(READOUT).enabled(!busy).min_width(reserve)).clicked() {
            let (wake, dir) = (app.wake.clone(), app.dir.clone());
            app.specs.check_now(&wake, &dir, &app.settings);
        }
        let verdict = crate::specs::verdict(ok, &detail);
        let tint = match (detail.is_empty(), ok) {
            (true, _) => LEGEND,
            (false, true) => OK,
            (false, false) => FAULT,
        };
        let hover = if detail.is_empty() { "the specs have not been checked yet".to_string() } else { detail };
        let line = Line::new()
            .legend("specs")
            .value(app.settings.specs().source())
            .gap(14.0)
            .legend("checked")
            .value(crate::specs::when(state.checked_at))
            .gap(14.0)
            .legend("result")
            .tint(tint)
            .value(verdict)
            .size(11.0)
            .elided(ui);
        line.on_hover_text(hover);
    });
}

fn library_pane(app: &mut App, ui: &mut egui::Ui) {
    if app.scan.busy() && app.library.is_empty() {
        note(ui, "scanning the hub cache and model folders", LEGEND);
    }
    let n = app.library.len();
    let usable = app.library.iter().filter(|m| runs_here(m)).count();
    let mut go_discover = false;
    field_then(ui, &mut app.models.library_query, "search my models by name, architecture or quant", 150.0, |ui| {
        go_discover = ui.add(Key::new("search hugging face").outline(READOUT).min_width(140.0)).clicked();
    });
    if go_discover {
        app.models.query = app.models.library_query.clone();
        app.models.pane = Pane::Discover;
        return;
    }
    let query = app.models.library_query.clone();
    let mut order: Vec<usize> = (0..n).filter(|&i| matches(&app.library[i], &query)).collect();
    order.sort_by_key(|&i| !runs_here(&app.library[i]));
    let visible: Vec<usize> = order.into_iter().filter(|&i| app.models.show_all || runs_here(&app.library[i])).collect();
    ui.horizontal(|ui| {
        Line::new()
            .legend("models")
            .value(n.to_string())
            .gap(14.0)
            .legend("shown")
            .value(visible.len().to_string())
            .gap(14.0)
            .legend("usable")
            .value(usable.to_string())
            .show(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut all = app.models.show_all;
            if switch(ui, "show unusable", &mut all, "", "") {
                app.models.show_all = all;
            }
        });
    });
    specs_bar(app, ui);
    for &i in &visible {
        model_card(app, ui, i);
    }
    if n == 0 && !app.scan.busy() {
        note(ui, "no models found: download one from discover, or add a folder in settings", LEGEND);
    } else if visible.is_empty() && !query.trim().is_empty() {
        note(ui, format!("no model matches \"{}\": try fewer words, or search Hugging Face", query.trim()), LEGEND);
    }
}

pub fn budget_line(gpus: &[inference_bench_core::gpu::GpuStats], seqs: usize) -> String {
    let available: u64 = gpus.iter().map(|g| g.available_bytes()).sum();
    let other: u64 = gpus.iter().map(|g| g.other_bytes()).sum();
    format!("{} available for a model ({} used by other processes), {seqs} parallel requests", human_size(available), human_size(other))
}

fn need_line(f: &discover::Assessment) -> String {
    format!("needs {}, {} free", human_size(f.need), f.free.map_or("-".into(), human_size))
}

fn best_line(s: &Scored) -> String {
    let Some(f) = &s.fit else { return String::new() };
    match f.verdict {
        Verdict::Unknown => format!("{} on the gpu, no gpu reading to check fit", human_size(f.weights)),
        Verdict::DoesNotFit => need_line(f),
        _ if f.full() => format!("full {} context, {}", discover::context_label(f.native), f.chosen.kv.label()),
        _ => format!("{} of {} context, {}", discover::context_label(f.chosen.context), discover::context_label(f.native), f.chosen.kv.label()),
    }
}

fn fits_somewhere(plan: &RepoPlan) -> bool {
    plan.ranked.iter().any(|s| s.downloadable() && s.fit.as_ref().is_some_and(|f| f.verdict != Verdict::DoesNotFit))
}

pub fn headline(plan: &RepoPlan) -> Option<String> {
    let s = plan.best()?;
    if plan.unsupported.is_some() || !s.downloadable() {
        return None;
    }
    let f = s.fit.as_ref()?;
    match f.verdict {
        Verdict::DoesNotFit => Some(format!("Does not fit: {}", need_line(f))),
        _ => Some(format!("Best fit: {}, {}", s.variant.label, best_line(s))),
    }
}

fn verdict_chip(ui: &mut egui::Ui, s: &Scored) {
    let Some(f) = &s.fit else { return };
    let (tint, lit) = match f.verdict {
        Verdict::Fits => (OK, true),
        Verdict::Tight => (WARN, true),
        Verdict::DoesNotFit => (FAULT, true),
        Verdict::Unknown => (LEGEND, false),
    };
    chip(ui, f.verdict.label(), tint, lit);
}

pub fn can_download(s: &Scored) -> bool {
    s.downloadable() && s.fit.as_ref().is_none_or(|f| f.verdict != Verdict::DoesNotFit)
}

fn repo_row(app: &mut App, ui: &mut egui::Ui, r: &Repo, plan: Option<&RepoPlan>, token: &str) {
    card(
        ui,
        None,
        |ui| {
            ui.label(name_job(&r.id, 13.0, false));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                Line::new().legend("downloads").measured(r.downloads.to_string()).gap(10.0).legend("likes").measured(r.likes.to_string()).size(11.0).show(ui);
            });
        },
        |ui| {
            let Some(plan) = plan else {
                match app.models.infos.get(&r.id) {
                    Some(Err(e)) => note(ui, e.clone(), LEGEND),
                    _ => note(ui, "reading the model header", LEGEND),
                }
                return;
            };
            let Some(best) = plan.best() else { return };
            let chosen_label = app.models.picks.get(&r.id).cloned().unwrap_or_else(|| best.variant.label.clone());
            let shown = plan.ranked.iter().find(|s| s.variant.label == chosen_label).unwrap_or(best).clone();
            if let Some(target) = &plan.draft {
                ui.horizontal_wrapped(|ui| {
                    chip(ui, "draft model", WARN, false);
                    note(ui, format!("draft model for {target}: speeds up its target model and cannot run alone"), LEGEND);
                });
                return;
            }
            if let Some(why) = plan.unsupported.clone().or_else(|| shown.runnable.clone().err()) {
                ui.horizontal_wrapped(|ui| {
                    chip(ui, "not supported", WARN, false);
                    note(ui, why, LEGEND);
                });
                return;
            }
            let fits = shown.fit.as_ref().is_none_or(|f| f.verdict != Verdict::DoesNotFit);
            if shown.variant.label == best.variant.label {
                if let Some(line) = headline(plan) {
                    let tint = if fits { VALUE } else { FAULT };
                    ui.label(egui::RichText::new(line).color(tint).size(12.5));
                }
            }
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                Line::new().legend("quant").value(shown.variant.label.clone()).show(ui);
                if let Some(f) = &shown.fit {
                    Line::new().legend("on gpu").value(human_size(f.weights)).show(ui);
                    if fits {
                        Line::new().legend("context").value(format!("{} of {}", discover::context_label(f.chosen.context), discover::context_label(f.native))).show(ui);
                        Line::new().legend("kv").value(f.chosen.kv.label()).show(ui);
                    }
                }
                verdict_chip(ui, &shown);
                chip(ui, "supported", OK, true);
            });
            if !fits && shown.variant.label != best.variant.label {
                if let Some(f) = &shown.fit {
                    note(ui, need_line(f), FAULT);
                }
            }
            let info = app.models.infos.get(&r.id).and_then(|i| i.as_ref().ok()).cloned();
            let Some(info) = info else { return };
            let running = app.models.downloads.iter().any(|d| d.repo == r.id && d.result.is_none());
            ui.horizontal(|ui| {
                let key = Key::new(if running { "downloading" } else { "download" }).solid(READOUT).enabled(!running && can_download(&shown)).id_salt(("dl", &r.id));
                if ui.add(key).clicked() {
                    let files = discover::download_list(&info.grouped, &shown.variant);
                    let first = shown.variant.files.first().map(|f| f.path.clone()).unwrap_or_default();
                    let fitted = shown.fit.as_ref().filter(|f| f.verdict != Verdict::DoesNotFit && f.verdict != Verdict::Unknown).map(|f| (f.chosen.context, f.chosen.kv));
                    let label = format!("{} {}", r.id, shown.variant.label);
                    let wake = app.wake.clone();
                    app.models.start_download(&wake, token, &r.id, &label, files, fitted, first, shown.variant.format);
                }
                if plan.ranked.len() > 1 && (fits_somewhere(plan) || !fits) {
                    let mut picked = shown.variant.label.clone();
                    let options: Vec<(String, String)> = plan
                        .ranked
                        .iter()
                        .filter(|s| s.runnable.is_ok())
                        .map(|s| {
                            let tail = s.fit.as_ref().map_or(String::new(), |f| format!(", {} {}", human_size(f.weights), f.verdict.label()));
                            (s.variant.label.clone(), format!("{}{tail}", s.variant.label))
                        })
                        .collect();
                    if options.len() > 1 && choice(ui, ("quant", &r.id), &mut picked, options) {
                        app.models.picks.insert(r.id.clone(), picked);
                    }
                }
            });
        },
    );
}

fn discover_pane(app: &mut App, ui: &mut egui::Ui) {
    let mut go = false;
    let token = app.settings.hf_token.clone();
    let busy = app.models.search.busy();
    field_then(ui, &mut app.models.query, "search Hugging Face, e.g. qwen3 gguf", 110.0, |ui| {
        go = ui.add(Key::new(if busy { "searching" } else { "search" }).solid(READOUT).enabled(!busy)).clicked();
    });
    if go {
        app.models.relation = None;
        app.models.start_search(&app.wake.clone(), &token);
    }
    if let Some((relation, of)) = app.models.relation.clone() {
        ui.horizontal(|ui| {
            Line::new().legend(relation.plural()).gap(8.0).legend("of").gap(4.0).value(of).show(ui);
            if ui.add(Key::new("clear filter").outline(VALUE)).clicked() {
                app.models.relation = None;
                app.models.results.clear();
            }
        });
    }
    if let Some((ok, text)) = &app.models.message {
        status(ui, *ok, text);
    }
    let gpus = app.free_gpu_list();
    if !gpus.is_empty() {
        note(ui, budget_line(&app.gpu, app.settings.default_seqs), LEGEND);
    }
    app.models.refresh_rows(&gpus, app.settings.default_seqs);
    let mut rows = std::mem::take(&mut app.models.rows);
    let gap = ui.spacing().item_spacing.y;
    let view = ui.clip_rect().expand(300.0);
    for row in &mut rows {
        let top = ui.cursor().min.y;
        if let Some(h) = row.height
            && (top + h < view.min.y || top > view.max.y)
        {
            ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
            continue;
        }
        repo_row(app, ui, &row.repo, row.plan.as_ref(), &token);
        row.height = Some(ui.cursor().min.y - top - gap);
    }
    app.models.rows = rows;
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
                None => progress(ui, "model", done, Some(total), &format!("{} / {}", human_size(done as u64), human_size(total as u64))),
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
            let (wake, token) = (app.wake.clone(), app.settings.hf_token.clone());
            let first = d.files.iter().find(|f| f.path.ends_with(".gguf") && !f.path.starts_with("mmproj")).or(d.files.last()).map(|f| f.path.clone()).unwrap_or_default();
            let format = if first.ends_with(".gguf") { library::Format::Gguf } else { library::Format::Safetensors };
            app.models.start_download(&wake, &token, &d.repo, &d.file, d.files, d.fitted, first, format);
        }
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "models", |ui| {
        if app.models.detail.is_some() {
            crate::detail::draw(app, ui);
            return;
        }
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
