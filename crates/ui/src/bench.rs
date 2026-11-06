use crate::App;
use crate::chart;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::bench::{self, BenchModel, BenchPlan, Cell};
use inference_bench_core::report::{Grid, Metric, bars, megabytes, millis};
use inference_bench_core::result::{ErrorKind, RunResult};
use inference_bench_core::store;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::Arc;

pub enum Progress {
    Started(usize, Cell),
    Result(usize, Cell, RunResult),
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum View {
    Setup,
    Results,
    Compare,
}

pub struct BenchState {
    pub models: BTreeSet<String>,
    pub backends: BTreeSet<String>,
    pub lengths: Vec<(usize, bool)>,
    pub max_tokens: usize,
    pub view: View,
    pub metric: Metric,
    pub results: Vec<RunResult>,
    pub sessions: Vec<(String, Vec<RunResult>)>,
    pub left: usize,
    pub right: usize,
    pub total: usize,
    pub current: Option<Cell>,
    rx: Option<Receiver<Progress>>,
    cancel: Arc<AtomicBool>,
    pub export_note: Option<String>,
}

impl BenchState {
    pub fn load(dir: &Path) -> Self {
        let sessions = bench::load_sessions(dir);
        let results = sessions.last().map(|s| s.1.clone()).unwrap_or_default();
        Self {
            models: BTreeSet::new(),
            backends: BTreeSet::new(),
            lengths: vec![(128, true), (512, true), (2048, false), (8192, false)],
            max_tokens: 64,
            view: if results.is_empty() { View::Setup } else { View::Results },
            metric: Metric::DecodeTokS,
            results,
            left: sessions.len().saturating_sub(2),
            right: sessions.len().saturating_sub(1),
            sessions,
            total: 0,
            current: None,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            export_note: None,
        }
    }

    pub fn running(&self) -> bool {
        self.rx.is_some()
    }

    pub fn done_count(&self) -> usize {
        self.results.len()
    }

    pub fn start(&mut self, plan: BenchPlan, wake: inference_bench_core::engine::Wake) {
        let (tx, rx) = channel();
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        self.total = plan.cells().len();
        self.results.clear();
        self.view = View::Results;
        self.rx = Some(rx);
        std::thread::spawn(move || {
            let (a, b) = (tx.clone(), tx.clone());
            let (wa, wb) = (wake.clone(), wake.clone());
            bench::run_plan(
                &plan,
                &cancel,
                &mut |i, c| {
                    let _ = a.send(Progress::Started(i, c.clone()));
                    wa();
                },
                &mut |i, c, r| {
                    let _ = b.send(Progress::Result(i, c.clone(), r));
                    wb();
                },
            );
            let _ = tx.send(Progress::Done);
            wake();
        });
    }

    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn pump(&mut self, dir: &Path, notice: &mut Option<(bool, String)>) {
        let Some(rx) = &self.rx else { return };
        let mut finished = false;
        while let Ok(p) = rx.try_recv() {
            match p {
                Progress::Started(_, c) => self.current = Some(c),
                Progress::Result(_, _, r) => {
                    let _ = store::save(dir, &r);
                    self.results.push(r);
                }
                Progress::Done => finished = true,
            }
        }
        if finished {
            self.rx = None;
            self.current = None;
            let failed = self.results.iter().filter(|r| r.error.is_some()).count();
            *notice = Some((failed == 0, format!("bench finished: {} cells, {failed} failed", self.results.len())));
            if let Ok(name) = bench::save_session(dir, &self.results) {
                self.sessions.push((name, self.results.clone()));
                self.right = self.sessions.len() - 1;
                self.left = self.right.saturating_sub(1);
            }
        }
    }
}

pub fn build_plan(app: &App) -> Option<BenchPlan> {
    let st = &app.bench;
    let backends: Vec<(String, _)> = st
        .backends
        .iter()
        .filter_map(|b| app.settings.config_for(b).map(|c| (b.clone(), c)))
        .collect();
    let models: Vec<BenchModel> = app
        .library
        .iter()
        .filter(|m| st.models.contains(&m.name))
        .map(|m| {
            let mut ids = Vec::new();
            let mut skips = Vec::new();
            for (label, _) in &backends {
                let lb = app.settings.library_backend(label);
                match m.support_for(&lb) {
                    Some(s) if s.supported => ids.push((label.clone(), m.model_id(&lb))),
                    Some(s) => skips.push((label.clone(), format!("unsupported architecture: {}", s.reason))),
                    None => {}
                }
            }
            BenchModel { name: m.name.clone(), ids, skips }
        })
        .collect();
    let lengths: Vec<usize> = st.lengths.iter().filter(|(_, on)| *on).map(|(l, _)| *l).collect();
    (!backends.is_empty() && !models.is_empty() && !lengths.is_empty()).then_some(BenchPlan {
        models,
        backends,
        prompt_lengths: lengths,
        max_tokens: st.max_tokens,
    })
}

fn toggle_set(ui: &mut egui::Ui, set: &mut BTreeSet<String>, name: &str, tint: egui::Color32) {
    let on = set.contains(name);
    let r = chip(ui, name, if on { tint } else { LEGEND }, on).interact(egui::Sense::click());
    if r.clicked() {
        if on {
            set.remove(name);
        } else {
            set.insert(name.to_string());
        }
    }
}

fn setup(app: &mut App, ui: &mut egui::Ui) {
    section(ui, "models", "pick the models to run", |ui| {
        ui.horizontal_wrapped(|ui| {
            let names: Vec<String> = app.library.iter().map(|m| m.name.clone()).collect();
            for n in &names {
                toggle_set(ui, &mut app.bench.models, n, READOUT);
            }
            if names.is_empty() {
                note(ui, "the library is empty", LEGEND);
            }
        });
    });
    section(ui, "backends", "each one is loaded in turn", |ui| {
        ui.horizontal_wrapped(|ui| {
            for b in app.settings.backend_names() {
                toggle_set(ui, &mut app.bench.backends, &b, READOUT);
            }
        });
    });
    section(ui, "prompt lengths", "tokens, built from a synthetic text", |ui| {
        ui.horizontal_wrapped(|ui| {
            for (len, on) in &mut app.bench.lengths {
                if chip(ui, &format!("{len}"), if *on { READOUT } else { LEGEND }, *on).interact(egui::Sense::click()).clicked() {
                    *on = !*on;
                }
            }
        });
        row(ui, "new tokens", |ui| {
            ui.add(egui::DragValue::new(&mut app.bench.max_tokens).range(2..=4096));
        });
    });
    let plan = build_plan(app);
    let cells = plan.as_ref().map(|p| p.cells().len()).unwrap_or(0);
    ui.horizontal(|ui| {
        if app.bench.running() {
            if ui.add(Key::new("stop").solid(FAULT).min_width(140.0)).clicked() {
                app.bench.stop();
            }
        } else if ui.add(Key::new(format!("run {cells} cells")).solid(READOUT).enabled(plan.is_some()).min_width(140.0)).clicked()
            && let Some(p) = plan
        {
            app.bench.start(p, app.wake.clone());
        }
        if cells == 0 {
            hint(ui, "choose at least one model, backend and prompt length");
        }
    });
}

fn progress_card(app: &mut App, ui: &mut egui::Ui) {
    if !app.bench.running() {
        return;
    }
    let done = app.bench.done_count() as f32;
    let total = app.bench.total as f32;
    section(ui, "running", "results appear as each cell finishes", |ui| {
        let said = match &app.bench.current {
            Some(c) => format!("{} on {} at {} tokens", c.model, c.backend, c.prompt_len),
            None => "starting".into(),
        };
        progress(ui, "cells", done, Some(total), &said);
        if ui.add(Key::new("stop").outline(FAULT)).clicked() {
            app.bench.stop();
        }
    });
}

fn cell_text(r: &RunResult, metric: Metric) -> (String, egui::Color32) {
    if let Some(e) = &r.error {
        let tint = if e.kind == ErrorKind::UnsupportedArchitecture { WARN } else { FAULT };
        return (e.kind.label().to_string(), tint);
    }
    match metric.value(r) {
        Some(v) if metric == Metric::TtftMs => (format!("{v:.0} ms"), TRACE),
        Some(v) => (format!("{v:.1} tok/s"), TRACE),
        None => ("no data".into(), LEGEND),
    }
}

fn results(app: &mut App, ui: &mut egui::Ui) {
    let options: Vec<(Metric, &str, Option<usize>)> = Metric::ALL.iter().map(|m| (*m, m.label(), None)).collect();
    ui.push_id("metric", |ui| segments(ui, &mut app.bench.metric, &options));
    let metric = app.bench.metric;
    if app.bench.results.is_empty() {
        note(ui, "no results yet: set up a run", LEGEND);
        return;
    }
    let grid = Grid::latest(&app.bench.results);
    let width = ((ui.available_width() - 200.0) / grid.backends.len().max(1) as f32).clamp(110.0, 220.0);
    let mut columns: Vec<(&str, f32)> = vec![("model @ prompt", 200.0)];
    columns.extend(grid.backends.iter().map(|b| (b.as_str(), width)));
    section(ui, metric.label(), "latest cell per model, backend and prompt length", |ui| {
        Table::new(&columns, grid.models.len()).show(ui, |row, p, rect, at| {
            cell(p, rect, at(0), 200.0, &grid.models[row], VALUE);
            for b in 0..grid.backends.len() {
                let (text, tint) = match grid.cell(row, b) {
                    Some(r) => cell_text(r, metric),
                    None => ("not run".into(), LEGEND),
                };
                cell(p, rect, at(b + 1), width, &text, tint);
            }
        });
    });
    section(ui, "chart", "best cell in each group is bright", |ui| {
        chart::bars(ui, &bars(&grid, metric), metric, grid.backends.len());
    });
    detail_table(app, ui);
    export_row(app, ui);
}

fn detail_table(app: &mut App, ui: &mut egui::Ui) {
    let cols: [(&str, f32); 7] = [("model", 190.0), ("backend", 120.0), ("prefill", 110.0), ("decode", 110.0), ("ttft", 90.0), ("memory", 100.0), ("tokens", 90.0)];
    let rows: Vec<&RunResult> = app.bench.results.iter().filter(|r| r.error.is_none()).collect();
    if rows.is_empty() {
        return;
    }
    section(ui, "details", "every finished cell", |ui| {
        Table::new(&cols, rows.len()).show(ui, |i, p, rect, at| {
            let r = rows[i];
            let rate = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or("-".into());
            let f = [
                (r.model.clone(), VALUE),
                (r.backend.clone(), VALUE),
                (rate(r.prefill_tok_s), TRACE),
                (rate(r.decode_tok_s), TRACE),
                (r.ttft_secs.map(millis).unwrap_or("-".into()), TRACE),
                (r.peak_memory_bytes.map(megabytes).unwrap_or("-".into()), TRACE),
                (format!("{} + {}", r.prompt_tokens, r.completion_tokens), VALUE),
            ];
            for (n, (text, tint)) in f.iter().enumerate() {
                cell(p, rect, at(n), cols[n].1, text, *tint);
            }
        });
    });
}

fn export_row(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let dir = app.dir.join("exports");
        let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        if ui.add(Key::new("export json").outline(VALUE)).clicked() {
            let path = dir.join(format!("bench-{stamp}.json"));
            let text = serde_json::to_string_pretty(&app.bench.results).unwrap_or_default();
            app.bench.export_note = Some(write_export(&path, &text));
        }
        if ui.add(Key::new("export csv").outline(VALUE)).clicked() {
            let path = dir.join(format!("bench-{stamp}.csv"));
            app.bench.export_note = Some(write_export(&path, &bench::csv(&app.bench.results)));
        }
        if let Some(n) = &app.bench.export_note {
            Line::new().note(n.clone()).size(11.0).elided(ui);
        }
    });
}

fn write_export(path: &Path, text: &str) -> String {
    let made = path.parent().map(std::fs::create_dir_all).transpose();
    match made.and_then(|_| std::fs::write(path, text)) {
        Ok(()) => format!("wrote {}", path.display()),
        Err(e) => format!("export failed: {e}"),
    }
}

fn compare(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.bench;
    if st.sessions.len() < 2 {
        note(ui, "run the bench twice to compare two runs side by side", LEGEND);
        return;
    }
    let options: Vec<(usize, String)> = st.sessions.iter().enumerate().map(|(i, s)| (i, s.0.clone())).collect();
    ui.horizontal(|ui| {
        ui.allocate_ui(egui::vec2(260.0, KEY_H), |ui| {
            choice(ui, "cmp-left", &mut st.left, options.clone());
        });
        Line::new().legend("against").show(ui);
        ui.allocate_ui(egui::vec2(260.0, KEY_H), |ui| {
            choice(ui, "cmp-right", &mut st.right, options.clone());
        });
    });
    let metric = st.metric;
    let options: Vec<(Metric, &str, Option<usize>)> = Metric::ALL.iter().map(|m| (*m, m.label(), None)).collect();
    ui.push_id("cmp-metric", |ui| segments(ui, &mut st.metric, &options));
    let (Some(l), Some(r)) = (st.sessions.get(st.left), st.sessions.get(st.right)) else { return };
    let deltas = bench::compare(&l.1, &r.1, |x| x.error.is_none().then(|| metric.value(x)).flatten());
    let cols: [(&str, f32); 5] = [("cell", 300.0), ("left", 110.0), ("right", 110.0), ("change", 100.0), ("backend", 120.0)];
    section(ui, "side by side", metric.label(), |ui| {
        Table::new(&cols, deltas.len()).show(ui, |i, p, rect, at| {
            let d = &deltas[i];
            let fmt = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or("-".into());
            let change = d.change_percent();
            let better = change.is_some_and(|c| if metric.higher_is_better() { c >= 0.0 } else { c <= 0.0 });
            cell(p, rect, at(0), 300.0, &d.model, VALUE);
            cell(p, rect, at(1), 110.0, &fmt(d.left), TRACE);
            cell(p, rect, at(2), 110.0, &fmt(d.right), TRACE);
            cell(p, rect, at(3), 100.0, &change.map(|c| format!("{c:+.1}%")).unwrap_or("-".into()), if better { OK } else { FAULT });
            cell(p, rect, at(4), 120.0, &d.backend, LEGEND);
        });
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "bench", |ui| {
        let mut view = app.bench.view;
        ui.push_id("view", |ui| segments(ui, &mut view, &[(View::Setup, "setup", None), (View::Results, "results", Some(app.bench.results.len())), (View::Compare, "compare", Some(app.bench.sessions.len()))]));
        app.bench.view = view;
        progress_card(app, ui);
        match view {
            View::Setup => setup(app, ui),
            View::Results => results(app, ui),
            View::Compare => compare(app, ui),
        }
    });
}
