use crate::{App, chart};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::report::{Grid, Metric, bars, megabytes, millis};
use inference_bench_core::result::{ErrorKind, RunResult};

pub fn runner(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(legend("matrix file"));
        let busy = app.running();
        let mut go = false;
        field_then(ui, &mut app.matrix_path, "path to a matrix .toml", 120.0, |ui| {
            go = ui
                .add(Key::new(if busy { "running" } else { "run" }).solid(READOUT).enabled(!busy))
                .clicked();
        });
        if go {
            app.start_matrix();
        }
    });
}

pub fn cell_text(r: &RunResult) -> (String, egui::Color32) {
    if let Some(e) = &r.error {
        let tint = if e.kind == ErrorKind::UnsupportedArchitecture { WARN } else { FAULT };
        return (e.kind.label().to_string(), tint);
    }
    match r.decode_tok_s {
        Some(v) => (format!("{v:.1} tok/s"), TRACE),
        None => ("no decode".to_string(), LEGEND),
    }
}

pub fn matrix(app: &mut App, ui: &mut egui::Ui) {
    let grid = Grid::latest(&app.runs);
    if grid.models.is_empty() {
        note(ui, "no runs yet: run a matrix file or `inference-bench matrix`", LEGEND);
        return;
    }
    let width = ((ui.available_width() - 190.0) / grid.backends.len() as f32).clamp(110.0, 220.0);
    let mut columns: Vec<(&str, f32)> = vec![("model", 190.0)];
    columns.extend(grid.backends.iter().map(|b| (b.as_str(), width)));
    section(ui, "decode speed", "latest run per model and backend", |ui| {
        Table::new(&columns, grid.models.len()).show(ui, |row, p, rect, at| {
            cell(p, rect, at(0), 190.0, &grid.models[row], VALUE);
            for b in 0..grid.backends.len() {
                let (text, tint) = match grid.cell(row, b) {
                    Some(r) => cell_text(r),
                    None => ("not run".to_string(), LEGEND),
                };
                cell(p, rect, at(b + 1), width, &text, tint);
            }
        });
    });
    let failures: Vec<&RunResult> = (0..grid.models.len())
        .flat_map(|m| (0..grid.backends.len()).map(move |b| (m, b)))
        .filter_map(|(m, b)| grid.cell(m, b))
        .filter(|r| r.error.is_some())
        .collect();
    if !failures.is_empty() {
        section(ui, "errors", "what each backend said", |ui| {
            for r in failures {
                let e = r.error.as_ref().unwrap();
                let tint = if e.kind == ErrorKind::UnsupportedArchitecture { WARN } else { FAULT };
                Line::new().legend(&r.backend).gap(8.0).value(&r.model).show(ui);
                note(ui, format!("{}: {}", e.kind.label(), e.message), tint);
            }
        });
    }
    section(ui, "latest details", "prefill, first token and memory", |ui| {
        let cols: [(&str, f32); 7] = [
            ("model", 190.0),
            ("backend", 120.0),
            ("prefill", 110.0),
            ("decode", 110.0),
            ("ttft", 90.0),
            ("memory", 100.0),
            ("tokens", 90.0),
        ];
        let cells: Vec<&RunResult> = (0..grid.models.len())
            .flat_map(|m| (0..grid.backends.len()).map(move |b| (m, b)))
            .filter_map(|(m, b)| grid.cell(m, b))
            .filter(|r| r.error.is_none())
            .collect();
        Table::new(&cols, cells.len()).show(ui, |row, p, rect, at| detail_row(cells[row], p, rect, at, &cols));
    });
}

fn detail_row(
    r: &RunResult,
    p: &egui::Painter,
    rect: egui::Rect,
    at: &dyn Fn(usize) -> f32,
    cols: &[(&str, f32)],
) {
    let rate = |v: Option<f64>| v.map(|v| format!("{v:.1} tok/s")).unwrap_or("-".into());
    let fields = [
        (r.model.clone(), VALUE),
        (r.backend.clone(), VALUE),
        (rate(r.prefill_tok_s), TRACE),
        (rate(r.decode_tok_s), TRACE),
        (r.ttft_secs.map(millis).unwrap_or("-".into()), TRACE),
        (r.peak_memory_bytes.map(megabytes).unwrap_or("-".into()), TRACE),
        (format!("{} + {}", r.prompt_tokens, r.completion_tokens), VALUE),
    ];
    for (i, (text, tint)) in fields.iter().enumerate() {
        cell(p, rect, at(i), cols[i].1, text, *tint);
    }
}

pub fn history(app: &mut App, ui: &mut egui::Ui) {
    if app.runs.is_empty() {
        note(ui, "no runs yet", LEGEND);
        return;
    }
    let cols: [(&str, f32); 8] = [
        ("when", 140.0),
        ("backend", 110.0),
        ("model", 170.0),
        ("prefill", 100.0),
        ("decode", 100.0),
        ("ttft", 80.0),
        ("memory", 90.0),
        ("result", 140.0),
    ];
    let runs: Vec<&RunResult> = app.runs.iter().rev().collect();
    section(ui, "runs", "newest first", |ui| {
        Table::new(&cols, runs.len()).show(ui, |row, p, rect, at| {
            let r = runs[row];
            let rate = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or("-".into());
            let (result, tint) = match &r.error {
                Some(e) => (e.kind.label().to_string(), FAULT),
                None => ("ok".to_string(), OK),
            };
            let fields = [
                (r.started.format("%Y-%m-%d %H:%M:%S").to_string(), LEGEND),
                (r.backend.clone(), VALUE),
                (r.model.clone(), VALUE),
                (rate(r.prefill_tok_s), TRACE),
                (rate(r.decode_tok_s), TRACE),
                (r.ttft_secs.map(millis).unwrap_or("-".into()), TRACE),
                (r.peak_memory_bytes.map(megabytes).unwrap_or("-".into()), TRACE),
                (result, tint),
            ];
            for (i, (text, col)) in fields.iter().enumerate() {
                cell(p, rect, at(i), cols[i].1, text, *col);
            }
        });
    });
    section(ui, "output", "the text a run produced", |ui| {
        let options: Vec<(usize, String)> = app
            .runs
            .iter()
            .enumerate()
            .rev()
            .map(|(i, r)| (i, format!("{} {} {}", r.started.format("%H:%M:%S"), r.backend, r.model)))
            .collect();
        ui.allocate_ui(egui::vec2(360.0, KEY_H), |ui| {
            choice(ui, "run", &mut app.shown_run, options);
        });
        match app.runs.get(app.shown_run) {
            Some(r) => match &r.error {
                Some(e) => note(ui, format!("{}: {}", e.kind.label(), e.message), FAULT),
                None => note(ui, r.text.clone(), VALUE),
            },
            None => {}
        }
    });
}

pub fn charts(app: &mut App, ui: &mut egui::Ui) {
    let options: Vec<(Metric, &str, Option<usize>)> =
        Metric::ALL.iter().map(|m| (*m, m.label(), None)).collect();
    segments(ui, &mut app.metric, &options);
    let grid = Grid::latest(&app.runs);
    let all = bars(&grid, app.metric);
    section(ui, app.metric.label(), "latest run per model and backend", |ui| {
        chart::bars(ui, &all, app.metric, grid.backends.len());
    });
}
