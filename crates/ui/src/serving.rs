use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::live::{HISTORY_SAMPLES, Point, Reading, RequestRecord};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Span {
    Five,
    Fifteen,
}

impl Span {
    pub fn samples(self) -> usize {
        match self {
            Span::Five => 300,
            Span::Fifteen => HISTORY_SAMPLES,
        }
    }
}

pub struct Snapshot {
    pub reading: Option<Reading>,
    pub last: Option<Point>,
    pub per_minute: f64,
    pub generation: Vec<f32>,
    pub prefill: Vec<f32>,
    pub ttft_p50: Vec<f32>,
    pub ttft_p95: Vec<f32>,
    pub itl_p50: Vec<f32>,
    pub itl_p95: Vec<f32>,
    pub running: Vec<f32>,
    pub kv: Vec<f32>,
    pub recent: Vec<RequestRecord>,
}

pub fn snapshot(app: &App, span: Span) -> Snapshot {
    let handle = app.engine.handle();
    let history = handle.history();
    let h = history.lock().unwrap();
    let take = span.samples();
    let ms = |f: fn(&Point) -> f64| move |p: &Point| f(p) * 1000.0;
    Snapshot {
        reading: h.latest().cloned(),
        last: h.last().cloned(),
        per_minute: h.requests_per_minute(),
        generation: h.series(take, |p| p.gen_tok_s),
        prefill: h.series(take, |p| p.prefill_tok_s),
        ttft_p50: h.series(take, ms(|p| p.ttft_p50)),
        ttft_p95: h.series(take, ms(|p| p.ttft_p95)),
        itl_p50: h.series(take, ms(|p| p.itl_p50)),
        itl_p95: h.series(take, ms(|p| p.itl_p95)),
        running: h.series(take, |p| p.running as f64),
        kv: h.series(take, |p| p.kv_used_bytes / (1u64 << 30) as f64),
        recent: {
            drop(h);
            handle.recent_requests()
        },
    }
}

fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
    }
}

fn percent(num: u64, den: u64) -> Option<String> {
    (den > 0).then(|| crate::fmt::percent(num as f64 / den as f64))
}

pub fn readings(r: &Reading, per_minute: f64) -> Vec<(&'static str, String, egui::Color32)> {
    let mut out = vec![
        ("requests per min", format!("{per_minute:.1}"), TRACE),
        ("running", format!("{} / {}", r.running, r.max_seqs), TRACE),
        ("waiting", r.waiting.to_string(), TRACE),
    ];
    if let Some(s) = &r.serving {
        let block = s.kv_block_bytes as u64;
        out.push((
            "kv cache",
            format!("{} / {}", size(s.kv_blocks_used as u64 * block), size(s.kv_blocks_total as u64 * block)),
            TRACE,
        ));
        out.push(("kv blocks", format!("{} / {}", s.kv_blocks_used, s.kv_blocks_total), TRACE));
        out.push(("reusable kv", size(s.kv_blocks_cached as u64 * block), TRACE));
        out.push(("prefix hits", percent(s.prefix_hits, s.prefix_queries).unwrap_or_else(|| "none yet".into()), TRACE));
        if s.spec_drafted > 0 {
            out.push(("draft accepted", percent(s.spec_accepted, s.spec_drafted).unwrap_or_default(), TRACE));
        }
        out.push(("preemptions", s.preemptions.to_string(), TRACE));
    }
    out
}

fn top(series: &[&[f32]], floor: f32) -> f32 {
    let peak = series.iter().flat_map(|s| s.iter().copied()).fold(0.0f32, f32::max);
    (peak * 1.15).max(floor)
}

fn chart(ui: &mut egui::Ui, label: &str, now: String, series: &[&[f32]], floor: f32) {
    Line::new().legend(label).gap(8.0).measured(now).size(11.0).show(ui);
    let size = egui::vec2(ui.available_width(), 56.0);
    let hi = top(series, floor);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    for (i, s) in series.iter().enumerate() {
        let tint = if i == 0 { TRACE } else { TRACE.gamma_multiply(0.5) };
        ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
            ui.add(Trace::new(s).size(size).tint(tint).range(0.0, hi).filled(i == 0));
        });
    }
    ui.add_space(6.0);
}

fn ms(v: f64) -> String {
    crate::fmt::duration(v)
}

fn charts(ui: &mut egui::Ui, s: &Snapshot, max_seqs: usize, with_kv: bool) {
    let p = s.last.clone().unwrap_or_default();
    let (kv_used, kv_total) = (p.kv_used_bytes, p.kv_total_bytes);
    ui.columns(2, |cols| {
        chart(&mut cols[0], "generation tok/s", format!("{:.1}", p.gen_tok_s), &[&s.generation], 10.0);
        chart(&mut cols[0], "prompt tok/s", format!("{:.0}", p.prefill_tok_s), &[&s.prefill], 100.0);
        chart(&mut cols[0], "ttft p50 and p95", format!("{} / {}", ms(p.ttft_p50), ms(p.ttft_p95)), &[&s.ttft_p50, &s.ttft_p95], 50.0);
        chart(&mut cols[1], "inter-token p50 and p95", format!("{} / {}", ms(p.itl_p50), ms(p.itl_p95)), &[&s.itl_p50, &s.itl_p95], 20.0);
        chart(&mut cols[1], "running sequences", format!("{} of {max_seqs}", p.running), &[&s.running], max_seqs.max(1) as f32);
        if with_kv {
            let used = format!("{} of {}", size(kv_used as u64), size(kv_total as u64));
            chart(&mut cols[1], "kv cache used", used, &[&s.kv], (kv_total / (1u64 << 30) as f64).max(0.1) as f32);
        }
    });
}

fn right(p: &egui::Painter, row: egui::Rect, x: f32, w: f32, text: &str, col: egui::Color32) {
    let r = egui::Rect::from_min_max(egui::pos2(x, row.top()), egui::pos2(x + w - 12.0, row.bottom()));
    p.with_clip_rect(r.intersect(p.clip_rect())).text(egui::pos2(r.right(), r.center().y), egui::Align2::RIGHT_CENTER, text, figure(11.0), col);
}

pub fn request_cells(r: &RequestRecord, with_cache: bool) -> Vec<String> {
    let mut out = vec![crate::fmt::duration(r.ttft_secs), crate::fmt::count(r.prompt_tokens)];
    if with_cache {
        let hit = if r.prompt_tokens > 0 { crate::fmt::percent(r.cached_tokens as f64 / r.prompt_tokens as f64) } else { "-".into() };
        out.push(hit);
        out.push(crate::fmt::count(r.cached_tokens));
        out.push(crate::fmt::count(r.prompt_tokens - r.cached_tokens));
    }
    out.push(crate::fmt::count(r.completion_tokens));
    out.push(match (r.prefill_tok_s > 0.0, r.prompt_tokens > 0 && r.cached_tokens >= r.prompt_tokens) {
        (true, _) => crate::fmt::rate(r.prefill_tok_s),
        (false, true) => "cached".into(),
        _ => "-".into(),
    });
    out.push(if r.decode_tok_s > 0.0 { crate::fmt::rate(r.decode_tok_s) } else { "-".into() });
    out
}

pub fn request_columns(with_cache: bool) -> Vec<(&'static str, f32)> {
    let mut cols = vec![("time", 80.0), ("ttft", 90.0), ("prompt", 80.0)];
    if with_cache {
        cols.extend([("cache hit", 90.0), ("cache read", 100.0), ("cache write", 110.0)]);
    }
    cols.extend([("completion", 100.0), ("prefill", 130.0), ("decode", 130.0)]);
    cols
}

pub fn per_request(ui: &mut egui::Ui, recent: &[RequestRecord]) {
    section(ui, "per request", "newest first", |ui| {
        if recent.is_empty() {
            note(ui, "no completed requests yet", LEGEND);
            return;
        }
        let with_cache = recent.iter().any(|r| r.cached_tokens > 0);
        let cols = request_columns(with_cache);
        Table::new(&cols, recent.len().min(12)).show(ui, |i, p, rect, at| {
            let r = &recent[recent.len() - 1 - i];
            cell(p, rect, at(0), cols[0].1, &r.at.with_timezone(&chrono::Local).format("%H:%M:%S").to_string(), LEGEND);
            for (k, text) in request_cells(r, with_cache).iter().enumerate() {
                let tint = if matches!(cols[k + 1].0, "ttft" | "prefill" | "decode") { TRACE } else { VALUE };
                right(p, rect, at(k + 1), cols[k + 1].1, text, tint);
            }
        });
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(loaded) = app.loaded() else { return };
    let snap = snapshot(app, app.server.span);
    let mut span = app.server.span;
    let note_text = match &loaded.parallel_note {
        Some(n) => n.clone(),
        None if loaded.max_num_seqs > 1 => format!("{} requests at once, chat and API clients share the engine", loaded.max_num_seqs),
        None => "one request at a time: raise parallel requests in the load dialog".to_string(),
    };
    section(ui, "serving", &note_text, |ui| match &snap.reading {
        None => note(ui, "waiting for the first sample", LEGEND),
        Some(r) => {
            readouts(ui, &readings(r, snap.per_minute));
            segments(ui, &mut span, &[(Span::Five, "last 5 minutes", None), (Span::Fifteen, "last 15 minutes", None)]);
            charts(ui, &snap, r.max_seqs, r.serving.is_some());
        }
    });
    app.server.span = span;
    per_request(ui, &snap.recent);
}
