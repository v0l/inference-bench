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
    let model = app.loaded().map(|l| l.display);
    let history = handle.history_of(model.as_deref());
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
            handle.recent_requests_of(model.as_deref())
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

pub fn offload_readings(r: &Reading, stream_bytes_s: f64) -> Vec<(&'static str, String, egui::Color32)> {
    let Some(o) = r.serving.as_ref().and_then(|s| s.offload) else { return Vec::new() };
    vec![
        ("weight hits", crate::fmt::percent(o.hit_ratio()), TRACE),
        ("weight stream", format!("{:.2} GB/s", stream_bytes_s / 1e9), TRACE),
        ("nvme reads", o.nvme_reads.to_string(), TRACE),
        ("nvme read", format!("{:.2} GB", o.nvme_read_bytes as f64 / 1e9), TRACE),
        ("nvme wait", format!("{:.2} s", o.nvme_wait_nanos as f64 / 1e9), TRACE),
        ("nvme failures", o.nvme_read_failures.to_string(), TRACE),
    ]
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

pub fn request_cells(r: &RequestRecord, columns: &[(&'static str, f32)]) -> Vec<String> {
    columns[1..].iter().map(|(name, _)| request_cell(r, name)).collect()
}

fn request_cell(r: &RequestRecord, name: &str) -> String {
    let dash = |v: f64| if v > 0.0 { crate::fmt::rate(v) } else { "-".into() };
    match name {
        "ttft" => crate::fmt::duration(r.ttft_secs),
        "queue" => crate::fmt::millis(r.queue_secs),
        "admit" => crate::fmt::duration(r.admission_secs),
        "wait" => crate::fmt::duration(r.prefill_wait_secs),
        "held by" | "held" => held_label(r.held_by.as_deref()).into(),
        "prompt" => crate::fmt::count(r.prompt_tokens),
        "cache hit" if r.prompt_tokens > 0 => crate::fmt::percent(r.cached_tokens as f64 / r.prompt_tokens as f64),
        "cache read" => crate::fmt::count(r.cached_tokens),
        "new" => crate::fmt::count(r.prompt_tokens - r.cached_tokens),
        "completion" => crate::fmt::count(r.completion_tokens),
        "prefill" => dash(r.prefill_tok_s),
        "decode" => dash(r.decode_tok_s),
        _ => "-".into(),
    }
}

pub fn held_label(reason: Option<&str>) -> &str {
    match reason {
        None => "-",
        Some("long_cap") => "long cap",
        Some(r) => r,
    }
}

pub const WIDE_TABLE: f32 = 1000.0;
pub const MEDIUM_TABLE: f32 = 780.0;

pub fn request_columns(with_cache: bool, width: f32) -> Vec<(&'static str, f32)> {
    if width >= MEDIUM_TABLE {
        return fill(wide_columns(with_cache, width), &[("held by", 80.0), ("admit", 80.0), ("wait", 80.0)], width, "queue");
    }
    let base = vec![("time", 66.0), ("ttft", 56.0), ("new", 52.0), ("prefill", 92.0)];
    let mut cols = fill(base, &[("held", 66.0), ("decode", 92.0), ("admit", 60.0), ("wait", 60.0)], width, "ttft");
    if let Some(i) = cols.iter().position(|c| c.0 == "decode") {
        let d = cols.remove(i);
        cols.push(d);
    }
    cols
}

fn fill(mut cols: Vec<(&'static str, f32)>, extras: &[(&'static str, f32)], width: f32, after: &str) -> Vec<(&'static str, f32)> {
    let at = cols.iter().position(|c| c.0 == after).map_or(1, |i| i + 1);
    let mut used: f32 = cols.iter().map(|c| c.1).sum();
    let mut added = 0;
    for &extra in extras {
        if used + extra.1 + 6.0 > width {
            break;
        }
        used += extra.1;
        cols.insert(at + added, extra);
        added += 1;
    }
    cols
}

fn wide_columns(with_cache: bool, width: f32) -> Vec<(&'static str, f32)> {
    let mut cols = vec![("time", 80.0), ("ttft", 90.0), ("queue", 90.0), ("prompt", 80.0)];
    if with_cache {
        cols.push(("cache hit", 90.0));
        if width >= WIDE_TABLE {
            cols.push(("cache read", 100.0));
        }
    }
    cols.push(("new", 80.0));
    if width >= WIDE_TABLE {
        cols.push(("completion", 100.0));
    }
    cols.extend([("prefill", 130.0), ("decode", 130.0)]);
    cols
}

pub fn per_request(ui: &mut egui::Ui, recent: &[RequestRecord]) {
    section(ui, "per request", "newest first", |ui| {
        if recent.is_empty() {
            note(ui, "no completed requests yet", LEGEND);
            return;
        }
        let with_cache = recent.iter().any(|r| r.cached_tokens > 0);
        let cols = request_columns(with_cache, ui.available_width());
        Table::new(&cols, recent.len().min(12)).show(ui, |i, p, rect, at| {
            let r = &recent[recent.len() - 1 - i];
            cell(p, rect, at(0), cols[0].1, &r.at.with_timezone(&chrono::Local).format("%H:%M:%S").to_string(), LEGEND);
            for (k, text) in request_cells(r, &cols).iter().enumerate() {
                let tint = if matches!(cols[k + 1].0, "ttft" | "queue" | "admit" | "wait" | "prefill" | "decode") { TRACE } else { VALUE };
                right(p, rect, at(k + 1), cols[k + 1].1, text, tint);
            }
        });
    });
}

pub fn host_reading(load: f64, cores: usize) -> Option<(&'static str, String, egui::Color32)> {
    inference_bench_core::live::host_busy(load, cores).then(|| ("host busy", format!("load {load:.1} on {cores} cores"), WARN))
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
    let several = app.loaded_all().len() > 1;
    section(ui, "serving", &note_text, |ui| match &snap.reading {
        None => note(ui, "waiting for the first sample", LEGEND),
        Some(r) => {
            if several {
                let label = format!("stats for {}", loaded.display);
                let line = Line::new().legend("showing").column(ui, 120.0).value(loaded.display.clone()).show(ui);
                line.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, label.clone()));
            }
            let mut shown = readings(r, snap.per_minute);
            shown.extend(offload_readings(r, snap.last.as_ref().map_or(0.0, |p| p.stream_bytes_s)));
            if let Some((load, cores)) = inference_bench_core::live::host_load() {
                shown.extend(host_reading(load, cores));
            }
            readouts(ui, &shown);
            segments(ui, &mut span, &[(Span::Five, "last 5 minutes", None), (Span::Fifteen, "last 15 minutes", None)]);
            charts(ui, &snap, r.max_seqs, r.serving.is_some());
        }
    });
    app.server.span = span;
    per_request(ui, &snap.recent);
}
