use crate::detail::View;
use crate::{App, Tab};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{LocalModel, human_size};
use specrt::anatomy::ModelAnatomy;

pub const MIN_CTX: usize = 512;
pub const STEP: usize = 512;
pub const MAX_SEQS: usize = inference_bench_core::engine::MAX_PARALLEL;
pub const GPU_UTILIZATION: f64 = 0.9;
pub const TYPICAL_REQUEST: usize = 4096;
const GB: f64 = (1u64 << 30) as f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    pub weights: u64,
    pub kv: u64,
    pub state: u64,
    pub free: Option<u64>,
}

impl Estimate {
    pub fn total(&self) -> u64 {
        self.weights + self.kv + self.state
    }

    pub fn fits(&self) -> Option<bool> {
        self.free.map(|f| self.total() <= f)
    }
}

pub fn estimate(a: &ModelAnatomy, kv_bytes: u64, seqs: usize, free: Option<u64>) -> Estimate {
    let seqs = seqs.max(1) as u64;
    Estimate { weights: a.total_bytes, kv: kv_bytes, state: a.state_bytes_per_seq * seqs, free }
}

pub fn pool_bytes(a: &ModelAnatomy, ctx: usize, seqs: usize, kv_gb: f64) -> u64 {
    if seqs <= 1 {
        a.kv_bytes_per_token * ctx as u64
    } else {
        (kv_gb * GB) as u64
    }
}

pub fn pool_tokens(a: &ModelAnatomy, kv_bytes: u64) -> usize {
    if a.kv_bytes_per_token == 0 { usize::MAX } else { (kv_bytes / a.kv_bytes_per_token) as usize }
}

pub fn pool_holds_one(a: &ModelAnatomy, kv_bytes: u64, ctx: usize) -> bool {
    pool_tokens(a, kv_bytes) >= ctx
}

pub fn concurrency(a: &ModelAnatomy, kv_bytes: u64, per_request: usize) -> f64 {
    if per_request == 0 { 0.0 } else { pool_tokens(a, kv_bytes).min(usize::MAX / 2) as f64 / per_request as f64 }
}

pub fn room_bytes(a: &ModelAnatomy, free: Option<u64>, seqs: usize) -> Option<u64> {
    free.map(|f| f.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64))
}

pub fn default_kv_gb(a: &ModelAnatomy, free: Option<u64>, seqs: usize, ctx: usize) -> f64 {
    let needed = a.kv_bytes_per_token * ctx as u64;
    let bytes = match free {
        Some(f) => {
            let usable = (f as f64 * GPU_UTILIZATION) as u64;
            usable.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64)
        }
        None => needed,
    };
    round_gb(bytes.max(needed))
}

pub fn max_kv_gb(a: &ModelAnatomy, free: Option<u64>, seqs: usize, ctx: usize) -> f64 {
    let needed = a.kv_bytes_per_token * ctx as u64;
    let room = room_bytes(a, free, seqs).unwrap_or(needed * 4);
    round_gb(room.max(needed)).max(default_kv_gb(a, free, seqs, ctx)).max(0.25)
}

fn round_gb(bytes: u64) -> f64 {
    ((bytes as f64 / GB) * 4.0).ceil() / 4.0
}

pub fn max_context(a: &ModelAnatomy) -> usize {
    a.max_context.max(1)
}

pub fn initial_ctx(remembered: Option<usize>, default: usize, max: usize) -> usize {
    let floor = MIN_CTX.min(max);
    remembered.unwrap_or(default).clamp(floor, max)
}

pub fn largest_seqs(a: &ModelAnatomy, free: Option<u64>) -> usize {
    let Some(free) = free else { return MAX_SEQS };
    if a.state_bytes_per_seq == 0 {
        return MAX_SEQS;
    }
    let room = free.saturating_sub(a.total_bytes);
    ((room / a.state_bytes_per_seq) as usize).clamp(1, MAX_SEQS)
}

pub fn initial_seqs(remembered: Option<usize>, default: usize, a: &ModelAnatomy, free: Option<u64>) -> usize {
    match remembered {
        Some(r) => r.clamp(1, MAX_SEQS),
        None => default.min(largest_seqs(a, free)).clamp(1, MAX_SEQS),
    }
}

pub fn parse_names(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for n in text.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        if !out.iter().any(|o| o == n) {
            out.push(n.to_string());
        }
    }
    out
}

pub struct LoadDialog {
    pub model: LocalModel,
    pub view: Result<View, String>,
    pub ctx: usize,
    pub max: usize,
    pub seqs: usize,
    pub kv_gb: f64,
    pub served_names: String,
    pub remember: bool,
}

impl LoadDialog {
    pub fn open(app: &App, model: &LocalModel) -> Self {
        let view = View::load(model);
        let free = app.free_gpu_bytes();
        let remembered_seqs = app.settings.model_seqs.get(&model.name).copied();
        let (ctx, max, seqs, kv_gb) = match &view {
            Ok(v) => {
                let max = max_context(&v.anatomy);
                let remembered = app.settings.model_ctx.get(&model.name).copied();
                let seqs = initial_seqs(remembered_seqs, app.settings.default_seqs, &v.anatomy, free);
                let ctx = initial_ctx(remembered, app.settings.default_ctx, max);
                let kv_gb = app.settings.kv_gb_for(&model.name).unwrap_or_else(|| default_kv_gb(&v.anatomy, free, seqs, ctx));
                (ctx, max, seqs, kv_gb)
            }
            Err(_) => (app.settings.ctx_for(&model.name), app.settings.ctx_for(&model.name), app.settings.seqs_for(&model.name), app.settings.kv_gb_for(&model.name).unwrap_or(0.0)),
        };
        let served_names = app.settings.served_names_for(&model.name).join(", ");
        Self { model: model.clone(), view, ctx, max, seqs, kv_gb, served_names, remember: true }
    }

    pub fn kv_bytes(&self) -> Option<u64> {
        self.view.as_ref().ok().map(|v| pool_bytes(&v.anatomy, self.ctx, self.seqs, self.kv_gb))
    }

    pub fn holds_one_request(&self) -> bool {
        match (&self.view, self.kv_bytes()) {
            (Ok(v), Some(kv)) => pool_holds_one(&v.anatomy, kv, self.ctx),
            _ => true,
        }
    }

    pub fn estimate(&self, free: Option<u64>) -> Option<Estimate> {
        self.view.as_ref().ok().map(|v| estimate(&v.anatomy, pool_bytes(&v.anatomy, self.ctx, self.seqs, self.kv_gb), self.seqs, free))
    }
}

pub fn modal(app: &mut App, ctx: &egui::Context) {
    let Some(mut dialog) = app.load.take() else { return };
    let free = app.free_gpu_bytes();
    let est = dialog.estimate(free);
    let kv_range = match &dialog.view {
        Ok(v) => (0.25, max_kv_gb(&v.anatomy, free, dialog.seqs, dialog.ctx).max(dialog.kv_gb)),
        Err(_) => (0.25, 0.25),
    };
    let (mut go, mut close) = (false, false);
    let r = egui::Modal::new(egui::Id::new("load-model"))
        .backdrop_color(egui::Color32::from_black_alpha(150))
        .show(ctx, |ui| {
            ui.set_width(520.0);
            modal_title(ui, &format!("Load {}", dialog.model.name));
            if let Err(e) = &dialog.view {
                status(ui, false, e);
                note(ui, "The engine will read the model itself and report what is wrong.", LEGEND);
            }
            section(ui, "context", "tokens one conversation can hold, and how many run at once", |ui| {
                row_help(ui, "context length", "Bigger contexts keep more of the conversation and cost more memory. The slider stops at what the model supports.", |ui| {
                    if dialog.view.is_ok() {
                        let lo = MIN_CTX.min(dialog.max);
                        ui.add(egui::Slider::new(&mut dialog.ctx, lo..=dialog.max).step_by(STEP as f64).suffix(" tokens"));
                    } else {
                        Line::new().set(format!("{} tokens", dialog.ctx)).show(ui);
                    }
                });
                if dialog.view.is_ok() {
                    Line::new().legend("model maximum").value(format!("{} tokens", dialog.max)).size(11.0).show(ui);
                }
                row_help(ui, "parallel requests", "How many requests the engine runs at once, chat and API clients together. They share one KV cache, and the engine queues or pauses requests when it is full. 1 keeps the plain single-request engine.", |ui| {
                    let suffix = if dialog.seqs == 1 { " request" } else { " requests" };
                    ui.add(egui::Slider::new(&mut dialog.seqs, 1..=MAX_SEQS).suffix(suffix));
                });
                if dialog.view.is_ok() && dialog.seqs > 1 {
                    let (lo, hi) = kv_range;
                    row_help(ui, "kv cache", "One cache shared by every request, like vLLM. It must hold at least one request of the full context length.", |ui| {
                        ui.add(egui::Slider::new(&mut dialog.kv_gb, lo..=hi.max(lo)).step_by(0.25).suffix(" GB"));
                    });
                }
            });
            ui.add_space(8.0);
            section(ui, "api", "extra names API clients can use for this model", |ui| {
                row_help(ui, "served names", "Comma separated aliases, like vLLM's served model name. /v1/models lists the model's own id, requests may use any of these.", |ui| {
                    ui.add(egui::TextEdit::singleline(&mut dialog.served_names).hint_text("qwen3.8-27b").desired_width(260.0));
                });
            });
            ui.add_space(8.0);
            section(ui, "memory", "weights and cache, against what the card has free", |ui| match &est {
                Some(e) => {
                    Line::new().legend("weights").column(ui, 120.0).value(human_size(e.weights)).show(ui);
                    let held = if dialog.seqs > 1 { format!("{} tokens shared by {} requests", pool_tokens(&dialog.view.as_ref().unwrap().anatomy, e.kv), dialog.seqs) } else { format!("{} tokens", dialog.ctx) };
                    let kv_line = Line::new().legend("kv cache").column(ui, 120.0);
                    let kv_line = if dialog.holds_one_request() { kv_line.value(human_size(e.kv)) } else { kv_line.value(human_size(e.kv)).tint(FAULT) };
                    kv_line.gap(8.0).note(held).show(ui);
                    if let Some(v) = dialog.view.as_ref().ok().filter(|_| dialog.seqs > 1) {
                        let full = concurrency(&v.anatomy, e.kv, dialog.ctx);
                        let typical = concurrency(&v.anatomy, e.kv, TYPICAL_REQUEST.min(dialog.ctx));
                        Line::new().legend("concurrency").column(ui, 120.0).value(format!("{full:.2}x at {} tokens, {typical:.1}x at {} tokens", dialog.ctx, TYPICAL_REQUEST.min(dialog.ctx))).size(11.0).show(ui);
                    }
                    if !dialog.holds_one_request() {
                        status(ui, false, &format!("the KV cache cannot hold one request of {} tokens: raise it or lower the context length", dialog.ctx));
                    }
                    if e.state > 0 {
                        let each = if dialog.seqs > 1 { format!("{} each", human_size(e.state / dialog.seqs as u64)) } else { String::new() };
                        Line::new().legend("recurrent state").column(ui, 120.0).value(human_size(e.state)).gap(8.0).note(each).show(ui);
                    }
                    let total = Line::new().legend("total").column(ui, 120.0);
                    let total = if e.fits() == Some(false) { total.value(human_size(e.total())).tint(FAULT) } else { total.value(human_size(e.total())) };
                    total.show(ui);
                    match e.free {
                        Some(f) => {
                            Line::new().legend("free on the gpu").column(ui, 120.0).measured(human_size(f)).show(ui);
                            match e.fits() {
                                Some(false) => status(ui, false, &format!("does not fit in free memory, {} over", human_size(e.total() - f))),
                                _ => status(ui, true, "fits in free memory"),
                            }
                        }
                        None => note(ui, "No GPU reading yet, so the estimate cannot be checked against free memory.", LEGEND),
                    }
                }
                None => note(ui, "No estimate: the model's layout could not be read.", LEGEND),
            });
            ui.add_space(8.0);
            switch(ui, "remember for this model", &mut dialog.remember, "use this context and parallel requests next time", "Saved with the app settings, per model.");
            footer(ui, |ui| {
                if ui.add_enabled(dialog.holds_one_request(), Key::new("load").solid(READOUT).min_width(100.0)).clicked() {
                    go = true;
                }
                if ui.add(Key::new("cancel").outline(VALUE).min_width(100.0)).clicked() {
                    close = true;
                }
            });
        });
    if r.should_close() {
        close = true;
    }
    if go {
        let model = dialog.model.clone();
        let names = parse_names(&dialog.served_names);
        if dialog.remember {
            app.settings.model_ctx.insert(model.name.clone(), dialog.ctx);
            app.settings.model_seqs.insert(model.name.clone(), dialog.seqs);
            if dialog.seqs > 1 && dialog.view.is_ok() {
                app.settings.model_kv_gb.insert(model.name.clone(), dialog.kv_gb);
            }
            app.settings.model_served_names.insert(model.name.clone(), names.clone());
            let _ = app.settings.save(&app.dir);
        }
        app.chat.load_error = None;
        let kv = dialog.view.is_ok().then(|| dialog.kv_bytes()).flatten().filter(|_| dialog.seqs > 1);
        app.load_model_named(&model, dialog.ctx, dialog.seqs, kv, names);
        if app.tab == Tab::Models {
            app.tab = Tab::Chat;
        }
    } else if !close {
        app.load = Some(dialog);
    }
}
