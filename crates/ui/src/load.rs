use crate::detail::View;
use crate::{App, Tab};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{LocalModel, human_size};
use specrt::anatomy::ModelAnatomy;

pub const MIN_CTX: usize = 512;
pub const STEP: usize = 512;
pub const MAX_SEQS: usize = inference_bench_core::engine::MAX_PARALLEL;

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

pub fn estimate(a: &ModelAnatomy, ctx: usize, seqs: usize, free: Option<u64>) -> Estimate {
    let seqs = seqs.max(1) as u64;
    Estimate { weights: a.total_bytes, kv: a.kv_bytes_per_token * ctx as u64 * seqs, state: a.state_bytes_per_seq * seqs, free }
}

pub fn max_context(a: &ModelAnatomy) -> usize {
    a.max_context.max(1)
}

pub fn largest_fit(a: &ModelAnatomy, free: Option<u64>, seqs: usize, max: usize) -> usize {
    let floor = MIN_CTX.min(max);
    let Some(free) = free else { return max };
    if a.kv_bytes_per_token == 0 {
        return max;
    }
    let seqs = seqs.max(1) as u64;
    let room = free.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs);
    let tokens = (room / (a.kv_bytes_per_token * seqs)) as usize;
    (tokens / STEP * STEP).clamp(floor, max)
}

pub fn initial_ctx(remembered: Option<usize>, default: usize, a: &ModelAnatomy, free: Option<u64>, seqs: usize, max: usize) -> usize {
    let floor = MIN_CTX.min(max);
    match remembered {
        Some(r) => r.clamp(floor, max),
        None => default.min(largest_fit(a, free, seqs, max)).clamp(floor, max),
    }
}

pub fn largest_seqs(a: &ModelAnatomy, ctx: usize, free: Option<u64>) -> usize {
    let Some(free) = free else { return MAX_SEQS };
    let per = a.kv_bytes_per_token * ctx as u64 + a.state_bytes_per_seq;
    if per == 0 {
        return MAX_SEQS;
    }
    let room = free.saturating_sub(a.total_bytes);
    ((room / per) as usize).clamp(1, MAX_SEQS)
}

pub fn initial_seqs(remembered: Option<usize>, default: usize, a: &ModelAnatomy, ctx: usize, free: Option<u64>) -> usize {
    match remembered {
        Some(r) => r.clamp(1, MAX_SEQS),
        None => default.min(largest_seqs(a, ctx, free)).clamp(1, MAX_SEQS),
    }
}

pub struct LoadDialog {
    pub model: LocalModel,
    pub view: Result<View, String>,
    pub ctx: usize,
    pub max: usize,
    pub seqs: usize,
    pub remember: bool,
}

impl LoadDialog {
    pub fn open(app: &App, model: &LocalModel) -> Self {
        let view = View::load(model);
        let free = app.free_gpu_bytes();
        let remembered_seqs = app.settings.model_seqs.get(&model.name).copied();
        let (ctx, max, seqs) = match &view {
            Ok(v) => {
                let max = max_context(&v.anatomy);
                let remembered = app.settings.model_ctx.get(&model.name).copied();
                let seqs = remembered_seqs.unwrap_or(app.settings.default_seqs).clamp(1, MAX_SEQS);
                let ctx = initial_ctx(remembered, app.settings.default_ctx, &v.anatomy, free, seqs, max);
                let seqs = initial_seqs(remembered_seqs, app.settings.default_seqs, &v.anatomy, ctx, free);
                (ctx, max, seqs)
            }
            Err(_) => (app.settings.ctx_for(&model.name), app.settings.ctx_for(&model.name), app.settings.seqs_for(&model.name)),
        };
        Self { model: model.clone(), view, ctx, max, seqs, remember: true }
    }

    pub fn estimate(&self, free: Option<u64>) -> Option<Estimate> {
        self.view.as_ref().ok().map(|v| estimate(&v.anatomy, self.ctx, self.seqs, free))
    }
}

pub fn modal(app: &mut App, ctx: &egui::Context) {
    let Some(mut dialog) = app.load.take() else { return };
    let free = app.free_gpu_bytes();
    let est = dialog.estimate(free);
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
                row_help(ui, "parallel requests", "How many requests the engine runs at once, chat and API clients together. Each one gets its own cache at the full context length, so more of them cost more memory. 1 keeps the plain single-request engine.", |ui| {
                    let suffix = if dialog.seqs == 1 { " request" } else { " requests" };
                    ui.add(egui::Slider::new(&mut dialog.seqs, 1..=MAX_SEQS).suffix(suffix));
                });
            });
            ui.add_space(8.0);
            section(ui, "memory", "weights and cache, against what the card has free", |ui| match &est {
                Some(e) => {
                    Line::new().legend("weights").column(ui, 120.0).value(human_size(e.weights)).show(ui);
                    let held = if dialog.seqs > 1 { format!("{} tokens x {}", dialog.ctx, dialog.seqs) } else { format!("{} tokens", dialog.ctx) };
                    Line::new().legend("kv cache").column(ui, 120.0).value(human_size(e.kv)).gap(8.0).note(held).show(ui);
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
                if ui.add(Key::new("load").solid(READOUT).min_width(100.0)).clicked() {
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
        if dialog.remember {
            app.settings.model_ctx.insert(model.name.clone(), dialog.ctx);
            app.settings.model_seqs.insert(model.name.clone(), dialog.seqs);
            let _ = app.settings.save(&app.dir);
        }
        app.chat.load_error = None;
        app.chat.pick(&model.name);
        app.load_model_with(&model, dialog.ctx, dialog.seqs);
        if app.tab == Tab::Models {
            app.tab = Tab::Chat;
        }
    } else if !close {
        app.load = Some(dialog);
    }
}
