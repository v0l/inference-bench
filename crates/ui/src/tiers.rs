use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;
use specrt::offload::{OffloadPlan, Tier};

pub fn tier_cells(plan: &OffloadPlan) -> Vec<(&'static str, String)> {
    vec![
        ("vram", human_size(plan.vram_bytes)),
        ("host ram", human_size(plan.host_bytes)),
        ("nvme", human_size(plan.nvme_bytes)),
    ]
}

pub fn layer_text(plan: &OffloadPlan) -> String {
    let streamed = plan.layers.iter().filter(|l| l.tier != Tier::Vram).count();
    format!(
        "{streamed} of {} layers stream in, {} staging, {} per token",
        plan.layers.len(),
        human_size(plan.staging_bytes),
        human_size(plan.streamed_bytes_per_token())
    )
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let plans: Vec<_> = app
        .loaded_all()
        .into_iter()
        .filter_map(|l| handle.serving_stats_of(&l.display).and_then(|s| s.offload_plan).map(|p| (l.display, p)))
        .collect();
    if plans.is_empty() {
        return;
    }
    let visible = ui.ctx().content_rect().right() - ui.max_rect().left() - 12.0;
    ui.scope(|ui| {
        ui.set_max_width(ui.available_width().min(visible).max(120.0));
        section(ui, "where the weights live", "per loaded model, from the offload plan", |ui| {
            for (name, plan) in &plans {
                Line::new().legend("model").column(ui, 120.0).value(name.clone()).wrapped(ui);
                for (label, value) in tier_cells(plan) {
                    Line::new().legend(label).column(ui, 120.0).value(value).wrapped(ui);
                }
                Line::new().legend("layers").column(ui, 120.0).value(layer_text(plan)).wrapped(ui);
            }
        });
    });
}
