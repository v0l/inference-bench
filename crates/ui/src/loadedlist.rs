use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::engine::Loaded;
use inference_bench_core::library::human_size;

pub fn row_text(l: &Loaded, stats: Option<&specrt::serve::Stats>) -> String {
    let vram = l.vram_bytes.map_or_else(|| "vram unknown".to_string(), |b| format!("{} vram", human_size(b)));
    let live = stats.map_or(String::new(), |s| format!("  {} running, {} waiting", s.running, s.waiting));
    format!("{vram}  {} ctx  {} parallel{live}", l.ctx, l.max_num_seqs)
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let all = app.loaded_all();
    if all.len() < 2 && app.max_loaded() < 2 {
        return;
    }
    let handle = app.engine.handle();
    let mut unload = None;
    let mut chosen = None;
    let shown = app.loaded().map(|l| l.display);
    let sub = format!("{} of {} allowed", all.len(), app.max_loaded());
    section(ui, "loaded models", &sub, |ui| {
        if all.is_empty() {
            note(ui, "nothing loaded: pick a model on the Models tab", LEGEND);
        }
        for l in &all {
            let stats = handle.serving_stats_of(&l.display);
            ui.horizontal(|ui| {
                let current = shown.as_deref() == Some(l.display.as_str());
                if ui.add(Key::new(l.display.clone()).outline(if current { READOUT } else { VALUE }).id_salt(("use", l.model_id.clone()))).clicked() {
                    chosen = Some(l.display.clone());
                }
                Line::new().value(row_text(l, stats.as_ref())).size(11.0).show(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(Key::new("unload").outline(FAULT).id_salt(("unload", l.model_id.clone()))).clicked() {
                        unload = Some(l.display.clone());
                    }
                });
            });
        }
    });
    if let Some(name) = chosen {
        app.chat_model = Some(name);
    }
    if let Some(name) = unload {
        app.unload_model(&name);
    }
}
