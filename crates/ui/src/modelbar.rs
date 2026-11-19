use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;

const PICKER_W: f32 = 360.0;

pub fn usable(app: &App, query: &str) -> Vec<usize> {
    (0..app.library.len())
        .filter(|&i| crate::models::runs_here(&app.library[i]) && crate::models::matches(&app.library[i], query))
        .collect()
}

fn picker(app: &mut App, ui: &mut egui::Ui) {
    let open = ui.add(Key::new("select a model").outline(VALUE).min_width(150.0).id_salt("model-bar-pick"));
    let mut chosen = None;
    egui::Popup::menu(&open)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .width(PICKER_W)
        .show(|ui| {
            ui.set_width(PICKER_W);
            field(ui, &mut app.picker_query, "search your models");
            ui.add_space(4.0);
            let found = usable(app, &app.picker_query);
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for &i in &found {
                    let m = &app.library[i];
                    let label = format!("{}  {}  {}", m.name, m.quant, human_size(m.size_bytes));
                    if ui.add(Key::new(label).outline(VALUE).min_width(ui.available_width()).id_salt(("pick", i))).clicked() {
                        chosen = Some(i);
                    }
                }
                if found.is_empty() {
                    note(ui, "no model on disk matches", LEGEND);
                }
            });
        });
    if let Some(i) = chosen {
        egui::Popup::close_all(ui.ctx());
        app.picker_query.clear();
        let m = app.library[i].clone();
        app.open_load_dialog(&m);
    }
}

pub fn progress_text(p: &crate::LoadProgress) -> String {
    if p.total == 0 {
        return p.stage.clone();
    }
    let secs = p.weights_started.map_or(0.0, |t| t.elapsed().as_secs_f64());
    if p.stage == "reading weights" {
        let rate = if secs > 0.2 && p.done > 0 { format!("  {:.1} GB/s", p.done as f64 / secs / 1e9) } else { String::new() };
        format!("{}  {:.1} / {:.1} GB{rate}", p.stage, p.done as f64 / 1e9, p.total as f64 / 1e9)
    } else if p.stage == "downloading the CUDA compiler" {
        format!("{}  {:.0} / {:.0} MB", p.stage, p.done as f64 / 1e6, p.total as f64 / 1e6)
    } else {
        p.stage.clone()
    }
}

pub fn progress_fraction(p: &crate::LoadProgress) -> f32 {
    if p.total == 0 { 0.0 } else { (p.done as f64 / p.total as f64).clamp(0.0, 1.0) as f32 }
}

pub fn load_bar(ui: &mut egui::Ui, p: &crate::LoadProgress, width: f32) {
    let text = progress_text(p);
    ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
        ui.set_width(width);
        ui.spacing_mut().item_spacing.y = 1.0;
        let total = (p.total > 0).then_some(1.0);
        progress(ui, "", progress_fraction(p), total, "");
        Line::new().value(text).size(10.5).elided(ui);
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui, busy: bool, right: impl FnOnce(&mut egui::Ui)) {
    let loaded = app.loaded();
    let loading = app.loading.clone();
    let failed = app.chat.load_error.is_some().then(|| app.chat.attempted.clone()).filter(|m| !m.is_empty());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        match (&loading, &loaded, &failed) {
            (Some(name), _, _) => {
                lamp(ui, "loading", true, false);
                chip(ui, name, READOUT_DIM, false);
                if let Some(p) = &app.load_progress {
                    load_bar(ui, p, 320.0);
                }
            }
            (None, Some(l), _) => {
                lamp(ui, if busy { "generating" } else { "ready" }, true, false);
                chip(ui, &l.display, READOUT, true);
                chip(ui, &format!("{} ctx", l.ctx), READOUT, false);
                for other in app.loaded_all().into_iter().filter(|o| o.display != l.display) {
                    let key = ui.add(Key::new(other.display.clone()).outline(VALUE).id_salt(("switch", other.model_id.clone())));
                    let key = key.on_hover_text(format!("chat with {} and show its stats", other.display));
                    if key.clicked() {
                        app.chat_model = Some(other.display);
                    }
                }
            }
            (None, None, Some(name)) => {
                lamp(ui, "failed", false, true);
                chip(ui, name, FAULT, false);
            }
            _ => {
                lamp(ui, "no model", false, false);
            }
        }
        ui.add_space(8.0);
        picker(app, ui);
        if loading.is_some() {
            ui.add(Key::new("loading").outline(READOUT).enabled(false).min_width(100.0));
        } else if loaded.is_some() && ui.add(Key::new("unload").outline(FAULT).min_width(100.0)).clicked() {
            match app.loaded_all().len() {
                0 | 1 => app.unload(),
                _ => app.unload_model(&loaded.as_ref().map(|l| l.display.clone()).unwrap_or_default()),
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            right(ui);
            if let Some((used, total)) = inference_bench_core::gpu::used_total(&app.gpu) {
                let legend = if app.gpu.len() > 1 { format!("vram x{}", app.gpu.len()) } else { "vram".into() };
                Line::new().legend(&legend).measured(format!("{} / {}", human_size(used), human_size(total))).size(11.0).show(ui);
            }
        });
    });
}
