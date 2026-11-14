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

pub fn draw(app: &mut App, ui: &mut egui::Ui, busy: bool, right: impl FnOnce(&mut egui::Ui)) {
    let loaded = app.loaded();
    let loading = app.loading.clone();
    let failed = app.chat.load_error.is_some().then(|| app.chat.conv().model.clone()).filter(|m| !m.is_empty());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        match (&loading, &loaded, &failed) {
            (Some(name), _, _) => {
                lamp(ui, "loading", true, false);
                chip(ui, name, READOUT_DIM, false);
            }
            (None, Some(l), _) => {
                lamp(ui, if busy { "generating" } else { "ready" }, true, false);
                chip(ui, &l.display, READOUT, true);
                chip(ui, &format!("{} ctx", l.ctx), READOUT, false);
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
            app.unload();
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            right(ui);
            if let Some(g) = app.gpu.first() {
                Line::new().legend("vram").measured(format!("{} / {}", human_size(g.used_bytes), human_size(g.total_bytes))).size(11.0).show(ui);
            }
        });
    });
}
