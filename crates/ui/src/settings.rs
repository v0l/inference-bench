use crate::App;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use std::path::PathBuf;

#[derive(Default)]
pub struct SettingsState {
    pub new_folder: String,
    pub new_spec_folder: String,
}

fn spec_updates(app: &mut App, ui: &mut egui::Ui) {
    section(ui, "spec updates", "new model specs from GitHub, without reinstalling", |ui| {
        let mut on = app.settings.specs_auto_update;
        if switch(ui, "automatic", &mut on, "check at start and on a timer", "one HTTPS request to GitHub per check, no identifiers") {
            app.settings.specs_auto_update = on;
        }
        row(ui, "repository", |ui| {
            field(ui, &mut app.settings.specs_repo, "v0l/inference-bench");
        });
        row(ui, "branch", |ui| {
            field(ui, &mut app.settings.specs_branch, "master");
        });
        row(ui, "pin to tag", |ui| {
            field(ui, &mut app.settings.specs_tag, "empty follows the branch, e.g. v0.2.0");
        });
        row(ui, "every (hours)", |ui| {
            ui.add(egui::DragValue::new(&mut app.settings.specs_update_hours).range(0..=720));
        });
        let state = app.specs.state.clone();
        reading(ui, "source", app.settings.specs().source());
        reading(ui, "last check", crate::specs::when(state.checked_at));
        reading(ui, "last success", crate::specs::when(state.last_success));
        if !state.last_result.is_empty() {
            status(ui, state.last_ok, &state.last_result);
        }
        ui.horizontal(|ui| {
            let busy = app.specs.busy();
            if ui.add(Key::new(if busy { "checking" } else { "check now" }).outline(READOUT).enabled(!busy)).clicked() {
                let (wake, dir) = (app.wake.clone(), app.dir.clone());
                app.specs.check_now(&wake, &dir, &app.settings);
            }
            if ui.add(Key::new("roll back").outline(VALUE)).clicked() {
                let r = crate::specs::rollback(&app.dir);
                app.specs.load_state(&app.dir);
                app.notice = Some(match r {
                    Ok(t) => (true, t),
                    Err(e) => (false, e),
                });
                app.rescan();
            }
        });
        hint(ui, "0 hours checks only at start. With automatic off, the app uses the specs built into this version and your own folders.");
        let mut remove = None;
        for (i, f) in app.settings.spec_folders.iter().enumerate() {
            ui.push_id(("specdir", i), |ui| {
                ui.horizontal(|ui| {
                    Line::new().legend("your specs").value(f.display().to_string()).elided(ui);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(Key::new("remove").outline(FAULT)).clicked() {
                            remove = Some(i);
                        }
                    });
                });
            });
        }
        if let Some(i) = remove {
            app.settings.spec_folders.remove(i);
        }
        let st = &mut app.settings_ui;
        let mut add = false;
        let ready = !st.new_spec_folder.trim().is_empty();
        field_then(ui, &mut st.new_spec_folder, "/path/to/your/specs, wins over downloaded ones", 96.0, |ui| {
            add = ui.add(Key::new("add").outline(VALUE).enabled(ready).id_salt("add-spec-dir")).clicked();
        });
        if add {
            app.settings.spec_folders.push(PathBuf::from(st.new_spec_folder.trim()));
            st.new_spec_folder.clear();
        }
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "settings", |ui| {
        let mut save = false;
        section(ui, "model folders", "searched for GGUF and safetensors, on top of the Hugging Face cache", |ui| {
            let mut remove = None;
            for (i, f) in app.settings.model_folders.iter().enumerate() {
                ui.horizontal(|ui| {
                    Line::new().value(f.display().to_string()).elided(ui);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(Key::new("remove").outline(FAULT)).clicked() {
                            remove = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = remove {
                app.settings.model_folders.remove(i);
            }
            let st = &mut app.settings_ui;
            let mut add = false;
            let ready = !st.new_folder.trim().is_empty();
            field_then(ui, &mut st.new_folder, "/path/to/models", 96.0, |ui| {
                add = ui.add(Key::new("add").outline(VALUE).enabled(ready)).clicked();
            });
            if add {
                app.settings.model_folders.push(PathBuf::from(st.new_folder.trim()));
                st.new_folder.clear();
            }
            Line::new().legend("hub cache").value(inference_bench_core::library::hub_dir().display().to_string()).elided(ui);
        });
        section(ui, "hugging face", "for gated repos and faster downloads", |ui| {
            row(ui, "token", |ui| {
                secret(ui, &mut app.settings.hf_token);
            });
        });
        spec_updates(app, ui);
        section(ui, "app", "where this app keeps its files", |ui| {
            reading(ui, "theme", "dark, shared with the other bench-suite apps");
            reading(ui, "data folder", app.dir.display().to_string());
        });
        ui.horizontal(|ui| {
            if ui.add(Key::new("save settings").solid(READOUT).min_width(140.0)).clicked() {
                save = true;
            }
            if ui.add(Key::new("rescan models").outline(VALUE).min_width(140.0)).clicked() {
                save = true;
                app.rescan();
            }
        });
        if save {
            app.save_settings();
        }
    });
}
