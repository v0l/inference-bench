use crate::App;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::settings::Endpoint;
use std::path::PathBuf;

#[derive(Default)]
pub struct SettingsState {
    pub new_folder: String,
    pub new_endpoint: String,
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "settings", |ui| {
        let mut save = false;
        section(ui, "backend endpoints", "OpenAI-compatible servers the chat and bench tabs can talk to", |ui| {
            let mut remove = None;
            for (i, e) in app.settings.endpoints.iter_mut().enumerate() {
                ui.push_id(i, |ui| {
                    row(ui, &e.label, |ui| {
                        field_then(ui, &mut e.url, "http://127.0.0.1:8080", 96.0, |ui| {
                            if ui.add(Key::new("remove").outline(FAULT)).clicked() {
                                remove = Some(i);
                            }
                        });
                    });
                    row(ui, "api key", |ui| {
                        secret(ui, &mut e.api_key);
                    });
                });
            }
            if let Some(i) = remove {
                app.settings.endpoints.remove(i);
            }
            let st = &mut app.settings_ui;
            let mut add = false;
            let ready = !st.new_endpoint.trim().is_empty();
            field_then(ui, &mut st.new_endpoint, "name for a new endpoint", 96.0, |ui| {
                add = ui.add(Key::new("add").outline(VALUE).enabled(ready)).clicked();
            });
            if add {
                app.settings.endpoints.push(Endpoint {
                    label: st.new_endpoint.trim().to_string(),
                    backend: "vllm".into(),
                    url: "http://127.0.0.1:8000".into(),
                    api_key: String::new(),
                });
                st.new_endpoint.clear();
            }
            hint(ui, "Keys are kept in settings.json in the data folder, readable only by you.");
        });
        section(ui, "local runtimes", "specrt and transformers run as worker processes", |ui| {
            row(ui, "specrt worker", |ui| {
                field(ui, &mut app.settings.specrt_worker, "inference-bench-specrt next to this app");
            });
            row(ui, "specrt context", |ui| {
                ui.add(egui::DragValue::new(&mut app.settings.specrt_ctx).range(512..=131072).speed(64));
            });
            row(ui, "python", |ui| {
                field(ui, &mut app.settings.python, "python3");
            });
        });
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
        section(ui, "appearance", "shared by every bench-suite app", |ui| {
            reading(ui, "theme", "bench instrument, dark");
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
