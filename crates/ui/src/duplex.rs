use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{LocalModel, human_size};
use specrt::duplex::service::{Footprint, footprint};

pub const MIN_CTX: usize = 512;
pub const MAX_CTX: usize = 262_144;

pub struct DuplexLoad {
    pub model: LocalModel,
    pub sessions: usize,
    pub ctx: usize,
    pub device: Option<usize>,
    pub footprint: Result<Footprint, String>,
}

pub fn is_duplex(m: &LocalModel) -> bool {
    inference_bench_core::library::is_duplex(m)
}

impl DuplexLoad {
    pub fn open(app: &App, model: &LocalModel) -> Self {
        let sessions = app.settings.model_seqs.get(&model.name).copied().unwrap_or(4).clamp(1, 32);
        let ctx = app.settings.model_ctx.get(&model.name).copied().unwrap_or(8192).clamp(MIN_CTX, MAX_CTX);
        let device = app.inventory().local().filter(|c| c.selected).max_by_key(|c| c.free_bytes).and_then(|c| c.ordinal);
        Self { model: model.clone(), sessions, ctx, device, footprint: footprint(&model.path, ctx).map_err(|e| format!("{e:#}")) }
    }

    fn refresh(&mut self) {
        self.footprint = footprint(&self.model.path, self.ctx).map_err(|e| format!("{e:#}"));
    }
}

pub fn load_page(app: &mut App, ui: &mut egui::Ui) -> bool {
    let Some(mut d) = app.duplex_load.take() else { return false };
    let (mut go, mut back) = (false, false);
    if !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        back = true;
    }
    let inv = app.inventory();
    let cards: Vec<(usize, String, u64)> = inv.local().filter(|c| c.selected).filter_map(|c| c.ordinal.map(|o| (o, c.short(), c.free_bytes))).collect();
    let free = d.device.and_then(|o| cards.iter().find(|c| c.0 == o)).map(|c| c.2).or_else(|| app.free_gpu_bytes());
    egui::Panel::top("duplex-load-head").frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 8 })).show_inside(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("load speech model").font(legend_font(16.0)).extra_letter_spacing(2.0).color(VALUE));
            chip(ui, &d.model.name, READOUT, true);
            chip(ui, "duplex", TRACE, false);
        });
    });
    egui::Panel::bottom("duplex-load-foot").frame(egui::Frame::NONE.fill(WELL).inner_margin(egui::Margin::symmetric(12, 8))).show_inside(ui, |ui| {
        footer(ui, |ui| {
            let label = if app.loaded_all().is_empty() { "load model" } else { "replace current" };
            if ui.add(Key::new(label).solid(READOUT).min_width(120.0).id_salt("duplex-load-go")).clicked() {
                go = true;
            }
            if ui.add(Key::new("cancel").outline(VALUE).min_width(100.0).id_salt("duplex-load-cancel")).clicked() {
                back = true;
            }
        });
    });
    egui::CentralPanel::no_frame().show_inside(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(12, 6)).show(ui, |ui| {
            let w = ui.available_width();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.set_max_width(w.min(760.0));
                ui.spacing_mut().item_spacing.y = 8.0;
                let mut changed = false;
                section(ui, "sessions", "conversations served at once", |ui| {
                    row(ui, "sessions", |ui| {
                        changed |= ui.add(egui::DragValue::new(&mut d.sessions).range(1..=32).speed(0.1)).changed();
                    });
                    row(ui, "context", |ui| {
                        changed |= ui.add(egui::DragValue::new(&mut d.ctx).range(MIN_CTX..=MAX_CTX).speed(64.0)).changed();
                    });
                    if let Ok(f) = &d.footprint {
                        let minutes = d.ctx as f64 * f.frame_seconds / 60.0;
                        Line::new().legend("longest call").column(ui, 120.0).value(format!("{minutes:.0} min")).gap(10.0).note(format!("one position per {:.0} ms frame", f.frame_seconds * 1e3)).wrapped(ui);
                    }
                });
                if changed {
                    d.refresh();
                }
                section(ui, "placement", "where the model runs", |ui| {
                    if !cards.is_empty() {
                        row(ui, "gpu", |ui| {
                            let mut picked = d.device.unwrap_or(cards[0].0);
                            let wide = ui.available_width().clamp(120.0, 220.0);
                            ui.allocate_ui(egui::vec2(wide, KEY_H), |ui| {
                                choice(ui, "duplex-gpu", &mut picked, cards.iter().map(|(o, name, f)| (*o, format!("{name}, {} free", human_size(*f)))));
                            });
                            d.device = Some(picked);
                        });
                    }
                    match &d.footprint {
                        Ok(f) => {
                            Line::new().legend("weights").column(ui, 120.0).value(human_size(f.weights)).wrapped(ui);
                            Line::new().legend("per session").column(ui, 120.0).value(human_size(f.per_session)).gap(10.0).note("KV for the context, recurrent and codec state").wrapped(ui);
                            let need = f.total(d.sessions);
                            Line::new().legend("total").column(ui, 120.0).set(human_size(need)).wrapped(ui);
                            match free {
                                Some(fr) if fr >= need => status(ui, true, &format!("fits: {} free on the GPU", human_size(fr))),
                                Some(fr) => status(ui, false, &format!("{} more than the {} free", human_size(need - fr), human_size(fr))),
                                None => {}
                            }
                        }
                        Err(e) => status(ui, false, e),
                    }
                });
                hint(ui, "A duplex model talks and listens at once. Once loaded, start the server and connect a client to the /v1/duplex websocket: frames of 24 kHz PCM16 audio go both ways, the model's words come back as text events.");
            });
        });
    });
    if go {
        app.settings.model_seqs.insert(d.model.name.clone(), d.sessions);
        app.settings.model_ctx.insert(d.model.name.clone(), d.ctx);
        let _ = app.settings.save(&app.dir);
        let mut engine = app.settings.engine_for(&d.model.name);
        if let Some(dev) = d.device {
            engine.overrides.insert(specrt::media::DEVICE_KEY.into(), serde_json::json!(dev));
        }
        app.load_model_placed(&d.model, d.ctx, d.sessions, None, Vec::new(), engine, false);
        if app.tab == crate::Tab::Models {
            app.tab = crate::Tab::Chat;
        }
    } else if !back {
        app.duplex_load = Some(d);
    }
    true
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(l) = app.loaded() else { return };
    let frame = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(16, 10));
    egui::Panel::top("duplex-header").frame(frame).show_inside(ui, |ui| crate::modelbar::draw(app, ui, false, |_| {}));
    let url = app.server.running.as_ref().map(|h| h.url());
    egui::CentralPanel::no_frame().show_inside(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(12, 6)).show(ui, |ui| {
            let w = ui.available_width();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.set_max_width(w.min(760.0));
                ui.spacing_mut().item_spacing.y = 8.0;
                section(ui, "duplex speech", "a voice model, not a text chat", |ui| {
                    Line::new().legend("model").column(ui, 120.0).value(&l.display).wrapped(ui);
                    Line::new().legend("sessions").column(ui, 120.0).value(l.max_num_seqs.to_string()).wrapped(ui);
                    Line::new().legend("context").column(ui, 120.0).value(format!("{} frames", l.ctx)).wrapped(ui);
                    for (gpu, bytes) in &l.landed {
                        Line::new().legend(gpu).column(ui, 120.0).value(human_size(*bytes)).wrapped(ui);
                    }
                });
                section(ui, "connect", "stream audio to it", |ui| {
                    match &url {
                        Some(u) => {
                            let ws = inference_bench_core::duplex_client::ws_url(u, "/v1/duplex");
                            Line::new().legend("websocket").column(ui, 120.0).value(&ws).wrapped(ui);
                            Line::new().legend("try it").column(ui, 120.0).value(format!("inference-bench duplex --url {u} --wav in.wav --out out.wav")).wrapped(ui);
                        }
                        None => {
                            status(ui, false, "the server is off: start it on the Server tab to open /v1/duplex");
                            if ui.add(Key::new("server tab").outline(VALUE).id_salt("duplex-server")).clicked() {
                                app.tab = crate::Tab::Server;
                            }
                        }
                    }
                });
                hint(ui, "Send a JSON start message with the card and voice, then 6400 byte frames of 24 kHz mono PCM16. The OpenAI Realtime subset is on /v1/realtime.");
            });
        });
    });
}
