use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::engine::MediaUpdate;
use inference_bench_core::library::{LocalModel, human_size};
use inference_bench_core::settings::ImageDefaults;
use specrt::media::{MediaEvent, MediaInfo, MediaJob, MediaOutput, ModelKind, Picture, parse_size};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Instant;

pub const NARROW: f32 = 720.0;

pub struct Result1 {
    pub picture: Picture,
    pub texture: Option<egui::TextureHandle>,
    pub prompt: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobRow {
    pub prompt: String,
    pub size: String,
    pub images: usize,
    pub steps: usize,
    pub steps_per_sec: f64,
    pub total_secs: f64,
    pub outcome: String,
}

pub struct Running {
    rx: Receiver<MediaUpdate>,
    cancel: Arc<AtomicBool>,
    pub step: usize,
    pub steps: usize,
    pub image: usize,
    pub images: usize,
    started: Instant,
    job: MediaJob,
}

pub struct GenerateState {
    pub prompt: String,
    pub negative: String,
    pub show_negative: bool,
    pub size: String,
    pub seed: u64,
    pub lock_seed: bool,
    pub steps: usize,
    pub guidance: f32,
    pub n: usize,
    pub running: Option<Running>,
    pub results: Vec<Result1>,
    pub rows: Vec<JobRow>,
    pub error: Option<String>,
    pub enlarged: Option<usize>,
    pub for_model: String,
}

impl Default for GenerateState {
    fn default() -> Self {
        Self { prompt: String::new(), negative: String::new(), show_negative: false, size: "1024x1024".into(), seed: 1, lock_seed: false, steps: 4, guidance: 1.0, n: 1, running: None, results: Vec::new(), rows: Vec::new(), error: None, enlarged: None, for_model: String::new() }
    }
}

impl GenerateState {
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    pub fn follow(&mut self, display: &str, info: &MediaInfo, defaults: &ImageDefaults) {
        if self.for_model == display {
            return;
        }
        self.for_model = display.to_string();
        self.size = if info.sizes.contains(&defaults.size) { defaults.size.clone() } else { info.default_size.clone() };
        self.steps = defaults.steps.max(1);
        self.guidance = defaults.guidance;
    }

    pub fn job(&self, offload_text_encoder: bool) -> Result<MediaJob, String> {
        let (w, h) = parse_size(&self.size).ok_or_else(|| format!("size {:?} is not WIDTHxHEIGHT", self.size))?;
        Ok(MediaJob { negative_prompt: self.negative.clone(), n: self.n.max(1), steps: Some(self.steps), guidance: Some(self.guidance), offload_text_encoder: Some(offload_text_encoder), ..MediaJob::image(&self.prompt, w, h, self.seed) })
    }

    pub fn start(&mut self, handle: &inference_bench_core::engine::Handle, model: &str, job: MediaJob) {
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = handle.generate_media(Some(model), job.clone(), cancel.clone());
        self.error = None;
        self.running = Some(Running { rx, cancel, step: 0, steps: job.steps.unwrap_or(0), image: 0, images: job.n, started: Instant::now(), job });
    }

    pub fn stop(&mut self) {
        if let Some(r) = &self.running {
            r.cancel.store(true, Ordering::SeqCst);
        }
    }

    pub fn pump(&mut self, ctx: &egui::Context) {
        let Some(r) = self.running.as_mut() else { return };
        let mut finished: Option<Result<MediaOutput, String>> = None;
        while let Ok(u) = r.rx.try_recv() {
            match u {
                MediaUpdate::Event(MediaEvent::Progress { image, step, steps }) => {
                    r.image = image;
                    r.step = step;
                    r.steps = steps;
                }
                MediaUpdate::Event(MediaEvent::Preview { .. }) => {}
                MediaUpdate::Done(o) => finished = Some(Ok(o)),
                MediaUpdate::Failed(e) => finished = Some(Err(e.message)),
            }
        }
        let Some(outcome) = finished else { return };
        let r = self.running.take().unwrap();
        let size = format!("{}x{}", r.job.width, r.job.height);
        match outcome {
            Ok(out) => {
                self.rows.push(JobRow { prompt: r.job.prompt.clone(), size, images: out.images.len(), steps: out.steps, steps_per_sec: out.steps_per_sec(), total_secs: out.total_secs, outcome: "done".into() });
                for p in out.images {
                    let img = egui::ColorImage::from_rgb([p.width, p.height], &p.rgb);
                    let texture = Some(ctx.load_texture(format!("gen-{}-{}", p.seed, self.results.len()), img, egui::TextureOptions::LINEAR));
                    self.results.insert(0, Result1 { picture: p, texture, prompt: r.job.prompt.clone() });
                }
                if !self.lock_seed {
                    self.seed = self.seed.wrapping_add(r.job.n as u64);
                }
            }
            Err(e) => {
                let stopped = r.cancel.load(Ordering::SeqCst);
                self.rows.push(JobRow { prompt: r.job.prompt.clone(), size, images: 0, steps: r.step, steps_per_sec: 0.0, total_secs: r.started.elapsed().as_secs_f64(), outcome: if stopped { "stopped".into() } else { "failed".into() } });
                if !stopped {
                    self.error = Some(e);
                }
            }
        }
    }
}

fn dice() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7);
    (t ^ (t >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 24
}

fn controls(app: &mut App, ui: &mut egui::Ui, info: &MediaInfo, model: &str) {
    let busy = app.generate.busy();
    let offload = app.settings.model_image.get(model).is_none_or(|d| d.offload_text_encoder);
    let g = &mut app.generate;
    section(ui, "prompt", "what to draw", |ui| {
        prose(ui, &mut g.prompt, "a red fox in studio light", 4);
        let label = if g.show_negative { "hide negative prompt" } else { "negative prompt" };
        if ui.add(Key::new(label).outline(LEGEND).id_salt("gen-negative-toggle")).clicked() {
            g.show_negative = !g.show_negative;
        }
        if g.show_negative {
            prose(ui, &mut g.negative, "what to keep out of the image", 2);
            hint(ui, "Used when guidance is above 1 on a model that runs classifier free guidance.");
        }
    });
    ui.add_space(8.0);
    section(ui, "settings", &format!("{} family", info.family), |ui| {
        row(ui, "size", |ui| {
            let mut sizes = info.sizes.clone();
            if !sizes.contains(&g.size) {
                sizes.insert(0, g.size.clone());
            }
            ui.allocate_ui(egui::vec2(150.0, KEY_H), |ui| {
                choice(ui, "gen-size", &mut g.size, sizes.into_iter().map(|s| (s.clone(), s)));
            });
        });
        row(ui, "seed", |ui| {
            ui.add(egui::DragValue::new(&mut g.seed).speed(1.0));
            if ui.add(Key::new("dice").outline(VALUE).id_salt("gen-dice")).clicked() {
                g.seed = dice();
            }
            ui.checkbox(&mut g.lock_seed, "lock");
        });
        row(ui, "steps", |ui| {
            ui.add(egui::DragValue::new(&mut g.steps).range(1..=200).speed(0.2));
        });
        row(ui, "guidance", |ui| {
            ui.add(egui::DragValue::new(&mut g.guidance).range(0.0..=30.0).speed(0.05).max_decimals(2));
        });
        row(ui, "images", |ui| {
            ui.add(egui::DragValue::new(&mut g.n).range(1..=info.max_images).speed(0.1));
        });
    });
    ui.add_space(8.0);
    let ready = !g.prompt.trim().is_empty() && !busy;
    let mut go = false;
    ui.horizontal(|ui| {
        if busy {
            if ui.add(Key::new("stop").outline(FAULT).min_width(120.0).id_salt("gen-stop")).clicked() {
                g.stop();
            }
        } else if ui.add(Key::new("generate image").solid(READOUT).enabled(ready).min_width(140.0).id_salt("gen-go")).clicked() {
            go = true;
        }
    });
    if let Some(r) = &g.running {
        let done = (r.image * r.steps.max(1) + r.step) as f32;
        let total = (r.images * r.steps.max(1)) as f32;
        progress(ui, "step", done, Some(total.max(1.0)), &format!("image {} of {}, step {} of {}", r.image + 1, r.images, r.step, r.steps));
    }
    if let Some(e) = &g.error {
        status(ui, false, e);
    }
    if go {
        match g.job(offload) {
            Ok(job) => {
                let handle = app.engine.handle();
                app.generate.start(&handle, model, job);
            }
            Err(e) => app.generate.error = Some(e),
        }
    }
}

fn results(app: &mut App, ui: &mut egui::Ui, columns: usize) {
    let g = &mut app.generate;
    if g.results.is_empty() {
        note(ui, "Generated images appear here. Each keeps its seed, so the same settings give the same picture.", LEGEND);
        return;
    }
    let gap = 8.0;
    let w = ((ui.available_width() - gap * (columns as f32 - 1.0)) / columns as f32).max(80.0);
    let mut save: Option<usize> = None;
    let mut copy: Option<usize> = None;
    let mut reuse: Option<usize> = None;
    for (chunk_i, chunk) in g.results.chunks(columns).enumerate() {
        ui.horizontal_top(|ui| {
            for (j, r) in chunk.iter().enumerate() {
                let i = chunk_i * columns + j;
                if j > 0 {
                    ui.add_space(gap - ui.spacing().item_spacing.x);
                }
                ui.allocate_ui_with_layout(egui::vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.set_width(w);
                    if let Some(t) = &r.texture {
                        let h = w * r.picture.height as f32 / r.picture.width.max(1) as f32;
                        let img = ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(w, h)).sense(egui::Sense::click()));
                        if img.clicked() {
                            g.enlarged = Some(i);
                        }
                    }
                    Line::new().legend("seed").set(r.picture.seed.to_string()).gap(8.0).value(format!("{}x{}", r.picture.width, r.picture.height)).size(10.5).elided(ui);
                    ui.horizontal_wrapped(|ui| {
                        if ui.add(Key::new("save").outline(VALUE).id_salt(("gen-save", i))).clicked() {
                            save = Some(i);
                        }
                        if ui.add(Key::new("copy").outline(VALUE).id_salt(("gen-copy", i))).clicked() {
                            copy = Some(i);
                        }
                        if ui.add(Key::new("reuse").outline(LEGEND).id_salt(("gen-reuse", i))).clicked() {
                            reuse = Some(i);
                        }
                    });
                });
            }
        });
        ui.add_space(gap);
    }
    if let Some(i) = save {
        let p = &g.results[i].picture;
        let dir = app.dir.join("images");
        let path = dir.join(format!("image-{}-{}.png", chrono::Utc::now().format("%Y%m%d-%H%M%S"), p.seed));
        app.notice = Some(match std::fs::create_dir_all(&dir).map_err(|e| e.to_string()).and_then(|_| p.png().map_err(|e| e.to_string())).and_then(|png| std::fs::write(&path, png).map_err(|e| e.to_string())) {
            Ok(()) => (true, format!("saved {}", path.display())),
            Err(e) => (false, format!("save: {e}")),
        });
    }
    if let Some(i) = copy {
        let p = &g.results[i].picture;
        ui.ctx().copy_image(egui::ColorImage::from_rgb([p.width, p.height], &p.rgb));
        app.notice = Some((true, format!("image with seed {} copied", p.seed)));
    }
    if let Some(i) = reuse {
        let r = &g.results[i];
        g.prompt = r.prompt.clone();
        g.seed = r.picture.seed;
        g.size = format!("{}x{}", r.picture.width, r.picture.height);
        g.lock_seed = true;
    }
}

const COLUMNS: [(&str, f32); 6] = [("prompt", 220.0), ("size", 80.0), ("images", 60.0), ("steps", 50.0), ("steps/s", 70.0), ("total", 70.0)];

fn requests(ui: &mut egui::Ui, rows: &[JobRow]) {
    if rows.is_empty() {
        return;
    }
    heading(ui, "requests");
    let shown: Vec<&JobRow> = rows.iter().rev().take(8).collect();
    let narrow = ui.available_width() < 560.0;
    let cols: &[(&str, f32)] = if narrow { &[("size", 76.0), ("steps", 46.0), ("steps/s", 64.0), ("total", 64.0), ("", 60.0)] } else { &COLUMNS };
    Table::new(cols, shown.len()).show(ui, |i, p, row, at| {
        let r = shown[i];
        let total = format!("{:.1} s", r.total_secs);
        let rate = if r.steps_per_sec > 0.0 { format!("{:.2}", r.steps_per_sec) } else { r.outcome.clone() };
        if narrow {
            cell(p, row, at(0), 76.0, &r.size, VALUE);
            cell(p, row, at(1), 46.0, &r.steps.to_string(), VALUE);
            cell(p, row, at(2), 64.0, &rate, TRACE);
            cell(p, row, at(3), 64.0, &total, TRACE);
            cell(p, row, at(4), 60.0, &r.outcome, LEGEND);
        } else {
            cell(p, row, at(0), 220.0, &r.prompt, VALUE);
            cell(p, row, at(1), 80.0, &r.size, VALUE);
            cell(p, row, at(2), 60.0, &r.images.to_string(), VALUE);
            cell(p, row, at(3), 50.0, &r.steps.to_string(), VALUE);
            cell(p, row, at(4), 70.0, &rate, TRACE);
            cell(p, row, at(5), 70.0, &total, TRACE);
        }
    });
}

fn enlarged(app: &mut App, ctx: &egui::Context) {
    let Some(i) = app.generate.enlarged else { return };
    let Some(r) = app.generate.results.get(i) else {
        app.generate.enlarged = None;
        return;
    };
    let mut close = false;
    let screen = ctx.content_rect();
    let side = (screen.width() - 80.0).min(screen.height() - 160.0).max(120.0);
    let m = egui::Modal::new(egui::Id::new("gen-enlarged")).backdrop_color(egui::Color32::from_black_alpha(150)).show(ctx, |ui| {
        ui.set_width(side);
        modal_title(ui, &format!("seed {}", r.picture.seed));
        if let Some(t) = &r.texture {
            let h = side * r.picture.height as f32 / r.picture.width.max(1) as f32;
            ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(side, h)));
        }
        note(ui, &r.prompt, LEGEND);
        footer(ui, |ui| {
            if ui.add(Key::new("close").outline(VALUE).min_width(100.0)).clicked() {
                close = true;
            }
        });
    });
    if close || m.should_close() {
        app.generate.enlarged = None;
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(l) = app.loaded() else { return };
    let Some(info) = l.media.clone() else { return };
    let defaults = app.settings.model_image.get(&l.display).cloned().unwrap_or_else(|| ImageDefaults::from_info(&info));
    app.generate.follow(&l.display, &info, &defaults);
    app.generate.pump(ui.ctx());
    if app.generate.busy() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
    let model = l.display.clone();
    let narrow = ui.available_width() < NARROW;
    let frame = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(16, 10));
    egui::Panel::top("gen-header").frame(frame).show_inside(ui, |ui| crate::modelbar::draw(app, ui, app.generate.busy(), |_| {}));
    if narrow {
        egui::CentralPanel::no_frame().show_inside(ui, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin::symmetric(10, 4)).show(ui, |ui| {
                let w = ui.available_width();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_max_width(w);
                    ui.spacing_mut().item_spacing.y = 8.0;
                    controls(app, ui, &info, &model);
                    results(app, ui, 1);
                    requests(ui, &app.generate.rows);
                });
            });
        });
    } else {
        egui::Panel::left("gen-controls").exact_size(340.0).resizable(false).frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin { left: 12, right: 8, top: 4, bottom: 4 })).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 8.0;
                controls(app, ui, &info, &model);
            });
        });
        egui::CentralPanel::no_frame().show_inside(ui, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin { left: 8, right: 12, top: 4, bottom: 4 }).show(ui, |ui| {
                let w = ui.available_width();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_max_width(w);
                    ui.spacing_mut().item_spacing.y = 8.0;
                    let columns = ((w / 260.0).floor() as usize).clamp(1, 4);
                    results(app, ui, columns);
                    requests(ui, &app.generate.rows);
                });
            });
        });
    }
    enlarged(app, ui.ctx());
}

pub fn is_image(m: &LocalModel) -> Option<MediaInfo> {
    specrt::media::is_pipeline(&m.path).then(|| specrt::media::info(&m.path).ok()).flatten().filter(|i| i.kind != ModelKind::Text)
}

pub struct ImageLoad {
    pub model: LocalModel,
    pub info: MediaInfo,
    pub defaults: ImageDefaults,
}

impl ImageLoad {
    pub fn open(app: &App, model: &LocalModel, info: MediaInfo) -> Self {
        let defaults = app.settings.model_image.get(&model.name).cloned().unwrap_or_else(|| ImageDefaults::from_info(&info));
        Self { model: model.clone(), info, defaults }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.info.components.iter().filter(|c| c.resident || (c.name == "text_encoder" && !self.defaults.offload_text_encoder)).map(|c| c.bytes).sum()
    }
}

pub fn load_page(app: &mut App, ui: &mut egui::Ui) -> bool {
    let Some(mut d) = app.image_load.take() else { return false };
    let (mut go, mut back) = (false, false);
    if !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        back = true;
    }
    let free = app.free_gpu_bytes();
    egui::Panel::top("img-load-head").frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 8 })).show_inside(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("load image model").font(legend_font(16.0)).extra_letter_spacing(2.0).color(VALUE));
            chip(ui, &d.model.name, READOUT, true);
            chip(ui, d.info.kind.label(), TRACE, false);
        });
    });
    egui::Panel::bottom("img-load-foot").frame(egui::Frame::NONE.fill(WELL).inner_margin(egui::Margin::symmetric(12, 8))).show_inside(ui, |ui| {
        footer(ui, |ui| {
            let label = if app.loaded_all().is_empty() { "load model" } else { "replace current" };
            if ui.add(Key::new(label).solid(READOUT).min_width(120.0).id_salt("img-load-go")).clicked() {
                go = true;
            }
            if ui.add(Key::new("cancel").outline(VALUE).min_width(100.0).id_salt("img-load-cancel")).clicked() {
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
                section(ui, "generation defaults", "what the Generate page starts with", |ui| {
                    row(ui, "size", |ui| {
                        ui.allocate_ui(egui::vec2(150.0, KEY_H), |ui| {
                            choice(ui, "img-size", &mut d.defaults.size, d.info.sizes.iter().map(|s| (s.clone(), s.clone())));
                        });
                    });
                    row(ui, "steps", |ui| {
                        ui.add(egui::DragValue::new(&mut d.defaults.steps).range(1..=200).speed(0.2));
                    });
                    row(ui, "guidance", |ui| {
                        ui.add(egui::DragValue::new(&mut d.defaults.guidance).range(0.0..=30.0).speed(0.05).max_decimals(2));
                    });
                    switch(ui, "text encoder", &mut d.defaults.offload_text_encoder, "release after encoding", "The text encoder runs once per prompt. Released, its memory goes back to the GPU before the denoiser steps; kept, the next prompt starts sooner.");
                });
                section(ui, "components", "where each part lives", |ui| {
                    for c in &d.info.components {
                        let place = match (c.resident, c.name.as_str()) {
                            (true, _) => "on the GPU for every request",
                            (false, "text_encoder") if d.defaults.offload_text_encoder => "loaded per prompt, then released",
                            (false, _) => "on the GPU while loaded",
                        };
                        Line::new().legend(&c.name.replace('_', " ")).column(ui, 120.0).value(human_size(c.bytes)).gap(10.0).note(place).wrapped(ui);
                    }
                    let need = d.resident_bytes();
                    Line::new().legend("resident").column(ui, 120.0).set(human_size(need)).gap(10.0).note("plus the activations the admission check reserves per request").wrapped(ui);
                    match free {
                        Some(f) if f >= need => status(ui, true, &format!("fits: {} free on the GPU", human_size(f))),
                        Some(f) => status(ui, false, &format!("{} more than the {} free", human_size(need - f), human_size(f))),
                        None => {}
                    }
                });
                hint(ui, "Requests are checked against free memory before any step runs: a VAE that does not fit decodes in tiles, guidance that does not fit runs as two passes, anything else is refused with the bytes it needs.");
            });
        });
    });
    if go {
        app.settings.model_image.insert(d.model.name.clone(), d.defaults.clone());
        let _ = app.settings.save(&app.dir);
        app.generate.for_model.clear();
        let engine = app.settings.engine_for(&d.model.name);
        app.load_model_placed(&d.model, 0, 1, None, Vec::new(), engine, false);
        if app.tab == crate::Tab::Models {
            app.tab = crate::Tab::Chat;
        }
    } else if !back {
        app.image_load = Some(d);
    }
    true
}
