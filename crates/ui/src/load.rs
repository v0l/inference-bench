use crate::detail::View;
use crate::{App, Tab};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{LocalModel, human_size};
use serde_json::{Value, json};
use specrt::anatomy::ModelAnatomy;
use specrt::options::{EngineChoice, Field, Group, Kind, Profile};
use std::collections::BTreeMap;

pub use inference_bench_core::fit::*;


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
    pub choice: EngineChoice,
    pub advanced: bool,
    pub texts: BTreeMap<String, String>,
    pub anatomy: Option<ModelAnatomy>,
    native_max: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfilePick {
    Profile(Profile),
    Custom,
}

pub fn display_value(v: &Value) -> String {
    match v {
        Value::Null => "auto".into(),
        Value::String(s) => s.clone(),
        Value::Array(a) if a.iter().all(Value::is_string) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "),
        other => other.to_string(),
    }
}

fn parse_text(f: &Field, text: &str) -> Option<Value> {
    let t = text.trim();
    if t.is_empty() {
        return Some(if f.key.ends_with(".unsplit") { json!([]) } else { Value::Null });
    }
    match f.key.as_str() {
        k if k.ends_with(".unsplit") => Some(json!(t.split(',').map(str::trim).filter(|n| !n.is_empty()).collect::<Vec<_>>())),
        k if k.ends_with("_tile") => serde_json::from_str::<Vec<u64>>(t).ok().filter(|a| a.len() == 4).map(|a| json!(a)),
        _ => Some(json!(t)),
    }
}

fn edit_field(ui: &mut egui::Ui, f: &Field, choice: &mut EngineChoice, texts: &mut BTreeMap<String, String>) {
    let current = choice.value(&f.key);
    match &f.kind {
        Kind::Bool => {
            let mut on = current.as_bool().unwrap_or(false);
            let text = if on { "on" } else { "off" };
            if ui.checkbox(&mut on, text).changed() {
                choice.set(&f.key, json!(on));
            }
        }
        Kind::Int { min, max } => match current.as_u64() {
            Some(mut n) => {
                if ui.add(egui::DragValue::new(&mut n).range(*min..=(*max).min(1 << 40)).speed(1.0)).changed() {
                    choice.set(&f.key, json!(n));
                }
                if f.optional && ui.add(Key::new("auto").outline(LEGEND)).clicked() {
                    choice.set(&f.key, Value::Null);
                }
            }
            None => {
                Line::new().set("auto").show(ui);
                if ui.add(Key::new("set").outline(VALUE)).clicked() {
                    choice.set(&f.key, json!((*min).max(1)));
                }
            }
        },
        Kind::Choice(options) => {
            let mut picked = display_value(&current);
            let mut all: Vec<(String, String)> = options.iter().map(|o| (o.to_string(), o.to_string())).collect();
            if f.optional {
                all.insert(0, ("auto".into(), "auto".into()));
            }
            ui.allocate_ui(egui::vec2(150.0, KEY_H), |ui| {
                if choice_of(ui, &f.key, &mut picked, all) {
                    choice.set(&f.key, if picked == "auto" { Value::Null } else { json!(picked) });
                }
            });
        }
        Kind::Text | Kind::Json => {
            let buf = texts.entry(f.key.clone()).or_insert_with(|| if current.is_null() { String::new() } else { display_value(&current) });
            let r = ui.add(egui::TextEdit::singleline(buf).hint_text("auto").desired_width(180.0));
            if r.changed() {
                if let Some(v) = parse_text(f, buf) {
                    choice.set(&f.key, v);
                }
            }
        }
    }
}

fn choice_of(ui: &mut egui::Ui, id: &str, picked: &mut String, options: Vec<(String, String)>) -> bool {
    choice(ui, id.to_string(), picked, options)
}

fn advanced_groups(ui: &mut egui::Ui, dialog: &mut LoadDialog) {
    let fields = specrt::options::schema();
    for group in Group::ALL.into_iter().filter(|g| g.advanced()) {
        let members: Vec<&Field> = fields.iter().filter(|f| f.group == group && f.key != "serve.max_model_len" && f.key != "serve.kv_cache_bytes").collect();
        if members.is_empty() {
            continue;
        }
        ui.add_space(8.0);
        section(ui, group.label(), "", |ui| {
            for f in members {
                row_help(ui, f.label, f.help, |ui| {
                    if f.key == "serve.max_num_seqs" {
                        let mut n = dialog.seqs;
                        ui.add(egui::Slider::new(&mut n, 1..=MAX_SEQS));
                        dialog.seqs = n;
                        dialog.choice.set(&f.key, json!(n));
                    } else {
                        edit_field(ui, f, &mut dialog.choice, &mut dialog.texts);
                    }
                    let at_default = if f.key == "serve.max_num_seqs" { Some(json!(dialog.seqs)) == f.default.as_u64().map(|d| json!(d)) } else { dialog.choice.is_default(&f.key) };
                    if !at_default {
                        if ui.add(Key::new("reset").outline(READOUT)).clicked() {
                            dialog.reset(&f.key);
                        }
                    }
                    Line::new().note(format!("default {}", display_value(&dialog.choice.base(&f.key)))).size(10.5).show(ui);
                });
            }
        });
    }
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
        let mut choice = app.settings.engine_for(&model.name);
        choice.set("serve.max_num_seqs", json!(seqs));
        let mut dialog = Self { model: model.clone(), view, ctx, max, seqs, kv_gb, served_names, remember: true, choice, advanced: false, texts: BTreeMap::new(), anatomy: None, native_max: max };
        dialog.sync();
        dialog
    }

    pub fn anatomy(&self) -> Option<&ModelAnatomy> {
        self.anatomy.as_ref()
    }

    pub fn extension_factor(&self) -> usize {
        self.choice.value("runtime.rope_extension.factor").as_u64().map_or(1, |f| f.max(1) as usize)
    }

    pub fn kv_dtype(&self) -> specrt::spec::KvDtype {
        match self.choice.value("runtime.kv_dtype").as_str() {
            Some("fp8") => specrt::spec::KvDtype::Fp8,
            _ => specrt::spec::KvDtype::F16,
        }
    }

    pub fn sync(&mut self) {
        let Ok(v) = &self.view else { return };
        let mut a = v.anatomy.clone();
        let mut spec = v.spec.clone();
        spec.runtime.kv_dtype = self.kv_dtype();
        a.kv_bytes_per_token = (specrt::runtime::kv_block_bytes(&spec, 1) / specrt::codegen::PAGE) as u64;
        if self.native_max == 0 {
            self.native_max = a.max_context;
        }
        a.max_context = self.native_max * self.extension_factor();
        self.max = max_context(&a);
        self.ctx = self.ctx.min(self.max);
        self.anatomy = Some(a);
    }

    pub fn can_load(&self, free: Option<u64>) -> bool {
        self.holds_one_request() && self.estimate(free).is_none_or(|e| e.fits() != Some(false))
    }

    pub fn kv_bytes(&self) -> Option<u64> {
        self.anatomy().map(|a| pool_bytes(a, self.ctx, self.seqs, self.kv_gb))
    }

    pub fn holds_one_request(&self) -> bool {
        match (self.anatomy(), self.kv_bytes()) {
            (Some(a), Some(kv)) => pool_holds_one(a, kv, self.ctx),
            _ => true,
        }
    }

    pub fn reset(&mut self, key: &str) {
        self.choice.reset(key);
        self.texts.remove(key);
        if key == "serve.max_num_seqs" {
            if let Some(n) = specrt::options::default_of(key).and_then(|d| d.as_u64()) {
                self.seqs = n as usize;
            }
        }
    }

    pub fn serve_options(&self) -> Option<specrt::serve::ServeOptions> {
        let base = specrt::serve::ServeOptions { max_model_len: Some(self.ctx), max_num_seqs: self.seqs, kv_cache_bytes: self.kv_bytes().filter(|_| self.seqs > 1), ..Default::default() };
        self.choice.apply(base).ok()
    }

    pub fn memory_gb(&self, free: Option<u64>) -> Option<f64> {
        self.estimate(free).map(|e| (e.weights + e.state + e.kv) as f64 / GB)
    }

    pub fn estimate(&self, free: Option<u64>) -> Option<Estimate> {
        self.anatomy().map(|a| estimate(a, pool_bytes(a, self.ctx, self.seqs, self.kv_gb), self.seqs, free))
    }
}

pub fn modal(app: &mut App, ctx: &egui::Context) {
    let Some(mut dialog) = app.load.take() else { return };
    let free = app.free_gpu_bytes();
    dialog.sync();
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
            section(ui, "profile", "one switch for the measured best, or for the least memory", |ui| {
                let mut pick = if dialog.choice.is_custom() { ProfilePick::Custom } else { ProfilePick::Profile(dialog.choice.profile) };
                let before = pick;
                let mut options: Vec<(ProfilePick, &str, Option<usize>)> = Profile::ALL.iter().map(|p| (ProfilePick::Profile(*p), p.label(), None)).collect();
                if dialog.choice.is_custom() {
                    options.push((ProfilePick::Custom, "custom", None));
                }
                segments(ui, &mut pick, &options);
                if pick != before {
                    if let ProfilePick::Profile(p) = pick {
                        dialog.choice.select(p);
                        dialog.seqs = dialog.choice.value("serve.max_num_seqs").as_u64().map_or(dialog.seqs, |n| n as usize);
                    }
                }
                match dialog.choice.profile {
                    Profile::MaxPerformance if !dialog.choice.is_custom() => hint(ui, "MTP, CUDA graphs, fused kernels, the fast loader and the prefix cache are on."),
                    Profile::LowMemory if !dialog.choice.is_custom() => hint(ui, "No MTP draft head copy, a small loader budget and no prefix cache snapshots."),
                    _ => hint(ui, &format!("{} advanced option(s) changed from the profile.", dialog.choice.overrides.len())),
                }
            });
            ui.add_space(8.0);
            section(ui, "memory and context", "the two settings most people touch", |ui| {
                if let Some(base) = dialog.anatomy().map(|a| (a.total_bytes + a.state_bytes_per_seq * dialog.seqs.max(1) as u64) as f64 / GB) {
                    if dialog.seqs > 1 {
                        let hi = free.map(|f| f as f64 / GB).unwrap_or(base + dialog.kv_gb).max(base + 0.5).max(base + dialog.kv_gb);
                        let mut mem = base + dialog.kv_gb;
                        row_help(ui, "gpu memory", "How much GPU memory this model may use, weights included. What is left after the weights becomes the KV cache every request shares.", |ui| {
                            if ui.add(egui::Slider::new(&mut mem, (base + 0.25)..=hi).step_by(0.25).fixed_decimals(1).suffix(" GB")).changed() {
                                dialog.kv_gb = ((mem - base) * 4.0).round() / 4.0;
                            }
                        });
                    } else {
                        Line::new().legend("gpu memory").column(ui, 120.0).value("sized by the context length with one request").size(11.0).show(ui);
                    }
                }
                row_help(ui, "context length", "Bigger contexts keep more of the conversation and cost more memory. The slider stops at what the model supports.", |ui| {
                    if dialog.view.is_ok() {
                        let lo = MIN_CTX.min(dialog.max);
                        ui.add(egui::Slider::new(&mut dialog.ctx, lo..=dialog.max).step_by(STEP as f64).suffix(" tokens"));
                    } else {
                        Line::new().set(format!("{} tokens", dialog.ctx)).show(ui);
                    }
                });
                if dialog.view.is_ok() {
                    let note = if dialog.extension_factor() > 1 { format!("{} tokens, {} trained x {} YaRN", dialog.max, dialog.native_max, dialog.extension_factor()) } else { format!("{} tokens", dialog.max) };
                    Line::new().legend("model maximum").value(note).size(11.0).show(ui);
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
                    let held = if dialog.seqs > 1 { format!("{} tokens shared by {} requests", pool_tokens(dialog.anatomy().unwrap(), e.kv), dialog.seqs) } else { format!("{} tokens", dialog.ctx) };
                    let kv_line = Line::new().legend("kv cache").column(ui, 120.0);
                    let kv_line = if dialog.holds_one_request() { kv_line.value(human_size(e.kv)) } else { kv_line.value(human_size(e.kv)).tint(FAULT) };
                    kv_line.gap(8.0).note(held).show(ui);
                    if let Some(a) = dialog.anatomy() {
                        let full = concurrency(a, e.kv, dialog.ctx);
                        let typical = concurrency(a, e.kv, TYPICAL_REQUEST.min(dialog.ctx));
                        Line::new().legend("concurrency").column(ui, 120.0).value(format!("{full:.2}x at {} tokens, {typical:.1}x at {} tokens", dialog.ctx, TYPICAL_REQUEST.min(dialog.ctx))).size(11.0).show(ui);
                    }
                    if let Ok(v) = &dialog.view {
                        use specrt::spec::KvDtype::{F16, Fp8};
                        let at = dialog.max;
                        Line::new().legend("kv at maximum").column(ui, 120.0).value(format!("{} f16, {} fp8 at {} tokens", human_size(kv_cost_at(&v.spec, F16, at)), human_size(kv_cost_at(&v.spec, Fp8, at)), at)).size(11.0).show(ui);
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
            ui.horizontal(|ui| {
                let label = if dialog.advanced { "hide advanced options" } else { "advanced options" };
                if ui.add(Key::new(label).outline(VALUE)).clicked() {
                    dialog.advanced = !dialog.advanced;
                }
                hint(ui, "every engine option, with its default");
            });
            if dialog.advanced {
                let w = ui.available_width();
                egui::ScrollArea::vertical().auto_shrink([false, true]).max_height((ui.ctx().content_rect().height() * 0.45).clamp(240.0, 560.0)).show(ui, |ui| {
                    ui.set_max_width(w);
                    advanced_groups(ui, &mut dialog);
                });
            }
            ui.add_space(8.0);
            switch(ui, "remember for this model", &mut dialog.remember, "use this context and parallel requests next time", "Saved with the app settings, per model.");
            footer(ui, |ui| {
                if ui.add_enabled(dialog.can_load(free), Key::new("load").solid(READOUT).min_width(100.0)).clicked() {
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
            app.settings.model_engine.insert(model.name.clone(), dialog.choice.clone());
            let _ = app.settings.save(&app.dir);
        }
        app.chat.load_error = None;
        let kv = dialog.view.is_ok().then(|| dialog.kv_bytes()).flatten().filter(|_| dialog.seqs > 1);
        app.load_model_named(&model, dialog.ctx, dialog.seqs, kv, names, dialog.choice.clone());
        if app.tab == Tab::Models {
            app.tab = Tab::Chat;
        }
    } else if !close {
        app.load = Some(dialog);
    }
}
