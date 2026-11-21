use crate::detail::View;
use crate::{App, Tab};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::cluster::{Planned, Sizing};
use specrt::solve::{PIN_POOL, PIN_SLOTS, PlacementRequest, Solved, Target};
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
    pub pin_seqs: bool,
    pub pin_kv: bool,
    pub solved: Option<Result<Solved, String>>,
    solve_key: String,
    native_max: usize,
    auto_extension: bool,
}

const MAX_EXTENSION: usize = 16;
const DEFAULT_EXTENSION: usize = 4;

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
            if r.changed()
                && let Some(v) = parse_text(f, buf) {
                    choice.set(&f.key, v);
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
                        if ui.add(egui::Slider::new(&mut n, 1..=MAX_SEQS)).changed() {
                            dialog.seqs = n;
                            dialog.pin_seqs = true;
                        }
                    } else {
                        edit_field(ui, f, &mut dialog.choice, &mut dialog.texts);
                    }
                    let at_default = if f.key == "serve.max_num_seqs" { !dialog.pin_seqs } else { dialog.choice.is_default(&f.key) };
                    if !at_default
                        && ui.add(Key::new("reset").outline(READOUT)).clicked() {
                            dialog.reset(&f.key);
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
        if remembered_seqs.is_some() {
            choice.set("serve.max_num_seqs", json!(seqs));
        }
        let fitted = app.models.preselect.iter().find(|p| p.model == model.name);
        let (ctx, kv_gb) = match (fitted, &view) {
            (Some(p), Ok(v)) => {
                let ctx = p.context.clamp(MIN_CTX.min(max), max);
                if p.kv != inference_bench_core::discover::Kv::F16 {
                    choice.set("runtime.kv_dtype", json!("fp8"));
                    if p.kv == inference_bench_core::discover::Kv::Fp8 {
                        choice.set("runtime.kv_f16_layers", json!(0));
                    }
                }
                (ctx, default_kv_gb(&v.anatomy, free, seqs, ctx))
            }
            _ => (ctx, kv_gb),
        };
        let pin_kv = app.settings.kv_gb_for(&model.name).is_some() || fitted.is_some();
        let mut dialog = Self { model: model.clone(), view, ctx, max, seqs, kv_gb, served_names, remember: true, choice, advanced: false, texts: BTreeMap::new(), anatomy: None, pin_seqs: remembered_seqs.is_some(), pin_kv, solved: None, solve_key: String::new(), native_max: max, auto_extension: false };
        dialog.sync();
        dialog.resolve(&crate::hardware::load_devices(app));
        dialog
    }

    pub fn resolve(&mut self, devices: &[specrt::placement::DeviceConfig]) {
        let key = format!("{:?}|{}|{}|{}|{}|{}|{:?}|{:?}", self.choice.profile, self.ctx, self.seqs, self.kv_gb, self.pin_seqs, self.pin_kv, self.choice.overrides, devices.iter().map(|d| (d.name.clone(), d.memory_gib.to_bits())).collect::<Vec<_>>());
        if key == self.solve_key {
            return;
        }
        self.solve_key = key;
        self.solved = self.solve(devices);
        if let Some(Ok(s)) = &self.solved {
            if !self.pin_seqs {
                self.seqs = s.slots;
            }
            if !self.pin_kv && s.slots > 1 {
                self.kv_gb = s.kv_pool_bytes as f64 / GB;
            }
        }
    }

    pub fn solve(&self, devices: &[specrt::placement::DeviceConfig]) -> Option<Result<Solved, String>> {
        if devices.is_empty() {
            return None;
        }
        let spec = self.kv_spec()?;
        self.anatomy()?;
        let mut pins = self.choice.overrides.clone();
        if self.pin_seqs {
            pins.insert(PIN_SLOTS.into(), json!(self.seqs));
        }
        if self.pin_kv && self.seqs > 1 {
            pins.insert(PIN_POOL.into(), json!(self.kv_bytes()?));
        }
        let target = Target { context: Some(self.ctx), memory: None, slots: None };
        let request = PlacementRequest { mode: self.choice.profile, target, pins, devices, spec: &spec };
        Some(specrt::solve::solve(&request).map_err(|e| inference_bench_core::cluster::friendly(&format!("{e:#}"))))
    }

    pub fn planned(&self, devices: &[specrt::placement::DeviceConfig]) -> Option<Result<Planned, String>> {
        let solved = self.solved.as_ref()?;
        Some(solved.clone().map(|s| Planned { sizing: Sizing { weight_bytes: self.anatomy().map(|a| a.total_bytes), kv_pool_bytes: Some(s.kv_pool_bytes) }, ctx: s.context, slots: s.slots, objective: specrt::placement::Objective::Throughput, devices: devices.to_vec(), placement: s.placement }))
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

    fn raw_kv_spec(&self) -> Option<specrt::spec::Spec> {
        let mut spec = self.view.as_ref().ok()?.spec.clone();
        spec.runtime.kv_dtype = self.kv_dtype();
        spec.runtime.kv_f16_layers = self.choice.value("runtime.kv_f16_layers").as_u64().map(|n| n as usize);
        Some(spec)
    }

    pub fn kv_spec(&self) -> Option<specrt::spec::Spec> {
        Some(self.raw_kv_spec()?.for_kv_context(self.ctx))
    }

    pub fn kv_note(&self) -> Option<(bool, String)> {
        self.raw_kv_spec()?.kv_context_note(self.ctx)
    }

    pub fn sync(&mut self) {
        let Ok(v) = &self.view else { return };
        let mut a = v.anatomy.clone();
        let spec = self.kv_spec().unwrap();
        a.kv_bytes_per_token = (specrt::runtime::kv_block_bytes(&spec, 1) / specrt::codegen::PAGE) as u64;
        if self.native_max == 0 {
            self.native_max = a.max_context;
        }
        a.max_context = self.native_max * MAX_EXTENSION;
        self.max = max_context(&a);
        self.ctx = self.ctx.min(self.max);
        self.anatomy = Some(a);
        self.follow_context();
    }

    fn follow_context(&mut self) {
        let native = self.native_max.max(1);
        let yarn = self.choice.value("runtime.rope_extension.type").as_str() == Some("yarn");
        if yarn && self.choice.value("runtime.rope_extension.factor").as_u64().is_none() {
            self.choice.set("runtime.rope_extension.factor", json!(DEFAULT_EXTENSION));
        }
        if self.ctx > native {
            let needed = self.ctx.div_ceil(native).min(MAX_EXTENSION);
            if !yarn {
                self.choice.set("runtime.rope_extension.type", json!("yarn"));
                self.auto_extension = true;
            }
            if needed > self.extension_factor() || !yarn || self.auto_extension {
                self.choice.set("runtime.rope_extension.factor", json!(needed));
            }
        } else if self.auto_extension {
            self.choice.reset("runtime.rope_extension.type");
            self.choice.reset("runtime.rope_extension.factor");
            self.auto_extension = false;
        }
    }

    pub fn plan_context(&self) -> Option<usize> {
        let a = self.anatomy()?;
        Some(pool_tokens(a, self.kv_bytes()?).min(self.max).div_ceil(self.seqs.max(1)).max(1))
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
        if key == "serve.max_num_seqs"
            && let Some(n) = specrt::options::default_of(key).and_then(|d| d.as_u64()) {
                self.seqs = n as usize;
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

pub fn card_line(c: &inference_bench_core::inventory::Card) -> String {
    let mut text = format!("{}  {}  {} free of {}", c.short(), c.label, human_size(c.free_bytes), human_size(c.total_bytes));
    if !c.local {
        text = format!("{}  on {}", text, c.host);
    }
    if c.display {
        text.push_str("  drives a display");
    }
    text
}

fn hardware_section(app: &mut App, ui: &mut egui::Ui, inv: &inference_bench_core::inventory::Inventory) {
    section(ui, "hardware", "the GPUs this model may use, the same switches as the Hardware tab", |ui| {
        if inv.cards.is_empty() {
            note(ui, "No GPU found yet.", LEGEND);
            return;
        }
        for c in &inv.cards {
            let mut on = c.selected;
            let r = ui.push_id(&c.name, |ui| ui.checkbox(&mut on, card_line(c))).inner;
            if r.changed() {
                inference_bench_core::inventory::toggle(&mut app.settings.cluster, inv, &c.name);
                let _ = app.settings.save(&app.dir);
            }
        }
        if inv.selected().next().is_none() {
            status(ui, false, "no GPU is ticked: tick one to load the model");
        }
    });
}

fn placement_bars(ui: &mut egui::Ui, inv: &inference_bench_core::inventory::Inventory, solved: &Solved) {
    ui.add_space(4.0);
    let head = Line::new().legend("load plan").size(10.5).show(ui);
    head.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, "load plan"));
    for s in &solved.placement.stages {
        let bytes = s.weight_bytes + s.kv_bytes + s.state_bytes;
        for name in s.device_name.split('+') {
            let card = inv.card(name);
            let label = card.map_or_else(|| specrt::hardware::display_name(name), |c| if c.local { c.short() } else { specrt::hardware::display_name(&c.name) });
            let free = card.map_or(0, |c| c.free_bytes);
            let text = format!("layers {}-{}  {} of {} free", s.start, s.end.saturating_sub(1), human_size(bytes), human_size(free));
            let r = Line::new().legend(&label).column(ui, 120.0).value(text.clone()).size(11.0).show(ui);
            r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("{label} {text}")));
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
            let fraction = if free == 0 { 1.0 } else { bytes as f32 / free as f32 };
            bar(&ui.painter_at(rect), rect, fraction.min(1.0), if fraction > 1.0 { FAULT } else { TRACE });
        }
    }
}

pub fn dialog_size(screen: egui::Vec2, advanced: bool) -> (f32, f32) {
    let width = 520.0f32.min(screen.x - 32.0).max(280.0);
    let room = (screen.y - 190.0).max(160.0);
    (width, if advanced { room } else { room.min(900.0) })
}

pub fn modal(app: &mut App, ctx: &egui::Context) {
    let Some(mut dialog) = app.load.take() else { return };
    let inv = app.inventory();
    let gpus = inv.free_local();
    let free = (!gpus.is_empty()).then(|| gpus.iter().sum::<u64>());
    dialog.sync();
    let est = dialog.estimate(free);
    let load_devices = inv.devices(app.hardware.machine.as_ref(), app.settings.cluster.network_link());
    dialog.resolve(&load_devices);
    let spread = dialog.planned(&load_devices).filter(|_| load_devices.len() > 1);
    let (mut go, mut close, mut alongside) = (false, false, false);
    let loaded_now = app.loaded_all();
    let alongside_block = (!loaded_now.is_empty()).then(|| app.alongside_blocked().or_else(|| est.as_ref().filter(|e| e.fits() == Some(false)).map(|e| format!("{} more than the free memory: unload a model first", human_size(e.total() - e.free.unwrap_or(0)))))).flatten();
    let r = egui::Modal::new(egui::Id::new("load-model"))
        .backdrop_color(egui::Color32::from_black_alpha(150))
        .show(ctx, |ui| {
            let (width, body_h) = dialog_size(ui.ctx().content_rect().size(), dialog.advanced);
            ui.set_width(width);
            modal_title(ui, &format!("Load {}", dialog.model.name));
            if let Err(e) = &dialog.view {
                status(ui, false, e);
                note(ui, "The engine will read the model itself and report what is wrong.", LEGEND);
            }
            let body_w = ui.available_width();
            egui::ScrollArea::vertical().auto_shrink([false, true]).max_height(body_h).show(ui, |ui| {
            ui.set_max_width(body_w);
            hardware_section(app, ui, &inv);
            ui.add_space(8.0);
            section(ui, "profile", "one switch for the measured best, or for the least memory", |ui| {
                let mut pick = if dialog.choice.is_custom() { ProfilePick::Custom } else { ProfilePick::Profile(dialog.choice.profile) };
                let before = pick;
                let mut options: Vec<(ProfilePick, &str, Option<usize>)> = Profile::ALL.iter().map(|p| (ProfilePick::Profile(*p), p.label(), None)).collect();
                if dialog.choice.is_custom() {
                    options.push((ProfilePick::Custom, "custom", None));
                }
                segments(ui, &mut pick, &options);
                if pick != before
                    && let ProfilePick::Profile(p) = pick {
                        dialog.choice.select(p);
                        dialog.pin_seqs = false;
                        dialog.pin_kv = false;
                        dialog.solve_key.clear();
                    }
                match dialog.choice.profile {
                    Profile::MaxPerformance if !dialog.choice.is_custom() => hint(ui, "MTP, CUDA graphs, fused kernels, the fast loader and the prefix cache are on."),
                    Profile::LowMemory if !dialog.choice.is_custom() => hint(ui, "fp8 KV cache, 2 parallel requests, smaller batches and prefill chunks, no MTP draft head, no prefix cache, small loader and image budgets."),
                    _ => hint(ui, &format!("{} advanced option(s) changed from the profile.", dialog.choice.overrides.len())),
                }
                match &dialog.solved {
                    Some(Ok(s)) => {
                        let r = Line::new().legend("placement").column(ui, 120.0).value(s.strategy.clone()).size(11.0).show(ui);
                        r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("placement {}", s.strategy)));
                        for why in &s.reasons {
                            hint(ui, why);
                        }
                        if let Some(short) = &s.shortfall {
                            status(ui, false, short);
                        }
                    }
                    Some(Err(why)) => status(ui, false, &format!("no placement fits: {why}")),
                    None => {}
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
                                dialog.pin_kv = true;
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
                    let note = if dialog.max > dialog.native_max { format!("{} tokens trained, up to {} with YaRN", dialog.native_max, dialog.max) } else { format!("{} tokens", dialog.max) };
                    Line::new().legend("model maximum").value(note).size(11.0).show(ui);
                    Line::new().note("Going past the trained length turns on YaRN, which changes short-context numerics slightly. Keep the context at or under the trained length unless you need more.").size(10.5).show(ui);
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
                    if let Some((ok, note)) = dialog.kv_note() {
                        status(ui, ok, &note);
                    }
                    if !dialog.holds_one_request() {
                        status(ui, false, &format!("the KV cache cannot hold one request of {} tokens: raise it or lower the context length. You can still try to load it.", dialog.ctx));
                    }
                    if e.state > 0 {
                        let each = if dialog.seqs > 1 { format!("{} each", human_size(e.state / dialog.seqs as u64)) } else { String::new() };
                        Line::new().legend("recurrent state").column(ui, 120.0).value(human_size(e.state)).gap(8.0).note(each).show(ui);
                    }
                    let total = Line::new().legend("total").column(ui, 120.0);
                    let total = if e.fits() == Some(false) { total.value(human_size(e.total())).tint(FAULT) } else { total.value(human_size(e.total())) };
                    total.show(ui);
                    match e.free {
                        Some(f) if gpus.len() > 1 => {
                            let label = format!("free on {} gpus", gpus.len());
                            let r = Line::new().legend(&label).column(ui, 120.0).measured(human_size(f)).show(ui);
                            r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, label.clone()));
                            match &spread {
                                Some(Ok(p)) => {
                                    let largest = gpus.iter().copied().max().unwrap_or(0);
                                    let line = if p.spans() && e.total() > largest {
                                        format!("{} is more than the largest gpu ({}): it will span {} gpus, split as below", human_size(e.total()), human_size(largest), p.placement.stages.len())
                                    } else {
                                        format!("fits across {} gpus in {} stage(s)", gpus.len(), p.placement.stages.len())
                                    };
                                    status(ui, true, &line);
                                }
                                Some(Err(why)) if e.total() > gpus.iter().copied().max().unwrap_or(0) => status(ui, false, &format!("the planner says it does not fit across {} gpus: {why}. Loading is blocked: lower the memory, the context or the parallel requests.", gpus.len())),
                                Some(Err(why)) => status(ui, false, &format!("the planner says it does not fit across {} gpus: {why}. You can still try to load it.", gpus.len())),
                                None => {}
                            }
                        }
                        Some(f) => {
                            Line::new().legend("free on the gpu").column(ui, 120.0).measured(human_size(f)).show(ui);
                            match e.fits() {
                                Some(false) => status(ui, false, &format!("the planner says it is {} over free memory. You can still try to load it.", human_size(e.total() - f))),
                                _ => status(ui, true, "fits in free memory"),
                            }
                        }
                        None => note(ui, "No GPU reading yet, so the estimate cannot be checked against free memory.", LEGEND),
                    }
                    if let Some(Ok(s)) = &dialog.solved {
                        placement_bars(ui, &inv, s);
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
                advanced_groups(ui, &mut dialog);
            }
            ui.add_space(8.0);
            switch(ui, "remember for this model", &mut dialog.remember, "use this context and parallel requests next time", "Saved with the app settings, per model.");
            });
            if let Some(why) = &alongside_block {
                status(ui, false, &format!("cannot load alongside: {why}"));
            }
            footer(ui, |ui| {
                let label = if loaded_now.is_empty() { "load" } else { "replace current" };
                if ui.add(Key::new(label).solid(READOUT).min_width(100.0).id_salt("load-replace")).clicked() {
                    go = true;
                }
                if !loaded_now.is_empty() {
                    let key = ui.add(Key::new("load alongside").outline(READOUT).enabled(alongside_block.is_none()).min_width(100.0).id_salt("load-alongside"));
                    if key.clicked() {
                        go = true;
                        alongside = true;
                    }
                    if let Some(why) = &alongside_block {
                        key.on_hover_text(why.clone());
                    }
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
            if dialog.pin_seqs {
                app.settings.model_seqs.insert(model.name.clone(), dialog.seqs);
            } else {
                app.settings.model_seqs.remove(&model.name);
            }
            if dialog.pin_kv && dialog.seqs > 1 && dialog.view.is_ok() {
                app.settings.model_kv_gb.insert(model.name.clone(), dialog.kv_gb);
            } else {
                app.settings.model_kv_gb.remove(&model.name);
            }
            app.settings.model_served_names.insert(model.name.clone(), names.clone());
            app.settings.model_engine.insert(model.name.clone(), dialog.choice.clone());
            let _ = app.settings.save(&app.dir);
        }
        app.chat.load_error = None;
        let own = crate::hardware::planning_machine(app).map_or_else(specrt::hardware::host_name, |m| m.host);
        let outcome = match dialog.solved.clone() {
            Some(Ok(solved)) if solved.placement.stages.iter().any(|s| specrt::hardware::host_of(s.device_name.split('+').next().unwrap_or_default()) != own) => {
                if alongside {
                    Err("not loaded alongside: this model spans several hosts and needs them all".to_string())
                } else {
                    let planned = dialog.planned(&load_devices).and_then(Result::ok);
                    match planned {
                        Some(planned) => crate::hardware::load_spanning(app, &model, planned, solved.context, solved.slots, Some(solved.kv_pool_bytes), names, dialog.choice.clone()),
                        None => Err("no plan to load".to_string()),
                    }
                }
            }
            Some(Ok(solved)) if alongside && solved.spans() => Err("not loaded alongside: this model spans several gpus and needs them all".to_string()),
            Some(Ok(solved)) => {
                let mut engine = dialog.choice.clone();
                let (ctx, slots, pool) = (solved.context, solved.slots, solved.kv_pool_bytes);
                engine.solved = Some(Box::new(solved));
                app.load_model_placed(&model, ctx, slots, Some(pool), names, engine, alongside);
                Ok(())
            }
            Some(Err(why)) => Err(format!("not loaded: {why}")),
            None if load_devices.is_empty() && !inv.cards.is_empty() => Err("not loaded: no gpu selected. Tick a gpu under hardware in this dialog".to_string()),
            None if inv.local().any(|c| c.ordinal == Some(0) && !c.selected) => Err("not loaded: there is no placement plan for this model, and without one it would load on GPU 0, which is switched off".to_string()),
            None => {
                let kv = dialog.view.is_ok().then(|| dialog.kv_bytes()).flatten().filter(|_| dialog.seqs > 1);
                app.load_model_placed(&model, dialog.ctx, dialog.seqs, kv, names, dialog.choice.clone(), alongside);
                Ok(())
            }
        };
        if let Err(e) = outcome {
            app.notice = Some((false, e));
        }
        if app.tab == Tab::Models {
            app.tab = Tab::Chat;
        }
    } else if !close {
        app.load = Some(dialog);
    }
}
