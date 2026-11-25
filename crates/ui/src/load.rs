use crate::detail::View;
use crate::{App, Tab};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::cluster::{Planned, Sizing};
use specrt::solve::{PIN_POOL, PIN_SLOTS, PlacementRequest, Solved, Target};
use inference_bench_core::library::{LocalModel, human_size};
use serde_json::{Value, json};
use specrt::anatomy::ModelAnatomy;
use specrt::options::{EngineChoice, Field, Kind, Profile};
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
    pub open_sections: std::collections::BTreeSet<&'static str>,
    pub search: String,
    pub details: bool,
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

impl LoadDialog {
    pub fn open(app: &App, model: &LocalModel) -> Self {
        let view = View::served(model);
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
        let mut dialog = Self { model: model.clone(), view, ctx, max, seqs, kv_gb, served_names, remember: true, choice, advanced: false, open_sections: Default::default(), search: String::new(), details: false, texts: BTreeMap::new(), anatomy: None, pin_seqs: remembered_seqs.is_some(), pin_kv, solved: None, solve_key: String::new(), native_max: max, auto_extension: false };
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

fn gpu_rows(app: &mut App, ui: &mut egui::Ui, inv: &inference_bench_core::inventory::Inventory) {
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

const WIDE: f32 = 860.0;
const SIDE_W: f32 = 460.0;
const LABEL_COL: f32 = 180.0;
const GRID_MIN: f32 = 520.0;

struct Env {
    inv: inference_bench_core::inventory::Inventory,
    free: Option<u64>,
    gpus: Vec<u64>,
    devices: Vec<specrt::placement::DeviceConfig>,
    has_mtp: bool,
}

pub fn tokens(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn shorten_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 3 {
        return s.to_string();
    }
    let keep = max - 1;
    let head = keep.div_ceil(2);
    let tail = keep / 2;
    let a: String = chars[..head].iter().collect();
    let b: String = chars[chars.len() - tail..].iter().collect();
    format!("{a}…{b}")
}

fn split_unit(text: String) -> (String, String) {
    match text.split_once(' ') {
        Some((n, u)) => (n.to_string(), u.to_string()),
        None => (text, String::new()),
    }
}

fn changed_marker(ui: &mut egui::Ui) {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(8.0, 14.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.0, READOUT);
    r.on_hover_text("changed from its default");
}

fn is_default(d: &LoadDialog, f: &Field) -> bool {
    match f.key.as_str() {
        "serve.max_num_seqs" => !d.pin_seqs,
        "serve.kv_cache_bytes" => !d.pin_kv,
        "serve.max_model_len" => true,
        "runtime.kv_dtype" => kv_kind_of(&d.choice) == KvKind::Auto,
        _ => d.choice.is_default(&f.key),
    }
}

fn label_block(ui: &mut egui::Ui, label: &str, changed: bool) -> bool {
    let mut reset = false;
    ui.add(egui::Label::new(legend(label)).wrap());
    if changed {
        ui.horizontal(|ui| {
            changed_marker(ui);
            if ui.add(Key::new("reset").outline(READOUT).id_salt(("reset", label))).clicked() {
                reset = true;
            }
        });
    }
    reset
}

fn help_line(ui: &mut egui::Ui, help: &str) {
    let edge = ui.ctx().content_rect().right() - 10.0;
    let w = ui.available_width().min((edge - ui.cursor().left()).max(80.0));
    ui.allocate_ui_with_layout(egui::vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
        ui.set_max_width(w);
        ui.add(egui::Label::new(egui::RichText::new(help).size(10.5).color(LEGEND)).wrap());
    });
}

fn grid_row(ui: &mut egui::Ui, label: &str, help: &str, changed: bool, control: impl FnOnce(&mut egui::Ui)) -> bool {
    let mut reset = false;
    if ui.available_width() >= GRID_MIN {
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(LABEL_COL, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(LABEL_COL);
                reset = label_block(ui, label, changed);
            });
            ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                control(ui);
                if !help.is_empty() {
                    help_line(ui, help);
                }
            });
        });
    } else {
        reset = label_block(ui, label, changed);
        control(ui);
        if !help.is_empty() {
            help_line(ui, help);
        }
    }
    ui.add_space(6.0);
    reset
}

fn slider_width(ui: &mut egui::Ui) {
    let room = ui.available_width() - 130.0;
    ui.spacing_mut().slider_width = room.clamp(60.0, 420.0);
}

fn with_default(d: &LoadDialog, f: &Field) -> String {
    format!("{} Default {}.", f.help, display_value(&d.choice.base(&f.key)))
}

fn context_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field, env: &Env) {
    let fits = if d.view.is_ok() { d.anatomy().and_then(|a| fitting_context(a, env.free, d.seqs)).map(|n| n.min(d.max)) } else { None };
    let (lo, max, native, ok, ctx) = (MIN_CTX.min(d.max), d.max, d.native_max, d.view.is_ok(), d.ctx);
    let mut value = ctx;
    grid_row(ui, "context length", f.help, false, |ui| {
        if ok {
            slider_width(ui);
            let slider = egui::Slider::new(&mut value, lo..=max)
                .step_by(STEP as f64)
                .suffix(" tokens")
                .custom_formatter(|v, _| tokens(v as usize))
                .custom_parser(|s| s.replace([',', ' '], "").parse::<f64>().ok());
            ui.add(slider);
            ui.horizontal_wrapped(|ui| {
                if let Some(n) = fits {
                    Line::new().legend("max that fits").value(format!("{} tokens", tokens(n))).size(11.0).show(ui);
                    if n != ctx && ui.add(Key::new("use max that fits").outline(READOUT).id_salt("ctx-fit")).clicked() {
                        value = n.max(lo);
                    }
                }
                let note = if max > native { format!("trained {}, up to {} with YaRN", tokens(native), tokens(max)) } else { format!("model maximum {}", tokens(max)) };
                Line::new().note(note).size(10.5).show(ui);
            });
            if ctx > native {
                hint(ui, "Going past the trained length turns on YaRN, which changes short-context numerics slightly.");
            }
        } else {
            Line::new().set(format!("{} tokens", ctx)).show(ui);
        }
    });
    d.ctx = value;
}

fn kv_type_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field) {
    let current = kv_kind_of(&d.choice);
    let mut pick = current;
    let effective = kv_kind_effective(&d.choice);
    let sizes: Vec<(KvKind, u64)> = KvKind::ALL
        .iter()
        .map(|k| {
            let shown = if *k == KvKind::Auto { effective } else { *k };
            let bytes = d.view.as_ref().map_or(0, |v| kv_bytes_per_token(&v.spec, shown, d.ctx) * d.ctx as u64);
            (*k, bytes)
        })
        .collect();
    let labels: Vec<String> = sizes.iter().map(|(k, b)| if *b == 0 { k.label().to_string() } else { format!("{} {}", k.label(), human_size(*b)) }).collect();
    let options: Vec<(KvKind, &str, Option<usize>)> = sizes.iter().zip(&labels).map(|((k, _), l)| (*k, l.as_str(), None)).collect();
    let f16_bytes = sizes.iter().find(|(k, _)| *k == KvKind::F16).map_or(0, |s| s.1);
    let picked_bytes = sizes.iter().find(|(k, _)| *k == pick).map_or(0, |s| s.1);
    let saving = if f16_bytes > 0 && picked_bytes > 0 { format!("{:.2}x the size of fp16 at {} tokens. ", picked_bytes as f64 / f16_bytes as f64, tokens(d.ctx)) } else { String::new() };
    let help = format!("{saving}{}", pick.help());
    let changed = !is_default(d, f);
    let reset = grid_row(ui, "kv cache type", &help, changed, |ui| {
        segments(ui, &mut pick, &options);
    });
    if reset {
        set_kv_kind(&mut d.choice, KvKind::Auto);
    } else if pick != current {
        set_kv_kind(&mut d.choice, pick);
    }
}

fn kv_size_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field, env: &Env) {
    let Some(a) = d.anatomy().cloned() else { return };
    let per = a.kv_bytes_per_token.max(1);
    let pooled = d.seqs > 1;
    let mut gb = if pooled { d.kv_gb } else { kv_gb_for_tokens(per, d.ctx) };
    let mut toks = if pooled { tokens_for_kv_gb(per, d.kv_gb) } else { d.ctx };
    let top = max_kv_gb(&a, env.free, d.seqs, d.max).max(gb);
    let token_cap = d.max.max(1) * d.seqs.max(1);
    let (mut gb_changed, mut tok_changed) = (false, false);
    let help = if pooled { format!("Shared by {} parallel requests. Change either number and the other follows.", d.seqs) } else { "With one request the cache follows the context length, so the two move together.".to_string() };
    let changed = !is_default(d, f);
    let reset = grid_row(ui, "kv cache size", &help, changed, |ui| {
        ui.horizontal_wrapped(|ui| {
            gb_changed = ui.add(egui::DragValue::new(&mut gb).speed(0.1).range(0.1..=top.max(0.2)).suffix(" GB").fixed_decimals(2)).changed();
            tok_changed = ui.add(egui::DragValue::new(&mut toks).speed(256.0).range(1..=token_cap).suffix(" tokens")).changed();
        });
    });
    if reset {
        d.pin_kv = false;
    } else if gb_changed {
        if pooled {
            d.kv_gb = gb;
            d.pin_kv = true;
        } else {
            d.ctx = tokens_for_kv_gb(per, gb).clamp(MIN_CTX.min(d.max), d.max);
        }
    } else if tok_changed {
        if pooled {
            d.kv_gb = kv_gb_for_tokens(per, toks);
            d.pin_kv = true;
        } else {
            d.ctx = toks.clamp(MIN_CTX.min(d.max), d.max);
        }
    }
}

fn slots_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field) {
    let mut n = d.seqs;
    let help = with_default(d, f);
    let changed = !is_default(d, f);
    let reset = grid_row(ui, "parallel requests", &help, changed, |ui| {
        slider_width(ui);
        ui.add(egui::Slider::new(&mut n, 1..=MAX_SEQS));
    });
    if reset {
        d.reset(&f.key);
    } else if n != d.seqs {
        d.seqs = n;
        d.pin_seqs = true;
    }
}

fn offload_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field) {
    let Some(total) = d.anatomy().map(|a| a.total_bytes as f64 / GB) else { return };
    let budget = d.choice.value(&f.key).as_u64().unwrap_or(0);
    let mut on_gpu = if budget == 0 { total } else { (budget as f64 / GB).min(total) };
    let before = on_gpu;
    let lo = 0.5f64.min(total);
    let help = if budget > 0 { format!("The other {:.1} GB of weights stay in host memory and stream in while earlier layers compute.", (total - budget as f64 / GB).max(0.0)) } else { "Every layer lives on the GPU, the fastest setting. Lower it to stream the rest from host memory.".to_string() };
    let changed = !is_default(d, f);
    let reset = grid_row(ui, "gpu offload", &help, changed, |ui| {
        slider_width(ui);
        let slider = egui::Slider::new(&mut on_gpu, lo..=total.max(lo + 0.01))
            .custom_formatter(move |v, _| if v >= total - 0.01 { "all on GPU".to_string() } else { format!("{v:.1} GB on GPU") })
            .custom_parser(|s| s.trim().trim_end_matches("GB").trim().parse().ok());
        ui.add(slider);
    });
    if reset {
        d.reset(&f.key);
    } else if (on_gpu - before).abs() > f64::EPSILON {
        if on_gpu >= total - 0.01 {
            d.choice.set(&f.key, json!(0));
        } else {
            d.choice.set(&f.key, json!((on_gpu * GB) as u64));
        }
    }
}

fn generic_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field) {
    let help = with_default(d, f);
    let changed = !is_default(d, f);
    let reset = grid_row(ui, f.label, &help, changed, |ui| {
        ui.horizontal_wrapped(|ui| edit_field(ui, f, &mut d.choice, &mut d.texts));
    });
    if reset {
        d.reset(&f.key);
    }
}

fn option_row(ui: &mut egui::Ui, d: &mut LoadDialog, f: &Field, env: &Env) {
    match f.key.as_str() {
        "serve.max_model_len" => context_row(ui, d, f, env),
        "runtime.kv_dtype" => kv_type_row(ui, d, f),
        "serve.kv_cache_bytes" => kv_size_row(ui, d, f, env),
        "serve.max_num_seqs" => slots_row(ui, d, f),
        "runtime.offload.vram_budget" => offload_row(ui, d, f),
        _ => generic_row(ui, d, f),
    }
}

fn gpu_block(app: &mut App, ui: &mut egui::Ui, env: &Env) {
    grid_row(ui, "gpu selection", "The cards this model may use. The same switches as the Hardware tab.", false, |ui| gpu_rows(app, ui, &env.inv));
}

fn profile_block(ui: &mut egui::Ui, d: &mut LoadDialog) {
    let mut pick = if d.choice.is_custom() { ProfilePick::Custom } else { ProfilePick::Profile(d.choice.profile) };
    let before = pick;
    let mut options: Vec<(ProfilePick, &str, Option<usize>)> = Profile::ALL.iter().map(|p| (ProfilePick::Profile(*p), p.label(), None)).collect();
    if d.choice.is_custom() {
        options.push((ProfilePick::Custom, "custom", None));
    }
    segments(ui, &mut pick, &options);
    if pick != before
        && let ProfilePick::Profile(p) = pick {
            d.choice.select(p);
            d.pin_seqs = false;
            d.pin_kv = false;
            d.solve_key.clear();
        }
    match d.choice.profile {
        Profile::MaxPerformance if !d.choice.is_custom() => hint(ui, "MTP, CUDA graphs, fused kernels, the fast loader and the prefix cache are on."),
        Profile::LowMemory if !d.choice.is_custom() => hint(ui, "fp8 KV cache, 2 parallel requests, smaller batches and prefill chunks, no MTP draft head, no prefix cache, small loader and image budgets."),
        _ => hint(ui, &format!("{} advanced option(s) changed from the profile.", d.choice.overrides.len())),
    }
}

fn section_block(ui: &mut egui::Ui, d: &mut LoadDialog, heading: &'static str, fields: &[Field], env: &Env) {
    let open = d.advanced || d.open_sections.contains(heading);
    let changed = fields.iter().filter(|f| !is_default(d, f)).count();
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui.add(Key::new(if open { "hide" } else { "show" }).outline(VALUE).min_width(64.0).id_salt(("section", heading))).clicked() {
            if open {
                d.open_sections.remove(heading);
            } else {
                d.open_sections.insert(heading);
            }
        }
        ui.label(legend(heading));
        let tally = if changed > 0 { format!("{} options, {changed} changed", fields.len()) } else { format!("{} options", fields.len()) };
        Line::new().note(tally).size(10.5).show(ui);
    });
    if open {
        ui.add_space(2.0);
        for f in fields {
            option_row(ui, d, f, env);
        }
    }
}

fn settings(ui: &mut egui::Ui, app: &mut App, d: &mut LoadDialog, env: &Env) {
    ui.spacing_mut().item_spacing.y = 8.0;
    ui.label(legend("profile"));
    profile_block(ui, d);
    let mut query = std::mem::take(&mut d.search);
    field(ui, &mut query, "search every option");
    d.search = query;
    if !d.search.trim().is_empty() {
        let found = specrt::options::matching(&specrt::options::schema(), &d.search);
        if found.is_empty() {
            note(ui, "No option matches that name.", LEGEND);
        }
        section(ui, "matching options", &format!("{} found", found.len()), |ui| {
            for f in &found {
                option_row(ui, d, f, env);
            }
        });
        return;
    }
    let layout = specrt::options::layout(env.has_mtp);
    section(ui, "common", "the settings most people change", |ui| {
        for f in &layout.common {
            option_row(ui, d, f, env);
            if f.key == "serve.max_num_seqs" {
                gpu_block(app, ui, env);
            }
        }
    });
    section(ui, "api", "extra names API clients can use for this model", |ui| {
        row_help(ui, "served names", "Comma separated aliases, like vLLM's served model name. /v1/models lists the model's own id, requests may use any of these.", |ui| {
            ui.add(egui::TextEdit::singleline(&mut d.served_names).hint_text("qwen3.8-27b").desired_width(220.0));
        });
        switch(ui, "remember for this model", &mut d.remember, "use these settings next time", "Saved with the app settings, per model.");
    });
    ui.label(legend("more options"));
    for (heading, fields) in &layout.sections {
        section_block(ui, d, heading, fields, env);
    }
}

pub fn concurrency_text(full: f64, ctx: usize) -> String {
    if full < 1.0 {
        format!("cannot hold one request of {} tokens", tokens(ctx))
    } else {
        let n = full.floor() as usize;
        format!("serves {} request{} at {} tokens each", n, if n == 1 { "" } else { "s" }, tokens(ctx))
    }
}

fn fit_line(d: &LoadDialog, env: &Env, e: &Estimate, spread: &Option<Result<Planned, String>>) -> Option<(bool, String)> {
    match e.free {
        Some(_) if env.gpus.len() > 1 => match spread {
            Some(Ok(p)) => {
                let largest = env.gpus.iter().copied().max().unwrap_or(0);
                Some((true, if p.spans() && e.total() > largest {
                    format!("{} is more than the largest gpu ({}): it will span {} gpus, split as below", human_size(e.total()), human_size(largest), p.placement.stages.len())
                } else {
                    format!("fits across {} gpus in {} stage(s)", env.gpus.len(), p.placement.stages.len())
                }))
            }
            Some(Err(why)) if e.total() > env.gpus.iter().copied().max().unwrap_or(0) => Some((false, format!("the planner says it does not fit across {} gpus: {why}. Loading is blocked: lower the memory, the context or the parallel requests.", env.gpus.len()))),
            Some(Err(why)) => Some((false, format!("the planner says it does not fit across {} gpus: {why}. You can still try to load it.", env.gpus.len()))),
            None => None,
        },
        Some(f) => Some(match e.fits() {
            Some(false) => (false, format!("the planner says it is {} over free memory. You can still try to load it.", human_size(e.total() - f))),
            _ => (true, "fits in free memory".to_string()),
        }),
        None => {
            let _ = d;
            None
        }
    }
}

fn tile(ui: &mut egui::Ui, width: f32, label: &str, value: String, tint: egui::Color32) {
    let (number, unit) = split_unit(value);
    ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
        ui.set_width(width);
        ui.spacing_mut().item_spacing.y = 1.0;
        ui.add(egui::Label::new(legend(label)).truncate());
        ui.add(egui::Label::new(egui::RichText::new(number).font(figure(19.0)).color(tint)).truncate());
        ui.add(egui::Label::new(egui::RichText::new(unit).size(10.5).color(LEGEND)).truncate());
    });
}

fn result_card(ui: &mut egui::Ui, d: &LoadDialog, env: &Env, spread: &Option<Result<Planned, String>>, compact: bool, details: &mut bool) {
    let est = d.estimate(env.free);
    card(ui, Some(READOUT), |ui| {
        ui.add(egui::Label::new(egui::RichText::new("WHAT YOU WILL GET").font(legend_font(13.0)).extra_letter_spacing(1.5).color(VALUE)).truncate());
        if !compact {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                Line::new().note("updates as you change settings").size(10.5).elided(ui);
            });
        }
    }, |ui| {
        let Some(e) = &est else {
            note(ui, "No estimate: the model's layout could not be read.", LEGEND);
            return;
        };
        let a = d.anatomy().unwrap();
        let fits = fitting_context(a, env.free, d.seqs).map(|n| n.min(d.max));
        let offload = d.choice.value("runtime.offload.vram_budget").as_u64().unwrap_or(0);
        let total_tint = if e.fits() == Some(false) { FAULT } else { VALUE };
        let kv_tint = if d.holds_one_request() { VALUE } else { FAULT };
        let tiles = [
            ("gpu memory", human_size(e.total()), total_tint),
            ("context", format!("{} tokens", tokens(d.ctx)), VALUE),
            ("max that fits", fits.map_or("-".to_string(), |n| format!("{} tokens", tokens(n))), VALUE),
            ("kv cache", human_size(e.kv), kv_tint),
        ];
        let gap = 8.0;
        let across = if ui.available_width() >= 400.0 { 4 } else { 2 };
        let w = (ui.available_width() - gap * (across as f32 - 1.0)) / across as f32;
        for row in tiles.chunks(across) {
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for (label, value, tint) in row {
                    tile(ui, w, label, value.clone(), *tint);
                }
            });
            ui.add_space(4.0);
        }
        if let Some((ok, text)) = fit_line(d, env, e, spread) {
            status(ui, ok, &text);
        } else {
            note(ui, "No GPU reading yet, so the estimate cannot be checked against free memory.", LEGEND);
        }
        if !d.holds_one_request() {
            status(ui, false, &format!("the KV cache cannot hold one request of {} tokens: raise it or lower the context length. You can still try to load it.", tokens(d.ctx)));
        }
        if let Some((ok, text)) = d.kv_note() {
            status(ui, ok, &text);
        }
        if compact {
            let label = if *details { "hide details" } else { "details" };
            if ui.add(Key::new(label).outline(VALUE).id_salt("result-details")).clicked() {
                *details = !*details;
            }
            if !*details {
                return;
            }
        }
        ui.add_space(4.0);
        let kv_text = if d.seqs > 1 { format!("{} tokens shared by {} requests", tokens(pool_tokens(a, e.kv)), d.seqs) } else { format!("{} tokens", tokens(d.ctx)) };
        let r = Line::new().legend("weights").column(ui, 120.0).value(human_size(e.weights)).show(ui);
        r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("weights {}", human_size(e.weights))));
        Line::new().legend("kv cache").column(ui, 120.0).note(kv_text).show(ui);
        Line::new().legend("offload").column(ui, 120.0).value(if offload > 0 { human_size(a.total_bytes.saturating_sub(offload)) } else { "none".to_string() }).show(ui);
        let full = concurrency(a, e.kv, d.ctx);
        let typical = concurrency(a, e.kv, TYPICAL_REQUEST.min(d.ctx));
        let words = if d.ctx > TYPICAL_REQUEST && typical >= 1.0 { format!("{}, or {} at {} tokens each", concurrency_text(full, d.ctx), typical.floor() as usize, tokens(TYPICAL_REQUEST)) } else { concurrency_text(full, d.ctx) };
        Line::new().legend("concurrency").column(ui, 120.0).note(words).wrapped(ui);
        if e.state > 0 {
            let each = if d.seqs > 1 { format!("{} each", human_size(e.state / d.seqs as u64)) } else { String::new() };
            Line::new().legend("recurrent state").column(ui, 120.0).value(human_size(e.state)).gap(8.0).note(each).show(ui);
        }
        if let Some(f) = e.free {
            if env.gpus.len() > 1 {
                let label = format!("free on {} gpus", env.gpus.len());
                let r = Line::new().legend(&label).column(ui, 120.0).measured(human_size(f)).show(ui);
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, label.clone()));
            } else {
                Line::new().legend("free on the gpu").column(ui, 120.0).measured(human_size(f)).show(ui);
            }
        }
        match &d.solved {
            Some(Ok(s)) => {
                let r = Line::new().legend("placement").column(ui, 120.0).value(s.strategy.clone()).size(11.0).show(ui);
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("placement {}", s.strategy)));
                for why in &s.reasons {
                    hint(ui, why);
                }
                if let Some(short) = &s.shortfall {
                    status(ui, false, short);
                }
                placement_bars(ui, &env.inv, s);
            }
            Some(Err(why)) => status(ui, false, &format!("no placement fits: {why}")),
            None => {}
        }
    });
}

fn header(ui: &mut egui::Ui, d: &LoadDialog, back: &mut bool) {
    ui.horizontal(|ui| {
        if ui.add(Key::new("back").outline(VALUE).min_width(80.0).id_salt("load-back")).clicked() {
            *back = true;
        }
        ui.add_space(6.0);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            let room = (ui.available_width() / 12.0).floor().max(12.0) as usize;
            let shown = shorten_middle(&format!("Load {}", d.model.name), room);
            let title = ui.add(egui::Label::new(egui::RichText::new(shown.to_uppercase()).font(legend_font(16.0)).extra_letter_spacing(2.0).color(VALUE)).truncate());
            title.on_hover_text(d.model.name.clone());
            let m = &d.model;
            let facts = [human_size(m.size_bytes), m.architecture.clone(), m.quant.clone()].into_iter().filter(|s| !s.is_empty() && s != "-").collect::<Vec<_>>().join("  ");
            ui.add(egui::Label::new(Line::new().note(facts).size(11.0).job()).truncate());
        });
    });
    if let Err(e) = &d.view {
        status(ui, false, e);
        note(ui, "The engine will read the model itself and report what is wrong.", LEGEND);
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(mut d) = app.load.take() else { return };
    let inv = app.inventory();
    let gpus = inv.free_local();
    let free = (!gpus.is_empty()).then(|| gpus.iter().sum::<u64>());
    d.sync();
    let devices = inv.devices(app.hardware.machine.as_ref(), app.settings.cluster.network_link());
    d.resolve(&devices);
    let spread = d.planned(&devices).filter(|_| devices.len() > 1);
    let has_mtp = d.view.as_ref().is_ok_and(|v| v.spec.speculative().is_some() || v.spec.mtp.is_some());
    let env = Env { inv, free, gpus, devices, has_mtp };
    let est = d.estimate(free);
    let loaded_now = app.loaded_all();
    let alongside_block = (!loaded_now.is_empty()).then(|| app.alongside_blocked().or_else(|| est.as_ref().filter(|e| e.fits() == Some(false)).map(|e| format!("{} more than the free memory: unload a model first", human_size(e.total() - e.free.unwrap_or(0)))))).flatten();
    let (mut go, mut back, mut alongside) = (false, false, false);
    let typing = ui.ctx().egui_wants_keyboard_input();
    if !typing && !egui::Popup::is_any_open(ui.ctx()) {
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            back = true;
        }
        if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            go = true;
        }
    }
    let wide = ui.available_width() >= WIDE;
    let mut details = d.details;
    egui::Panel::top("load-head").frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 8 })).show(ui, |ui| header(ui, &d, &mut back));
    egui::Panel::bottom("load-foot").frame(egui::Frame::NONE.fill(WELL).inner_margin(egui::Margin::symmetric(12, 8))).show(ui, |ui| {
        if let Some(why) = &alongside_block {
            status(ui, false, &format!("cannot load alongside: {why}"));
        }
        footer(ui, |ui| {
            let label = if loaded_now.is_empty() { "load model" } else { "replace current" };
            if ui.add(Key::new(label).solid(READOUT).min_width(120.0).id_salt("load-replace")).clicked() {
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
                back = true;
            }
        });
    });
    if wide {
        egui::Panel::left("load-result").exact_size(SIDE_W).resizable(false).frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin { left: 12, right: 8, top: 4, bottom: 4 })).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| result_card(ui, &d, &env, &spread, false, &mut details));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin { left: 8, right: 12, top: 4, bottom: 4 }).show(ui, |ui| {
                let w = ui.available_width();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_max_width(w);
                    settings(ui, app, &mut d, &env);
                });
            });
        });
    } else {
        egui::CentralPanel::no_frame().show(ui, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin::symmetric(10, 4)).show(ui, |ui| {
                let w = ui.available_width();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_max_width(w);
                    ui.spacing_mut().item_spacing.y = 8.0;
                    result_card(ui, &d, &env, &spread, true, &mut details);
                    settings(ui, app, &mut d, &env);
                });
            });
        });
    }
    d.details = details;
    if go {
        launch(app, &d, alongside, &env);
    } else if !back {
        app.load = Some(d);
    }
}

fn launch(app: &mut App, dialog: &LoadDialog, alongside: bool, env: &Env) {
    let load_devices = &env.devices;
    let inv = &env.inv;
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
                let planned = dialog.planned(load_devices).and_then(Result::ok);
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
        None if load_devices.is_empty() && !inv.cards.is_empty() => Err("not loaded: no gpu selected. Tick a gpu under GPU selection on this page".to_string()),
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
}
