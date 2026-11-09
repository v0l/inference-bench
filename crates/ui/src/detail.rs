use crate::App;
use crate::{block, strip};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{Format, LocalModel, human_size};
use serde_json::Value;
use specrt::anatomy::{LayerKind, ModelAnatomy};
use specrt::spec::Spec;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    SpecToml,
    HfConfig,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::SpecToml => "spec TOML",
            Source::HfConfig => "config.json",
        }
    }
}

pub struct View {
    pub name: String,
    pub spec: Spec,
    pub anatomy: ModelAnatomy,
    pub source: Source,
}

impl View {
    pub fn new(name: &str, spec: Spec, source: Source, mtp_layers: usize, max_ctx: Option<usize>) -> Self {
        let anatomy = ModelAnatomy::new(&spec, mtp_layers, max_ctx);
        View { name: name.to_string(), spec, anatomy, source }
    }

    pub fn load(m: &LocalModel) -> Result<View, String> {
        match m.format {
            Format::Gguf => Err("GGUF files carry no layer spec, so there is nothing to draw".into()),
            Format::Spec => {
                let spec = Spec::load(&m.path).map_err(|e| format!("{e:#}"))?;
                let cfg = m.files.first().and_then(|f| f.parent()).map(|d| d.join("config.json"));
                let (mtp, ctx) = cfg.as_deref().map(config_extras).unwrap_or((0, None));
                Ok(View::new(&m.name, spec, Source::SpecToml, mtp, ctx.or(m.context_length.map(|c| c as usize))))
            }
            Format::Safetensors => {
                let path = m.path.join("config.json");
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let mut spec = Spec::from_hf_config(&text).map_err(|e| format!("{e:#}"))?;
                spec.name = m.name.clone();
                let (mtp, ctx) = config_extras(&path);
                Ok(View::new(&m.name, spec, Source::HfConfig, mtp, ctx.or(m.context_length.map(|c| c as usize))))
            }
        }
    }
}

fn config_extras(path: &Path) -> (usize, Option<usize>) {
    let Some(v) = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else {
        return (0, None);
    };
    let text = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    let mtp = text["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    (mtp, text["max_position_embeddings"].as_u64().map(|c| c as usize))
}

pub struct Detail {
    pub model: String,
    pub view: Result<View, String>,
    pub kind: LayerKind,
    pub layer: Option<usize>,
    pub compare: Option<String>,
    pub other: Option<Result<View, String>>,
}

impl Detail {
    pub fn open(m: &LocalModel) -> Self {
        let view = View::load(m);
        let kind = match &view {
            Ok(v) if v.anatomy.count(LayerKind::DeltaNet) > 0 => LayerKind::DeltaNet,
            _ => LayerKind::Attention,
        };
        Detail { model: m.name.clone(), view, kind, layer: None, compare: None, other: None }
    }

    pub fn from_view(view: View) -> Self {
        let kind = if view.anatomy.count(LayerKind::DeltaNet) > 0 { LayerKind::DeltaNet } else { LayerKind::Attention };
        Detail { model: view.name.clone(), view: Ok(view), kind, layer: None, compare: None, other: None }
    }

    pub fn compare_with(&mut self, view: View) {
        self.compare = Some(view.name.clone());
        self.other = Some(Ok(view));
    }
}

fn pick_compare(app: &App, detail: &mut Detail, ui: &mut egui::Ui) {
    let mut chosen = detail.compare.clone().unwrap_or_default();
    let mut options = vec![(String::new(), "no comparison".to_string())];
    for m in &app.library {
        if m.format != Format::Gguf && m.name != detail.model {
            options.push((m.name.clone(), m.name.clone()));
        }
    }
    ui.allocate_ui(egui::vec2(320.0, KEY_H), |ui| {
        choice(ui, "compare-with", &mut chosen, options);
    });
    if chosen != detail.compare.clone().unwrap_or_default() {
        if chosen.is_empty() {
            detail.compare = None;
            detail.other = None;
        } else if let Some(m) = app.library.iter().find(|m| m.name == chosen) {
            detail.compare = Some(chosen);
            detail.other = Some(View::load(m));
        }
    }
}

fn header(app: &App, detail: &mut Detail, ui: &mut egui::Ui) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if ui.add(Key::new("back").outline(VALUE)).clicked() {
            back = true;
        }
        ui.label(egui::RichText::new(&detail.model).font(legend_font(14.0)).color(VALUE));
        if let Ok(v) = &detail.view {
            chip(ui, v.source.label(), LEGEND, false);
            chip(ui, specrt::anatomy::quant_tag(v.spec.quant), READOUT, true);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            pick_compare(app, detail, ui);
            ui.label(legend("compare with"));
        });
    });
    back
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(mut detail) = app.models.detail.take() else { return };
    let back = header(app, &mut detail, ui);
    if !back {
        body(&mut detail, ui);
        app.models.detail = Some(detail);
    }
}

fn body(detail: &mut Detail, ui: &mut egui::Ui) {
    let Detail { view, other, kind, layer, .. } = detail;
    let view = match view {
        Ok(v) => v,
        Err(e) => {
            status(ui, false, e);
            return;
        }
    };
    let other = match other {
        Some(Ok(v)) => Some(&*v),
        Some(Err(e)) => {
            status(ui, false, e);
            None
        }
        None => None,
    };
    let views: Vec<&View> = std::iter::once(&*view).chain(other).collect();
    let scale = strip::shared_scale(views.iter().map(|v| &v.anatomy));
    section(ui, "layers", &layers_note(&view.anatomy), |ui| {
        for (n, v) in views.iter().enumerate() {
            if views.len() > 1 {
                Line::new().legend(&v.name).gap(10.0).value(human_size(v.anatomy.total_bytes)).size(11.0).show(ui);
            }
            strip::draw(ui, n, &v.anatomy, scale, kind, layer);
        }
        strip::legend(ui);
    });
    let layer_for_block = layer.unwrap_or_else(|| view.anatomy.layers.iter().find(|l| l.kind == *kind).map_or(0, |l| l.index));
    section(ui, "block diagram", &block::title(*kind, *layer, &view.spec), |ui| {
        block::draw(ui, &view.spec, *kind, layer_for_block);
    });
}

fn layers_note(a: &ModelAnatomy) -> String {
    let attention = a.count(LayerKind::Attention);
    let delta = a.count(LayerKind::DeltaNet);
    match delta {
        0 => format!("{attention} attention layers"),
        _ => format!("{attention} attention, {delta} delta-net"),
    }
}
