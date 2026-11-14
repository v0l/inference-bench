use crate::App;
use crate::{block, breakdown, engine, strip, table};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::{Format, LocalModel, human_size};
use serde_json::Value;
use specrt::anatomy::{LayerKind, ModelAnatomy};
use specrt::spec::Spec;
use specrt::structure::Structure;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    SpecToml,
    HfConfig,
    GgufMetadata,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::SpecToml => "spec TOML",
            Source::HfConfig => "config.json",
            Source::GgufMetadata => "GGUF metadata",
        }
    }
}

pub struct View {
    pub name: String,
    pub spec: Spec,
    pub anatomy: ModelAnatomy,
    pub source: Source,
    pub structure: Option<Structure>,
    pub unread: Vec<(String, String)>,
}

impl View {
    pub fn new(name: &str, spec: Spec, source: Source, mtp_layers: usize, max_ctx: Option<usize>) -> Self {
        let anatomy = ModelAnatomy::new(&spec, mtp_layers, max_ctx);
        View { name: name.to_string(), spec, anatomy, source, structure: None, unread: Vec::new() }
    }

    pub fn with_structure(mut self, structure: Structure) -> Self {
        self.spec.apply_structure(&structure);
        self.anatomy = ModelAnatomy::new(&self.spec, self.anatomy.mtp_layers, Some(self.anatomy.max_context));
        self.structure = Some(structure);
        self
    }

    pub fn load(m: &LocalModel) -> Result<View, String> {
        match m.format {
            Format::Gguf => {
                let g = specrt::gguf::Gguf::open(&m.path).map_err(|e| format!("{e:#}"))?;
                let mut spec = g.spec().map_err(|e| format!("{e:#}"))?;
                spec.name = m.name.clone();
                Ok(View::new(&m.name, spec, Source::GgufMetadata, 0, m.context_length.map(|c| c as usize)))
            }
            Format::Spec => {
                let mut spec = Spec::load(&m.path).map_err(|e| format!("{e:#}"))?;
                let cfg = m.files.first().and_then(|f| f.parent()).map(|d| d.join("config.json"));
                if let Some(text) = cfg.as_deref().and_then(|p| std::fs::read_to_string(p).ok()) {
                    let _ = spec.complete_from_checkpoint_config(&text);
                }
                let (mtp, ctx) = cfg.as_deref().map(config_extras).unwrap_or((0, None));
                let view = View::new(&m.name, spec, Source::SpecToml, mtp, ctx.or(m.context_length.map(|c| c as usize)));
                Ok(with_header(view, m))
            }
            Format::Safetensors => {
                let path = m.path.join("config.json");
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let mut spec = Spec::describe_hf_config(&text).map_err(|e| format!("{e:#}"))?;
                spec.name = m.name.clone();
                spec.vision = specrt::vision::Vision::from_hf_config(&text).ok().flatten();
                let (mtp, ctx) = config_extras(&path);
                let mut view = View::new(&m.name, spec, Source::HfConfig, mtp, ctx.or(m.context_length.map(|c| c as usize)));
                view.unread = specrt::spec::unread_hf_keys(&text);
                Ok(with_header(view, m))
            }
        }
    }
}

fn with_header(view: View, m: &LocalModel) -> View {
    match Structure::read(&m.files) {
        Ok(structure) => view.with_structure(structure),
        Err(_) => view,
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
    pub diagram: Diagram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Diagram {
    #[default]
    Checkpoint,
    EnginePrefill,
    EngineDecode,
}

impl Diagram {
    fn phase(self) -> Option<specrt::plan::Phase> {
        match self {
            Diagram::Checkpoint => None,
            Diagram::EnginePrefill => Some(specrt::plan::Phase::Prefill),
            Diagram::EngineDecode => Some(specrt::plan::Phase::Decode),
        }
    }
}

impl Detail {
    pub fn open(m: &LocalModel) -> Self {
        let view = View::load(m);
        let kind = match &view {
            Ok(v) if v.anatomy.count(LayerKind::DeltaNet) > 0 => LayerKind::DeltaNet,
            _ => LayerKind::Attention,
        };
        Detail { model: m.name.clone(), view, kind, layer: None, compare: None, other: None, diagram: Diagram::default() }
    }

    pub fn from_view(view: View) -> Self {
        let kind = if view.anatomy.count(LayerKind::DeltaNet) > 0 { LayerKind::DeltaNet } else { LayerKind::Attention };
        Detail { model: view.name.clone(), view: Ok(view), kind, layer: None, compare: None, other: None, diagram: Diagram::default() }
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
        if m.name != detail.model {
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
            let mix = v.anatomy.quant_mix();
            let total: u64 = mix.iter().map(|m| m.1).sum::<u64>().max(1);
            for (q, bytes) in &mix {
                let text = match mix.len() {
                    1 => specrt::anatomy::quant_tag(*q).to_string(),
                    _ => format!("{} {:.0}%", specrt::anatomy::quant_tag(*q), *bytes as f64 / total as f64 * 100.0),
                };
                chip(ui, &text, READOUT, true);
            }
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
        let free = app.gpu.first().map(|g| g.total_bytes.saturating_sub(g.used_bytes));
        body(&mut detail, ui, free);
        app.models.detail = Some(detail);
    }
}

fn body(detail: &mut Detail, ui: &mut egui::Ui, free: Option<u64>) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 8.0;
        content(detail, ui, free);
    });
}

fn content(detail: &mut Detail, ui: &mut egui::Ui, free: Option<u64>) {
    let Detail { view, other, kind, layer, diagram, .. } = detail;
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
    section(ui, "layers", &layers_note(&view.anatomy, view.spec.layers.window), |ui| {
        for (n, v) in views.iter().enumerate() {
            if views.len() > 1 {
                Line::new().legend(&v.name).gap(10.0).value(human_size(v.anatomy.total_bytes)).size(11.0).show(ui);
            }
            strip::draw(ui, n, &v.anatomy, scale, kind, layer);
        }
        strip::legend(ui, &view.anatomy);
    });
    let layer_for_block = layer.unwrap_or_else(|| view.anatomy.layers.iter().find(|l| l.kind == *kind).map_or(0, |l| l.index));
    block_section(ui, view, *kind, *layer, layer_for_block, diagram);
    for v in &views {
        if let Some(vision) = &v.spec.vision {
            section(ui, "vision tower", &tower_note(&vision.config), |ui| {
                if views.len() > 1 {
                    Line::new().legend(&v.name).show(ui);
                }
                tower(ui, &vision.config);
            });
        }
    }
    let anatomies: Vec<&specrt::anatomy::ModelAnatomy> = views.iter().map(|v| &v.anatomy).collect();
    let totals = breakdown::max_totals(anatomies.iter().copied());
    section(ui, "parameters and memory", "bars share one scale", |ui| {
        for (n, v) in views.iter().enumerate() {
            if views.len() > 1 {
                Line::new().legend(&v.name).show(ui);
            }
            breakdown::bars(ui, &v.anatomy, &v.spec, totals, &format!("{n}"));
            ui.add_space(6.0);
        }
    });
    let y_max = breakdown::memory_scale(&anatomies, free);
    section(ui, "memory against context", "weights, recurrent state and kv cache from 1k tokens to the context limit", |ui| {
        for v in &views {
            if views.len() > 1 {
                Line::new().legend(&v.name).show(ui);
            }
            breakdown::memory_summary(ui, &v.anatomy);
            breakdown::memory_chart(ui, &v.anatomy, y_max, free);
        }
    });
    section(ui, "numbers", "every spec field and where it came from", |ui| {
        for v in &views {
            if views.len() > 1 {
                Line::new().legend(&v.name).show(ui);
            }
            table::draw(ui, v);
        }
    });
}

fn tower_note(c: &specrt::vision::VisionConfig) -> String {
    format!(
        "{} blocks, hidden {}, {} heads, patch {} merged {}x{} to {}, {} params",
        c.depth,
        c.hidden_size,
        c.num_heads,
        c.patch_size,
        c.spatial_merge_size,
        c.spatial_merge_size,
        c.out_hidden_size,
        human_count(specrt::vision_encoder::tower_params(c))
    )
}

fn human_count(n: u64) -> String {
    match n {
        1_000_000_000.. => format!("{:.2} B", n as f64 / 1e9),
        1_000_000.. => format!("{:.1} M", n as f64 / 1e6),
        1_000.. => format!("{:.1} K", n as f64 / 1e3),
        _ => n.to_string(),
    }
}

fn tower(ui: &mut egui::Ui, c: &specrt::vision::VisionConfig) {
    let parts = specrt::vision_encoder::tower_parts(c);
    for p in &parts {
        let times = if p.count > 1 { format!("{} x ", p.count) } else { String::new() };
        Line::new().legend(&p.name).gap(10.0).value(format!("{times}{}", human_count(p.params))).show(ui);
    }
    let ops = specrt::vision_encoder::plan(c);
    let labels: Vec<String> = ops.iter().take(9).map(|o| o.label()).collect();
    Line::new().legend("op plan").value(format!("{} ops: {} ...", ops.len(), labels.join(", "))).size(10.5).elided(ui);
    if !c.deepstack_visual_indexes.is_empty() {
        Line::new().legend("deepstack after blocks").value(format!("{:?}", c.deepstack_visual_indexes)).show(ui);
    }
}

fn layers_note(a: &ModelAnatomy, window: Option<usize>) -> String {
    let attention = a.count(LayerKind::Attention);
    let sliding = a.count(LayerKind::SlidingAttention);
    let delta = a.count(LayerKind::DeltaNet);
    let mut parts = Vec::new();
    match (sliding, delta) {
        (0, 0) => return format!("{attention} attention layers"),
        _ => parts.push(format!("{attention} attention")),
    }
    if sliding > 0 {
        parts.push(match window {
            Some(w) => format!("{sliding} sliding attention (window {w})"),
            None => format!("{sliding} sliding attention"),
        });
    }
    if delta > 0 {
        parts.push(format!("{delta} delta-net"));
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_bench_core::library::{hub_dir, scan_hub};
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures")
    }

    fn local(name: &str, dir: &Path) -> LocalModel {
        LocalModel {
            name: name.into(),
            repo: String::new(),
            path: dir.to_path_buf(),
            root: dir.to_path_buf(),
            format: Format::Safetensors,
            size_bytes: 0,
            architecture: String::new(),
            quant: String::new(),
            context_length: None,
            files: vec![],
            support: vec![],
            origin: None,
            vision: false,
        }
    }

    fn fixture_models() -> Vec<LocalModel> {
        let scratch = std::env::temp_dir().join(format!("ib-detail-open-{}", std::process::id()));
        let mut out = Vec::new();
        for entry in std::fs::read_dir(fixtures()).unwrap().flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".config.json") {
                let dir = scratch.join(stem);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::copy(&path, dir.join("config.json")).unwrap();
                out.push(local(stem, &dir));
            } else if path.join("config.json").is_file() {
                out.push(local(&name, &path));
            }
        }
        out
    }

    fn hub_models() -> Vec<LocalModel> {
        let mut hubs = vec![hub_dir()];
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            hubs.push(home.join("spark/huggingface/hub"));
        }
        hubs.iter().flat_map(|h| scan_hub(h)).filter(|m| m.format == Format::Safetensors).collect()
    }

    #[test]
    fn a_vision_checkpoint_shows_its_tower_and_a_text_model_does_not() {
        let dir = std::env::temp_dir().join(format!("ib-detail-vision-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(fixtures().join("qwen3.8-27b.multimodal.json"), dir.join("config.json")).unwrap();
        let view = View::load(&local("qwen3.8-27b", &dir)).unwrap();
        let vision = view.spec.vision.as_ref().expect("vision_config is read");
        assert_eq!(vision.config.depth, 27);
        assert!(tower_note(&vision.config).contains("27 blocks"));
        let text = std::env::temp_dir().join(format!("ib-detail-text-{}", std::process::id()));
        std::fs::create_dir_all(&text).unwrap();
        std::fs::copy(fixtures().join("qwen3-0.6b.config.json"), text.join("config.json")).unwrap();
        assert!(View::load(&local("qwen3-0.6b", &text)).unwrap().spec.vision.is_none());
    }

    #[test]
    fn the_35b_a3b_moe_config_opens_with_its_experts() {
        let models = fixture_models();
        let m = models.iter().find(|m| m.name == "qwen3.6-35b-a3b-fp8").unwrap();
        let detail = Detail::open(m);
        let view = detail.view.unwrap_or_else(|e| panic!("{e}"));
        let moe = view.anatomy.moe.as_ref().unwrap();
        assert_eq!((moe.experts, moe.active, moe.moe_layers), (256, Some(8), 40));
        assert!(view.anatomy.decode_params * 8 < view.anatomy.total_params);
    }

    #[test]
    fn every_fixture_and_local_model_opens_or_says_why_it_cannot_be_drawn() {
        let mut models = fixture_models();
        models.extend(hub_models());
        assert!(models.len() >= 8);
        for m in &models {
            if let Err(e) = Detail::open(m).view {
                assert!(!e.contains("missing field"), "{}: {e}", m.name);
                assert!(e.contains("not supported") || e.contains("no config.json") || e.contains("config.json has no"), "{}: {e}", m.name);
            }
        }
    }

    #[test]
    fn a_config_without_an_intermediate_size_shows_unknown_not_an_error() {
        let json = std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap();
        let mut v: Value = serde_json::from_str(&json).unwrap();
        v.as_object_mut().unwrap().remove("intermediate_size");
        let dir = std::env::temp_dir().join(format!("ib-detail-nointer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        let view = Detail::open(&local("no-inter", &dir)).view.unwrap_or_else(|e| panic!("{e}"));
        let entries = crate::table::entries(&view);
        assert!(entries.iter().any(|e| e.field == "intermediate" && e.value == "unknown"));
    }
}

fn block_section(ui: &mut egui::Ui, view: &View, _kind: LayerKind, layer: Option<usize>, layer_for_block: usize, diagram: &mut Diagram) {
    let Some(structure) = view.structure.as_ref() else {
        section(ui, "block diagram", "no checkpoint headers", |ui| {
            Line::new().note("the safetensors headers are not on disk, so there are no tensors to draw").size(11.0).show(ui);
        });
        return;
    };
    let Some(l) = block::layer_for(structure, layer_for_block) else { return };
    let title = match diagram.phase() {
        None => block::title(layer, &view.spec, l),
        Some(phase) => engine::title(&view.spec, l.index, phase),
    };
    section(ui, "block diagram", &title, |ui| {
        let options = [
            (Diagram::Checkpoint, "checkpoint".to_string()),
            (Diagram::EnginePrefill, "engine, prefill".to_string()),
            (Diagram::EngineDecode, "engine, decode".to_string()),
        ];
        ui.allocate_ui(egui::vec2(240.0, KEY_H), |ui| {
            choice(ui, "diagram-view", diagram, options);
        });
        match diagram.phase() {
            None => block::draw(ui, &block::rows_for(l, &view.spec)),
            Some(phase) => match engine::rows_for(&view.spec, structure, l.index, phase) {
                Ok(rows) => block::draw(ui, &rows),
                Err(reason) => status(ui, false, &reason),
            },
        }
    });
}

#[cfg(test)]
mod note_tests {
    use super::*;

    #[test]
    fn the_layer_note_counts_sliding_layers_with_the_window() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures");
        let spec = Spec::describe_hf_config(&std::fs::read_to_string(dir.join("gemma4-31b.config.json")).unwrap()).unwrap();
        let a = ModelAnatomy::new(&spec, 0, None);
        assert_eq!(layers_note(&a, spec.layers.window), "10 attention, 50 sliding attention (window 1024)");
    }
}
