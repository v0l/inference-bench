use crate::App;
use crate::{block, breakdown, engine, strip, table};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::hf::{Hub, ModelTree, model_tree_now};
use inference_bench_core::library::{Format, LocalModel, Lineage, Relation, human_size};
use specrt::anatomy::{LayerKind, ModelAnatomy};
use specrt::spec::Spec;
use specrt::structure::Structure;

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
        let d = inference_bench_core::fit::describe(m)?;
        let source = match m.format {
            Format::Gguf => Source::GgufMetadata,
            Format::Spec => Source::SpecToml,
            Format::Safetensors => Source::HfConfig,
        };
        Ok(View { name: m.name.clone(), spec: d.spec, anatomy: d.anatomy, source, structure: d.structure, unread: d.unread })
    }
}

pub struct Detail {
    pub model: String,
    pub view: Result<View, String>,
    pub kind: LayerKind,
    pub layer: Option<usize>,
    pub compare: Option<String>,
    pub other: Option<Result<View, String>>,
    pub diagram: Diagram,
    pub repo: String,
    pub tree: Option<Result<ModelTree, String>>,
    tree_job: crate::job::Pending<Result<ModelTree, String>>,
    tree_started: bool,
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
        Detail { model: m.name.clone(), view, kind, layer: None, compare: None, other: None, diagram: Diagram::default(), repo: m.repo.clone(), tree: None, tree_job: Default::default(), tree_started: false }
    }

    pub fn from_view(view: View) -> Self {
        let kind = if view.anatomy.count(LayerKind::DeltaNet) > 0 { LayerKind::DeltaNet } else { LayerKind::Attention };
        Detail { model: view.name.clone(), view: Ok(view), kind, layer: None, compare: None, other: None, diagram: Diagram::default(), repo: String::new(), tree: None, tree_job: Default::default(), tree_started: false }
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

fn header_chip(ui: &mut egui::Ui, text: &str, tint: egui::Color32, lit: bool) {
    let r = chip(ui, text, tint, lit);
    r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("header chip {text}")));
}

fn header(app: &mut App, detail: &mut Detail, ui: &mut egui::Ui) -> bool {
    let mut back = false;
    let mut load = false;
    ui.horizontal(|ui| {
        if ui.add(Key::new("back").outline(VALUE)).clicked() {
            back = true;
        }
        ui.label(egui::RichText::new(&detail.model).font(legend_font(14.0)).color(VALUE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let runs = app.library.iter().find(|m| m.name == detail.model).is_some_and(crate::models::runs_here);
            if runs && ui.add(Key::new("load").solid(READOUT).min_width(100.0).id_salt("detail-load")).clicked() {
                load = true;
            }
            pick_compare(app, detail, ui);
            ui.label(legend("compare with"));
        });
    });
    if let Ok(v) = &detail.view {
        ui.horizontal_wrapped(|ui| {
            header_chip(ui, v.source.label(), LEGEND, false);
            let mix = v.anatomy.quant_mix();
            let total: u64 = mix.iter().map(|m| m.1).sum::<u64>().max(1);
            for (q, bytes) in &mix {
                let text = match mix.len() {
                    1 => specrt::anatomy::quant_tag(*q).to_string(),
                    _ => format!("{} {:.0}%", specrt::anatomy::quant_tag(*q), *bytes as f64 / total as f64 * 100.0),
                };
                header_chip(ui, &text, READOUT, true);
            }
        });
    }
    if load && let Some(m) = app.library.iter().find(|m| m.name == detail.model).cloned() {
        app.open_load_dialog(&m);
    }
    back
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(mut detail) = app.models.detail.take() else { return };
    let back = header(app, &mut detail, ui);
    if !back {
        pump_tree(app, &mut detail);
        let free = app.free_gpu_bytes();
        let mut act = TreeAction::default();
        let running = app.engine.loaded().filter(|l| l.display == detail.model);
        body(&mut detail, &app.library, ui, free, &mut act, running.as_ref());
        if let Some(i) = act.open.filter(|i| *i < app.library.len()) {
            app.models.detail = Some(Detail::open(&app.library[i]));
            return;
        }
        if let Some(name) = act.find {
            app.models.query = name;
            app.models.relation = None;
            app.models.pane = crate::models::Pane::Discover;
            app.models.detail = None;
            return;
        }
        if let Some((relation, of)) = act.discover {
            let token = app.settings.hf_token.clone();
            let wake = app.wake.clone();
            app.models.discover_relation(&wake, &token, relation, &of);
            app.models.detail = None;
            return;
        }
        app.models.detail = Some(detail);
    }
}

fn pump_tree(app: &mut App, detail: &mut Detail) {
    if let Some(r) = detail.tree_job.poll() {
        detail.tree = Some(r);
    }
    if detail.tree_started || detail.repo.is_empty() {
        return;
    }
    detail.tree_started = true;
    let known: Vec<(String, Lineage)> = app.library.iter().filter_map(|m| m.lineage.clone().map(|l| (m.repo.clone(), l))).collect();
    let (base, token, repo) = (app.models.hub_base.clone(), app.settings.hf_token.clone(), detail.repo.clone());
    let cache = app.dir.join("model-tree-cache.json");
    detail.tree_job.start(crate::job::spawn(&app.wake, move || {
        let local = |r: &str| known.iter().find(|(k, _)| k == r).map(|(_, l)| l.clone());
        model_tree_now(&Hub::new(&base, &token), &repo, &local, &cache)
    }));
}

#[derive(Default)]
struct TreeAction {
    discover: Option<(Relation, String)>,
    find: Option<String>,
    open: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct LineageRow {
    pub name: String,
    pub depth: usize,
    pub relation: Option<String>,
    pub local: Option<usize>,
    pub current: bool,
}

pub fn lineage_rows(chain: &[String], relations: &[String], library: &[LocalModel]) -> Vec<LineageRow> {
    let held = |repo: &str| library.iter().position(|m| !repo.is_empty() && m.repo == repo);
    let mut rows = Vec::new();
    let n = chain.len();
    for k in 0..n {
        let at = n - 1 - k;
        let name = &chain[at];
        rows.push(LineageRow {
            name: name.clone(),
            depth: k,
            relation: (k > 0).then(|| relations.get(at).cloned().unwrap_or_default()),
            local: held(name),
            current: at == 0,
        });
        let next = at.checked_sub(1).map(|a| chain[a].as_str());
        for (i, m) in library.iter().enumerate() {
            let Some(l) = &m.lineage else { continue };
            if l.base() != name || m.repo.is_empty() || m.repo == *name || Some(m.repo.as_str()) == next || chain.contains(&m.repo) {
                continue;
            }
            rows.push(LineageRow { name: m.repo.clone(), depth: k + 1, relation: Some(l.relation.label().to_string()), local: Some(i), current: false });
        }
    }
    rows
}

fn tree_section(detail: &Detail, library: &[LocalModel], ui: &mut egui::Ui, act: &mut TreeAction) {
    if detail.repo.is_empty() {
        return;
    }
    section(ui, "model tree", "where this model comes from, and what else you downloaded from its base", |ui| {
        match &detail.tree {
            None => crate::working::working(ui, "looking up the base model"),
            Some(Err(e)) => status(ui, false, e),
            Some(Ok(t)) => {
                for (n, row) in lineage_rows(&t.chain, &t.relations, library).iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add_space(row.depth as f32 * 18.0);
                        if let Some(relation) = &row.relation {
                            chip(ui, relation, READOUT_DIM, false);
                        }
                        let dim = row.local.is_none() && !row.current;
                        let label = ui.add(egui::Label::new(crate::models::name_job(&row.name, 12.5, dim)).sense(egui::Sense::click()));
                        if let (Some(i), false) = (row.local, row.current) {
                            let text = format!("open {}", row.name);
                            label.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, text.clone()));
                            if label.clicked() {
                                act.open = Some(i);
                            }
                        }
                        if row.current {
                            let r = chip(ui, "this model", READOUT, true);
                            r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, "this model"));
                        } else if dim {
                            Line::new().note("not downloaded").size(10.5).show(ui);
                            if ui.add(Key::new("find on hugging face").outline(READOUT).id_salt(("find-base", n))).clicked() {
                                act.find = Some(row.name.clone());
                            }
                        }
                    });
                }
                if t.chain.len() == 1 {
                    note(ui, "no base model is listed for this repo", LEGEND);
                }
                let top = t.chain[t.chain.len() - 1].clone();
                ui.horizontal_wrapped(|ui| {
                    for (label, n) in &t.counts {
                        let Some(relation) = Relation::parse(label) else { continue };
                        let text = format!("{} {n}", relation.plural());
                        if ui.add(Key::new(text).outline(READOUT).id_salt(("tree-count", label))).clicked() {
                            act.discover = Some((relation, top.clone()));
                        }
                    }
                });
            }
        }
    });
}

fn body(detail: &mut Detail, library: &[LocalModel], ui: &mut egui::Ui, free: Option<u64>, act: &mut TreeAction, running: Option<&inference_bench_core::engine::Loaded>) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 8.0;
        if let Some(l) = running {
            crate::running::draw(ui, l);
        }
        tree_section(detail, library, ui, act);
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
    section(ui, "summary", "what this model is and what loading it costs", |ui| {
        for v in &views {
            summary(ui, v, free, views.len() > 1);
        }
    });
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

fn summary(ui: &mut egui::Ui, v: &View, free: Option<u64>, named: bool) {
    let a = &v.anatomy;
    if named {
        Line::new().legend(&v.name).show(ui);
    }
    let quants: Vec<&str> = a.quant_mix().iter().map(|(q, _)| specrt::anatomy::quant_tag(*q)).collect();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 14.0;
        Line::new().legend("parameters").value(crate::strip::params_text(a.total_params)).show(ui);
        Line::new().legend("active per token").value(crate::strip::params_text(a.decode_params)).show(ui);
        Line::new().legend("quant").value(quants.join(", ")).show(ui);
        Line::new().legend("max context").value(format!("{} tokens", a.max_context)).show(ui);
        Line::new().legend("vision").value(if v.spec.vision.is_some() { "yes" } else { "no" }).show(ui);
        Line::new().legend("mtp").value(if a.mtp_layers > 0 { format!("{} layer", a.mtp_layers) } else { "no".into() }).show(ui);
    });
    let at_default = crate::load::estimate(a, a.kv_bytes_per_token * crate::load::max_context(a).min(8192) as u64, 1, free);
    let mut line = Line::new().legend("weights").measured(human_size(a.total_bytes)).gap(14.0).legend("kv per 1k tokens").measured(human_size(a.kv_bytes_per_token * 1024));
    if a.state_bytes_per_seq > 0 {
        line = line.gap(14.0).legend("state").measured(human_size(a.state_bytes_per_seq));
    }
    line.gap(14.0).legend("loads at 8k in").measured(human_size(at_default.total())).show(ui);
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
        crate::fmt::count(specrt::vision_encoder::tower_params(c))
    )
}

fn tower(ui: &mut egui::Ui, c: &specrt::vision::VisionConfig) {
    let parts = specrt::vision_encoder::tower_parts(c);
    for p in &parts {
        let times = if p.count > 1 { format!("{} x ", p.count) } else { String::new() };
        Line::new().legend(&p.name).gap(10.0).value(format!("{times}{}", crate::fmt::count(p.params))).show(ui);
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
    use serde_json::Value;
    use std::path::{Path, PathBuf};

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
            spec_source: None,
            vision: false,
            lineage: None,
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
    use std::path::Path;

    #[test]
    fn the_layer_note_counts_sliding_layers_with_the_window() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures");
        let spec = Spec::describe_hf_config(&std::fs::read_to_string(dir.join("gemma4-31b.config.json")).unwrap()).unwrap();
        let a = ModelAnatomy::new(&spec, 0, None);
        assert_eq!(layers_note(&a, spec.layers.window), "10 attention, 50 sliding attention (window 1024)");
    }
}
