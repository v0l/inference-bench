use eframe::egui::{self, Color32, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2};
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;
use specrt::anatomy::{Component, LayerInfo, LayerKind, ModelAnatomy, quant_tag};

pub const STRIP_H: f32 = 84.0;
const CAP_W: f32 = 38.0;

pub fn kind_tint(kind: LayerKind) -> Color32 {
    match kind {
        LayerKind::Attention => READOUT,
        LayerKind::DeltaNet => TRACE,
    }
}

pub fn shared_scale<'a>(models: impl Iterator<Item = &'a ModelAnatomy>) -> u64 {
    models.flat_map(|a| a.layers.iter().map(|l| l.params)).max().unwrap_or(1).max(1)
}

pub fn params_text(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2} B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1} M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1} K", n as f64 / 1e3),
        n => n.to_string(),
    }
}

pub fn layer_tip(l: &LayerInfo) -> String {
    let quants = if l.mixer_quant == l.mlp_quant {
        quant_tag(l.mixer_quant).to_string()
    } else {
        format!("{} mixer, {} mlp", quant_tag(l.mixer_quant), quant_tag(l.mlp_quant))
    };
    format!(
        "layer {}: {}\n{} parameters ({} mixer, {} mlp, {} norm)\n{} at {}",
        l.index,
        l.kind.label(),
        params_text(l.params),
        params_text(l.mixer_params),
        params_text(l.mlp_params),
        params_text(l.norm_params),
        human_size(l.bytes),
        quants
    )
}

fn cap_tip(c: &Component, quant: &str) -> String {
    format!("{}\n{} parameters\n{} at {}", c.name, params_text(c.params), human_size(c.bytes), quant)
}

fn height_for(params: u64, scale: u64) -> f32 {
    (params as f32 / scale as f32 * STRIP_H * 0.78).clamp(3.0, STRIP_H)
}

#[allow(clippy::too_many_arguments)]
fn cap(ui: &mut egui::Ui, id: egui::Id, rect: Rect, label: &str, params: u64, scale: u64, tint: Color32, tip: String) {
    let h = height_for(params, scale);
    let bar = Rect::from_min_max(pos2(rect.left(), rect.bottom() - h), rect.right_bottom());
    let resp = ui.interact(rect, id, Sense::hover());
    let p = ui.painter();
    p.rect_filled(bar, 2.0, tint.gamma_multiply(0.55));
    p.rect_stroke(bar, 2.0, Stroke::new(1.0, tint), StrokeKind::Inside);
    let galley = p.layout_no_wrap(label.to_uppercase(), legend_font(9.0), LEGEND);
    p.galley(pos2(rect.center().x - galley.size().x / 2.0, rect.bottom() + 3.0), galley, LEGEND);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("{label} cap")));
    resp.on_hover_text(tip);
}

fn find<'a>(a: &'a ModelAnatomy, name: &str) -> Option<&'a Component> {
    a.components.iter().find(|c| c.name == name)
}

pub fn draw(ui: &mut egui::Ui, salt: usize, a: &ModelAnatomy, scale: u64, kind: &mut LayerKind, picked: &mut Option<usize>) {
    let width = ui.available_width();
    let (outer, _) = ui.allocate_exact_size(vec2(width, STRIP_H + 18.0), Sense::hover());
    let base = egui::Id::new(("strip", salt));
    let area = Rect::from_min_size(outer.min, vec2(width, STRIP_H));
    ui.painter().rect_filled(area.expand2(vec2(4.0, 2.0)), 2.0, WELL);
    let has_mtp = a.mtp_params > 0;
    let caps = 2.0 + f32::from(has_mtp);
    let n = a.layers.len() as f32;
    let gap = if (width - caps * (CAP_W + 6.0)) / n > 7.0 { 1.0 } else { 0.0 };
    let cell_w = ((width - caps * (CAP_W + 6.0)) / n - gap).max(1.0);
    let mut x = area.left();
    if let Some(c) = find(a, "embedding") {
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(CAP_W, STRIP_H));
        cap(ui, base.with("embed"), r, "embed", c.params, scale, LEGEND, cap_tip(c, "bf16"));
    }
    x += CAP_W + 6.0;
    for l in &a.layers {
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(cell_w, STRIP_H));
        layer_cell(ui, base.with(l.index), r, l, scale, kind, picked);
        x += cell_w + gap;
    }
    x += 6.0 - gap;
    if let Some(c) = find(a, "lm_head") {
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(CAP_W, STRIP_H));
        let (params, tint) = if a.head_tied { (a.embedding_params, LEGEND) } else { (c.params, VALUE) };
        let tip = if a.head_tied { format!("lm_head is tied to the embedding\n{} parameters shared", params_text(params)) } else { cap_tip(c, quant_tag(a.head_quant)) };
        cap(ui, base.with("head"), r, if a.head_tied { "tied" } else { "head" }, params, scale, tint, tip);
    }
    x += CAP_W + 6.0;
    if let Some(c) = find(a, "mtp") {
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(CAP_W, STRIP_H));
        let tip = format!("{} layer(s)\n{} parameters\n{}", a.mtp_layers, params_text(c.params), human_size(c.bytes));
        cap(ui, base.with("mtp"), r, "mtp", c.params, scale, OK, tip);
    }
}

fn layer_cell(ui: &mut egui::Ui, id: egui::Id, rect: Rect, l: &LayerInfo, scale: u64, kind: &mut LayerKind, picked: &mut Option<usize>) {
    let resp = ui.interact(rect, id, Sense::click());
    let tint = kind_tint(l.kind);
    let h = height_for(l.params, scale);
    let bar = Rect::from_min_max(pos2(rect.left(), rect.bottom() - h), rect.right_bottom());
    let split = h * l.mlp_params as f32 / l.params.max(1) as f32;
    let mlp = Rect::from_min_max(pos2(bar.left(), bar.bottom() - split), bar.right_bottom());
    let hot = resp.hovered() || *picked == Some(l.index);
    let p = ui.painter();
    p.rect_filled(bar, 1.0, tint);
    p.rect_filled(mlp, 1.0, tint.gamma_multiply(0.45));
    if hot {
        p.rect_stroke(bar.expand(1.0), 1.0, Stroke::new(1.5, VALUE), StrokeKind::Outside);
    } else if l.kind == *kind {
        p.rect_stroke(bar, 1.0, Stroke::new(1.0, VALUE.gamma_multiply(0.35)), StrokeKind::Inside);
    }
    if l.index.is_multiple_of(8) && rect.width() >= 6.0 {
        let g = p.layout_no_wrap(l.index.to_string(), figure(8.5), LEGEND);
        p.galley(pos2(rect.left(), rect.bottom() + 3.0), g, LEGEND);
    }
    if resp.clicked() {
        *kind = l.kind;
        *picked = Some(l.index);
    }
    let name = format!("layer {} {}", l.index, l.kind.label());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, name.clone()));
    resp.on_hover_text(layer_tip(l));
}

pub fn legend(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        for kind in [LayerKind::Attention, LayerKind::DeltaNet] {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter().rect_filled(r, 1.0, kind_tint(kind));
            Line::new().note(kind.label()).size(11.0).show(ui);
            ui.add_space(8.0);
        }
        Line::new().note("darker base is the mlp share, height is parameters, click a layer for its block diagram").size(10.5).show(ui);
    });
}
