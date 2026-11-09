use eframe::egui::{self, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use egui_bench::prelude::*;
use specrt::spec::{Activation, RopeScalingKind, Spec};
use specrt::structure::{Group, Layer, Structure};

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub title: String,
    pub shape: String,
    pub dtype: Option<String>,
    pub tensors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    Io(Node),
    Boxes(Vec<Node>),
    Add,
}

fn computed(title: &str, shape: impl Into<String>) -> Node {
    Node { title: title.into(), shape: shape.into(), dtype: None, tensors: Vec::new() }
}

fn dims_text(shape: &[usize]) -> String {
    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" x ")
}

fn stored(g: &Group) -> Node {
    let weight = g.weight();
    let mut shape = weight.map(|w| dims_text(&w.shape)).unwrap_or_default();
    let extras: Vec<&str> = g.tensors.iter().filter(|t| Some(*t) != weight).map(|t| t.leaf.as_str()).collect();
    if !extras.is_empty() {
        shape = format!("{shape}, + {}", extras.join(", "));
    }
    let title = match g.instances {
        1 => g.path.clone(),
        n => format!("{n} x {}", g.path),
    };
    Node { title, shape, dtype: weight.map(|w| w.dtype.to_lowercase()), tensors: g.tensor_names() }
}

fn dtype_tint(dtype: &str) -> Color32 {
    match dtype {
        "bf16" | "f16" | "f32" => LEGEND,
        "u8" | "i32" | "u32" | "i8" => OK,
        _ => READOUT,
    }
}

fn norm_weight_width(g: &Group) -> String {
    g.weight().map(|w| dims_text(&w.shape)).unwrap_or_default()
}

fn is_output(g: &Group) -> bool {
    let n = g.name();
    n == "o_proj" || n.contains("out") || n == "wo" || n == "c_proj"
}

fn is_down(g: &Group) -> bool {
    let n = g.name();
    n.contains("down") || n == "w2"
}

fn mlp_module(layer: &Layer) -> Option<&str> {
    layer.modules().into_iter().find(|m| layer.in_module(m).any(|g| is_down(g) || g.path.contains("experts")))
}

fn mixer_modules<'a>(layer: &'a Layer, mlp: Option<&str>) -> Vec<&'a str> {
    layer.modules().into_iter().filter(|m| Some(*m) != mlp && layer.in_module(m).next().is_some()).collect()
}

fn rope_text(spec: &Spec, head_dim: usize) -> String {
    let scaling = match &spec.rope.scaling {
        Some(s) => format!(", {} x {}", match s.kind { RopeScalingKind::Linear => "linear", RopeScalingKind::Llama3 => "llama3" }, s.factor),
        None => String::new(),
    };
    format!("theta {}, {} of {head_dim} dims{scaling}", theta_text(spec.rope.theta), spec.rotary_dim().min(head_dim))
}

fn mixer_rows(layer: &Layer, module: &str, spec: &Spec) -> Vec<Row> {
    let groups: Vec<&Group> = layer.in_module(module).collect();
    let (outputs, inputs): (Vec<&Group>, Vec<&Group>) = groups.iter().copied().partition(|g| is_output(g) && g.is_matrix());
    let projections: Vec<Node> = inputs.iter().filter(|g| g.is_matrix() && !g.is_norm()).map(|g| stored(g)).collect();
    let small: Vec<Node> = inputs.iter().filter(|g| !(g.is_matrix() && !g.is_norm())).map(|g| stored(g)).collect();
    let mut rows = Vec::new();
    if !projections.is_empty() {
        rows.push(Row::Boxes(projections));
    }
    if !small.is_empty() {
        rows.push(Row::Boxes(small));
    }
    let has = |name: &str| groups.iter().any(|g| g.name() == name);
    let a = &spec.attention;
    if let (true, Some(q)) = (has("q_proj") || has("q_a_proj"), groups.iter().find(|g| g.name() == "q_norm")) {
        let head_dim = q.weight().and_then(|w| w.shape.first().copied()).unwrap_or(a.head_dim);
        rows.push(Row::Boxes(vec![computed("rope", rope_text(spec, head_dim))]));
    } else if has("q_proj") {
        rows.push(Row::Boxes(vec![computed("rope", rope_text(spec, a.head_dim))]));
    }
    if has("q_proj") {
        let group = a.heads / a.kv_heads.max(1);
        let gating = if a.output_gate { ", sigmoid output gate" } else { "" };
        rows.push(Row::Boxes(vec![computed(
            "attention",
            format!("{} heads, {} kv, {group} per group{gating}", a.heads, a.kv_heads),
        )]));
    } else if let Some(la) = spec.linear_attention.as_ref().filter(|_| has("conv1d")) {
        rows.push(Row::Boxes(vec![computed(
            "gated delta rule",
            format!("{} key, {} value heads, state {} x {} per head", la.key_heads, la.value_heads, la.key_head_dim, la.value_head_dim),
        )]));
    }
    if !outputs.is_empty() {
        rows.push(Row::Boxes(outputs.iter().map(|g| stored(g)).collect()));
    }
    rows
}

fn expert_node(layer: &Layer, module: &str, spec: &Spec) -> Option<Node> {
    let parts: Vec<&Group> = layer.in_module(module).filter(|g| g.path.contains("experts") && g.is_matrix()).collect();
    let first = parts.first()?;
    let count = first.instances;
    let each: Vec<String> = parts.iter().map(|g| format!("{} {}", g.name(), g.weight().map(|w| dims_text(&w.shape)).unwrap_or_default())).collect();
    let top = spec.moe.as_ref().and_then(|m| m.active).map_or(String::new(), |k| format!(", top {k}"));
    Some(Node {
        title: format!("{count} x experts{top}"),
        shape: each.join(", "),
        dtype: first.weight().map(|w| w.dtype.to_lowercase()),
        tensors: parts.iter().flat_map(|g| g.tensor_names()).collect(),
    })
}

fn mlp_rows(layer: &Layer, module: &str, spec: &Spec) -> Vec<Row> {
    let groups: Vec<&Group> = layer.in_module(module).collect();
    let routed = groups.iter().any(|g| g.path.contains("experts") && !g.path.contains("shared"));
    let act = match spec.mlp.activation {
        Activation::Swiglu => "swiglu",
        Activation::Geglu => "geglu",
    };
    if routed {
        let shared: Vec<&Group> = groups.iter().copied().filter(|g| g.path.contains("shared")).collect();
        let router: Vec<Node> = groups.iter().filter(|g| !g.path.contains("experts") && !g.path.contains("shared") && g.is_matrix()).map(|g| stored(g)).collect();
        let mut rows = Vec::new();
        if !router.is_empty() {
            rows.push(Row::Boxes(router));
        }
        let mut cols: Vec<Node> = expert_node(layer, module, spec).into_iter().collect();
        if !shared.is_empty() {
            let width: Vec<String> = shared.iter().filter(|g| g.is_matrix()).map(|g| format!("{} {}", g.name(), g.weight().map(|w| dims_text(&w.shape)).unwrap_or_default())).collect();
            cols.push(Node {
                title: "shared expert".into(),
                shape: width.join(", "),
                dtype: shared.iter().find(|g| g.is_matrix()).and_then(|g| g.weight()).map(|w| w.dtype.to_lowercase()),
                tensors: shared.iter().flat_map(|g| g.tensor_names()).collect(),
            });
        }
        rows.push(Row::Boxes(cols));
        let gates: Vec<Node> = groups.iter().filter(|g| g.path.contains("shared") && !g.is_matrix()).map(|g| stored(g)).collect();
        if !gates.is_empty() {
            rows.push(Row::Boxes(gates));
        }
        rows.push(Row::Boxes(vec![computed("weighted sum", "routed outputs by router weight, plus shared")]));
        return rows;
    }
    let (down, up): (Vec<&Group>, Vec<&Group>) = groups.iter().copied().partition(|g| is_down(g));
    let mut rows = vec![Row::Boxes(up.iter().map(|g| stored(g)).collect())];
    rows.push(Row::Boxes(vec![computed(act, "")]));
    rows.push(Row::Boxes(down.iter().map(|g| stored(g)).collect()));
    rows
}

pub fn rows_for(layer: &Layer, spec: &Spec) -> Vec<Row> {
    let hidden = layer
        .groups
        .iter()
        .find(|g| g.is_norm() && !g.path.contains('.'))
        .map(norm_weight_width)
        .unwrap_or_else(|| spec.dims.hidden.to_string());
    let top: Vec<&Group> = layer.groups.iter().filter(|g| !g.path.contains('.')).collect();
    let before_mixer: Vec<Node> = top.iter().filter(|g| g.is_norm() && g.name().contains("input")).map(|g| stored(g)).collect();
    let before_mlp: Vec<Node> = top.iter().filter(|g| g.is_norm() && !g.name().contains("input") && !g.name().contains("post_feedforward")).map(|g| stored(g)).collect();
    let after_mlp: Vec<Node> = top.iter().filter(|g| g.name().contains("post_feedforward") || !g.is_norm()).map(|g| stored(g)).collect();
    let mlp = mlp_module(layer);
    let mut rows = vec![Row::Io(computed("input", hidden.clone()))];
    if !before_mixer.is_empty() {
        rows.push(Row::Boxes(before_mixer));
    }
    for m in mixer_modules(layer, mlp) {
        rows.extend(mixer_rows(layer, m, spec));
    }
    rows.push(Row::Add);
    if !before_mlp.is_empty() {
        rows.push(Row::Boxes(before_mlp));
    }
    if let Some(m) = mlp {
        rows.extend(mlp_rows(layer, m, spec));
    }
    if !after_mlp.is_empty() {
        rows.push(Row::Boxes(after_mlp));
    }
    rows.push(Row::Add);
    rows.push(Row::Io(computed("output", hidden)));
    rows
}

pub fn layer_for(structure: &Structure, index: usize) -> Option<&Layer> {
    let stack = structure.main()?;
    stack.layers.iter().min_by_key(|l| l.index.abs_diff(index))
}

const BOX_H: f32 = 38.0;
const GAP: f32 = 14.0;
const ADD_R: f32 = 8.0;

fn row_height(row: &Row) -> f32 {
    match row {
        Row::Add => ADD_R * 2.0,
        _ => BOX_H,
    }
}

pub fn total_height(rows: &[Row]) -> f32 {
    rows.iter().map(row_height).sum::<f32>() + GAP * rows.len().saturating_sub(1) as f32 + 4.0
}

pub fn theta_text(theta: f64) -> String {
    match theta {
        t if t >= 1e6 && (t / 1e6).fract() == 0.0 => format!("{}M", t / 1e6),
        t if t >= 1e3 && (t / 1e3).fract() == 0.0 => format!("{}k", t / 1e3),
        t => format!("{t}"),
    }
}

pub fn shape_job(text: &str, width: f32) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::simple(text.to_string(), figure(10.0), VALUE, width);
    for s in &mut job.sections {
        s.format.strikethrough = Stroke::NONE;
        s.format.underline = Stroke::NONE;
    }
    job
}

fn paint_node(ui: &egui::Ui, rect: Rect, n: &Node, tint: Color32) {
    let p = ui.painter();
    p.rect_filled(rect, 2.0, PANEL);
    p.rect_stroke(rect, 2.0, Stroke::new(1.0, tint.gamma_multiply(0.7)), StrokeKind::Inside);
    let title = p.layout_no_wrap(n.title.to_uppercase(), legend_font(10.5), tint);
    p.galley(pos2(rect.left() + 6.0, rect.top() + 4.0), title, tint);
    let shape = p.layout_job(shape_job(&n.shape, (rect.width() - 12.0).max(10.0)));
    p.galley(pos2(rect.left() + 6.0, rect.top() + 19.0), shape, VALUE);
    if let Some(d) = &n.dtype {
        let tint = dtype_tint(d);
        let tag = p.layout_no_wrap(d.to_uppercase(), legend_font(8.5), tint);
        let tr = Rect::from_min_size(pos2(rect.right() - tag.size().x - 10.0, rect.top() + 3.0), tag.size() + vec2(6.0, 2.0));
        p.rect_filled(tr, 2.0, tint.gamma_multiply(0.18));
        p.galley(tr.min + vec2(3.0, 1.0), tag, tint);
    }
    if !n.tensors.is_empty() {
        ui.interact(rect, ui.id().with(&n.title).with(rect.min.x as i32), Sense::hover()).on_hover_text(n.tensors.join("\n"));
    }
}

pub fn draw(ui: &mut egui::Ui, rows: &[Row]) {
    let rows = rows;
    let width = ui.available_width().min(720.0);
    let (area, _) = ui.allocate_exact_size(vec2(width, total_height(rows)), Sense::hover());
    let skip_x = area.left() + 8.0;
    let left = area.left() + 24.0;
    let inner = width - 24.0;
    let mut y = area.top();
    let mut adds: Vec<f32> = Vec::new();
    let mut first_input = None;
    let mut prev_bottom: Option<f32> = None;
    for (n, row) in rows.iter().enumerate() {
        let h = row_height(row);
        let mid = y + h / 2.0;
        if let Some(b) = prev_bottom {
            let x = left + inner / 2.0;
            let p = ui.painter();
            p.line_segment([pos2(x, b), pos2(x, y)], Stroke::new(1.0, ETCH));
            p.line_segment([pos2(x - 3.0, y - 4.0), pos2(x, y)], Stroke::new(1.0, ETCH));
            p.line_segment([pos2(x + 3.0, y - 4.0), pos2(x, y)], Stroke::new(1.0, ETCH));
        }
        match row {
            Row::Io(nd) => {
                let r = Rect::from_min_size(pos2(left + inner * 0.3, y), vec2(inner * 0.4, h));
                paint_node(ui, r, nd, VALUE);
                if n == 0 {
                    first_input = Some(r.bottom());
                }
            }
            Row::Boxes(nodes) => {
                let count = nodes.len() as f32;
                let bw = (inner - 8.0 * (count - 1.0)) / count;
                let bw = bw.min(inner * 0.8);
                let total = bw * count + 8.0 * (count - 1.0);
                let start = left + (inner - total) / 2.0;
                for (i, nd) in nodes.iter().enumerate() {
                    let r = Rect::from_min_size(pos2(start + i as f32 * (bw + 8.0), y), vec2(bw, h));
                    let tint = if nd.dtype.is_some() { VALUE } else { TRACE };
                    paint_node(ui, r, nd, tint);
                }
            }
            Row::Add => {
                let c = pos2(left + inner / 2.0, mid);
                let p = ui.painter();
                p.circle_filled(c, ADD_R, WELL);
                p.circle_stroke(c, ADD_R, Stroke::new(1.0, READOUT));
                p.line_segment([c - vec2(4.0, 0.0), c + vec2(4.0, 0.0)], Stroke::new(1.0, READOUT));
                p.line_segment([c - vec2(0.0, 4.0), c + vec2(0.0, 4.0)], Stroke::new(1.0, READOUT));
                adds.push(mid);
            }
        }
        prev_bottom = Some(y + h);
        y += h + GAP;
    }
    let p = ui.painter();
    let skip = Stroke::new(1.0, READOUT.gamma_multiply(0.5));
    let mut from = first_input.unwrap_or(area.top());
    for add in adds {
        p.line_segment([pos2(left + inner * 0.3, from), pos2(skip_x, from)], skip);
        p.line_segment([pos2(skip_x, from), pos2(skip_x, add)], skip);
        p.line_segment([pos2(skip_x, add), pos2(left + inner / 2.0 - ADD_R, add)], skip);
        from = add;
    }
}

pub fn title(layer: Option<usize>, spec: &Spec, structure: &Layer) -> String {
    match layer {
        Some(i) => format!("layer {i}"),
        None => format!("layer {} ({})", structure.index, spec.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixtures() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures")
    }

    fn drawn(config: &str, header: &str, layer: usize) -> Vec<String> {
        let json = std::fs::read_to_string(fixtures().join(config)).unwrap();
        let mut spec = Spec::describe_hf_config(&json).unwrap();
        let structure = Structure::from_json(&std::fs::read_to_string(fixtures().join("headers").join(header)).unwrap()).unwrap();
        spec.apply_structure(&structure);
        let rows = rows_for(layer_for(&structure, layer).unwrap(), &spec);
        rows.iter()
            .flat_map(|r| match r {
                Row::Io(n) => vec![format!("{} {}", n.title, n.shape)],
                Row::Boxes(ns) => ns.iter().map(|n| format!("{} {}", n.title, n.shape)).collect(),
                Row::Add => vec!["add".into()],
            })
            .collect()
    }

    #[test]
    fn the_27b_attention_layer_is_drawn_from_its_tensors() {
        let t = drawn("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json", 3);
        assert!(t.iter().any(|s| s.starts_with("self_attn.q_proj 12288 x 5120")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("self_attn.k_proj 1024 x 5120")));
        assert!(t.iter().any(|s| s.starts_with("self_attn.q_norm 256")));
        assert!(t.iter().any(|s| s.starts_with("attention 24 heads, 4 kv, 6 per group, sigmoid output gate")));
        assert!(t.iter().any(|s| s.starts_with("rope theta 10M, 64 of 256 dims")), "{t:?}");
        assert_eq!(t.iter().filter(|s| *s == "add").count(), 2);
    }

    #[test]
    fn the_delta_net_layer_shows_conv_and_projections_that_exist_in_the_checkpoint() {
        let t = drawn("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json", 0);
        assert!(t.iter().any(|s| s.starts_with("linear_attn.conv1d 10240 x 1 x 4")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("linear_attn.in_proj_qkv 10240 x 5120")));
        assert!(t.iter().any(|s| s.starts_with("gated delta rule 16 key, 48 value heads")));
        assert!(t.iter().any(|s| s.starts_with("linear_attn.out_proj 5120 x 6144")));
        assert!(!t.iter().any(|s| s.starts_with("rope")));
    }

    #[test]
    fn a_moe_layer_shows_the_router_the_expert_count_and_the_shared_expert() {
        let t = drawn("qwen3.6-35b-a3b-fp8.config.json", "qwen3.6-35b-a3b-fp8.json", 3);
        assert!(t.iter().any(|s| s.starts_with("mlp.gate 256 x 2048")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("256 x experts, top 8")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("shared expert")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("mlp.shared_expert_gate")), "{t:?}");
        assert!(!t.iter().any(|s| s.starts_with("mlp.gate_proj")));
    }

    #[test]
    fn a_llama_layer_has_no_norm_boxes_inside_attention() {
        let t = drawn("llama-3.2-1b.config.json", "llama-3.2-1b.json", 0);
        assert!(!t.iter().any(|s| s.contains("q_norm")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("mlp.gate_proj 8192 x 2048")));
        assert!(t.iter().any(|s| s.starts_with("swiglu")));
    }

    #[test]
    fn a_family_the_engine_has_not_seen_draws_with_its_extra_norms() {
        let t = drawn("gemma4-31b.config.json", "gemma4-31b.json", 0);
        assert!(t.iter().any(|s| s.starts_with("pre_feedforward_layernorm")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("post_feedforward_layernorm")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("geglu")), "{t:?}");
        assert!(t.iter().any(|s| s.starts_with("self_attn.q_proj 8192 x 5376")), "{t:?}");
    }

    #[test]
    fn the_rope_label_has_no_strikethrough_and_a_compact_theta() {
        assert_eq!(theta_text(10_000_000.0), "10M");
        let job = shape_job("theta 10M, 64 of 256 dims", 300.0);
        assert!(!job.sections.is_empty());
        for s in &job.sections {
            assert_eq!(s.format.strikethrough, Stroke::NONE);
            assert_eq!(s.format.underline, Stroke::NONE);
        }
    }
}
