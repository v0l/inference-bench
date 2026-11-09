use eframe::egui::{self, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use egui_bench::prelude::*;
use specrt::anatomy::{LayerKind, quant_tag};
use specrt::spec::{Activation, ActQuant, Quant, RopeScalingKind, Spec};

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub title: String,
    pub shape: String,
    pub quant: Option<Quant>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    Io(Node),
    Boxes(Vec<Node>),
    Add,
}

fn node(title: &str, shape: impl Into<String>) -> Node {
    Node { title: title.into(), shape: shape.into(), quant: None }
}

fn linear(title: &str, shape: impl Into<String>, quant: Quant) -> Node {
    Node { title: title.into(), shape: shape.into(), quant: Some(quant) }
}

fn norm_node(spec: &Spec) -> Node {
    let offset = if spec.norm.unit_offset { ", 1 + w" } else { "" };
    node("rmsnorm", format!("{}{offset}", spec.dims.hidden))
}

fn mlp_rows(spec: &Spec, layer: usize) -> Vec<Row> {
    let (h, i) = (spec.dims.hidden, spec.dims.intermediate);
    let q = spec.mlp_quant(layer);
    let act = match spec.mlp.activation {
        Activation::Swiglu => "swiglu",
        Activation::Geglu => "geglu",
    };
    let act_in = match (q, spec.mlp.act_quant, spec.act_quant) {
        (Quant::Nvfp4G16, ActQuant::Fp4, _) => ", fp4 input",
        (_, _, ActQuant::Fp8) => ", fp8 input",
        _ => "",
    };
    vec![
        Row::Boxes(vec![linear("gate", format!("{h} → {i}"), q), linear("up", format!("{h} → {i}"), q)]),
        Row::Boxes(vec![node(act, format!("{i}{act_in}"))]),
        Row::Boxes(vec![linear("down", format!("{i} → {h}"), q)]),
    ]
}

fn attention_rows(spec: &Spec) -> Vec<Row> {
    let a = &spec.attention;
    let (h, d) = (spec.dims.hidden, a.head_dim);
    let q = spec.quant;
    let gate = if a.output_gate { "2 × " } else { "" };
    let mut rows = vec![Row::Boxes(vec![
        linear("q", format!("{h} → {gate}{} × {d}", a.heads), q),
        linear("k", format!("{h} → {} × {d}", a.kv_heads), q),
        linear("v", format!("{h} → {} × {d}", a.kv_heads), q),
    ])];
    if a.qkv_bias {
        rows.push(Row::Boxes(vec![node("qkv bias", format!("{}", spec.qkv_rows()))]));
    }
    if a.qk_norm {
        rows.push(Row::Boxes(vec![node("q norm", format!("{d}")), node("k norm", format!("{d}"))]));
    }
    let scaling = match &spec.rope.scaling {
        Some(s) => format!(", {} × {}", match s.kind { RopeScalingKind::Linear => "linear", RopeScalingKind::Llama3 => "llama3" }, s.factor),
        None => String::new(),
    };
    rows.push(Row::Boxes(vec![node(
        "rope neox",
        format!("theta {}, {} of {d} dims{scaling}", spec.rope.theta, spec.rotary_dim()),
    )]));
    let group = a.heads / a.kv_heads;
    let gating = if a.output_gate { ", sigmoid output gate" } else { "" };
    rows.push(Row::Boxes(vec![node(
        "gqa attention",
        format!("{} heads, {} kv, {group} per group{gating}", a.heads, a.kv_heads),
    )]));
    rows.push(Row::Boxes(vec![linear("o", format!("{} × {d} → {h}", a.heads), q)]));
    rows
}

fn delta_rows(spec: &Spec) -> Vec<Row> {
    let Some(la) = &spec.linear_attention else { return Vec::new() };
    let h = spec.dims.hidden;
    let q = spec.quant;
    let (kh, vh, kd, vd) = (la.key_heads, la.value_heads, la.key_head_dim, la.value_head_dim);
    vec![
        Row::Boxes(vec![
            linear("qkv", format!("{h} → 2 × {kh} × {kd} + {vh} × {vd}"), q),
            linear("z gate", format!("{h} → {vh} × {vd}"), q),
            linear("b, a", format!("{h} → 2 × {vh}"), Quant::Bf16),
        ]),
        Row::Boxes(vec![node("causal conv", format!("{} channels, width {}", la.conv_dim(), la.conv_kernel))]),
        Row::Boxes(vec![node(
            "gated delta rule",
            format!("{kh} key, {vh} value heads, state {kd} × {vd} per head"),
        )]),
        Row::Boxes(vec![node("gated rmsnorm", format!("{vd} per head, silu(z)"))]),
        Row::Boxes(vec![linear("out", format!("{vh} × {vd} → {h}"), q)]),
    ]
}

pub fn rows_for(spec: &Spec, kind: LayerKind, layer: usize) -> Vec<Row> {
    let h = spec.dims.hidden;
    let mut rows = vec![Row::Io(node("input", format!("{h}"))), Row::Boxes(vec![norm_node(spec)])];
    rows.extend(match kind {
        LayerKind::Attention => attention_rows(spec),
        LayerKind::DeltaNet => delta_rows(spec),
    });
    rows.push(Row::Add);
    rows.push(Row::Boxes(vec![norm_node(spec)]));
    rows.extend(mlp_rows(spec, layer));
    rows.push(Row::Add);
    rows.push(Row::Io(node("output", format!("{h}"))));
    rows
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

fn quant_tint(q: Quant) -> Color32 {
    match q {
        Quant::Bf16 => LEGEND,
        Quant::Nvfp4G16 => OK,
        _ => READOUT,
    }
}

fn paint_node(ui: &egui::Ui, rect: Rect, n: &Node, tint: Color32) {
    let p = ui.painter();
    p.rect_filled(rect, 2.0, PANEL);
    p.rect_stroke(rect, 2.0, Stroke::new(1.0, tint.gamma_multiply(0.7)), StrokeKind::Inside);
    let title = p.layout_no_wrap(n.title.to_uppercase(), legend_font(10.5), tint);
    p.galley(pos2(rect.left() + 6.0, rect.top() + 4.0), title, tint);
    let shape = p.layout(n.shape.clone(), figure(10.0), VALUE, (rect.width() - 12.0).max(10.0));
    p.galley(pos2(rect.left() + 6.0, rect.top() + 19.0), shape, VALUE);
    if let Some(q) = n.quant {
        let t = quant_tag(q).to_uppercase();
        let tag = p.layout_no_wrap(t, legend_font(8.5), quant_tint(q));
        let tr = Rect::from_min_size(pos2(rect.right() - tag.size().x - 10.0, rect.top() + 3.0), tag.size() + vec2(6.0, 2.0));
        p.rect_filled(tr, 2.0, quant_tint(q).gamma_multiply(0.18));
        p.galley(tr.min + vec2(3.0, 1.0), tag, quant_tint(q));
    }
}

pub fn draw(ui: &mut egui::Ui, spec: &Spec, kind: LayerKind, layer: usize) {
    let rows = rows_for(spec, kind, layer);
    let width = ui.available_width().min(720.0);
    let (area, _) = ui.allocate_exact_size(vec2(width, total_height(&rows)), Sense::hover());
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
                    let tint = if nd.quant.is_some() { VALUE } else { TRACE };
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

pub fn title(kind: LayerKind, layer: Option<usize>, spec: &Spec) -> String {
    match layer {
        Some(i) => format!("layer {i}: {}", kind.label()),
        None => format!("{} ({})", kind.label(), spec.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn spec(name: &str) -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures").join(name)).unwrap()
    }

    fn texts(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .flat_map(|r| match r {
                Row::Io(n) => vec![format!("{} {}", n.title, n.shape)],
                Row::Boxes(ns) => ns.iter().map(|n| format!("{} {}", n.title, n.shape)).collect(),
                Row::Add => vec!["add".into()],
            })
            .collect()
    }

    #[test]
    fn the_27b_attention_block_shows_gqa_and_tensor_shapes() {
        let rows = rows_for(&spec("qwen3.8-27b.toml"), LayerKind::Attention, 3);
        let t = texts(&rows);
        assert!(t.contains(&"q 5120 → 2 × 24 × 256".to_string()), "{t:?}");
        assert!(t.contains(&"k 5120 → 4 × 256".to_string()));
        assert!(t.iter().any(|s| s.starts_with("gqa attention 24 heads, 4 kv, 6 per group")));
        assert!(t.iter().any(|s| s.starts_with("rope neox") && s.contains("64 of 256")));
        assert!(t.contains(&"q norm 256".to_string()));
        assert_eq!(rows.iter().filter(|r| **r == Row::Add).count(), 2);
    }

    #[test]
    fn the_delta_net_block_shows_conv_and_heads() {
        let t = texts(&rows_for(&spec("qwen3.8-27b.toml"), LayerKind::DeltaNet, 0));
        assert!(t.iter().any(|s| s.starts_with("causal conv 10240 channels, width 4")), "{t:?}");
        assert!(t.iter().any(|s| s.contains("16 key, 48 value heads")));
        assert!(t.contains(&"out 48 × 128 → 5120".to_string()));
    }

    #[test]
    fn linears_carry_the_quant_of_their_layer() {
        let s = spec("qwen3.8-27b-nvfp4.toml");
        let early = rows_for(&s, LayerKind::DeltaNet, 0);
        let late = rows_for(&s, LayerKind::Attention, 63);
        let gate = |rows: &[Row]| {
            rows.iter()
                .find_map(|r| match r {
                    Row::Boxes(ns) => ns.iter().find(|n| n.title == "gate").and_then(|n| n.quant),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(gate(&early), Quant::Nvfp4G16);
        assert_eq!(gate(&late), Quant::Fp8Channel);
    }
}
