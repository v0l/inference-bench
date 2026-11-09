use crate::strip::params_text;
use eframe::egui::{self, Color32, Mesh, Rect, Sense, Shape, Stroke, StrokeKind, pos2, vec2};
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;
use specrt::anatomy::{Component, ModelAnatomy};

const BAR_H: f32 = 22.0;
const POINTS: usize = 48;

pub fn component_tint(name: &str) -> Color32 {
    match name {
        "embedding" => LEGEND,
        "attention" => READOUT,
        "delta-net" => TRACE,
        "mlp" => VALUE,
        "experts" => VALUE,
        "shared expert" => VALUE.gamma_multiply(0.6),
        "router" => WARN.gamma_multiply(0.7),
        "norms" => WARN,
        "lm_head" => LEGEND.gamma_multiply(0.6),
        _ => OK,
    }
}

pub fn max_totals<'a>(models: impl Iterator<Item = &'a ModelAnatomy>) -> (u64, u64) {
    models.fold((1, 1), |(p, b), a| (p.max(a.total_params), b.max(a.total_bytes)))
}

fn stacked(ui: &mut egui::Ui, a: &ModelAnatomy, scale: u64, bytes: bool, label: &str) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, BAR_H), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, WELL);
    let value = |c: &Component| if bytes { c.bytes } else { c.params };
    let mut x = rect.left();
    for c in &a.components {
        let w = value(c) as f32 / scale as f32 * width;
        if w <= 0.0 {
            continue;
        }
        let r = Rect::from_min_size(pos2(x, rect.top()), vec2(w, BAR_H));
        let tint = component_tint(c.name);
        ui.painter().rect_filled(r.shrink2(vec2(0.0, 0.0)), 0.0, tint.gamma_multiply(0.8));
        ui.painter().rect_stroke(r, 0.0, Stroke::new(1.0, WELL), StrokeKind::Inside);
        let id = egui::Id::new(("component", label, c.name, bytes));
        let resp = ui.interact(r, id, Sense::hover());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("{label} {} {}", c.name, if bytes { "bytes" } else { "params" })));
        resp.on_hover_text(format!(
            "{}\n{} parameters\n{}\n{:.1}% of the bar",
            c.name,
            params_text(c.params),
            human_size(c.bytes),
            value(c) as f32 / if bytes { a.total_bytes } else { a.total_params } as f32 * 100.0
        ));
        x += w;
    }
}

fn effects(a: &ModelAnatomy, spec: &specrt::spec::Spec) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(m) = &a.moe {
        let top = m.active.map_or("unknown".to_string(), |k| k.to_string());
        out.push(format!(
            "moe: {} experts per layer, top {top} routed plus {} shared, {} of {} layers",
            m.experts,
            if m.shared_intermediate > 0 { "1" } else { "no" },
            m.moe_layers,
            a.layers.len()
        ));
        out.push(format!("experts hold {} of {}, {} per token", human_size(m.routed_bytes), human_size(a.total_bytes), human_size(a.decode_bytes)));
    }
    out.push(format!("head_quant {}", specrt::anatomy::quant_tag(a.head_quant)));
    if a.head_tied {
        out.push("lm_head tied to the embedding".into());
    }
    let act = match (spec.act_quant, spec.mlp.act_quant) {
        (specrt::spec::ActQuant::Fp8, _) => "act_quant fp8 on the matmul inputs",
        (_, specrt::spec::ActQuant::Fp4) => "mlp act_quant fp4 on the matmul inputs",
        _ => "act_quant none, activations stay bf16",
    };
    out.push(act.into());
    out
}

fn key(ui: &mut egui::Ui, a: &ModelAnatomy) {
    ui.horizontal_wrapped(|ui| {
        for c in &a.components {
            let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 1.0, component_tint(c.name));
            Line::new().note(format!("{} {}", c.name, human_size(c.bytes))).size(11.0).show(ui);
            ui.add_space(6.0);
        }
    });
}

pub fn bars(ui: &mut egui::Ui, a: &ModelAnatomy, spec: &specrt::spec::Spec, scales: (u64, u64), label: &str) {
    Line::new().legend("parameters").gap(8.0).value(params_text(a.total_params)).size(11.0).gap(14.0).legend("active per token").gap(8.0).value(params_text(a.decode_params)).size(11.0).show(ui);
    stacked(ui, a, scales.0, false, label);
    ui.add_space(4.0);
    Line::new().legend("bytes").gap(8.0).measured(human_size(a.total_bytes)).size(11.0).gap(14.0).legend("read per token").gap(8.0).measured(human_size(a.decode_bytes)).size(11.0).show(ui);
    stacked(ui, a, scales.1, true, label);
    key(ui, a);
    for e in effects(a, spec) {
        Line::new().note(e).size(10.5).show(ui);
    }
}

pub fn memory_summary(ui: &mut egui::Ui, a: &ModelAnatomy) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 14.0;
        Line::new().legend("kv per 1k tokens").gap(8.0).measured(human_size(a.kv_bytes_per_token * 1024)).size(11.0).show(ui);
        Line::new().legend("state per sequence").gap(8.0).measured(human_size(a.state_bytes_per_seq)).size(11.0).show(ui);
        Line::new().legend("weights").gap(8.0).measured(human_size(a.total_bytes)).size(11.0).show(ui);
    });
}

pub fn memory_scale(models: &[&ModelAnatomy], free: Option<u64>) -> u64 {
    let top = models.iter().map(|a| a.memory_at(a.max_context)).max().unwrap_or(1);
    top.max(free.unwrap_or(0)).max(1)
}

fn points(a: &ModelAnatomy) -> Vec<usize> {
    let top = a.max_context.max(2048);
    (0..POINTS).map(|i| 1024 + (top - 1024) * i / (POINTS - 1)).collect()
}

pub fn memory_chart(ui: &mut egui::Ui, a: &ModelAnatomy, y_max: u64, free: Option<u64>) {
    let width = ui.available_width();
    let height = 150.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(width, height + 16.0), Sense::hover());
    let plot = Rect::from_min_size(rect.min, vec2(width, height));
    let p = ui.painter();
    p.rect_filled(plot, 2.0, WELL);
    let xs = points(a);
    let (x0, x1) = (xs[0] as f32, *xs.last().unwrap() as f32);
    let at = |ctx: usize, bytes: u64| pos2(plot.left() + (ctx as f32 - x0) / (x1 - x0) * plot.width(), plot.bottom() - bytes as f32 / y_max as f32 * plot.height());
    let layers: [(u64, Color32); 3] = [(0, LEGEND), (a.state_bytes_per_seq, TRACE), (0, READOUT)];
    let mut floor = vec![0u64; xs.len()];
    for (li, (flat, tint)) in layers.iter().enumerate() {
        let tops: Vec<u64> = xs
            .iter()
            .enumerate()
            .map(|(i, &c)| match li {
                0 => a.total_bytes,
                1 => floor[i] + flat,
                _ => floor[i] + a.kv_bytes_per_token * c as u64,
            })
            .collect();
        for i in 0..xs.len() - 1 {
            let quad = vec![at(xs[i], floor[i]), at(xs[i + 1], floor[i + 1]), at(xs[i + 1], tops[i + 1]), at(xs[i], tops[i])];
            let mut m = Mesh::default();
            let c = tint.gamma_multiply(0.35);
            for q in &quad {
                m.colored_vertex(*q, c);
            }
            m.add_triangle(0, 1, 2);
            m.add_triangle(0, 2, 3);
            p.add(Shape::mesh(m));
        }
        let line: Vec<_> = xs.iter().zip(&tops).map(|(&c, &t)| at(c, t)).collect();
        p.add(Shape::line(line, Stroke::new(1.5, *tint)));
        floor = tops;
    }
    if let Some(f) = free {
        let y = plot.bottom() - f as f32 / y_max as f32 * plot.height();
        p.line_segment([pos2(plot.left(), y), pos2(plot.right(), y)], Stroke::new(1.5, FAULT));
        let g = p.layout_no_wrap(format!("gpu free {}", human_size(f)), figure(10.0), FAULT);
        p.galley(pos2(plot.right() - g.size().x - 6.0, (y - 14.0).max(plot.top() + 2.0)), g, FAULT);
    }
    for (frac, text) in [(0.0, "1k".to_string()), (0.5, format!("{}k", (x0 + (x1 - x0) / 2.0) as usize / 1024)), (1.0, format!("{}k", x1 as usize / 1024))] {
        let g = p.layout_no_wrap(text, figure(9.5), LEGEND);
        let x = (plot.left() + frac * plot.width() - g.size().x * frac).max(plot.left());
        p.galley(pos2(x, plot.bottom() + 3.0), g, LEGEND);
    }
    let top = p.layout_no_wrap(human_size(y_max), figure(9.5), LEGEND);
    p.galley(pos2(plot.left() + 4.0, plot.top() + 2.0), top, LEGEND);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, "memory against context"));
    if let Some(pos) = resp.hover_pos().filter(|p| plot.contains(*p)) {
        let ctx = (x0 + (pos.x - plot.left()) / plot.width() * (x1 - x0)) as usize;
        resp.on_hover_text(format!("{} tokens\n{} in total\n{} weights, {} state, {} kv", ctx, human_size(a.memory_at(ctx)), human_size(a.total_bytes), human_size(a.state_bytes_per_seq), human_size(a.kv_bytes_per_token * ctx as u64)));
    }
    ui.horizontal(|ui| {
        for (name, tint) in [("weights", LEGEND), ("state", TRACE), ("kv", READOUT)] {
            let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 1.0, tint);
            Line::new().note(name).size(11.0).show(ui);
            ui.add_space(6.0);
        }
    });
}
