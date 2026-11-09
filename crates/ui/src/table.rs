use crate::detail::{Source, View};
use crate::strip::params_text;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;
use specrt::anatomy::quant_tag;
use specrt::spec::{ActQuant, Activation, Block, TileChoice};

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub group: &'static str,
    pub field: &'static str,
    pub value: String,
    pub source: &'static str,
}

fn yes(b: bool) -> String {
    if b { "yes".into() } else { "no".into() }
}

pub fn entries(view: &View, bandwidth_gbs: Option<u64>) -> Vec<Entry> {
    let s = &view.spec;
    let a = &view.anatomy;
    let src = view.source.label();
    let mut out = Vec::new();
    let mut add = |group: &'static str, field: &'static str, value: String, source: &'static str| {
        out.push(Entry { group, field, value, source });
    };
    add("dims", "hidden", s.dims.hidden.to_string(), src);
    add("dims", "intermediate", s.dims.intermediate.to_string(), src);
    add("dims", "vocab", s.dims.vocab.to_string(), src);
    add("dims", "tie_embeddings", yes(s.dims.tie_embeddings), src);
    add("dims", "layers", s.layers.count.to_string(), src);
    let pattern: Vec<&str> = s.layers.pattern.iter().map(|b| match b { Block::Dense => "dense", Block::Full => "full", Block::Linear => "linear" }).collect();
    add("dims", "pattern", pattern.join(", "), src);
    add("attention", "heads", s.attention.heads.to_string(), src);
    add("attention", "kv_heads", s.attention.kv_heads.to_string(), src);
    add("attention", "head_dim", s.attention.head_dim.to_string(), src);
    add("attention", "qkv_bias", yes(s.attention.qkv_bias), src);
    add("attention", "qk_norm", yes(s.attention.qk_norm), src);
    add("attention", "output_gate", yes(s.attention.output_gate), src);
    add("attention", "rotary_dim", s.rotary_dim().to_string(), src);
    if let Some(la) = &s.linear_attention {
        add("linear attention", "key_heads", la.key_heads.to_string(), src);
        add("linear attention", "value_heads", la.value_heads.to_string(), src);
        add("linear attention", "key_head_dim", la.key_head_dim.to_string(), src);
        add("linear attention", "value_head_dim", la.value_head_dim.to_string(), src);
        add("linear attention", "conv_kernel", la.conv_kernel.to_string(), src);
    }
    let act = match s.mlp.activation {
        Activation::Swiglu => "swiglu",
        Activation::Geglu => "geglu",
    };
    add("mlp", "activation", act.into(), src);
    add("mlp", "experts", "dense, none".into(), "derived");
    add("norm", "kind", "rmsnorm".into(), src);
    add("norm", "eps", format!("{:e}", s.norm.eps), src);
    add("norm", "unit_offset", yes(s.norm.unit_offset), src);
    add("rope and scaling", "theta", format!("{}", s.rope.theta), src);
    add("rope and scaling", "style", "neox".into(), src);
    let scaling = s.rope.scaling.as_ref().map_or("none".to_string(), |r| format!("{:?} x {}", r.kind, r.factor).to_lowercase());
    add("rope and scaling", "scaling", scaling, src);
    add("quant", "quant", quant_tag(s.quant).into(), src);
    add("quant", "head_quant", quant_tag(s.lm_head_quant()).into(), if s.head_quant.is_some() { src } else { "default" });
    let mlp_quant = match (s.mlp.quant, s.mlp.quant_until) {
        (Some(q), Some(u)) => format!("{} for layers 0 to {}", quant_tag(q), u - 1),
        (Some(q), None) => quant_tag(q).into(),
        _ => "same as quant".into(),
    };
    add("quant", "mlp quant", mlp_quant, src);
    let act_quant = |q: ActQuant| match q { ActQuant::None => "none", ActQuant::Fp8 => "fp8", ActQuant::Fp4 => "fp4" };
    add("quant", "act_quant", act_quant(s.act_quant).into(), src);
    add("quant", "mlp act_quant", act_quant(s.mlp.act_quant).into(), src);
    let mtp = if a.mtp_layers > 0 { format!("{} layer, {} parameters", a.mtp_layers, params_text(a.mtp_params)) } else { "none".into() };
    add("mtp and runtime", "mtp", mtp, if view.source == Source::HfConfig { "config.json" } else { "config.json beside the weights" });
    let r = &s.runtime;
    let tile = match r.tile { TileChoice::Auto => "auto", TileChoice::Small => "small", TileChoice::Big => "big", TileChoice::Narrow => "narrow" };
    add("mtp and runtime", "graphs", yes(r.graphs), src);
    add("mtp and runtime", "fuse", yes(r.fuse), src);
    add("mtp and runtime", "tile", tile.into(), src);
    add("mtp and runtime", "gemm_fp32", yes(r.gemm_fp32), src);
    add("derived", "total parameters", params_text(a.total_params), "derived");
    add("derived", "active parameters", params_text(a.decode_params), "derived");
    add("derived", "weights", human_size(a.total_bytes), "derived");
    add("derived", "read per decoded token", human_size(a.decode_bytes), "derived");
    add("derived", "kv per 1k tokens", human_size(a.kv_bytes_per_token * 1024), "derived");
    add("derived", "state per sequence", human_size(a.state_bytes_per_seq), "derived");
    match bandwidth_gbs {
        Some(bw) => add("derived", "decode ceiling", format!("{:.1} tok/s at {bw} GB/s", a.decode_tokens_per_second(bw as f64 * 1e9)), "derived"),
        None => add("derived", "decode ceiling", "no bandwidth recorded".into(), "derived"),
    }
    out
}

pub fn draw(ui: &mut egui::Ui, view: &View, bandwidth_gbs: Option<u64>) {
    let mut group = "";
    for e in entries(view, bandwidth_gbs) {
        if e.group != group {
            group = e.group;
            ui.add_space(4.0);
            Line::new().legend(group).show(ui);
        }
        ui.horizontal(|ui| {
            Line::new().note(e.field).column(ui, 170.0).value(e.value.clone()).size(11.0).gap(10.0).note(format!("from {}", e.source)).size(10.0).elided(ui);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn view(name: &str) -> View {
        let spec = specrt::spec::Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures").join(name)).unwrap();
        View::new(name, spec, Source::SpecToml, 1, None)
    }

    #[test]
    fn every_group_is_listed_with_its_source() {
        let e = entries(&view("qwen3.8-27b-nvfp4.toml"), Some(800));
        for g in ["dims", "attention", "linear attention", "mlp", "norm", "rope and scaling", "quant", "mtp and runtime", "derived"] {
            assert!(e.iter().any(|x| x.group == g), "{g}");
        }
        assert!(e.iter().any(|x| x.field == "mlp quant" && x.value.contains("0 to 55")));
        assert!(e.iter().filter(|x| x.group != "derived").all(|x| !x.source.is_empty()));
        assert!(e.iter().any(|x| x.field == "decode ceiling" && x.value.contains("tok/s at 800 GB/s")));
    }

    #[test]
    fn the_ceiling_says_so_when_no_bandwidth_is_recorded() {
        let e = entries(&view("qwen3.8-27b.toml"), None);
        assert!(e.iter().any(|x| x.field == "decode ceiling" && x.value.starts_with("no bandwidth")));
    }
}
