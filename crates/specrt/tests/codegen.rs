use specrt::codegen::{Module, render};
use specrt::spec::{Quant, Spec};
use std::collections::BTreeMap;
use std::path::Path;

fn spec(file: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs").join(file)).unwrap()
}

#[test]
fn render_substitutes_every_slot() {
    let vars = BTreeMap::from([("A", "1".to_string()), ("B", "two".to_string())]);
    assert_eq!(render("x{{A}}y{{B}}{{A}}", &vars).unwrap(), "x1ytwo1");
}

#[test]
fn render_rejects_unknown_slot() {
    let vars = BTreeMap::from([("A", "1".to_string())]);
    assert!(render("{{A}}{{MISSING}}", &vars).is_err());
}

#[test]
fn module_bakes_spec_constants() {
    let m = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("constexpr int HEAD_DIM = 128;"));
    assert!(m.source.contains("constexpr int GROUP = 2;"));
    assert!(m.source.contains("constexpr bool QK_NORM = true;"));
    assert!(m.source.contains("constexpr int MAX_CTX = 256;"));
    assert!(m.source.contains("linear_bf16<4096, 1024, false, false>"));
    assert!(m.source.contains("linear_bf16<151936, 1024, false, false>"));
}

#[test]
fn second_model_differs_only_by_spec() {
    let a = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    let b = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    assert_ne!(a.source, b.source);
    assert_eq!(a.kernels, b.kernels);
    assert!(b.source.contains("constexpr bool QK_NORM = false;"));
    assert!(b.source.contains("linear_bf16<1152, 896, true, false>"));
}

#[test]
fn q8_spec_selects_q8_linears() {
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("linear_q8<4096, 1024, false, false>"));
    assert!(m.source.contains("linear_q8<151936, 1024, false, false>"));
    assert!(!m.source.contains("linear_bf16<"));
}

#[test]
fn generation_is_deterministic() {
    let s = spec("qwen3-0.6b.toml");
    assert_eq!(
        Module::generate(&s, 256).unwrap().source,
        Module::generate(&s, 256).unwrap().source
    );
}

#[test]
fn module_has_batched_gemm_for_every_linear() {
    let m = Module::generate(&spec("qwen2.5-0.5b.toml"), 256).unwrap();
    for name in ["qkv_gemm", "o_gemm", "gate_up_gemm", "down_gemm", "lm_head_gemm"] {
        assert!(m.kernels.contains(&name), "{name}");
        assert!(m.source.contains(&format!("void {name}(")), "{name}");
    }
    assert!(m.source.contains("gemm<1152, 896, true, false, false>"));
}

#[test]
fn rope_scaling_and_activation_are_baked() {
    let mut s = spec("qwen3-0.6b.toml");
    let plain = Module::generate(&s, 256).unwrap();
    assert!(plain.source.contains("constexpr int ROPE_KIND = 0;"));
    assert!(plain.source.contains("constexpr int ACT_KIND = 0;"));
    s.rope.scaling = Some(specrt::spec::RopeScaling {
        kind: specrt::spec::RopeScalingKind::Linear,
        factor: 4.0,
        low_freq_factor: None,
        high_freq_factor: None,
        original_max_positions: None,
    });
    s.mlp.activation = specrt::spec::Activation::Geglu;
    let m = Module::generate(&s, 256).unwrap();
    assert!(m.source.contains("constexpr int ROPE_KIND = 1;"));
    assert!(m.source.contains("constexpr int ACT_KIND = 1;"));
    assert!(!m.source.contains("{{"));
}
