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
fn bf16_prefill_splits_activations_and_q8_does_not() {
    let bf16 = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(bf16.source.contains("EPI_ACT, false, 0, true>"));
    let mut s = spec("qwen3-0.6b.toml");
    s.quant = Quant::Q8;
    let q8 = Module::generate(&s, 256).unwrap();
    assert!(q8.source.contains("EPI_ACT, false, 1, false>"));
    assert!(!q8.source.contains(", true>(w, s, b, xh"));
}

#[test]
fn fp8_spec_uses_fp8_projections_and_keeps_lm_head_and_gates_bf16() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml");
    let s = Spec::load(&path).unwrap();
    let m = Module::generate(&s, 4096).unwrap();
    assert!(!m.source.contains("{{"));
    assert!(m.source.contains("linear_fp8<5120, 6144, false, true>"));
    assert!(m.source.contains("linear_bf16<248320, 5120, false, false>"));
    assert!(m.source.contains("linear_norm<2, LA_IN_ROWS, false>"));
    assert!(m.source.contains("constexpr int LA_IN_ROWS = LA_CONV_DIM + LA_VALUE_DIM;"));
    assert!(m.source.contains("linear_norm<WF_BF16, LA_BA_ROWS, false>"));
    let tc: Vec<&str> = m.source.lines().filter(|l| l.contains("gemm_tc<")).collect();
    let has = |s: &str| tc.iter().any(|l| l.contains(s));
    assert!(has("gemm_tc<34816, 5120, 64, 128, 2, 4, 64, 2, EPI_ACT, false, 2, true>"), "{tc:#?}");
    assert!(has("gemm_tc<248320, 5120, 64, 128, 2, 4, 32, 2, EPI_STORE, false, 0, true>"));
    assert!(has("gemm_tc<96, 5120, 64, 64, 2, 2, 32, 2, EPI_STORE, false, 0, true>"));
    assert!(m.kernels.contains(&"lin_ba_mma"));
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
    assert!(m.source.contains("gemm<1152, 896, true, false, 0>"));
}

#[test]
fn hybrid_spec_adds_the_delta_net_kernels_and_dense_specs_do_not() {
    let json = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5/config.json"),
    )
    .unwrap();
    let hybrid = Module::generate(&Spec::from_hf_config(&json).unwrap(), 256).unwrap();
    assert!(!hybrid.source.contains("{{"));
    for name in specrt::codegen::GDN_KERNELS {
        assert!(hybrid.kernels.contains(name), "{name}");
        assert!(hybrid.source.contains(&format!("void {name}(")) || hybrid.source.contains(&format!(") {name}(")), "{name}");
    }
    assert!(hybrid.source.contains("constexpr int QKV_ROW = 1536;"));
    assert!(hybrid.source.contains("constexpr int ROT = 32;"));
    assert!(hybrid.source.contains("constexpr bool OUT_GATE = true;"));
    assert!(hybrid.source.contains("constexpr int LA_VH = 6;"));
    assert!(hybrid.smem.contains_key("gdn_chunk_prep"));
    let dense = Module::generate(&spec("qwen3-0.6b.toml"), 256).unwrap();
    assert!(!dense.source.contains("gdn_step"));
    assert!(dense.kernels.iter().all(|k| !k.starts_with("gdn_") && !k.starts_with("lin_")));
    assert!(dense.source.contains("constexpr int ROT = 128;"));
    assert!(dense.source.contains("constexpr bool OUT_GATE = false;"));
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
