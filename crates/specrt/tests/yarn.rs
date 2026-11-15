use serde_json::Value;
use specrt::spec::{RopeScalingKind, Spec, yarn_inv_freq};
use std::path::Path;

fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yarn_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn spec_for(params: &Value) -> Spec {
    let mut rope = params.clone();
    let rope_type = rope.as_object_mut().unwrap();
    rope_type.remove("rope_theta");
    rope_type.remove("partial_rotary_factor");
    let mut spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.config.json")).unwrap();
    let mut cfg: Value = serde_json::from_str(&json).unwrap();
    cfg["text_config"]["rope_parameters"] = params.clone();
    let parsed = Spec::from_hf_config(&cfg.to_string()).unwrap();
    spec.rope = parsed.rope;
    assert_eq!(spec.rope.scaling.as_ref().unwrap().kind, RopeScalingKind::Yarn);
    spec
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn yarn_tables_match_transformers_for_qwen3_8_at_the_context_boundaries() {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let spec = spec_for(&case["rope_parameters"]);
        let rot = spec.rotary_dim();
        assert_eq!(rot, 64, "{name}");
        let scaling = spec.rope.scaling.as_ref().unwrap();
        let ours = yarn_inv_freq(spec.rope.theta, rot, scaling);
        let want = floats(&case["inv_freq"]);
        assert_eq!(ours.len(), want.len(), "{name}");
        for (i, (a, b)) in ours.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-12) + 1e-12, "{name} inv_freq[{i}] {a} vs {b}");
        }
        let mscale = case["attention_factor"].as_f64().unwrap();
        assert!((scaling.yarn.as_ref().unwrap().attention_factor - mscale).abs() < 1e-6, "{name}");
        assert!((spec.rope_constants().mscale - mscale).abs() < 1e-6, "{name}");
        let positions = floats(&case["positions"]);
        for (p, pos) in positions.iter().enumerate() {
            let cos = floats(&case["cos"][p]);
            let sin = floats(&case["sin"][p]);
            for (i, f) in ours.iter().enumerate() {
                let angle = *pos * *f;
                let (c, s) = (angle.cos() * mscale, angle.sin() * mscale);
                let tolerance = 1e-3 + pos * f.abs() * 1.2e-7 * 2.0;
                assert!((c - cos[i]).abs() < tolerance && (s - sin[i]).abs() < tolerance, "{name} pos {pos} dim {i}: {c} {s} vs {} {}", cos[i], sin[i]);
            }
        }
    }
}

#[test]
fn yarn_leaves_the_fast_dimensions_alone_and_stretches_the_slow_ones_by_the_factor() {
    let case = &reference()["cases"][0];
    let spec = spec_for(&case["rope_parameters"]);
    let f = yarn_inv_freq(spec.rope.theta, spec.rotary_dim(), spec.rope.scaling.as_ref().unwrap());
    let plain = |i: usize| 1.0 / spec.rope.theta.powf(2.0 * i as f64 / spec.rotary_dim() as f64);
    assert_eq!(f[0], plain(0));
    let last = f.len() - 1;
    assert!((f[last] * 4.0 - plain(last)).abs() < 1e-12 * plain(last));
    assert!(f.windows(2).all(|w| w[0] > w[1]));
}

#[test]
fn the_generated_kernel_bakes_the_yarn_constants() {
    let case = &reference()["cases"][0];
    let spec = spec_for(&case["rope_parameters"]);
    let m = specrt::codegen::Module::generate(&spec, 256).unwrap();
    assert!(m.source.contains("constexpr int ROPE_KIND = 3;"));
    assert!(m.source.contains("constexpr float ROPE_MSCALE = "));
    assert!(!m.source.contains("{{"));
    let plain = specrt::codegen::Module::generate(&Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap(), 256).unwrap();
    assert!(plain.source.contains("constexpr float ROPE_MSCALE = 1e0f;"));
}
