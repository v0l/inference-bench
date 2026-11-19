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
        let ours = yarn_inv_freq(spec.theta(), rot, scaling);
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
    let f = yarn_inv_freq(spec.theta(), spec.rotary_dim(), spec.rope.scaling.as_ref().unwrap());
    let plain = |i: usize| 1.0 / spec.theta().powf(2.0 * i as f64 / spec.rotary_dim() as f64);
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

fn extended(factor: u32) -> specrt::spec::RuntimeOptions {
    specrt::spec::RuntimeOptions { rope_extension: Some(specrt::spec::RopeExtension { kind: specrt::spec::RopeExtensionKind::Yarn, factor }), ..Default::default() }
}

fn tiny_spec() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec
}

#[test]
fn rope_extension_reads_from_the_runtime_table_and_rejects_bad_values() {
    let t: specrt::spec::RuntimeOptions = toml::from_str("rope_extension = { type = \"yarn\", factor = 4 }").unwrap();
    assert_eq!(t, extended(4));
    assert_eq!(toml::from_str::<specrt::spec::RuntimeOptions>("rope_extension = { factor = 2 }").unwrap(), extended(2));
    assert!(toml::from_str::<specrt::spec::RuntimeOptions>("rope_extension = { type = \"ntk\", factor = 4 }").is_err());
    assert!(toml::from_str::<specrt::spec::RuntimeOptions>("rope_extension = { type = \"yarn\", factor = 4, bogus = 1 }").is_err());
    assert_eq!(specrt::spec::RuntimeOptions::default().rope_extension, None);
}

#[test]
fn rope_extension_multiplies_the_model_maximum_context_and_stays_off_by_default() {
    let mut spec = tiny_spec();
    assert_eq!(specrt::engine::model_max_context(&spec), Some(4096));
    spec.runtime = extended(4);
    assert_eq!(specrt::engine::model_max_context(&spec), Some(16384));
    assert!(specrt::engine::limit_context(16384, specrt::engine::model_max_context(&spec)).is_ok());
    assert!(specrt::engine::limit_context(16385, specrt::engine::model_max_context(&spec)).is_err());
    let off = specrt::engine::apply_rope_extension(&tiny_spec()).unwrap();
    assert!(off.rope.scaling.is_none());
}

#[test]
fn rope_extension_applies_yarn_with_the_original_context_and_transformers_mscale() {
    let mut spec = tiny_spec();
    spec.runtime = extended(4);
    let s = specrt::engine::apply_rope_extension(&spec).unwrap();
    let r = s.rope.scaling.clone().unwrap();
    assert_eq!((r.kind, r.factor, r.original_max_positions), (RopeScalingKind::Yarn, 4.0, Some(4096.0)));
    assert!((r.yarn.unwrap().attention_factor - (1.0 + 0.1 * 4f64.ln())).abs() < 1e-12);
    assert!(s.runtime.rope_extension.is_none());
}

#[test]
fn rope_extension_is_refused_on_a_model_that_already_scales_its_rope() {
    let mut spec = tiny_spec();
    spec.runtime = extended(4);
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5/config.json")).unwrap();
    let mut cfg: Value = serde_json::from_str(&json).unwrap();
    cfg["rope_parameters"]["rope_type"] = "yarn".into();
    cfg["rope_parameters"]["factor"] = 2.0.into();
    cfg["rope_parameters"]["original_max_position_embeddings"] = 4096.into();
    spec.rope = Spec::from_hf_config(&cfg.to_string()).unwrap().rope;
    let e = specrt::engine::apply_rope_extension(&spec).unwrap_err();
    assert!(format!("{e:#}").contains("already declares rope scaling"), "{e:#}");
}

#[test]
fn a_serve_option_rope_extension_lifts_the_context_limit_and_the_prefix_cache_key() {
    let spec = tiny_spec();
    let mut opts = specrt::serve::ServeOptions { max_model_len: Some(16384), ..Default::default() };
    assert!(opts.model_len(&spec).is_err());
    opts.runtime = Some(extended(4));
    assert_eq!(opts.model_len(&spec).unwrap(), 16384);
    let mut with = spec.clone();
    with.runtime = extended(4);
    assert_ne!(specrt::prefix::namespace(&spec, 1, ""), specrt::prefix::namespace(&with, 1, ""));
}

#[test]
fn the_extension_options_are_in_the_load_schema_and_reach_the_runtime() {
    use specrt::options::{EngineChoice, default_of};
    assert_eq!(default_of("runtime.rope_extension.factor"), Some(Value::Null));
    let mut c = EngineChoice::default();
    c.set("runtime.rope_extension.factor", 4.into());
    let rt = c.apply(specrt::serve::ServeOptions::default()).unwrap().runtime.unwrap();
    assert_eq!(rt.rope_extension, extended(4).rope_extension);
}

fn yarn_tiny() -> Spec {
    let mut spec = tiny_spec();
    spec.rope.scaling = Some(specrt::spec::RopeScaling {
        kind: RopeScalingKind::Yarn,
        factor: 4.0,
        low_freq_factor: None,
        high_freq_factor: None,
        original_max_positions: Some(262144.0),
        yarn: Some(specrt::spec::Yarn { beta_fast: 32.0, beta_slow: 1.0, attention_factor: specrt::spec::yarn_mscale(4.0, 1.0), truncate: true }),
        longrope: None,
    });
    spec
}

fn past_a_million(spec: &Spec, reversed: bool) -> Vec<f32> {
    use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
    let max_ctx = 1_100_000usize;
    let pages = max_ctx.div_ceil(specrt::codegen::PAGE);
    let layout = Layout { max_ctx, slots: 1, kv_blocks: pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(spec, layout).unwrap();
    let table: Vec<u32> = if reversed { (0..pages as u32).rev().collect() } else { (0..pages as u32).collect() };
    rt.set_block_table(0, &table).unwrap();
    let mut st = rt.stepper().unwrap();
    let start = 1_048_000;
    let prompt: Vec<u32> = (0..100u32).map(|i| (i * 37 + 11) % 1024).collect();
    rt.prefill_slot(&mut st, 0, &prompt, start, Some((Sampling::GREEDY, 0))).unwrap();
    let mut logits = rt.last_logits().unwrap();
    for step in 0..40usize {
        let row = DecodeRow { slot: 0, token: (step as u32 * 7 + 3) % 1024, pos: start + prompt.len() + step, sampling: Sampling::GREEDY, step: step as u32 };
        rt.decode(&mut st, &[row]).unwrap();
    }
    rt.synchronize().unwrap();
    logits.extend(rt.last_logits().unwrap());
    logits
}

#[test]
#[ignore = "needs a CUDA GPU with 2 GB free"]
fn yarn_paging_and_index_maths_hold_past_a_million_positions() {
    let spec = yarn_tiny();
    let straight = past_a_million(&spec, false);
    let shuffled = past_a_million(&spec, true);
    assert!(straight.iter().all(|v| v.is_finite()));
    assert_eq!(straight, shuffled, "page order must not change the numbers");
    let mut plain = tiny_spec();
    plain.rope.scaling = None;
    let unscaled = past_a_million(&plain, false);
    assert!(unscaled.iter().all(|v| v.is_finite()));
    assert_ne!(straight, unscaled, "yarn must change the rotation at these positions");
}
