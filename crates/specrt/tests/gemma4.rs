use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

fn popular(repo: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular").join(repo).join("config.json");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn param(spec: &Spec, name: &str) -> f64 {
    spec.attention.params[name]
}

#[test]
fn the_dense_31b_maps_two_attention_shapes_by_spec_alone() {
    let spec = Spec::from_hf_config_serving(&popular("google__gemma-4-31B-it")).unwrap();
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Sliding, Block::Sliding, Block::Sliding, Block::Sliding, Block::Full]);
    assert_eq!(spec.layers.window, Some(1024));
    assert_eq!((param(&spec, "hd_s"), param(&spec, "hd_g")), (256.0, 512.0));
    assert_eq!((param(&spec, "kvh_s"), param(&spec, "kvh_g"), param(&spec, "k_eq_v")), (16.0, 4.0, 1.0));
    assert_eq!(param(&spec, "angles_g"), 64.0);
    assert_eq!(spec.qkv_rows(), 32 * 512 + 2 * 16 * 256);
    assert_eq!(spec.o_dim(), 32 * 512);
    assert_eq!((spec.o_cols_in(Block::Sliding), spec.o_cols_in(Block::Full)), (32 * 256, 32 * 512));
    assert_eq!(spec.dims.logit_softcap, Some(30.0));
    assert_eq!(spec.norm.layer_scale.as_deref(), Some("layer_scalar"));
    assert!(spec.norm.sandwich && !spec.norm.unit_offset);
    assert!(spec.vision.is_none());
    let states: Vec<(&str, usize)> = spec.state.iter().map(|s| (s.name.as_str(), s.layer_count(&spec))).collect();
    assert_eq!(states, vec![("k_s", 50), ("v_s", 50), ("k_g", 10), ("v_g", 10)]);
}

#[test]
fn the_12b_unified_config_maps_like_the_31b() {
    let spec = Spec::from_hf_config_serving(&popular("google__gemma-4-12B-it")).unwrap();
    assert_eq!((param(&spec, "kvh_s"), param(&spec, "kvh_g")), (8.0, 1.0));
    assert_eq!(spec.layers.count, 48);
}

#[test]
fn the_parts_not_built_yet_are_refused_by_name() {
    let moe = Spec::from_hf_config_serving(&popular("google__gemma-4-26B-A4B-it")).unwrap_err().to_string();
    assert!(moe.contains("enable_moe_block"), "{moe}");
    let small = Spec::from_hf_config_serving(&popular("google__gemma-4-E2B-it")).unwrap_err().to_string();
    assert!(small.contains("hidden_size_per_layer_input"), "{small}");
}

#[test]
fn the_kv_cache_holds_each_layer_kind_at_its_own_width() {
    let spec = Spec::from_hf_config_serving(&popular("google__gemma-4-31B-it")).unwrap();
    let sliding = specrt::state::kv_bytes_per_token_at(&spec, 0);
    let full = specrt::state::kv_bytes_per_token_at(&spec, 5);
    assert_eq!((sliding, full), (2 * 16 * 256 * 2, 2 * 4 * 512 * 2));
}

#[test]
fn the_tiny_fixture_has_a_spec_that_builds_every_kernel() {
    let spec = tiny();
    specrt::codegen::Module::generate(&spec, 256).unwrap();
    let states: Vec<(&str, usize)> = spec.state.iter().map(|s| (s.name.as_str(), s.layer_count(&spec))).collect();
    assert_eq!(states, vec![("k_s", 4), ("v_s", 4), ("k_g", 2), ("v_g", 2)]);
    assert_eq!((param(&spec, "hd_s"), param(&spec, "hd_g"), param(&spec, "kvh_g"), param(&spec, "angles_g")), (32.0, 64.0, 1.0, 8.0));
}

fn tiny() -> Spec {
    let dir = fixture("tiny-gemma4");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = "tiny-gemma4".into();
    spec.repo = dir.display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).expect("SPECRT_REF_DIR holds scripts/tiny_gemma4.py output");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tiny-gemma4.ref.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join("tiny-gemma4.ref.bin")).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_layer_states_match_transformers() {
    let (tokens, _, _, _) = reference();
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap();
    let bytes = std::fs::read(dir.join("tiny-gemma4.layers.bin")).unwrap();
    let want: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut rt = specrt::runtime::Runtime::load(&tiny(), 256).unwrap();
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * 128;
    for layer in 0..6 {
        let (a, b) = (&trace.layers[layer * per..(layer + 1) * per], &want[layer * per..(layer + 1) * per]);
        let max_abs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let size = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
        eprintln!("layer {layer} max_abs {max_abs} of {size}");
        assert!(max_abs < 2e-2 * size.max(1.0), "layer {layer} max abs diff {max_abs}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_logits_and_greedy_match_transformers() {
    let (tokens, greedy, vocab, want) = reference();
    let mut rt = specrt::runtime::Runtime::load(&tiny(), 256).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    for r in 0..tokens.len() {
        let row = |v: &[f32]| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(row(&ours), row(&want), "row {r}");
    }
    let got = rt.generate(&tokens, greedy.len(), specrt::runtime::Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, greedy);
}
