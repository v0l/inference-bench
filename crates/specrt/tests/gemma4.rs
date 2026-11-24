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
fn the_e2b_config_maps_its_shared_tail_and_wide_mlp() {
    let mut e2b: serde_json::Value = serde_json::from_str(&popular("google__gemma-4-E2B-it")).unwrap();
    e2b["text_config"]["hidden_size_per_layer_input"] = 0.into();
    let spec = Spec::from_hf_config(&e2b.to_string()).unwrap();
    assert_eq!(spec.layers.count, 35);
    assert_eq!(spec.layers.shared_tail, 20);
    assert!(spec.mlp.wide_shared_tail);
    assert!(!spec.shares_kv(14) && spec.shares_kv(15));
    assert_eq!(spec.layer_intermediate(14), spec.dims.intermediate);
    assert_eq!(spec.layer_intermediate(15), 2 * spec.dims.intermediate);
    for layer in 15..35 {
        let source = spec.kv_source(layer);
        assert!(source < 15 && spec.block(source) == spec.block(layer));
        assert_eq!(source, (0..15).rev().find(|&j| spec.block(j) == spec.block(layer)).unwrap());
    }
    let dense = Spec::from_hf_config_serving(&popular("google__gemma-4-31B-it")).unwrap();
    assert_eq!(dense.layers.shared_tail, 0);
    assert!(!dense.mlp.wide_shared_tail);
    assert!((0..dense.layers.count).all(|l| dense.kv_source(l) == l));
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

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_served_two_slots_match_transformers_greedy() {
    use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
    let (tokens, greedy, _, _) = reference();
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    let layout = Layout { max_ctx, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&tiny(), layout).unwrap();
    rt.set_block_table(0, &(0..pages as u32).rev().collect::<Vec<_>>()).unwrap();
    rt.set_block_table(1, &(pages as u32..2 * pages as u32).collect::<Vec<_>>()).unwrap();
    let mut st = rt.stepper().unwrap();
    let short = &tokens[..40];
    let mut firsts = [None, None];
    for (slot, prompt) in [(0usize, &tokens[..]), (1, short)] {
        let mut start = 0;
        while start < prompt.len() {
            let end = (start + 64).min(prompt.len());
            let last = end == prompt.len();
            firsts[slot] = rt.prefill_slot(&mut st, slot, &prompt[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap();
            start = end;
        }
    }
    let mut out = vec![firsts[0].unwrap()];
    let mut other = vec![firsts[1].unwrap()];
    while out.len() < greedy.len() {
        let rows = [
            DecodeRow { slot: 0, token: *out.last().unwrap(), pos: tokens.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 },
            DecodeRow { slot: 1, token: *other.last().unwrap(), pos: short.len() + other.len() - 1, sampling: Sampling::GREEDY, step: other.len() as u32 },
        ];
        let next = rt.decode(&mut st, &rows).unwrap();
        out.push(next[0]);
        other.push(next[1]);
    }
    assert_eq!(out, greedy, "served greedy beside a second sequence");
}
