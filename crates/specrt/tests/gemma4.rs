use specrt::plan::{Buf, NormKind, Op, Phase, Plan, Weight};
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
    for repo in ["google__gemma-4-E2B-it", "google__gemma-4-E4B-it"] {
        let spec = Spec::from_hf_config_serving(&popular(repo)).unwrap();
        assert!(spec.layers.shared_tail > 0 && spec.per_layer().is_some(), "{repo}");
    }
    let e2b = Spec::from_hf_config_serving(&popular("google__gemma-4-E2B-it")).unwrap();
    let e4b = Spec::from_hf_config_serving(&popular("google__gemma-4-E4B-it")).unwrap();
    assert!(e2b.mlp.wide_shared_tail, "E2B uses the double wide mlp");
    assert!(!e4b.mlp.wide_shared_tail, "E4B does not use the double wide mlp");
    assert_eq!((e4b.layers.count, e4b.layers.shared_tail, e4b.per_layer().unwrap().dim), (42, 18, 256));
    assert_eq!((param(&e4b, "hd_s"), param(&e4b, "hd_g")), (256.0, 512.0));
    let gemma3n = Spec::from_hf_config_serving(&popular("google__gemma-3n-E2B-it")).unwrap();
    assert!(gemma3n.gated_residual().is_some() && gemma3n.per_layer().is_some(), "gemma-3n is its own family on the same per layer inputs");
}

fn unrefused() -> specrt::family::Registry {
    let texts: Vec<String> = specrt::spec::EMBEDDED
        .iter()
        .map(|(_, text)| {
            let mut table: toml::Table = toml::from_str(text).unwrap();
            if table.get("name").and_then(|n| n.as_str()) == Some("gemma4") {
                table.remove("refuse");
            }
            table.to_string()
        })
        .collect();
    specrt::family::Registry::from_texts(texts.iter().map(String::as_str)).unwrap()
}

#[test]
fn the_e2b_spec_carries_per_layer_inputs() {
    let spec = Spec::from_hf_config_serving_in(&unrefused(), &popular("google__gemma-4-E2B-it")).unwrap();
    let p = spec.per_layer().expect("the E2B config has per layer inputs");
    assert_eq!(p.dim, 256);
    assert_eq!(spec.weight_len("ple_width * hidden").unwrap(), 256 * 35 * 1536);
    assert_eq!(spec.weight_len("ple_dim * hidden").unwrap(), 256 * 1536);
    assert!(spec.hyper.is_none());
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let head = plan.head.iter().filter(|o| matches!(o, Op::PerLayer { stage: specrt::plan::PLE_HEAD, .. })).count();
    let tail = plan.layer_tail().iter().filter(|o| matches!(o, Op::PerLayer { stage: specrt::plan::PLE_LAYER, .. })).count();
    assert_eq!((head, tail), (2, 4));
    specrt::signature::validate_plan(&plan, &spec).unwrap();
    let dense = Spec::from_hf_config_serving(&popular("google__gemma-4-31B-it")).unwrap();
    assert!(dense.per_layer().is_none() && Plan::build(&dense, Phase::Prefill, true).layer_tail().is_empty());
}

#[test]
fn the_e2b_config_maps_its_shared_tail_and_wide_mlp() {
    let spec = Spec::from_hf_config_serving_in(&unrefused(), &popular("google__gemma-4-E2B-it")).unwrap();
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
fn the_wide_tail_plan_runs_the_wide_mlp_and_the_rest_is_unchanged() {
    let spec = Spec::from_hf_config_serving_in(&unrefused(), &popular("google__gemma-4-E2B-it")).unwrap();
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let wide = |ops: &[Op]| ops.iter().any(|o| matches!(o, Op::Linear { weight: Weight::GateUpWide | Weight::DownWide, .. }));
    let narrow = |ops: &[Op]| ops.iter().any(|o| matches!(o, Op::Linear { weight: Weight::GateUp | Weight::Down, .. }));
    for layer in 0..spec.layers.count {
        let ops = plan.layer_for(spec.block(layer), false, spec.shares_kv(layer));
        assert_eq!((wide(ops), narrow(ops)), (spec.shares_kv(layer), !spec.shares_kv(layer)), "layer {layer}");
    }
    assert_eq!(Weight::GateUpWide.rows(&spec), 4 * spec.dims.intermediate);
    let dense = Spec::from_hf_config_serving(&popular("google__gemma-4-31B-it")).unwrap();
    assert!(Plan::build(&dense, Phase::Prefill, true).blocks.iter().all(|b| !b.tail));
}

#[test]
fn shared_layers_own_no_cache_and_read_their_source() {
    let spec = Spec::from_hf_config_serving_in(&unrefused(), &popular("google__gemma-4-E2B-it")).unwrap();
    let owners = (0..spec.layers.count).filter(|&l| spec.owns_kv(l)).count();
    assert_eq!(owners, 15);
    let states: Vec<(&str, usize)> = spec.state.iter().map(|s| (s.name.as_str(), s.layer_count(&spec))).collect();
    assert_eq!(states, vec![("k_s", 12), ("v_s", 12), ("k_g", 3), ("v_g", 3)]);
    for layer in 15..35 {
        assert_eq!(spec.kv_index(layer), spec.attention_before(spec.kv_source(layer)));
        assert!(spec.kv_index(layer) < 15);
    }
    assert_eq!(spec.kind_index(34), spec.kind_before(spec.kv_source(34)));
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let writes = |tail: bool, block: Block| plan.layer_for(block, false, tail).iter().filter(|o| matches!(o, Op::Kernel { node } if spec.attention.node[*node as usize].kernel.as_deref().is_some_and(|k| k.contains("write")))).count();
    assert_eq!((writes(false, Block::Sliding), writes(true, Block::Sliding)), (2, 0));
    assert_eq!((writes(false, Block::Full), writes(true, Block::Full)), (2, 0));
    let per_token: u64 = (0..owners).map(|i| specrt::state::kv_bytes_per_token_at(&spec, i)).sum();
    let mut all_own = spec.clone();
    all_own.layers.shared_tail = 0;
    let full: u64 = (0..35).map(|i| specrt::state::kv_bytes_per_token_at(&all_own, i)).sum();
    assert!(per_token < full);
}

#[test]
fn the_26b_a4b_maps_a_dense_mlp_beside_its_experts() {
    let spec = Spec::from_hf_config_serving(&popular("google__gemma-4-26B-A4B-it")).unwrap();
    let m = spec.moe.as_ref().unwrap();
    assert_eq!((m.experts, m.top_k(), m.expert_intermediate, spec.dims.intermediate), (128, 8, 704, 2112));
    assert!(m.parallel_dense && m.norm_topk && !m.shared_gate && m.shared_intermediate == 0);
    assert!(m.dense_layers.is_empty());
    assert!((m.router_norm_scale - 1.0 / 2816f64.sqrt()).abs() < 1e-12);
    assert_eq!(m.router_norm.as_deref(), Some("mlp.router.scale"));
    assert_eq!(m.expert_scale.as_deref(), Some("mlp.router.per_expert_scale"));
    assert_eq!(m.input_norm.as_deref(), Some("pre_feedforward_layernorm_2.weight"));
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        for block in [Block::Sliding, Block::Full] {
            let ops = plan.layer_of(block, true);
            let route = ops.iter().position(|o| *o == Op::MoeRoute).unwrap();
            let down = ops.iter().position(|o| matches!(o, Op::Linear { weight: Weight::Down, .. })).unwrap();
            assert!(down < route, "{block:?}: the dense mlp runs before the router");
            let adds: Vec<(NormKind, Buf, Buf)> = ops.iter().filter_map(|o| match *o { Op::NormAdd { which, from, into } => Some((which, from, into)), _ => None }).collect();
            assert_eq!(
                adds,
                vec![
                    (NormKind::PostAttention, Buf::Branch, Buf::X),
                    (NormKind::PostDense, Buf::Branch, Buf::Merged),
                    (NormKind::PostExperts, Buf::Branch, Buf::Merged),
                    (NormKind::PostMlp, Buf::Merged, Buf::X),
                ]
            );
            assert!(ops.contains(&Op::MoeCombineInto { into: Buf::Branch }));
            assert!(!ops.contains(&Op::MoeCombine));
        }
        specrt::signature::validate_plan(&plan, &spec).unwrap();
    }
    specrt::codegen::Module::generate(&spec, 256).unwrap();
}

#[test]
fn the_26b_quantised_copies_are_held_back_only_by_their_quant_tile_sizes() {
    let fp8 = Spec::from_hf_config_serving(&popular("RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic")).unwrap_err().to_string();
    assert!(fp8.contains("multiples of 128"), "{fp8}");
    let nvfp4 = Spec::from_hf_config_serving(&popular("nvidia__Gemma-4-26B-A4B-NVFP4")).unwrap_err().to_string();
    assert!(nvfp4.contains("hidden to be a multiple of 256"), "{nvfp4}");
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
    tiny_named("tiny-gemma4")
}

fn tiny_named(name: &str) -> Spec {
    let dir = fixture(name);
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = name.into();
    spec.repo = dir.display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

#[test]
fn the_tiny_moe_fixture_routes_every_layer_beside_its_dense_mlp() {
    let spec = tiny_named("tiny-gemma4-moe");
    let m = spec.moe.as_ref().unwrap();
    assert_eq!((m.experts, m.top_k(), m.expert_intermediate), (4, 2, 64));
    assert!(spec.parallel_moe() && (0..6).all(|l| m.is_sparse(l)));
    let module = specrt::codegen::Module::generate(&spec, 256).unwrap();
    for k in ["moe_route_logits(const float* x, const float* ln, const float* xn_ln,", "const float* bias, const float* expert_scale,", "moe_mma_up_b16", "moe_up_bf16"] {
        assert!(module.source.contains(k), "{k}");
    }
    let target = kir::Target::sm_120();
    for phase in [Phase::Prefill, Phase::Decode] {
        specrt::compile::compile(&spec, Plan::build(&spec, phase, true), &target, &specrt::compile::Options::planes(256)).unwrap();
    }
}

#[test]
#[ignore = "needs nvrtc, no GPU"]
fn the_moe_modules_compile_to_ptx() {
    let big = Spec::from_hf_config_serving(&popular("google__gemma-4-26B-A4B-it")).unwrap();
    for spec in [tiny_named("tiny-gemma4-moe"), big] {
        let m = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
        assert!(m.source.contains("const float* expert_scale,"));
        let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120a"), options: vec!["--std=c++17".to_string()], ..Default::default() };
        cudarc::nvrtc::compile_ptx_with_opts(m.source.as_str(), opts).unwrap_or_else(|e| panic!("{}: {e:?}", spec.name));
    }
}

#[test]
fn the_tiny_moe_layer_loads_router_scales_and_fused_experts_from_the_checkpoint() {
    let spec = tiny_named("tiny-gemma4-moe");
    let ck = specrt::weights::Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    let layer = ck.layer(&spec, 1).unwrap();
    let moe = layer.moe.as_ref().unwrap();
    let n = moe.norms.as_ref().unwrap();
    let raw = |name: &str| -> Vec<f32> {
        let st = safetensors::SafeTensors::deserialize(Box::leak(std::fs::read(fixture("tiny-gemma4-moe").join("model.safetensors")).unwrap().into_boxed_slice())).unwrap();
        st.tensor(name).unwrap().data().as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect()
    };
    let scale = raw("model.layers.1.router.scale");
    for (a, b) in n.router.iter().zip(&scale) {
        assert!((a - b / 128f32.sqrt()).abs() < 1e-6);
    }
    assert_eq!(n.expert_scale, raw("model.layers.1.router.per_expert_scale"));
    assert_eq!(n.input, raw("model.layers.1.pre_feedforward_layernorm_2.weight"));
    assert_eq!(n.post_experts, raw("model.layers.1.post_feedforward_layernorm_2.weight"));
    let fused = raw("model.layers.1.experts.gate_up_proj");
    let specrt::weights::Stacked::Bf16(up) = &moe.up else { panic!("bf16 experts") };
    let (h, i) = (128, 64);
    let up_e2_r3 = &fused[(2 * 2 * i + i + 3) * h..(2 * 2 * i + i + 4) * h];
    let got: Vec<f32> = up[(2 * i + 3) * h..(2 * i + 4) * h].iter().map(|b| f32::from_bits((*b as u32) << 16)).collect();
    assert_eq!(got, up_e2_r3);
    assert!(matches!(layer.gate_up, specrt::weights::Matrix::Bf16(ref w) if w.len() == 2 * 256 * h));
}

fn tiny_e() -> Spec {
    let dir = fixture("tiny-gemma4-e");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(&unrefused(), &json).unwrap();
    spec.name = "tiny-gemma4-e".into();
    spec.repo = dir.display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

#[test]
fn the_tiny_e_fixture_has_a_spec_that_builds_every_kernel() {
    let spec = tiny_e();
    specrt::codegen::Module::generate(&spec, 256).unwrap();
    let states: Vec<(&str, usize)> = spec.state.iter().map(|s| (s.name.as_str(), s.layer_count(&spec))).collect();
    assert_eq!(states, vec![("k_s", 3), ("v_s", 3), ("k_g", 1), ("v_g", 1)]);
    assert_eq!(spec.layers.shared_tail, 2);
    assert!(spec.mlp.wide_shared_tail);
    assert_eq!(spec.per_layer().unwrap().dim, 16);
    assert_eq!((param(&spec, "kvh_s"), param(&spec, "kvh_g")), (1.0, 1.0));
    assert_eq!(spec.max_intermediate(), 512);
}

fn e_reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR_E").map(PathBuf::from).expect("SPECRT_REF_DIR_E holds scripts/tiny_gemma4.py output for the E fixture");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tiny-gemma4-e.ref.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join("tiny-gemma4-e.ref.bin")).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py E output in SPECRT_REF_DIR_E"]
fn gemma4_e_reading_the_wrong_kv_layer_or_layer_slice_fails_the_match() {
    let (tokens, _, _, _) = e_reference();
    let mut spec = tiny_e();
    spec.layers.shared_tail = 0;
    let red = match specrt::runtime::Runtime::load(&spec, 256).and_then(|mut rt| rt.prompt_logits(&tokens)) {
        Ok(_) => panic!("with no shared layers every layer owned a cache and the run still matched transformers"),
        Err(e) => format!("{e:#}"),
    };
    eprintln!("red kv: {red}");
    assert!(red.contains("k_proj") || red.contains("state_write"), "the red run failed for an unrelated reason: {red}");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py E output in SPECRT_REF_DIR_E"]
fn gemma4_e_logits_and_greedy_match_transformers() {
    let (tokens, greedy, vocab, want) = e_reference();
    let mut rt = specrt::runtime::Runtime::load(&tiny_e(), 256).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("e max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    for r in 0..tokens.len() {
        let row = |v: &[f32]| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(row(&ours), row(&want), "row {r}");
    }
    let got = rt.generate(&tokens, greedy.len(), specrt::runtime::Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, greedy);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py E output in SPECRT_REF_DIR_E"]
fn gemma4_e_layer_states_match_transformers() {
    let (tokens, _, _, _) = e_reference();
    let dir = std::env::var_os("SPECRT_REF_DIR_E").map(PathBuf::from).unwrap();
    let bytes = std::fs::read(dir.join("tiny-gemma4-e.layers.bin")).unwrap();
    let want: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut rt = specrt::runtime::Runtime::load(&tiny_e(), 256).unwrap();
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * 128;
    for layer in 0..6 {
        let (a, b) = (&trace.layers[layer * per..(layer + 1) * per], &want[layer * per..(layer + 1) * per]);
        let max_abs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let size = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
        eprintln!("e layer {layer} max_abs {max_abs} of {size}");
        assert!(max_abs < 2e-2 * size.max(1.0), "layer {layer} max abs diff {max_abs}");
    }
}

#[test]
fn e_kv_cache_is_smaller_by_the_shared_layers() {
    let spec = tiny_e();
    let owners: u64 = (0..spec.attention_layers()).map(|i| specrt::state::kv_bytes_per_token_at(&spec, i)).sum();
    let mut all = spec.clone();
    all.layers.shared_tail = 0;
    let full: u64 = (0..all.attention_layers()).map(|i| specrt::state::kv_bytes_per_token_at(&all, i)).sum();
    assert_eq!((owners, full), (3 * 32 * 2 + 3 * 32 * 2 + 64 * 2 + 64 * 2, 4 * 32 * 2 + 4 * 32 * 2 + 2 * 64 * 2 + 2 * 64 * 2));
    assert!(owners < full);
}

fn reference(name: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).expect("SPECRT_REF_DIR holds scripts/tiny_gemma4.py output");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_layer_states_match_transformers() {
    layer_states_match("tiny-gemma4");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_logits_and_greedy_match_transformers() {
    logits_and_greedy_match("tiny-gemma4");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_served_two_slots_match_transformers_greedy() {
    served_two_slots_match("tiny-gemma4");
}

#[test]
#[ignore = "needs a CUDA GPU and G4_MOE=1 scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_moe_layer_states_match_transformers() {
    layer_states_match("tiny-gemma4-moe");
}

#[test]
#[ignore = "needs a CUDA GPU and G4_MOE=1 scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_moe_logits_and_greedy_match_transformers() {
    logits_and_greedy_match("tiny-gemma4-moe");
}

#[test]
#[ignore = "needs a CUDA GPU and G4_MOE=1 scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_moe_served_two_slots_match_transformers_greedy() {
    served_two_slots_match("tiny-gemma4-moe");
}

#[test]
#[ignore = "needs a CUDA GPU and G4_MOE=1 scripts/tiny_gemma4.py output in SPECRT_REF_DIR"]
fn gemma4_moe_router_fed_the_normed_input_misses_transformers() {
    let name = "tiny-gemma4-moe";
    let (tokens, _, vocab, want) = reference(name);
    let mut spec = tiny_named(name);
    let m = spec.moe.as_mut().unwrap();
    m.router_norm = m.input_norm.clone();
    m.router_norm_scale = 1.0;
    let mut rt = specrt::runtime::Runtime::load(&spec, 256).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let row = |v: &[f32], r: usize| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let missed = (0..tokens.len()).filter(|&r| row(&ours, r) != row(&want, r)).count();
    eprintln!("{name} router on the normed input: max_abs {max_abs}, {missed} of {} argmax rows differ", tokens.len());
    assert!(max_abs >= 5e-2, "the logit bound must catch a router that reads the normed input, max abs {max_abs}");
}

fn layer_states_match(name: &str) {
    let (tokens, _, _, _) = reference(name);
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap();
    let bytes = std::fs::read(dir.join(format!("{name}.layers.bin"))).unwrap();
    let want: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut rt = specrt::runtime::Runtime::load(&tiny_named(name), 256).unwrap();
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * 128;
    for layer in 0..6 {
        let (a, b) = (&trace.layers[layer * per..(layer + 1) * per], &want[layer * per..(layer + 1) * per]);
        let max_abs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let size = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
        eprintln!("{name} layer {layer} max_abs {max_abs} of {size}");
        assert!(max_abs < 2e-2 * size.max(1.0), "{name} layer {layer} max abs diff {max_abs}");
    }
}

fn logits_and_greedy_match(name: &str) {
    let (tokens, greedy, vocab, want) = reference(name);
    let mut rt = specrt::runtime::Runtime::load(&tiny_named(name), 256).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{name} max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    for r in 0..tokens.len() {
        let row = |v: &[f32]| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(row(&ours), row(&want), "row {r}");
    }
    let got = rt.generate(&tokens, greedy.len(), specrt::runtime::Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, greedy);
}

fn served_two_slots_match(name: &str) {
    use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
    let (tokens, greedy, _, _) = reference(name);
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    let layout = Layout { max_ctx, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&tiny_named(name), layout).unwrap();
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
