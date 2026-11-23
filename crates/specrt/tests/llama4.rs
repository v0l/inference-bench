use specrt::family::Registry;
use specrt::spec::Spec;
use std::path::Path;

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn registry() -> Registry {
    Registry::embedded().clone()
}

fn scout() -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("llama4-scout.config.json")).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("vision_config");
    v
}

fn spec_of(config: &serde_json::Value) -> Spec {
    Spec::from_hf_config_serving_in(&registry(), &config.to_string()).unwrap()
}

#[test]
fn dense_layers_take_the_mlp_width_and_moe_layers_the_expert_width() {
    let mut config = scout();
    config["text_config"]["interleave_moe_layer_step"] = 2.into();
    let spec = spec_of(&config);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((spec.dims.intermediate, moe.expert_intermediate, moe.shared_intermediate), (16384, 8192, 8192));
    assert_eq!((moe.experts, moe.top_k()), (16, 1));
    assert!(!moe.is_sparse(0) && moe.is_sparse(1) && !moe.is_sparse(2) && moe.is_sparse(3));
}

#[test]
fn rope_pairs_and_weightless_qk_norm_come_from_the_family() {
    let spec = spec_of(&scout());
    assert_eq!(spec.rope.style, specrt::spec::RopeStyle::Pairs);
    assert!(spec.attention.qk_norm && spec.attention.qk_norm_unit);
    assert_eq!(spec.rope.nope_layers, (0..48).filter(|l| l % 4 == 3).collect::<Vec<_>>());
    let shape = specrt::codegen::attention_shape(&spec, 1, 8192);
    assert!(shape.rope.interleaved && shape.qk_norm_unit);
    let mut off = scout();
    off["text_config"]["use_qk_norm"] = false.into();
    assert!(!spec_of(&off).attention.qk_norm);
    let toml = spec.to_toml().unwrap();
    assert_eq!(Spec::parse(&toml).unwrap(), spec);
}

#[test]
fn top1_sigmoid_routing_scales_the_expert_input_and_the_shared_expert_has_no_gate() {
    let spec = spec_of(&scout());
    let moe = spec.moe.as_ref().unwrap();
    assert!(moe.input_scaled && !moe.norm_topk && !moe.shared_gate);
    assert_eq!(moe.router, specrt::spec::RouterAct::Sigmoid);
    assert_eq!((moe.top_k(), moe.shared_slots(), moe.pairs()), (1, 1, 2));
    let target = kir::Target::sm_120();
    let source = specrt::codegen::Module::generate_kv(&spec, 256, 1, &target).unwrap().source;
    assert!(source.contains("constexpr bool MOE_INPUT_SCALED = true;") && source.contains("moe_scale_inputs("));
    let plain = specrt::codegen::moe_kernels(&Spec::from_hf_config_serving_in(Registry::embedded(), &std::fs::read_to_string(fixtures().join("glm4.5-air.config.json")).unwrap()).unwrap(), &target).unwrap();
    assert!(!plain.is_empty() && plain.iter().all(|k| !k.shape.input_scaled));
}

#[test]
fn glm_air_route_logits_stage_through_dynamic_shared_memory() {
    let spec = Spec::from_hf_config_serving_in(Registry::embedded(), &std::fs::read_to_string(fixtures().join("glm4.5-air.config.json")).unwrap()).unwrap();
    let target = kir::Target::sm_120();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &target).unwrap();
    assert!(module.source.contains("extern __shared__ __align__(16) float xs[];"));
    assert_eq!(module.smem["moe_route_logits"], 4 * 4096 * 4);
    assert!(module.smem["moe_route_logits"] <= target.smem_limit);
    let mut small = target;
    small.smem_limit = 48 * 1024;
    let err = specrt::codegen::moe_kernels(&spec, &small).unwrap_err().to_string();
    assert!(err.contains("moe_route_logits needs"), "{err}");
}

#[test]
fn chunked_attention_nope_and_temperature_tuning_come_from_the_config() {
    let spec = spec_of(&scout());
    use specrt::spec::Block;
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Sliding, Block::Sliding, Block::Full]);
    assert_eq!(spec.layers.window, Some(8192));
    assert_eq!((spec.attention.params["attn_scale"], spec.attention.params["floor_scale"], spec.attention.params["tuning"]), (0.1, 8192.0, 1.0));
    assert!(spec.attention.node.iter().any(|n| n.kernel.as_deref() == Some("attn_chunked")));
    assert_eq!(spec.tensors.get(".feed_forward.").map(String::as_str), Some(".mlp."));
}

#[test]
fn the_tiny_checkpoint_reads_as_a_transposed_fused_expert_model() {
    let json = std::fs::read_to_string(fixtures().join("tiny-llama4/config.json")).unwrap();
    let spec = Spec::from_hf_config_serving_in(&registry(), &json).unwrap();
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((spec.dims.intermediate, moe.expert_intermediate, moe.experts), (256, 128, 8));
    assert!(!moe.is_sparse(0) && moe.is_sparse(1) && !moe.is_sparse(2) && moe.is_sparse(3));
    let ck = specrt::weights::Checkpoint::open(&fixtures().join("tiny-llama4")).unwrap().renamed(&spec.tensors);
    assert_eq!(ck.shape_of("model.layers.1.mlp.experts.gate_up_proj").unwrap(), vec![8, 128, 256]);
    assert_eq!(ck.shape_of("model.layers.0.mlp.gate_proj.weight").unwrap(), vec![256, 128]);
    assert!(ck.has("model.layers.1.mlp.router.weight") && ck.has("model.layers.1.mlp.shared_expert.up_proj.weight"));
    assert!(!ck.has("model.layers.0.self_attn.q_norm.weight"));
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::path::PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tiny-llama4.ref.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join("tiny-llama4.ref.bin")).unwrap();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn the_tiny_checkpoint_matches_transformers_logits_and_greedy_tokens() {
    let json = std::fs::read_to_string(fixtures().join("tiny-llama4/config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(&registry(), &json).unwrap();
    spec.repo = fixtures().join("tiny-llama4").display().to_string();
    spec.mlp.act_quant = specrt::spec::ActQuant::None;
    let (tokens, greedy, vocab, want) = reference();
    let mut rt = specrt::runtime::Runtime::load_batched(&spec, 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("tiny-llama4 prefill max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    for r in 0..tokens.len() {
        let span = r * vocab..(r + 1) * vocab;
        assert_eq!(top(&ours[span.clone()]), top(&want[span]), "row {r}");
    }
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "decode greedy");
}

#[test]
fn an_explicit_moe_layer_list_overrides_the_step() {
    let mut config = scout();
    config["text_config"]["moe_layers"] = serde_json::json!([1, 2, 5]);
    let spec = spec_of(&config);
    let moe = spec.moe.as_ref().unwrap();
    let sparse: Vec<usize> = (0..48).filter(|l| moe.is_sparse(*l)).collect();
    assert_eq!(sparse, vec![1, 2, 5]);
}
