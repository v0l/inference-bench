use specrt::family::Registry;
use specrt::spec::{EMBEDDED, Spec};
use std::path::Path;

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn registry() -> Registry {
    let family = std::fs::read_to_string(fixtures().join("family-llama4.toml")).unwrap();
    Registry::from_texts(EMBEDDED.iter().map(|(_, t)| *t).chain([family.as_str()])).unwrap()
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
fn an_explicit_moe_layer_list_overrides_the_step() {
    let mut config = scout();
    config["text_config"]["moe_layers"] = serde_json::json!([1, 2, 5]);
    let spec = spec_of(&config);
    let moe = spec.moe.as_ref().unwrap();
    let sparse: Vec<usize> = (0..48).filter(|l| moe.is_sparse(*l)).collect();
    assert_eq!(sparse, vec![1, 2, 5]);
}
