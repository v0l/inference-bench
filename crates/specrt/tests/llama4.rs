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
fn an_explicit_moe_layer_list_overrides_the_step() {
    let mut config = scout();
    config["text_config"]["moe_layers"] = serde_json::json!([1, 2, 5]);
    let spec = spec_of(&config);
    let moe = spec.moe.as_ref().unwrap();
    let sparse: Vec<usize> = (0..48).filter(|l| moe.is_sparse(*l)).collect();
    assert_eq!(sparse, vec![1, 2, 5]);
}
