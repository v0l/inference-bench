use specrt::spec::{Quant, Spec};
use std::path::Path;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn derived(fixture: &str) -> Spec {
    let json = std::fs::read_to_string(root().join("tests/fixtures").join(fixture)).unwrap();
    Spec::from_hf_config(&json).unwrap()
}

#[test]
fn qwen3_spec_matches_hf_config() {
    let spec = Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap();
    let mut from_hf = derived("qwen3-0.6b.config.json");
    from_hf.name = spec.name.clone();
    from_hf.repo = spec.repo.clone();
    assert_eq!(spec, from_hf);
    assert!(spec.attention.qk_norm);
    assert!(!spec.attention.qkv_bias);
    assert_eq!(spec.attention.head_dim, 128);
}

#[test]
fn qwen25_spec_matches_hf_config() {
    let spec = Spec::load(&root().join("specs/qwen2.5-0.5b.toml")).unwrap();
    let mut from_hf = derived("qwen2.5-0.5b.config.json");
    from_hf.name = spec.name.clone();
    from_hf.repo = spec.repo.clone();
    assert_eq!(spec, from_hf);
    assert!(!spec.attention.qk_norm);
    assert!(spec.attention.qkv_bias);
    assert_eq!(spec.attention.head_dim, 64);
}

#[test]
fn spec_round_trips_through_toml() {
    let spec = derived("qwen3-0.6b.config.json");
    let text = spec.to_toml().unwrap();
    assert_eq!(Spec::parse(&text).unwrap(), spec);
}

#[test]
fn quant_override_parses() {
    let text = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml"))
        .unwrap()
        .replace("quant = \"bf16\"", "quant = \"q8\"");
    assert_eq!(Spec::parse(&text).unwrap().quant, Quant::Q8);
}

#[test]
fn rejects_unsupported_rope_scaling() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3-0.6b.config.json"))
        .unwrap()
        .replace(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"type\": \"yarn\", \"factor\": 4.0}",
        );
    assert!(Spec::from_hf_config(&json).is_err());
}

#[test]
fn rejects_non_dividing_gqa_groups() {
    let text = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml"))
        .unwrap()
        .replace("kv_heads = 8", "kv_heads = 5");
    assert!(Spec::parse(&text).is_err());
}

fn fixture_with(from: &str, to: &str) -> String {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3-0.6b.config.json"))
        .unwrap();
    assert!(json.contains(from), "{from}");
    json.replace(from, to)
}

#[test]
fn llama3_rope_scaling_is_parsed() {
    let json = fixture_with(
        "\"rope_scaling\": null",
        "\"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0, \"low_freq_factor\": 1.0, \"high_freq_factor\": 4.0, \"original_max_position_embeddings\": 8192}",
    );
    let spec = Spec::from_hf_config(&json).unwrap();
    let s = spec.rope.scaling.clone().unwrap();
    assert_eq!(s.kind, specrt::spec::RopeScalingKind::Llama3);
    assert_eq!(s.factor, 8.0);
    assert_eq!(s.original_max_positions, Some(8192.0));
    assert_eq!(Spec::parse(&spec.to_toml().unwrap()).unwrap(), spec);
}

#[test]
fn linear_rope_scaling_is_parsed() {
    let json = fixture_with(
        "\"rope_scaling\": null",
        "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 4.0}",
    );
    let s = Spec::from_hf_config(&json).unwrap().rope.scaling.unwrap();
    assert_eq!(s.kind, specrt::spec::RopeScalingKind::Linear);
}

#[test]
fn gelu_activation_and_f16_dtype_are_accepted() {
    let json = fixture_with("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu_pytorch_tanh\"")
        .replace("\"bfloat16\"", "\"float16\"");
    let spec = Spec::from_hf_config(&json).unwrap();
    assert_eq!(spec.mlp.activation, specrt::spec::Activation::Geglu);
}

#[test]
fn rejects_unknown_activation_and_dtype() {
    let json = fixture_with("\"hidden_act\": \"silu\"", "\"hidden_act\": \"relu2\"");
    assert!(Spec::from_hf_config(&json).is_err());
    let json = fixture_with("\"bfloat16\"", "\"int8\"");
    assert!(Spec::from_hf_config(&json).is_err());
}
