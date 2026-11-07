use specrt::spec::{Checked, EMBEDDED, Quant, SPEC_FORMAT, Spec, check};
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
    let json =
        std::fs::read_to_string(root().join("tests/fixtures/qwen3-0.6b.config.json")).unwrap();
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

fn matches_hf(spec_file: &str, fixture: &str) {
    let spec = Spec::load(&root().join("specs").join(spec_file)).unwrap();
    let mut from_hf = derived(fixture);
    from_hf.name = spec.name.clone();
    from_hf.repo = spec.repo.clone();
    assert_eq!(spec, from_hf);
}

#[test]
fn llama32_spec_matches_hf_config() {
    matches_hf("llama-3.2-1b.toml", "llama-3.2-1b.config.json");
}

#[test]
fn mistral_f16_spec_matches_hf_config() {
    matches_hf("tinymistral-248m.toml", "tinymistral-248m.config.json");
}

#[test]
fn gemma_and_phi3_stay_unsupported_with_the_model_type_in_the_error() {
    for model_type in ["gemma2", "gemma3_text", "phi3"] {
        let json =
            fixture_with("\"model_type\": \"qwen3\"", &format!("\"model_type\": \"{model_type}\""));
        let e = Spec::from_hf_config(&json).unwrap_err().to_string();
        assert_eq!(e, format!("model_type {model_type} is not supported"));
    }
}

#[test]
fn the_embedded_snapshot_equals_the_specs_folder() {
    let mut on_disk: Vec<(String, String)> = std::fs::read_dir(root().join("specs"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".toml"))
        .map(|n| (n.clone(), std::fs::read_to_string(root().join("specs").join(&n)).unwrap()))
        .collect();
    on_disk.sort();
    let embedded: Vec<(String, String)> =
        EMBEDDED.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect();
    assert_eq!(embedded, on_disk);
    assert!(embedded.len() >= 7);
    for (name, text) in EMBEDDED {
        assert!(Spec::parse(text).is_ok(), "{name}");
    }
}

#[test]
fn min_engine_defaults_to_zero_and_a_newer_one_is_refused_by_name() {
    let text = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    assert_eq!(Spec::parse(&text).unwrap().min_engine, 0);
    let current = format!("min_engine = {SPEC_FORMAT}\n{text}");
    assert_eq!(Spec::parse(&current).unwrap().min_engine, SPEC_FORMAT);
    let newer = format!("min_engine = {}\n{}", SPEC_FORMAT + 1, text.replace("[mlp]", "[future]\nthing = 1\n[mlp]"));
    assert_eq!(
        check(&newer).unwrap(),
        Checked::NeedsNewer { name: "qwen3-0.6b".into(), min_engine: SPEC_FORMAT + 1 }
    );
    let e = Spec::parse(&newer).unwrap_err().to_string();
    assert!(e.contains("needs a newer inference-bench"), "{e}");
    assert!(check("min_engine = -1\nname = \"x\"").is_err());
}
