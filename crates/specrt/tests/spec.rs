use specrt::spec::{Block, Checked, EMBEDDED, LinearKind, Quant, SPEC_FORMAT, Spec, check};
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
fn fp8_block_checkpoints_derive_the_fp8_quant_and_others_are_refused() {
    let spec = derived("qwen3.8-27b.config.json");
    assert_eq!(spec.quant, Quant::Fp8Block128);
    assert!(spec.to_toml().unwrap().contains("quant = \"fp8_block128\""));
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b.config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["quantization_config"]["weight_block_size"] = serde_json::json!([1, 128]);
    assert!(Spec::from_hf_config(&v.to_string()).is_err());
    v["quantization_config"] = serde_json::json!({"quant_method": "compressed-tensors"});
    assert!(Spec::from_hf_config(&v.to_string()).is_err());
    let text = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml"))
        .unwrap()
        .replace("quant = \"bf16\"", "quant = \"fp8_block128\"")
        .replace("intermediate = 3072", "intermediate = 3000");
    assert!(Spec::parse(&text).is_err(), "fp8 blocks need dims that are multiples of 128");
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
fn qwen38_27b_config_parses_into_a_hybrid_spec_and_round_trips() {
    let mut spec = derived("qwen3.8-27b.config.json");
    assert_eq!(spec.layers.pattern, [Block::Linear, Block::Linear, Block::Linear, Block::Full]);
    assert_eq!(spec.layers.count, 64);
    assert_eq!(spec.linear_layers(), 48);
    assert_eq!(spec.attention_layers(), 16);
    let la = spec.linear_attention.clone().unwrap();
    assert_eq!(la.kind, LinearKind::GatedDeltaNet);
    assert_eq!((la.key_heads, la.value_heads, la.key_head_dim, la.value_head_dim), (16, 48, 128, 128));
    assert_eq!(la.conv_kernel, 4);
    assert_eq!(la.conv_dim(), 10240);
    assert_eq!(spec.rotary_dim(), 64);
    assert!(spec.attention.output_gate && spec.attention.qk_norm && spec.norm.unit_offset);
    assert_eq!(spec.qkv_rows(), 2 * 6144 + 2 * 1024);
    assert_eq!(Spec::parse(&spec.to_toml().unwrap()).unwrap(), spec);
    let file = Spec::load(&root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    spec.name = file.name.clone();
    spec.repo = file.repo.clone();
    spec.min_engine = file.min_engine;
    assert_eq!(file, spec);
    assert_eq!(file.min_engine, 3);
}

#[test]
fn compressed_tensors_nvfp4_checkpoint_derives_fp8_channel_with_nvfp4_mlps_until_56() {
    let mut spec = derived("qwen3.8-27b-nvfp4.config.json");
    assert_eq!(spec.quant, Quant::Fp8Channel);
    assert_eq!((spec.mlp.quant, spec.mlp.quant_until), (Some(Quant::Nvfp4G16), Some(56)));
    assert_eq!(spec.mlp_quant(0), Quant::Nvfp4G16);
    assert_eq!(spec.mlp_quant(55), Quant::Nvfp4G16);
    assert_eq!(spec.mlp_quant(56), Quant::Fp8Channel);
    assert_eq!(spec.mlp_quant(63), Quant::Fp8Channel);
    assert_eq!(Spec::parse(&spec.to_toml().unwrap()).unwrap(), spec);
    let file = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    spec.name = file.name.clone();
    spec.repo = file.repo.clone();
    spec.min_engine = file.min_engine;
    assert_eq!(file, spec);
    assert_eq!(file.min_engine, SPEC_FORMAT);
}

fn nvfp4_config() -> serde_json::Value {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.config.json")).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[test]
fn compressed_tensors_groups_that_are_not_fp8_channel_or_nvfp4_g16_are_refused() {
    let mut v = nvfp4_config();
    v["quantization_config"]["config_groups"]["group_1"]["weights"]["group_size"] = serde_json::json!(32);
    assert!(Spec::from_hf_config(&v.to_string()).unwrap_err().to_string().contains("group_1"));
    let mut v = nvfp4_config();
    v["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] = serde_json::json!("tensor");
    assert!(Spec::from_hf_config(&v.to_string()).is_err());
    let mut v = nvfp4_config();
    let targets = v["quantization_config"]["config_groups"]["group_1"]["targets"].as_array_mut().unwrap();
    targets.retain(|t| t.as_str().unwrap() != "model.language_model.layers.7.mlp.down_proj");
    targets.push(serde_json::json!("model.language_model.layers.7.self_attn.q_proj"));
    assert!(Spec::from_hf_config(&v.to_string()).is_err(), "an nvfp4 attention projection");
}

#[test]
fn mlp_quant_needs_fp8_channel_projections_and_a_layer_inside_the_count() {
    let text = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    assert!(Spec::parse(&text).is_ok());
    assert!(Spec::parse(&text.replace("quant = \"fp8_channel\"", "quant = \"fp8_block128\"")).is_err());
    assert!(Spec::parse(&text.replace("quant = \"fp8_channel\"", "quant = \"nvfp4_g16\"")).is_err());
    assert!(Spec::parse(&text.replace("quant_until = 56", "quant_until = 64")).is_err());
    assert!(Spec::parse(&text.replace("quant = \"nvfp4_g16\"\n", "")).is_err(), "quant_until alone");
    let all = Spec::parse(&text.replace("quant_until = 56\n", "")).unwrap();
    assert_eq!(all.mlp_quant(63), Quant::Nvfp4G16);
}

fn hybrid_text() -> String {
    std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap()
}

#[test]
fn hybrid_spec_validation_rejects_inconsistent_linear_attention() {
    let text = hybrid_text();
    assert!(Spec::parse(&text).is_ok());
    let bad = [
        ("value_heads = 48", "value_heads = 40"),
        ("value_head_dim = 128", "value_head_dim = 64"),
        ("conv_kernel = 4", "conv_kernel = 1"),
        ("count = 64", "count = 66"),
        ("rotary_dim = 64", "rotary_dim = 300"),
        ("pattern = [\"linear\", \"linear\", \"linear\", \"full\"]", "pattern = [\"linear\", \"dense\"]"),
        ("pattern = [\"linear\", \"linear\", \"linear\", \"full\"]", "pattern = [\"full\"]"),
        ("kind = \"gated_delta_net\"", "kind = \"mamba2\""),
    ];
    for (from, to) in bad {
        assert!(text.contains(from), "{from}");
        assert!(Spec::parse(&text.replace(from, to)).is_err(), "{to}");
    }
    let cut = text.split("[linear_attention]").next().unwrap();
    assert!(Spec::parse(cut).is_err(), "a linear layer needs [linear_attention]");
}

#[test]
fn layer_types_reduce_to_their_shortest_period() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b.config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let text = &mut v["text_config"];
    text["num_hidden_layers"] = 3.into();
    text["layer_types"] = serde_json::json!(["linear_attention", "linear_attention", "full_attention"]);
    let spec = Spec::from_hf_config(&v.to_string()).unwrap();
    assert_eq!(spec.layers.pattern, [Block::Linear, Block::Linear, Block::Full]);
    v["text_config"]["layer_types"] = serde_json::json!(["linear_attention", "sliding_attention", "full_attention"]);
    assert!(Spec::from_hf_config(&v.to_string()).is_err());
}

#[test]
fn dense_specs_serialize_without_the_hybrid_fields() {
    let text = derived("qwen3-0.6b.config.json").to_toml().unwrap();
    for field in ["unit_offset", "output_gate", "rotary_dim", "linear_attention"] {
        assert!(!text.contains(field), "{field} in {text}");
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
