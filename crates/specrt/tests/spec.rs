use specrt::spec::{Activation, Block, Checked, EMBEDDED, LinearKind, Quant, SPEC_FORMAT, Spec, check};
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
            "\"rope_scaling\": {\"type\": \"dynamic\", \"factor\": 4.0}",
        );
    assert!(Spec::from_hf_config(&json).is_err());
}

#[test]
fn yarn_without_the_original_context_is_refused() {
    let json = fixture_with("\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 4.0}");
    assert!(format!("{:#}", Spec::from_hf_config(&json).unwrap_err()).contains("original_max_position_embeddings"));
}

#[test]
fn yarn_rope_scaling_is_parsed_with_the_transformers_defaults() {
    let json = fixture_with(
        "\"rope_scaling\": null",
        "\"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 4.0, \"original_max_position_embeddings\": 262144}",
    );
    let spec = Spec::from_hf_config(&json).unwrap();
    let s = spec.rope.scaling.clone().unwrap();
    assert_eq!(s.kind, specrt::spec::RopeScalingKind::Yarn);
    assert_eq!((s.factor, s.original_max_positions), (4.0, Some(262144.0)));
    let y = s.yarn.clone().unwrap();
    assert_eq!((y.beta_fast, y.beta_slow, y.truncate), (32.0, 1.0, true));
    assert!((y.attention_factor - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
    assert_eq!(Spec::parse(&spec.to_toml().unwrap()).unwrap(), spec);
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
    assert_eq!(s.kind, specrt::spec::RopeScalingKind::Banded);
    assert_eq!(s.factor, 8.0);
    assert_eq!(s.original_max_positions, Some(8192.0));
    assert_eq!(Spec::parse(&spec.to_toml().unwrap()).unwrap(), spec);
}

#[test]
fn an_old_llama3_rope_scaling_kind_in_a_spec_still_loads_as_banded() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(root.join("specs/llama-3.2-1b.toml")).unwrap();
    assert!(text.contains("kind = \"banded\""));
    let old = text.replace("kind = \"banded\"", "kind = \"llama3\"");
    assert_eq!(Spec::parse(&old).unwrap(), Spec::parse(&text).unwrap());
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
    from_hf.plain_activations = false;
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
fn gemma2_and_unknown_types_stay_unsupported_with_the_model_type_in_the_error() {
    for model_type in ["gemma2", "not_a_model"] {
        let json =
            fixture_with("\"model_type\": \"qwen3\"", &format!("\"model_type\": \"{model_type}\""));
        let e = Spec::from_hf_config(&json).unwrap_err().to_string();
        assert_eq!(e, format!("model_type {model_type} is not supported"));
    }
}

#[test]
fn gemma3_is_supported() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/tiny-gemma3/config.json")).unwrap();
    let spec = Spec::from_hf_config(&json).unwrap();
    assert!(spec.norm.sandwich);
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
        assert!(matches!(check(text), Ok(Checked::Ready(_) | Checked::Family(_) | Checked::Kernel(_))), "{name}");
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
    assert_eq!(spec.rope.mrope, Some(specrt::spec::Mrope { section: [11, 11, 10], interleaved: true }));
    let file = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-exact.toml")).unwrap();
    assert!(file.rope.mrope.is_none() && file.vision.is_none(), "toml specs are completed from the snapshot config at load, not stored");
    spec.rope.mrope = None;
    spec.vision = None;
    spec.name = file.name.clone();
    spec.repo = file.repo.clone();
    spec.min_engine = file.min_engine;
    spec.plain_activations = file.plain_activations;
    assert_eq!(file, spec);
    assert_eq!(file.min_engine, 7);
    assert_eq!(file.mtp, Some(specrt::spec::Mtp::new(1)));
    assert_eq!(file.kv_layers(), 17);
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
    assert_eq!(spec.rope.mrope, Some(specrt::spec::Mrope { section: [11, 11, 10], interleaved: true }));
    assert_eq!((spec.act_quant, spec.mlp.act_quant), (specrt::spec::ActQuant::None, specrt::spec::ActQuant::None));
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.config.json")).unwrap();
    let mut serving = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!((serving.act_quant, serving.mlp.act_quant), (specrt::spec::ActQuant::Fp8, specrt::spec::ActQuant::Fp4));
    let served = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    let file = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap();
    serving.rope.mrope = None;
    serving.vision = None;
    serving.name = served.name.clone();
    serving.repo = served.repo.clone();
    serving.min_engine = served.min_engine;
    serving.plain_activations = served.plain_activations;
    serving.runtime = served.runtime.clone();
    assert_eq!(served, serving);
    spec.rope.mrope = None;
    spec.vision = None;
    spec.name = file.name.clone();
    spec.repo = file.repo.clone();
    spec.min_engine = file.min_engine;
    spec.plain_activations = file.plain_activations;
    spec.runtime = file.runtime.clone();
    assert_eq!(file, spec);
    assert_eq!(file.min_engine, 7);
    assert_eq!(file.mtp, Some(specrt::spec::Mtp::new(1)));
}

#[test]
fn compressed_tensors_checkpoint_without_dynamic_input_activations_derives_weight_only() {
    let mut v = nvfp4_config();
    for g in ["group_0", "group_1"] {
        v["quantization_config"]["config_groups"][g]["input_activations"] = serde_json::Value::Null;
    }
    let spec = Spec::from_hf_config_serving(&v.to_string()).unwrap();
    assert_eq!(spec.act_quant, specrt::spec::ActQuant::None);
    assert_eq!(spec.mlp.act_quant, specrt::spec::ActQuant::None);
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
    let text = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap();
    assert!(Spec::parse(&text).is_ok());
    assert!(Spec::parse(&text.replace("quant = \"fp8_channel\"", "quant = \"fp8_block128\"")).is_err());
    assert!(Spec::parse(&text.replace("quant = \"fp8_channel\"", "quant = \"nvfp4_g16\"")).is_err());
    assert!(Spec::parse(&text.replace("quant_until = 56", "quant_until = 64")).is_err());
    assert!(Spec::parse(&text.replace("quant = \"nvfp4_g16\"\n", "")).is_err(), "quant_until alone");
    let all = Spec::parse(&text.replace("quant_until = 56\n", "")).unwrap();
    assert_eq!(all.mlp_quant(63), Quant::Nvfp4G16);
}

#[test]
fn head_quant_defaults_to_the_checkpoint_head_round_trips_and_rejects_other_formats() {
    let text = nvfp4_text();
    let plain = Spec::parse(&text).unwrap();
    assert_eq!((plain.head_quant, plain.lm_head_quant()), (None, Quant::Bf16));
    assert_eq!(Spec::parse(&std::fs::read_to_string(root().join("specs/qwen3-0.6b-q8.toml")).unwrap()).unwrap().lm_head_quant(), Quant::Q8);
    let with = |v: &str| text.replace("quant = \"fp8_channel\"\n", &format!("quant = \"fp8_channel\"\nhead_quant = {v:?}\n"));
    for (v, q) in [("bf16", Quant::Bf16), ("fp8_channel", Quant::Fp8Channel), ("nvfp4_g16", Quant::Nvfp4G16)] {
        let s = Spec::parse(&with(v)).unwrap();
        assert_eq!((s.head_quant, s.lm_head_quant()), (Some(q), q));
        assert!(s.to_toml().unwrap().contains(&format!("head_quant = \"{v}\"")));
        assert_eq!(Spec::parse(&s.to_toml().unwrap()).unwrap(), s);
    }
    assert!(!plain.to_toml().unwrap().contains("head_quant"));
    let e = format!("{:#}", Spec::parse(&with("fp8_block128")).unwrap_err());
    assert!(e.contains("head_quant") && e.contains("not supported"), "{e}");
    let e = format!("{:#}", Spec::parse(&with("q8")).unwrap_err());
    assert!(e.contains("head_quant"), "{e}");
    let e = format!("{:#}", Spec::parse(&with("fp4")).unwrap_err());
    assert!(e.contains("fp4"), "{e}");
    for (file, q) in [("qwen3.8-27b-nvfp4-head-fp8", Quant::Fp8Channel), ("qwen3.8-27b-nvfp4-head-nvfp4", Quant::Nvfp4G16)] {
        let mut f = Spec::load(&root().join(format!("tests/fixtures/{file}.toml"))).unwrap();
        assert_eq!(f.head_quant, Some(q));
        f.name = plain.name.clone();
        f.min_engine = plain.min_engine;
        f.head_quant = None;
        assert_eq!(f, plain, "{file} differs from the base spec only in head_quant");
    }
}

fn hybrid_text() -> String {
    std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-exact.toml")).unwrap()
}

fn nvfp4_text() -> String {
    std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap()
}

#[test]
fn act_quant_fields_default_to_none_round_trip_and_match_the_act_fixtures() {
    use specrt::spec::ActQuant;
    let plain = Spec::parse(&nvfp4_text()).unwrap();
    assert_eq!((plain.act_quant, plain.mlp.act_quant), (ActQuant::None, ActQuant::None));
    let text = plain.to_toml().unwrap();
    assert!(!text.contains("act_quant") && !text.contains("[runtime]"), "{text}");
    assert_eq!(Spec::parse(&text).unwrap(), plain);
    let act = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    assert_eq!((act.act_quant, act.mlp.act_quant), (ActQuant::Fp8, ActQuant::Fp4));
    assert_eq!(Spec::parse(&act.to_toml().unwrap()).unwrap(), act);
    let mut same = act.clone();
    same.name = plain.name.clone();
    same.min_engine = plain.min_engine;
    same.act_quant = ActQuant::None;
    same.mlp.act_quant = ActQuant::None;
    same.runtime = plain.runtime.clone();
    assert_eq!(same, plain);
    let fp8 = Spec::load(&root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    assert_eq!((fp8.act_quant, fp8.mlp.act_quant), (ActQuant::Fp8, ActQuant::None));
}

#[test]
fn act_quant_rejects_unknown_values_and_formats_the_weights_cannot_take() {
    let text = nvfp4_text();
    let top = |v: &str| text.replace("quant = \"fp8_channel\"\n", &format!("quant = \"fp8_channel\"\nact_quant = {v:?}\n"));
    let mlp = |v: &str| text.replace("quant_until = 56\n", &format!("quant_until = 56\nact_quant = {v:?}\n"));
    let e = format!("{:#}", Spec::parse(&top("int8")).unwrap_err());
    assert!(e.contains("int8") && e.contains("act_quant"), "{e}");
    let e = format!("{:#}", Spec::parse(&mlp("fp6")).unwrap_err());
    assert!(e.contains("fp6"), "{e}");
    assert!(Spec::parse(&top("fp8")).is_ok());
    assert!(Spec::parse(&mlp("fp4")).is_ok());
    assert!(Spec::parse(&top("none")).is_ok());
    let e = format!("{:#}", Spec::parse(&top("fp4")).unwrap_err());
    assert!(e.contains("[mlp] only"), "{e}");
    let e = format!("{:#}", Spec::parse(&mlp("fp8")).unwrap_err());
    assert!(e.contains("[mlp] act_quant"), "{e}");
    let bf16 = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    let e = format!("{:#}", Spec::parse(&bf16.replace("[dims]", "act_quant = \"fp8\"\n\n[dims]")).unwrap_err());
    assert!(e.contains("needs fp8 weights"), "{e}");
    let e = format!("{:#}", Spec::parse(&hybrid_text().replace("activation = \"swiglu\"", "activation = \"swiglu\"\nact_quant = \"fp4\"")).unwrap_err());
    assert!(e.contains("nvfp4_g16"), "{e}");
}

#[test]
fn offload_table_parses_round_trips_and_defaults_off() {
    let base = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    assert!(!Spec::parse(&base).unwrap().runtime.offload.enabled());
    let s = Spec::parse(&format!("{base}\n[runtime.offload]\nvram_budget = 100\nprefetch_depth = 2\npolicy = \"frequency\"\n")).unwrap();
    assert_eq!((s.runtime.offload.vram_budget, s.runtime.offload.prefetch_depth), (100, 2));
    assert_eq!(Spec::parse(&s.to_toml().unwrap()).unwrap(), s);
    assert!(Spec::parse(&format!("{base}\n[runtime.offload]\nbogus = 1\n")).is_err());
}

#[test]
fn runtime_table_defaults_round_trips_and_rejects_bad_values() {
    use specrt::spec::{RuntimeOptions, TileChoice};
    let base = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    assert_eq!(Spec::parse(&base).unwrap().runtime, RuntimeOptions::default());
    let d = RuntimeOptions::default();
    assert!(d.graphs && d.fuse && !d.gemm_fp32 && d.tile == TileChoice::Auto && d.cache_dir.is_none());
    let with = format!("{base}\n[runtime]\nbackend = \"wgpu\"\ngraphs = false\nfuse = false\ntile = \"big\"\nbig_tile = [64, 128, 2, 4]\nfp4_tile = [64, 256, 2, 4]\nfp4_stages = 4\nprefill_chunk = 4096\nunsplit = [\"lm_head\"]\ngemm_fp32 = true\ncache_dir = \"/tmp/x\"\n");
    let s = Spec::parse(&with).unwrap();
    assert_eq!(
        s.runtime,
        RuntimeOptions {
            backend: specrt::spec::BackendChoice::Wgpu,
            graphs: false,
            fuse: false,
            tile: TileChoice::Big,
            big_tile: Some([64, 128, 2, 4]),
            fp4_tile: Some([64, 256, 2, 4]),
            fp4_stages: Some(4),
            prefill_chunk: Some(4096),
            unsplit: vec!["lm_head".into()],
            gemm_fp32: true,
            cache_dir: Some("/tmp/x".into()),
            ..RuntimeOptions::default()
        }
    );
    assert_eq!(Spec::parse(&s.to_toml().unwrap()).unwrap(), s);
    let partial = Spec::parse(&format!("{base}\n[runtime]\nfuse = false\n")).unwrap();
    assert_eq!(partial.runtime, RuntimeOptions { fuse: false, ..RuntimeOptions::default() });
    let bad = |extra: &str| format!("{:#}", Spec::parse(&format!("{base}\n[runtime]\n{extra}\n")).unwrap_err());
    assert!(bad("tile = \"huge\"").contains("huge"));
    assert!(bad("bogus = 1").contains("bogus"));
    assert!(bad("backend = \"metal\"").contains("metal"));
    assert!(bad("unsplit = [\"nope\"]").contains("nope"));
    assert!(bad("big_tile = [60, 128, 2, 4]").contains("big_tile"));
    assert!(bad("fp4_tile = [60, 128, 2, 4]").contains("fp4_tile"));
    assert!(bad("fp4_tile = [64, 32, 2, 4]").contains("16 * wn"));
    assert!(bad("fp4_stages = 1").contains("fp4_stages"));
    let vision = Spec::parse(&format!("{base}\n[runtime.vision]\nprecision = \"exact\"\ncache_mb = 64\nmax_batch = 2\n")).unwrap();
    assert_eq!(
        vision.runtime.vision,
        specrt::spec::VisionOptions { precision: specrt::vision_encoder::VisionPrecision::Exact, cache_mb: 64, max_batch: 2, pan_and_scan: false }
    );
    assert_eq!(Spec::parse(&vision.to_toml().unwrap()).unwrap(), vision);
    assert!(format!("{:#}", Spec::parse(&format!("{base}\n[runtime.vision]\nprecision = \"half\"\n")).unwrap_err()).contains("half"));
    assert!(format!("{:#}", Spec::parse(&format!("{base}\n[runtime.vision]\nmax_batch = 0\n")).unwrap_err()).contains("max_batch"));
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
    for field in ["unit_offset", "output_gate", "rotary_dim", "linear_attention", "mtp"] {
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

fn nvfp4_with_mtp(block: &str) -> anyhow::Result<Spec> {
    let text = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap();
    let (head, _) = text.split_once("[mtp]").unwrap();
    Spec::parse(&format!("{head}[mtp]\n{block}"))
}

#[test]
fn mtp_block_defaults_to_two_draft_tokens_and_rejects_what_cannot_run() {
    let s = nvfp4_with_mtp("layers = 1\n").unwrap();
    let m = s.mtp.clone().unwrap();
    assert_eq!((m.draft_tokens, m.speculate, m.max_batch), (2, true, None));
    assert_eq!(Spec::parse(&s.to_toml().unwrap()).unwrap(), s);
    let s = nvfp4_with_mtp("layers = 1\ndraft_tokens = 3\nmax_batch = 4\n").unwrap();
    assert_eq!(s.speculative().map(|m| (m.draft_tokens, m.max_batch)), Some((3, Some(4))));
    let off = nvfp4_with_mtp("layers = 1\nspeculate = false\n").unwrap();
    assert!(off.speculative().is_none());
    assert_eq!(off.kv_layers(), 16);
    for bad in ["layers = 2\n", "layers = 1\ndraft_tokens = 0\n", "layers = 1\ndraft_tokens = 4\n", "layers = 1\nmax_batch = 0\n", "layers = 1\nbogus = 1\n"] {
        assert!(nvfp4_with_mtp(bad).is_err(), "{bad}");
    }
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.config.json")).unwrap()).unwrap();
    v["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(0);
    assert_eq!(Spec::from_hf_config(&v.to_string()).unwrap().mtp, None);
}

#[test]
fn the_35b_fp8_moe_config_is_runnable_with_mtp_and_block_fp8_experts() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.6-35b-a3b-fp8.config.json")).unwrap();
    let served = Spec::from_hf_config(&json).unwrap();
    assert!(served.quant.is_fp8() && served.moe.is_some() && served.speculative().is_some());
    served.runnable().unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!(spec.dims.intermediate, 0);
    let moe = spec.moe.unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (256, Some(8), 512, 512));
}

#[test]
fn a_dense_config_without_an_intermediate_size_names_the_missing_field() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3-0.6b.config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v.as_object_mut().unwrap().remove("intermediate_size");
    let e = Spec::from_hf_config(&v.to_string()).unwrap_err().to_string();
    assert!(e.contains("no intermediate_size") && !e.contains("missing field"), "{e}");
}

#[test]
fn a_config_of_a_family_the_engine_has_not_seen_is_still_described() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/gemma4-31b.config.json")).unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!((spec.dims.hidden, spec.layers.count, spec.attention.heads, spec.attention.head_dim), (5376, 60, 32, 256));
    assert_eq!(spec.mlp.activation, Activation::Geglu);
    assert!(Spec::from_hf_config(&json).is_err());
}

#[test]
fn tensors_decide_qk_norm_and_the_output_gate_not_the_family() {
    use specrt::structure::Structure;
    let fixtures = root().join("tests/fixtures");
    let describe = |config: &str, header: &str| {
        let json = std::fs::read_to_string(fixtures.join(config)).unwrap();
        let mut spec = Spec::describe_hf_config(&json).unwrap();
        let rows = std::fs::read_to_string(fixtures.join("headers").join(header)).unwrap();
        spec.apply_structure(&Structure::from_json(&rows).unwrap());
        spec
    };
    let llama = describe("llama-3.2-1b.config.json", "llama-3.2-1b.json");
    assert!(!llama.attention.qk_norm && !llama.attention.qkv_bias && !llama.attention.output_gate);
    let qwen3 = describe("qwen3-0.6b.config.json", "qwen3-0.6b.json");
    assert!(qwen3.attention.qk_norm && !qwen3.attention.output_gate);
    let big = describe("qwen3.8-27b.config.json", "qwen3.8-27b-fp8.json");
    assert!(big.attention.qk_norm && big.attention.output_gate);
    let gemma = describe("gemma4-31b.config.json", "gemma4-31b.json");
    assert!(gemma.attention.qk_norm);
}

#[test]
fn layer_types_in_any_config_give_sliding_and_full_layers_with_the_window() {
    use specrt::anatomy::{LayerKind, ModelAnatomy};
    let json = std::fs::read_to_string(root().join("tests/fixtures/gemma4-31b.config.json")).unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!(spec.layers.window, Some(1024));
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Sliding, Block::Sliding, Block::Sliding, Block::Sliding, Block::Full]);
    let a = ModelAnatomy::new(&spec, 0, None);
    assert_eq!((a.count(LayerKind::SlidingAttention), a.count(LayerKind::Attention), a.count(LayerKind::DeltaNet)), (50, 10, 0));
    assert!(a.pattern().starts_with("SSSSSA"));
}

#[test]
fn mtp_head_quant_defaults_to_the_target_head_and_validates() {
    let hf = std::fs::read_to_string(format!("{}/tests/fixtures/tiny-qwen3.5-mtp/config.json", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let s = Spec::from_hf_config(&hf).unwrap();
    assert_eq!(s.mtp.as_ref().unwrap().head_quant, None);
    assert_eq!(s.draft_head_quant(), Some(Quant::Nvfp4G16));
    let mut fp32 = s.clone();
    fp32.runtime.gemm_fp32 = true;
    assert_eq!(fp32.draft_head_quant(), None);
    let mut bf = s.clone();
    bf.mtp.as_mut().unwrap().head_quant = Some(Quant::Bf16);
    assert_eq!(bf.draft_head_quant(), None);
    let mut q = s.clone();
    q.mtp.as_mut().unwrap().head_quant = Some(Quant::Fp8Channel);
    q.validate().unwrap();
    assert_eq!(q.draft_head_quant(), Some(Quant::Fp8Channel));
    let again = Spec::parse(&q.to_toml().unwrap()).unwrap();
    assert_eq!(again.mtp.unwrap().head_quant, Some(Quant::Fp8Channel));
    let mut bad = s;
    bad.mtp.as_mut().unwrap().head_quant = Some(Quant::Q8);
    assert!(bad.validate().unwrap_err().to_string().contains("[mtp] head_quant"));
}

fn core_fixture(name: &str) -> String {
    std::fs::read_to_string(root().join("../core/tests/fixtures/hf").join(name)).unwrap()
}

fn kinds(json: &str) -> std::collections::BTreeMap<(usize, String), Quant> {
    specrt::spec::module_quants(json).unwrap().into_iter().map(|(l, m, q)| ((l, m), q)).collect()
}

#[test]
fn compressed_tensors_regex_targets_resolve_to_per_layer_dtypes() {
    let json = core_fixture("primitive-ai-config.json");
    let refusal = Spec::from_hf_config_serving(&json).unwrap_err().to_string();
    assert!(refusal.contains("layer 8 linear_attn.in_proj_qkv") && refusal.contains("Nvfp4G16"), "{refusal}");
    let k = kinds(&json);
    let at = |l: usize, m: &str| k.get(&(l, m.to_string())).copied();
    assert_eq!(at(3, "self_attn.q_proj"), Some(Quant::Fp8Channel));
    assert_eq!(at(5, "linear_attn.in_proj_qkv"), Some(Quant::Fp8Channel));
    assert_eq!(at(9, "linear_attn.in_proj_qkv"), Some(Quant::Nvfp4G16));
    assert_eq!(at(9, "linear_attn.out_proj"), Some(Quant::Fp8Channel));
    assert_eq!(at(0, "mlp.up_proj"), Some(Quant::Nvfp4G16));
    assert_eq!(at(0, "linear_attn.in_proj_a"), None);
}

#[test]
fn modelopt_mixed_config_maps_quant_algos_to_the_same_kernels() {
    let json = core_fixture("darkstar-config.json");
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.quant, Quant::Fp8Channel);
    assert_eq!((spec.mlp.quant, spec.mlp.quant_until), (Some(Quant::Nvfp4G16), None));
    assert_eq!(spec.mlp.act_quant, specrt::spec::ActQuant::None);
    let k = kinds(&json);
    assert_eq!(k[&(0, "linear_attn.out_proj".to_string())], Quant::Fp8Channel);
    assert_eq!(k[&(63, "self_attn.o_proj".to_string())], Quant::Fp8Channel);
    assert_eq!(k[&(63, "mlp.down_proj".to_string())], Quant::Nvfp4G16);
    let nvfp4 = Spec::from_hf_config_serving(&core_fixture("nvfp4-qwen3.8-27b-config.json")).unwrap();
    assert_eq!((nvfp4.quant, nvfp4.mlp.quant), (Quant::Fp8Channel, Some(Quant::Nvfp4G16)));
}

#[test]
fn glm5_3_flash_describes_with_its_glm_keys_and_refuses_to_serve() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/glm5.3-flash.config.json")).unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!(spec.rope.theta, None, "glm5.3 has no rope_theta");
    assert_eq!(spec.theta(), 10_000.0);
    assert_eq!(spec.attention.head_dim, 4096 / 64, "head_dim 0 falls back to hidden / heads");
    assert_eq!(spec.layers.pattern.len(), 45, "the 3-then-1 rhythm has no period that divides 45");
    assert!(spec.layers.pattern.iter().enumerate().all(|(i, b)| if i % 4 == 3 { *b == Block::Full } else { b.is_linear() }), "deepseek_sparse_attention is full, linear_attention is linear");
    let la = spec.linear_attention.as_ref().unwrap();
    assert_eq!((la.key_heads, la.value_heads, la.key_head_dim, la.conv_kernel), (64, 64, 128, 4), "read from linear_attn_config");
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate), (288, Some(8), 2048));
    assert_eq!(moe.shared_intermediate, 2048, "n_shared_experts x moe_intermediate_size");
    assert_eq!(moe.dense_layers, vec![0, 1, 2], "first_k_dense_replace");
    assert!(moe.shared_gate);
    assert_eq!(spec.speculative().map(|m| m.layers), Some(1), "num_nextn_predict_layers");
    assert!(spec.runnable().is_err(), "glm5.3 cannot run yet (no family spec, no rope theta)");
    let mut no_theta = derived("qwen3-0.6b.config.json");
    no_theta.rope.theta = None;
    assert!(no_theta.runnable().unwrap_err().to_string().contains("rope theta is required to run"));
    let refusal = Spec::from_hf_config_serving(&json).unwrap_err().to_string();
    assert!(refusal.contains("model_type glm5_next is not supported"), "{refusal}");
}

#[test]
fn glm4_5_air_reads_partial_rotary_sigmoid_routing_shared_expert_and_mtp_layer() {
    let spec = derived("glm4.5-air.config.json");
    assert_eq!((spec.attention.head_dim, spec.rotary_dim()), (128, 64));
    assert!(spec.attention.qkv_bias && !spec.attention.qk_norm);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k(), moe.expert_intermediate, moe.shared_slots()), (128, 8, 1408, 1));
    assert_eq!((moe.router, moe.score_bias, moe.norm_topk, moe.shared_gate, moe.groups), (specrt::spec::RouterAct::Sigmoid, true, true, false, 0));
    assert_eq!(moe.dense_layers, vec![0]);
    assert_eq!(spec.dims.intermediate, 10944);
    assert_eq!(spec.layers.count, 46);
    spec.runnable().unwrap();
}

#[test]
fn a_plain_attention_mtp_block_validates_only_with_a_layout() {
    let mut s = derived("glm4.5-air.config.json");
    assert!(s.linear_attention.is_none() && !s.attention.output_gate);
    let m = s.mtp.clone().expect("the family names the nextn tensors");
    assert!(m.layout.is_some());
    s.validate().unwrap();
    s.mtp.as_mut().unwrap().layout = None;
    let err = s.validate().unwrap_err().to_string();
    assert!(err.contains("[mtp.layout]"), "{err}");
    let mut bad = derived("glm4.5-air.config.json");
    bad.mtp.as_mut().unwrap().layout.as_mut().unwrap().fc = "fc".into();
    assert!(bad.validate().unwrap_err().to_string().contains(".weight"));
    let mut again = derived("glm4.5-air.config.json");
    again.mtp.as_mut().unwrap().layout.as_mut().unwrap().norm = String::new();
    assert!(again.validate().is_err());
    let round = Spec::parse(&derived("glm4.5-air.config.json").to_toml().unwrap()).unwrap();
    assert_eq!(round.mtp_layout().layer, "model.layers.46");
}

#[test]
fn llama4_scout_is_served_through_its_family_spec() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/llama4-scout.config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.layers.count, 48);
    assert_eq!(spec.moe.as_ref().unwrap().experts, 16);
    spec.runnable().unwrap();
}

#[test]
fn qwen3_next_nvfp4_config_reads_fp4_experts_and_fused_linear_projections() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3-next-80b-nvfp4.config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.quant, Quant::Bf16);
    assert_eq!(spec.mlp.quant, Some(Quant::Nvfp4G16));
    assert_eq!(spec.mlp.act_quant, specrt::spec::ActQuant::Fp4);
    let la = spec.linear_attention.as_ref().unwrap();
    assert!(la.fused_interleaved);
    assert_eq!((la.key_heads, la.value_heads), (16, 32));
    let tensors = specrt::plan::Weight::LinIn.tensors(&spec);
    assert_eq!(tensors, vec!["linear_attn.in_proj_qkvz.weight".to_string()]);
    assert_eq!(specrt::plan::Weight::LinBa.tensors(&spec), vec!["linear_attn.in_proj_ba.weight".to_string()]);
}
