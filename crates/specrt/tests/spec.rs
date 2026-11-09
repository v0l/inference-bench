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
    let file = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
    spec.name = file.name.clone();
    spec.repo = file.repo.clone();
    spec.min_engine = file.min_engine;
    assert_eq!(file, spec);
    assert_eq!(file.min_engine, 7);
    assert_eq!(file.mtp, Some(specrt::spec::Mtp::new(1)));
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

#[test]
fn deprecated_env_overrides_still_switch_the_resolved_spec() {
    use specrt::spec::{ActQuant, TileChoice};
    let plain = Spec::parse(&nvfp4_text()).unwrap();
    assert_eq!(plain.with_env_overrides(), plain);
    unsafe {
        std::env::set_var("SPECRT_FP4_ACT", "1");
        std::env::set_var("SPECRT_FP8_ACT", "1");
        std::env::set_var("SPECRT_NO_GRAPH", "1");
        std::env::set_var("SPECRT_TILE", "narrow");
    }
    let r = plain.with_env_overrides();
    unsafe {
        for v in ["SPECRT_FP4_ACT", "SPECRT_FP8_ACT", "SPECRT_NO_GRAPH", "SPECRT_TILE"] {
            std::env::remove_var(v);
        }
    }
    assert_eq!((r.act_quant, r.mlp.act_quant), (ActQuant::Fp8, ActQuant::Fp4));
    assert!(!r.runtime.graphs);
    assert_eq!(r.runtime.tile, TileChoice::Narrow);
    assert!(r.runtime.fuse);
    r.validate().unwrap();
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
    std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap()
}

fn nvfp4_text() -> String {
    std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap()
}

#[test]
fn act_quant_fields_default_to_none_round_trip_and_match_the_act_fixtures() {
    use specrt::spec::ActQuant;
    let plain = Spec::parse(&nvfp4_text()).unwrap();
    assert_eq!((plain.act_quant, plain.mlp.act_quant), (ActQuant::None, ActQuant::None));
    let text = plain.to_toml().unwrap();
    assert!(!text.contains("act_quant") && !text.contains("[runtime]"), "{text}");
    assert_eq!(Spec::parse(&text).unwrap(), plain);
    let act = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4-act.toml")).unwrap();
    assert_eq!((act.act_quant, act.mlp.act_quant), (ActQuant::Fp8, ActQuant::Fp4));
    assert_eq!(Spec::parse(&act.to_toml().unwrap()).unwrap(), act);
    let mut same = act.clone();
    same.name = plain.name.clone();
    same.min_engine = plain.min_engine;
    same.act_quant = ActQuant::None;
    same.mlp.act_quant = ActQuant::None;
    assert_eq!(same, plain);
    let fp8 = Spec::load(&root().join("tests/fixtures/qwen3.8-27b-act.toml")).unwrap();
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
fn runtime_table_defaults_round_trips_and_rejects_bad_values() {
    use specrt::spec::{RuntimeOptions, TileChoice};
    let base = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    assert_eq!(Spec::parse(&base).unwrap().runtime, RuntimeOptions::default());
    let d = RuntimeOptions::default();
    assert!(d.graphs && d.fuse && !d.gemm_fp32 && d.tile == TileChoice::Auto && d.cache_dir.is_none());
    let with = format!("{base}\n[runtime]\ngraphs = false\nfuse = false\ntile = \"big\"\nbig_tile = [64, 128, 2, 4]\nfp4_tile = [64, 256, 2, 4]\nfp4_stages = 4\nunsplit = [\"lm_head\"]\ngemm_fp32 = true\ncache_dir = \"/tmp/x\"\n");
    let s = Spec::parse(&with).unwrap();
    assert_eq!(
        s.runtime,
        RuntimeOptions {
            graphs: false,
            fuse: false,
            tile: TileChoice::Big,
            big_tile: Some([64, 128, 2, 4]),
            fp4_tile: Some([64, 256, 2, 4]),
            fp4_stages: Some(4),
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
    assert!(bad("unsplit = [\"nope\"]").contains("nope"));
    assert!(bad("big_tile = [60, 128, 2, 4]").contains("big_tile"));
    assert!(bad("fp4_tile = [60, 128, 2, 4]").contains("fp4_tile"));
    assert!(bad("fp4_stages = 1").contains("fp4_stages"));
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
    let text = std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap();
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
fn moe_configs_without_an_intermediate_size_are_described_but_refused_for_serving() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/qwen3.6-35b-a3b-fp8.config.json")).unwrap();
    let e = Spec::from_hf_config(&json).unwrap_err().to_string();
    assert!(e.contains("mixture-of-experts"), "{e}");
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
    assert_eq!(s.draft_head_quant(), None);
    let mut q = s.clone();
    q.mtp.as_mut().unwrap().head_quant = Some(Quant::Nvfp4G16);
    q.validate().unwrap();
    assert_eq!(q.draft_head_quant(), Some(Quant::Nvfp4G16));
    let again = Spec::parse(&q.to_toml().unwrap()).unwrap();
    assert_eq!(again.mtp.unwrap().head_quant, Some(Quant::Nvfp4G16));
    let mut bad = s;
    bad.mtp.as_mut().unwrap().head_quant = Some(Quant::Q8);
    assert!(bad.validate().unwrap_err().to_string().contains("[mtp] head_quant"));
}
