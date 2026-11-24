use safetensors::tensor::{Dtype, TensorView};
use specrt::quant_map::{QuantMap, TensorQuant};
use specrt::spec::Spec;
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

struct Tensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
}

fn t(name: &str, dtype: Dtype, shape: &[usize]) -> Tensor {
    Tensor { name: name.to_string(), dtype, shape: shape.to_vec() }
}

fn quant_config() -> serde_json::Value {
    serde_json::json!({
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "exclude_modules": [],
        "quantized_layers": {
            "lm_head": { "quant_algo": "MXFP8", "weight_block_size": [1, 32] },
            "model.layers.0.self_attn.q_proj": { "quant_algo": "MXFP8", "group_size": 32 },
            "model.layers.0.self_attn.o_proj": { "quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128] },
            "model.layers.0.mlp.experts": { "quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1" },
            "model.layers.0.mlp.shared_experts.down_proj": { "quant_algo": "W4A16_NVFP4", "group_size": 16 },
            "model.layers.1.mlp.down_proj": { "quant_algo": "NVFP4", "group_size": 16 }
        }
    })
}

fn layout() -> Vec<Tensor> {
    let mut out = vec![
        t("model.embed_tokens.weight", Dtype::BF16, &[16, 64]),
        t("lm_head.weight", Dtype::F8_E4M3, &[16, 64]),
        t("lm_head.weight_scale", Dtype::U8, &[16, 2]),
        t("model.layers.0.self_attn.q_proj.weight", Dtype::F8_E4M3, &[32, 64]),
        t("model.layers.0.self_attn.q_proj.weight_scale", Dtype::U8, &[32, 2]),
        t("model.layers.0.self_attn.q_proj.bias", Dtype::BF16, &[32]),
        t("model.layers.0.self_attn.o_proj.weight", Dtype::F8_E4M3, &[200, 256]),
        t("model.layers.0.self_attn.o_proj.weight_scale", Dtype::F32, &[2, 1, 2, 1]),
        t("model.layers.0.self_attn.kv_b_proj.weight", Dtype::BF16, &[32, 64]),
        t("model.layers.0.mlp.shared_experts.down_proj.weight", Dtype::U8, &[64, 32]),
        t("model.layers.0.mlp.shared_experts.down_proj.weight_scale", Dtype::F8_E4M3, &[64, 4]),
        t("model.layers.0.mlp.shared_experts.down_proj.weight_scale_2", Dtype::F32, &[]),
        t("model.layers.1.mlp.down_proj.weight", Dtype::U8, &[64, 32]),
        t("model.layers.1.mlp.down_proj.weight_scale", Dtype::F8_E4M3, &[64, 4]),
        t("model.layers.1.mlp.down_proj.weight_scale_2", Dtype::F32, &[]),
        t("model.layers.1.mlp.down_proj.input_scale", Dtype::F32, &[]),
    ];
    for e in 0..2 {
        let m = format!("model.layers.0.mlp.experts.{e}.gate_proj");
        out.extend([
            t(&format!("{m}.weight"), Dtype::U8, &[64, 32]),
            t(&format!("{m}.weight_scale.csf32_bases"), Dtype::U8, &[64]),
            t(&format!("{m}.weight_scale.csf32_codes"), Dtype::U8, &[128]),
            t(&format!("{m}.weight_scale.csf32_page_bases"), Dtype::U32, &[1]),
            t(&format!("{m}.weight_scale.csf32_offsets"), Dtype::U16, &[5]),
            t(&format!("{m}.weight_scale.csf32_exceptions"), Dtype::U16, &[0]),
            t(&format!("{m}.weight_scale_2"), Dtype::F32, &[]),
            t(&format!("{m}.input_scale"), Dtype::F32, &[]),
        ]);
    }
    out
}

fn write(tag: &str, tensors: &[Tensor], config: &serde_json::Value) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("specrt-quant-map-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data: Vec<Vec<u8>> = tensors.iter().map(|t| vec![0u8; t.shape.iter().product::<usize>() * t.dtype.bitsize() / 8]).collect();
    let views: Vec<(String, TensorView)> = tensors.iter().zip(&data).map(|(t, d)| (t.name.clone(), TensorView::new(t.dtype, t.shape.clone(), d).unwrap())).collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    std::fs::write(dir.join("hf_quant_config.json"), config.to_string()).unwrap();
    dir
}

fn refusal(tag: &str, tensors: &[Tensor], config: &serde_json::Value) -> String {
    let dir = write(tag, tensors, config);
    let ck = Checkpoint::open(&dir).unwrap();
    let err = ck.quant_map().and_then(|m| ck.check_quant_layout(&m)).expect_err("must refuse").to_string();
    std::fs::remove_dir_all(&dir).ok();
    err
}

fn edited(edit: impl Fn(&mut Vec<Tensor>)) -> Vec<Tensor> {
    let mut l = layout();
    edit(&mut l);
    l
}

#[test]
fn hf_quant_config_maps_every_weight_tensor_to_its_algo() {
    let dir = write("ok", &layout(), &quant_config());
    let ck = Checkpoint::open(&dir).unwrap();
    let map = ck.quant_map().unwrap();
    ck.check_quant_layout(&map).unwrap();
    let at = |n: &str| map.tensor(n);
    assert_eq!(at("lm_head.weight"), Some(TensorQuant::Mxfp8));
    assert_eq!(at("model.layers.0.self_attn.q_proj.weight"), Some(TensorQuant::Mxfp8));
    assert_eq!(at("model.layers.0.self_attn.o_proj.weight"), Some(TensorQuant::Fp8Block128));
    assert_eq!(at("model.layers.0.mlp.experts.1.gate_proj.weight"), Some(TensorQuant::Nvfp4Csf));
    assert_eq!(at("model.layers.0.mlp.shared_experts.down_proj.weight"), Some(TensorQuant::W4a16Nvfp4));
    assert_eq!(at("model.layers.1.mlp.down_proj.weight"), Some(TensorQuant::Nvfp4));
    assert_eq!(at("model.layers.0.self_attn.kv_b_proj.weight"), None, "bf16 module");
    assert_eq!(at("model.layers.0.self_attn.q_proj_extra.weight"), None, "a module prefix only matches at a dot");
    assert_eq!(TensorQuant::Fp8Block128.activations(), None, "FP8_PB_WO is weight only");
    assert_eq!(TensorQuant::W4a16Nvfp4.activations(), None);
    assert_eq!(TensorQuant::Nvfp4.activations(), Some(specrt::spec::ActQuant::Fp4));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_spec_from_a_mixed_config_names_the_format_of_each_tensor() {
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap()).unwrap();
    config["quantization_config"] = quant_config();
    let spec = Spec::describe_hf_config(&config.to_string()).unwrap();
    assert_eq!(spec.tensor_quant("model.layers.0.mlp.experts.0.up_proj.weight"), Some(TensorQuant::Nvfp4Csf));
    assert_eq!(spec.tensor_quant("model.layers.0.self_attn.o_proj.weight"), Some(TensorQuant::Fp8Block128));
    assert_eq!(spec.tensor_quant("model.embed_tokens.weight"), None);
    let text = spec.to_toml().unwrap();
    assert_eq!(Spec::parse(&text).unwrap().quant_map, spec.quant_map, "the map round trips through the spec file");
    let plain = Spec::describe_hf_config(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap()).unwrap();
    assert!(plain.quant_map.is_empty() && !plain.to_toml().unwrap().contains("quant_map"));
}

#[test]
fn the_glm5_3_config_maps_all_494_modules() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5.3-flash.config.json")).unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    let counts = spec.quant_map.counts();
    assert_eq!(counts.get(&TensorQuant::Nvfp4Csf), Some(&42));
    assert_eq!(counts.get(&TensorQuant::W4a16Nvfp4), Some(&77));
    assert_eq!(counts.get(&TensorQuant::Mxfp8), Some(&205));
    assert_eq!(counts.get(&TensorQuant::Fp8Block128), Some(&170));
    assert_eq!(spec.tensor_quant("model.language_model.layers.3.mlp.experts.287.down_proj.weight"), Some(TensorQuant::Nvfp4Csf));
    assert_eq!(spec.tensor_quant("model.language_model.layers.45.mlp.experts.0.down_proj.weight"), Some(TensorQuant::W4a16Nvfp4));
    assert_eq!(spec.tensor_quant("model.language_model.layers.3.self_attn.kv_b_proj.weight"), None);
}

#[test]
fn an_unknown_algo_is_refused_with_the_module_name() {
    let mut config = quant_config();
    config["quantized_layers"]["model.layers.0.self_attn.q_proj"] = serde_json::json!({ "quant_algo": "INT4_AWQ", "group_size": 128 });
    let err = refusal("algo", &layout(), &config);
    assert!(err.contains("model.layers.0.self_attn.q_proj") && err.contains("INT4_AWQ"), "{err}");
    let mut config = quant_config();
    config["quantized_layers"]["model.layers.0.mlp.experts"]["weight_scale_encoding"] = serde_json::json!("csf_64x1_v9");
    let err = refusal("encoding", &layout(), &config);
    assert!(err.contains("model.layers.0.mlp.experts") && err.contains("csf_64x1_v9"), "{err}");
}

#[test]
fn a_scale_in_the_wrong_layout_is_refused_with_the_tensor_name() {
    let cases: Vec<(&str, Vec<Tensor>, &str)> = vec![
        ("mx-scale", edited(|l| l[2] = t("lm_head.weight_scale", Dtype::U8, &[16, 4])), "lm_head.weight_scale"),
        ("mx-dtype", edited(|l| l[4] = t("model.layers.0.self_attn.q_proj.weight_scale", Dtype::F32, &[32, 2])), "model.layers.0.self_attn.q_proj.weight_scale"),
        ("pb-grid", edited(|l| l[7] = t("model.layers.0.self_attn.o_proj.weight_scale", Dtype::F32, &[1, 1, 2, 1])), "model.layers.0.self_attn.o_proj.weight_scale"),
        ("pb-weight", edited(|l| l[6] = t("model.layers.0.self_attn.o_proj.weight", Dtype::BF16, &[200, 256])), "model.layers.0.self_attn.o_proj.weight"),
        ("w4-scale", edited(|l| l[10] = t("model.layers.0.mlp.shared_experts.down_proj.weight_scale", Dtype::F8_E4M3, &[64, 8])), "model.layers.0.mlp.shared_experts.down_proj.weight_scale"),
        ("nv-global", edited(|l| { l.remove(14); }), "model.layers.1.mlp.down_proj.weight_scale_2"),
        ("csf-missing", edited(|l| l.retain(|t| t.name != "model.layers.0.mlp.experts.1.gate_proj.weight_scale.csf32_offsets")), "model.layers.0.mlp.experts.1.gate_proj.weight_scale.csf32_offsets"),
        ("csf-plain", edited(|l| l.push(t("model.layers.1.mlp.down_proj.weight_scale.csf32_bases", Dtype::U8, &[64]))), "model.layers.1.mlp.down_proj.weight_scale.csf32_bases"),
        ("orphan", edited(|l| l[8] = t("model.layers.0.self_attn.kv_b_proj.weight", Dtype::F8_E4M3, &[32, 64])), "model.layers.0.self_attn.kv_b_proj.weight"),
    ];
    for (tag, tensors, name) in cases {
        let err = refusal(tag, &tensors, &quant_config());
        assert!(err.contains(&format!("{name} ")) || err.ends_with(name), "{tag}: {err}");
    }
}

#[test]
#[ignore = "needs the GLM-5.3 Flash checkpoint on disk; reads only the shard headers"]
fn the_real_glm5_3_checkpoint_layout_is_understood() {
    let ck = Checkpoint::for_repo("Piecrust/GLM-5.3-Flash-FP4-Abliterated").unwrap();
    let map = ck.quant_map().unwrap();
    assert_eq!(map.0.len(), 494);
    ck.check_quant_layout(&map).unwrap();
}

#[test]
fn a_checkpoint_without_quantized_layers_has_an_empty_map() {
    let dir = write("none", &[t("model.embed_tokens.weight", Dtype::BF16, &[16, 64])], &serde_json::json!({ "quant_method": "fp8" }));
    let ck = Checkpoint::open(&dir).unwrap();
    let map = ck.quant_map().unwrap();
    assert!(map.is_empty());
    ck.check_quant_layout(&map).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let _: QuantMap = Default::default();
}
