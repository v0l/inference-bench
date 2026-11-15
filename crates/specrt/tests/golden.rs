use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> PathBuf {
    root().join("tests/fixtures")
}

fn golden_dir() -> PathBuf {
    root().join("tests/golden")
}

fn config_files() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(fixtures()).unwrap().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if path.join("config.json").is_file() && name.starts_with("tiny-") {
                out.push((name, path.join("config.json")));
            }
        } else if let Some(stem) = name.strip_suffix(".config.json") {
            out.push((stem.to_string(), path));
        }
    }
    out.push(("qwen3.8-27b.multimodal".into(), fixtures().join("qwen3.8-27b.multimodal.json")));
    out.sort();
    out
}

fn render(result: anyhow::Result<Spec>) -> String {
    match result {
        Ok(spec) => serde_json::to_string_pretty(&spec).unwrap(),
        Err(e) => format!("refused: {e:#}\n"),
    }
}

fn check(name: &str, text: String) {
    let path = golden_dir().join(name);
    if std::env::var_os("GOLDEN_UPDATE").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, &text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is missing, run with GOLDEN_UPDATE=1", path.display()));
    assert_eq!(want, text, "{name}");
}

fn variants() -> Vec<(String, String)> {
    let read = |f: &str| serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(fixtures().join(f)).unwrap()).unwrap();
    let mut out = Vec::new();
    let mut add = |name: &str, base: &str, edit: &dyn Fn(&mut serde_json::Value)| {
        let mut v = read(base);
        edit(&mut v);
        out.push((name.to_string(), v.to_string()));
    };
    use serde_json::json;
    add("v-qwen3-next", "qwen3.8-27b.config.json", &|v| v["model_type"] = json!("qwen3_next"));
    add("v-qwen3-moe-dense", "qwen3-0.6b.config.json", &|v| v["model_type"] = json!("qwen3_moe"));
    add("v-qwen3-bias", "qwen3-0.6b.config.json", &|v| v["attention_bias"] = json!(true));
    add("v-llama-bias", "llama-3.2-1b.config.json", &|v| v["attention_bias"] = json!(true));
    add("v-qwen2-no-bias-key", "qwen2.5-0.5b.config.json", &|v| { v.as_object_mut().unwrap().remove("attention_bias"); });
    add("v-weird", "qwen3-0.6b.config.json", &|v| { v["model_type"] = json!("weird"); v["architectures"] = json!(["WeirdForCausalLM"]); });
    add("v-gelu", "qwen3-0.6b.config.json", &|v| v["hidden_act"] = json!("gelu_pytorch_tanh"));
    add("v-relu2", "qwen3-0.6b.config.json", &|v| v["hidden_act"] = json!("relu2"));
    add("v-no-act", "qwen3-0.6b.config.json", &|v| { v.as_object_mut().unwrap().remove("hidden_act"); });
    add("v-int8", "qwen3-0.6b.config.json", &|v| v["torch_dtype"] = json!("int8"));
    add("v-f16", "qwen3-0.6b.config.json", &|v| v["torch_dtype"] = json!("float16"));
    add("v-linear-rope", "llama-3.2-1b.config.json", &|v| v["rope_scaling"] = json!({"type": "linear", "factor": 4.0}));
    add("v-bad-rope", "llama-3.2-1b.config.json", &|v| v["rope_scaling"] = json!({"type": "yarn", "factor": 4.0}));
    add("v-fp8-block", "qwen3-0.6b.config.json", &|v| v["quantization_config"] = json!({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}));
    add("v-gptq", "qwen3-0.6b.config.json", &|v| v["quantization_config"] = json!({"quant_method": "gptq", "bits": 4}));
    add("v-no-intermediate", "qwen3-0.6b.config.json", &|v| { v.as_object_mut().unwrap().remove("intermediate_size"); });
    add("v-no-kv-heads", "qwen3-0.6b.config.json", &|v| { v.as_object_mut().unwrap().remove("num_key_value_heads"); });
    add("v-no-head-dim", "qwen3-0.6b.config.json", &|v| { v.as_object_mut().unwrap().remove("head_dim"); });
    add("v-no-theta", "qwen3-0.6b.config.json", &|v| { v.as_object_mut().unwrap().remove("rope_theta"); });
    add("v-27b-no-mtp", "qwen3.8-27b-nvfp4.config.json", &|v| v["text_config"]["mtp_num_hidden_layers"] = json!(0));
    add("v-27b-sliding", "qwen3.8-27b.config.json", &|v| v["text_config"]["layer_types"][1] = json!("sliding_attention"));
    add("v-27b-no-layer-types", "qwen3.8-27b.config.json", &|v| { v["text_config"].as_object_mut().unwrap().remove("layer_types"); v["text_config"]["full_attention_interval"] = json!(4); });
    add("v-27b-no-vision", "qwen3.8-27b.multimodal.json", &|v| v["language_model_only"] = json!(true));
    add("v-nvfp4-no-act", "qwen3.8-27b-nvfp4.config.json", &|v| {
        for g in ["group_0", "group_1"] {
            v["quantization_config"]["config_groups"][g]["input_activations"] = serde_json::Value::Null;
        }
    });
    add("v-nvfp4-group32", "qwen3.8-27b-nvfp4.config.json", &|v| v["quantization_config"]["config_groups"]["group_1"]["weights"]["group_size"] = json!(32));
    add("v-partial-rotary-top", "qwen3-0.6b.config.json", &|v| v["partial_rotary_factor"] = json!(0.5));
    add("v-top-level-qwen3_5", "tiny-qwen3.5/config.json", &|v| v["model_type"] = json!("qwen3_5_text"));
    out
}

#[test]
fn every_variant_builds_the_spec_master_built() {
    for (name, json) in variants() {
        check(&format!("{name}.serving.json"), render(Spec::from_hf_config_serving(&json)));
        check(&format!("{name}.plain.json"), render(Spec::from_hf_config(&json)));
        check(&format!("{name}.describe.json"), render(Spec::describe_hf_config(&json)));
    }
}

#[test]
fn every_config_builds_the_spec_master_built() {
    let configs = config_files();
    assert!(configs.len() >= 18, "{} configs", configs.len());
    for (name, path) in configs {
        let json = std::fs::read_to_string(&path).unwrap();
        check(&format!("{name}.serving.json"), render(Spec::from_hf_config_serving(&json)));
        check(&format!("{name}.plain.json"), render(Spec::from_hf_config(&json)));
        check(&format!("{name}.describe.json"), render(Spec::describe_hf_config(&json)));
    }
}

fn gguf_dir() -> PathBuf {
    std::env::var_os("SPECRT_GGUF_DIR").filter(|d| !d.is_empty()).map(Into::into).unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/specrt-gguf"))
}

#[test]
fn every_gguf_on_disk_builds_the_spec_master_built() {
    let mut seen = 0;
    for file in ["Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"] {
        let path = gguf_dir().join(file);
        if !path.is_file() {
            continue;
        }
        seen += 1;
        let mut spec = Spec::from_gguf_file(&path);
        if let Ok(s) = spec.as_mut() {
            s.repo = String::new();
        }
        check(&format!("{file}.json"), render(spec));
    }
    if std::env::var_os("GOLDEN_UPDATE").is_none() && seen == 0 {
        eprintln!("no GGUF files in {}", gguf_dir().display());
    }
}
