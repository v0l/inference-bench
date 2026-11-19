use specrt::family::Registry;
use specrt::spec::{EMBEDDED, Spec};
use std::path::Path;

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn registry_with(extra: &[&str]) -> Registry {
    Registry::from_texts(EMBEDDED.iter().map(|(_, t)| *t).chain(extra.iter().copied())).unwrap()
}

#[test]
fn a_family_added_by_spec_file_alone_builds_a_spec() {
    let acme = std::fs::read_to_string(fixtures().join("family-acme.toml")).unwrap();
    let qwen3 = std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap();
    let renamed = qwen3.replace("\"model_type\": \"qwen3\"", "\"model_type\": \"acmelm\"").replace("Qwen3ForCausalLM", "AcmeForCausalLM");
    assert!(renamed.contains("acmelm"));
    assert!(Spec::from_hf_config_serving_in(&registry_with(&[]), &renamed).unwrap_err().to_string().contains("model_type acmelm is not supported"));
    let built = Spec::from_hf_config_serving_in(&registry_with(&[&acme]), &renamed).unwrap();
    let mut want = Spec::from_hf_config_serving_in(&registry_with(&[]), &qwen3).unwrap();
    want.name = "acmelm".into();
    assert_eq!(built, want);
}

#[test]
fn a_family_spec_file_can_change_a_flag_the_code_used_to_fix() {
    let acme = std::fs::read_to_string(fixtures().join("family-acme.toml")).unwrap().replace("qk_norm = true", "qk_norm = false");
    let qwen3 = std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap().replace("\"model_type\": \"qwen3\"", "\"model_type\": \"acmelm\"");
    let built = Spec::from_hf_config_serving_in(&registry_with(&[&acme]), &qwen3).unwrap();
    assert!(!built.attention.qk_norm);
}

#[test]
fn every_embedded_family_has_a_name_and_a_match_or_is_generic() {
    let r = Registry::embedded();
    for f in r.families() {
        assert!(matches!(f.name.as_str(), "generic" | "base") || !f.matches.model_types.is_empty(), "{}", f.name);
    }
    assert!(r.for_model_type("qwen3_5_text").is_some() && r.for_model_type("mistral").is_some());
}

fn production_lines(file: &str) -> Vec<(usize, String)> {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(file)).unwrap();
    let mut out = Vec::new();
    let mut skipping = false;
    for (n, line) in text.lines().enumerate() {
        if line.starts_with("#[cfg(test)]") {
            break;
        }
        if line.starts_with("const ") && line.contains("_ARCHS") {
            skipping = true;
        }
        if skipping {
            skipping = !line.trim_end().ends_with("];");
            continue;
        }
        out.push((n + 1, line.to_string()));
    }
    out
}

#[test]
fn the_mapper_names_no_architecture_activation_or_quant_method() {
    let r = Registry::embedded();
    let mut forbidden: Vec<String> = r.families().iter().flat_map(|f| f.matches.model_types.iter().cloned()).collect();
    forbidden.extend(r.families().iter().flat_map(|f| f.activation.map.keys().cloned()));
    forbidden.extend(r.families().iter().flat_map(|f| f.gguf.pretokenizers.keys().cloned()));
    forbidden.extend(["gelu", "silu", "swish", "compressed-tensors", "float-quantized", "nvfp4-pack-quantized", "e4m3", "gptq", "awq", "bfloat16", "float16", "float32", "gemma", "mistral"].map(String::from));
    forbidden.sort();
    forbidden.dedup();
    for file in ["src/spec.rs", "src/gguf.rs", "src/family.rs"] {
        for (n, line) in production_lines(file) {
            for word in &forbidden {
                assert!(!line.contains(&format!("\"{word}\"")) && !line.contains(&format!("\"{word}")), "{file}:{n} names {word}: {line}");
            }
        }
    }
}

#[test]
fn spec_rs_and_library_rs_name_no_model_type() {
    let r = Registry::embedded();
    let mut names: Vec<&str> = r.families().iter().flat_map(|f| f.matches.model_types.iter().map(String::as_str)).collect();
    names.extend(["gemma", "phi3"]);
    for file in ["src/spec.rs", "../core/src/library.rs"] {
        let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(file)).unwrap();
        let mut skipping = false;
        for (n, line) in text.lines().enumerate() {
            if line.starts_with("#[cfg(test)]") {
                break;
            }
            if line.starts_with("const ") && line.contains("_ARCHS") {
                skipping = true;
            }
            if skipping {
                skipping = !line.trim_end().ends_with("];");
                continue;
            }
            for name in &names {
                assert!(!line.contains(&format!("\"{name}\"")), "{file}:{}: {line}", n + 1);
            }
        }
    }
}

#[test]
fn a_family_with_its_own_key_names_and_activation_builds_by_toml_alone() {
    let acme = std::fs::read_to_string(fixtures().join("family-acme.toml")).unwrap();
    let base = std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
    v["model_type"] = "acmelm".into();
    let hidden = v.as_object_mut().unwrap().remove("hidden_size").unwrap();
    v["n_embd"] = hidden;
    v.as_object_mut().unwrap().remove("hidden_act");
    v["ffn_act"] = "tanh_glu".into();
    let built = Spec::from_hf_config_serving_in(&registry_with(&[&acme]), &v.to_string()).unwrap();
    assert_eq!(built.dims.hidden, 1024);
    assert_eq!(built.mlp.activation, specrt::spec::Activation::Geglu);
    assert!(Spec::from_hf_config_serving_in(&registry_with(&[]), &v.to_string()).is_err());
}

#[test]
fn a_gguf_architecture_needs_only_a_family_toml() {
    let family = Registry::from_texts(EMBEDDED.iter().map(|(_, t)| *t).chain([include_str!("fixtures/family-acme.toml")])).unwrap();
    assert!(family.for_gguf("acmelm").is_none());
    let acme = std::fs::read_to_string(fixtures().join("family-acme.toml")).unwrap() + "\n";
    let with_gguf = acme.replace("model_types = [\"acmelm\"]", "model_types = [\"acmelm\"]\ngguf = [\"acmelm\"]");
    let r = registry_with(&[&with_gguf]);
    let f = r.for_gguf("acmelm").unwrap();
    assert_eq!(f.gguf.key("block_count", "acmelm").unwrap(), "acmelm.block_count");
    assert_eq!(f.gguf.tensor("embedding").unwrap(), "token_embd.weight");
}

fn clamped_config() -> String {
    let base = std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
    v["model_type"] = "clampedlm".into();
    v["swiglu_limit"] = 6.0.into();
    v["swiglu_alpha"] = 1.5.into();
    v.to_string()
}

#[test]
fn a_clamped_swiglu_with_alpha_comes_from_the_family_toml_into_every_gated_kernel() {
    let clamped = std::fs::read_to_string(fixtures().join("family-clamped.toml")).unwrap();
    let spec = Spec::from_hf_config_serving_in(&registry_with(&[&clamped]), &clamped_config()).unwrap();
    let gated = spec.mlp.gated.as_ref().unwrap();
    assert_eq!((gated.params["alpha"], gated.params["limit"]), (1.5, 6.0));
    let plain = Spec::from_hf_config_serving_in(&registry_with(&[]), &std::fs::read_to_string(fixtures().join("qwen3-0.6b.config.json")).unwrap()).unwrap();
    assert!(plain.mlp.gated.is_none());
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.source.contains("#define GATED_BODY(g, u) (") && module.source.contains("fminf(g, 6e0f)") && module.source.contains("1.5e0f"), "clamp and alpha are lowered into the gated body");
    let stock = specrt::codegen::Module::generate_kv(&plain, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(stock.source.contains("#define GATED_BODY(g, u) (activation(g) * u)"));
}

#[test]
fn a_gated_expression_with_an_unknown_name_is_a_load_error_naming_it() {
    let clamped = std::fs::read_to_string(fixtures().join("family-clamped.toml")).unwrap().replace("* alpha", "* beta");
    let spec = Spec::from_hf_config_serving_in(&registry_with(&[&clamped]), &clamped_config()).unwrap();
    let err = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_err().to_string();
    assert!(err.contains("beta"), "{err}");
}

#[test]
fn block_quantised_experts_are_declared_in_the_family_toml() {
    let clamped = std::fs::read_to_string(fixtures().join("family-clamped.toml")).unwrap();
    let fam = registry_with(&[&clamped]);
    let f = fam.for_model_type("clampedlm").unwrap();
    assert_eq!(f.moe.expert_block, Some(specrt::dequant::MXFP4));
    assert!(f.moe.gate_up_interleaved);
}
