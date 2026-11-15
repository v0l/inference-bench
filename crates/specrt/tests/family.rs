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
