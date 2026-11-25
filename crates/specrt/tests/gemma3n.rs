use specrt::spec::Spec;
use std::path::Path;

fn config(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular").join(name).join("config.json")).unwrap()
}

#[test]
fn gemma_3n_configs_name_the_first_missing_piece_not_the_model_type() {
    for name in ["google__gemma-3n-E2B-it", "google__gemma-3n-E4B-it"] {
        let err = format!("{:#}", Spec::from_hf_config_serving(&config(name)).unwrap_err());
        assert!(!err.contains("model_type gemma3n is not supported"), "{name}: {err}");
        assert!(err.contains("Gemma 3n AltUp residual streams"), "{name}: {err}");
    }
}

#[test]
fn each_gemma_3n_piece_is_refused_by_name() {
    let base: serde_json::Value = serde_json::from_str(&config("google__gemma-3n-E2B-it")).unwrap();
    let pieces = [
        ("altup_num_inputs", serde_json::json!(1), "Laurel blocks"),
        ("laurel_rank", serde_json::json!(0), "per-layer input embeddings"),
        ("hidden_size_per_layer_input", serde_json::json!(0), "KV sharing across layers"),
    ];
    let mut v = base.clone();
    for (key, off, next) in pieces {
        v["text_config"][key] = off;
        let err = format!("{:#}", Spec::from_hf_config_serving(&v.to_string()).unwrap_err());
        assert!(err.contains(next), "with {key} off: {err}");
    }
}
