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
        Err(_) => "refused\n".to_string(),
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
