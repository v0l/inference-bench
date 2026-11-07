use crate::target::{Target, TargetConfig, Request};
use crate::result::RunResult;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendEntry {
    pub label: String,
    #[serde(flatten)]
    pub config: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    #[serde(default)]
    pub ids: BTreeMap<String, String>,
}

impl ModelEntry {
    pub fn id_for(&self, backend: &str) -> &str {
        self.ids.get(backend).map(String::as_str).unwrap_or(&self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixSpec {
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub backends: Vec<BackendEntry>,
    pub models: Vec<ModelEntry>,
}

fn default_max_tokens() -> usize {
    64
}

impl MatrixSpec {
    pub fn parse(text: &str) -> Result<Self, String> {
        let spec: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        if spec.backends.is_empty() || spec.models.is_empty() {
            return Err("a matrix needs at least one backend and one model".into());
        }
        Ok(spec)
    }

    pub fn prompt_text(&self, base: &std::path::Path) -> Result<String, String> {
        match (&self.prompt, &self.prompt_file) {
            (Some(p), _) => Ok(p.clone()),
            (None, Some(f)) => {
                let path = base.join(f);
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
            }
            _ => Err("the matrix has no prompt or prompt_file".into()),
        }
    }
}

pub fn run_model(
    backend: &mut dyn Target,
    label: &str,
    display_model: &str,
    model_id: &str,
    request: &Request,
    on_text: &mut dyn FnMut(&str),
) -> RunResult {
    let started = Utc::now();
    let outcome = backend.load(model_id).and_then(|_| backend.generate(request, on_text));
    match outcome {
        Ok(g) => RunResult::ok(label, display_model, started, g),
        Err(e) => RunResult::failed(label, display_model, started, e),
    }
}

pub fn run_matrix(
    spec: &MatrixSpec,
    request: &Request,
    on_result: &mut dyn FnMut(RunResult),
) {
    for entry in &spec.backends {
        let mut backend = entry.config.build();
        for model in &spec.models {
            let result = run_model(backend.as_mut(), &entry.label, &model.name, model.id_for(&entry.label), request, &mut |_| {});
            on_result(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeServer, Script};

    fn script(models: &[&str]) -> Script {
        Script {
            models: models.iter().map(|m| m.to_string()).collect(),
            chunks: vec!["a ".into(), "b ".into(), "c ".into()],
            chunk_delay_ms: 5,
            first_delay_ms: 10,
            usage: Some((10, 3)),
            fail_status: None,
            files: Default::default(),
        }
    }

    #[test]
    fn spec_parses_with_per_backend_model_ids() {
        let spec = MatrixSpec::parse(
            r#"
prompt = "hi"
[[backends]]
label = "llama.cpp"
kind = "openai"
url = "http://x:8080"
[[backends]]
label = "specrt"
kind = "specrt"
[[models]]
name = "Qwen3-0.6B"
ids = { specrt = "Qwen/Qwen3-0.6B" }
"#,
        )
        .unwrap();
        assert_eq!(spec.max_tokens, 64);
        assert_eq!(spec.models[0].id_for("specrt"), "Qwen/Qwen3-0.6B");
        assert_eq!(spec.models[0].id_for("llama.cpp"), "Qwen3-0.6B");
        assert_eq!(spec.backends[1].config.kind(), "native");
    }

    #[test]
    fn an_empty_matrix_is_rejected() {
        assert!(MatrixSpec::parse("prompt = \"x\"\nbackends = []\nmodels = []").is_err());
    }

    #[test]
    fn matrix_runs_every_cell_and_keeps_each_cells_error() {
        let good = FakeServer::start(script(&["m1", "m2"]));
        let partial = FakeServer::start(script(&["m1"]));
        let spec = MatrixSpec {
            prompt: Some("hello".into()),
            prompt_file: None,
            max_tokens: 3,
            backends: vec![
                BackendEntry {
                    label: "good".into(),
                    config: TargetConfig::Openai { url: good.url.clone(), api_key: None, label: None },
                },
                BackendEntry {
                    label: "partial".into(),
                    config: TargetConfig::Openai { url: partial.url.clone(), api_key: None, label: None },
                },
            ],
            models: vec![
                ModelEntry { name: "m1".into(), ids: Default::default() },
                ModelEntry { name: "m2".into(), ids: Default::default() },
            ],
        };
        let mut results = Vec::new();
        run_matrix(&spec, &Request { prompt: "hello".into(), max_tokens: 3 }, &mut |r| results.push(r));
        let cell = |b: &str, m: &str| results.iter().find(|r| r.backend == b && r.model == m).unwrap();
        assert_eq!(results.len(), 4);
        assert!(cell("good", "m1").error.is_none() && cell("good", "m2").error.is_none());
        assert!(cell("partial", "m1").error.is_none());
        assert_eq!(cell("good", "m1").completion_tokens, 3);
        let e = cell("partial", "m2").error.as_ref().unwrap();
        assert_eq!(e.kind, crate::result::ErrorKind::ModelNotFound);
    }
}
