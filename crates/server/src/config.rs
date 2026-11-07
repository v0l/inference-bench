use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn legacy_native_ctx(table: &toml::Table) -> Result<Option<usize>, String> {
    match table.get("kind").and_then(|k| k.as_str()) {
        Some("specrt" | "native") | None => {
            Ok(table.get("max_ctx").and_then(|v| v.as_integer()).map(|n| n as usize))
        }
        Some(other) => Err(other.to_string()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default, skip_serializing)]
    pub backend: Option<toml::Table>,
    #[serde(default)]
    pub max_ctx: Option<usize>,
    #[serde(default)]
    pub vram_bytes: Option<u64>,
    #[serde(default)]
    pub preload: bool,
}

impl ModelConfig {
    pub fn upstream_name(&self) -> &str {
        self.model.as_deref().unwrap_or(&self.id)
    }

    pub fn spec_hash(&self) -> String {
        let name = self.upstream_name();
        let bytes = match name.ends_with(".toml") {
            true => std::fs::read(name).unwrap_or_else(|_| name.as_bytes().to_vec()),
            false => name.as_bytes().to_vec(),
        };
        let digest = Sha256::digest(&bytes);
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    pub cert: PathBuf,
    pub key: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: String,
    pub api_keys: Vec<String>,
    pub api_keys_env: String,
    pub protect_metrics: bool,
    pub cors_origins: Vec<String>,
    pub tls: Option<TlsConfig>,
    pub models_dir: Option<PathBuf>,
    pub max_ctx: Option<usize>,
    #[serde(skip_serializing)]
    pub default_backend: Option<toml::Table>,
    pub default_model: Option<String>,
    pub models: Vec<ModelConfig>,
    pub max_concurrent: usize,
    pub queue_depth: usize,
    pub retry_after_secs: u64,
    pub idle_unload_secs: Option<u64>,
    pub max_loaded_models: usize,
    pub vram_budget_bytes: Option<u64>,
    pub default_max_tokens: usize,
    pub shutdown_grace_secs: u64,
    pub access_log: Option<PathBuf>,
    pub log_prompts: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".into(),
            api_keys: Vec::new(),
            api_keys_env: "INFERENCE_BENCH_API_KEYS".into(),
            protect_metrics: false,
            cors_origins: Vec::new(),
            tls: None,
            models_dir: None,
            max_ctx: None,
            default_backend: None,
            default_model: None,
            models: Vec::new(),
            max_concurrent: 1,
            queue_depth: 16,
            retry_after_secs: 1,
            idle_unload_secs: None,
            max_loaded_models: 1,
            vram_budget_bytes: None,
            default_max_tokens: 256,
            shutdown_grace_secs: 30,
            access_log: None,
            log_prompts: false,
        }
    }
}

impl ServerConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        Self::parse(&text).with_context(|| path.display().to_string())
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut config: Self = toml::from_str(text)?;
        config.migrate();
        config.discover_models()?;
        config.validate()?;
        Ok(config)
    }

    fn migrate(&mut self) {
        match self.default_backend.take().as_ref().map(legacy_native_ctx) {
            Some(Ok(max_ctx)) => self.max_ctx = self.max_ctx.or(max_ctx),
            Some(Err(kind)) => tracing::warn!(%kind, "default_backend is no longer used: the server runs models on the native engine"),
            None => {}
        }
        let mut dropped = Vec::new();
        self.models.retain_mut(|m| match m.backend.take().as_ref().map(legacy_native_ctx) {
            Some(Ok(max_ctx)) => {
                m.max_ctx = m.max_ctx.or(max_ctx);
                true
            }
            Some(Err(kind)) => {
                tracing::warn!(model = %m.id, %kind, "dropping a model served by another engine: the server runs models on the native engine");
                dropped.push(m.id.clone());
                false
            }
            None => true,
        });
        if self.default_model.as_ref().is_some_and(|d| dropped.contains(d)) {
            tracing::warn!(model = ?self.default_model, "default_model was dropped with its backend");
            self.default_model = None;
        }
    }

    pub fn discover_models(&mut self) -> Result<()> {
        let Some(dir) = self.models_dir.clone() else { return Ok(()) };
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| dir.display().to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        found.sort();
        for path in found {
            let id = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            if self.models.iter().any(|m| m.id == id) {
                continue;
            }
            self.models.push(ModelConfig {
                id,
                model: Some(path.display().to_string()),
                ..ModelConfig::default()
            });
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.max_concurrent >= 1, "max_concurrent must be at least 1");
        anyhow::ensure!(self.max_loaded_models >= 1, "max_loaded_models must be at least 1");
        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            anyhow::ensure!(!m.id.is_empty(), "a model has an empty id");
            anyhow::ensure!(seen.insert(&m.id), "model {} is listed twice", m.id);
        }
        if let Some(d) = &self.default_model {
            anyhow::ensure!(self.model(d).is_some(), "default_model {d} is not a configured model");
        }
        Ok(())
    }

    pub fn model(&self, id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn max_ctx_for(&self, m: &ModelConfig) -> Option<usize> {
        m.max_ctx.or(self.max_ctx)
    }

    pub fn resolved_keys(&self) -> Vec<String> {
        let mut keys = self.api_keys.clone();
        if let Ok(env) = std::env::var(&self.api_keys_env) {
            keys.extend(env.split(',').map(str::trim).filter(|k| !k.is_empty()).map(String::from));
        }
        keys.retain(|k| !k.is_empty());
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_models_and_limits() {
        let c = ServerConfig::parse(
            r#"
bind = "0.0.0.0:9000"
api_keys = ["k1"]
max_concurrent = 4
idle_unload_secs = 600
default_model = "qwen"
max_ctx = 4096

[[models]]
id = "qwen"
model = "specs/qwen3-0.6b.toml"
vram_bytes = 2000000000

[[models]]
id = "long"
max_ctx = 32768
"#,
        )
        .unwrap();
        assert_eq!(c.bind, "0.0.0.0:9000");
        assert_eq!(c.models.len(), 2);
        assert_eq!(c.max_ctx_for(&c.models[0]), Some(4096));
        assert_eq!(c.max_ctx_for(&c.models[1]), Some(32768));
        assert_eq!(c.idle_unload_secs, Some(600));
    }

    #[test]
    fn an_old_config_with_other_backends_loads_and_keeps_the_native_models() {
        let c = ServerConfig::parse(
            r#"
default_model = "llama"

[default_backend]
kind = "specrt"
worker = "/opt/inference-bench-specrt"
max_ctx = 4096

[[models]]
id = "qwen"

[[models]]
id = "llama"
[models.backend]
kind = "openai"
url = "http://127.0.0.1:8081"

[[models]]
id = "hf"
[models.backend]
kind = "specrt"
max_ctx = 1024

[[models]]
id = "torch"
[models.backend]
kind = "transformers"
python = "python3"
"#,
        )
        .unwrap();
        let ids: Vec<&str> = c.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["qwen", "hf"]);
        assert_eq!(c.max_ctx_for(&c.models[0]), Some(4096));
        assert_eq!(c.max_ctx_for(&c.models[1]), Some(1024));
        assert_eq!(c.default_model, None);
    }

    #[test]
    fn unknown_keys_and_duplicate_ids_are_rejected() {
        assert!(ServerConfig::parse("bogus = 1").is_err());
        assert!(ServerConfig::parse("[[models]]\nid='a'\n[[models]]\nid='a'").is_err());
        assert!(ServerConfig::parse("default_model='x'").is_err());
    }

    #[test]
    fn models_dir_registers_spec_files() {
        let dir = std::env::temp_dir().join(format!("ib-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tiny.toml"), "x = 1").unwrap();
        std::fs::write(dir.join("notes.txt"), "").unwrap();
        let c = ServerConfig::parse(&format!("models_dir = {:?}", dir.display().to_string())).unwrap();
        assert_eq!(c.models.len(), 1);
        assert_eq!(c.models[0].id, "tiny");
        assert!(c.models[0].upstream_name().ends_with("tiny.toml"));
    }
}
