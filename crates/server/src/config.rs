use anyhow::{Context, Result};
use inference_bench_core::specs::SpecsConfig;
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
        spec_hash_of(self.upstream_name())
    }
}

pub fn spec_hash_of(name: &str) -> String {
    let bytes = match name.ends_with(".toml") {
        true => std::fs::read(name).unwrap_or_else(|_| name.as_bytes().to_vec()),
        false => name.as_bytes().to_vec(),
    };
    let digest = Sha256::digest(&bytes);
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpecsSection {
    pub auto_update: bool,
    pub repo: String,
    pub branch: String,
    pub tag: String,
    pub update_hours: u64,
    pub url: Option<String>,
    pub data_dir: Option<PathBuf>,
    pub folders: Vec<PathBuf>,
}

impl Default for SpecsSection {
    fn default() -> Self {
        let d = SpecsConfig::default();
        Self {
            auto_update: d.auto_update,
            repo: d.repo,
            branch: d.branch,
            tag: d.tag,
            update_hours: d.update_hours,
            url: None,
            data_dir: None,
            folders: Vec::new(),
        }
    }
}

impl SpecsSection {
    pub fn updater(&self) -> SpecsConfig {
        let d = SpecsConfig::default();
        SpecsConfig {
            auto_update: self.auto_update,
            repo: self.repo.clone(),
            branch: self.branch.clone(),
            tag: self.tag.clone(),
            update_hours: self.update_hours,
            base: self.url.clone().unwrap_or(d.base),
            ..d
        }
    }

    pub fn data(&self) -> PathBuf {
        self.data_dir.clone().unwrap_or_else(inference_bench_core::store::data_dir)
    }

    pub fn resolve(&self, name: &str) -> String {
        inference_bench_core::specs::resolve_model(&self.data(), &self.folders, name)
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
    pub prefix_cache_share_across_keys: bool,
    pub vram_budget_bytes: Option<u64>,
    pub default_max_tokens: Option<usize>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub shutdown_grace_secs: u64,
    pub access_log: Option<PathBuf>,
    pub log_prompts: bool,
    pub specs: SpecsSection,
    pub max_num_seqs: Option<usize>,
    pub max_model_len: Option<usize>,
    pub kv_cache_gb: Option<f64>,
    pub max_batched_tokens: Option<usize>,
    pub itl_target_ms: Option<u64>,
    pub runtime: Option<inference_bench_core::specrt_serve::RuntimeOptions>,
    pub speculative: inference_bench_core::specrt_serve::SpeculativeOptions,
    pub placement: inference_bench_core::specrt_placement::PlacementConfig,
    pub tool_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    pub chat_template_kwargs: toml::Table,
    pub media: inference_bench_core::media::MediaLimits,
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
            prefix_cache_share_across_keys: false,
            vram_budget_bytes: None,
            default_max_tokens: None,
            top_k: None,
            top_p: None,
            min_p: None,
            shutdown_grace_secs: 30,
            access_log: None,
            log_prompts: false,
            specs: SpecsSection::default(),
            max_num_seqs: None,
            max_model_len: None,
            kv_cache_gb: None,
            max_batched_tokens: None,
            itl_target_ms: None,
            runtime: None,
            speculative: Default::default(),
            placement: Default::default(),
            tool_parser: None,
            reasoning_parser: None,
            chat_template_kwargs: toml::Table::new(),
            media: Default::default(),
        }
    }
}

impl ServerConfig {
    pub fn with_thinking(mut self, thinking: inference_bench_core::specrt_chat::Thinking) -> Self {
        if self.reasoning_parser.is_none() && thinking == inference_bench_core::specrt_chat::Thinking::OpensInPrompt {
            self.reasoning_parser = Some("qwen3".into());
        }
        self
    }

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
        anyhow::ensure!(self.media.max_image_bytes > 0 && self.media.max_video_bytes > 0, "media byte limits must be positive");
        anyhow::ensure!(self.max_concurrent >= 1, "max_concurrent must be at least 1");
        anyhow::ensure!(self.max_num_seqs != Some(0), "max_num_seqs must be at least 1");
        anyhow::ensure!(self.max_model_len.is_none_or(|n| n >= 16), "max_model_len must be at least 16");
        anyhow::ensure!(self.default_max_tokens != Some(0), "default_max_tokens must be at least 1");
        anyhow::ensure!(self.kv_cache_gb.is_none_or(|g| g > 0.0), "kv_cache_gb must be positive");
        anyhow::ensure!(self.max_loaded_models >= 1, "max_loaded_models must be at least 1");
        if let Some(p) = &self.tool_parser {
            anyhow::ensure!(matches!(p.as_str(), "qwen3_coder" | "qwen3_xml" | "hermes"), "tool_parser {p} is not supported: use qwen3_coder or hermes");
        }
        if let Some(p) = &self.reasoning_parser {
            anyhow::ensure!(p == "qwen3", "reasoning_parser {p} is not supported: use qwen3");
        }
        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            anyhow::ensure!(!m.id.is_empty(), "a model has an empty id");
            anyhow::ensure!(seen.insert(&m.id), "model {} is listed twice", m.id);
        }
        if let Some(r) = &self.runtime {
            r.prefix_cache.validate()?;
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

    pub fn serve_options(&self, m: &ModelConfig) -> Option<inference_bench_core::specrt_serve::ServeOptions> {
        let seqs = self.max_num_seqs?;
        let d = inference_bench_core::specrt_serve::ServeOptions::default();
        Some(inference_bench_core::specrt_serve::ServeOptions {
            max_model_len: self.max_model_len.or(self.max_ctx_for(m)),
            max_num_seqs: seqs,
            kv_cache_bytes: self.kv_cache_gb.map(|g| (g * (1u64 << 30) as f64) as u64),
            kv_cache_tokens: None,
            max_batched_tokens: self.max_batched_tokens.unwrap_or(d.max_batched_tokens),
            itl_target: self.itl_target_ms.map(std::time::Duration::from_millis).unwrap_or(d.itl_target),
            runtime: self.runtime.clone(),
            speculative: self.speculative,
            placement: Some(self.placement.clone()),
        })
    }

    pub fn concurrency(&self) -> usize {
        self.max_concurrent.max(self.max_num_seqs.unwrap_or(0))
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
    fn the_packaged_example_config_parses() {
        let c = ServerConfig::parse(include_str!("../../../packaging/server.example.toml")).unwrap();
        assert_eq!(c.specs.data(), PathBuf::from("/var/lib/inference-bench"));
        assert_eq!(c.model("qwen3-0.6b").unwrap().upstream_name(), "qwen3-0.6b");
    }

    #[test]
    fn the_specs_section_has_defaults_and_can_pin_a_tag() {
        let c = ServerConfig::parse("").unwrap();
        assert!(c.specs.auto_update);
        assert_eq!(c.specs.updater().source(), "v0l/inference-bench@master");
        let c = ServerConfig::parse("[specs]\nauto_update = false\ntag = \"v0.2.0\"\nupdate_hours = 6\nurl = \"http://mirror\"").unwrap();
        let u = c.specs.updater();
        assert!(!u.auto_update);
        assert_eq!(u.url(), "http://mirror/repos/v0l/inference-bench/commits/refs/tags/v0.2.0");
        assert!(ServerConfig::parse("[specs]\nbogus = 1").is_err());
    }

    #[test]
    fn serving_options_come_from_the_config_and_raise_concurrency() {
        let c = ServerConfig::parse(
            "max_num_seqs = 32\nmax_model_len = 262144\nkv_cache_gb = 7\nmax_batched_tokens = 16384\nitl_target_ms = 80\n[[models]]\nid='m'",
        )
        .unwrap();
        let o = c.serve_options(&c.models[0]).unwrap();
        assert_eq!((o.max_num_seqs, o.max_model_len, o.max_batched_tokens), (32, Some(262144), 16384));
        assert_eq!(o.kv_cache_bytes, Some(7 << 30));
        assert_eq!(o.itl_target, std::time::Duration::from_millis(80));
        assert_eq!(c.concurrency(), 32);
        let plain = ServerConfig::parse("max_ctx = 4096\n[[models]]\nid='m'").unwrap();
        assert!(plain.serve_options(&plain.models[0]).is_none());
        assert_eq!(plain.concurrency(), 1);
        let c = ServerConfig::parse("max_num_seqs = 4\nmax_ctx = 8192\n[[models]]\nid='m'").unwrap();
        assert_eq!(c.serve_options(&c.models[0]).unwrap().max_model_len, Some(8192));
        assert!(ServerConfig::parse("max_model_len = 8").is_err());
        assert!(ServerConfig::parse("max_num_seqs = 0").is_err());
    }

    #[test]
    fn the_speculative_table_reaches_the_serving_options() {
        use inference_bench_core::specrt_serve::SpeculativeOptions;
        let c = ServerConfig::parse("max_num_seqs = 4\n[speculative]\ndraft_tokens = 3\nmax_batch = 2\n[[models]]\nid='m'").unwrap();
        let o = c.serve_options(&c.models[0]).unwrap();
        assert_eq!(o.speculative, SpeculativeOptions { enabled: None, draft_tokens: Some(3), max_batch: Some(2) });
        let off = ServerConfig::parse("max_num_seqs = 4\n[speculative]\nenabled = false\n[[models]]\nid='m'").unwrap();
        assert_eq!(off.serve_options(&off.models[0]).unwrap().speculative.enabled, Some(false));
        let plain = ServerConfig::parse("max_num_seqs = 4\n[[models]]\nid='m'").unwrap();
        assert_eq!(plain.serve_options(&plain.models[0]).unwrap().speculative, SpeculativeOptions::default());
        assert!(ServerConfig::parse("[speculative]\nk = 2").is_err());
    }

    #[test]
    fn the_runtime_table_reaches_the_serving_options_with_defaults_for_missing_keys() {
        use inference_bench_core::specrt_serve::{RuntimeOptions, TileChoice};
        let c = ServerConfig::parse("max_num_seqs = 4\n[runtime]\ngraphs = false\ntile = \"narrow\"\ncache_dir = \"/tmp/ptx\"\n[[models]]\nid='m'").unwrap();
        let r = c.serve_options(&c.models[0]).unwrap().runtime.unwrap();
        assert_eq!(r, RuntimeOptions { graphs: false, tile: TileChoice::Narrow, cache_dir: Some("/tmp/ptx".into()), ..RuntimeOptions::default() });
        assert!(r.fuse && !r.gemm_fp32);
        let plain = ServerConfig::parse("max_num_seqs = 4\n[[models]]\nid='m'").unwrap();
        assert_eq!(plain.serve_options(&plain.models[0]).unwrap().runtime, None);
        let e = ServerConfig::parse("[runtime]\ntile = \"huge\"").unwrap_err();
        assert!(format!("{e:#}").contains("huge"), "{e:#}");
        assert!(ServerConfig::parse("[runtime]\nbogus = 1").is_err());
    }

    #[test]
    fn the_prefix_cache_table_reaches_the_serving_options() {
        let c = ServerConfig::parse("max_num_seqs = 4\n[runtime.prefix_cache]\nenabled = true\nsnapshot_interval = 1024\nmax_snapshots = 9\n[[models]]\nid='m'").unwrap();
        let p = c.serve_options(&c.models[0]).unwrap().runtime.unwrap().prefix_cache;
        assert!(p.enabled);
        assert_eq!((p.snapshot_interval, p.max_snapshots, p.max_blocks), (1024, 9, 0));
        assert!(!c.prefix_cache_share_across_keys);
        let shared = ServerConfig::parse("prefix_cache_share_across_keys = true").unwrap();
        assert!(shared.prefix_cache_share_across_keys);
        assert!(ServerConfig::parse("[runtime.prefix_cache]\nsnapshot_interval = 100").is_err());
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
