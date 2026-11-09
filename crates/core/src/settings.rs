use crate::target::{NATIVE, TargetConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    pub label: String,
    pub backend: String,
    pub url: String,
    #[serde(default)]
    pub api_key: String,
}

impl Endpoint {
    fn is_http(&self) -> bool {
        self.url.starts_with("http://") || self.url.starts_with("https://")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerSettings {
    pub port: u16,
    pub api_key: String,
    pub lan: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self { port: 1234, api_key: String::new(), lan: false }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub endpoints: Vec<Endpoint>,
    #[serde(default)]
    pub model_folders: Vec<PathBuf>,
    #[serde(default)]
    pub server: ServerSettings,
    #[serde(default = "default_ctx")]
    pub specrt_ctx: usize,
    #[serde(default)]
    pub gpu_bandwidth_gbs: Option<u64>,
    #[serde(default)]
    pub hf_token: String,
    #[serde(default)]
    pub spec_folders: Vec<PathBuf>,
    #[serde(default = "yes")]
    pub specs_auto_update: bool,
    #[serde(default = "default_specs_repo")]
    pub specs_repo: String,
    #[serde(default = "default_specs_branch")]
    pub specs_branch: String,
    #[serde(default)]
    pub specs_tag: String,
    #[serde(default = "default_specs_hours")]
    pub specs_update_hours: u64,
}

fn yes() -> bool {
    true
}

fn default_specs_repo() -> String {
    crate::specs::DEFAULT_REPO.into()
}

fn default_specs_branch() -> String {
    crate::specs::DEFAULT_BRANCH.into()
}

fn default_specs_hours() -> u64 {
    crate::specs::DEFAULT_HOURS
}

fn default_ctx() -> usize {
    8192
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            endpoints: vec![
                Endpoint { label: "llama.cpp".into(), backend: "llama.cpp".into(), url: "http://127.0.0.1:8080".into(), api_key: String::new() },
                Endpoint { label: "vllm".into(), backend: "vllm".into(), url: "http://127.0.0.1:8000".into(), api_key: String::new() },
            ],
            model_folders: Vec::new(),
            server: ServerSettings::default(),
            specrt_ctx: default_ctx(),
            gpu_bandwidth_gbs: None,
            hf_token: String::new(),
            spec_folders: Vec::new(),
            specs_auto_update: true,
            specs_repo: default_specs_repo(),
            specs_branch: default_specs_branch(),
            specs_tag: String::new(),
            specs_update_hours: default_specs_hours(),
        }
    }
}

impl Settings {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join("settings.json")
    }

    pub fn load(dir: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(Self::path(dir)) else { return Self::default() };
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Self {
        let Ok(mut s) = serde_json::from_str::<Self>(text) else { return Self::default() };
        for old in ["specrt_worker", "python"] {
            if serde_json::from_str::<serde_json::Value>(text).is_ok_and(|v| v.get(old).is_some()) {
                tracing::info!(setting = old, "ignoring a setting for a removed backend");
            }
        }
        s.endpoints.retain(|e| {
            let keep = e.is_http() && e.label != NATIVE;
            if !keep {
                tracing::info!(label = %e.label, url = %e.url, "dropping a backend entry that is not an OpenAI-compatible bench target");
            }
            keep
        });
        s
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = Self::path(dir);
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    pub fn native(&self) -> TargetConfig {
        TargetConfig::native(Some(self.specrt_ctx))
    }

    pub fn specs(&self) -> crate::specs::SpecsConfig {
        crate::specs::SpecsConfig {
            auto_update: self.specs_auto_update,
            repo: self.specs_repo.clone(),
            branch: self.specs_branch.clone(),
            tag: self.specs_tag.clone(),
            update_hours: self.specs_update_hours,
            ..crate::specs::SpecsConfig::default()
        }
    }

    pub fn bench_targets(&self) -> Vec<String> {
        let mut names = vec![NATIVE.to_string()];
        names.extend(self.endpoints.iter().map(|e| e.label.clone()));
        names
    }

    pub fn target_for(&self, label: &str) -> Option<TargetConfig> {
        let opt = |s: &str| (!s.trim().is_empty()).then(|| s.trim().to_string());
        match label {
            NATIVE => Some(self.native()),
            other => self.endpoints.iter().find(|e| e.label == other).map(|e| TargetConfig::Openai {
                url: e.url.clone(),
                api_key: opt(&e.api_key),
                label: Some(e.label.clone()),
            }),
        }
    }

    pub fn library_backend(&self, label: &str) -> String {
        match self.endpoints.iter().find(|e| e.label == label) {
            Some(e) => e.backend.clone(),
            None => label.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn settings_round_trip_and_keys_are_private() {
        let dir = temp("a");
        let mut s = Settings::default();
        s.endpoints[0].api_key = "sk-secret".into();
        s.save(&dir).unwrap();
        assert_eq!(Settings::load(&dir), s);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(Settings::path(&dir)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn a_missing_or_broken_file_gives_defaults() {
        let dir = temp("b");
        assert_eq!(Settings::load(&dir), Settings::default());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(Settings::path(&dir), "{nope").unwrap();
        assert_eq!(Settings::load(&dir), Settings::default());
    }

    #[test]
    fn bench_targets_are_the_native_engine_and_the_http_endpoints() {
        let s = Settings::default();
        assert_eq!(s.target_for("specrt").unwrap().kind(), "native");
        assert_eq!(s.target_for("llama.cpp").unwrap().kind(), "openai");
        assert!(s.target_for("transformers").is_none());
        assert_eq!(s.bench_targets(), vec!["specrt", "llama.cpp", "vllm"]);
    }

    #[test]
    fn old_settings_with_removed_backends_load_and_keep_their_bench_targets() {
        let old = r#"{
            "endpoints": [
                {"label": "llama.cpp", "backend": "llama.cpp", "url": "http://127.0.0.1:8080", "api_key": ""},
                {"label": "transformers", "backend": "transformers", "url": "", "api_key": ""},
                {"label": "lmstudio", "backend": "lmstudio", "url": "https://box:1234/v1", "api_key": "k"}
            ],
            "model_folders": ["/models"],
            "server": {"port": 9000, "api_key": "", "lan": true},
            "specrt_worker": "/opt/inference-bench-specrt",
            "python": "/usr/bin/python3",
            "specrt_ctx": 4096,
            "hf_token": "hf_x"
        }"#;
        let s = Settings::parse(old);
        assert_eq!(s.bench_targets(), vec!["specrt", "llama.cpp", "lmstudio"]);
        assert_eq!((s.specrt_ctx, s.server.port, s.hf_token.as_str()), (4096, 9000, "hf_x"));
        assert_eq!(s.model_folders, vec![PathBuf::from("/models")]);
        assert_eq!(s.native(), TargetConfig::native(Some(4096)));
        let specs = s.specs();
        assert!(specs.auto_update);
        assert_eq!((specs.repo.as_str(), specs.branch.as_str(), specs.update_hours), ("v0l/inference-bench", "master", 24));
        assert!(s.spec_folders.is_empty());
    }

    #[test]
    fn spec_update_settings_round_trip() {
        let s = Settings::parse(r#"{"specs_auto_update": false, "specs_tag": "v0.2.0", "specs_update_hours": 0, "spec_folders": ["/mine"]}"#);
        let c = s.specs();
        assert!(!c.auto_update);
        assert_eq!(c.source(), "v0l/inference-bench@v0.2.0");
        assert_eq!(c.interval(), None);
        assert_eq!(s.spec_folders, vec![PathBuf::from("/mine")]);
    }
}
