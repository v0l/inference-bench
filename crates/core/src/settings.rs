use crate::backend::BackendConfig;
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
    pub endpoints: Vec<Endpoint>,
    pub model_folders: Vec<PathBuf>,
    pub server: ServerSettings,
    #[serde(default)]
    pub specrt_worker: String,
    #[serde(default)]
    pub python: String,
    #[serde(default = "default_ctx")]
    pub specrt_ctx: usize,
    #[serde(default)]
    pub hf_token: String,
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
            specrt_worker: String::new(),
            python: String::new(),
            specrt_ctx: default_ctx(),
            hf_token: String::new(),
        }
    }
}

impl Settings {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join("settings.json")
    }

    pub fn load(dir: &Path) -> Self {
        std::fs::read_to_string(Self::path(dir))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
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

    pub fn backend_names(&self) -> Vec<String> {
        let mut names = vec!["specrt".to_string()];
        names.extend(self.endpoints.iter().map(|e| e.label.clone()));
        names.push("transformers".into());
        names
    }

    pub fn config_for(&self, label: &str) -> Option<BackendConfig> {
        let opt = |s: &str| (!s.trim().is_empty()).then(|| s.trim().to_string());
        match label {
            "specrt" => Some(BackendConfig::Specrt {
                worker: opt(&self.specrt_worker).map(PathBuf::from),
                max_ctx: Some(self.specrt_ctx),
            }),
            "transformers" => Some(BackendConfig::Transformers { python: opt(&self.python), device: None }),
            other => self.endpoints.iter().find(|e| e.label == other).map(|e| BackendConfig::Openai {
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
    fn backend_labels_map_to_configs() {
        let s = Settings::default();
        assert_eq!(s.config_for("specrt").unwrap().kind(), "specrt");
        assert_eq!(s.config_for("llama.cpp").unwrap().kind(), "openai");
        assert_eq!(s.config_for("transformers").unwrap().kind(), "transformers");
        assert!(s.config_for("nope").is_none());
        assert_eq!(s.backend_names(), vec!["specrt", "llama.cpp", "vllm", "transformers"]);
    }
}
