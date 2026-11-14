use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub model_folders: Vec<PathBuf>,
    #[serde(default)]
    pub server: ServerSettings,
    #[serde(default = "default_ctx", alias = "specrt_ctx")]
    pub default_ctx: usize,
    #[serde(default)]
    pub model_ctx: BTreeMap<String, usize>,
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
            model_folders: Vec::new(),
            server: ServerSettings::default(),
            default_ctx: default_ctx(),
            model_ctx: BTreeMap::new(),
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
        serde_json::from_str::<Self>(text).unwrap_or_default()
    }

    pub fn ctx_for(&self, model: &str) -> usize {
        self.model_ctx.get(model).copied().unwrap_or(self.default_ctx)
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
        s.hf_token = "hf_secret".into();
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
    fn an_old_settings_file_with_a_global_context_and_bandwidth_still_loads() {
        let old = r#"{
            "endpoints": [{"label": "vllm", "backend": "vllm", "url": "http://127.0.0.1:8000", "api_key": "k"}],
            "model_folders": ["/models"],
            "server": {"port": 9000, "api_key": "", "lan": true},
            "specrt_worker": "/opt/inference-bench-specrt",
            "python": "/usr/bin/python3",
            "specrt_ctx": 4096,
            "gpu_bandwidth_gbs": 1792,
            "placement": {"context": 2048, "devices": [], "stages": []},
            "hf_token": "hf_x"
        }"#;
        let s = Settings::parse(old);
        assert_eq!((s.default_ctx, s.server.port, s.hf_token.as_str()), (4096, 9000, "hf_x"));
        assert_eq!(s.model_folders, vec![PathBuf::from("/models")]);
        assert_eq!(s.ctx_for("anything"), 4096);
        let specs = s.specs();
        assert!(specs.auto_update);
        assert_eq!((specs.repo.as_str(), specs.branch.as_str(), specs.update_hours), ("v0l/inference-bench", "master", 24));
        assert!(s.spec_folders.is_empty());
        let saved = serde_json::to_string(&s).unwrap();
        assert!(!saved.contains("specrt_ctx") && !saved.contains("gpu_bandwidth") && !saved.contains("endpoints"));
    }

    #[test]
    fn a_remembered_context_wins_over_the_default() {
        let mut s = Settings::default();
        s.model_ctx.insert("Qwen3-0.6B".into(), 32768);
        assert_eq!((s.ctx_for("Qwen3-0.6B"), s.ctx_for("other")), (32768, 8192));
        let dir = temp("c");
        s.save(&dir).unwrap();
        assert_eq!(Settings::load(&dir).ctx_for("Qwen3-0.6B"), 32768);
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
