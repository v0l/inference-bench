use crate::result::RunResult;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn data_dir() -> PathBuf {
    std::env::var_os("INFERENCE_BENCH_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::data_dir().map(|d| d.join("inference-bench")))
        .unwrap_or_else(|| PathBuf::from("inference-bench-data"))
}

fn slug(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' }).collect()
}

pub fn save(dir: &Path, result: &RunResult) -> Result<PathBuf> {
    let runs = dir.join("runs");
    std::fs::create_dir_all(&runs).with_context(|| runs.display().to_string())?;
    let stamp = result.started.format("%Y%m%dT%H%M%S%3f");
    let base = format!("{stamp}-{}-{}", slug(&result.backend), slug(&result.model));
    let mut path = runs.join(format!("{base}.json"));
    let mut n = 1;
    while path.exists() {
        path = runs.join(format!("{base}-{n}.json"));
        n += 1;
    }
    std::fs::write(&path, serde_json::to_string_pretty(result)?)?;
    Ok(path)
}

pub fn load_all(dir: &Path) -> Vec<RunResult> {
    let Ok(entries) = std::fs::read_dir(dir.join("runs")) else { return Vec::new() };
    let mut runs: Vec<RunResult> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok())
        .collect();
    runs.sort_by_key(|r| r.started);
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{BackendError, Generation};
    use chrono::Utc;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn saved_runs_come_back_in_time_order() {
        let dir = tmp("order");
        let t = Utc::now();
        let a = RunResult::ok("llama.cpp", "qwen/3", t, Generation { text: "a".into(), ..Default::default() });
        let b = RunResult::failed("vllm", "qwen/3", t + chrono::Duration::seconds(1), BackendError::unsupported("x"));
        save(&dir, &b).unwrap();
        save(&dir, &a).unwrap();
        let back = load_all(&dir);
        assert_eq!(back, vec![a, b]);
    }

    #[test]
    fn the_same_backend_and_model_in_one_instant_do_not_overwrite() {
        let dir = tmp("clash");
        let r = RunResult::ok("x", "m", Utc::now(), Generation::default());
        save(&dir, &r).unwrap();
        save(&dir, &r).unwrap();
        assert_eq!(load_all(&dir).len(), 2);
    }

    #[test]
    fn an_empty_directory_has_no_runs() {
        assert!(load_all(&tmp("none")).is_empty());
    }
}
