use crate::backend::{BackendConfig, Request};
use crate::matrix::run_model;
use crate::result::RunResult;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub struct BenchModel {
    pub name: String,
    pub ids: Vec<(String, String)>,
    pub skips: Vec<(String, String)>,
}

impl BenchModel {
    pub fn id_for(&self, backend: &str) -> Option<&str> {
        self.ids.iter().find(|(b, _)| b == backend).map(|(_, id)| id.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct BenchPlan {
    pub models: Vec<BenchModel>,
    pub backends: Vec<(String, BackendConfig)>,
    pub prompt_lengths: Vec<usize>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub model: String,
    pub backend: String,
    pub prompt_len: usize,
}

impl BenchPlan {
    pub fn cells(&self) -> Vec<Cell> {
        let mut out = Vec::new();
        for (backend, _) in &self.backends {
            for model in &self.models {
                for &len in &self.prompt_lengths {
                    out.push(Cell { model: model.name.clone(), backend: backend.clone(), prompt_len: len });
                }
            }
        }
        out
    }
}

const WORDS: [&str; 16] = [
    "the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog", "while", "rain", "falls",
    "on", "quiet", "stone", "streets",
];

pub fn synthetic_prompt(tokens: usize) -> String {
    let mut out = String::from("Summarise the following text in one sentence.\n\n");
    let words = tokens.saturating_sub(10);
    for i in 0..words {
        out.push_str(WORDS[(i * 7 + i / 16) % WORDS.len()]);
        out.push(' ');
    }
    out
}

pub fn run_plan(
    plan: &BenchPlan,
    cancel: &AtomicBool,
    on_start: &mut dyn FnMut(usize, &Cell),
    on_result: &mut dyn FnMut(usize, &Cell, RunResult),
) {
    let mut index = 0;
    for (label, config) in &plan.backends {
        let mut backend = config.build();
        for model in &plan.models {
            let mut loaded = false;
            for &len in &plan.prompt_lengths {
                let cell = Cell { model: model.name.clone(), backend: label.clone(), prompt_len: len };
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                on_start(index, &cell);
                let request = Request { prompt: synthetic_prompt(len), max_tokens: plan.max_tokens };
                let result = match (backend.as_mut(), model.id_for(label)) {
                    (Ok(b), Some(id)) => {
                        let mut r = if loaded {
                            run_model_loaded(b.as_mut(), label, &model.name, &request)
                        } else {
                            run_model(b.as_mut(), label, &model.name, id, &request, &mut |_| {})
                        };
                        loaded = r.error.is_none();
                        r.text = r.text.chars().take(2000).collect();
                        r.model = tagged(&model.name, len);
                        r
                    }
                    (Ok(_), None) => {
                        let why = model
                            .skips
                            .iter()
                            .find(|(b, _)| b == label)
                            .map(|(_, r)| r.clone())
                            .unwrap_or_else(|| format!("{label} has no way to load {}", model.name));
                        RunResult::failed(label, &tagged(&model.name, len), chrono::Utc::now(), crate::result::BackendError::classify(why))
                    }
                    (Err(e), _) => RunResult::failed(label, &tagged(&model.name, len), chrono::Utc::now(), e.clone()),
                };
                on_result(index, &cell, result);
                index += 1;
            }
        }
    }
}

pub fn tagged(model: &str, len: usize) -> String {
    format!("{model} @{len}")
}

pub fn save_session(dir: &std::path::Path, results: &[RunResult]) -> std::io::Result<String> {
    let folder = dir.join("sessions");
    std::fs::create_dir_all(&folder)?;
    let name = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let file = folder.join(format!("{}.json", name.replace([' ', ':'], "-")));
    std::fs::write(file, serde_json::to_string_pretty(results)?)?;
    Ok(name)
}

pub fn load_sessions(dir: &std::path::Path) -> Vec<(String, Vec<RunResult>)> {
    let Ok(entries) = std::fs::read_dir(dir.join("sessions")) else { return Vec::new() };
    let mut out: Vec<(String, Vec<RunResult>)> = entries
        .flatten()
        .filter_map(|e| {
            let stem = e.path().file_stem()?.to_string_lossy().into_owned();
            let results = serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
            Some((stem, results))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn run_model_loaded(
    backend: &mut dyn crate::backend::Backend,
    label: &str,
    model: &str,
    request: &Request,
) -> RunResult {
    let started = chrono::Utc::now();
    match backend.generate(request, &mut |_| {}) {
        Ok(g) => RunResult::ok(label, model, started, g),
        Err(e) => RunResult::failed(label, model, started, e),
    }
}

pub fn csv(results: &[RunResult]) -> String {
    let mut out = String::from("started,backend,model,prompt_tokens,completion_tokens,prefill_tok_s,decode_tok_s,ttft_ms,peak_memory_mib,error\n");
    let num = |v: Option<f64>| v.map(|v| format!("{v:.2}")).unwrap_or_default();
    let quote = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
    for r in results {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            r.started.to_rfc3339(),
            quote(&r.backend),
            quote(&r.model),
            r.prompt_tokens,
            r.completion_tokens,
            num(r.prefill_tok_s),
            num(r.decode_tok_s),
            num(r.ttft_secs.map(|s| s * 1000.0)),
            num(r.peak_memory_bytes.map(|b| b as f64 / 1048576.0)),
            quote(&r.error.as_ref().map(|e| format!("{}: {}", e.kind.label(), e.message)).unwrap_or_default()),
        ));
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    pub backend: String,
    pub model: String,
    pub prompt_tokens: u64,
    pub left: Option<f64>,
    pub right: Option<f64>,
}

impl Delta {
    pub fn change_percent(&self) -> Option<f64> {
        match (self.left, self.right) {
            (Some(l), Some(r)) if l > 0.0 => Some((r - l) / l * 100.0),
            _ => None,
        }
    }
}

pub fn compare(left: &[RunResult], right: &[RunResult], value: impl Fn(&RunResult) -> Option<f64>) -> Vec<Delta> {
    let mut keys: Vec<(String, String, u64)> = Vec::new();
    for r in left.iter().chain(right) {
        let key = (r.backend.clone(), r.model.clone(), r.prompt_tokens);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    let pick = |set: &[RunResult], k: &(String, String, u64)| {
        set.iter().filter(|r| r.backend == k.0 && r.model == k.1 && r.prompt_tokens == k.2).last().and_then(&value)
    };
    keys.into_iter()
        .map(|k| Delta { left: pick(left, &k), right: pick(right, &k), backend: k.0, model: k.1, prompt_tokens: k.2 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeServer, Script};
    use crate::result::Generation;

    fn script(models: &[&str]) -> Script {
        Script {
            models: models.iter().map(|m| m.to_string()).collect(),
            chunks: vec!["a ".into(), "b ".into(), "c ".into()],
            chunk_delay_ms: 3,
            first_delay_ms: 8,
            usage: Some((50, 3)),
            fail_status: None,
        }
    }

    fn openai(url: &str) -> BackendConfig {
        BackendConfig::Openai { url: url.into(), api_key: None, label: None }
    }

    fn plan(a: &FakeServer, b: &FakeServer) -> BenchPlan {
        BenchPlan {
            models: vec![
                BenchModel { name: "m1".into(), ids: vec![("a".into(), "m1".into()), ("b".into(), "m1".into())], skips: vec![] },
                BenchModel { name: "m2".into(), ids: vec![("a".into(), "m2".into()), ("b".into(), "m2".into())], skips: vec![] },
            ],
            backends: vec![("a".into(), openai(&a.url)), ("b".into(), openai(&b.url))],
            prompt_lengths: vec![32, 128],
            max_tokens: 3,
        }
    }

    #[test]
    fn the_grid_runs_every_model_backend_and_length_against_fake_http_backends() {
        let a = FakeServer::start(script(&["m1", "m2"]));
        let b = FakeServer::start(script(&["m1"]));
        let plan = plan(&a, &b);
        assert_eq!(plan.cells().len(), 8);
        let mut started = Vec::new();
        let mut results = Vec::new();
        run_plan(&plan, &AtomicBool::new(false), &mut |i, c| started.push((i, c.clone())), &mut |_, c, r| results.push((c.clone(), r)));
        assert_eq!(results.len(), 8);
        assert_eq!(started.iter().map(|(i, _)| *i).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
        let get = |b: &str, m: &str, l: usize| results.iter().find(|(c, _)| c.backend == b && c.model == m && c.prompt_len == l).unwrap();
        assert!(get("a", "m2", 128).1.error.is_none());
        assert_eq!(get("a", "m1", 32).1.completion_tokens, 3);
        let miss = get("b", "m2", 32).1.error.as_ref().unwrap();
        assert_eq!(miss.kind, crate::result::ErrorKind::ModelNotFound);
        let sent = a.requests.lock().unwrap().iter().filter(|r| r.starts_with("/v1/completions")).count();
        assert_eq!(sent, 6, "one warmup and two timed runs per model");
    }

    #[test]
    fn results_carry_the_prompt_length_and_sessions_round_trip() {
        let a = FakeServer::start(script(&["m1", "m2"]));
        let b = FakeServer::start(script(&["m1"]));
        let mut results = Vec::new();
        run_plan(&plan(&a, &b), &AtomicBool::new(false), &mut |_, _| {}, &mut |_, _, r| results.push(r));
        assert!(results.iter().any(|r| r.model == "m1 @32") && results.iter().any(|r| r.model == "m2 @128"));
        let dir = std::env::temp_dir().join(format!("ib-sessions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_session(&dir, &results).unwrap();
        let back = load_sessions(&dir);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].1.len(), results.len());
        assert_eq!(back[0].1[0].model, "m1 @32");
    }

    #[test]
    fn a_cancelled_plan_stops_before_the_next_cell() {
        let a = FakeServer::start(script(&["m1", "m2"]));
        let b = FakeServer::start(script(&["m1"]));
        let cancel = AtomicBool::new(false);
        let mut n = 0;
        run_plan(&plan(&a, &b), &cancel, &mut |_, _| {}, &mut |_, _, _| {
            n += 1;
            cancel.store(true, Ordering::Relaxed);
        });
        assert_eq!(n, 1);
    }

    #[test]
    fn synthetic_prompts_grow_with_the_requested_length() {
        assert!(synthetic_prompt(512).split_whitespace().count() > 4 * synthetic_prompt(128).split_whitespace().count() - 40);
        assert!(synthetic_prompt(4).starts_with("Summarise"));
    }

    fn run(backend: &str, tok_s: f64, prompt: u64) -> RunResult {
        let mut r = RunResult::ok(backend, "m", chrono::Utc::now(), Generation { prompt_tokens: prompt, ..Default::default() });
        r.decode_tok_s = Some(tok_s);
        r
    }

    #[test]
    fn two_runs_are_compared_cell_by_cell() {
        let left = vec![run("a", 100.0, 32), run("b", 50.0, 32)];
        let right = vec![run("a", 150.0, 32), run("c", 10.0, 32)];
        let d = compare(&left, &right, |r| r.decode_tok_s);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0].change_percent(), Some(50.0));
        assert_eq!((d[1].right, d[2].left), (None, None));
        assert_eq!(d[1].change_percent(), None);
    }

    #[test]
    fn csv_quotes_errors_and_lists_numbers() {
        let mut ok = run("a", 12.5, 32);
        ok.backend = "my,backend".into();
        let bad = RunResult::failed("b", "m", chrono::Utc::now(), crate::result::BackendError::failed("said \"no\""));
        let text = csv(&[ok, bad]);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("started,backend"));
        assert!(lines[1].contains("\"my,backend\"") && lines[1].contains(",12.50,"));
        assert!(lines[2].ends_with("\"failed: said \"\"no\"\"\""), "{}", lines[2]);
    }
}
