use inference_bench_core::fake::{FakeServer, Script};
use inference_bench_core::result::{ErrorKind, RunResult};
use inference_bench_core::store;
use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_inference-bench"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn server(models: &[&str]) -> FakeServer {
    FakeServer::start(Script {
        models: models.iter().map(|m| m.to_string()).collect(),
        chunks: vec!["one ".into(), "two ".into(), "three ".into()],
        chunk_delay_ms: 10,
        first_delay_ms: 30,
        usage: Some((12, 3)),
        fail_status: None,
    })
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn bench_streams_prints_and_saves_one_result() {
    let out = tmp("run");
    let prompt = out.join("p.txt");
    std::fs::write(&prompt, "hello there").unwrap();
    let s = server(&["qwen3"]);
    let o = bin()
        .args(["--out", out.to_str().unwrap(), "bench", "--target", &format!("openai:{}", s.url)])
        .args(["--label", "fake", "--model", "qwen3", "--prompt-file", prompt.to_str().unwrap()])
        .args(["--max-tokens", "3", "--json"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let printed: Vec<RunResult> = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(printed[0].text, "one two three ");
    assert_eq!(printed[0].backend, "fake");
    assert_eq!(printed[0].completion_tokens, 3);
    let saved = store::load_all(&out);
    assert_eq!(saved.len(), 1);
    assert_eq!(saved[0].model, "qwen3");
    assert!(s.requests.lock().unwrap().iter().any(|r| r.contains("hello there")));
}

#[test]
fn bench_against_a_missing_model_exits_nonzero_and_saves_the_error() {
    let out = tmp("missing");
    let s = server(&["qwen3"]);
    let o = bin()
        .args(["--out", out.to_str().unwrap(), "bench", "--target", &format!("openai:{}", s.url)])
        .args(["--model", "nope", "--prompt", "x"])
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(stdout(&o).contains("model not found"), "{}", stdout(&o));
    let saved = store::load_all(&out);
    assert_eq!(saved[0].error.as_ref().unwrap().kind, ErrorKind::ModelNotFound);
}

#[test]
fn matrix_runs_every_model_on_every_backend_and_persists_json() {
    let out = tmp("matrix");
    let a = server(&["m1", "m2"]);
    let b = server(&["m1"]);
    let file = out.join("matrix.toml");
    std::fs::write(
        &file,
        format!(
            r#"
prompt = "say numbers"
max_tokens = 3
[[backends]]
label = "alpha"
kind = "openai"
url = "{}"
[[backends]]
label = "beta"
kind = "openai"
url = "{}"
[[models]]
name = "m1"
[[models]]
name = "m2"
"#,
            a.url, b.url
        ),
    )
    .unwrap();
    let o = bin().args(["--out", out.to_str().unwrap(), "matrix", file.to_str().unwrap()]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let saved = store::load_all(&out);
    assert_eq!(saved.len(), 4);
    let beta_m2 = saved.iter().find(|r| r.backend == "beta" && r.model == "m2").unwrap();
    assert_eq!(beta_m2.error.as_ref().unwrap().kind, ErrorKind::ModelNotFound);
    assert!(saved.iter().filter(|r| r.error.is_none()).all(|r| r.decode_tok_s.is_some()));
    let listed = bin().args(["--out", out.to_str().unwrap(), "results", "--json"]).output().unwrap();
    let again: Vec<RunResult> = serde_json::from_str(&stdout(&listed)).unwrap();
    assert_eq!(again.len(), 4);
}

#[test]
fn batch_runs_each_size_concurrently_and_prints_json() {
    let s = server(&["m"]);
    let o = bin()
        .args(["batch", "--target", &format!("openai:{}", s.url), "--label", "fake", "--model", "m"])
        .args(["--prompt-len", "64", "--max-tokens", "3", "--sizes", "1,4", "--json"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let printed: Vec<inference_bench_core::batch::BatchResult> = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(printed.iter().map(|r| r.batch).collect::<Vec<_>>(), vec![1, 4]);
    assert_eq!(printed[1].completion_tokens, 12);
    assert!(printed.iter().all(|r| r.backend == "fake" && r.decode_tok_s > 0.0));
    let sent = s.requests.lock().unwrap();
    assert!(sent.iter().any(|r| r.contains("Summarise the following text")));
}

#[test]
fn an_unreachable_openai_backend_is_a_failed_run_not_a_crash() {
    let out = tmp("dead");
    let o = bin()
        .args(["--out", out.to_str().unwrap(), "bench", "--target", "openai:http://127.0.0.1:1"])
        .args(["--model", "m", "--prompt", "x"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(store::load_all(&out)[0].error.as_ref().unwrap().kind, ErrorKind::Unreachable);
}

#[test]
fn run_without_a_backend_flag_runs_the_native_engine() {
    let out = tmp("native");
    let spec = out.join("missing.toml");
    let o = bin()
        .args(["--out", out.to_str().unwrap(), "run", "--model", spec.to_str().unwrap(), "--prompt", "x", "--json"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1), "{}", String::from_utf8_lossy(&o.stderr));
    let printed: Vec<RunResult> = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(printed[0].backend, "specrt");
    assert_eq!(printed[0].error.as_ref().unwrap().kind, ErrorKind::ModelNotFound);
}

#[test]
fn the_removed_backend_flag_explains_where_other_engines_went() {
    for args in [
        vec!["run", "--backend", "openai", "--model", "m", "--prompt", "x"],
        vec!["run", "--backend", "transformers", "--model", "m", "--prompt", "x"],
        vec!["bench", "--backend", "openai", "--model", "m", "--prompt", "x"],
    ] {
        let o = bin().args(&args).output().unwrap();
        assert!(!o.status.success());
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("--backend was removed") && err.contains("--target openai:URL"), "{err}");
    }
}

#[test]
fn an_unknown_flag_or_target_is_a_clear_error() {
    let o = bin().args(["run", "--url", "http://x", "--model", "m", "--prompt", "x"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unexpected argument '--url'"));
    let o = bin().args(["bench", "--target", "transformers", "--model", "m", "--prompt", "x"]).output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown target transformers"));
}
