mod common;

use common::*;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

fn serving_rig(extra: &str) -> Rig {
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs/qwen3-0.6b.toml");
    start_with(
        config(&format!("max_num_seqs = 8\nmax_model_len = 4096\nkv_cache_gb = 0.25\n{extra}[[models]]\nid='q'\nmodel={model:?}\n")),
        inference_bench_server::NativeFactory,
        Default::default(),
    )
}

fn completion(prompt: &str, n: usize, stream: bool) -> Value {
    json!({"model": "q", "prompt": prompt, "max_tokens": n, "temperature": 0, "stream": stream})
}

fn metric(text: &str, name: &str) -> f64 {
    text.lines().find_map(|l| l.strip_prefix(&format!("{name} "))).unwrap_or_else(|| panic!("{name} in {text}")).parse().unwrap()
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn concurrent_requests_are_batched_and_match_their_solo_output() {
    let rig = serving_rig("");
    let prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time in a small village",
        "The three laws of motion are",
        "SELECT name FROM users WHERE",
        "A haiku about the sea:",
    ];
    let solo: Vec<String> = prompts
        .iter()
        .map(|p| {
            let (s, body, _) = rig.post("/v1/completions", None, completion(p, 48, false));
            assert_eq!(s, 200, "{body}");
            serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["text"].as_str().unwrap().to_string()
        })
        .collect();
    let url = rig.url.clone();
    let started = Instant::now();
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (url, body) = (url.clone(), completion(p, 48, false));
            std::thread::spawn(move || {
                let mut r = agent().post(format!("{url}/v1/completions")).send_json(&body).unwrap();
                let v: Value = serde_json::from_str(&r.body_mut().read_to_string().unwrap()).unwrap();
                (v["choices"][0]["text"].as_str().unwrap().to_string(), v["usage"]["completion_tokens"].as_u64().unwrap())
            })
        })
        .collect();
    let together: Vec<(String, u64)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let wall = started.elapsed();
    for (i, (text, n)) in together.iter().enumerate() {
        assert_eq!(text, &solo[i], "prompt {i}");
        assert!(*n > 0 && *n <= 48);
    }
    let m = rig.metrics();
    assert!(metric(&m, "inference_batch_size_sum") > metric(&m, "inference_batch_size_count") * 1.5, "{m}");
    assert_eq!(metric(&m, "inference_kv_blocks_used"), 0.0);
    assert_eq!(metric(&m, "inference_running_seqs"), 0.0);
    assert!(metric(&m, "inference_kv_blocks_total") > 0.0);
    assert!(metric(&m, "inference_inter_token_seconds_count") > 100.0);
    eprintln!("6 concurrent completions of 48 tokens in {wall:?}");
    rig.stop();
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn a_dropped_stream_frees_its_kv_blocks_within_a_step() {
    let rig = serving_rig("");
    let body = completion("Write a very long story about a dragon.", 1500, true).to_string();
    let mut s = rig.raw_post("/v1/completions", &body);
    let seen = read_until(&mut s, "data:");
    assert!(seen.contains("data:"), "{seen}");
    let m = rig.metrics();
    assert!(metric(&m, "inference_kv_blocks_used") > 0.0, "{seen}\n{m}");
    drop(s);
    let dropped = Instant::now();
    wait_for("blocks freed", || metric(&rig.metrics(), "inference_kv_blocks_used") == 0.0);
    assert!(dropped.elapsed() < Duration::from_secs(2), "{:?}", dropped.elapsed());
    wait_for("cancel counted", || rig.metrics().contains("inference_requests_cancelled_total 1"));
    rig.stop();
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn a_request_longer_than_the_model_length_is_a_400_not_a_crash() {
    let rig = serving_rig("");
    let long = "word ".repeat(5000);
    let (status, body, _) = rig.post("/v1/completions", None, completion(&long, 16, false));
    assert!((400..500).contains(&status), "{status} {body}");
    assert!(body.contains("exceeds the context"), "{body}");
    let (status, body, _) = rig.post("/v1/completions", None, completion("hi", 3000, false));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("can never fit"), "{body}");
    assert_eq!(rig.post("/v1/completions", None, completion("hi", 4, false)).0, 200);
    rig.stop();
}
