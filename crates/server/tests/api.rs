mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use std::time::Duration;

fn chat(model: &str, stream: bool) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "say hello now"}], "max_tokens": 4, "stream": stream})
}

#[test]
fn models_lists_configured_models() {
    let rig = start(config("[[models]]\nid='a'\n[[models]]\nid='b'"), Shape::default());
    let (status, body, _) = rig.get("/v1/models", None);
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    let ids: Vec<_> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(v["data"][0]["object"], "model");
    rig.stop();
}

#[test]
fn non_streaming_chat_matches_the_openai_schema() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["model"], "m");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "w0 w1 w2 w3 ");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert_eq!(v["usage"]["total_tokens"].as_u64().unwrap(), 3 + 4);
    rig.stop();
}

#[test]
fn streaming_chat_emits_openai_chunks_then_done() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let mut body = chat("m", true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, text, headers) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 200);
    assert!(header(&headers, "content-type").unwrap().starts_with("text/event-stream"));
    let events: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim).collect();
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["model"], "m");
        assert!(c["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }
    let content: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "w0 w1 w2 w3 ");
    let finals: Vec<_> = chunks.iter().filter(|c| c["choices"][0]["finish_reason"] == "length").collect();
    assert_eq!(finals.len(), 1);
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 4);
    rig.stop();
}

#[test]
fn completions_stream_and_plain() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (_, body, _) = rig.post("/v1/completions", None, json!({"model": "m", "prompt": "hi there", "max_tokens": 2}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "w0 w1 ");
    let (_, text, _) =
        rig.post("/v1/completions", None, json!({"model": "m", "prompt": "hi", "max_tokens": 2, "stream": true}));
    let first = text.lines().find_map(|l| l.strip_prefix("data:")).unwrap();
    let c: Value = serde_json::from_str(first.trim()).unwrap();
    assert_eq!(c["choices"][0]["text"], "w0 ");
    rig.stop();
}

#[test]
fn bad_requests_and_unknown_models_use_openai_errors() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("nope", false));
    assert_eq!(status, 404);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "not_found_error");
    let (status, _, _) = rig.post("/v1/chat/completions", None, json!({"model": "m", "messages": []}));
    assert_eq!(status, 400);
    let (status, _, _) = rig.post("/v1/embeddings", None, json!({"model": "m", "input": "x"}));
    assert_eq!(status, 501);
    rig.stop();
}

#[test]
fn missing_or_wrong_key_is_401_and_health_stays_open() {
    let rig = start(config("api_keys = ['sk-secret-one']\n[[models]]\nid='m'"), Shape::default());
    let (status, body, headers) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 401);
    assert_eq!(header(&headers, "www-authenticate"), Some("Bearer"));
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "authentication_error");
    assert_eq!(rig.post("/v1/chat/completions", Some("sk-wrong-key-xx"), chat("m", false)).0, 401);
    assert_eq!(rig.post("/v1/chat/completions", Some("sk-secret-one"), chat("m", false)).0, 200);
    assert_eq!(rig.get("/health", None).0, 200);
    assert_eq!(rig.get("/v1/models", None).0, 401);
    assert_eq!(rig.get("/admin/models", None).0, 401);
    rig.stop();
}

#[test]
fn keys_from_the_environment_are_accepted() {
    unsafe { std::env::set_var("IB_TEST_KEYS", "env-key-1, env-key-2") };
    let rig = start(config("api_keys_env = 'IB_TEST_KEYS'\n[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.get("/v1/models", Some("env-key-2")).0, 200);
    assert_eq!(rig.get("/v1/models", Some("env-key-3")).0, 401);
    rig.stop();
}

#[test]
fn full_queue_returns_429_with_retry_after() {
    let shape = Shape { tokens: 3, first_delay_ms: 600, token_delay_ms: 1 };
    let rig = start(config("max_concurrent = 1\nqueue_depth = 1\nretry_after_secs = 7\n[[models]]\nid='m'"), shape);
    let mut held = Vec::new();
    for _ in 0..2 {
        held.push(rig.raw_post("/v1/chat/completions", &chat("m", true).to_string()));
    }
    wait_for("two admitted", || rig.metrics().contains("inference_queue_depth 1"));
    let (status, _, headers) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 429);
    assert_eq!(header(&headers, "retry-after"), Some("7"));
    for mut s in held {
        assert!(read_until(&mut s, "[DONE]").contains("[DONE]"));
    }
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    let m = rig.metrics();
    assert!(m.contains("inference_requests_total{endpoint=\"chat_completions\",status=\"429\"} 1"), "{m}");
    rig.stop();
}

#[test]
fn client_disconnect_cancels_generation() {
    let shape = Shape { tokens: 500, first_delay_ms: 0, token_delay_ms: 20 };
    let rig = start(config("[[models]]\nid='m'"), shape);
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "go"}], "max_tokens": 500, "stream": true});
    let mut s = rig.raw_post("/v1/chat/completions", &body.to_string());
    assert!(read_until(&mut s, "w1 ").contains("w1 "));
    drop(s);
    wait_for("cancellation", || rig.stats.cancelled.load(Ordering::SeqCst) == 1);
    assert_eq!(rig.stats.generated.load(Ordering::SeqCst), 0);
    wait_for("metric", || rig.metrics().contains("inference_requests_cancelled_total 1"));
    wait_for("slot released", || rig.metrics().contains("inference_in_flight 0"));
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    rig.stop();
}

#[test]
fn models_load_lazily_and_unload_when_idle() {
    let rig = start(config("idle_unload_secs = 1\n[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 0);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1, "second request reuses the model");
    assert!(rig.metrics().contains("inference_models_loaded 1"));
    wait_for("idle unload", || rig.metrics().contains("inference_models_loaded 0"));
    assert!(rig.metrics().contains("inference_model_unloads_total 1"));
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 2, "reloaded on demand");
    rig.stop();
}

#[test]
fn admin_endpoints_load_and_unload() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.post("/admin/models/m/load", None, json!({})).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1);
    let (_, body, _) = rig.get("/admin/models", None);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["models"][0]["loaded"], true);
    assert_eq!(rig.post("/admin/models/m/unload", None, json!({})).0, 200);
    let (_, body, _) = rig.get("/admin/models", None);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["models"][0]["loaded"], false);
    assert_eq!(rig.post("/admin/models/zzz/load", None, json!({})).0, 404);
    rig.stop();
}

#[test]
fn one_model_at_a_time_evicts_the_idle_one() {
    let rig = start(config("max_loaded_models = 1\n[[models]]\nid='a'\n[[models]]\nid='b'"), Shape::default());
    rig.post("/v1/chat/completions", None, chat("a", false));
    rig.post("/v1/chat/completions", None, chat("b", false));
    let (_, body, _) = rig.get("/admin/models", None);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["models"][0]["loaded"], false);
    assert_eq!(v["models"][1]["loaded"], true);
    rig.stop();
}

#[test]
fn vram_budget_keeps_several_models_until_it_is_full() {
    let rig = start(
        config(
            "max_loaded_models = 3\nvram_budget_bytes = 250\n[[models]]\nid='a'\nvram_bytes = 100\n[[models]]\nid='b'\nvram_bytes = 100\n[[models]]\nid='c'\nvram_bytes = 100",
        ),
        Shape::default(),
    );
    for id in ["a", "b"] {
        rig.post("/v1/chat/completions", None, chat(id, false));
    }
    assert!(rig.metrics().contains("inference_models_loaded 2"));
    rig.post("/v1/chat/completions", None, chat("c", false));
    assert!(rig.metrics().contains("inference_models_loaded 2"), "{}", rig.metrics());
    rig.stop();
}

#[test]
fn a_failed_load_is_reported_and_counted() {
    let rig = start(config("[[models]]\nid='x'\nmodel='broken'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("x", false));
    assert_eq!(status, 404, "{body}");
    assert!(body.contains("no such checkpoint"));
    assert!(rig.metrics().contains("inference_model_load_failures_total 1"));
    rig.stop();
}

#[test]
fn metrics_counters_move() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let before = rig.metrics();
    assert!(before.contains("inference_ttft_seconds_count 0"));
    rig.post("/v1/chat/completions", None, chat("m", false));
    rig.post("/v1/chat/completions", None, chat("m", true));
    rig.post("/v1/chat/completions", None, chat("nope", false));
    let m = rig.metrics();
    assert!(m.contains("inference_requests_total{endpoint=\"chat_completions\",status=\"200\"} 2"), "{m}");
    assert!(m.contains("inference_requests_total{endpoint=\"chat_completions\",status=\"404\"} 1"), "{m}");
    assert!(m.contains("inference_completion_tokens_total{model=\"m\"} 8"), "{m}");
    assert!(m.contains("inference_prompt_tokens_total{model=\"m\"} 6"), "{m}");
    assert!(m.contains("inference_ttft_seconds_count 2"), "{m}");
    assert!(m.contains("inference_queue_depth 0"));
    assert!(m.contains("inference_gpu_memory_used_bytes 1000000"));
    assert!(m.contains("inference_decode_tokens_per_second{window=\"mean\"}"));
    rig.stop();
}

#[test]
fn bench_runs_a_prompt_set_and_returns_stats() {
    let rig = start(config("[[models]]\nid='m'"), Shape { tokens: 5, first_delay_ms: 5, token_delay_ms: 2 });
    let (status, body, _) =
        rig.post("/v1/bench", None, json!({"model": "m", "prompts": ["one two", "three"], "max_tokens": 5, "repeat": 2}));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["summary"]["requests"], 4);
    assert_eq!(v["summary"]["failures"], 0);
    assert_eq!(v["summary"]["completion_tokens"], 20);
    assert!(v["summary"]["tokens_per_sec"].as_f64().unwrap() > 0.0);
    assert!(v["summary"]["ttft_secs"]["p95"].as_f64().unwrap() >= v["summary"]["ttft_secs"]["p50"].as_f64().unwrap());
    rig.stop();
}

#[test]
fn cors_headers_are_sent_for_allowed_origins() {
    let rig = start(config("cors_origins = ['http://app.example']\n[[models]]\nid='m'"), Shape::default());
    let resp = agent()
        .get(format!("{}/v1/models", rig.url))
        .header("Origin", "http://app.example")
        .call()
        .unwrap();
    assert_eq!(resp.headers()["access-control-allow-origin"], "http://app.example");
    rig.stop();
}

#[test]
fn graceful_shutdown_drains_in_flight_requests() {
    let shape = Shape { tokens: 5, first_delay_ms: 300, token_delay_ms: 20 };
    let mut rig = start(config("[[models]]\nid='m'"), shape);
    let mut s = rig.raw_post("/v1/chat/completions", &chat("m", true).to_string());
    wait_for("in flight", || rig.metrics().contains("inference_in_flight 1"));
    let handle = rig.handle.take().unwrap();
    handle.shutdown();
    let out = read_until(&mut s, "[DONE]");
    assert!(out.contains("w3 ") && out.contains("[DONE]"), "{out}");
    rig.rt.block_on(async { tokio::time::timeout(Duration::from_secs(5), handle.wait()).await.unwrap().unwrap() });
}
