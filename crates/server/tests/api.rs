mod common;

use common::*;
use serde_json::{Value, json};
use std::sync::Arc;
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
fn a_served_name_from_server_toml_resolves_to_its_model_and_is_not_listed() {
    let rig = start(config("[[models]]\nid='m'\nserved_names=['qwen3.8-27b']\n[[models]]\nid='n'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("qwen3.8-27b", false));
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["model"], "m");
    let (_, body, _) = rig.get("/v1/models", None);
    assert!(!body.contains("qwen3.8-27b"), "{body}");
    rig.stop();
    assert!(inference_bench_server::ServerConfig::parse("[[models]]\nid='m'\nserved_names=['n']\n[[models]]\nid='n'").is_err());
}

#[test]
fn a_developer_message_reaches_the_template_as_system() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let body = json!({"model": "m", "messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}], "max_tokens": 2});
    let (status, body, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 200, "{body}");
    let requests = rig.stats.requests.lock().unwrap();
    let sent = requests.last().unwrap();
    assert_eq!(sent.openai_messages.as_ref().unwrap()[0]["role"], "system");
    assert_eq!(sent.messages[0].role, inference_bench_core::chat::Role::System);
    drop(requests);
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
fn usage_reports_cached_prompt_tokens_in_the_vllm_shape() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (_, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let mut streamed = chat("m", true);
    streamed["stream_options"] = json!({"include_usage": true});
    let (_, body, _) = rig.post("/v1/chat/completions", None, streamed);
    assert!(body.contains("\"prompt_tokens_details\":{\"cached_tokens\":0}"), "{body}");
    rig.stop();
}

#[test]
fn the_cache_salt_is_the_body_salt_without_keys_and_per_key_with_keys() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let mut body = chat("m", false);
    body["cache_salt"] = json!("tenant");
    let (status, text, _) = rig.post("/v1/chat/completions", None, body.clone());
    assert_eq!(status, 200, "{text}");
    rig.post("/v1/chat/completions", None, chat("m", false));
    let salts: Vec<_> = rig.stats.requests.lock().unwrap().iter().map(|r| r.cache_salt.clone().unwrap()).collect();
    assert_eq!(salts, ["tenant", ""]);
    rig.stop();

    let rig = start(config("api_keys = [\"k1\", \"k2\"]\n[[models]]\nid='m'"), Shape::default());
    rig.post("/v1/chat/completions", Some("k1"), chat("m", false));
    rig.post("/v1/chat/completions", Some("k2"), chat("m", false));
    rig.post("/v1/chat/completions", Some("k1"), body.clone());
    let salts: Vec<_> = rig.stats.requests.lock().unwrap().iter().map(|r| r.cache_salt.clone().unwrap()).collect();
    assert_ne!(salts[0], salts[1], "two keys never share a cache");
    assert_ne!(salts[0], salts[2], "a body salt narrows a key further");
    assert!(salts[2].ends_with("|tenant") && !salts[2].contains("k1"));
    rig.stop();

    let rig = start(config("api_keys = [\"k1\", \"k2\"]\nprefix_cache_share_across_keys = true\n[[models]]\nid='m'"), Shape::default());
    rig.post("/v1/chat/completions", Some("k1"), chat("m", false));
    rig.post("/v1/chat/completions", Some("k2"), chat("m", false));
    let salts: Vec<_> = rig.stats.requests.lock().unwrap().iter().map(|r| r.cache_salt.clone().unwrap()).collect();
    assert_eq!(salts[0], salts[1]);
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
    let shape = Shape { tokens: 3, first_delay_ms: 600, token_delay_ms: 1, ..Shape::default() };
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
fn a_hung_engine_marks_the_model_failed_answers_503_and_reloads_on_demand() {
    let rig = start(config("max_num_seqs = 1\n[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    rig.stats.hung.store(true, Ordering::SeqCst);
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 200, "reload_failed_models defaults on: {body}");
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 2);
    rig.stop();
    let rig = start(config("max_num_seqs = 1\nreload_failed_models = false\n[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    rig.stats.hung.store(true, Ordering::SeqCst);
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 503, "{body}");
    assert!(body.contains("stopped responding"), "{body}");
    let (_, admin, _) = rig.get("/admin/models", None);
    assert!(admin.contains("stopped responding"), "{admin}");
    rig.stats.hung.store(false, Ordering::SeqCst);
    assert_eq!(rig.post("/admin/models/m/load", None, json!({})).0, 200);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    rig.stop();
}

#[test]
fn client_disconnect_cancels_generation() {
    let shape = Shape { tokens: 500, first_delay_ms: 0, token_delay_ms: 20, ..Shape::default() };
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
    let rig = start(config("[[models]]\nid='m'"), Shape { tokens: 5, first_delay_ms: 5, token_delay_ms: 2, ..Shape::default() });
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
    let shape = Shape { tokens: 5, first_delay_ms: 300, token_delay_ms: 20, ..Shape::default() };
    let mut rig = start(config("[[models]]\nid='m'"), shape);
    let mut s = rig.raw_post("/v1/chat/completions", &chat("m", true).to_string());
    wait_for("in flight", || rig.metrics().contains("inference_in_flight 1"));
    let handle = rig.handle.take().unwrap();
    handle.shutdown();
    let out = read_until(&mut s, "[DONE]");
    assert!(out.contains("w3 ") && out.contains("[DONE]"), "{out}");
    rig.rt.block_on(async { tokio::time::timeout(Duration::from_secs(5), handle.wait()).await.unwrap().unwrap() });
}

#[test]
fn streaming_sends_several_chunks_spread_over_time_before_done() {
    let shape = Shape { tokens: 8, first_delay_ms: 0, token_delay_ms: 60, ..Shape::default() };
    let rig = start(config("[[models]]\nid='m'"), shape);
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "go"}], "max_tokens": 8, "stream": true});
    let mut s = rig.raw_post("/v1/chat/completions", &body.to_string());
    let started = std::time::Instant::now();
    let mut first_content = None;
    let mut seen = String::new();
    while !seen.contains("[DONE]") {
        let more = read_until(&mut s, "\n\n");
        if more.is_empty() {
            break;
        }
        seen.push_str(&more);
        if first_content.is_none() && seen.contains("\"content\":\"w0 ") {
            first_content = Some(started.elapsed());
        }
    }
    let total = started.elapsed();
    let chunks = seen.matches("\"content\":\"w").count();
    assert!(chunks >= 6, "{chunks} content chunks in {seen}");
    assert!(first_content.unwrap() < total / 2, "first {:?} total {total:?}", first_content);
    rig.stop();
}

#[test]
fn the_server_serves_a_model_through_the_in_process_native_engine() {
    use inference_bench_core::fake::{Script, ScriptedModel};
    use inference_bench_core::native::Native;
    let script = Script {
        models: vec!["Qwen/Qwen3-0.6B".into()],
        chunks: vec!["Hel".into(), "lo".into()],
        usage: Some((5, 2)),
        ..Script::default()
    };
    let seen_ctx = std::sync::Arc::new(std::sync::Mutex::new(None));
    let seen = seen_ctx.clone();
    let factory = move |max_ctx: Option<usize>| -> Result<Box<dyn inference_bench_core::target::Target>, inference_bench_core::result::BackendError> {
        *seen.lock().unwrap() = max_ctx;
        Ok(Box::new(Native::with_factory(max_ctx, ScriptedModel::factory(script.clone()))))
    };
    let rig = start_with(
        config("max_ctx = 4096\n[[models]]\nid='qwen'\nmodel='Qwen/Qwen3-0.6B'"),
        factory,
        Default::default(),
    );
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("qwen", false));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello");
    assert_eq!(*seen_ctx.lock().unwrap(), Some(4096));
    let (_, body, _) = rig.get("/admin/models", None);
    assert!(body.contains("\"backend\":\"specrt\""), "{body}");
    rig.stop();
}

#[test]
fn a_native_model_missing_on_disk_is_a_clean_error() {
    use inference_bench_core::fake::{Script, ScriptedModel};
    use inference_bench_core::native::Native;
    let factory = |max_ctx: Option<usize>| -> Result<Box<dyn inference_bench_core::target::Target>, inference_bench_core::result::BackendError> {
        Ok(Box::new(Native::with_factory(max_ctx, ScriptedModel::factory(Script::default()))))
    };
    let rig = start_with(config("[[models]]\nid='gone'"), factory, Default::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("gone", false));
    assert!(status >= 400, "{status} {body}");
    assert!(body.contains("no config.json"), "{body}");
    rig.stop();
}

#[test]
fn the_server_checks_specs_at_start_reports_them_and_loads_a_new_spec_by_name() {
    use inference_bench_core::fake::{FakeServer, Script, publish_specs};
    let data = std::env::temp_dir().join(format!("ib-server-specs-update-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs/qwen3-0.6b.toml")).unwrap();
    let fresh = base.replace("name = \"qwen3-0.6b\"", "name = \"brand-new\"");
    let script = Script::default();
    let github = FakeServer::start(script.clone());
    publish_specs(&script, "v0l/inference-bench", "refs/heads/master", "cafe123", "e1", &[("brand-new.toml", fresh.as_bytes())]);
    let extra = format!(
        "[[models]]\nid = 'new'\nmodel = 'brand-new'\n[specs]\nurl = {:?}\ndata_dir = {:?}\n",
        github.url,
        data.display().to_string()
    );
    let rig = start(config(&extra), Shape::default());
    wait_for("spec check", || rig.metrics().contains("inference_specs_checks_total{result=\"updated\"} 1"));
    let m = rig.metrics();
    let at: i64 = m.lines().find_map(|l| l.strip_prefix("inference_specs_last_success_timestamp_seconds ")).unwrap().parse().unwrap();
    assert!(at > 1_700_000_000, "{m}");
    let (status, body, _) = rig.get("/admin/specs", None);
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["source"], "v0l/inference-bench@master");
    assert_eq!(v["generation"]["commit"], "cafe123");
    let new = v["specs"].as_array().unwrap().iter().find(|s| s["name"] == "brand-new").unwrap();
    assert_eq!(new["origin"], "github");
    assert!(v["specs"].as_array().unwrap().iter().any(|s| s["name"] == "llama-3.2-1b" && s["origin"] == "embedded"));
    assert_eq!(rig.post("/v1/chat/completions", None, chat("new", false)).0, 200);
    let loaded = rig.stats.loaded_names.lock().unwrap().clone();
    assert_eq!(loaded, vec![data.join("specs/current/brand-new.toml").display().to_string()]);
    rig.stop();
}

#[test]
fn the_server_makes_no_spec_request_when_updates_are_off() {
    use inference_bench_core::fake::{FakeServer, Script};
    let github = FakeServer::start(Script::default());
    let extra = format!("[specs]\nauto_update = false\nurl = {:?}\ndata_dir = {:?}\n", github.url, std::env::temp_dir().join("ib-server-specs-off").display().to_string());
    let rig = start(config(&extra), Shape::default());
    assert!(rig.metrics().contains("inference_specs_last_success_timestamp_seconds 0"));
    std::thread::sleep(Duration::from_millis(200));
    assert!(github.requests.lock().unwrap().is_empty());
    rig.stop();
}

fn image_chat(url: &str) -> Value {
    json!({"model": "m", "max_tokens": 2, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]})
}

const PIXEL: &str = "data:image/png;base64,iVBORw0KGgo=";

#[test]
fn the_image_refusal_names_the_reason_the_engine_gives_not_the_config() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    *rig.stats.refusal.lock().unwrap() = Some("the engine did not load a vision encoder (no GPU had room for it)".into());
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(PIXEL));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("no GPU had room for it"), "{body}");
    assert!(!body.contains("vision_config"), "{body}");
    rig.stop();
}

#[test]
fn image_input_to_a_model_without_vision_is_a_400() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(PIXEL));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("vision_config"), "{body}");
    rig.stop();
}

#[test]
fn image_input_reaches_a_vision_model_as_media() {
    let rig = start(config("[[models]]\nid='m'"), Shape { vision: true, ..Shape::default() });
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(PIXEL));
    assert_eq!(status, 200, "{body}");
    let seen = rig.stats.requests.lock().unwrap();
    assert_eq!(seen[0].media.len(), 1);
    assert_eq!(seen[0].media[0].bytes.as_slice(), &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    drop(seen);
    rig.stop();
}

#[test]
fn url_fetching_is_off_by_default_and_the_limits_come_from_config() {
    let rig = start(config("[[models]]\nid='m'"), Shape { vision: true, ..Shape::default() });
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat("http://127.0.0.1:1/a.png"));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("disabled"), "{body}");
    rig.stop();
    let rig = start(config("[[models]]\nid='m'\n[media]\nmax_images = 0"), Shape { vision: true, ..Shape::default() });
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(PIXEL));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("at most 0 images"), "{body}");
    rig.stop();
}

#[test]
fn usage_counts_the_vision_tokens_of_media_and_text_requests_stay_unchanged() {
    let rig = start(config("[[models]]\nid='m'"), Shape { vision: true, ..Shape::default() });
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(PIXEL));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["usage"]["prompt_tokens_details"]["vision_tokens"], 100);
    let (_, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["usage"]["prompt_tokens_details"].get("vision_tokens").is_none());
    rig.stop();
}

#[test]
fn a_request_body_above_the_old_two_megabyte_default_is_accepted_when_media_limits_allow_it() {
    let rig = start(config("[[models]]\nid='m'"), Shape { vision: true, ..Shape::default() });
    let big = format!("data:image/png;base64,{}", "A".repeat(3 << 20));
    let (status, body, _) = rig.post("/v1/chat/completions", None, image_chat(&big));
    assert_ne!(status, 413, "{body}");
    rig.stop();
    let rig = start(config("[[models]]\nid='m'\n[media]\nmax_images = 1\nmax_videos = 0\nmax_image_bytes = 1000"), Shape { vision: true, ..Shape::default() });
    match agent().post(format!("{}/v1/chat/completions", rig.url)).send_json(image_chat(&big)) {
        Ok(resp) => assert_eq!(resp.status().as_u16(), 413),
        Err(ureq::Error::Io(e)) => assert!(matches!(e.kind(), std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset), "{e}"),
        Err(e) => panic!("{e}"),
    }
    rig.stop();
}

fn get_accept(rig: &Rig, path: &str, accept: &str) -> (u16, String) {
    let mut resp = agent().get(format!("{}{path}", rig.url)).header("Accept", accept).call().unwrap();
    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap())
}

const ROUTES: &[&str] = &[
    "/health",
    "/metrics",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/bench",
    "/admin/models",
    "/admin/models/{id}/load",
    "/admin/models/{id}/unload",
    "/admin/specs",
];

#[test]
fn landing_page_is_html_and_lists_every_route() {
    let rig = start(config("[[models]]\nid='alpha'\nmax_ctx=4096"), Shape::default());
    let (status, body, headers) = rig.get("/", None);
    assert_eq!(status, 200);
    assert!(header(&headers, "content-type").unwrap().starts_with("text/html"));
    for path in ROUTES {
        assert!(body.contains(path), "{path} missing");
    }
    assert!(body.contains("alpha") && body.contains("4096") && body.contains("/v1</code>"));
    rig.stop();
}

#[test]
fn landing_page_returns_json_when_asked() {
    let rig = start(config("[[models]]\nid='alpha'"), Shape::default());
    let (status, body) = get_accept(&rig, "/", "application/json");
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["name"], "inference-bench");
    assert_eq!(v["auth_required"], false);
    assert_eq!(v["models"][0]["id"], "alpha");
    let paths: Vec<_> = v["routes"].as_array().unwrap().iter().map(|r| r["path"].as_str().unwrap()).collect();
    for path in ROUTES {
        assert!(paths.contains(path), "{path} missing");
    }
    rig.stop();
}

#[test]
fn landing_page_is_public_and_hides_the_key() {
    let rig = start(config("api_keys = ['sk-secret-one']\n[[models]]\nid='m'"), Shape::default());
    let (status, body, _) = rig.get("/", None);
    assert_eq!(status, 200);
    assert!(!body.contains("sk-secret-one"));
    assert!(body.contains("required"));
    let (_, json) = get_accept(&rig, "/", "application/json");
    assert!(!json.contains("sk-secret-one"));
    rig.stop();
}

fn vllm_shapes() -> Vec<(&'static str, Value)> {
    let user = json!([{"role": "user", "content": "say hello now"}]);
    let tool = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    let base = |extra: Value| {
        let mut b = json!({"model": "qwen3.8-27b", "messages": user, "max_tokens": 4});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    vec![
        ("plain", base(json!({}))),
        ("stream", base(json!({"stream": true}))),
        ("max_completion_tokens", base(json!({"max_completion_tokens": 4}))),
        ("tools", base(json!({"tools": tool, "tool_choice": "auto"}))),
        ("reasoning_effort", base(json!({"reasoning_effort": "low"}))),
        ("parallel_tool_calls", base(json!({"tools": tool, "parallel_tool_calls": true}))),
        ("chat_template_kwargs", base(json!({"chat_template_kwargs": {"enable_thinking": false}}))),
        ("stop list", base(json!({"stop": ["\n\n", "END"]}))),
        ("temperature 2.5", base(json!({"temperature": 2.5}))),
        ("n 2", base(json!({"n": 2}))),
        ("logprobs", base(json!({"logprobs": true, "top_logprobs": 2}))),
        ("logprobs stream", base(json!({"logprobs": true, "top_logprobs": 2, "stream": true}))),
        ("developer role", base(json!({"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]}))),
        ("image_url", base(json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": PIXEL}},
        ]}]}))),
    ]
}

fn compat_rig() -> Rig {
    start(
        config("max_num_seqs = 4\n[[models]]\nid='m'\nserved_names=['qwen3.8-27b']"),
        Shape { vision: true, batched: true, ..Shape::default() },
    )
}

#[test]
fn every_vllm_client_request_shape_is_a_200() {
    let rig = compat_rig();
    for (name, body) in vllm_shapes() {
        let (status, text, _) = rig.post("/v1/chat/completions", None, body);
        assert_eq!(status, 200, "{name}: {text}");
    }
    rig.stop();
}

#[test]
fn response_format_json_object_is_served_by_the_batched_engine() {
    let rig = compat_rig();
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}});
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 200, "{text}");
    rig.stop();
}

#[test]
fn response_format_json_schema_is_served_and_bad_schemas_are_a_400() {
    let rig = compat_rig();
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
    let ok = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}});
    let (status, text, _) = rig.post("/v1/chat/completions", None, ok);
    assert_eq!(status, 200, "{text}");
    let bad = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "wat"}}}});
    let (status, text, _) = rig.post("/v1/chat/completions", None, bad);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("response_format"), "{text}");
    rig.stop();
}

#[test]
fn response_format_json_object_is_a_clear_400_on_the_single_sequence_engine() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}});
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("response_format") && text.contains("max_num_seqs"), "{text}");
    rig.stop();
}

#[test]
fn unknown_response_format_types_are_a_400() {
    let rig = compat_rig();
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "xml"}});
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("response_format"), "{text}");
    rig.stop();
}

#[test]
fn n_choices_come_back_indexed_with_summed_usage() {
    let rig = compat_rig();
    let mut body = chat("m", false);
    body["n"] = json!(3);
    let (status, text, _) = rig.post("/v1/chat/completions", None, body.clone());
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let choices = v["choices"].as_array().unwrap();
    assert_eq!(choices.iter().map(|c| c["index"].as_u64().unwrap()).collect::<Vec<_>>(), [0, 1, 2]);
    assert!(choices.iter().all(|c| c["message"]["content"] == "w0 w1 w2 w3 "));
    assert_eq!(v["usage"]["completion_tokens"], 12);
    body["stream"] = json!(true);
    let (_, text, _) = rig.post("/v1/chat/completions", None, body);
    for i in 0..3 {
        assert_eq!(text.matches(&format!("\"index\":{i},\"delta\":{{}},\"finish_reason\":\"length\"")).count(), 1, "{i}: {text}");
    }
    assert_eq!(text.matches("[DONE]").count(), 1, "{text}");
    for bad in [json!(0), json!(17), json!("2")] {
        let mut b = chat("m", false);
        b["n"] = bad;
        assert_eq!(rig.post("/v1/chat/completions", None, b).0, 400);
    }
    rig.stop();
}

#[test]
fn n_above_one_on_a_single_sequence_engine_is_a_400_naming_max_num_seqs() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let mut body = chat("m", false);
    body["n"] = json!(2);
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("max_num_seqs"), "{text}");
    rig.stop();
}

#[test]
fn chat_logprobs_list_every_token_with_its_top_alternatives() {
    let rig = compat_rig();
    let mut body = chat("m", false);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let (status, text, _) = rig.post("/v1/chat/completions", None, body.clone());
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    assert_eq!(content[0]["token"], "w0 ");
    assert_eq!(content[0]["logprob"], -0.25);
    assert_eq!(content[0]["bytes"], json!("w0 ".as_bytes()));
    assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 2);
    body["stream"] = json!(true);
    let (_, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(text.matches("\"top_logprobs\":[").count(), 4, "{text}");
    let mut plain = chat("m", false);
    plain["top_logprobs"] = json!(2);
    assert_eq!(rig.post("/v1/chat/completions", None, plain).0, 400);
    rig.stop();
}

#[test]
fn completion_logprobs_use_the_legacy_shape() {
    let rig = compat_rig();
    let body = json!({"model": "m", "prompt": "once upon", "max_tokens": 3, "logprobs": 1});
    let (status, text, _) = rig.post("/v1/completions", None, body);
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"], json!(["w0 ", "w1 ", "w2 "]));
    assert_eq!(lp["text_offset"], json!([0, 3, 6]));
    assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), 3);
    rig.stop();
}

#[test]
fn logprobs_on_a_single_sequence_engine_are_a_400() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let mut body = chat("m", false);
    body["logprobs"] = json!(true);
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 400, "{text}");
    rig.stop();
}

#[test]
fn a_busy_gpu_refuses_a_load_with_503_and_retry_after_until_memory_is_free() {
    let free = Arc::new(std::sync::atomic::AtomicU64::new(500));
    let rig = start_probed(config("min_free_vram_mib = 4000\nretry_after_secs = 7\n[[models]]\nid='m'"), Shape::default(), free.clone());
    let (status, body, headers) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 503, "{body}");
    assert_eq!(header(&headers, "retry-after"), Some("7"));
    assert!(body.contains("500 MiB free"), "{body}");
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 0);
    assert_eq!(rig.post("/load", None, json!({})).0, 503);
    free.store(8000, Ordering::SeqCst);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1);
    free.store(10, Ordering::SeqCst);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200, "a loaded model keeps serving");
    rig.stop();
}

#[test]
fn unload_and_load_endpoints_give_the_gpu_back_and_report_unloaded() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.post("/load", None, json!({})).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1);
    let health: Value = serde_json::from_str(&rig.get("/health", None).1).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(rig.post("/unload", None, json!({})).0, 200);
    let health: Value = serde_json::from_str(&rig.get("/health", None).1).unwrap();
    assert_eq!(health["status"], "unloaded");
    assert_eq!(health["unloaded_models"][0], "m");
    let models: Value = serde_json::from_str(&rig.get("/v1/models", None).1).unwrap();
    assert_eq!(models["data"][0]["status"], "unloaded");
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 2);
    rig.stop();
}

#[test]
fn unload_is_409_while_busy_and_waits_when_asked() {
    let shape = Shape { tokens: 30, token_delay_ms: 20, ..Shape::default() };
    let rig = start(config("[[models]]\nid='m'"), shape);
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "go"}], "max_tokens": 30, "stream": true});
    let mut s = rig.raw_post("/v1/chat/completions", &body.to_string());
    assert!(read_until(&mut s, "w1 ").contains("w1 "));
    assert_eq!(rig.post("/unload", None, json!({})).0, 409);
    let (status, body, _) = rig.post("/unload?wait=1", None, json!({}));
    assert_eq!(status, 200, "{body}");
    assert!(read_until(&mut s, "[DONE]").contains("[DONE]"));
    let health: Value = serde_json::from_str(&rig.get("/health", None).1).unwrap();
    assert_eq!(health["status"], "unloaded");
    rig.stop();
}

#[test]
fn lazy_load_skips_the_preload_until_the_first_request() {
    let rig = start(config("lazy_load = true\n[[models]]\nid='m'\npreload = true"), Shape::default());
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 0);
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 1);
    rig.stop();
}

#[test]
fn eager_preload_loads_at_start() {
    let rig = start(config("[[models]]\nid='m'\npreload = true"), Shape::default());
    wait_for("preload", || rig.stats.loads.load(Ordering::SeqCst) == 1);
    rig.stop();
}

#[test]
fn before_load_runs_first_and_a_failure_refuses_the_load() {
    let marker = std::env::temp_dir().join(format!("ib-before-load-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let rig = start(config(&format!("before_load = \"touch {}\"\n[[models]]\nid='m'", marker.display())), Shape::default());
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    assert!(marker.exists());
    rig.stop();
    let rig = start(config("before_load = \"exit 3\"\n[[models]]\nid='m'"), Shape::default());
    let (status, body, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 503, "{body}");
    assert_eq!(rig.stats.loads.load(Ordering::SeqCst), 0);
    rig.stop();
}

#[test]
fn metrics_carry_vllm_names_beside_our_own() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.post("/v1/chat/completions", None, chat("m", false)).0, 200);
    let text = rig.metrics();
    for name in [
        "vllm:num_requests_running{engine=\"0\",model_name=\"m\"}",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total{engine=\"0\",model_name=\"m\"}",
        "vllm:generation_tokens_total{engine=\"0\",model_name=\"m\"}",
        "vllm:request_success_total{engine=\"0\",finished_reason=\"length\",model_name=\"m\"} 1",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:time_to_first_token_seconds_count",
        "inference_ttft_seconds_count",
    ] {
        assert!(text.contains(name), "{name} missing from\n{text}");
    }
    rig.stop();
}

fn session_rig(tag: &str) -> (Rig, Arc<inference_bench_core::specrt_sessions::Shared>, String, String) {
    use inference_bench_core::specrt_sessions::*;
    let dir = std::env::temp_dir().join(format!("api-sessions-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let compat = Compat { model_hash: [1; 32], spec_hash: [2; 32], kv_dtype: 0, layers: 1, block_size: 16 };
    let shared = Arc::new(Shared::new(SessionStore::open(&dir, compat).unwrap(), SessionOptions { enabled: true, ..Default::default() }));
    let session = Session { scope: [0; 32], tokens: (0..5000).collect(), sections: vec![vec![0; 32], vec![1; 100], vec![2; 100], vec![]] };
    let on_disk = shared.store.write(&session, now_seconds()).unwrap().id;
    shared.resident.lock().unwrap().push(Resident { id: "resident-1".into(), tokens: 6000, last_used: now_seconds() });
    let rig = start(config("max_num_seqs = 2\n[[models]]\nid='m'"), Shape::default());
    *rig.stats.sessions.lock().unwrap() = Some(shared.clone());
    assert_eq!(rig.post("/admin/models/m/load", None, json!({})).0, 200);
    (rig, shared, on_disk, dir.display().to_string())
}

#[test]
fn admin_sessions_list_restore_strip_and_delete() {
    let (rig, shared, id, _) = session_rig("ops");
    let (code, body, _) = rig.get("/admin/sessions", None);
    assert_eq!(code, 200);
    let list: Value = serde_json::from_str(&body).unwrap();
    let rows = list["data"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let disk = rows.iter().find(|r| r["id"] == id.as_str()).unwrap();
    assert_eq!((disk["state"].as_str(), disk["tokens"].as_u64(), disk["model"].as_str()), (Some("disk"), Some(5000), Some("m")));
    assert!(disk["bytes"].as_u64().unwrap() > 0);
    assert_eq!(disk["expires"].as_u64().unwrap() - disk["last_used"].as_u64().unwrap(), 8 * 3600);
    assert_eq!(rows.iter().find(|r| r["id"] == "resident-1").unwrap()["state"], "resident");

    let (code, body, _) = rig.post(&format!("/admin/sessions/{id}/restore"), None, json!({}));
    assert_eq!(code, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["verified"], true);

    assert_eq!(rig.post("/admin/sessions/resident-1/save", None, json!({})).0, 200);
    assert_eq!(shared.force.lock().unwrap().as_slice(), ["resident-1".to_string()]);
    assert_eq!(rig.post(&format!("/admin/sessions/{id}/save"), None, json!({})).0, 409);

    let (code, _, _) = rig.post(&format!("/admin/sessions/{id}/strip"), None, json!({}));
    assert_eq!(code, 200);
    let rows: Value = serde_json::from_str(&rig.get("/admin/sessions", None).1).unwrap();
    let stripped = rows["data"].as_array().unwrap().iter().find(|r| r["id"] == id.as_str()).unwrap().clone();
    assert_eq!((stripped["state"].as_str(), stripped["tokens"].as_u64()), (Some("stripped"), Some(5000)));
    assert_eq!(rig.post(&format!("/admin/sessions/{id}/restore"), None, json!({})).0, 409);

    let req = agent().delete(format!("{}/admin/sessions/{id}", rig.url));
    assert_eq!(req.call().unwrap().status().as_u16(), 200);
    assert!(shared.store.list().is_empty());
    assert_eq!(rig.post("/admin/sessions/zzz/restore", None, json!({})).0, 404);
    rig.stop();
}

#[test]
fn a_corrupt_snapshot_is_dropped_when_a_restore_is_requested() {
    let (rig, shared, id, dir) = session_rig("corrupt");
    let path = std::fs::read_dir(shared.store.dir()).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "kvs")).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes.len() - 40;
    bytes[at] ^= 0xff;
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(rig.post(&format!("/admin/sessions/{id}/restore"), None, json!({})).0, 409);
    assert!(!path.exists(), "{dir}");
    rig.stop();
}

#[test]
fn session_routes_need_the_api_key_like_other_admin_routes() {
    let rig = start(config("api_keys = ['k']\n[[models]]\nid='m'"), Shape::default());
    assert_eq!(rig.get("/admin/sessions", None).0, 401);
    assert_eq!(rig.get("/admin/sessions", Some("k")).0, 200);
    rig.stop();
}

#[test]
fn the_sessions_table_reaches_the_serving_options_and_defaults_to_off() {
    let c = inference_bench_server::ServerConfig::parse("max_num_seqs = 4\n[sessions]\nenabled = true\nidle_seconds = 60\nttl_hours = 2\nmin_tokens = 128\n[[models]]\nid='m'").unwrap();
    let o = c.serve_options(&c.models[0]).unwrap().sessions.unwrap();
    assert_eq!((o.enabled, o.idle_seconds, o.ttl_hours, o.min_tokens), (true, 60, 2, 128));
    let d = inference_bench_server::ServerConfig::parse("max_num_seqs = 4\n[[models]]\nid='m'").unwrap();
    let o = d.serve_options(&d.models[0]).unwrap().sessions.unwrap();
    assert_eq!((o.enabled, o.idle_seconds, o.ttl_hours, o.min_tokens), (false, 300, 8, 4096));
    assert!(inference_bench_server::ServerConfig::parse("[sessions]\nidle_seconds = 0").is_err());
}

#[test]
fn purge_sessions_deletes_every_snapshot_under_the_configured_dir() {
    use inference_bench_core::specrt_sessions::*;
    let dir = std::env::temp_dir().join(format!("api-purge-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let compat = Compat { model_hash: [1; 32], spec_hash: [2; 32], kv_dtype: 0, layers: 1, block_size: 16 };
    let store = SessionStore::open(&dir, compat).unwrap();
    for n in [100u32, 101] {
        store.write(&Session { scope: [0; 32], tokens: (0..n).collect(), sections: vec![vec![]; 4] }, 0).unwrap();
    }
    let cfg = dir.join("server.toml");
    std::fs::write(&cfg, format!("[sessions]\ndir = '{}'\n", dir.display())).unwrap();
    let args = inference_bench_server::cli::ServeArgs { config: Some(cfg), purge_sessions: true, ..Default::default() };
    assert_eq!(args.purge().unwrap(), 2);
    assert!(SessionStore::open(&dir, store.compat().clone()).unwrap().list().is_empty());
}
