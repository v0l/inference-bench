mod common;

use common::*;
use serde_json::{Value, json};

#[test]
fn request_id_is_returned_and_present_on_every_line_for_the_request() {
    let rig = start(
        config("api_keys = ['sk-very-secret-key']\n[[models]]\nid='m'"),
        Shape::default(),
    );
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "my private question about taxes"}], "max_tokens": 3});
    let (status, _, headers) = rig.post("/v1/chat/completions", Some("sk-very-secret-key"), body.clone());
    assert_eq!(status, 200);
    let id = header(&headers, "x-request-id").expect("request id header").to_string();
    assert!(id.starts_with("req-"));

    let lines = lines_for(&id);
    assert!(lines.len() >= 3, "{lines:?}");
    let messages: Vec<&str> = lines.iter().map(|l| l["fields"]["message"].as_str().unwrap_or("")).collect();
    assert!(messages.contains(&"model loaded"), "{messages:?}");
    assert!(messages.contains(&"generation finished"), "{messages:?}");
    assert!(messages.contains(&"request finished"), "{messages:?}");

    let finished = lines.iter().find(|l| l["fields"]["message"] == "generation finished").unwrap();
    let f = &finished["fields"];
    assert_eq!(f["prompt_tokens"], 5);
    assert_eq!(f["completion_tokens"], 3);
    assert_eq!(f["finish_reason"], "length");
    assert!(f["ttft_ms"].is_number() && f["decode_tok_s"].is_number());
    let span = finished["spans"].as_array().unwrap().iter().find(|s| s["name"] == "request").unwrap();
    assert_eq!(span["model"], "m");
    assert_eq!(span["backend"], "fake");
    assert_eq!(span["path"], "/v1/chat/completions");

    let loaded = lines.iter().find(|l| l["fields"]["message"] == "model loaded").unwrap();
    assert!(loaded["fields"]["spec_hash"].is_string());
    assert!(loaded["fields"]["load_secs"].is_number());
    assert!(loaded["fields"].get("vram_before_bytes").is_some());

    let (_, _, h2) = rig.post("/v1/chat/completions", Some("sk-very-secret-key"), body);
    assert_ne!(header(&h2, "x-request-id").unwrap(), id);
    rig.stop();
}

#[test]
fn a_client_supplied_request_id_is_kept() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let resp = agent()
        .get(format!("{}/v1/models", rig.url))
        .header("x-request-id", "trace-abc-123")
        .call()
        .unwrap();
    assert_eq!(resp.headers()["x-request-id"], "trace-abc-123");
    assert!(!lines_for("trace-abc-123").is_empty());
    rig.stop();
}

#[test]
fn keys_and_prompts_never_reach_the_logs_at_info() {
    let rig = start(config("api_keys = ['sk-redact-me-please']\n[[models]]\nid='m'"), Shape::default());
    rig.post(
        "/v1/chat/completions",
        Some("sk-redact-me-please"),
        json!({"model": "m", "messages": [{"role": "user", "content": "unique-prompt-marker-77"}], "max_tokens": 2}),
    );
    let (status, _, _) = rig.post(
        "/v1/chat/completions",
        Some("sk-bad-guess-0000000"),
        json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
    );
    assert_eq!(status, 401);
    let all = serde_json::to_string(&log_lines()).unwrap();
    assert!(!all.contains("sk-redact-me-please"), "configured key leaked");
    assert!(!all.contains("sk-bad-guess-0000000"), "presented key leaked");
    let info_and_up: Vec<Value> = log_lines()
        .into_iter()
        .filter(|l| matches!(l["level"].as_str(), Some("INFO" | "WARN" | "ERROR")))
        .collect();
    assert!(!serde_json::to_string(&info_and_up).unwrap().contains("unique-prompt-marker-77"));
    let rejected = log_lines().into_iter().find(|l| l["fields"]["message"] == "rejected request with an invalid API key");
    assert!(rejected.is_some(), "invalid key attempts are logged redacted");
    rig.stop();
}

#[test]
fn prompts_are_logged_when_asked_for() {
    let rig = start(config("log_prompts = true\n[[models]]\nid='m'"), Shape::default());
    rig.post(
        "/v1/completions",
        None,
        json!({"model": "m", "prompt": "visible-prompt-marker-42", "max_tokens": 1}),
    );
    let line = log_lines()
        .into_iter()
        .find(|l| l["fields"]["prompt"].as_str().is_some_and(|p| p.contains("visible-prompt-marker-42")))
        .expect("prompt line");
    assert_eq!(line["level"], "INFO");
    rig.stop();
}

#[test]
fn the_access_log_gets_one_json_line_per_request() {
    let path = std::env::temp_dir().join(format!("ib-access-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let rig = start(
        config(&format!("access_log = {:?}\n[[models]]\nid='m'", path.display().to_string())),
        Shape::default(),
    );
    let (_, _, h) = rig.get("/v1/models", None);
    rig.stop();
    let text = std::fs::read_to_string(&path).unwrap();
    let v: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(v["path"], "/v1/models");
    assert_eq!(v["status"], 200);
    assert_eq!(v["request_id"], header(&h, "x-request-id").unwrap());
}
