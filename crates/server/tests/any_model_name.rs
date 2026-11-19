mod common;

use common::*;
use serde_json::{Value, json};

const ONE: &str = "any_model_name = true\n[[models]]\nid='m'\n";

fn chat(model: &str, stream: bool) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "say hello now"}], "max_tokens": 4, "stream": stream})
}

#[test]
fn one_model_answers_to_any_name_and_echoes_it() {
    let rig = start(config(ONE), Shape::default());
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let body = match path {
            "/v1/completions" => json!({"model": "old-name", "prompt": "hello", "max_tokens": 4}),
            _ => chat("old-name", false),
        };
        let (status, text, _) = rig.post(path, None, body);
        assert_eq!(status, 200, "{text}");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "old-name");
    }
    let (_, text, _) = rig.get("/v1/models", None);
    assert!(text.contains("\"m\"") && !text.contains("old-name"), "{text}");
    rig.stop();
}

#[test]
fn the_real_name_is_still_echoed_as_is() {
    let rig = start(config(ONE), Shape::default());
    let (status, text, _) = rig.post("/v1/chat/completions", None, chat("m", false));
    assert_eq!(status, 200, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "m");
    rig.stop();
}

#[test]
fn streaming_chunks_carry_the_requested_name() {
    let rig = start(config(ONE), Shape::default());
    let (status, text, _) = rig.post("/v1/chat/completions", None, chat("old-name", true));
    assert_eq!(status, 200, "{text}");
    let chunks: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|c| c["model"] == "old-name"), "{text}");
    rig.stop();
}

#[test]
fn the_anthropic_and_responses_dialects_follow_the_same_path() {
    let rig = start(config(ONE), Shape::default());
    let body = json!({"model": "claude-x", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]});
    let (status, text, _) = rig.post("/v1/messages", None, body.clone());
    assert_eq!(status, 200, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "claude-x");
    let streamed = json!({"model": "claude-x", "max_tokens": 4, "stream": true, "messages": [{"role": "user", "content": "hi"}]});
    let (status, text, _) = rig.post("/v1/messages", None, streamed);
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("\"model\":\"claude-x\""), "{text}");
    let resp = json!({"model": "gpt-x", "input": "hi", "max_output_tokens": 16});
    let (status, text, _) = rig.post("/v1/responses", None, resp);
    assert_eq!(status, 200, "{text}");
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["model"], "gpt-x");
    rig.stop();
}

#[test]
fn two_models_still_refuse_an_unknown_name() {
    let rig = start(config("any_model_name = true\n[[models]]\nid='m'\n[[models]]\nid='n'"), Shape::default());
    let (status, text, _) = rig.post("/v1/chat/completions", None, chat("old-name", false));
    assert_eq!(status, 404, "{text}");
    rig.stop();
}

#[test]
fn the_option_off_refuses_an_unknown_name() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (status, text, _) = rig.post("/v1/chat/completions", None, chat("old-name", false));
    assert_eq!(status, 404, "{text}");
    let (status, text, _) = rig.post("/v1/messages", None, json!({"model": "old-name", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(status, 404, "{text}");
    rig.stop();
}
