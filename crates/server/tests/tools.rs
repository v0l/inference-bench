mod common;

use common::*;
use serde_json::{Value, json};

const AGENT: &str = "tool_parser = 'qwen3_coder'\nreasoning_parser = 'qwen3'\n[chat_template_kwargs]\nreasoning_effort = 'medium'\n[[models]]\nid='m'\n";

fn pieces(text: &[&str]) -> Shape {
    Shape { pieces: text.iter().map(|s| s.to_string()).collect(), token_delay_ms: 0, ..Shape::default() }
}

fn weather() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}])
}

const CALL: [&str; 9] = [
    "Checking.\n",
    "</thi",
    "nk>\n\nLet me look.",
    "<tool_",
    "call>\n<function=get_weather>\n<parameter=city>\nDal",
    "las\n</parameter>\n<parameter=days>\n",
    "3\n</parameter>\n</function>\n</tool_call>",
    "<|im_end|>",
    "",
];

fn agent_body(stream: bool) -> Value {
    json!({
        "model": "m",
        "stream": stream,
        "max_tokens": 50,
        "tools": weather(),
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "sunny"},
            {"role": "user", "content": "and Dallas?"}
        ]
    })
}

fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

#[test]
fn a_tool_call_comes_back_parsed_with_reasoning_split_off() {
    let rig = start(config(AGENT), pieces(&CALL[..7]));
    let (status, body, _) = rig.post("/v1/chat/completions", None, agent_body(false));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let m = &choice["message"];
    assert_eq!(m["reasoning_content"], "Checking.\n");
    assert_eq!(m["content"], "Let me look.");
    assert_eq!(m["tool_calls"][0]["type"], "function");
    assert_eq!(m["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], r#"{"city": "Dallas", "days": 3}"#);
    assert!(m["tool_calls"][0]["id"].as_str().unwrap().starts_with("chatcmpl-tool-"));
    let r = rig.stats.requests.lock().unwrap()[0].clone();
    assert_eq!(r.template_vars["reasoning_effort"], "medium");
    assert_eq!(r.template_vars["tools"], weather());
    let history = r.openai_messages.unwrap();
    assert_eq!(history[1]["tool_calls"][0]["id"], "c0");
    assert_eq!(history[2]["role"], "tool");
    rig.stop();
}

#[test]
fn a_streamed_tool_call_split_across_chunks_arrives_as_openai_deltas() {
    let rig = start(config(AGENT), pieces(&CALL[..7]));
    let (status, body, _) = rig.post("/v1/chat/completions", None, agent_body(true));
    assert_eq!(status, 200, "{body}");
    let ev = events(&body);
    let deltas: Vec<&Value> = ev.iter().map(|e| &e["choices"][0]["delta"]).collect();
    let text = |k: &str| deltas.iter().filter_map(|d| d[k].as_str()).collect::<String>();
    assert_eq!(text("reasoning_content"), "Checking.\n");
    assert_eq!(text("content"), "\n\nLet me look.");
    let calls: Vec<&Value> = deltas.iter().filter_map(|d| d["tool_calls"].get(0)).collect();
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert!(calls[0]["id"].is_string());
    let args: String = calls.iter().filter_map(|c| c["function"]["arguments"].as_str()).collect();
    assert_eq!(args, r#"{"city": "Dallas", "days": 3}"#);
    assert!(calls.iter().all(|c| c["index"] == 0));
    assert_eq!(ev.iter().rev().find_map(|e| e["choices"][0]["finish_reason"].as_str()), Some("tool_calls"));
    assert!(body.trim_end().ends_with("data: [DONE]"));
    rig.stop();
}

#[test]
fn tool_choice_none_leaves_the_markup_as_content() {
    let rig = start(config(AGENT), pieces(&["</think>", "<tool_call>x</tool_call>"]));
    let mut body = agent_body(false);
    body["tool_choice"] = json!("none");
    let (_, body, _) = rig.post("/v1/chat/completions", None, body);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "<tool_call>x</tool_call>");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    rig.stop();
}

#[test]
fn enable_thinking_false_keeps_everything_in_content_and_reaches_the_template() {
    let rig = start(config(AGENT), pieces(&["plain ", "answer"]));
    let mut body = agent_body(false);
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    let (_, body, _) = rig.post("/v1/chat/completions", None, body);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "plain answer");
    assert!(v["choices"][0]["message"].get("reasoning_content").is_none());
    assert_eq!(rig.stats.requests.lock().unwrap()[0].template_vars["enable_thinking"], false);
    rig.stop();
}

#[test]
fn reasoning_effort_sets_the_template_kwargs() {
    let rig = start(config(AGENT), pieces(&["a</think>b"]));
    let mut body = agent_body(false);
    body["reasoning_effort"] = json!("low");
    rig.post("/v1/chat/completions", None, body);
    let vars = rig.stats.requests.lock().unwrap()[0].template_vars.clone();
    assert_eq!((vars["reasoning_effort"].as_str(), vars["enable_thinking"].as_bool()), (Some("low"), Some(true)));
    rig.stop();
}

#[test]
fn without_parsers_configured_thinking_is_off_and_text_is_raw() {
    let rig = start(config("[[models]]\nid='m'\n"), pieces(&["<think>x</think>y"]));
    let (_, body, _) = rig.post("/v1/chat/completions", None, json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "<think>x</think>y");
    assert_eq!(rig.stats.requests.lock().unwrap()[0].template_vars["enable_thinking"], false);
    rig.stop();
}

#[test]
fn a_stop_string_split_across_chunks_cuts_the_stream_and_ends_with_stop() {
    let shape = pieces(&["one two ", "EN", "D three", " four"]);
    let rig = start(config("[[models]]\nid='m'\n"), shape);
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": ["END"], "stream": true, "max_tokens": 4});
    let (_, text, _) = rig.post("/v1/chat/completions", None, body);
    let ev = events(&text);
    let content: String = ev.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "one two ");
    assert_eq!(ev.iter().rev().find_map(|e| e["choices"][0]["finish_reason"].as_str()), Some("stop"));
    let (_, text, _) = rig.post("/v1/completions", None, json!({"model": "m", "prompt": "x", "stop": "two", "max_tokens": 4}));
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["choices"][0]["text"], "one ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    rig.stop();
}

#[test]
fn malformed_tool_fields_are_rejected() {
    let rig = start(config(AGENT), Shape::default());
    for (field, value) in [("tools", json!("x")), ("tool_choice", json!("required")), ("stop", json!(3)), ("chat_template_kwargs", json!([]))] {
        let mut body = agent_body(false);
        body[field] = value;
        let (status, text, _) = rig.post("/v1/chat/completions", None, body);
        assert_eq!(status, 400, "{field}: {text}");
    }
    rig.stop();
}
