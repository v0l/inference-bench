mod common;

use common::*;
use serde_json::{Value, json};

const AGENT: &str = "tool_parser = 'xml_function'\nreasoning_parser = 'think_tags'\n[[models]]\nid='m'\n";

fn pieces(text: &[&str]) -> Shape {
    Shape { pieces: text.iter().map(|s| s.to_string()).collect(), token_delay_ms: 0, ..Shape::default() }
}

fn weather() -> Value {
    json!([{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}])
}

fn fn_weather() -> Value {
    json!([{"type": "function", "name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}])
}

const CALL: [&str; 6] = [
    "Let me look.\n</thi",
    "nk>\n\nChecking.",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nDal",
    "las\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    "\n",
    "",
];

fn sse(body: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut name = String::new();
    for line in body.lines() {
        if let Some(n) = line.strip_prefix("event: ") {
            name = n.to_string();
        } else if let Some(d) = line.strip_prefix("data: ") {
            if d != "[DONE]" {
                out.push((name.clone(), serde_json::from_str(d).unwrap()));
            }
            name.clear();
        }
    }
    out
}

fn names(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(n, _)| n.as_str()).filter(|n| *n != "ping").collect()
}

fn anthropic(stream: bool) -> Value {
    json!({
        "model": "m", "max_tokens": 200, "stream": stream, "tools": weather(),
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [{"role": "user", "content": "weather in Dallas?"}],
    })
}

#[test]
fn messages_stream_has_the_anthropic_event_sequence_for_thinking_text_and_a_tool() {
    let rig = start(config(AGENT), pieces(&CALL));
    let (status, body, headers) = rig.post("/v1/messages", None, anthropic(true));
    assert_eq!(status, 200, "{body}");
    assert!(header(&headers, "content-type").unwrap().starts_with("text/event-stream"));
    let events: Vec<(String, Value)> = sse(&body).into_iter().filter(|(n, _)| n != "ping").collect();
    assert_eq!(
        names(&events),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[0].1["message"]["role"], "assistant");
    assert_eq!(events[1].1["content_block"]["type"], "thinking");
    assert_eq!(events[2].1["delta"], json!({"type": "thinking_delta", "thinking": "Let me look.\n"}));
    assert_eq!(events[4].1["content_block"]["type"], "text");
    assert_eq!(events[7].1["content_block"]["type"], "tool_use");
    assert_eq!(events[7].1["content_block"]["name"], "get_weather");
    let args = events[8].1["delta"]["partial_json"].as_str().unwrap();
    assert_eq!(serde_json::from_str::<Value>(args).unwrap(), json!({"city": "Dallas", "days": 3}));
    let delta = &events[10].1;
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert!(delta["usage"]["output_tokens"].as_u64().unwrap() > 0);
    let indexes: Vec<u64> = events.iter().filter(|(n, _)| n == "content_block_start").map(|(_, v)| v["index"].as_u64().unwrap()).collect();
    assert_eq!(indexes, [0, 1, 2]);
    rig.stop();
}

#[test]
fn messages_non_stream_returns_blocks_and_usage() {
    let rig = start(config(AGENT), pieces(&CALL));
    let (status, body, _) = rig.post("/v1/messages", None, anthropic(false));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "tool_use");
    let kinds: Vec<&str> = v["content"].as_array().unwrap().iter().map(|b| b["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["thinking", "text", "tool_use"]);
    assert_eq!(v["content"][2]["input"], json!({"city": "Dallas", "days": 3}));
    assert!(v["content"][2]["id"].as_str().unwrap().starts_with("toolu_"));
    assert!(v["usage"]["input_tokens"].as_u64().is_some());
    rig.stop();
}

#[test]
fn messages_without_a_thinking_field_asks_for_no_thinking() {
    let rig = start(config(AGENT), pieces(&["plain answer", ""]));
    let mut body = anthropic(false);
    body.as_object_mut().unwrap().remove("thinking");
    body["tools"] = Value::Null;
    let (status, text, _) = rig.post("/v1/messages", None, body);
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["content"], json!([{"type": "text", "text": "plain answer"}]));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(rig.stats.requests.lock().unwrap()[0].template_vars["enable_thinking"], false);
    rig.stop();
}

#[test]
fn messages_effort_and_budget_reach_the_template() {
    let rig = start(config(AGENT), pieces(&["a</think>b"]));
    let mut body = anthropic(false);
    body.as_object_mut().unwrap().remove("thinking");
    body["output_config"] = json!({"effort": "low"});
    rig.post("/v1/messages", None, body);
    let mut body = anthropic(false);
    body["thinking"] = json!({"type": "disabled"});
    rig.post("/v1/messages", None, body);
    let seen = rig.stats.requests.lock().unwrap();
    assert_eq!((seen[0].template_vars["reasoning_effort"].as_str(), seen[0].template_vars["enable_thinking"].as_bool()), (Some("low"), Some(true)));
    assert_eq!(seen[1].template_vars["enable_thinking"], false);
    drop(seen);
    rig.stop();
}

#[test]
fn messages_tool_results_and_images_reach_the_model_as_chat_turns() {
    let rig = start(config(AGENT), Shape { vision: true, ..pieces(&["ok", ""]) });
    let body = json!({
        "model": "m", "max_tokens": 50, "tools": weather(), "thinking": {"type": "disabled"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="}}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}]},
        ],
    });
    let (status, text, _) = rig.post("/v1/messages", None, body);
    assert_eq!(status, 200, "{text}");
    let seen = rig.stats.requests.lock().unwrap();
    let list = seen[0].openai_messages.clone().unwrap();
    assert_eq!(list[1]["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(list[2], json!({"role": "tool", "tool_call_id": "t1", "content": "sunny"}));
    assert_eq!(seen[0].media.len(), 1);
    drop(seen);
    rig.stop();
}

#[test]
fn messages_max_tokens_maps_to_max_tokens_stop_reason_and_errors_use_the_anthropic_shape() {
    let rig = start(config(AGENT), pieces(&["w ", "w ", "w ", "w "]));
    let body = json!({"model": "m", "max_tokens": 2, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "hi"}]});
    let (_, text, _) = rig.post("/v1/messages", None, body);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["stop_reason"], "max_tokens");
    let (status, text, _) = rig.post("/v1/messages", None, json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(status, 400);
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((v["type"].as_str(), v["error"]["type"].as_str()), (Some("error"), Some("invalid_request_error")));
    rig.stop();
}

#[test]
fn messages_accept_x_api_key() {
    let rig = start(config(&format!("api_keys=['k']\n{AGENT}")), pieces(&["hi"]));
    let body = json!({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]});
    let r = ureq_post(&rig, "/v1/messages", Some("k"), &body);
    assert_eq!(r, 200);
    let r = ureq_post(&rig, "/v1/messages", None, &body);
    assert_eq!(r, 401);
    rig.stop();
}

fn ureq_post(rig: &Rig, path: &str, key: Option<&str>, body: &Value) -> u16 {
    let mut req = agent().post(format!("{}{path}", rig.url));
    if let Some(k) = key {
        req = req.header("x-api-key", k);
    }
    req.send_json(body).unwrap().status().as_u16()
}

fn responses_body(stream: bool) -> Value {
    json!({
        "model": "m", "stream": stream, "max_output_tokens": 200, "tools": fn_weather(),
        "reasoning": {"effort": "medium"}, "instructions": "be brief",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather in Dallas?"}]}],
    })
}

#[test]
fn responses_stream_emits_reasoning_message_and_function_call_items() {
    let rig = start(config(AGENT), pieces(&CALL));
    let (status, body, _) = rig.post("/v1/responses", None, responses_body(true));
    assert_eq!(status, 200, "{body}");
    let events = sse(&body);
    let n = names(&events);
    assert_eq!(n.first(), Some(&"response.created"));
    assert_eq!(n.last(), Some(&"response.completed"));
    for want in [
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ] {
        assert!(n.contains(&want), "{want} in {n:?}");
    }
    let seq: Vec<u64> = events.iter().map(|(_, v)| v["sequence_number"].as_u64().unwrap()).collect();
    assert!(seq.windows(2).all(|w| w[1] == w[0] + 1));
    let done = &events.last().unwrap().1["response"];
    let kinds: Vec<&str> = done["output"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "message", "function_call"]);
    assert_eq!(done["output"][2]["name"], "get_weather");
    assert_eq!(serde_json::from_str::<Value>(done["output"][2]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Dallas", "days": 3}));
    assert!(done["usage"]["output_tokens"].as_u64().unwrap() > 0);
    rig.stop();
}

#[test]
fn responses_non_stream_returns_output_items() {
    let rig = start(config(AGENT), pieces(&CALL));
    let (status, body, _) = rig.post("/v1/responses", None, responses_body(false));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((v["object"].as_str(), v["status"].as_str()), (Some("response"), Some("completed")));
    assert_eq!(v["output"][2]["type"], "function_call");
    let seen = rig.stats.requests.lock().unwrap();
    assert_eq!(seen[0].template_vars["reasoning_effort"], "medium");
    assert_eq!(seen[0].openai_messages.as_ref().unwrap()[0], json!({"role": "system", "content": "be brief"}));
    drop(seen);
    rig.stop();
}

#[test]
fn responses_function_call_history_is_replayed_and_previous_response_id_is_refused() {
    let rig = start(config(AGENT), pieces(&["done", ""]));
    let body = json!({
        "model": "m", "tools": fn_weather(), "reasoning": {"effort": "none"},
        "input": [
            {"type": "message", "role": "user", "content": "weather?"},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ],
    });
    let (status, text, _) = rig.post("/v1/responses", None, body.clone());
    assert_eq!(status, 200, "{text}");
    let list = rig.stats.requests.lock().unwrap()[0].openai_messages.clone().unwrap();
    assert_eq!(list[1]["tool_calls"][0]["id"], "c1");
    assert_eq!(list[2]["role"], "tool");
    let mut stateful = body;
    stateful["previous_response_id"] = json!("resp_1");
    let (status, _, _) = rig.post("/v1/responses", None, stateful);
    assert_eq!(status, 400);
    rig.stop();
}

#[test]
fn reasoning_effort_aliases_set_the_template_on_chat_completions() {
    let rig = start(config(AGENT), pieces(&["a</think>b"]));
    let chat = |extra: Value| {
        let mut b = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        rig.post("/v1/chat/completions", None, b);
    };
    chat(json!({"reasoning": {"effort": "high"}}));
    chat(json!({"reasoning_effort": "none"}));
    chat(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    let seen = rig.stats.requests.lock().unwrap();
    assert_eq!((seen[0].template_vars["reasoning_effort"].as_str(), seen[0].template_vars["enable_thinking"].as_bool()), (Some("high"), Some(true)));
    assert_eq!(seen[1].template_vars["enable_thinking"], false);
    assert_eq!(seen[2].template_vars["enable_thinking"], false);
    drop(seen);
    rig.stop();
}

const THINK: [&str; 8] = ["t1 ", "t2 ", "t3 ", "t4 ", "t5 ", "t6 ", "</think>\n\n", "answer"];

#[test]
fn a_reasoning_budget_closes_the_thinking_with_a_wrapup_and_the_model_answers() {
    let rig = start(config(AGENT), pieces(&THINK));
    let body = json!({"model": "m", "max_tokens": 50, "reasoning_budget_tokens": 3, "messages": [{"role": "user", "content": "hi"}]});
    let (status, text, _) = rig.post("/v1/chat/completions", None, body);
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let m = &v["choices"][0]["message"];
    assert!(m["reasoning_content"].as_str().unwrap().starts_with("t1 t2 t3 "), "{m}");
    assert!(m["reasoning_content"].as_str().unwrap().contains("answer now"), "{m}");
    assert!(m["content"].as_str().unwrap().ends_with("answer"), "{m}");
    assert!(rig.metrics().contains("inference_thinking_wrapups_total 1"));
    rig.stop();
}

#[test]
fn a_budget_that_is_not_reached_changes_nothing() {
    let rig = start(config(AGENT), pieces(&THINK));
    let body = json!({"model": "m", "max_tokens": 50, "reasoning_budget_tokens": 40, "messages": [{"role": "user", "content": "hi"}]});
    let (_, text, _) = rig.post("/v1/chat/completions", None, body);
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "\n\nanswer");
    assert!(!rig.metrics().contains("inference_thinking_wrapups_total 1"));
    rig.stop();
}

#[test]
fn a_max_tokens_cap_leaves_room_for_the_answer_when_there_is_no_budget() {
    let rig = start(config(AGENT), pieces(&["t "; 1200].iter().copied().chain(["</think>", "answer"]).collect::<Vec<_>>()));
    let body = json!({"model": "m", "max_tokens": 1024, "messages": [{"role": "user", "content": "hi"}]});
    let (_, text, _) = rig.post("/v1/chat/completions", None, body);
    let v: Value = serde_json::from_str(&text).unwrap();
    let m = &v["choices"][0]["message"];
    assert!(m["reasoning_content"].as_str().unwrap().contains("answer now"));
    assert!(rig.metrics().contains("inference_thinking_wrapups_total 1"));
    rig.stop();
}

#[test]
fn props_and_slots_report_llama_cpp_style_fields() {
    let rig = start(config(&format!("max_model_len = 4096\ndefault_max_tokens = 256\n{AGENT}")), pieces(&["hi"]));
    let (status, body, _) = rig.get("/props", None);
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model_alias"], "m");
    assert_eq!(v["default_generation_settings"]["n_predict"], 256);
    assert!(v["chat_template_caps"].is_object());
    assert_eq!(v["is_sleeping"], true);
    let (status, body, _) = rig.get("/slots", None);
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v[0]["id"], 0);
    assert_eq!(v[0]["is_processing"], false);
    rig.stop();
}

#[test]
fn tool_call_repairs_show_in_metrics() {
    let rig = start(config(AGENT), pieces(&["<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</tool_call>", ""]));
    let mut body = anthropic(false);
    body["thinking"] = json!({"type": "disabled"});
    let (_, text, _) = rig.post("/v1/messages", None, body);
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["content"][0]["type"], "tool_use", "{text}");
    assert!(rig.metrics().contains("inference_tool_call_repairs_total"));
    rig.stop();
}
