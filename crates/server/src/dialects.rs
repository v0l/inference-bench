use crate::app::{AppState, CancelOnDrop, Gen, Prepared, cache_salt, collect_all, finish_reason, now, prepare, state_shares_keys};
use crate::error::ApiError;
use crate::tools::{Delta, OutputParser, Parsed};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::{self, Stream, StreamExt};
use inference_bench_core::result::Generation;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) enum Piece {
    Content(String),
    Reasoning(String),
    Call { id: String, name: String },
    Args { arguments: String },
    Finished { generation: Generation, reason: &'static str, stopped: bool },
    Failed(ApiError),
}

struct PieceState {
    rx: mpsc::UnboundedReceiver<Gen>,
    parser: Option<OutputParser>,
    streamed: bool,
    done: bool,
    max_tokens: usize,
    _guard: CancelOnDrop,
}

fn from_delta(d: Delta) -> Piece {
    match d {
        Delta::Content(t) => Piece::Content(t),
        Delta::Reasoning(t) => Piece::Reasoning(t),
        Delta::Call { id, name, .. } => Piece::Call { id, name },
        Delta::Args { arguments, .. } => Piece::Args { arguments },
    }
}

fn pieces(rx: mpsc::UnboundedReceiver<Gen>, guard: CancelOnDrop, parser: Option<OutputParser>, max_tokens: usize) -> impl Stream<Item = Piece> {
    let state = PieceState { rx, parser, streamed: false, done: false, max_tokens, _guard: guard };
    stream::unfold(state, |mut st| async move {
        if st.done {
            return None;
        }
        let out: Vec<Piece> = match st.rx.recv().await {
            None => {
                st.done = true;
                vec![Piece::Failed(ApiError::internal("generation ended without a result"))]
            }
            Some(Gen::Logprob(_)) => Vec::new(),
            Some(Gen::Text(t)) => {
                st.streamed = true;
                match st.parser.as_mut() {
                    Some(p) => p.feed(&t).into_iter().map(from_delta).collect(),
                    None => vec![Piece::Content(t)],
                }
            }
            Some(Gen::Done(Err(e), _)) => {
                st.done = true;
                vec![Piece::Failed(e)]
            }
            Some(Gen::Done(Ok(g), stopped)) => {
                st.done = true;
                let mut reason = finish_reason(&g, st.max_tokens, stopped);
                let mut out = Vec::new();
                match st.parser.as_mut() {
                    Some(p) => {
                        let mut d = if st.streamed { Vec::new() } else { p.feed(&g.text) };
                        d.extend(p.finish());
                        out.extend(d.into_iter().map(from_delta));
                        if p.called() {
                            reason = "tool_calls";
                        }
                    }
                    None if !st.streamed && !g.text.is_empty() => out.push(Piece::Content(g.text.clone())),
                    None => {}
                }
                out.push(Piece::Finished { generation: g, reason, stopped });
                out
            }
        };
        Some((out, st))
    })
    .flat_map(stream::iter)
}

fn parsed_output(text: &str, output: &crate::app::OutputMode) -> Parsed {
    if !output.parses() {
        return Parsed { content: (!text.is_empty()).then(|| text.to_string()), ..Parsed::default() };
    }
    let mut parser = output.parser();
    parser.feed(text);
    parser.finish();
    parser.result()
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn content_value(texts: Vec<String>, images: Vec<String>) -> Value {
    if images.is_empty() {
        return Value::String(texts.join("\n"));
    }
    let mut parts: Vec<Value> = texts.into_iter().map(|t| json!({"type": "text", "text": t})).collect();
    parts.extend(images.into_iter().map(|u| json!({"type": "image_url", "image_url": {"url": u}})));
    Value::Array(parts)
}

fn anthropic_error(e: ApiError) -> Response {
    let kind = match e.kind {
        "server_error" => "api_error",
        "upstream_error" => "api_error",
        other => other,
    };
    let status = match e.status {
        StatusCode::SERVICE_UNAVAILABLE => StatusCode::from_u16(529).unwrap_or(e.status),
        s => s,
    };
    let mut response = (status, Json(json!({"type": "error", "error": {"type": kind, "message": e.message}}))).into_response();
    if let Some(secs) = e.retry_after {
        response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from(secs));
    }
    response
}

fn effort_for_budget(budget: u64) -> &'static str {
    match budget {
        0..=2047 => "low",
        2048..=8191 => "medium",
        _ => "high",
    }
}

fn anthropic_image(block: &Value) -> Option<String> {
    let source = &block["source"];
    match source["type"].as_str() {
        Some("base64") => Some(format!("data:{};base64,{}", source["media_type"].as_str().unwrap_or("image/png"), source["data"].as_str()?)),
        Some("url") => source["url"].as_str().map(String::from),
        _ => None,
    }
}

fn tool_result_text(block: &Value) -> String {
    match &block["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p["type"].as_str() {
                Some("text") => p["text"].as_str().unwrap_or("").to_string(),
                Some("image") => "[image]".to_string(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn anthropic_messages(body: &Value) -> Result<Vec<Value>, ApiError> {
    let mut out: Vec<Value> = Vec::new();
    match &body["system"] {
        Value::String(s) if !s.is_empty() => out.push(json!({"role": "system", "content": s})),
        Value::Array(blocks) => {
            let text = blocks.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("\n\n");
            if !text.is_empty() {
                out.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }
    let messages = body["messages"].as_array().filter(|m| !m.is_empty()).ok_or_else(|| ApiError::bad_request("messages: at least one message is required"))?;
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        let blocks: Vec<Value> = match &m["content"] {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        if role == "assistant" {
            let (mut texts, mut calls, mut thinking) = (Vec::new(), Vec::new(), Vec::new());
            for b in &blocks {
                match b["type"].as_str() {
                    Some("text") => texts.push(b["text"].as_str().unwrap_or("").to_string()),
                    Some("thinking") => thinking.push(b["thinking"].as_str().unwrap_or("").to_string()),
                    Some("tool_use") => calls.push(json!({
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": b["input"].to_string()},
                    })),
                    _ => {}
                }
            }
            let mut msg = json!({"role": "assistant", "content": texts.join("\n")});
            if !thinking.is_empty() {
                msg["reasoning_content"] = json!(thinking.join("\n"));
            }
            if !calls.is_empty() {
                msg["tool_calls"] = Value::Array(calls);
            }
            out.push(msg);
            continue;
        }
        let (mut texts, mut images) = (Vec::new(), Vec::new());
        let mut results = Vec::new();
        for b in &blocks {
            match b["type"].as_str() {
                Some("text") => texts.push(b["text"].as_str().unwrap_or("").to_string()),
                Some("image") => images.extend(anthropic_image(b)),
                Some("tool_result") => results.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"], "content": tool_result_text(b)})),
                _ => {}
            }
        }
        out.extend(results);
        if !texts.is_empty() || !images.is_empty() {
            out.push(json!({"role": "user", "content": content_value(texts, images)}));
        }
    }
    Ok(out)
}

fn anthropic_tools(body: &Value) -> Value {
    let tools: Vec<Value> = body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["name"].is_string() && t["type"].as_str().is_none_or(|k| k == "custom"))
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": if t["input_schema"].is_object() { t["input_schema"].clone() } else { json!({"type": "object", "properties": {}}) },
            }})
        })
        .collect();
    match tools.is_empty() {
        true => Value::Null,
        false => Value::Array(tools),
    }
}

pub(crate) fn anthropic_to_chat(state: &AppState, body: &Value) -> Result<Value, ApiError> {
    let max_tokens = body["max_tokens"].as_u64().filter(|n| *n > 0).ok_or_else(|| ApiError::bad_request("max_tokens: a positive integer is required"))?;
    let mut effort: Option<String> = body["output_config"]["effort"].as_str().map(|e| match e {
        "max" | "xhigh" => "high".to_string(),
        other => other.to_string(),
    });
    let mut budget = Value::Null;
    match body["thinking"]["type"].as_str() {
        Some("disabled") => effort = Some("none".into()),
        Some("enabled") => {
            if let Some(b) = body["thinking"]["budget_tokens"].as_u64() {
                budget = json!(b);
                effort.get_or_insert_with(|| effort_for_budget(b).to_string());
            }
            effort.get_or_insert_with(|| "high".to_string());
        }
        _ => {}
    }
    if effort.is_none() && body["thinking"].is_null() && state.config.anthropic_thinking_on_request {
        effort = Some("none".into());
    }
    let tools = anthropic_tools(body);
    let choice = match body["tool_choice"]["type"].as_str() {
        Some("none") => json!("none"),
        _ => Value::Null,
    };
    Ok(json!({
        "model": body["model"],
        "messages": anthropic_messages(body)?,
        "max_tokens": max_tokens,
        "temperature": body["temperature"],
        "top_p": body["top_p"],
        "top_k": body["top_k"],
        "stop": body["stop_sequences"],
        "stream": body["stream"],
        "tools": tools,
        "tool_choice": choice,
        "reasoning_effort": effort,
        "reasoning_budget_tokens": budget,
    }))
}

fn anthropic_stop_reason(reason: &str, stopped: bool) -> &'static str {
    match (reason, stopped) {
        ("tool_calls", _) => "tool_use",
        ("length", _) => "max_tokens",
        (_, true) => "stop_sequence",
        _ => "end_turn",
    }
}

fn anthropic_id(id: &str) -> String {
    format!("toolu_{}", id.strip_prefix("chatcmpl-tool-").unwrap_or(id))
}

fn anthropic_usage(g: &Generation) -> Value {
    json!({"input_tokens": g.prompt_tokens, "output_tokens": g.completion_tokens, "cache_read_input_tokens": g.cached_tokens})
}

fn only_stop(stops: &[String]) -> Value {
    match stops {
        [one] => json!(one),
        _ => Value::Null,
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Open {
    None,
    Text,
    Thinking,
    Tool,
}

struct MessageEvents {
    id: String,
    model: String,
    index: usize,
    open: Open,
    stops: Vec<String>,
}

impl MessageEvents {
    fn start(&self) -> Vec<(&'static str, Value)> {
        vec![(
            "message_start",
            json!({"type": "message_start", "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
                "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0},
            }}),
        )]
    }

    fn close(&mut self, out: &mut Vec<(&'static str, Value)>) {
        if self.open != Open::None {
            out.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
            self.index += 1;
            self.open = Open::None;
        }
    }

    fn begin(&mut self, kind: Open, block: Value, out: &mut Vec<(&'static str, Value)>) {
        self.close(out);
        out.push(("content_block_start", json!({"type": "content_block_start", "index": self.index, "content_block": block})));
        self.open = kind;
    }

    fn on(&mut self, piece: Piece) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        match piece {
            Piece::Content(t) if !t.is_empty() && (self.open == Open::Text || !t.trim().is_empty()) => {
                if self.open != Open::Text {
                    self.begin(Open::Text, json!({"type": "text", "text": ""}), &mut out);
                }
                out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": {"type": "text_delta", "text": t}})));
            }
            Piece::Reasoning(t) if !t.is_empty() => {
                if self.open != Open::Thinking {
                    self.begin(Open::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}), &mut out);
                }
                out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": {"type": "thinking_delta", "thinking": t}})));
            }
            Piece::Call { id, name, .. } => {
                self.begin(Open::Tool, json!({"type": "tool_use", "id": anthropic_id(&id), "name": name, "input": {}}), &mut out);
            }
            Piece::Args { arguments, .. } => {
                out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": {"type": "input_json_delta", "partial_json": arguments}})));
            }
            Piece::Finished { generation, reason, stopped } => {
                self.close(&mut out);
                let stop = anthropic_stop_reason(reason, stopped);
                let sequence = if stop == "stop_sequence" { only_stop(&self.stops) } else { Value::Null };
                out.push(("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": sequence}, "usage": anthropic_usage(&generation)})));
                out.push(("message_stop", json!({"type": "message_stop"})));
            }
            Piece::Failed(e) => {
                let kind = if e.kind == "server_error" { "api_error" } else { e.kind };
                out.push(("error", json!({"type": "error", "error": {"type": kind, "message": e.message}})));
            }
            _ => {}
        }
        out
    }
}

fn sse(name: &str, data: &Value) -> Result<Event, Infallible> {
    Ok(Event::default().event(name).data(data.to_string()))
}

fn ping() -> Event {
    Event::default().event("ping").data(json!({"type": "ping"}).to_string())
}

fn keep_alive() -> KeepAlive {
    KeepAlive::new().interval(Duration::from_secs(10)).event(ping())
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", &uuid::Uuid::new_v4().simple().to_string()[..24])
}

pub(crate) async fn messages(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    match run_messages(state, headers, body).await {
        Ok(r) => r,
        Err(e) => anthropic_error(e),
    }
}

async fn run_messages(state: Arc<AppState>, headers: HeaderMap, body: Value) -> Result<Response, ApiError> {
    let chat = anthropic_to_chat(&state, &body)?;
    let salt = cache_salt(&headers, "", state_shares_keys(&state));
    let Prepared { p, mut receivers, guard, max_tokens } = prepare(state, chat, true, salt).await?;
    let rx = receivers.remove(0);
    let mut events = MessageEvents { id: new_id("msg"), model: p.model.clone(), index: 0, open: Open::None, stops: p.stop.clone() };
    if p.stream {
        let parser = p.output.parses().then(|| p.output.parser());
        let opening = events.start();
        let head = stream::iter(opening.into_iter().map(|(n, v)| sse(n, &v)).chain(std::iter::once(Ok(ping()))));
        let tail = pieces(rx, guard, parser, max_tokens).flat_map(move |piece| stream::iter(events.on(piece).into_iter().map(|(n, v)| sse(n, &v))));
        return Ok(Sse::new(head.chain(tail)).keep_alive(keep_alive()).into_response());
    }
    let done = collect_all(rx).await?;
    drop(guard);
    let parsed = parsed_output(&done.generation.text, &p.output);
    let mut reason = finish_reason(&done.generation, max_tokens, done.stopped);
    if !parsed.tool_calls.is_empty() {
        reason = "tool_calls";
    }
    let mut content = Vec::new();
    if let Some(r) = &parsed.reasoning {
        content.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
    }
    if let Some(t) = parsed.content.as_deref().filter(|t| !t.trim().is_empty()) {
        content.push(json!({"type": "text", "text": t}));
    }
    for c in &parsed.tool_calls {
        let input: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
        content.push(json!({"type": "tool_use", "id": anthropic_id(&c.id), "name": c.name, "input": input}));
    }
    let stop = anthropic_stop_reason(reason, done.stopped);
    let sequence = if stop == "stop_sequence" { only_stop(&p.stop) } else { Value::Null };
    Ok(Json(json!({
        "id": events.id, "type": "message", "role": "assistant", "model": p.model, "content": content,
        "stop_reason": stop, "stop_sequence": sequence, "usage": anthropic_usage(&done.generation),
    }))
    .into_response())
}

pub(crate) async fn count_tokens(Json(body): Json<Value>) -> Response {
    let chars: usize = body["messages"].to_string().len() + body["system"].to_string().len() + body["tools"].to_string().len();
    Json(json!({"input_tokens": inference_bench_core::chat::estimate_tokens(&"x".repeat(chars))})).into_response()
}

fn responses_text(content: &Value) -> (Vec<String>, Vec<String>) {
    let (mut texts, mut images) = (Vec::new(), Vec::new());
    match content {
        Value::String(s) => texts.push(s.clone()),
        Value::Array(parts) => {
            for p in parts {
                match p["type"].as_str() {
                    Some("input_text") | Some("output_text") | Some("text") => texts.push(p["text"].as_str().unwrap_or("").to_string()),
                    Some("input_image") | Some("image_url") => {
                        let url = p["image_url"].as_str().or_else(|| p["image_url"]["url"].as_str());
                        images.extend(url.map(String::from));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    (texts, images)
}

fn responses_output_text(output: &Value) -> String {
    match output {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        other => other.to_string(),
    }
}

fn responses_messages(body: &Value) -> Result<Vec<Value>, ApiError> {
    let mut out: Vec<Value> = Vec::new();
    if let Some(i) = body["instructions"].as_str().filter(|i| !i.is_empty()) {
        out.push(json!({"role": "system", "content": i}));
    }
    let items: Vec<Value> = match &body["input"] {
        Value::String(s) => vec![json!({"role": "user", "content": s})],
        Value::Array(a) if !a.is_empty() => a.clone(),
        _ => return Err(ApiError::bad_request("input: a string or a non-empty array is required")),
    };
    let mut reasoning: Option<String> = None;
    let assistant_tail = |out: &mut Vec<Value>, reasoning: &mut Option<String>| {
        if out.last().is_some_and(|m| m["role"] == "assistant") {
            return;
        }
        let mut m = json!({"role": "assistant", "content": ""});
        if let Some(r) = reasoning.take() {
            m["reasoning_content"] = json!(r);
        }
        out.push(m);
    };
    for item in &items {
        let kind = item["type"].as_str().unwrap_or(if item["role"].is_string() { "message" } else { "" });
        match kind {
            "message" => {
                let role = match item["role"].as_str().unwrap_or("user") {
                    "developer" => "system",
                    r => r,
                };
                let (texts, images) = responses_text(&item["content"]);
                if role == "assistant" {
                    let mut m = json!({"role": "assistant", "content": texts.join("")});
                    if let Some(r) = reasoning.take() {
                        m["reasoning_content"] = json!(r);
                    }
                    out.push(m);
                } else {
                    out.push(json!({"role": role, "content": content_value(texts, images)}));
                }
            }
            "reasoning" => {
                let mut text = text_of(&item["content"]);
                if text.is_empty() {
                    text = text_of(&item["summary"]);
                }
                if !text.is_empty() {
                    reasoning = Some(text);
                }
            }
            "function_call" => {
                assistant_tail(&mut out, &mut reasoning);
                let call = json!({
                    "id": item["call_id"].as_str().or_else(|| item["id"].as_str()), "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"].as_str().unwrap_or("{}")},
                });
                let last = out.last_mut().unwrap();
                match last["tool_calls"].as_array_mut() {
                    Some(list) => list.push(call),
                    None => last["tool_calls"] = json!([call]),
                }
            }
            "function_call_output" => {
                out.push(json!({"role": "tool", "tool_call_id": item["call_id"], "content": responses_output_text(&item["output"])}));
            }
            _ => {}
        }
    }
    Ok(out)
}

fn responses_tools(body: &Value) -> Value {
    let tools: Vec<Value> = body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["type"] == "function" && t["name"].is_string())
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": if t["parameters"].is_object() { t["parameters"].clone() } else { json!({"type": "object", "properties": {}}) },
            }})
        })
        .collect();
    match tools.is_empty() {
        true => Value::Null,
        false => Value::Array(tools),
    }
}

fn responses_format(body: &Value) -> Value {
    let f = &body["text"]["format"];
    match f["type"].as_str() {
        Some("json_schema") => json!({"type": "json_schema", "json_schema": {"name": f["name"], "schema": f["schema"], "strict": f["strict"]}}),
        Some("json_object") => json!({"type": "json_object"}),
        _ => Value::Null,
    }
}

pub(crate) fn responses_to_chat(body: &Value) -> Result<Value, ApiError> {
    if !body["previous_response_id"].is_null() {
        return Err(ApiError::bad_request("previous_response_id: responses are stateless, send the whole conversation in input"));
    }
    let choice = match body["tool_choice"].as_str() {
        Some("none") => json!("none"),
        _ => Value::Null,
    };
    Ok(json!({
        "model": body["model"],
        "messages": responses_messages(body)?,
        "max_tokens": body["max_output_tokens"],
        "temperature": body["temperature"],
        "top_p": body["top_p"],
        "stream": body["stream"],
        "tools": responses_tools(body),
        "tool_choice": choice,
        "reasoning": body["reasoning"],
        "reasoning_budget_tokens": body["reasoning_budget_tokens"],
        "response_format": responses_format(body),
    }))
}

fn response_usage(g: &Generation) -> Value {
    json!({
        "input_tokens": g.prompt_tokens,
        "input_tokens_details": {"cached_tokens": g.cached_tokens},
        "output_tokens": g.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": g.prompt_tokens + g.completion_tokens,
    })
}

fn response_object(id: &str, model: &str, created: i64, status: &str, output: Vec<Value>, usage: Value, reason: &str) -> Value {
    let incomplete = if status == "incomplete" { json!({"reason": reason}) } else { Value::Null };
    json!({
        "id": id, "object": "response", "created_at": created, "status": status, "model": model,
        "output": output, "usage": usage, "incomplete_details": incomplete, "error": null,
        "parallel_tool_calls": true, "tool_choice": "auto", "tools": [], "store": false,
    })
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "message", "status": status, "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"id": id, "type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]})
}

fn call_item(id: &str, call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
    json!({"id": id, "type": "function_call", "status": status, "call_id": call_id, "name": name, "arguments": arguments})
}

fn responses_status(reason: &str) -> (&'static str, &'static str) {
    match reason {
        "length" => ("incomplete", "max_output_tokens"),
        _ => ("completed", ""),
    }
}

enum Current {
    None,
    Message { id: String, text: String },
    Reasoning { id: String, text: String },
    Call { id: String, call_id: String, name: String, arguments: String },
}

struct ResponseEvents {
    id: String,
    model: String,
    created: i64,
    sequence: usize,
    current: Current,
    output: Vec<Value>,
}

impl ResponseEvents {
    fn emit(&mut self, out: &mut Vec<(String, Value)>, name: &str, mut data: Value) {
        data["type"] = json!(name);
        data["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        out.push((name.to_string(), data));
    }

    fn start(&mut self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        let r = response_object(&self.id, &self.model, self.created, "in_progress", Vec::new(), Value::Null, "");
        self.emit(&mut out, "response.created", json!({"response": r.clone()}));
        self.emit(&mut out, "response.in_progress", json!({"response": r}));
        out
    }

    fn close(&mut self, out: &mut Vec<(String, Value)>) {
        let index = self.output.len();
        match std::mem::replace(&mut self.current, Current::None) {
            Current::None => {}
            Current::Message { id, text } => {
                self.emit(out, "response.output_text.done", json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}));
                self.emit(out, "response.content_part.done", json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}));
                let item = message_item(&id, &text, "completed");
                self.emit(out, "response.output_item.done", json!({"output_index": index, "item": item.clone()}));
                self.output.push(item);
            }
            Current::Reasoning { id, text } => {
                self.emit(out, "response.reasoning_text.done", json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}));
                let item = reasoning_item(&id, &text);
                self.emit(out, "response.output_item.done", json!({"output_index": index, "item": item.clone()}));
                self.output.push(item);
            }
            Current::Call { id, call_id, name, arguments } => {
                self.emit(out, "response.function_call_arguments.done", json!({"item_id": id, "output_index": index, "arguments": arguments}));
                let item = call_item(&id, &call_id, &name, &arguments, "completed");
                self.emit(out, "response.output_item.done", json!({"output_index": index, "item": item.clone()}));
                self.output.push(item);
            }
        }
    }

    fn on(&mut self, piece: Piece) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        let index = self.output.len();
        match piece {
            Piece::Content(t) if !t.is_empty() && (matches!(self.current, Current::Message { .. }) || !t.trim().is_empty()) => {
                if !matches!(self.current, Current::Message { .. }) {
                    self.close(&mut out);
                    let index = self.output.len();
                    let id = new_id("msg");
                    self.emit(&mut out, "response.output_item.added", json!({"output_index": index, "item": message_item(&id, "", "in_progress")}));
                    self.emit(&mut out, "response.content_part.added", json!({"item_id": id, "output_index": index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}));
                    self.current = Current::Message { id, text: String::new() };
                }
                if let Current::Message { id, text } = &mut self.current {
                    text.push_str(&t);
                    let id = id.clone();
                    self.emit(&mut out, "response.output_text.delta", json!({"item_id": id, "output_index": self.output.len(), "content_index": 0, "delta": t}));
                }
            }
            Piece::Reasoning(t) if !t.is_empty() => {
                if !matches!(self.current, Current::Reasoning { .. }) {
                    self.close(&mut out);
                    let index = self.output.len();
                    let id = new_id("rs");
                    self.emit(&mut out, "response.output_item.added", json!({"output_index": index, "item": {"id": id, "type": "reasoning", "summary": [], "content": []}}));
                    self.current = Current::Reasoning { id, text: String::new() };
                }
                if let Current::Reasoning { id, text } = &mut self.current {
                    text.push_str(&t);
                    let id = id.clone();
                    self.emit(&mut out, "response.reasoning_text.delta", json!({"item_id": id, "output_index": self.output.len(), "content_index": 0, "delta": t}));
                }
            }
            Piece::Call { id: call_id, name, .. } => {
                self.close(&mut out);
                let index = self.output.len();
                let id = new_id("fc");
                self.emit(&mut out, "response.output_item.added", json!({"output_index": index, "item": call_item(&id, &call_id, &name, "", "in_progress")}));
                self.current = Current::Call { id, call_id, name, arguments: String::new() };
            }
            Piece::Args { arguments, .. } => {
                if let Current::Call { id, arguments: acc, .. } = &mut self.current {
                    acc.push_str(&arguments);
                    let id = id.clone();
                    self.emit(&mut out, "response.function_call_arguments.delta", json!({"item_id": id, "output_index": index, "delta": arguments}));
                }
            }
            Piece::Finished { generation, reason, .. } => {
                self.close(&mut out);
                let (status, why) = responses_status(reason);
                let r = response_object(&self.id, &self.model, self.created, status, self.output.clone(), response_usage(&generation), why);
                let name = if status == "completed" { "response.completed" } else { "response.incomplete" };
                self.emit(&mut out, name, json!({"response": r}));
            }
            Piece::Failed(e) => {
                let r = {
                    let mut r = response_object(&self.id, &self.model, self.created, "failed", self.output.clone(), Value::Null, "");
                    r["error"] = json!({"code": e.kind, "message": e.message});
                    r
                };
                self.emit(&mut out, "response.failed", json!({"response": r}));
            }
            _ => {}
        }
        out
    }
}

pub(crate) async fn responses(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let chat = responses_to_chat(&body)?;
    let salt = cache_salt(&headers, "", state_shares_keys(&state));
    let Prepared { p, mut receivers, guard, max_tokens } = prepare(state, chat, true, salt).await?;
    let rx = receivers.remove(0);
    let mut events = ResponseEvents { id: new_id("resp"), model: p.model.clone(), created: now(), sequence: 0, current: Current::None, output: Vec::new() };
    if p.stream {
        let parser = p.output.parses().then(|| p.output.parser());
        let head = stream::iter(events.start().into_iter().map(|(n, v)| sse(&n, &v)));
        let tail = pieces(rx, guard, parser, max_tokens).flat_map(move |piece| stream::iter(events.on(piece).into_iter().map(|(n, v)| sse(&n, &v))));
        return Ok(Sse::new(head.chain(tail)).keep_alive(KeepAlive::default()).into_response());
    }
    let done = collect_all(rx).await?;
    drop(guard);
    let parsed = parsed_output(&done.generation.text, &p.output);
    let mut reason = finish_reason(&done.generation, max_tokens, done.stopped);
    if !parsed.tool_calls.is_empty() {
        reason = "tool_calls";
    }
    let mut output = Vec::new();
    if let Some(r) = &parsed.reasoning {
        output.push(reasoning_item(&new_id("rs"), r));
    }
    if let Some(t) = parsed.content.as_deref().filter(|t| !t.trim().is_empty()) {
        output.push(message_item(&new_id("msg"), t, "completed"));
    }
    for c in &parsed.tool_calls {
        output.push(call_item(&new_id("fc"), &c.id, &c.name, &c.arguments, "completed"));
    }
    let (status, why) = responses_status(reason);
    Ok(Json(response_object(&events.id, &p.model, events.created, status, output, response_usage(&done.generation), why)).into_response())
}

fn slot_model(state: &AppState, query: &HashMap<String, String>) -> Option<String> {
    query.get("model").cloned().or_else(|| state.manager.loaded_snapshot().into_iter().next().map(|(id, _)| id)).or_else(|| state.manager.default_model())
}

pub(crate) async fn props(State(state): State<Arc<AppState>>, Query(query): Query<HashMap<String, String>>) -> Result<Json<Value>, ApiError> {
    let id = slot_model(&state, &query).ok_or_else(|| ApiError::not_found("no model is configured"))?;
    let id = state.manager.resolve(&id)?.id;
    let engine = state.manager.props(&id);
    let loaded = state.manager.is_loaded(&id);
    let engine = engine.unwrap_or_else(|| json!({}));
    let defaults = &engine["generation"];
    let n_ctx = engine["n_ctx"].as_u64().or(state.config.max_model_len.map(|n| n as u64));
    let tool_format = state.manager.tool_format(&id);
    Ok(Json(json!({
        "tool_parser": state.config.tool_parser_for(tool_format),
        "tool_format": format!("{tool_format:?}"),
        "reasoning_parser": state.config.reasoning_parser,
        "model_alias": id,
        "model_path": state.config.model(&id).and_then(|m| m.model.clone()),
        "total_slots": state.config.concurrency(),
        "chat_template": engine["chat_template"],
        "chat_template_caps": if engine["chat_template_caps"].is_object() { engine["chat_template_caps"].clone() } else { json!({}) },
        "bos_token": engine["bos_token"],
        "eos_token": engine["eos_token"],
        "modalities": {"vision": engine["vision"].as_bool().unwrap_or(false), "audio": false},
        "default_generation_settings": {
            "n_ctx": n_ctx,
            "n_predict": state.config.default_max_tokens.map(|n| n as i64).unwrap_or(-1),
            "params": {
                "top_k": state.config.top_k.map(Value::from).unwrap_or_else(|| defaults["top_k"].clone()),
                "top_p": state.config.top_p.map(Value::from).unwrap_or_else(|| defaults["top_p"].clone()),
                "min_p": state.config.min_p.map(Value::from).unwrap_or_else(|| defaults["min_p"].clone()),
                "max_tokens": state.config.default_max_tokens.map(|n| n as i64).unwrap_or(-1),
            },
        },
        "reasoning_budget_tokens": state.config.reasoning_budget_tokens,
        "is_sleeping": !loaded,
        "build_info": format!("inference-bench {}", env!("CARGO_PKG_VERSION")),
    })))
}

pub(crate) async fn slots(State(state): State<Arc<AppState>>, Query(query): Query<HashMap<String, String>>) -> Result<Json<Value>, ApiError> {
    let id = slot_model(&state, &query).ok_or_else(|| ApiError::not_found("no model is configured"))?;
    let id = state.manager.resolve(&id)?.id;
    let n_ctx = state.manager.props(&id).and_then(|p| p["n_ctx"].as_u64()).or(state.config.max_model_len.map(|n| n as u64));
    let active = state.manager.active_of(&id);
    let list: Vec<Value> = (0..state.config.concurrency())
        .map(|i| json!({"id": i, "n_ctx": n_ctx, "is_processing": i < active, "model": id}))
        .collect();
    Ok(Json(Value::Array(list)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_blocks_become_chat_messages() {
        let body = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "x"}, {"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "done"}]}, {"type": "text", "text": "next"}]},
            ],
        });
        let m = anthropic_messages(&body).unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "be brief"}));
        assert_eq!(m[1]["content"][1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(m[2]["reasoning_content"], "hm");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(m[3], json!({"role": "tool", "tool_call_id": "t1", "content": "done"}));
        assert_eq!(m[4], json!({"role": "user", "content": "next"}));
    }

    #[test]
    fn responses_items_become_chat_messages() {
        let body = json!({
            "instructions": "sys",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "think"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "one"},
            ],
        });
        let m = responses_messages(&body).unwrap();
        assert_eq!(m[1]["role"], "system");
        assert_eq!(m[3]["reasoning_content"], "think");
        assert_eq!(m[3]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(m[4], json!({"role": "tool", "tool_call_id": "c1", "content": "one"}));
    }

    #[test]
    fn budgets_map_to_efforts() {
        assert_eq!([effort_for_budget(1024), effort_for_budget(4096), effort_for_budget(16000)], ["low", "medium", "high"]);
    }
}
