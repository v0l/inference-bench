use crate::config::ServerConfig;
use crate::error::ApiError;
use crate::manager::{EngineFactory, GpuProbe, Manager};
use crate::metrics::{Gauges, Metrics, percentile};
use crate::stop::StopFilter;
use crate::tools::{Delta, OutputParser};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream, StreamExt};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling};
use inference_bench_core::logging::{preview, redact_key};
use inference_bench_core::result::{BackendError, ErrorKind, Generation};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{Instrument, Span};

pub const REQUEST_ID_HEADER: &str = "x-request-id";

pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub manager: Arc<Manager>,
    pub metrics: Arc<Metrics>,
    keys: Vec<String>,
    admitted: AtomicUsize,
    running: AtomicUsize,
    access_log: Option<Mutex<std::fs::File>>,
    pub started: Instant,
}

impl AppState {
    pub fn new(
        config: Arc<ServerConfig>,
        factory: Arc<dyn EngineFactory>,
        probe: GpuProbe,
    ) -> anyhow::Result<Arc<Self>> {
        let metrics = Arc::new(Metrics::default());
        let access_log = match &config.access_log {
            Some(path) => {
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                Some(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(path)?))
            }
            None => None,
        };
        Ok(Arc::new(Self {
            manager: Arc::new(Manager::new(config.clone(), factory, probe, metrics.clone())),
            keys: config.resolved_keys(),
            metrics,
            config,
            admitted: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            access_log,
            started: Instant::now(),
        }))
    }

    pub fn gauges(&self) -> Gauges {
        let admitted = self.admitted.load(Ordering::SeqCst);
        let running = self.running.load(Ordering::SeqCst);
        Gauges {
            queue_depth: admitted.saturating_sub(running),
            in_flight: running,
            loaded_models: self.manager.loaded_snapshot(),
            gpu_used_bytes: self.manager.gpu_used(),
            engine: self.manager.engine_metrics(),
        }
    }

    pub fn auth_required(&self) -> bool {
        !self.keys.is_empty()
    }

    fn key_valid(&self, presented: &str) -> bool {
        self.keys.iter().fold(false, |ok, k| ok | constant_time_eq(k.as_bytes(), presented.as_bytes()))
    }

    fn admit(self: &Arc<Self>) -> Result<Admission, ApiError> {
        let limit = self.config.concurrency() + self.config.queue_depth;
        let mut current = self.admitted.load(Ordering::SeqCst);
        loop {
            if current >= limit {
                return Err(ApiError::busy(self.config.retry_after_secs));
            }
            match self.admitted.compare_exchange(current, current + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
        Ok(Admission { state: self.clone(), running: false })
    }

    fn write_access(&self, line: Value) {
        if let Some(f) = &self.access_log {
            let mut f = f.lock().unwrap();
            let _ = writeln!(f, "{line}");
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

struct Admission {
    state: Arc<AppState>,
    running: bool,
}

impl Admission {
    fn start(&mut self) {
        if !self.running {
            self.running = true;
            self.state.running.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if self.running {
            self.state.running.fetch_sub(1, Ordering::SeqCst);
        }
        self.state.admitted.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/bench", post(bench))
        .route("/admin/models", get(admin_models))
        .route("/admin/models/{id}/load", post(admin_load))
        .route("/admin/models/{id}/unload", post(admin_unload))
        .route("/admin/specs", get(admin_specs))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(middleware::from_fn_with_state(state.clone(), request_scope))
        .with_state(state.clone());
    if !state.config.cors_origins.is_empty() {
        app = app.layer(cors(&state.config.cors_origins));
    }
    app
}

fn cors(origins: &[String]) -> tower_http::cors::CorsLayer {
    use tower_http::cors::{AllowOrigin, Any, CorsLayer};
    let layer = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .expose_headers([HeaderName::from_static(REQUEST_ID_HEADER)]);
    match origins.iter().any(|o| o == "*") {
        true => layer.allow_origin(Any),
        false => layer.allow_origin(AllowOrigin::list(
            origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()),
        )),
    }
}

fn new_request_id() -> String {
    format!("req-{}", &uuid::Uuid::new_v4().simple().to_string()[..16])
}

async fn request_scope(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    let incoming = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 64 && v.chars().all(|c| c.is_ascii_graphic()))
        .map(String::from);
    let request_id = incoming.unwrap_or_else(new_request_id);
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %method,
        path = %path,
        model = tracing::field::Empty,
        backend = tracing::field::Empty,
        prompt_tokens = tracing::field::Empty,
        completion_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        decode_tok_s = tracing::field::Empty,
        finish_reason = tracing::field::Empty,
        status = tracing::field::Empty,
    );
    let started = Instant::now();
    let mut response = next.run(req).instrument(span.clone()).await;
    let status = response.status().as_u16();
    span.record("status", status);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    span.in_scope(|| tracing::info!(status, elapsed_ms, "request finished"));
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(HeaderName::from_static(REQUEST_ID_HEADER), v);
    }
    state.metrics.request(endpoint_label(&path), status);
    state.write_access(json!({
        "time": chrono::Utc::now().to_rfc3339(),
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status,
        "elapsed_ms": elapsed_ms,
    }));
    response
}

fn endpoint_label(path: &str) -> &'static str {
    match path {
        "/health" => "health",
        "/metrics" => "metrics",
        "/v1/models" => "models",
        "/v1/chat/completions" => "chat_completions",
        "/v1/completions" => "completions",
        "/v1/embeddings" => "embeddings",
        "/v1/bench" => "bench",
        p if p.starts_with("/admin/") => "admin",
        _ => "other",
    }
}

async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    let path = req.uri().path();
    let open = path == "/health" || (path == "/metrics" && !state.config.protect_metrics);
    if !state.auth_required() || open || req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }
    let presented = bearer(req.headers());
    match presented {
        Some(key) if state.key_valid(key) => next.run(req).await,
        Some(key) => {
            tracing::warn!(key = %redact_key(key), "rejected request with an invalid API key");
            ApiError::unauthorized("Incorrect API key provided").into_response()
        }
        None => {
            tracing::warn!("rejected request without an API key");
            ApiError::unauthorized("Missing bearer API key").into_response()
        }
    }
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "uptime_secs": state.started.elapsed().as_secs(),
        "loaded_models": state.manager.loaded_snapshot().into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
    }))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let gauges = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || state.gauges()).await.unwrap_or_default()
    };
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(&gauges),
    )
        .into_response()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = state
        .manager
        .status()
        .into_iter()
        .map(|m| json!({"id": m.id, "object": "model", "created": 0, "owned_by": "inference-bench", "loaded": m.loaded}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn admin_specs(State(state): State<Arc<AppState>>) -> Json<Value> {
    let section = state.config.specs.clone();
    let body = tokio::task::spawn_blocking(move || {
        use inference_bench_core::specs;
        let data = section.data();
        let store = specs::SpecStore::new(&data);
        let list: Vec<Value> = specs::list(&specs::standard_layers(&data, &section.folders))
            .into_iter()
            .map(|l| json!({"name": l.name, "path": l.path, "origin": l.origin.kind(), "source": l.origin.label(), "problem": l.problem}))
            .collect();
        json!({
            "source": section.updater().source(),
            "auto_update": section.auto_update,
            "state": store.state(),
            "generation": store.generation(),
            "specs": list,
        })
    })
    .await
    .unwrap_or_default();
    Json(body)
}

async fn admin_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"models": state.manager.status()}))
}

async fn admin_load(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let cfg = state.manager.resolve(&id)?;
    if !state.manager.is_loaded(&id) {
        state.manager.preload(&cfg).await?;
    }
    Ok(Json(json!({"id": id, "loaded": true})))
}

async fn admin_unload(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    state.manager.resolve(&id)?;
    if state.manager.is_busy(&id) {
        return Err(ApiError::new(StatusCode::CONFLICT, "conflict", format!("{id} has requests in flight")));
    }
    let unloaded = state.manager.unload(&id);
    Ok(Json(json!({"id": id, "loaded": !unloaded && state.manager.is_loaded(&id)})))
}

async fn embeddings() -> ApiError {
    ApiError::new(
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented",
        "none of the configured backends support embeddings",
    )
}

struct Params {
    model: String,
    messages: Vec<Message>,
    openai_messages: Option<Vec<Value>>,
    template_vars: Value,
    output: OutputMode,
    stop: Vec<String>,
    use_template: bool,
    sampling: Sampling,
    stream: bool,
    include_usage: bool,
    ignore_eos: bool,
}

#[derive(Clone, Default)]
struct OutputMode {
    tools: Vec<Value>,
    tool_parsing: bool,
    reasoning: bool,
    hermes: bool,
}

impl OutputMode {
    fn parses(&self) -> bool {
        self.tool_parsing || self.reasoning
    }

    fn parser(&self) -> OutputParser {
        let p = OutputParser::new(&self.tools, self.tool_parsing, self.reasoning);
        if self.hermes { p.hermes() } else { p }
    }
}

fn toml_json(t: &toml::Table) -> Value {
    serde_json::to_value(t).unwrap_or_else(|_| json!({}))
}

fn stop_strings(body: &Value) -> Result<Vec<String>, ApiError> {
    match &body["stop"] {
        Value::Null => Ok(Vec::new()),
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) if a.iter().all(Value::is_string) => Ok(a.iter().filter_map(|s| s.as_str().map(String::from)).collect()),
        _ => Err(ApiError::bad_request("stop must be a string or an array of strings")),
    }
}

const HANDLED_FIELDS: &[&str] = &[
    "model", "messages", "prompt", "prompts", "repeat", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "stop", "stream",
    "stream_options", "ignore_eos", "tools", "tool_choice", "chat_template_kwargs", "reasoning_effort",
];

const NEUTRAL_FIELDS: &[&str] = &["user", "parallel_tool_calls", "store", "metadata"];

fn reject_unsupported_fields(body: &Value) -> Result<(), ApiError> {
    let Some(obj) = body.as_object() else { return Ok(()) };
    for (key, value) in obj {
        let key = key.as_str();
        let known = HANDLED_FIELDS.contains(&key) || NEUTRAL_FIELDS.contains(&key);
        let ok = known
            || match key {
                "n" => value.is_null() || value.as_u64() == Some(1),
                "logprobs" => value.is_null() || value == &json!(false),
                "top_logprobs" => value.is_null() || value.as_u64() == Some(0),
                "response_format" => value.is_null() || value["type"] == "text",
                "frequency_penalty" | "presence_penalty" => value.is_null() || value.as_f64() == Some(0.0),
                _ => false,
            };
        if !ok {
            let why = match key {
                "n" | "logprobs" | "top_logprobs" | "response_format" | "frequency_penalty" | "presence_penalty" => "value is not supported",
                _ => "field is not supported",
            };
            return Err(ApiError::bad_request(format!("{key}: {why}")));
        }
    }
    Ok(())
}

fn chat_setup(state: &AppState, body: &Value) -> Result<(Value, OutputMode), ApiError> {
    let mut vars = toml_json(&state.config.chat_template_kwargs);
    let user = match &body["chat_template_kwargs"] {
        Value::Object(o) => o.clone(),
        Value::Null => Default::default(),
        _ => return Err(ApiError::bad_request("chat_template_kwargs must be an object")),
    };
    if let Some(effort) = body["reasoning_effort"].as_str() {
        vars["reasoning_effort"] = json!(effort);
        if !user.contains_key("enable_thinking") {
            vars["enable_thinking"] = json!(effort != "none");
        }
    }
    for (k, v) in user {
        vars[k] = v;
    }
    if state.config.reasoning_parser.is_none() && vars.get("enable_thinking").is_none() {
        vars["enable_thinking"] = json!(false);
    }
    let tools = match &body["tools"] {
        Value::Array(t) => t.clone(),
        Value::Null => Vec::new(),
        _ => return Err(ApiError::bad_request("tools must be an array")),
    };
    if tools.iter().any(|t| t["type"] != "function" || !t["function"]["name"].is_string()) {
        return Err(ApiError::bad_request("tools must be function tools with a name"));
    }
    let choice = match &body["tool_choice"] {
        Value::Null => "auto",
        Value::String(s) if s == "auto" || s == "none" => s.as_str(),
        _ => return Err(ApiError::bad_request("tool_choice must be auto or none: required and named tool choice are not supported")),
    };
    if !tools.is_empty() {
        vars["tools"] = Value::Array(tools.clone());
    }
    let output = OutputMode {
        tool_parsing: state.config.tool_parser.is_some() && !tools.is_empty() && choice == "auto",
        reasoning: state.config.reasoning_parser.is_some() && vars["enable_thinking"] != json!(false),
        hermes: state.config.tool_parser.as_deref() == Some("hermes"),
        tools,
    };
    Ok((vars, output))
}

fn parse_params(state: &AppState, body: &Value, chat: bool) -> Result<Params, ApiError> {
    reject_unsupported_fields(body)?;
    let model = match body["model"].as_str() {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => state.manager.default_model().ok_or_else(|| ApiError::bad_request("model is required"))?,
    };
    let messages = if chat { chat_messages(body)? } else { vec![Message::new(Role::User, completion_prompt(body)?)] };
    let max_tokens = body["max_completion_tokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(state.config.default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }
    let defaults = Sampling::default();
    let sampling = Sampling {
        temperature: body["temperature"].as_f64().map(|t| t as f32).unwrap_or(0.0),
        top_p: body["top_p"].as_f64().map(|t| t as f32).or(state.config.top_p).or(defaults.top_p),
        top_k: body["top_k"].as_u64().map(|t| t as u32).or(state.config.top_k).or(defaults.top_k),
        min_p: body["min_p"].as_f64().map(|t| t as f32).or(state.config.min_p).or(defaults.min_p),
        seed: body["seed"].as_u64(),
        max_tokens,
    };
    let (template_vars, output) = match chat {
        true => chat_setup(state, body)?,
        false => (Value::Null, OutputMode::default()),
    };
    Ok(Params {
        model,
        messages,
        openai_messages: chat.then(|| body["messages"].as_array().cloned().unwrap_or_default()),
        template_vars,
        output,
        stop: stop_strings(body)?,
        use_template: chat,
        sampling,
        stream: body["stream"].as_bool().unwrap_or(false),
        include_usage: body["stream_options"]["include_usage"].as_bool().unwrap_or(false),
        ignore_eos: body["ignore_eos"].as_bool().unwrap_or(false),
    })
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

pub fn chat_messages(body: &Value) -> Result<Vec<Message>, ApiError> {
    let messages = body["messages"]
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad_request("messages must be a non-empty array"))?;
    Ok(messages
        .iter()
        .map(|m| Message::new(Role::from_name(m["role"].as_str().unwrap_or("user")), content_text(&m["content"])))
        .collect())
}

fn completion_prompt(body: &Value) -> Result<String, ApiError> {
    match &body["prompt"] {
        Value::String(s) => Ok(s.clone()),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => Ok(a[0].as_str().unwrap().to_string()),
        Value::Array(_) => Err(ApiError::bad_request("batched prompts are not supported")),
        _ => Err(ApiError::bad_request("prompt is required")),
    }
}

enum Gen {
    Text(String),
    Done(Result<Generation, ApiError>, bool),
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn log_prompt(state: &AppState, request: &ChatRequest) {
    let prompt = request.flatten();
    let prompt = prompt.as_str();
    if state.config.log_prompts {
        tracing::info!(prompt = %preview(prompt, 2000), "prompt");
    } else {
        tracing::trace!(prompt = %prompt, prompt_bytes = prompt.len(), "prompt");
    }
}

fn start_generation(
    state: &Arc<AppState>,
    model: String,
    request: ChatRequest,
    max_tokens: usize,
    stop: Vec<String>,
) -> Result<mpsc::UnboundedReceiver<Gen>, ApiError> {
    let mut admission = state.admit()?;
    let (tx, rx) = mpsc::unbounded_channel();
    let state = state.clone();
    let span = Span::current();
    span.record("model", model.as_str());
    tokio::spawn(
        async move {
            let started = Instant::now();
            let lease = tokio::select! {
                l = state.manager.acquire(&model) => l,
                _ = tx.closed() => {
                    state.metrics.cancelled();
                    tracing::info!("client went away while queued");
                    return;
                }
            };
            let mut lease = match lease {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Gen::Done(Err(e), false));
                    return;
                }
            };
            admission.start();
            Span::current().record("backend", lease.model().backend_label.as_str());
            let queued_ms = started.elapsed().as_secs_f64() * 1000.0;
            tracing::debug!(queued_ms, "generation started");
            let flag = request.cancel.clone().unwrap_or_default();
            let text_tx = tx.clone();
            let span = Span::current();
            let outcome = tokio::task::spawn_blocking(move || {
                span.in_scope(|| {
                    let mut filter = StopFilter::new(stop);
                    let mut send = |t: &str| {
                        if !t.is_empty() && text_tx.send(Gen::Text(t.to_string())).is_err() {
                            flag.store(true, Ordering::SeqCst);
                        }
                    };
                    let result = lease.backend().chat(&request, &mut |t| {
                        let out = filter.feed(t);
                        send(&out);
                        if filter.hit {
                            flag.store(true, Ordering::SeqCst);
                        }
                    });
                    let tail = filter.finish();
                    send(&tail);
                    let hit = filter.hit;
                    let result = match result {
                        Ok(mut g) if hit => {
                            g.text = filter.text;
                            Ok(g)
                        }
                        Ok(_) if flag.load(Ordering::SeqCst) => {
                            Err(BackendError::new(ErrorKind::Cancelled, "cancelled"))
                        }
                        Ok(mut g) => {
                            g.text = filter.text;
                            Ok(g)
                        }
                        other => other,
                    };
                    (result, hit)
                })
            })
            .await;
            let (result, stopped) = match outcome {
                Ok((r, hit)) => (r.map_err(ApiError::from), hit),
                Err(e) => (Err(ApiError::internal(format!("generation task failed: {e}"))), false),
            };
            match &result {
                Ok(g) => record_generation(&state, &model, g, max_tokens, stopped),
                Err(e) if e.kind == "cancelled" => {
                    state.metrics.cancelled();
                    tracing::info!("generation cancelled by client disconnect");
                }
                Err(e) => tracing::error!(status = e.status.as_u16(), error = %e.message, "generation failed"),
            }
            let _ = tx.send(Gen::Done(result, stopped));
        }
        .instrument(span),
    );
    Ok(rx)
}

fn finish_reason(g: &Generation, max_tokens: usize, stopped: bool) -> &'static str {
    match !stopped && g.completion_tokens as usize >= max_tokens {
        true => "length",
        false => "stop",
    }
}

fn decode_rate(g: &Generation) -> f64 {
    match g.decode_secs > 0.0 {
        true => g.completion_tokens.saturating_sub(1) as f64 / g.decode_secs,
        false => 0.0,
    }
}

fn record_generation(state: &AppState, model: &str, g: &Generation, max_tokens: usize, stopped: bool) {
    let span = Span::current();
    let reason = finish_reason(g, max_tokens, stopped);
    span.record("prompt_tokens", g.prompt_tokens);
    span.record("completion_tokens", g.completion_tokens);
    span.record("ttft_ms", g.ttft_secs * 1000.0);
    span.record("decode_tok_s", decode_rate(g));
    span.record("finish_reason", reason);
    state.metrics.generation(
        model,
        g.prompt_tokens,
        g.completion_tokens,
        g.ttft_secs,
        g.completion_tokens.saturating_sub(1),
        g.decode_secs,
    );
    tracing::info!(
        prompt_tokens = g.prompt_tokens,
        completion_tokens = g.completion_tokens,
        ttft_ms = g.ttft_secs * 1000.0,
        decode_tok_s = decode_rate(g),
        finish_reason = reason,
        "generation finished"
    );
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn usage(g: &Generation) -> Value {
    json!({
        "prompt_tokens": g.prompt_tokens,
        "completion_tokens": g.completion_tokens,
        "total_tokens": g.prompt_tokens + g.completion_tokens,
    })
}

async fn collect(mut rx: mpsc::UnboundedReceiver<Gen>) -> Result<(Generation, bool), ApiError> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Gen::Text(t) => text.push_str(&t),
            Gen::Done(Ok(mut g), stopped) => {
                if g.text.is_empty() {
                    g.text = text;
                }
                return Ok((g, stopped));
            }
            Gen::Done(Err(e), _) => return Err(e),
        }
    }
    Err(ApiError::internal("generation ended without a result"))
}

async fn chat_completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    generate_endpoint(state, body, true).await
}

async fn completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    generate_endpoint(state, body, false).await
}

async fn generate_endpoint(state: Arc<AppState>, body: Value, chat: bool) -> Result<Response, ApiError> {
    let p = parse_params(&state, &body, chat)?;
    state.manager.resolve(&p.model)?;
    let max_tokens = p.sampling.max_tokens;
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let mut request = ChatRequest::new(p.messages, p.sampling);
    request.use_template = p.use_template;
    request.ignore_eos = p.ignore_eos;
    request.cancel = Some(cancel);
    request.openai_messages = p.openai_messages;
    if p.use_template {
        request.template_vars = p.template_vars;
    }
    log_prompt(&state, &request);
    let rx = start_generation(&state, p.model.clone(), request, max_tokens, p.stop)?;
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, &uuid::Uuid::new_v4().simple().to_string()[..24]);
    if p.stream {
        let events = sse_events(rx, guard, id, p.model, max_tokens, chat, p.include_usage, p.output);
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    let (g, stopped) = collect(rx).await?;
    drop(guard);
    let reason = finish_reason(&g, max_tokens, stopped);
    let created = now();
    let body = if chat {
        let (message, reason) = chat_message(&g.text, &p.output, reason);
        json!({
            "id": id, "object": "chat.completion", "created": created, "model": p.model,
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": usage(&g),
        })
    } else {
        json!({
            "id": id, "object": "text_completion", "created": created, "model": p.model,
            "choices": [{"index": 0, "text": g.text, "logprobs": null, "finish_reason": reason}],
            "usage": usage(&g),
        })
    };
    Ok(Json(body).into_response())
}

fn chat_message(text: &str, output: &OutputMode, reason: &'static str) -> (Value, &'static str) {
    if !output.parses() {
        return (json!({"role": "assistant", "content": text}), reason);
    }
    let mut parser = output.parser();
    parser.feed(text);
    parser.finish();
    let parsed = parser.result();
    let mut message = json!({"role": "assistant", "content": parsed.content});
    if let Some(r) = &parsed.reasoning {
        message["reasoning"] = json!(r);
        message["reasoning_content"] = json!(r);
    }
    if parsed.tool_calls.is_empty() {
        return (message, reason);
    }
    message["tool_calls"] = parsed
        .tool_calls
        .iter()
        .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}))
        .collect();
    (message, "tool_calls")
}

fn delta_json(d: &Delta) -> Value {
    match d {
        Delta::Content(t) => json!({"content": t}),
        Delta::Reasoning(t) => json!({"reasoning": t, "reasoning_content": t}),
        Delta::Call { index, id, name } => {
            json!({"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]})
        }
        Delta::Args { index, arguments } => json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}),
    }
}

fn delta_chunk(id: &str, model: &str, delta: Value) -> Value {
    let mut c = chunk(id, model, true, None, None);
    c["choices"][0]["delta"] = delta;
    c
}

fn chunk(id: &str, model: &str, chat: bool, delta: Option<(&str, &str)>, reason: Option<&str>) -> Value {
    let choice = match chat {
        true => {
            let d = match delta {
                Some((role, content)) if !role.is_empty() => json!({"role": role, "content": content}),
                Some((_, content)) => json!({"content": content}),
                None => json!({}),
            };
            json!({"index": 0, "delta": d, "finish_reason": reason})
        }
        false => json!({"index": 0, "text": delta.map(|d| d.1).unwrap_or(""), "logprobs": null, "finish_reason": reason}),
    };
    json!({
        "id": id,
        "object": if chat { "chat.completion.chunk" } else { "text_completion" },
        "created": now(),
        "model": model,
        "choices": [choice],
    })
}

fn sse_events(
    rx: mpsc::UnboundedReceiver<Gen>,
    guard: CancelOnDrop,
    id: String,
    model: String,
    max_tokens: usize,
    chat: bool,
    include_usage: bool,
    output: OutputMode,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let data = |v: Value| Ok(Event::default().data(v.to_string()));
    let opening = match chat {
        true => vec![data(chunk(&id, &model, true, Some(("assistant", "")), None))],
        false => Vec::new(),
    };
    let body = stream::unfold((rx, false), |(mut rx, done)| async move {
        if done {
            return None;
        }
        rx.recv().await.map(|ev| {
            let finished = matches!(ev, Gen::Done(..));
            (ev, (rx, finished))
        })
    });
    let mut streamed = false;
    let mut parser = output.parses().then(|| output.parser());
    let tail = body.flat_map(move |ev| {
        let _keep = &guard;
        let deltas = |_: &mut OutputParser, deltas: Vec<Delta>| -> Vec<Result<Event, Infallible>> {
            deltas.iter().map(|d| data(delta_chunk(&id, &model, delta_json(d)))).collect()
        };
        let items: Vec<Result<Event, Infallible>> = match ev {
            Gen::Text(t) => {
                streamed = true;
                match parser.as_mut() {
                    Some(p) => {
                        let d = p.feed(&t);
                        deltas(p, d)
                    }
                    None => vec![data(chunk(&id, &model, chat, Some(("", &t)), None))],
                }
            }
            Gen::Done(Ok(g), stopped) => {
                let mut out = Vec::new();
                let mut reason = finish_reason(&g, max_tokens, stopped);
                match parser.as_mut() {
                    Some(p) => {
                        let mut d = match streamed {
                            true => Vec::new(),
                            false => p.feed(&g.text),
                        };
                        d.extend(p.finish());
                        out.extend(deltas(p, d));
                        if p.called() {
                            reason = "tool_calls";
                        }
                    }
                    None if !streamed && !g.text.is_empty() => {
                        out.push(data(chunk(&id, &model, chat, Some(("", &g.text)), None)))
                    }
                    None => {}
                }
                out.push(data(chunk(&id, &model, chat, None, Some(reason))));
                if include_usage {
                    let mut u = chunk(&id, &model, chat, None, None);
                    u["choices"] = json!([]);
                    u["usage"] = usage(&g);
                    out.push(data(u));
                }
                out.push(Ok(Event::default().data("[DONE]")));
                out
            }
            Gen::Done(Err(e), _) => vec![
                data(json!({"error": {"message": e.message, "type": e.kind, "code": e.status.as_u16().to_string()}})),
                Ok(Event::default().data("[DONE]")),
            ],
        };
        stream::iter(items)
    });
    stream::iter(opening).chain(tail)
}

async fn bench(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let model = match body["model"].as_str() {
        Some(m) => m.to_string(),
        None => state.manager.default_model().ok_or_else(|| ApiError::bad_request("model is required"))?,
    };
    state.manager.resolve(&model)?;
    let prompts: Vec<String> = match body["prompts"].as_array() {
        Some(a) if !a.is_empty() => a.iter().filter_map(|p| p.as_str().map(String::from)).collect(),
        _ => return Err(ApiError::bad_request("prompts must be a non-empty array of strings")),
    };
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(64) as usize;
    let repeat = body["repeat"].as_u64().unwrap_or(1).clamp(1, 100) as usize;
    let mut runs = Vec::new();
    let mut ttfts = Vec::new();
    let mut rates = Vec::new();
    let (mut prompt_tokens, mut completion_tokens, mut failures) = (0u64, 0u64, 0u64);
    let started = Instant::now();
    for _ in 0..repeat {
        for prompt in &prompts {
            let cancel = Arc::new(AtomicBool::new(false));
            let _guard = CancelOnDrop(cancel.clone());
            let sampling = Sampling { temperature: 0.0, max_tokens, ..Sampling::default() };
            let mut request = ChatRequest::new(vec![Message::new(Role::User, prompt.clone())], sampling);
            request.use_template = false;
            request.cancel = Some(cancel);
            let outcome = match start_generation(&state, model.clone(), request, max_tokens, Vec::new()) {
                Ok(rx) => collect(rx).await.map(|(g, _)| g),
                Err(e) => Err(e),
            };
            match outcome {
                Ok(g) => {
                    prompt_tokens += g.prompt_tokens;
                    completion_tokens += g.completion_tokens;
                    ttfts.push(g.ttft_secs);
                    rates.push(decode_rate(&g));
                    runs.push(json!({
                        "prompt_tokens": g.prompt_tokens,
                        "completion_tokens": g.completion_tokens,
                        "ttft_secs": g.ttft_secs,
                        "decode_tok_s": decode_rate(&g),
                        "peak_memory_bytes": g.peak_memory_bytes,
                    }));
                }
                Err(e) => {
                    failures += 1;
                    runs.push(json!({"error": e.message}));
                }
            }
        }
    }
    ttfts.sort_by(f64::total_cmp);
    rates.sort_by(f64::total_cmp);
    let wall = started.elapsed().as_secs_f64();
    Ok(Json(json!({
        "model": model,
        "runs": runs,
        "summary": {
            "requests": runs.len(),
            "failures": failures,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_secs": wall,
            "tokens_per_sec": if wall > 0.0 { completion_tokens as f64 / wall } else { 0.0 },
            "ttft_secs": {"p50": percentile(&ttfts, 0.5), "p95": percentile(&ttfts, 0.95)},
            "decode_tok_s": {"p50": percentile(&rates, 0.5), "p95": percentile(&rates, 0.95)},
        },
    })))
}
