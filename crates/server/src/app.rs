use crate::config::ServerConfig;
use crate::error::ApiError;
use crate::manager::{BackendFactory, GpuProbe, Manager};
use crate::metrics::{Gauges, Metrics, percentile};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream, StreamExt};
use inference_bench_core::backend::Request as GenRequest;
use inference_bench_core::logging::{preview, redact_key};
use inference_bench_core::result::Generation;
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
        factory: Arc<dyn BackendFactory>,
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
        }
    }

    pub fn auth_required(&self) -> bool {
        !self.keys.is_empty()
    }

    fn key_valid(&self, presented: &str) -> bool {
        self.keys.iter().fold(false, |ok, k| ok | constant_time_eq(k.as_bytes(), presented.as_bytes()))
    }

    fn admit(self: &Arc<Self>) -> Result<Admission, ApiError> {
        let limit = self.config.max_concurrent + self.config.queue_depth;
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
    prompt: String,
    max_tokens: usize,
    stream: bool,
    include_usage: bool,
}

fn parse_params(state: &AppState, body: &Value, chat: bool) -> Result<Params, ApiError> {
    let model = match body["model"].as_str() {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => state.manager.default_model().ok_or_else(|| ApiError::bad_request("model is required"))?,
    };
    let prompt = if chat { chat_prompt(body)? } else { completion_prompt(body)? };
    let max_tokens = body["max_completion_tokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(state.config.default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }
    Ok(Params {
        model,
        prompt,
        max_tokens,
        stream: body["stream"].as_bool().unwrap_or(false),
        include_usage: body["stream_options"]["include_usage"].as_bool().unwrap_or(false),
    })
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

pub fn chat_prompt(body: &Value) -> Result<String, ApiError> {
    let messages = body["messages"]
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad_request("messages must be a non-empty array"))?;
    if messages.len() == 1 && messages[0]["role"].as_str().unwrap_or("user") == "user" {
        return Ok(content_text(&messages[0]["content"]));
    }
    let mut out = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        out.push_str(&format!("{role}: {}\n", content_text(&m["content"])));
    }
    out.push_str("assistant:");
    Ok(out)
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
    Done(Result<Generation, ApiError>),
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn log_prompt(state: &AppState, prompt: &str) {
    if state.config.log_prompts {
        tracing::info!(prompt = %preview(prompt, 2000), "prompt");
    } else {
        tracing::trace!(prompt = %prompt, prompt_bytes = prompt.len(), "prompt");
    }
}

fn start_generation(
    state: &Arc<AppState>,
    model: String,
    prompt: String,
    max_tokens: usize,
    cancel: Arc<AtomicBool>,
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
                    let _ = tx.send(Gen::Done(Err(e)));
                    return;
                }
            };
            admission.start();
            Span::current().record("backend", lease.model().backend_label.as_str());
            let queued_ms = started.elapsed().as_secs_f64() * 1000.0;
            tracing::debug!(queued_ms, "generation started");
            let flag = cancel.clone();
            let text_tx = tx.clone();
            let span = Span::current();
            let outcome = tokio::task::spawn_blocking(move || {
                span.in_scope(|| {
                    let backend = lease.backend();
                    backend.set_cancel(Some(flag.clone()));
                    let result = backend.generate(&GenRequest { prompt, max_tokens }, &mut |t| {
                        if text_tx.send(Gen::Text(t.to_string())).is_err() {
                            flag.store(true, Ordering::SeqCst);
                        }
                    });
                    backend.set_cancel(None);
                    result
                })
            })
            .await;
            let result = match outcome {
                Ok(r) => r.map_err(ApiError::from),
                Err(e) => Err(ApiError::internal(format!("generation task failed: {e}"))),
            };
            match &result {
                Ok(g) => record_generation(&state, &model, g, max_tokens),
                Err(e) if e.kind == "cancelled" => {
                    state.metrics.cancelled();
                    tracing::info!("generation cancelled by client disconnect");
                }
                Err(e) => tracing::error!(status = e.status.as_u16(), error = %e.message, "generation failed"),
            }
            let _ = tx.send(Gen::Done(result));
        }
        .instrument(span),
    );
    Ok(rx)
}

fn finish_reason(g: &Generation, max_tokens: usize) -> &'static str {
    match g.completion_tokens as usize >= max_tokens {
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

fn record_generation(state: &AppState, model: &str, g: &Generation, max_tokens: usize) {
    let span = Span::current();
    let reason = finish_reason(g, max_tokens);
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

async fn collect(mut rx: mpsc::UnboundedReceiver<Gen>) -> Result<Generation, ApiError> {
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Gen::Text(t) => text.push_str(&t),
            Gen::Done(Ok(mut g)) => {
                if g.text.is_empty() {
                    g.text = text;
                }
                return Ok(g);
            }
            Gen::Done(Err(e)) => return Err(e),
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
    log_prompt(&state, &p.prompt);
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let rx = start_generation(&state, p.model.clone(), p.prompt, p.max_tokens, cancel)?;
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, &uuid::Uuid::new_v4().simple().to_string()[..24]);
    if p.stream {
        let events = sse_events(rx, guard, id, p.model, p.max_tokens, chat, p.include_usage);
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    let g = collect(rx).await?;
    drop(guard);
    let reason = finish_reason(&g, p.max_tokens);
    let created = now();
    let body = if chat {
        json!({
            "id": id, "object": "chat.completion", "created": created, "model": p.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": g.text}, "finish_reason": reason}],
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
            let finished = matches!(ev, Gen::Done(_));
            (ev, (rx, finished))
        })
    });
    let mut streamed = false;
    let tail = body.flat_map(move |ev| {
        let _keep = &guard;
        let items: Vec<Result<Event, Infallible>> = match ev {
            Gen::Text(t) => {
                streamed = true;
                vec![data(chunk(&id, &model, chat, Some(("", &t)), None))]
            }
            Gen::Done(Ok(g)) => {
                let mut out = Vec::new();
                if !streamed && !g.text.is_empty() {
                    out.push(data(chunk(&id, &model, chat, Some(("", &g.text)), None)));
                }
                out.push(data(chunk(&id, &model, chat, None, Some(finish_reason(&g, max_tokens)))));
                if include_usage {
                    let mut u = chunk(&id, &model, chat, None, None);
                    u["choices"] = json!([]);
                    u["usage"] = usage(&g);
                    out.push(data(u));
                }
                out.push(Ok(Event::default().data("[DONE]")));
                out
            }
            Gen::Done(Err(e)) => vec![
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
            let outcome = match start_generation(&state, model.clone(), prompt.clone(), max_tokens, cancel) {
                Ok(rx) => collect(rx).await,
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
