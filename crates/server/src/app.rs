use crate::capabilities::Usage;
use crate::config::ServerConfig;
use crate::error::ApiError;
use crate::landing::{self, ApiRoute, routes_json};
use crate::manager::{EngineFactory, GpuProbe, Manager, ModelSource};
use crate::metrics::{Gauges, Metrics, percentile};
use crate::stop::StopFilter;
use crate::tools::{Delta, OutputParser};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{MethodRouter, delete, get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream, StreamExt};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling, TokenLogprob};
use inference_bench_core::specrt_logprobs;
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
        total: crate::manager::TotalProbe,
        source: Option<Arc<dyn ModelSource>>,
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
            manager: Arc::new(Manager::new(config.clone(), factory, probe, total, metrics.clone(), source)),
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

    pub fn admitted_now(&self) -> usize {
        self.admitted.load(Ordering::SeqCst)
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

fn api_routes() -> Vec<(ApiRoute, MethodRouter<Arc<AppState>>)> {
    let r = |method, path, purpose| ApiRoute { method, path, purpose };
    vec![
        (r("GET", "/", "This page; JSON with Accept: application/json"), get(landing)),
        (r("GET", "/health", "Liveness, uptime and loaded models"), get(health)),
        (r("GET", "/metrics", "Prometheus metrics"), get(metrics)),
        (r("GET", "/v1/models", "List models (OpenAI)"), get(list_models)),
        (r("GET", "/v1/capabilities", "Routes, the request fields each accepts, and loaded and loadable models with spec features"), get(capabilities)),
        (r("POST", "/v1/chat/completions", "Chat completions, streaming, tools"), post(chat_completions)),
        (r("POST", "/v1/completions", "Text completions"), post(completions)),
        (r("POST", "/v1/messages", "Anthropic Messages, streaming, tools, thinking"), post(crate::dialects::messages)),
        (r("POST", "/v1/responses", "OpenAI Responses (stateless), streaming, function tools"), post(crate::dialects::responses)),
        (r("POST", "/v1/messages/count_tokens", "Anthropic token estimate"), post(crate::dialects::count_tokens)),
        (r("GET", "/props", "llama.cpp style model properties and chat template"), get(crate::dialects::props)),
        (r("GET", "/slots", "llama.cpp style slot list"), get(crate::dialects::slots)),
        (r("POST", "/v1/embeddings", "Embeddings (not supported yet)"), post(embeddings)),
        (r("POST", "/v1/bench", "Run a timed generation"), post(bench)),
        (r("POST", "/load", "Load the default model, or ?model=ID"), post(load_default)),
        (r("POST", "/unload", "Unload to free the GPU: 409 while busy, ?wait waits for requests to finish"), post(unload_default)),
        (r("GET", "/admin/models", "Model status and VRAM"), get(admin_models)),
        (r("POST", "/admin/models/{id}/load", "Load a model"), post(admin_load)),
        (r("POST", "/admin/models/{id}/unload", "Unload a model"), post(admin_unload)),
        (r("GET", "/admin/sessions", "Idle session snapshots on disk and sessions held on the GPU"), get(admin_sessions)),
        (r("DELETE", "/admin/sessions/{id}", "Delete a session snapshot"), delete(admin_session_delete)),
        (r("POST", "/admin/sessions/{id}/save", "Write a resident session to disk now"), post(admin_session_save)),
        (r("POST", "/admin/sessions/{id}/restore", "Verify a snapshot, it is restored by the next request that continues it"), post(admin_session_restore)),
        (r("POST", "/admin/sessions/{id}/strip", "Keep the tokens, drop the KV payload"), post(admin_session_strip)),
        (r("GET", "/admin/specs", "Model spec layers and update state"), get(admin_specs)),
    ]
}

pub fn router(state: Arc<AppState>) -> Router {
    let mut app = Router::new();
    for (route, handler) in api_routes() {
        app = app.route(route.path, handler);
    }
    let mut app = app
        .layer(axum::extract::DefaultBodyLimit::max(state.config.media.request_body_limit()))
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
        "/v1/messages" => "messages",
        "/v1/messages/count_tokens" => "count_tokens",
        "/v1/responses" => "responses",
        "/props" => "props",
        "/slots" => "slots",
        "/v1/embeddings" => "embeddings",
        "/v1/bench" => "bench",
        p if p.starts_with("/admin/") => "admin",
        _ => "other",
    }
}

async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    let path = req.uri().path();
    let open = path == "/" || path == "/health" || (path == "/metrics" && !state.config.protect_metrics);
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
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return headers.get("x-api-key").and_then(|v| v.to_str().ok()).map(str::trim);
    };
    let value = value.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

async fn landing(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or("localhost").to_string();
    let total = tokio::task::spawn_blocking(crate::gpu::smi_total_bytes).await.ok().flatten();
    let used = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || state.manager.gpu_used()).await.ok().flatten()
    };
    let parallel = state.config.concurrency();
    let models: Vec<Value> = state
        .manager
        .status()
        .into_iter()
        .map(|m| {
            let max_ctx = state.config.models.iter().find(|c| c.id == m.id).and_then(|c| c.max_ctx);
            json!({
                "id": m.id,
                "state": if m.loaded { "loaded" } else { "not loaded" },
                "loaded": m.loaded,
                "max_ctx": max_ctx,
                "parallel_requests": parallel,
                "active_requests": m.active_requests,
                "vram_bytes": m.vram_bytes,
            })
        })
        .collect();
    let s = state.metrics.summary();
    let info = json!({
        "name": "inference-bench",
        "version": env!("CARGO_PKG_VERSION"),
        "git_hash": env!("INFERENCE_BENCH_GIT_HASH"),
        "uptime_secs": state.started.elapsed().as_secs(),
        "base_url": format!("http://{host}/v1"),
        "auth_required": state.auth_required(),
        "gpu": {"used_bytes": used, "total_bytes": total},
        "models": models,
        "stats": {
            "requests": s.requests,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "ttft_p50_ms": s.ttft_p50_secs.map(|v| v * 1000.0),
        },
        "routes": routes_json(&api_routes().into_iter().map(|(r, _)| r).collect::<Vec<_>>()),
    });
    let wants_json = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("application/json") && !a.contains("text/html"));
    match wants_json {
        true => Json(info).into_response(),
        false => Html(landing::render(&info)).into_response(),
    }
}

async fn capabilities(State(state): State<Arc<AppState>>) -> Json<Value> {
    let routes: Vec<ApiRoute> = api_routes().into_iter().map(|(r, _)| r).collect();
    let status = state.manager.status();
    let loaded_ids: Vec<String> = status.iter().filter(|m| m.loaded).map(|m| m.id.clone()).collect();
    let models = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            status
                .into_iter()
                .map(|m| {
                    let cfg = state.config.models.iter().find(|c| c.id == m.id);
                    let described = cfg.map(|c| crate::capabilities::model_features(&state.config.specs.resolve(c.upstream_name())));
                    let (features, error) = match described {
                        Some(Ok(f)) => (serde_json::to_value(f).unwrap_or(Value::Null), Value::Null),
                        Some(Err(e)) => (Value::Null, json!(e)),
                        None => (Value::Null, Value::Null),
                    };
                    json!({
                        "id": m.id,
                        "state": if m.loaded { "loaded" } else if m.loading.is_some() { "loading" } else { "loadable" },
                        "loaded": m.loaded,
                        "max_ctx": cfg.and_then(|c| c.max_ctx),
                        "vram_bytes": m.vram_bytes,
                        "features": features,
                        "error": error,
                    })
                })
                .collect::<Vec<_>>()
        }
    })
    .await
    .unwrap_or_default();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "auth_required": state.auth_required(),
        "parallel_requests": state.config.concurrency(),
        "routes": crate::capabilities::routes_with_fields(&routes),
        "loaded": loaded_ids,
        "models": models,
    }))
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let status = state.manager.status();
    let unloaded: Vec<String> = status.iter().filter(|m| !m.loaded && m.loading.is_none()).map(|m| m.id.clone()).collect();
    let state_word = match (state.manager.loading_snapshot().is_empty(), status.iter().any(|m| m.loaded), unloaded.is_empty()) {
        (false, _, _) => "loading",
        (true, true, _) => "ok",
        (true, false, false) => "unloaded",
        (true, false, true) => "ok",
    };
    Json(json!({
        "status": state_word,
        "unloaded_models": unloaded,
        "uptime_secs": state.started.elapsed().as_secs(),
        "loaded_models": state.manager.loaded_snapshot().into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
        "loading": state.manager.loading_snapshot().into_iter().map(|(id, l)| json!({"id": id, "stage": l.stage, "done": l.done, "total": l.total, "elapsed_secs": l.elapsed_secs, "bytes_per_sec": l.bytes_per_sec})).collect::<Vec<_>>(),
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
        .map(|m| json!({"id": m.id, "object": "model", "created": 0, "owned_by": "inference-bench", "loaded": m.loaded, "status": if m.loaded { "loaded" } else if m.loading.is_some() { "loading" } else { "unloaded" }}))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

fn find_session(state: &AppState, id: &str) -> Result<(String, Arc<inference_bench_core::specrt_sessions::Shared>), ApiError> {
    let all = state.manager.sessions_all();
    if let Some(found) = all.into_iter().find(|(_, s)| s.store.list().iter().any(|e| e.id == id) || s.resident.lock().unwrap().iter().any(|r| r.id == id)) {
        return Ok(found);
    }
    Err(ApiError::new(StatusCode::NOT_FOUND, "not_found", format!("no session {id}")))
}

async fn admin_sessions(State(state): State<Arc<AppState>>) -> Json<Value> {
    let ttl_for = |s: &inference_bench_core::specrt_sessions::Shared| s.options.ttl_seconds();
    let mut data = Vec::new();
    for (model, s) in state.manager.sessions_all() {
        let ttl = ttl_for(&s);
        for e in s.store.list() {
            data.push(json!({"id": e.id, "model": model, "state": if e.stripped { "stripped" } else { "disk" }, "tokens": e.tokens, "bytes": e.bytes, "created": e.created, "last_used": e.last_used, "expires": e.expires(ttl)}));
        }
        for r in s.resident.lock().unwrap().iter() {
            data.push(json!({"id": r.id, "model": model, "state": "resident", "tokens": r.tokens, "bytes": 0, "last_used": r.last_used, "expires": Value::Null}));
        }
    }
    Json(json!({"object": "list", "data": data}))
}

async fn admin_session_delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (_, s) = find_session(&state, &id)?;
    s.store.delete(&id).map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{e:#}")))?;
    Ok(Json(json!({"id": id, "deleted": true})))
}

async fn admin_session_save(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (_, s) = find_session(&state, &id)?;
    if !s.request_save(&id) {
        return Err(ApiError::new(StatusCode::CONFLICT, "conflict", format!("session {id} is not held on the GPU")));
    }
    Ok(Json(json!({"id": id, "queued": true})))
}

async fn admin_session_restore(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (_, s) = find_session(&state, &id)?;
    let e = s.store.verify(&id).map_err(|e| ApiError::new(StatusCode::CONFLICT, "conflict", format!("{e:#}")))?;
    Ok(Json(json!({"id": id, "verified": true, "tokens": e.tokens, "stripped": e.stripped})))
}

async fn admin_session_strip(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (_, s) = find_session(&state, &id)?;
    let e = s.store.strip(&id, inference_bench_core::specrt_sessions::now_seconds()).map_err(|e| ApiError::new(StatusCode::CONFLICT, "conflict", format!("{e:#}")))?;
    Ok(Json(json!({"id": id, "stripped": true, "bytes": e.bytes})))
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

#[derive(serde::Deserialize, Default)]
struct ModelQuery {
    model: Option<String>,
    wait: Option<String>,
}

impl ModelQuery {
    fn waits(&self) -> bool {
        self.wait.as_deref().is_some_and(|w| !matches!(w, "0" | "false" | "no"))
    }
}

async fn load_default(State(state): State<Arc<AppState>>, axum::extract::Query(q): axum::extract::Query<ModelQuery>) -> Result<Json<Value>, ApiError> {
    let id = state.manager.requested_model(q.model.as_deref())?;
    admin_load(State(state), Path(id)).await
}

async fn unload_default(State(state): State<Arc<AppState>>, axum::extract::Query(q): axum::extract::Query<ModelQuery>) -> Result<Json<Value>, ApiError> {
    let waits = q.waits();
    let id = state.manager.requested_model(q.model.as_deref())?;
    release_model(&state, &id, waits).await
}

const UNLOAD_WAIT: std::time::Duration = std::time::Duration::from_secs(300);

async fn release_model(state: &Arc<AppState>, id: &str, wait: bool) -> Result<Json<Value>, ApiError> {
    let cfg = state.manager.resolve(id)?;
    let deadline = Instant::now() + UNLOAD_WAIT;
    while state.manager.is_busy(&cfg.id) || state.admitted_now() > 0 {
        if !wait {
            return Err(ApiError::new(StatusCode::CONFLICT, "conflict", format!("{id} has requests in flight")));
        }
        if Instant::now() > deadline {
            return Err(ApiError::new(StatusCode::CONFLICT, "conflict", format!("{id} still has requests in flight after {} s", UNLOAD_WAIT.as_secs())));
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    state.manager.release(&cfg.id);
    Ok(Json(json!({"id": cfg.id, "loaded": state.manager.is_loaded(&cfg.id), "status": "unloaded"})))
}

async fn admin_load(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let cfg = state.manager.resolve(&id)?;
    if !state.manager.is_loaded(&id) {
        state.manager.preload(&cfg).await?;
    }
    Ok(Json(json!({"id": id, "loaded": true})))
}

async fn admin_unload(State(state): State<Arc<AppState>>, Path(id): Path<String>, axum::extract::Query(q): axum::extract::Query<ModelQuery>) -> Result<Json<Value>, ApiError> {
    release_model(&state, &id, q.waits()).await
}

async fn embeddings() -> ApiError {
    ApiError::new(
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented",
        "none of the configured backends support embeddings",
    )
}

pub(crate) struct Params {
    pub(crate) model: String,
    pub(crate) shown_model: String,
    messages: Vec<Message>,
    openai_messages: Option<Vec<Value>>,
    template_vars: Value,
    pub(crate) output: OutputMode,
    pub(crate) stop: Vec<String>,
    use_template: bool,
    pub(crate) sampling: Sampling,
    pub(crate) stream: bool,
    include_usage: bool,
    ignore_eos: bool,
    pub(crate) n: usize,
    logprobs: Option<u32>,
    response_format: Option<inference_bench_core::specrt_grammar::Schema>,
    think: Option<ThinkCap>,
}

#[derive(Clone)]
pub(crate) struct ThinkCap {
    budget: usize,
    wrapup: String,
}

#[derive(Clone, Default)]
pub(crate) struct OutputMode {
    tools: Vec<Value>,
    tool_parsing: bool,
    reasoning: bool,
    parser: Option<String>,
}

impl OutputMode {
    pub(crate) fn parses(&self) -> bool {
        self.tool_parsing || self.reasoning || self.parser.as_deref() == Some("harmony")
    }

    pub(crate) fn parser(&self) -> OutputParser {
        OutputParser::new(&self.tools, self.tool_parsing, self.reasoning).for_parser(self.parser.as_deref())
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

fn reject_unsupported_fields(body: &Value) -> Result<(), ApiError> {
    let Some(obj) = body.as_object() else { return Ok(()) };
    for (key, value) in obj {
        let key = key.as_str();
        let ok = match crate::capabilities::lookup(key).map(|f| f.usage) {
            Some(Usage::Read | Usage::Ignored) => true,
            Some(Usage::Validated) => inference_bench_core::specrt_grammar::parse_response_format(value).is_ok(),
            Some(Usage::ZeroOnly) => value.is_null() || value.as_f64() == Some(0.0),
            None => false,
        };
        if !ok {
            let why = match key {
                "response_format" => {
                    return Err(ApiError::bad_request(match inference_bench_core::specrt_grammar::parse_response_format(value) {
                        Err(e) => format!("response_format: {e}"),
                        Ok(_) => "response_format: invalid".to_string(),
                    }))
                }
                "frequency_penalty" | "presence_penalty" => "value is not supported",
                _ => "field is not supported",
            };
            return Err(ApiError::bad_request(format!("{key}: {why}")));
        }
    }
    Ok(())
}

pub const MAX_CHOICES: u64 = 16;

fn choice_count(body: &Value) -> Result<usize, ApiError> {
    match &body["n"] {
        Value::Null => Ok(1),
        v => match v.as_u64() {
            Some(n) if (1..=MAX_CHOICES).contains(&n) => Ok(n as usize),
            _ => Err(ApiError::bad_request(format!("n must be an integer from 1 to {MAX_CHOICES}"))),
        },
    }
}

fn logprob_request(body: &Value, chat: bool) -> Result<Option<u32>, ApiError> {
    let max = specrt_logprobs::MAX_TOP as u64;
    let top = |v: &Value, name: &str| match v.as_u64() {
        Some(n) if n <= max => Ok(n as u32),
        _ => Err(ApiError::bad_request(format!("{name} must be an integer from 0 to {max}"))),
    };
    if !chat {
        return match &body["logprobs"] {
            Value::Null => Ok(None),
            v => top(v, "logprobs").map(Some),
        };
    }
    let on = match &body["logprobs"] {
        Value::Null => false,
        Value::Bool(b) => *b,
        _ => return Err(ApiError::bad_request("logprobs must be true or false")),
    };
    let n = match &body["top_logprobs"] {
        Value::Null => 0,
        v => top(v, "top_logprobs")?,
    };
    if n > 0 && !on {
        return Err(ApiError::bad_request("top_logprobs needs logprobs: true"));
    }
    Ok(on.then_some(n))
}

fn normalize_effort(effort: &str) -> &str {
    match effort {
        "minimal" => "low",
        "xhigh" | "max" => "high",
        other => other,
    }
}

fn think_cap(state: &AppState, body: &Value, reasoning: bool, max_tokens: usize) -> Option<ThinkCap> {
    if !reasoning {
        return None;
    }
    let requested = body["reasoning_budget_tokens"]
        .as_u64()
        .or_else(|| body["thinking_budget_tokens"].as_u64())
        .map(|n| n as usize)
        .or(state.config.reasoning_budget_tokens)
        .filter(|n| *n > 0);
    let room = (state.config.answer_room && max_tokens != inference_bench_core::chat::UNTIL_FULL && max_tokens >= 1024)
        .then(|| max_tokens - (max_tokens / 4).max(512));
    let budget = requested.or(room)?;
    Some(ThinkCap { budget: budget.max(1), wrapup: state.config.reasoning_wrapup.clone() })
}

fn chat_setup(state: &AppState, body: &Value, model: &str) -> Result<(Value, OutputMode), ApiError> {
    let mut vars = toml_json(&state.config.chat_template_kwargs);
    let user = match &body["chat_template_kwargs"] {
        Value::Object(o) => o.clone(),
        Value::Null => Default::default(),
        _ => return Err(ApiError::bad_request("chat_template_kwargs must be an object")),
    };
    let effort = body["reasoning_effort"].as_str().or_else(|| body["reasoning"]["effort"].as_str()).map(normalize_effort);
    if let Some(effort) = effort {
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
        Value::String(s) if s == "none" => "none",
        _ => "auto",
    };
    if !tools.is_empty() {
        vars["tools"] = Value::Array(tools.clone());
    }
    let tool_parser = state.config.tool_parser_for(state.manager.tool_format(model));
    let output = OutputMode {
        tool_parsing: tool_parser.is_some() && !tools.is_empty() && choice != "none",
        reasoning: state.config.reasoning_parser.is_some() && vars["enable_thinking"] != json!(false),
        parser: tool_parser,
        tools,
    };
    Ok((vars, output))
}

fn forced_tool_schema(state: &AppState, body: &Value, model: &str, output: &OutputMode) -> Result<Option<inference_bench_core::specrt_grammar::Schema>, ApiError> {
    let choice = &body["tool_choice"];
    if !(choice == "required" || choice.is_object()) {
        inference_bench_core::specrt_grammar::parse_tool_choice(choice, &output.tools, true).map_err(ApiError::bad_request)?;
        return Ok(None);
    }
    let id = state.manager.resolve(model)?.id;
    let xml = match state.config.tool_parser_for(state.manager.tool_format(&id)).as_deref() {
        Some("xml_function") => true,
        Some("hermes") => false,
        _ => return Err(ApiError::bad_request("tool_choice required or named needs a model whose chat template has a known tool call format")),
    };
    inference_bench_core::specrt_grammar::parse_tool_choice(choice, &output.tools, xml).map_err(|e| ApiError::bad_request(format!("tool_choice: {e}")))
}

fn parse_params(state: &AppState, body: &Value, chat: bool) -> Result<Params, ApiError> {
    reject_unsupported_fields(body)?;
    let model = state.manager.requested_model(body["model"].as_str())?;
    let messages = if chat { chat_messages(body)? } else { vec![Message::new(Role::User, completion_prompt(body)?)] };
    let max_tokens = body["max_completion_tokens"]
        .as_u64()
        .or_else(|| body["max_tokens"].as_u64())
        .map(|n| n as usize)
        .or(state.config.default_max_tokens)
        .unwrap_or(inference_bench_core::chat::UNTIL_FULL);
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
        true => chat_setup(state, body, &state.manager.resolve(&model)?.id)?,
        false => (Value::Null, OutputMode::default()),
    };
    let think = think_cap(state, body, output.reasoning, max_tokens);
    let forced = match chat {
        true => forced_tool_schema(state, body, &model, &output)?,
        false => None,
    };
    if forced.is_some() && !body["response_format"].is_null() && body["response_format"]["type"] != "text" {
        return Err(ApiError::bad_request("tool_choice required or named cannot be combined with response_format"));
    }
    Ok(Params {
        shown_model: model.clone(),
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
        n: choice_count(body)?,
        logprobs: logprob_request(body, chat)?,
        response_format: match forced {
            Some(f) => Some(f),
            None => inference_bench_core::specrt_grammar::parse_response_format(&body["response_format"]).map_err(|e| ApiError::bad_request(format!("response_format: {e}")))?,
        },
        think,
    })
}

fn system_for_developer(mut list: Vec<Value>) -> Vec<Value> {
    for m in &mut list {
        if m["role"] == "developer" {
            m["role"] = json!("system");
        }
    }
    list
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

pub(crate) enum Gen {
    Text(String),
    Logprob(TokenLogprob),
    Done(Result<Generation, ApiError>, bool),
}

pub(crate) struct CancelOnDrop(Arc<AtomicBool>);

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
    mut request: ChatRequest,
    max_tokens: usize,
    stop: Vec<String>,
    needs_batched: bool,
    think: Option<ThinkCap>,
) -> Result<mpsc::UnboundedReceiver<Gen>, ApiError> {
    let mut admission = state.admit()?;
    let (tx, rx) = mpsc::unbounded_channel();
    let state = state.clone();
    let metrics = state.metrics.clone();
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
            let caps = lease.backend().capabilities();
            let refusal = if !request.media.is_empty() && !caps.vision {
                let why = lease.backend().media_refusal().unwrap_or_else(|| "this model's config has no vision_config".to_string());
                Some(format!("model {model} does not accept image or video input: {why}"))
            } else if needs_batched && !caps.batched {
                Some(format!("n > 1 needs the batched engine: load {model} with more than one parallel request (max_num_seqs above 1)"))
            } else if request.logprobs.is_some() && !caps.logprobs {
                Some(format!("logprobs need the batched engine: load {model} with more than one parallel request (max_num_seqs above 1)"))
            } else if request.response_format.is_some() && !caps.batched {
                Some(format!("response_format needs the batched engine: load {model} with more than one parallel request (max_num_seqs above 1)"))
            } else {
                None
            };
            if let Some(msg) = refusal {
                let _ = tx.send(Gen::Done(Err(ApiError::bad_request(msg)), false));
                return;
            }
            if request.logprobs.is_some() {
                let lp_tx = tx.clone();
                request.on_logprob = Some(Arc::new(move |lp: &TokenLogprob| {
                    let _ = lp_tx.send(Gen::Logprob(lp.clone()));
                }));
            }
            Span::current().record("backend", lease.model().backend_label.as_str());
            let queued_ms = started.elapsed().as_secs_f64() * 1000.0;
            tracing::debug!(queued_ms, "generation started");
            let flag = request.cancel.clone().unwrap_or_default();
            let text_tx = tx.clone();
            let span = Span::current();
            let outcome = tokio::task::spawn_blocking(move || {
                span.in_scope(|| {
                    let mut filter = StopFilter::new(stop);
                    let send = |t: &str| {
                        if !t.is_empty() && text_tx.send(Gen::Text(t.to_string())).is_err() {
                            flag.store(true, Ordering::SeqCst);
                        }
                    };
                    let acc = std::cell::RefCell::new(String::new());
                    let mut emit = |t: &str| {
                        if think.is_some() {
                            acc.borrow_mut().push_str(t);
                        }
                        let out = filter.feed(t);
                        send(&out);
                        if filter.hit {
                            flag.store(true, Ordering::SeqCst);
                        }
                    };
                    let total = request.sampling.max_tokens;
                    let first_cap = think.as_ref().map_or(total, |c| c.budget.min(total));
                    request.sampling.max_tokens = first_cap;
                    let mut result = lease.backend().chat(&request, &mut |t| emit(t));
                    if let (Some(cap), Ok(first)) = (&think, &result)
                        && first_cap < total
                        && first.completion_tokens as usize >= first_cap
                        && !flag.load(Ordering::SeqCst)
                        && thinking_open(&acc.borrow())
                    {
                        metrics.thinking_wrapup();
                        emit(&format!("{}</think>\n\n", cap.wrapup));
                        let first = first.clone();
                        request.continuation = Some(acc.borrow().clone());
                        request.sampling.max_tokens = total.saturating_sub(first.completion_tokens as usize).max(1);
                        result = lease.backend().chat(&request, &mut |t| emit(t)).map(|second| join_generations(first, second));
                    }
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

fn thinking_open(generated: &str) -> bool {
    !generated.contains("</think>")
}

fn join_generations(first: Generation, second: Generation) -> Generation {
    Generation {
        text: second.text,
        completion_tokens: first.completion_tokens + second.completion_tokens,
        decode_secs: first.decode_secs + second.decode_secs,
        peak_memory_bytes: second.peak_memory_bytes.or(first.peak_memory_bytes),
        ..first
    }
}

pub(crate) fn finish_reason(g: &Generation, max_tokens: usize, stopped: bool) -> &'static str {
    match !stopped && g.completion_tokens as usize >= max_tokens {
        true => "length",
        false => "stop",
    }
}

pub(crate) fn decode_rate(g: &Generation) -> f64 {
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
    state.metrics.finished(model, reason);
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

pub(crate) fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn usage(g: &Generation) -> Value {
    let mut details = json!({"cached_tokens": g.cached_tokens});
    if g.vision_tokens > 0 {
        details["vision_tokens"] = json!(g.vision_tokens);
    }
    json!({
        "prompt_tokens": g.prompt_tokens,
        "completion_tokens": g.completion_tokens,
        "total_tokens": g.prompt_tokens + g.completion_tokens,
        "prompt_tokens_details": details,
    })
}

pub(crate) struct Collected {
    pub(crate) generation: Generation,
    pub(crate) stopped: bool,
    logprobs: Vec<TokenLogprob>,
}

async fn collect(rx: mpsc::UnboundedReceiver<Gen>) -> Result<(Generation, bool), ApiError> {
    collect_all(rx).await.map(|c| (c.generation, c.stopped))
}

pub(crate) async fn collect_all(mut rx: mpsc::UnboundedReceiver<Gen>) -> Result<Collected, ApiError> {
    let mut text = String::new();
    let mut logprobs = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Gen::Text(t) => text.push_str(&t),
            Gen::Logprob(lp) => logprobs.push(lp),
            Gen::Done(Ok(mut g), stopped) => {
                if g.text.is_empty() {
                    g.text = text;
                }
                logprobs.truncate(g.completion_tokens as usize);
                return Ok(Collected { generation: g, stopped, logprobs });
            }
            Gen::Done(Err(e), _) => return Err(e),
        }
    }
    Err(ApiError::internal("generation ended without a result"))
}

pub(crate) async fn chat_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let salt = cache_salt(&headers, &body_salt(&body), state_shares_keys(&state));
    generate_endpoint(state, body, true, salt).await
}

async fn completions(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let salt = cache_salt(&headers, &body_salt(&body), state_shares_keys(&state));
    generate_endpoint(state, body, false, salt).await
}

pub(crate) fn state_shares_keys(state: &AppState) -> bool {
    state.config.prefix_cache_share_across_keys || !state.auth_required()
}

pub(crate) fn body_salt(body: &Value) -> String {
    body["cache_salt"].as_str().unwrap_or_default().to_string()
}

pub fn cache_salt(headers: &HeaderMap, body_salt: &str, shared: bool) -> String {
    use sha2::{Digest, Sha256};
    if shared {
        return body_salt.to_string();
    }
    let key = bearer(headers).unwrap_or_default();
    let digest: [u8; 32] = Sha256::digest(key.as_bytes()).into();
    let tag: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("key:{tag}|{body_salt}")
}

pub(crate) async fn prepare(state: Arc<AppState>, body: Value, chat: bool, salt: String) -> Result<Prepared, ApiError> {
    if chat && body["tools"].as_array().is_some_and(|t| !t.is_empty()) {
        let cfg = state.manager.resolve(&state.manager.requested_model(body["model"].as_str())?)?;
        if !state.manager.is_loaded(&cfg.id) {
            state.manager.preload(&cfg).await?;
        }
    }
    let mut p = parse_params(&state, &body, chat)?;
    let requested = p.model.clone();
    let (cfg, remapped) = state.manager.resolve_named(&requested)?;
    p.model = cfg.id;
    p.shown_model = if remapped { requested } else { p.model.clone() };
    let max_tokens = p.sampling.max_tokens;
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let mut request = ChatRequest::new(std::mem::take(&mut p.messages), p.sampling);
    request.use_template = p.use_template;
    request.ignore_eos = p.ignore_eos;
    request.cancel = Some(cancel);
    request.openai_messages = p.openai_messages.take().map(system_for_developer);
    request.cache_salt = Some(salt);
    if let Some(list) = &request.openai_messages
        && inference_bench_core::media::has_media(list)
    {
        let list = list.clone();
        let limits = state.config.media.clone();
        request.media_max_pixels = limits.max_pixels;
        request.media_max_video_secs = limits.max_video_secs;
        request.media_max_patches = limits.max_patches;
        request.media_max_video_frames = limits.max_video_frames;
        request.media = tokio::task::spawn_blocking(move || inference_bench_core::media::extract(&list, &limits))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .map_err(|e| ApiError::bad_request(e.0))?;
    }
    if p.use_template {
        request.template_vars = p.template_vars.take();
    }
    request.logprobs = p.logprobs;
    request.response_format = p.response_format.take();
    log_prompt(&state, &request);
    let mut receivers = Vec::with_capacity(p.n);
    for i in 0..p.n {
        let mut one = request.clone();
        if i > 0 {
            one.sampling.seed = Some(request.sampling.seed.unwrap_or(0).wrapping_add(i as u64));
        }
        receivers.push(start_generation(&state, p.model.clone(), one, max_tokens, p.stop.clone(), p.n > 1, p.think.clone())?);
    }
    Ok(Prepared { p, receivers, guard, max_tokens })
}

pub(crate) struct Prepared {
    pub(crate) p: Params,
    pub(crate) receivers: Vec<mpsc::UnboundedReceiver<Gen>>,
    pub(crate) guard: CancelOnDrop,
    pub(crate) max_tokens: usize,
}

async fn generate_endpoint(state: Arc<AppState>, body: Value, chat: bool, salt: String) -> Result<Response, ApiError> {
    let Prepared { p, receivers, guard, max_tokens } = prepare(state, body, chat, salt).await?;
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, &uuid::Uuid::new_v4().simple().to_string()[..24]);
    if p.stream {
        let events = sse_events(receivers, guard, id, p.shown_model, max_tokens, chat, p.include_usage, p.output);
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    let mut done = Vec::with_capacity(p.n);
    for rx in receivers {
        done.push(collect_all(rx).await?);
    }
    drop(guard);
    let created = now();
    let choices: Vec<Value> = done
        .iter()
        .enumerate()
        .map(|(index, c)| {
            let reason = finish_reason(&c.generation, max_tokens, c.stopped);
            let logprobs = p.logprobs.map(|_| logprobs_json(&c.logprobs, chat));
            if chat {
                let (message, reason) = chat_message(&c.generation.text, &p.output, reason);
                json!({"index": index, "message": message, "logprobs": logprobs, "finish_reason": reason})
            } else {
                json!({"index": index, "text": c.generation.text, "logprobs": logprobs, "finish_reason": reason})
            }
        })
        .collect();
    let total = merged_usage(done.iter().map(|c| &c.generation));
    let body = json!({
        "id": id, "object": if chat { "chat.completion" } else { "text_completion" }, "created": created, "model": p.shown_model,
        "choices": choices,
        "usage": total,
    });
    Ok(Json(body).into_response())
}

fn merged_usage<'a>(all: impl Iterator<Item = &'a Generation>) -> Value {
    let mut sum: Option<Generation> = None;
    for g in all {
        match sum.as_mut() {
            None => sum = Some(g.clone()),
            Some(s) => s.completion_tokens += g.completion_tokens,
        }
    }
    usage(&sum.unwrap_or_default())
}

fn chat_logprob_entry(lp: &TokenLogprob) -> Value {
    json!({
        "token": lp.token,
        "logprob": lp.logprob,
        "bytes": lp.bytes,
        "top_logprobs": lp.top.iter().map(|(t, b, l)| json!({"token": t, "logprob": l, "bytes": b})).collect::<Vec<_>>(),
    })
}

fn logprobs_json(list: &[TokenLogprob], chat: bool) -> Value {
    if chat {
        return json!({"content": list.iter().map(chat_logprob_entry).collect::<Vec<_>>()});
    }
    let mut offset = 0;
    let mut offsets = Vec::with_capacity(list.len());
    for lp in list {
        offsets.push(offset);
        offset += lp.token.len();
    }
    json!({
        "tokens": list.iter().map(|l| l.token.clone()).collect::<Vec<_>>(),
        "token_logprobs": list.iter().map(|l| l.logprob).collect::<Vec<_>>(),
        "top_logprobs": list.iter().map(|l| l.top.iter().map(|(t, _, v)| (t.clone(), json!(v))).collect::<serde_json::Map<_, _>>()).collect::<Vec<_>>(),
        "text_offset": offsets,
    })
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
        Delta::Reasoning(t) => json!({"reasoning_content": t}),
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

struct ChoiceStream {
    streamed: bool,
    parser: Option<OutputParser>,
}

fn indexed(mut v: Value, index: usize) -> Value {
    v["choices"][0]["index"] = json!(index);
    v
}

#[allow(clippy::too_many_arguments)]
fn sse_events(
    receivers: Vec<mpsc::UnboundedReceiver<Gen>>,
    guard: CancelOnDrop,
    id: String,
    model: String,
    max_tokens: usize,
    chat: bool,
    include_usage: bool,
    output: OutputMode,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let data = |v: Value| Ok(Event::default().data(v.to_string()));
    let n = receivers.len();
    let opening: Vec<Result<Event, Infallible>> = match chat {
        true => (0..n).map(|i| data(indexed(chunk(&id, &model, true, Some(("assistant", "")), None), i))).collect(),
        false => Vec::new(),
    };
    let tagged = receivers.into_iter().enumerate().map(|(i, rx)| {
        stream::unfold((rx, false), move |(mut rx, done)| async move {
            if done {
                return None;
            }
            rx.recv().await.map(|ev| {
                let finished = matches!(ev, Gen::Done(..));
                ((i, ev), (rx, finished))
            })
        })
        .boxed()
    });
    let body = stream::select_all(tagged);
    let mut choices: Vec<ChoiceStream> = (0..n).map(|_| ChoiceStream { streamed: false, parser: output.parses().then(|| output.parser()) }).collect();
    let mut finished = 0usize;
    let mut failed = false;
    let mut totals: Option<Generation> = None;
    let tail = body.flat_map(move |(index, ev)| {
        let _keep = &guard;
        let at = |v: Value| data(indexed(v, index));
        let deltas = |deltas: Vec<Delta>| -> Vec<Result<Event, Infallible>> {
            deltas.iter().map(|d| at(delta_chunk(&id, &model, delta_json(d)))).collect()
        };
        let c = &mut choices[index];
        let items: Vec<Result<Event, Infallible>> = match ev {
            _ if failed => Vec::new(),
            Gen::Text(t) => {
                c.streamed = true;
                match c.parser.as_mut() {
                    Some(p) => deltas(p.feed(&t)),
                    None => vec![at(chunk(&id, &model, chat, Some(("", &t)), None))],
                }
            }
            Gen::Logprob(lp) => {
                let mut v = chunk(&id, &model, chat, None, None);
                v["choices"][0]["logprobs"] = logprobs_json(std::slice::from_ref(&lp), chat);
                vec![at(v)]
            }
            Gen::Done(Ok(g), stopped) => {
                let mut out = Vec::new();
                let mut reason = finish_reason(&g, max_tokens, stopped);
                match c.parser.as_mut() {
                    Some(p) => {
                        let mut d = match c.streamed {
                            true => Vec::new(),
                            false => p.feed(&g.text),
                        };
                        d.extend(p.finish());
                        out.extend(deltas(d));
                        if p.called() {
                            reason = "tool_calls";
                        }
                    }
                    None if !c.streamed && !g.text.is_empty() => out.push(at(chunk(&id, &model, chat, Some(("", &g.text)), None))),
                    None => {}
                }
                out.push(at(chunk(&id, &model, chat, None, Some(reason))));
                match totals.as_mut() {
                    None => totals = Some(g.clone()),
                    Some(t) => t.completion_tokens += g.completion_tokens,
                }
                finished += 1;
                if finished == n {
                    if include_usage {
                        let mut u = chunk(&id, &model, chat, None, None);
                        u["choices"] = json!([]);
                        u["usage"] = usage(&totals.clone().unwrap_or_default());
                        out.push(data(u));
                    }
                    out.push(Ok(Event::default().data("[DONE]")));
                }
                out
            }
            Gen::Done(Err(e), _) => {
                failed = true;
                vec![
                    data(json!({"error": {"message": e.message, "type": e.kind, "code": e.status.as_u16().to_string()}})),
                    Ok(Event::default().data("[DONE]")),
                ]
            }
        };
        stream::iter(items)
    });
    stream::iter(opening).chain(tail)
}

async fn bench(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let model = state.manager.requested_model(body["model"].as_str())?;
    let model = state.manager.resolve(&model)?.id;
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
            let outcome = match start_generation(&state, model.clone(), request, max_tokens, Vec::new(), false, None) {
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
