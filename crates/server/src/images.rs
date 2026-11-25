use crate::app::{AppState, CancelOnDrop};
use crate::error::ApiError;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use inference_bench_core::specrt_media::{self as media, MediaEvent, MediaJob, MediaOutput, ModelKind};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::{Semaphore, mpsc};

const ACCEPTED: &[&str] = &["model", "prompt", "n", "size", "seed", "response_format", "output_format", "negative_prompt", "steps", "guidance", "sampler", "stream", "partial_images", "user", "quality", "style", "background", "moderation"];

pub struct Gate {
    running: Arc<Semaphore>,
    admitted: AtomicUsize,
    limit: usize,
    queue: usize,
}

impl Gate {
    pub fn new(jobs: usize, queue: usize) -> Gate {
        Gate { running: Arc::new(Semaphore::new(jobs.max(1))), admitted: AtomicUsize::new(0), limit: jobs.max(1) + queue, queue }
    }

    pub fn waiting(&self) -> usize {
        let running = self.limit - self.queue - self.running.available_permits();
        self.admitted().saturating_sub(running)
    }

    pub fn running(&self) -> usize {
        self.limit - self.queue - self.running.available_permits()
    }

    fn admit(self: &Arc<Self>, retry: u64) -> Result<Ticket, ApiError> {
        let mut current = self.admitted.load(Ordering::SeqCst);
        loop {
            if current >= self.limit {
                let mut e = ApiError::busy(retry);
                e.message = format!("the image queue is full ({current} jobs admitted, {} allowed): retry shortly", self.limit);
                return Err(e);
            }
            match self.admitted.compare_exchange(current, current + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Ok(Ticket { gate: self.clone() }),
                Err(c) => current = c,
            }
        }
    }

    pub fn admitted(&self) -> usize {
        self.admitted.load(Ordering::SeqCst)
    }
}

pub(crate) struct Ticket {
    gate: Arc<Gate>,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.gate.admitted.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageRequest {
    pub model: Option<String>,
    pub job: MediaJob,
    pub url: bool,
    pub stream: bool,
}

fn int(v: &Value, key: &str) -> Result<Option<u64>, ApiError> {
    match v {
        Value::Null => Ok(None),
        v => v.as_u64().map(Some).ok_or_else(|| ApiError::bad_request(format!("{key} must be a whole number")).with_code("invalid_value", "n")),
    }
}

pub fn parse(body: &Value, partial_default: usize) -> Result<ImageRequest, ApiError> {
    let obj = body.as_object().ok_or_else(|| ApiError::bad_request("the body must be a JSON object"))?;
    if let Some(k) = obj.keys().find(|k| !ACCEPTED.contains(&k.as_str())) {
        return Err(ApiError::bad_request(format!("unknown field {k}; this route reads {}", ACCEPTED.join(", "))));
    }
    let prompt = body["prompt"].as_str().filter(|p| !p.trim().is_empty()).ok_or_else(|| ApiError::bad_request("prompt is required").with_code("invalid_value", "prompt"))?;
    let size = body["size"].as_str().unwrap_or("auto");
    let (width, height) = match size {
        "auto" => (0, 0),
        s => media::parse_size(s).ok_or_else(|| ApiError::bad_request(format!("size {s:?} is not WIDTHxHEIGHT, for example 1024x1024")).with_code("invalid_value", "size"))?,
    };
    let url = match body["response_format"].as_str().unwrap_or("b64_json") {
        "b64_json" => false,
        "url" => true,
        other => return Err(ApiError::bad_request(format!("response_format {other:?} is not b64_json or url"))),
    };
    if let Some(f) = body["output_format"].as_str().filter(|f| *f != "png") {
        return Err(ApiError::bad_request(format!("output_format {f:?} is not supported: png only")));
    }
    let seed = int(&body["seed"], "seed")?.unwrap_or_else(|| uuid::Uuid::new_v4().as_u64_pair().0 >> 12);
    let guidance = match &body["guidance"] {
        Value::Null => None,
        v => Some(v.as_f64().ok_or_else(|| ApiError::bad_request("guidance must be a number"))? as f32),
    };
    let job = MediaJob {
        kind: ModelKind::Image,
        prompt: prompt.to_string(),
        negative_prompt: body["negative_prompt"].as_str().unwrap_or("").to_string(),
        n: int(&body["n"], "n")?.unwrap_or(1) as usize,
        width,
        height,
        seed,
        steps: int(&body["steps"], "steps")?.map(|s| s as usize),
        guidance,
        sampler: body["sampler"].as_str().map(String::from),
        frames: None,
        offload_text_encoder: None,
        fps: None,
        preview_every: int(&body["partial_images"], "partial_images")?.map(|k| k as usize).or((partial_default > 0).then_some(partial_default)),
    };
    Ok(ImageRequest { model: body["model"].as_str().map(String::from), job, url, stream: body["stream"].as_bool().unwrap_or(false) })
}

pub(crate) enum Update {
    Event(MediaEvent),
    Done(Result<MediaOutput, ApiError>),
}

fn sweep(dir: &std::path::Path, ttl: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > ttl);
        if old {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn picture_json(state: &AppState, p: &media::Picture, url: bool, host: &str) -> Result<Value, ApiError> {
    let png = p.png().map_err(|e| ApiError::internal(format!("png: {e}")))?;
    if !url {
        return Ok(json!({"b64_json": base64::engine::general_purpose::STANDARD.encode(&png), "seed": p.seed}));
    }
    let dir = state.config.images.dir();
    std::fs::create_dir_all(&dir).map_err(|e| ApiError::internal(format!("{}: {e}", dir.display())))?;
    sweep(&dir, Duration::from_secs(state.config.images.file_ttl_secs));
    let id = uuid::Uuid::new_v4().simple().to_string();
    std::fs::write(dir.join(format!("{id}.png")), &png).map_err(|e| ApiError::internal(format!("write image: {e}")))?;
    let scheme = if state.config.tls.is_some() { "https" } else { "http" };
    Ok(json!({"url": format!("{scheme}://{host}/v1/images/files/{id}.png"), "seed": p.seed}))
}

fn usage(out: &MediaOutput) -> Value {
    json!({"steps": out.steps, "steps_per_second": out.steps_per_sec(), "total_seconds": out.total_secs, "admission": out.admission})
}

async fn start(state: &Arc<AppState>, req: &ImageRequest) -> Result<(String, mpsc::UnboundedReceiver<Update>, CancelOnDrop), ApiError> {
    let ticket = state.images.admit(state.config.retry_after_secs)?;
    start_job(state, req.model.as_deref(), req.job.clone(), ticket, None).await
}

pub(crate) fn admit(state: &Arc<AppState>) -> Result<Ticket, ApiError> {
    state.images.admit(state.config.retry_after_secs)
}

pub(crate) async fn start_job(state: &Arc<AppState>, model: Option<&str>, job: MediaJob, ticket: Ticket, seconds: Option<f64>) -> Result<(String, mpsc::UnboundedReceiver<Update>, CancelOnDrop), ApiError> {
    let model = state.manager.requested_model(model)?;
    let flag = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop::new(flag.clone());
    let (tx, rx) = mpsc::unbounded_channel();
    let st = state.clone();
    let mut job = job;
    let route = if job.kind == ModelKind::Video { "/v1/videos" } else { "/v1/images/generations" };
    let shown = model.clone();
    let label = model.clone();
    let span = tracing::Span::current();
    span.record("model", model.as_str());
    tokio::spawn(async move {
        let _ticket = ticket;
        let permit = tokio::select! {
            p = st.images.running.clone().acquire_owned() => p.ok(),
            _ = tx.closed() => None,
        };
        let Some(_permit) = permit else {
            st.metrics.cancelled();
            return;
        };
        let mut lease = match st.manager.acquire(&model).await {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(Update::Done(Err(e)));
                return;
            }
        };
        let info = lease.backend().media_info();
        let checked = match &info {
            None => Err(ApiError::bad_request(format!("model {model} is a text model; {route} needs a {} model", job.kind.label()))),
            Some(i) => {
                if job.width == 0 {
                    let (w, h) = media::parse_size(&i.default_size).unwrap_or((1024, 1024));
                    job.width = w;
                    job.height = h;
                }
                if job.kind == ModelKind::Video {
                    match seconds {
                        Some(s) => job.frames = Some(crate::videos::frames_for(s, job.fps.unwrap_or(i.fps), i.frame_step)),
                        None if job.frames.is_none() => job.frames = Some(i.default_frames),
                        None => {}
                    }
                }
                i.check(&job).map_err(|e| ApiError::bad_request(e.to_string()))
            }
        };
        if let Err(e) = checked {
            let _ = tx.send(Update::Done(Err(e)));
            return;
        }
        let events = tx.clone();
        let cancel = flag.clone();
        let started = std::time::Instant::now();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut on_event = |e: MediaEvent| events.send(Update::Event(e)).is_ok() && !cancel.load(Ordering::SeqCst);
            lease.backend().generate_media(&job, &mut on_event)
        })
        .await;
        let result = match outcome {
            Ok(Ok(o)) => {
                st.metrics.image_job(&label, o.images.len() as u64, o.steps as u64, o.step_secs, started.elapsed().as_secs_f64());
                Ok(o)
            }
            Ok(Err(e)) => {
                let e = ApiError::from(e);
                if e.kind == "cancelled" {
                    st.metrics.cancelled();
                    tracing::info!("media job cancelled");
                }
                Err(e)
            }
            Err(e) => Err(ApiError::internal(format!("image task failed: {e}"))),
        };
        let _ = tx.send(Update::Done(result));
    });
    Ok((shown, rx, guard))
}

pub async fn generations(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let req = parse(&body, state.config.images.partial_images)?;
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or("localhost").to_string();
    let (model, mut rx, guard) = start(&state, &req).await?;
    let created = crate::app::now();
    if req.stream {
        let st = state.clone();
        let events = stream::unfold((rx, Some(guard), false), move |(mut rx, guard, done)| {
            let (st, host, model) = (st.clone(), host.clone(), model.clone());
            async move {
                if done {
                    drop(guard);
                    return None;
                }
                let update = rx.recv().await?;
                let (events, last) = match update {
                    Update::Event(MediaEvent::Progress { image, step, steps }) => (vec![("image_generation.progress", json!({"type": "image_generation.progress", "index": image, "step": step, "steps": steps}))], false),
                    Update::Event(MediaEvent::Preview { image, step, picture }) => match picture.png() {
                        Ok(png) => (vec![("image_generation.partial_image", json!({"type": "image_generation.partial_image", "index": image, "partial_image_index": step, "b64_json": base64::engine::general_purpose::STANDARD.encode(png), "size": format!("{}x{}", picture.width, picture.height)}))], false),
                        Err(_) => (Vec::new(), false),
                    },
                    Update::Event(MediaEvent::Frames { .. }) => (Vec::new(), false),
                    Update::Done(Ok(out)) => {
                        let mut list = Vec::new();
                        for (i, p) in out.images.iter().enumerate() {
                            let mut v = match picture_json(&st, p, req.url, &host) {
                                Ok(v) => v,
                                Err(e) => json!({"error": e.message}),
                            };
                            v["type"] = json!("image_generation.completed");
                            v["index"] = json!(i);
                            v["created_at"] = json!(created);
                            v["model"] = json!(model);
                            v["size"] = json!(format!("{}x{}", p.width, p.height));
                            v["output_format"] = json!("png");
                            v["usage"] = usage(&out);
                            list.push(("image_generation.completed", v));
                        }
                        (list, true)
                    }
                    Update::Done(Err(e)) => (vec![("error", json!({"type": "error", "error": {"message": e.message, "type": e.kind, "code": e.status.as_u16()}}))], true),
                };
                let items: Vec<Result<Event, Infallible>> = events.into_iter().map(|(name, v)| Ok(Event::default().event(name).data(v.to_string()))).collect();
                Some((stream::iter(items), (rx, guard, last)))
            }
        })
        .flatten();
        return Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response());
    }
    let out = loop {
        match rx.recv().await {
            Some(Update::Event(_)) => {}
            Some(Update::Done(r)) => break r?,
            None => return Err(ApiError::internal("the image job stopped without a result")),
        }
    };
    drop(guard);
    let data = out.images.iter().map(|p| picture_json(&state, p, req.url, &host)).collect::<Result<Vec<_>, _>>()?;
    let size = out.images.first().map(|p| format!("{}x{}", p.width, p.height));
    Ok(Json(json!({"created": created, "model": model, "data": data, "size": size, "output_format": "png", "usage": usage(&out)})).into_response())
}

pub async fn file(State(state): State<Arc<AppState>>, Path(name): Path<String>) -> Result<Response, ApiError> {
    let stem = name.strip_suffix(".png").filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit())).ok_or_else(|| ApiError::not_found(format!("no image {name}")))?;
    let path = state.config.images.dir().join(format!("{stem}.png"));
    let ttl = Duration::from_secs(state.config.images.file_ttl_secs);
    let fresh = std::fs::metadata(&path).and_then(|m| m.modified()).ok().and_then(|t| SystemTime::now().duration_since(t).ok()).is_some_and(|age| age <= ttl);
    if !fresh {
        let _ = std::fs::remove_file(&path);
        return Err(ApiError::not_found(format!("image {name} has expired or never existed")));
    }
    let bytes = std::fs::read(&path).map_err(|e| ApiError::internal(format!("read image: {e}")))?;
    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_parses_with_openai_names_and_the_extras() {
        let r = parse(&json!({"prompt": "a fox", "n": 2, "size": "64x32", "seed": 7, "response_format": "url", "negative_prompt": "blur", "steps": 3, "guidance": 4.5, "sampler": "euler", "stream": true}), 0).unwrap();
        assert_eq!((r.job.width, r.job.height, r.job.n, r.job.seed, r.job.steps, r.job.guidance), (64, 32, 2, 7, Some(3), Some(4.5)));
        assert!(r.url && r.stream);
        assert_eq!(r.job.negative_prompt, "blur");
        assert_eq!(r.job.kind, ModelKind::Image);
    }

    #[test]
    fn a_bad_request_is_refused_before_it_is_queued() {
        for (body, word) in [
            (json!({"size": "64x64"}), "prompt"),
            (json!({"prompt": "x", "size": "big"}), "WIDTHxHEIGHT"),
            (json!({"prompt": "x", "response_format": "jpeg"}), "b64_json or url"),
            (json!({"prompt": "x", "output_format": "webp"}), "png only"),
            (json!({"prompt": "x", "wat": 1}), "unknown field wat"),
            (json!({"prompt": "x", "n": -1}), "whole number"),
        ] {
            let e = parse(&body, 0).unwrap_err();
            assert_eq!(e.status.as_u16(), 400, "{body}");
            assert!(e.message.contains(word), "{body}: {}", e.message);
        }
    }

    #[test]
    fn the_gate_admits_jobs_plus_queue_then_refuses_with_retry_after() {
        let g = Arc::new(Gate::new(1, 2));
        let held: Vec<Ticket> = (0..3).map(|_| g.admit(2).unwrap()).collect();
        let e = g.admit(2).err().expect("the fourth job is refused");
        assert_eq!((e.status.as_u16(), e.retry_after), (429, Some(2)));
        assert!(e.message.contains("image queue is full"), "{}", e.message);
        drop(held);
        assert_eq!(g.admitted(), 0);
        g.admit(2).unwrap();
    }
}
