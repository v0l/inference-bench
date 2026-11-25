use crate::app::{AppState, CancelOnDrop};
use crate::error::ApiError;
use crate::images::Update;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use inference_bench_core::specrt_media::{self as media, MediaEvent, MediaJob, ModelKind};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const ACCEPTED: &[&str] = &["model", "prompt", "seconds", "size", "seed", "negative_prompt", "steps", "guidance", "fps", "frames", "sampler", "user"];
const KEEP: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Queued,
    InProgress,
    Completed,
    Failed(String),
    Cancelled,
}

impl Status {
    fn label(&self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::InProgress => "in_progress",
            Status::Completed => "completed",
            Status::Failed(_) => "failed",
            Status::Cancelled => "cancelled",
        }
    }
}

pub struct VideoJob {
    pub id: String,
    pub model: String,
    pub created: i64,
    pub size: String,
    pub seconds: f64,
    pub status: Status,
    pub step: usize,
    pub steps: usize,
    pub frames_done: usize,
    pub frames: usize,
    pub fps: u32,
    pub mp4: Option<Arc<Vec<u8>>>,
    pub usage: Value,
    guard: Option<CancelOnDrop>,
}

impl VideoJob {
    pub fn progress(&self) -> u32 {
        match self.status {
            Status::Completed => 100,
            _ => {
                let s = if self.steps == 0 { 0.0 } else { self.step as f64 / self.steps as f64 };
                let f = if self.frames == 0 { 0.0 } else { self.frames_done as f64 / self.frames as f64 };
                ((0.9 * s + 0.1 * f) * 100.0).floor().min(99.0) as u32
            }
        }
    }

    fn json(&self) -> Value {
        let mut v = json!({
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status.label(),
            "progress": self.progress(),
            "created_at": self.created,
            "size": self.size,
            "seconds": format!("{}", self.seconds),
            "frames": self.frames,
            "fps": self.fps,
            "step": self.step,
            "steps": self.steps,
            "frames_decoded": self.frames_done,
        });
        if let Status::Failed(m) = &self.status {
            v["error"] = json!({"message": m});
        }
        if self.status == Status::Completed {
            v["completed_at"] = json!(crate::app::now());
            v["usage"] = self.usage.clone();
        }
        v
    }
}

#[derive(Default)]
pub struct Jobs {
    jobs: Mutex<BTreeMap<String, VideoJob>>,
}

impl Jobs {
    fn insert(&self, job: VideoJob) {
        let mut map = self.jobs.lock().unwrap();
        if map.len() >= KEEP
            && let Some(old) = map.iter().filter(|(_, j)| matches!(j.status, Status::Completed | Status::Failed(_) | Status::Cancelled)).min_by_key(|(_, j)| j.created).map(|(k, _)| k.clone())
        {
            map.remove(&old);
        }
        map.insert(job.id.clone(), job);
    }

    fn with<T>(&self, id: &str, f: impl FnOnce(&mut VideoJob) -> T) -> Option<T> {
        self.jobs.lock().unwrap().get_mut(id).map(f)
    }

    fn remove(&self, id: &str) -> Option<VideoJob> {
        self.jobs.lock().unwrap().remove(id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoRequest {
    pub model: Option<String>,
    pub job: MediaJob,
    pub seconds: Option<f64>,
}

fn int(v: &Value, key: &'static str) -> Result<Option<u64>, ApiError> {
    match v {
        Value::Null => Ok(None),
        v => v.as_u64().map(Some).ok_or_else(|| ApiError::bad_request(format!("{key} must be a whole number")).with_code("invalid_value", key)),
    }
}

pub fn parse(body: &Value) -> Result<VideoRequest, ApiError> {
    let obj = body.as_object().ok_or_else(|| ApiError::bad_request("the body must be a JSON object"))?;
    if let Some(k) = obj.keys().find(|k| !ACCEPTED.contains(&k.as_str())) {
        return Err(ApiError::bad_request(format!("unknown field {k}; this route reads {}", ACCEPTED.join(", "))));
    }
    let prompt = body["prompt"].as_str().filter(|p| !p.trim().is_empty()).ok_or_else(|| ApiError::bad_request("prompt is required").with_code("invalid_value", "prompt"))?;
    let (width, height) = match body["size"].as_str() {
        None => (0, 0),
        Some(s) => media::parse_size(s).ok_or_else(|| ApiError::bad_request(format!("size {s:?} is not WIDTHxHEIGHT, for example 832x480")).with_code("invalid_value", "size"))?,
    };
    let seconds = match &body["seconds"] {
        Value::Null => None,
        Value::String(s) => Some(s.trim().parse::<f64>().map_err(|_| ApiError::bad_request(format!("seconds {s:?} is not a number")))?),
        v => Some(v.as_f64().ok_or_else(|| ApiError::bad_request("seconds must be a number"))?),
    };
    if seconds.is_some_and(|s| !(s > 0.0 && s <= 60.0)) {
        return Err(ApiError::bad_request("seconds must be above 0 and at most 60"));
    }
    let guidance = match &body["guidance"] {
        Value::Null => None,
        v => Some(v.as_f64().ok_or_else(|| ApiError::bad_request("guidance must be a number"))? as f32),
    };
    let job = MediaJob {
        kind: ModelKind::Video,
        prompt: prompt.to_string(),
        negative_prompt: body["negative_prompt"].as_str().unwrap_or("").to_string(),
        n: 1,
        width,
        height,
        seed: int(&body["seed"], "seed")?.unwrap_or_else(|| uuid::Uuid::new_v4().as_u64_pair().0 >> 12),
        steps: int(&body["steps"], "steps")?.map(|s| s as usize),
        guidance,
        sampler: body["sampler"].as_str().map(String::from),
        frames: int(&body["frames"], "frames")?.map(|f| f as usize),
        preview_every: None,
        offload_text_encoder: None,
        fps: int(&body["fps"], "fps")?.map(|f| f as u32),
    };
    Ok(VideoRequest { model: body["model"].as_str().map(String::from), job, seconds })
}

pub fn frames_for(seconds: f64, fps: u32, frame_step: usize) -> usize {
    let step = frame_step.max(1);
    let raw = (seconds * f64::from(fps)).round() as usize;
    (raw.saturating_sub(1) + step / 2) / step * step + 1
}

pub async fn create(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let req = parse(&body)?;
    let ticket = crate::images::admit(&state)?;
    let seconds = req.seconds.filter(|_| req.job.frames.is_none());
    let (model, mut rx, guard) = crate::images::start_job(&state, req.model.as_deref(), req.job.clone(), ticket, seconds).await?;
    let id = format!("video_{}", uuid::Uuid::new_v4().simple());
    let fps = req.job.fps.unwrap_or(16);
    let frames = req.job.frames.unwrap_or(0);
    let steps = req.job.steps.unwrap_or(0);
    let size = if req.job.width == 0 { String::new() } else { format!("{}x{}", req.job.width, req.job.height) };
    let job = VideoJob { id: id.clone(), model, created: crate::app::now(), size, seconds: frames as f64 / f64::from(fps.max(1)), status: Status::Queued, step: 0, steps, frames_done: 0, frames, fps, mp4: None, usage: Value::Null, guard: Some(guard) };
    let body = job.json();
    state.videos.insert(job);
    let st = state.clone();
    let jid = id.clone();
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            let done = st.videos.with(&jid, |j| match update {
                Update::Event(MediaEvent::Progress { step, steps, .. }) => {
                    j.status = Status::InProgress;
                    j.step = step;
                    j.steps = steps;
                    false
                }
                Update::Event(MediaEvent::Frames { done, total }) => {
                    j.frames_done = done;
                    j.frames = total;
                    false
                }
                Update::Event(_) => false,
                Update::Done(Ok(out)) => {
                    j.mp4 = out.mp4.clone().map(Arc::new);
                    j.fps = out.fps;
                    if let Some(f) = out.frames.as_ref().and_then(|v| v.first()) {
                        j.frames = f.len();
                        j.frames_done = f.len();
                        if let Some(p) = f.first() {
                            j.size = format!("{}x{}", p.width, p.height);
                        }
                    }
                    j.seconds = j.frames as f64 / f64::from(j.fps.max(1));
                    j.usage = json!({"steps": out.steps, "steps_per_second": out.steps_per_sec(), "total_seconds": out.total_secs, "admission": out.admission});
                    j.status = Status::Completed;
                    j.guard = None;
                    true
                }
                Update::Done(Err(e)) => {
                    j.status = if e.kind == "cancelled" { Status::Cancelled } else { Status::Failed(e.message) };
                    j.guard = None;
                    true
                }
            });
            if done != Some(false) {
                break;
            }
        }
    });
    Ok((axum::http::StatusCode::OK, Json(body)).into_response())
}

pub async fn get_one(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let v = state.videos.with(&id, |j| j.json()).ok_or_else(|| ApiError::not_found(format!("no video {id}")))?;
    Ok(Json(v).into_response())
}

pub async fn content(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let found = state.videos.with(&id, |j| (j.status.clone(), j.mp4.clone())).ok_or_else(|| ApiError::not_found(format!("no video {id}")))?;
    match found {
        (Status::Completed, Some(mp4)) => Ok(([(header::CONTENT_TYPE, "video/mp4")], mp4.as_ref().clone()).into_response()),
        (s, _) => Err(ApiError::not_found(format!("video {id} is {}, its content is not ready", s.label())).with_code("video_not_ready", "id")),
    }
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let mut job = state.videos.remove(&id).ok_or_else(|| ApiError::not_found(format!("no video {id}")))?;
    if job.guard.take().is_some() {
        job.status = Status::Cancelled;
    }
    let mut v = job.json();
    v["deleted"] = json!(true);
    Ok(Json(v).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_video_request_parses_openai_names_and_the_extras() {
        let r = parse(&json!({"prompt": "a fox", "seconds": "5", "size": "832x480", "seed": 3, "fps": 16, "steps": 50, "guidance": 5.0, "negative_prompt": "blur"})).unwrap();
        assert_eq!((r.job.width, r.job.height, r.job.seed, r.job.fps, r.job.steps, r.seconds), (832, 480, 3, Some(16), Some(50), Some(5.0)));
        assert_eq!(r.job.kind, ModelKind::Video);
        for (body, word) in [(json!({"size": "8x8"}), "prompt"), (json!({"prompt": "x", "seconds": "lots"}), "not a number"), (json!({"prompt": "x", "seconds": 0}), "above 0"), (json!({"prompt": "x", "n": 2}), "unknown field n")] {
            let e = parse(&body).unwrap_err();
            assert!(e.message.contains(word), "{body}: {}", e.message);
        }
    }

    #[test]
    fn seconds_become_a_frame_count_the_causal_vae_takes() {
        assert_eq!(frames_for(5.0, 16, 4), 81);
        assert_eq!(frames_for(0.3, 16, 2), 5);
        assert_eq!(frames_for(1.0, 16, 4), 17);
    }

    #[test]
    fn progress_counts_steps_then_decoded_frames() {
        let mut j = VideoJob { id: "v".into(), model: "m".into(), created: 0, size: String::new(), seconds: 0.0, status: Status::InProgress, step: 25, steps: 50, frames_done: 0, frames: 81, fps: 16, mp4: None, usage: Value::Null, guard: None };
        assert_eq!(j.progress(), 45);
        j.step = 50;
        j.frames_done = 81;
        assert_eq!(j.progress(), 99);
        j.status = Status::Completed;
        assert_eq!(j.progress(), 100);
    }
}
