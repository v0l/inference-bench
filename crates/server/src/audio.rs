use crate::app::AppState;
use crate::error::ApiError;
use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_media::ModelKind;
use inference_bench_core::specrt_speech::service::{Heard, Request};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Json,
    Text,
    VerboseJson,
}

impl Format {
    pub fn parse(s: &str) -> Result<Format, ApiError> {
        match s.trim() {
            "" | "json" => Ok(Format::Json),
            "text" => Ok(Format::Text),
            "verbose_json" => Ok(Format::VerboseJson),
            other => Err(ApiError::bad_request(format!("response_format {other:?} is not json, text or verbose_json")).with_code("invalid_value", "response_format")),
        }
    }
}

#[derive(Debug, Default)]
pub struct Form {
    pub file: Option<Vec<u8>>,
    pub model: String,
    pub response_format: String,
    pub language: String,
    pub prompt: String,
    pub temperature: String,
}

pub struct Parsed {
    pub pcm: Vec<f32>,
    pub rate: u32,
    pub model: String,
    pub format: Format,
    pub request: Request,
}

pub fn parse(form: Form, max_bytes: usize) -> Result<Parsed, ApiError> {
    let file = form.file.ok_or_else(|| ApiError::bad_request("the form has no file").with_code("invalid_value", "file"))?;
    if file.len() > max_bytes {
        return Err(ApiError::new(axum::http::StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", format!("the file is {} bytes, media.max_audio_bytes allows {max_bytes}", file.len())).with_code("invalid_value", "file"));
    }
    let (pcm, rate) = wav::read_wav(&file).map_err(|e| ApiError::bad_request(format!("file: {e:#}")).with_code("invalid_value", "file"))?;
    if rate == 0 {
        return Err(ApiError::bad_request("file: the WAV has a sample rate of 0").with_code("invalid_value", "file"));
    }
    let temperature = match form.temperature.trim() {
        "" => None,
        t => Some(t.parse::<f64>().ok().filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).ok_or_else(|| ApiError::bad_request(format!("temperature {t:?} is not a number from 0 to 1")).with_code("invalid_value", "temperature"))?),
    };
    let language = Some(form.language.trim().to_lowercase()).filter(|l| !l.is_empty());
    Ok(Parsed { pcm, rate, model: form.model.trim().to_string(), format: Format::parse(&form.response_format)?, request: Request { language, prompt: form.prompt, temperature } })
}

async fn read_form(mut multipart: Multipart, max_bytes: usize) -> Result<Form, ApiError> {
    let bad = |e: axum::extract::multipart::MultipartError| ApiError::bad_request(format!("the multipart form: {}", e.body_text()));
    let mut form = Form::default();
    while let Some(field) = multipart.next_field().await.map_err(bad)? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let bytes = field.bytes().await.map_err(bad)?;
                if bytes.len() > max_bytes {
                    return Err(ApiError::new(axum::http::StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", format!("the file is {} bytes, media.max_audio_bytes allows {max_bytes}", bytes.len())).with_code("invalid_value", "file"));
                }
                form.file = Some(bytes.to_vec());
            }
            "model" => form.model = field.text().await.map_err(bad)?,
            "response_format" => form.response_format = field.text().await.map_err(bad)?,
            "language" => form.language = field.text().await.map_err(bad)?,
            "prompt" => form.prompt = field.text().await.map_err(bad)?,
            "temperature" => form.temperature = field.text().await.map_err(bad)?,
            _ => {}
        }
    }
    Ok(form)
}

pub fn pick(state: &AppState, asked: &str) -> Result<String, ApiError> {
    let ids = state.manager.transcription_ids();
    if !asked.is_empty()
        && let Ok(cfg) = state.manager.resolve(asked)
        && (ids.contains(&cfg.id) || state.config.model(asked).is_some())
    {
        return Ok(cfg.id);
    }
    let loaded = state.manager.loaded_transcribers();
    loaded
        .first()
        .map(|(id, _)| id.clone())
        .or_else(|| ids.first().cloned())
        .ok_or_else(|| ApiError::not_found("no speech to text model is configured; add a Whisper model folder to the server config").with_code("model_not_found", "model"))
}

fn response(heard: &Heard, format: Format) -> Response {
    match format {
        Format::Json => Json(json!({"text": heard.text})).into_response(),
        Format::Text => ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], heard.text.clone()).into_response(),
        Format::VerboseJson => Json(json!({
            "task": "transcribe",
            "language": heard.language,
            "duration": heard.duration,
            "text": heard.text,
            "segments": heard.segments,
        }))
        .into_response(),
    }
}

pub async fn transcriptions(State(state): State<Arc<AppState>>, multipart: Multipart) -> Result<Response, ApiError> {
    let max = state.config.media.max_audio_bytes;
    let parsed = parse(read_form(multipart, max).await?, max)?;
    let ticket = crate::images::admit(&state)?;
    let id = pick(&state, &parsed.model)?;
    tracing::Span::current().record("model", id.as_str());
    let _permit = state.images.permit().await.ok_or_else(|| ApiError::unavailable("the media queue closed"))?;
    let (id, service, held) = state.manager.transcriber(&id).await?;
    let seconds = parsed.pcm.len() as f64 / f64::from(parsed.rate);
    let started = std::time::Instant::now();
    let (pcm, rate, request) = (parsed.pcm, parsed.rate, parsed.request);
    let heard = tokio::task::spawn_blocking(move || service.transcribe(pcm, rate, request))
        .await
        .map_err(|e| ApiError::internal(format!("transcription task failed: {e}")))?
        .map_err(|e| ApiError::internal(format!("{e:#}")))?;
    drop((held, ticket));
    let spent = started.elapsed().as_secs_f64();
    state.metrics.transcription(&id, seconds, spent);
    tracing::info!(model = %id, audio_secs = seconds, secs = spent, tokens = heard.tokens.len(), "transcribed");
    Ok(response(&heard, parsed.format))
}

fn size_mb(id: &str, state: &AppState) -> u64 {
    let Some(cfg) = state.config.model(id) else { return 0 };
    let Some(dir) = inference_bench_core::specrt_speech::service::resolve_dir(&state.config.specs.resolve(cfg.upstream_name())) else { return 0 };
    let bytes: u64 = std::fs::read_dir(dir).into_iter().flatten().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    bytes.div_ceil(1 << 20)
}

pub async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let listed = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let status: Vec<_> = state.manager.status().into_iter().filter(|m| m.kind == ModelKind::Transcription).collect();
            let default = status.iter().find(|m| m.loaded).or(status.first()).map(|m| m.id.clone());
            status
                .iter()
                .map(|m| json!({"id": m.id, "object": "model", "label": m.id, "size_mb": size_mb(&m.id, &state), "installed": true, "default": Some(&m.id) == default.as_ref(), "loaded": m.loaded}))
                .collect::<Vec<_>>()
        }
    })
    .await
    .unwrap_or_default();
    Json(json!({"object": "list", "data": listed}))
}

pub async fn status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let gpu = tokio::task::spawn_blocking(crate::gpu::smi_total_bytes).await.ok().flatten().is_some();
    let reading_on = state.manager.loaded_transcribers().first().map(|(_, s)| s.info.device.clone());
    let speaking_on = state.manager.loaded_speakers().first().map(|(_, s)| s.info.device.clone());
    Json(json!({
        "device": if gpu { "GPU" } else { "CPU" },
        "reading_on": reading_on,
        "speaking_on": speaking_on,
    }))
}

pub const INPUT_MAX: usize = 20_000;

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechRequest {
    pub input: String,
    pub voice: String,
    pub model: String,
    pub stream: bool,
    pub seed: Option<u64>,
}

pub fn parse_speech(body: &Value) -> Result<SpeechRequest, ApiError> {
    let text = |v: &Value, name: &'static str| -> Result<String, ApiError> {
        match v {
            Value::Null => Ok(String::new()),
            Value::String(s) => Ok(s.clone()),
            _ => Err(ApiError::bad_request(format!("{name} must be a string")).with_code("invalid_type", name)),
        }
    };
    let input = text(&body["input"], "input")?.trim().to_string();
    if input.is_empty() {
        return Err(ApiError::bad_request("input is empty: there is nothing to say").with_code("invalid_value", "input"));
    }
    let chars = input.chars().count();
    if chars > INPUT_MAX {
        return Err(ApiError::bad_request(format!("input is {chars} characters, too long to say in one go: {INPUT_MAX} at most")).with_code("invalid_value", "input"));
    }
    let seed = match &body["seed"] {
        Value::Null => None,
        v => Some(v.as_u64().ok_or_else(|| ApiError::bad_request("seed must be a non-negative integer").with_code("invalid_value", "seed"))?),
    };
    Ok(SpeechRequest {
        input,
        voice: text(&body["voice"], "voice").unwrap_or_default().trim().to_string(),
        model: text(&body["model"], "model")?.trim().to_string(),
        stream: text(&body["response_format"], "response_format")?.trim().eq_ignore_ascii_case("pcm"),
        seed,
    })
}

pub fn pick_speaker(state: &AppState, asked: &str) -> Result<String, ApiError> {
    let ids = state.manager.speech_ids();
    if !asked.is_empty()
        && let Ok(cfg) = state.manager.resolve(asked)
        && ids.contains(&cfg.id)
    {
        return Ok(cfg.id);
    }
    let loaded = state.manager.loaded_speakers();
    loaded
        .first()
        .map(|(id, _)| id.clone())
        .or_else(|| ids.first().cloned())
        .ok_or_else(|| ApiError::not_found("no text to speech model is configured; add a text to speech model folder to the server config").with_code("model_not_found", "model"))
}

type Frames = tokio::sync::mpsc::Receiver<Result<Vec<f32>, String>>;

fn pcm_body(first: Vec<f32>, rest: Frames) -> axum::body::Body {
    use futures_util::StreamExt;
    let head = futures_util::stream::once(async move { Ok::<_, std::io::Error>(axum::body::Bytes::from(wav::f32_to_pcm16(&first))) });
    let tail = futures_util::stream::unfold(rest, |mut rx| async move {
        match rx.recv().await {
            Some(Ok(pcm)) => Some((Ok::<_, std::io::Error>(axum::body::Bytes::from(wav::f32_to_pcm16(&pcm))), rx)),
            Some(Err(e)) => {
                tracing::warn!("speech stopped part way: {e}");
                None
            }
            None => None,
        }
    });
    axum::body::Body::from_stream(head.chain(tail).fuse())
}

fn random_seed() -> u64 {
    uuid::Uuid::new_v4().as_u64_pair().0
}

pub async fn speech(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    use inference_bench_core::specrt_speech::tts::service::{Emit, Job};
    let req = parse_speech(&body)?;
    let id = pick_speaker(&state, &req.model)?;
    tracing::Span::current().record("model", id.as_str());
    let ticket = crate::images::admit(&state)?;
    let permit = state.images.permit().await.ok_or_else(|| ApiError::unavailable("the media queue closed"))?;
    let (id, service, held) = state.manager.speaker(&id).await?;
    let picked = tokio::task::spawn_blocking({
        let (service, asked) = (service.clone(), req.voice.clone());
        move || service.pick(&asked)
    })
    .await
    .map_err(|e| ApiError::internal(format!("voice lookup failed: {e}")))?
    .map_err(|e| ApiError::internal(format!("{e:#}")))?;
    let voice_name = picked.as_ref().map(|(v, _)| v.name.clone());
    if picked.is_none() {
        tracing::warn!(model = %id, "the model has no voice catalogue, speaking with no voice prompt");
    }
    let job = Job { text: req.input.clone(), voice: picked.map(|(_, p)| p), seed: req.seed.unwrap_or_else(random_seed) };
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Vec<f32>, String>>(32);
    let frames = tx.clone();
    let emit: Emit = Box::new(move |f| frames.blocking_send(Ok(f.to_vec())).is_ok());
    let rate = service.info.sample_rate;
    let done = service.start(job, emit).map_err(|e| ApiError::internal(format!("{e:#}")))?;
    let started = std::time::Instant::now();
    let span = tracing::Span::current();
    let (metrics, model) = (state.metrics.clone(), id.clone());
    tokio::task::spawn_blocking(move || {
        let out = done.recv().unwrap_or_else(|_| Err(anyhow::anyhow!("the speech engine has stopped")));
        span.in_scope(|| match out {
            Ok(spoken) => {
                let (audio, secs) = (spoken.samples as f64 / f64::from(rate), started.elapsed().as_secs_f64());
                metrics.speech(&model, audio, secs);
                tracing::info!(model = %model, voice = ?voice_name, audio_secs = audio, secs, frames = spoken.frames, chunks = spoken.chunks, stopped = spoken.stopped, "spoke");
            }
            Err(e) => {
                let _ = tx.blocking_send(Err(format!("{e:#}")));
            }
        });
        drop((held, permit, ticket));
    });
    if req.stream {
        let first = match rx.recv().await {
            Some(Ok(f)) => f,
            Some(Err(e)) => return Err(ApiError::internal(e)),
            None => Vec::new(),
        };
        let headers = [
            (header::CONTENT_TYPE, "audio/L16".to_string()),
            (header::HeaderName::from_static("x-sample-rate"), rate.to_string()),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ];
        return Ok((headers, pcm_body(first, rx)).into_response());
    }
    let mut pcm = Vec::new();
    while let Some(f) = rx.recv().await {
        pcm.extend(f.map_err(ApiError::internal)?);
    }
    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav::wav_bytes(&pcm, rate)).into_response())
}

fn speaker_dir(state: &AppState, id: &str) -> Option<std::path::PathBuf> {
    let cfg = state.config.model(id)?;
    inference_bench_core::specrt_speech::tts::service::resolve_dir(&state.config.specs.resolve(cfg.upstream_name()))
}

pub async fn voices(State(state): State<Arc<AppState>>) -> Json<Value> {
    use inference_bench_core::specrt_speech::tts::voices::{Catalogue, HUB};
    let listed = tokio::task::spawn_blocking(move || {
        if let Some((_, s)) = state.manager.loaded_speakers().first() {
            return s.voices();
        }
        for id in state.manager.speech_ids() {
            let Some(mut c) = speaker_dir(&state, &id).and_then(|d| Catalogue::open(&d).ok()) else { continue };
            if c.voices.is_empty()
                && let Err(e) = c.fetch_catalogue(HUB)
            {
                tracing::warn!("could not fetch the voice catalogue: {e:#}");
            }
            return c.voices.iter().map(|v| (v.clone(), c.installed(v))).collect();
        }
        Vec::new()
    })
    .await
    .unwrap_or_default();
    let list: Vec<Value> = listed.into_iter().map(|(v, installed)| json!({"id": v.name, "label": v.label(), "installed": installed})).collect();
    Json(json!({"voices": list}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(file: Option<Vec<u8>>) -> Form {
        Form { file, ..Form::default() }
    }

    #[test]
    fn a_stereo_float_wav_at_any_rate_reads_as_mono_with_its_rate() {
        let mut f32_wav = wav::wav_bytes(&[0.0; 4], 44_100);
        f32_wav[20..22].copy_from_slice(&3u16.to_le_bytes());
        f32_wav[22..24].copy_from_slice(&2u16.to_le_bytes());
        f32_wav[34..36].copy_from_slice(&32u16.to_le_bytes());
        let data: Vec<u8> = [0.5f32, -0.5, 0.25, 0.75].iter().flat_map(|v| v.to_le_bytes()).collect();
        f32_wav.truncate(40);
        f32_wav.extend((data.len() as u32).to_le_bytes());
        f32_wav.extend(data);
        let p = parse(form(Some(f32_wav)), 1 << 20).unwrap();
        assert_eq!((p.pcm, p.rate, p.format), (vec![0.0, 0.5], 44_100, Format::Json));
    }

    #[test]
    fn a_bad_form_is_refused_with_the_field_named() {
        let ok = wav::wav_bytes(&[0.0; 16], 16_000);
        for (f, status, word) in [
            (form(None), 400, "no file"),
            (form(Some(b"not audio".to_vec())), 400, "WAV"),
            (form(Some(ok.clone())), 413, "max_audio_bytes"),
            (Form { response_format: "srt".into(), ..form(Some(ok.clone())) }, 400, "response_format"),
            (Form { temperature: "hot".into(), ..form(Some(ok.clone())) }, 400, "temperature"),
        ] {
            let limit = if status == 413 { 10 } else { 1 << 20 };
            let e = parse(f, limit).err().expect(word);
            assert_eq!(e.status.as_u16(), status, "{word}: {}", e.message);
            assert!(e.message.contains(word), "{word}: {}", e.message);
        }
        let p = parse(Form { language: " EN ".into(), prompt: "hi".into(), temperature: "0.2".into(), response_format: "verbose_json".into(), ..form(Some(ok)) }, 1 << 20).unwrap();
        assert_eq!((p.request.language.as_deref(), p.request.prompt.as_str(), p.request.temperature, p.format), (Some("en"), "hi", Some(0.2), Format::VerboseJson));
    }
}
