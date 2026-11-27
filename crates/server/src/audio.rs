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
    Json(json!({
        "device": if gpu { "GPU" } else { "CPU" },
        "reading_on": reading_on,
        "speaking_on": Value::Null,
    }))
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
