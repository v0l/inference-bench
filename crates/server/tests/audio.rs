mod common;

use common::*;
use inference_bench_core::result::{BackendError, Generation};
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_media::ModelKind;
use inference_bench_core::specrt_speech::service::{Heard, Request as Heard_request, Segment, Service, ServiceInfo, Transcribe};
use inference_bench_core::target::{Capabilities, Request, Target};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Seen {
    calls: Mutex<Vec<(usize, Heard_request)>>,
    hold: Mutex<bool>,
    loads: AtomicUsize,
}

struct Echo {
    seen: Arc<Seen>,
}

impl Transcribe for Echo {
    fn info(&self) -> ServiceInfo {
        ServiceInfo { name: "echo".into(), sample_rate: 16_000, window_seconds: 30.0, device: "CPU".into(), multilingual: true }
    }

    fn transcribe(&mut self, pcm: &[f32], request: &Heard_request) -> anyhow::Result<Heard> {
        self.seen.calls.lock().unwrap().push((pcm.len(), request.clone()));
        while *self.seen.hold.lock().unwrap() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let text = format!("heard {} samples", pcm.len());
        let duration = pcm.len() as f64 / 16_000.0;
        let segment = Segment { id: 0, seek: 0, start: 0.0, end: duration, text: text.clone(), tokens: vec![1, 2], temperature: 0.0, avg_logprob: -0.1, compression_ratio: 1.0, no_speech_prob: 0.01 };
        Ok(Heard { text, language: request.language.clone().or(Some("en".into())), duration, segments: vec![segment], tokens: vec![1, 2] })
    }
}

struct FakeSpeech {
    service: Option<Arc<Service>>,
    seen: Arc<Seen>,
}

impl Target for FakeSpeech {
    fn name(&self) -> &str {
        "fake-speech"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { kind: ModelKind::Transcription, ..Capabilities::default() }
    }

    fn load(&mut self, _model: &str) -> Result<(), BackendError> {
        self.seen.loads.fetch_add(1, Ordering::SeqCst);
        let seen = self.seen.clone();
        self.service = Some(Service::spawn(move || Ok(Box::new(Echo { seen }) as Box<dyn Transcribe>)).map_err(|e| BackendError::failed(e.to_string()))?);
        Ok(())
    }

    fn generate(&mut self, _: &Request, _: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        Err(BackendError::failed("speech only"))
    }

    fn transcriber(&self) -> Option<Arc<Service>> {
        self.service.clone()
    }
}

fn rig(extra: &str) -> (Rig, Arc<Seen>) {
    let seen = Arc::new(Seen::default());
    let s = seen.clone();
    let factory = move |_: Option<usize>| -> Result<Box<dyn Target>, BackendError> { Ok(Box::new(FakeSpeech { service: None, seen: s.clone() })) };
    let rig = start_with(config(&format!("{extra}\n[[models]]\nid='whisper-base.en'\npreload = true\n")), factory, Default::default());
    wait_for("the speech model to preload", || seen.loads.load(Ordering::SeqCst) == 1 && rig.get("/health", None).1.contains("whisper-base.en"));
    (rig, seen)
}

fn reqwest_form(boundary: &str, wav: &[u8], model: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend(wav);
    body.extend(format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n").as_bytes());
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n").as_bytes());
    body.extend(format!("--{boundary}--\r\n").as_bytes());
    body
}

fn agent_voice_form(wav: &[u8], model: &str) -> Vec<u8> {
    let boundary = "----agent-voice-speech";
    let mut body = Vec::new();
    let mut field = |name: &str, value: &str| body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    let model = if model.trim().is_empty() { "whisper-1" } else { model.trim() };
    field("model", model);
    field("response_format", "json");
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend(wav);
    body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

fn form(fields: &[(&str, &str)], wav: &[u8]) -> (String, Vec<u8>) {
    let boundary = "x-test-boundary";
    let mut body = Vec::new();
    for (k, v) in fields {
        body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend(wav);
    body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn post(rig: &Rig, content_type: &str, body: Vec<u8>, key: Option<&str>, request_id: Option<&str>) -> (u16, String, Vec<(String, String)>) {
    let mut req = agent().post(format!("{}/v1/audio/transcriptions", rig.url)).header("Content-Type", content_type);
    if let Some(k) = key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    if let Some(id) = request_id {
        req = req.header("x-request-id", id);
    }
    let mut resp = req.send(&body[..]).unwrap();
    let headers = resp.headers().iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string())).collect();
    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap(), headers)
}

fn json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("{e}: {s}"))
}

#[test]
fn the_desktop_app_and_agent_voice_bodies_come_back_as_text_behind_the_key_and_log_like_chat() {
    let log = std::env::temp_dir().join(format!("ib-audio-access-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let (rig, seen) = rig(&format!("api_keys = [\"sk-a\"]\naccess_log = {:?}\n", log.display().to_string()));
    let take = wav::wav_bytes(&vec![0.1; 16_000], 16_000);
    let boundary = "a1b2c3d4e5f6a7b8-c9d0e1f2a3b4c5d6";
    let reqwest_type = format!("multipart/form-data; boundary={boundary}");
    let (code, _, _) = post(&rig, &reqwest_type, reqwest_form(boundary, &take, "whisper-1"), None, None);
    assert_eq!(code, 401, "the same key as chat");
    let (code, body, headers) = post(&rig, &reqwest_type, reqwest_form(boundary, &take, "whisper-1"), Some("sk-a"), Some("rid-native-1"));
    assert_eq!(code, 200, "{body}");
    assert_eq!(json(&body), serde_json::json!({"text": "heard 16000 samples"}));
    assert_eq!(header(&headers, "x-request-id"), Some("rid-native-1"));
    let (code, body, headers) = post(&rig, "multipart/form-data; boundary=----agent-voice-speech", agent_voice_form(&take, ""), Some("sk-a"), None);
    assert_eq!(code, 200, "{body}");
    assert_eq!(json(&body)["text"], "heard 16000 samples");
    assert!(header(&headers, "x-request-id").is_some_and(|v| v.starts_with("req-")));
    let calls = seen.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|(n, r)| *n == 16_000 && *r == Heard_request::default()), "{calls:?}");
    let m = rig.metrics();
    assert!(m.contains("inference_requests_total{endpoint=\"audio_transcriptions\",status=\"200\"} 2"), "{m}");
    assert!(m.contains("inference_requests_total{endpoint=\"audio_transcriptions\",status=\"401\"} 1"), "{m}");
    assert!(m.contains("inference_transcriptions_total{model=\"whisper-base.en\"} 2"), "{m}");
    let lines: Vec<Value> = std::fs::read_to_string(&log).unwrap().lines().map(json).collect();
    assert!(lines.iter().any(|l| l["request_id"] == "rid-native-1" && l["path"] == "/v1/audio/transcriptions" && l["status"] == 200), "{lines:?}");
    assert!(lines_for("rid-native-1").iter().any(|l| l["fields"]["message"] == "transcribed"));
    rig.stop();
}

#[test]
fn every_openai_field_is_read_unknown_ones_are_ignored_and_audio_is_resampled_and_downmixed() {
    let (rig, seen) = rig("");
    let stereo: Vec<f32> = (0..48_000 * 2).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let mut take = wav::wav_bytes(&stereo, 48_000);
    take[22..24].copy_from_slice(&2u16.to_le_bytes());
    take[28..32].copy_from_slice(&(48_000u32 * 4).to_le_bytes());
    take[32..34].copy_from_slice(&4u16.to_le_bytes());
    let (ct, body) = form(&[("model", "some-unknown-id"), ("language", "de"), ("prompt", "Vorher."), ("temperature", "0.2"), ("timestamp_granularities[]", "word"), ("stream", "false")], &take);
    let (code, body, _) = post(&rig, &ct, body, None, None);
    assert_eq!(code, 200, "{body}");
    let (n, r) = seen.calls.lock().unwrap().last().cloned().unwrap();
    assert_eq!(n, 16_000, "one second at 48 kHz stereo is 16000 mono samples at the model's rate");
    assert_eq!(r, Heard_request { language: Some("de".into()), prompt: "Vorher.".into(), temperature: Some(0.2) });
    let (ct, body) = form(&[("response_format", "text")], &wav::wav_bytes(&[0.0; 8000], 8000));
    let (code, text, headers) = post(&rig, &ct, body, None, None);
    assert_eq!((code, text.as_str()), (200, "heard 16000 samples"));
    assert!(header(&headers, "content-type").unwrap().starts_with("text/plain"));
    let (ct, body) = form(&[("response_format", "verbose_json")], &wav::wav_bytes(&[0.0; 16000], 16000));
    let (code, body, _) = post(&rig, &ct, body, None, None);
    let v = json(&body);
    assert_eq!(code, 200, "{body}");
    assert_eq!((v["text"].as_str(), v["language"].as_str(), v["duration"].as_f64()), (Some("heard 16000 samples"), Some("en"), Some(1.0)));
    assert_eq!(v["segments"][0]["tokens"], serde_json::json!([1, 2]));
    for (fields, word) in [(vec![("response_format", "srt")], "response_format"), (vec![("temperature", "x")], "temperature")] {
        let (ct, body) = form(&fields, &wav::wav_bytes(&[0.0; 160], 16000));
        let (code, body, _) = post(&rig, &ct, body, None, None);
        assert_eq!(code, 400, "{body}");
        assert!(body.contains(word), "{body}");
    }
    let (ct, body) = form(&[], &wav::wav_bytes(&vec![0.0; 1_600_000], 16_000));
    assert_eq!(post(&rig, &ct, body, None, None).0, 200, "a 3.2 MB take fits the 64 MB default");
    let (ct, body) = form(&[], b"RIFF....WAVEjunk");
    assert_eq!(post(&rig, &ct, body, None, None).0, 400);
    assert_eq!(seen.loads.load(Ordering::SeqCst), 1, "one model, loaded once");
    rig.stop();
}

#[test]
fn a_file_over_max_audio_bytes_is_refused_and_a_full_media_queue_answers_429_with_retry_after() {
    let (rig, seen) = rig("retry_after_secs = 3\n[media]\nmax_audio_bytes = 40000\n[images]\njobs_per_gpu = 1\nqueue_depth = 0\n");
    let (ct, body) = form(&[], &wav::wav_bytes(&vec![0.0; 30_000], 16_000));
    let (code, body, _) = post(&rig, &ct, body, None, None);
    assert_eq!(code, 413, "{body}");
    assert!(body.contains("max_audio_bytes"), "{body}");
    *seen.hold.lock().unwrap() = true;
    let url = rig.url.clone();
    let (ct, small) = form(&[], &wav::wav_bytes(&vec![0.0; 1600], 16_000));
    let first = {
        let (ct, small) = (ct.clone(), small.clone());
        std::thread::spawn(move || agent().post(format!("{url}/v1/audio/transcriptions")).header("Content-Type", &ct).send(&small[..]).unwrap().status().as_u16())
    };
    wait_for("the first job to run", || !seen.calls.lock().unwrap().is_empty());
    let (code, body, headers) = post(&rig, &ct, small.clone(), None, None);
    assert_eq!(code, 429, "{body}");
    assert_eq!(header(&headers, "retry-after"), Some("3"));
    *seen.hold.lock().unwrap() = false;
    assert_eq!(first.join().unwrap(), 200);
    assert_eq!(post(&rig, &ct, small, None, None).0, 200, "the slot is free again");
    rig.stop();
}

#[test]
fn audio_models_status_v1_models_and_capabilities_describe_the_transcription_model() {
    let (rig, _) = rig("api_keys = [\"sk-a\"]\n");
    assert_eq!(rig.get("/v1/audio/models", None).0, 401);
    assert_eq!(rig.get("/v1/audio/status", None).0, 401);
    let (code, body, _) = rig.get("/v1/audio/status", Some("sk-a"));
    let v = json(&body);
    assert_eq!(code, 200);
    assert!(matches!(v["device"].as_str(), Some("GPU" | "CPU")), "{v}");
    assert_eq!((v["reading_on"].as_str(), v["speaking_on"].as_str()), (Some("CPU"), None), "the loaded speech model names its device, nothing speaks: {v}");
    let (ct, b) = form(&[], &wav::wav_bytes(&[0.0; 160], 16_000));
    assert_eq!(post(&rig, &ct, b, Some("sk-a"), None).0, 200);
    let v = json(&rig.get("/v1/audio/models", Some("sk-a")).1);
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    let m = &data[0];
    assert_eq!((m["id"].as_str(), m["object"].as_str(), m["label"].as_str(), m["installed"].as_bool(), m["default"].as_bool()), (Some("whisper-base.en"), Some("model"), Some("whisper-base.en"), Some(true), Some(true)));
    assert!(m["size_mb"].is_u64(), "{m}");
    let models = json(&rig.get("/v1/models", Some("sk-a")).1);
    assert_eq!(models["data"][0]["type"], "transcription");
    let caps = json(&rig.get("/v1/capabilities", Some("sk-a")).1);
    assert_eq!(caps["limits"]["max_audio_bytes"], 64 << 20);
    let route = caps["routes"].as_array().unwrap().iter().find(|r| r["path"] == "/v1/audio/transcriptions").unwrap().clone();
    let names: Vec<&str> = route["fields"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
    for f in ["file", "model", "response_format", "language", "prompt", "temperature"] {
        assert!(names.contains(&f), "{f} in {names:?}");
    }
    assert!(caps["routes"].as_array().unwrap().iter().any(|r| r["path"] == "/v1/audio/models"));
    let chat = rig.post("/v1/chat/completions", Some("sk-a"), serde_json::json!({"model": "whisper-base.en", "messages": [{"role": "user", "content": "hi"}]}));
    assert_ne!(chat.0, 200, "a speech model does not chat");
    rig.stop();
}

fn tiny_whisper() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/speech/tiny-whisper")
}

fn f32_file(name: &str) -> Vec<f32> {
    std::fs::read(tiny_whisper().join("reference").join(name)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
fn the_tiny_whisper_fixture_served_end_to_end_on_the_host_engine_gives_the_greedy_tokens() {
    let factory = |max_ctx: Option<usize>| -> Result<Box<dyn Target>, BackendError> {
        let host: inference_bench_core::native::SpeechLoader = Arc::new(|dir: &Path, _| Service::load_on_host(dir));
        Ok(Box::new(inference_bench_core::native::Native::new(max_ctx).with_speech_loader(host)))
    };
    let rig = start_with(config(&format!("[[models]]\nid = 'tiny-whisper'\nmodel = {:?}\n", tiny_whisper().display().to_string())), factory, Default::default());
    let meta = json(&std::fs::read_to_string(tiny_whisper().join("reference.json")).unwrap());
    let greedy: Vec<u64> = meta["greedy"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap()).collect();
    let pcm = f32_file("audio.full.bin");
    let float_wav = {
        let mut w = wav::wav_bytes(&vec![0.0; pcm.len() * 2], 16_000);
        w[20..22].copy_from_slice(&3u16.to_le_bytes());
        w[28..32].copy_from_slice(&(16_000u32 * 4).to_le_bytes());
        w[32..34].copy_from_slice(&4u16.to_le_bytes());
        w[34..36].copy_from_slice(&32u16.to_le_bytes());
        w.truncate(44);
        w.extend(pcm.iter().flat_map(|v| v.to_le_bytes()));
        w
    };
    let (ct, body) = form(&[("model", "whisper-1"), ("language", "en"), ("response_format", "verbose_json")], &float_wav);
    let (code, body, _) = post(&rig, &ct, body, None, None);
    assert_eq!(code, 200, "{body}");
    let v = json(&body);
    let tokens: Vec<u64> = v["segments"].as_array().unwrap().iter().flat_map(|s| s["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap())).collect();
    assert!(tokens.len() >= greedy.len() && tokens[..greedy.len()] == greedy[..], "served {tokens:?}, fixture greedy {greedy:?}");
    assert_eq!((v["language"].as_str(), v["duration"].as_f64()), (Some("en"), Some(1.0)));
    let models = json(&rig.get("/v1/models", None).1);
    assert_eq!((models["data"][0]["type"].as_str(), models["data"][0]["loaded"].as_bool()), (Some("transcription"), Some(true)));
    assert_eq!(json(&rig.get("/v1/audio/status", None).1)["reading_on"], "CPU");
    rig.stop();
}
