mod common;

use common::*;
use inference_bench_core::result::{BackendError, Generation};
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_media::ModelKind;
use inference_bench_core::specrt_speech::tts::service::{Job, Service, Speak, SpeakerInfo, Spoken};
use inference_bench_core::specrt_speech::tts::voices::Catalogue;
use inference_bench_core::specrt_spec::TtsVoices;
use inference_bench_core::target::{Capabilities, Request, Target};
use serde_json::{Value, json};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const FRAME: usize = 1920;

#[derive(Default)]
struct Seen {
    jobs: Mutex<Vec<Job>>,
    hold_after_first: AtomicBool,
    released: AtomicBool,
    emitted: AtomicUsize,
    finished: Mutex<Vec<Spoken>>,
    loads: AtomicUsize,
    frames: AtomicUsize,
    frame_ms: AtomicUsize,
}

fn fake_samples(seed: u64, frame: usize) -> Vec<f32> {
    (0..FRAME).map(|i| (((seed as usize + frame * 31 + i * 7) % 2001) as f32 / 1000.0 - 1.0) * 0.9).collect()
}

struct Talker {
    seen: Arc<Seen>,
    catalogue: Catalogue,
}

impl Speak for Talker {
    fn info(&self) -> SpeakerInfo {
        SpeakerInfo { name: "fake".into(), sample_rate: 24_000, frame_samples: FRAME, device: "CPU".into() }
    }

    fn catalogue(&self) -> Catalogue {
        self.catalogue.clone()
    }

    fn speak(&mut self, job: &Job, emit: &mut dyn FnMut(&[f32]) -> bool) -> anyhow::Result<Spoken> {
        self.seen.jobs.lock().unwrap().push(job.clone());
        let frames = self.seen.frames.load(Ordering::SeqCst).max(1);
        let mut out = Spoken { chunks: 1, ..Spoken::default() };
        for f in 0..frames {
            if f == 1 && self.seen.hold_after_first.load(Ordering::SeqCst) {
                while !self.seen.released.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
            let ms = self.seen.frame_ms.load(Ordering::SeqCst);
            if ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(ms as u64));
            }
            let pcm = fake_samples(job.seed, f);
            out.frames += 1;
            out.samples += pcm.len();
            self.seen.emitted.fetch_add(1, Ordering::SeqCst);
            if !emit(&pcm) {
                out.stopped = true;
                break;
            }
        }
        self.seen.finished.lock().unwrap().push(out.clone());
        Ok(out)
    }
}

struct FakeVoice {
    service: Option<Arc<Service>>,
    seen: Arc<Seen>,
    dir: PathBuf,
}

impl Target for FakeVoice {
    fn name(&self) -> &str {
        "fake-voice"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { kind: ModelKind::Speech, streaming: true, ..Capabilities::default() }
    }

    fn load(&mut self, _model: &str) -> Result<(), BackendError> {
        self.seen.loads.fetch_add(1, Ordering::SeqCst);
        let seen = self.seen.clone();
        let spec = TtsVoices { repo: String::new(), catalogue: "voices.jsonl".into(), default: "Hadley".into() };
        let catalogue = Catalogue::find(&self.dir, &spec, "en");
        self.service = Some(Service::spawn(move || Ok(Box::new(Talker { seen, catalogue }) as Box<dyn Speak>)).map_err(|e| BackendError::failed(e.to_string()))?);
        Ok(())
    }

    fn generate(&mut self, _: &Request, _: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        Err(BackendError::failed("speech only"))
    }

    fn speaker(&self) -> Option<Arc<Service>> {
        self.service.clone()
    }
}

fn voice_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ib-speech-voices-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("voices")).unwrap();
    std::fs::write(
        dir.join("voices.jsonl"),
        concat!(
            "{\"filename\": \"voices/z.safetensors\", \"name\": \"Zoey\", \"voice_id\": \"z\", \"language\": \"en\", \"locale\": \"en-us\", \"gender\": \"female\"}\n",
            "{\"filename\": \"voices/h.safetensors\", \"name\": \"Hadley\", \"voice_id\": \"h\", \"language\": \"en\", \"locale\": \"en-us\", \"gender\": \"female\"}\n",
            "{\"filename\": \"voices/a.safetensors\", \"name\": \"Alfie\", \"voice_id\": \"a\", \"language\": \"en\", \"locale\": \"en-gb\", \"gender\": \"male\"}\n",
            "{\"filename\": \"voices/l.safetensors\", \"name\": \"Lucie\", \"voice_id\": \"l\", \"language\": \"fr\", \"locale\": \"fr-fr\"}\n",
        ),
    )
    .unwrap();
    for f in ["z", "h"] {
        std::fs::write(dir.join(format!("voices/{f}.safetensors")), b"voice").unwrap();
    }
    dir
}

fn rig(extra: &str) -> (Rig, Arc<Seen>) {
    let seen = Arc::new(Seen::default());
    seen.frames.store(5, Ordering::SeqCst);
    let (s, dir) = (seen.clone(), voice_dir());
    let factory = move |_: Option<usize>| -> Result<Box<dyn Target>, BackendError> { Ok(Box::new(FakeVoice { service: None, seen: s.clone(), dir: dir.clone() })) };
    let rig = start_with(config(&format!("{extra}\n[[models]]\nid='phonon-31m-en'\npreload = true\n")), factory, Default::default());
    wait_for("the speech model to preload", || seen.loads.load(Ordering::SeqCst) == 1 && rig.get("/health", None).1.contains("phonon-31m-en"));
    (rig, seen)
}

fn reader_body(text: &str, voice: &str) -> Value {
    let mut body = json!({ "model": "kokoro", "input": text, "response_format": "pcm" });
    if !voice.trim().is_empty() {
        body["voice"] = voice.trim().into();
    }
    body
}

fn client_speak_body(text: &str, voice: &str) -> Value {
    let mut body = json!({ "model": "kokoro", "input": text });
    if !voice.trim().is_empty() {
        body["voice"] = voice.trim().into();
    }
    body
}

struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn send(rig: &Rig, body: &Value, token: &str, reader: bool, request_id: Option<&str>) -> Answer {
    let mut req = agent().post(format!("{}/v1/audio/speech", rig.url)).header("content-type", "application/json");
    if reader {
        req = req.header("accept-encoding", "identity");
    }
    if !token.trim().is_empty() {
        req = req.header("Authorization", &format!("Bearer {}", token.trim()));
    }
    if let Some(id) = request_id {
        req = req.header("x-request-id", id);
    }
    let mut resp = req.send(body.to_string()).unwrap();
    let headers = resp.headers().iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string())).collect();
    let status = resp.status().as_u16();
    Answer { status, headers, body: resp.body_mut().with_config().limit(256 << 20).read_to_vec().unwrap() }
}

fn reader_outcome(a: &Answer) -> Result<u32, String> {
    if (200..300).contains(&a.status) {
        return Ok(header(&a.headers, "x-sample-rate").and_then(|r| r.parse().ok()).unwrap_or(24_000));
    }
    if matches!(a.status, 404 | 405) {
        return Err("this agentbench server cannot read aloud yet: update and restart it".into());
    }
    if a.status == 401 {
        return Err("the server refused the token, so it cannot read aloud".into());
    }
    let said = serde_json::from_slice::<Value>(&a.body).ok().and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string)).unwrap_or_else(|| String::from_utf8_lossy(&a.body).trim().to_string());
    Err(format!("the server could not read that aloud: {}: {said}", a.status))
}

fn expected(seed: u64, frames: usize) -> Vec<f32> {
    (0..frames).flat_map(|f| fake_samples(seed, f)).collect()
}

fn json_of(b: &[u8]) -> Value {
    serde_json::from_slice(b).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(b)))
}

#[test]
fn the_desktop_reader_and_agent_voice_bodies_get_pcm_and_wav_behind_the_key_and_log_like_chat() {
    let log = std::env::temp_dir().join(format!("ib-speech-access-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let (rig, seen) = rig(&format!("api_keys = [\"sk-a\"]\naccess_log = {:?}\n", log.display().to_string()));
    let refused = send(&rig, &reader_body("Hello there.", "Hadley"), "", true, None);
    assert_eq!(refused.status, 401);
    assert_eq!(reader_outcome(&refused).unwrap_err(), "the server refused the token, so it cannot read aloud");
    assert_eq!(send(&rig, &reader_body("Hello there.", ""), "sk-wrong", true, None).status, 401);
    let a = send(&rig, &reader_body("Hello there.", "hadley"), "sk-a", true, Some("rid-reader-1"));
    assert_eq!(reader_outcome(&a), Ok(24_000));
    assert_eq!(header(&a.headers, "content-type"), Some("audio/L16"));
    assert_eq!(header(&a.headers, "cache-control"), Some("no-store"));
    assert_eq!(header(&a.headers, "x-request-id"), Some("rid-reader-1"));
    assert!(header(&a.headers, "content-encoding").is_none());
    let job = seen.jobs.lock().unwrap().last().cloned().unwrap();
    assert_eq!(job.text, "Hello there.");
    assert!(job.voice.as_ref().unwrap().ends_with("voices/h.safetensors"), "{job:?}");
    assert_eq!(a.body, wav::f32_to_pcm16(&expected(job.seed, 5)), "the stream is the engine's samples rounded to PCM16");
    let b = send(&rig, &client_speak_body("Hello there.", "zoey"), "sk-a", false, None);
    assert_eq!(b.status, 200, "{}", String::from_utf8_lossy(&b.body));
    assert_eq!(header(&b.headers, "content-type"), Some("audio/wav"));
    let job = seen.jobs.lock().unwrap().last().cloned().unwrap();
    assert!(job.voice.as_ref().unwrap().ends_with("voices/z.safetensors"));
    assert_eq!(b.body, wav::wav_bytes(&expected(job.seed, 5), 24_000), "one WAV of the same rounded samples");
    let (pcm, rate) = wav::read_wav(&b.body).unwrap();
    assert_eq!((pcm.len(), rate), (5 * FRAME, 24_000));
    let m = rig.metrics();
    assert!(m.contains("inference_requests_total{endpoint=\"audio_speech\",status=\"200\"} 2"), "{m}");
    assert!(m.contains("inference_requests_total{endpoint=\"audio_speech\",status=\"401\"} 2"), "{m}");
    wait_for("both jobs to be counted", || rig.metrics().contains("inference_speech_total{model=\"phonon-31m-en\"} 2"));
    let lines: Vec<Value> = std::fs::read_to_string(&log).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(lines.iter().any(|l| l["request_id"] == "rid-reader-1" && l["path"] == "/v1/audio/speech" && l["status"] == 200), "{lines:?}");
    wait_for("the spoke line", || lines_for("rid-reader-1").iter().any(|l| l["fields"]["message"] == "spoke"));
    rig.stop();
}

#[test]
fn voices_are_case_insensitive_unknown_ones_get_hadley_and_every_field_is_accepted() {
    let (rig, seen) = rig("");
    for (asked, file) in [("alloy", "h"), ("ZOEY", "z"), ("", "h"), ("af_bella", "h"), ("Lucie", "h")] {
        let mut body = client_speak_body("Hi.", asked);
        body["model"] = json!("tts-1");
        body["speed"] = json!(1.5);
        body["response_format"] = json!("mp3");
        body["instructions"] = json!("cheerful");
        let a = send(&rig, &body, "", false, None);
        assert_eq!((a.status, header(&a.headers, "content-type")), (200, Some("audio/wav")), "{asked}");
        let voice = seen.jobs.lock().unwrap().last().unwrap().voice.clone().unwrap();
        assert!(voice.ends_with(format!("voices/{file}.safetensors")), "{asked}: {voice:?}");
    }
    let mut seeded = client_speak_body("Same again.", "");
    seeded["seed"] = json!(42);
    seeded["model"] = json!("some-unknown-model");
    let (x, y) = (send(&rig, &seeded, "", false, None), send(&rig, &seeded, "", false, None));
    assert_eq!((x.status, y.status), (200, 200));
    assert_eq!(x.body, y.body, "a seed gives the same samples");
    assert_eq!(seen.jobs.lock().unwrap().last().unwrap().seed, 42);
    let mut pcm = reader_body("Upper case format.", "");
    pcm["response_format"] = json!("PCM");
    assert_eq!(header(&send(&rig, &pcm, "", true, None).headers, "content-type"), Some("audio/L16"));
    for (body, word) in [
        (json!({"model": "kokoro", "input": "   "}), "empty"),
        (json!({"model": "kokoro"}), "empty"),
        (json!({"model": "kokoro", "input": "x".repeat(20_001)}), "20000"),
        (json!({"model": "kokoro", "input": 5}), "input"),
        (json!({"model": "kokoro", "input": "hi", "seed": -1}), "seed"),
    ] {
        let a = send(&rig, &body, "", false, None);
        assert_eq!(a.status, 400, "{word}");
        assert!(String::from_utf8_lossy(&a.body).contains(word), "{word}: {}", String::from_utf8_lossy(&a.body));
        let r = send(&rig, &json!({"model": "kokoro", "input": body["input"], "response_format": "pcm"}), "", true, None);
        if word != "seed" {
            assert!(reader_outcome(&r).unwrap_err().contains("400"), "{word}");
        }
    }
    assert_eq!(send(&rig, &json!({"model": "kokoro", "input": "é".repeat(20_000)}), "", false, None).status, 200, "the limit counts characters");
    rig.stop();
}

#[test]
fn the_first_frame_arrives_before_the_sentence_is_done_and_a_dropped_listener_stops_the_work() {
    let (rig, seen) = rig("");
    seen.frames.store(4, Ordering::SeqCst);
    seen.hold_after_first.store(true, Ordering::SeqCst);
    let resp = agent().post(format!("{}/v1/audio/speech", rig.url)).header("content-type", "application/json").header("accept-encoding", "identity").send(reader_body("A long sentence.", "").to_string()).unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let mut body = resp.into_body().into_reader();
    let mut first = vec![0u8; 2 * FRAME];
    body.read_exact(&mut first).unwrap();
    assert_eq!(seen.emitted.load(Ordering::SeqCst), 1, "one frame made and already heard, the rest held back");
    let seed = seen.jobs.lock().unwrap().last().unwrap().seed;
    assert_eq!(first, wav::f32_to_pcm16(&fake_samples(seed, 0)));
    seen.released.store(true, Ordering::SeqCst);
    let mut rest = Vec::new();
    body.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, wav::f32_to_pcm16(&expected(seed, 4)[FRAME..]));
    seen.hold_after_first.store(false, Ordering::SeqCst);
    seen.emitted.store(0, Ordering::SeqCst);
    seen.frames.store(5000, Ordering::SeqCst);
    seen.frame_ms.store(1, Ordering::SeqCst);
    let before = seen.finished.lock().unwrap().len();
    {
        let resp = agent().post(format!("{}/v1/audio/speech", rig.url)).header("content-type", "application/json").send(reader_body("Stop me.", "").to_string()).unwrap();
        let mut body = resp.into_body().into_reader();
        let mut one = vec![0u8; 2 * FRAME];
        body.read_exact(&mut one).unwrap();
    }
    wait_for("the engine to notice the listener left", || seen.finished.lock().unwrap().len() > before);
    let last = seen.finished.lock().unwrap().last().cloned().unwrap();
    assert!(last.stopped && last.frames < 5000, "{last:?}");
    println!("a dropped listener stopped the work after {} of 5000 frames", last.frames);
    seen.frames.store(2, Ordering::SeqCst);
    seen.frame_ms.store(0, Ordering::SeqCst);
    assert_eq!(send(&rig, &client_speak_body("Still here.", ""), "", false, None).status, 200, "the voice is free again");
    rig.stop();
}

#[test]
fn a_server_with_no_speech_model_answers_404_which_the_reader_reads_as_not_yet() {
    let rig = start(config("[[models]]\nid='chat-only'\n"), Shape::default());
    let a = send(&rig, &reader_body("Hello.", ""), "", true, None);
    assert_eq!(a.status, 404, "{}", String::from_utf8_lossy(&a.body));
    assert_eq!(reader_outcome(&a).unwrap_err(), "this agentbench server cannot read aloud yet: update and restart it");
    let b = send(&rig, &client_speak_body("Hello.", ""), "", false, None);
    assert_eq!(b.status, 404);
    assert!(json_of(&b.body)["error"]["message"].as_str().unwrap().contains("text to speech"));
    let v = json_of(&rig.get("/v1/audio/voices", None).1.into_bytes());
    assert_eq!(v, json!({"voices": []}));
    rig.stop();
}

#[test]
fn voices_models_status_and_capabilities_describe_the_speech_model() {
    let (rig, _) = rig("api_keys = [\"sk-a\"]\n");
    assert_eq!(rig.get("/v1/audio/voices", None).0, 401);
    let v = json_of(&rig.get("/v1/audio/voices", Some("sk-a")).1.into_bytes());
    assert_eq!(
        v,
        json!({"voices": [
            {"id": "Zoey", "label": "Zoey, American", "installed": true},
            {"id": "Hadley", "label": "Hadley, American", "installed": true},
            {"id": "Alfie", "label": "Alfie, British man", "installed": false},
        ]})
    );
    let models = json_of(&rig.get("/v1/models", Some("sk-a")).1.into_bytes());
    assert_eq!(models["data"][0]["type"], "speech");
    let status = json_of(&rig.get("/v1/audio/status", Some("sk-a")).1.into_bytes());
    assert_eq!(status["speaking_on"], "CPU");
    let caps = json_of(&rig.get("/v1/capabilities", Some("sk-a")).1.into_bytes());
    let routes = caps["routes"].as_array().unwrap();
    let route = routes.iter().find(|r| r["path"] == "/v1/audio/speech").unwrap();
    let names: Vec<&str> = route["fields"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
    for f in ["input", "voice", "model", "response_format", "speed", "seed"] {
        assert!(names.contains(&f), "{f} in {names:?}");
    }
    let speed = route["fields"].as_array().unwrap().iter().find(|f| f["name"] == "speed").unwrap();
    assert_eq!(speed["status"], "ignored");
    assert!(routes.iter().any(|r| r["path"] == "/v1/audio/voices" && r["method"] == "GET"));
    let chat = rig.post("/v1/chat/completions", Some("sk-a"), json!({"model": "phonon-31m-en", "messages": [{"role": "user", "content": "hi"}]}));
    assert_ne!(chat.0, 200, "a speech model does not chat");
    rig.stop();
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/speech")
}

fn tiny_phonon_folder() -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-tiny-phonon-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("voices")).unwrap();
    let (flow, mimi) = (fixtures().join("tiny-phonon"), fixtures().join("tiny-mimi"));
    for f in ["config.json", "tokenizer.json"] {
        std::fs::copy(flow.join(f), d.join(f)).unwrap();
    }
    std::fs::copy(flow.join("model.safetensors"), d.join("flow.safetensors")).unwrap();
    std::fs::copy(mimi.join("model.safetensors"), d.join("mimi.safetensors")).unwrap();
    std::fs::copy(flow.join("voice.safetensors"), d.join("voices/tiny.safetensors")).unwrap();
    let mut map = serde_json::Map::new();
    for f in ["flow.safetensors", "mimi.safetensors"] {
        let bytes = std::fs::read(d.join(f)).unwrap();
        for n in safetensors_names(&bytes) {
            map.insert(n, Value::from(f));
        }
    }
    std::fs::write(d.join("model.safetensors.index.json"), json!({"weight_map": map}).to_string()).unwrap();
    std::fs::write(d.join("voices.jsonl"), "{\"filename\": \"voices/tiny.safetensors\", \"name\": \"Hadley\", \"voice_id\": \"tiny\", \"language\": \"en\", \"locale\": \"en-us\", \"gender\": \"female\"}\n").unwrap();
    d
}

fn safetensors_names(bytes: &[u8]) -> Vec<String> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    header.keys().filter(|k| *k != "__metadata__").cloned().collect()
}

#[test]
fn the_tiny_phonon_fixture_served_on_the_host_engine_streams_the_speaker_samples_rounded_to_pcm16() {
    use inference_bench_core::specrt_speech::tts::service::Speaker;
    let dir = tiny_phonon_folder();
    let factory = |max_ctx: Option<usize>| -> Result<Box<dyn Target>, BackendError> {
        let host: inference_bench_core::native::SpeakerLoader = Arc::new(|dir: &Path, _| Service::load_on_host(dir));
        Ok(Box::new(inference_bench_core::native::Native::new(max_ctx).with_speaker_loader(host)))
    };
    let rig = start_with(config(&format!("[[models]]\nid = 'tiny-phonon'\nmodel = {:?}\n", dir.display().to_string())), factory, Default::default());
    let models = json_of(&rig.get("/v1/models", None).1.into_bytes());
    assert_eq!(models["data"][0]["type"], "speech", "known as speech from disk before it loads: {models}");
    let text = "Hello there, how are you? I am fine.";
    let mut body = reader_body(text, "alloy");
    body["seed"] = json!(9);
    let a = send(&rig, &body, "", true, None);
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    assert_eq!(header(&a.headers, "x-sample-rate"), Some("24000"));
    let mut speaker = Speaker::open_host(&dir).unwrap();
    let mut want = Vec::new();
    let job = Job { text: text.into(), voice: Some(dir.join("voices/tiny.safetensors")), seed: 9 };
    speaker.speak(&job, &mut |f| {
        want.extend_from_slice(f);
        true
    })
    .unwrap();
    assert!(want.len() >= 2 * FRAME);
    assert_eq!(a.body, wav::f32_to_pcm16(&want), "streamed bytes are the engine samples rounded to PCM16");
    let mut wav_body = client_speak_body(text, "Hadley");
    wav_body["seed"] = json!(9);
    let b = send(&rig, &wav_body, "", false, None);
    assert_eq!(b.body, wav::wav_bytes(&want, 24_000));
    println!("tiny phonon through /v1/audio/speech: {} samples, stream and WAV equal to the speaker", want.len());
    let v = json_of(&rig.get("/v1/audio/voices", None).1.into_bytes());
    assert_eq!(v, json!({"voices": [{"id": "Hadley", "label": "Hadley, American", "installed": true}]}));
    assert_eq!(json_of(&rig.get("/v1/audio/status", None).1.into_bytes())["speaking_on"], "CPU");
    rig.stop();
    let _ = std::fs::remove_dir_all(&dir);
}
