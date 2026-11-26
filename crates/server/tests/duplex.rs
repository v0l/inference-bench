mod common;

use base64::Engine as _;
use common::*;
use inference_bench_core::duplex_client::{self, Options};
use inference_bench_core::result::{BackendError, Generation};
use inference_bench_core::specrt_duplex::clock::{Control, FrameEngine, Out, Reply, SessionId, StartRequest};
use inference_bench_core::specrt_duplex::service::{Service, ServiceInfo};
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_media::ModelKind;
use inference_bench_core::target::{Capabilities, Request, Target};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const SAMPLES: usize = 320;
const FRAME_MS: u64 = 20;

#[derive(Default)]
struct Seen {
    started: Mutex<Vec<StartRequest>>,
    ended: Mutex<Vec<SessionId>>,
    said: Mutex<Vec<String>>,
    parked: Mutex<Vec<SessionId>>,
    resumed: Mutex<Vec<String>>,
    max_frames: AtomicUsize,
}

struct Echo {
    seen: Arc<Seen>,
    next: SessionId,
    frames: BTreeMap<SessionId, u64>,
}

impl FrameEngine for Echo {
    fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> anyhow::Result<Vec<(SessionId, Out)>> {
        let limit = self.seen.max_frames.load(Ordering::SeqCst) as u64;
        Ok(frames
            .iter()
            .map(|(id, f)| {
                let n = self.frames.entry(*id).or_default();
                *n += 1;
                let text = if *n % 2 == 0 { format!("w{n} ") } else { String::new() };
                (*id, Out { audio: f.iter().map(|v| v * 0.5).collect(), token: Some(1), text, index: *n, late: false, done: limit > 0 && *n >= limit, error: None, input: None })
            })
            .collect())
    }

    fn start(&mut self, r: StartRequest) -> anyhow::Result<(SessionId, Out)> {
        if let Some(k) = &r.resume {
            self.seen.resumed.lock().unwrap().push(k.clone());
        }
        anyhow::ensure!(r.options.card != "boom", "the card does not fit");
        self.seen.started.lock().unwrap().push(StartRequest { options: r.options.clone(), card_fields: r.card_fields.clone(), prompt: None, resume: r.resume.clone() });
        self.next += 1;
        self.frames.insert(self.next, 0);
        Ok((self.next, Out { audio: vec![0.25; SAMPLES], token: Some(0), text: "hi".into(), index: 0, late: false, done: false, error: None, input: None }))
    }

    fn control(&mut self, _id: SessionId, c: Control) -> anyhow::Result<Reply> {
        match c {
            Control::Say(t) => {
                self.seen.said.lock().unwrap().push(t.clone());
                Ok(Reply::Queued(t.split_whitespace().count()))
            }
            Control::Mode(_) => Ok(Reply::Done),
            Control::Interrupt => Ok(Reply::Said { said: "half".into(), unsaid: 3 }),
        }
    }

    fn end(&mut self, id: SessionId) -> anyhow::Result<()> {
        self.seen.ended.lock().unwrap().push(id);
        Ok(())
    }

    fn park(&mut self, id: SessionId) -> anyhow::Result<String> {
        self.seen.parked.lock().unwrap().push(id);
        Ok(format!("key-{id}"))
    }
}

struct FakeDuplex {
    service: Option<Arc<Service>>,
    seen: Arc<Seen>,
    slots: usize,
    snapshots: bool,
}

impl Target for FakeDuplex {
    fn name(&self) -> &str {
        "fake-duplex"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { streaming: true, kind: ModelKind::Duplex, ..Capabilities::default() }
    }

    fn load(&mut self, _model: &str) -> Result<(), BackendError> {
        let (seen, slots, snapshots) = (self.seen.clone(), self.slots, self.snapshots);
        let s = Service::spawn(Duration::from_millis(FRAME_MS), SAMPLES, 8, move || {
            let info = ServiceInfo { name: "echo".into(), sample_rate: 16000, frame_samples: SAMPLES, slots, max_ctx: 256, device: 0, snapshots };
            Ok((Box::new(Echo { seen, next: 0, frames: BTreeMap::new() }) as Box<dyn FrameEngine>, info))
        })
        .map_err(|e| BackendError::failed(e.to_string()))?;
        self.service = Some(s);
        Ok(())
    }

    fn generate(&mut self, _: &Request, _: &mut dyn FnMut(&str)) -> Result<Generation, BackendError> {
        Err(BackendError::failed("duplex only"))
    }

    fn duplex(&self) -> Option<Arc<Service>> {
        self.service.clone()
    }
}

fn rig(slots: usize, extra: &str, snapshots: bool) -> (Rig, Arc<Seen>) {
    let seen = Arc::new(Seen::default());
    let s = seen.clone();
    let factory = move |_: Option<usize>| -> Result<Box<dyn Target>, BackendError> { Ok(Box::new(FakeDuplex { service: None, seen: s.clone(), slots, snapshots })) };
    let rig = start_with(config(&format!("{extra}\n[[models]]\nid='haru'\n")), factory, Default::default());
    (rig, seen)
}

fn opts(rig: &Rig, start: Value) -> Options {
    Options { url: duplex_client::ws_url(&rig.url, "/v1/duplex"), key: None, start, realtime: false, tail_frames: 2, paced: true }
}

fn input(frames: usize) -> Vec<f32> {
    (0..frames * SAMPLES).map(|i| ((i / SAMPLES) as f32 + 1.0) / 16.0).collect()
}

#[test]
fn a_session_streams_frames_both_ways_with_text_events_and_logs_like_chat() {
    let (rig, seen) = rig(2, "", false);
    let run = duplex_client::stream(&opts(&rig, json!({"type": "start", "model": "haru", "card": "Be calm.", "mode": "relay", "temperature": 0.4, "seed": 9})), &input(4)).unwrap();
    assert_eq!(run.ready["type"], "ready");
    assert_eq!(run.ready["frame_bytes"], SAMPLES * 2);
    assert_eq!(run.frames[0], wav::pcm16_round(&[0.25; SAMPLES]), "the first frame comes from the prompt");
    for (i, f) in run.frames[1..5].iter().enumerate() {
        let want = wav::pcm16_round(&vec![(i as f32 + 1.0) / 32.0; SAMPLES]);
        assert_eq!(f, &want, "frame {} is the engine's reply to user frame {i}", i + 1);
    }
    assert!(run.text.starts_with("hiw2 w4 "), "{}", run.text);
    assert!(!run.events.iter().any(|e| e["type"] == "text" && e["text"] == ""), "empty frames send no text event");
    assert_eq!(run.latency_ms.len(), 4 + 2, "one latency per user frame the engine consumed, matched by its index");
    let started = seen.started.lock().unwrap();
    assert_eq!((started[0].options.card.as_str(), started[0].options.temperature, started[0].options.seed), ("Be calm.", Some(0.4), 9));
    wait_for("session end", || !seen.ended.lock().unwrap().is_empty());
    let m = rig.metrics();
    assert!(m.contains("inference_duplex_sessions_total{model=\"haru\"} 1"), "{m}");
    assert!(m.contains("inference_requests_total{endpoint=\"duplex\",status=\"101\"}"), "{m}");
    let (_, models, _) = rig.get("/v1/models", None);
    let v: Value = serde_json::from_str(&models).unwrap();
    assert_eq!(v["data"][0]["type"], "duplex", "{v}");
}

fn socket(rig: &Rig, path: &str, key: Option<&str>, protocol: Option<&str>) -> tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>> {
    use tungstenite::client::IntoClientRequest;
    let mut r = duplex_client::ws_url(&rig.url, path).into_client_request().unwrap();
    if let Some(k) = key {
        r.headers_mut().insert("authorization", format!("Bearer {k}").parse().unwrap());
    }
    if let Some(p) = protocol {
        r.headers_mut().insert("sec-websocket-protocol", p.parse().unwrap());
    }
    tungstenite::connect(r).unwrap().0
}

fn next_json(s: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>) -> Value {
    loop {
        match s.read().unwrap() {
            tungstenite::Message::Text(t) => return serde_json::from_str(&t).unwrap(),
            tungstenite::Message::Close(_) => panic!("closed"),
            _ => {}
        }
    }
}

fn send(s: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, v: Value) {
    s.send(tungstenite::Message::Text(v.to_string().into())).unwrap();
}

fn until(s: &mut tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>, kind: &str) -> Value {
    for _ in 0..200 {
        let v = next_json(s);
        if v["type"] == kind {
            return v;
        }
    }
    panic!("no {kind} event");
}

#[test]
fn say_mode_interrupt_and_stop_are_answered() {
    let (rig, seen) = rig(1, "", false);
    let mut s = socket(&rig, "/v1/duplex", None, None);
    send(&mut s, json!({"type": "start", "card": {"name": "Haru", "backchannel": "mid"}}));
    until(&mut s, "ready");
    send(&mut s, json!({"type": "say", "text": "the build is green"}));
    assert_eq!(until(&mut s, "queued")["tokens"], 4);
    send(&mut s, json!({"type": "mode", "mode": "free"}));
    assert_eq!(until(&mut s, "mode")["mode"], "free");
    send(&mut s, json!({"type": "interrupt"}));
    let said = until(&mut s, "said");
    assert_eq!((said["text"].as_str(), said["unsaid_tokens"].as_u64()), (Some("half"), Some(3)));
    send(&mut s, json!({"type": "mode", "mode": "loud"}));
    assert!(until(&mut s, "error")["message"].as_str().unwrap().contains("free or relay"));
    send(&mut s, json!({"type": "stop"}));
    until(&mut s, "stopped");
    wait_for("end", || seen.ended.lock().unwrap().len() == 1);
    assert_eq!(seen.said.lock().unwrap().as_slice(), [" the build is green"]);
    assert_eq!(seen.started.lock().unwrap()[0].card_fields.get("backchannel").map(String::as_str), Some("mid"));
}

#[test]
fn a_full_gate_refuses_with_a_reason_and_a_dropped_socket_frees_its_slot() {
    let (rig, seen) = rig(1, "[duplex]\nqueue_depth = 0\n", false);
    let mut a = socket(&rig, "/v1/duplex", None, None);
    send(&mut a, json!({"type": "start"}));
    until(&mut a, "ready");
    let mut b = socket(&rig, "/v1/duplex", None, None);
    send(&mut b, json!({"type": "start"}));
    let e = next_json(&mut b);
    assert_eq!((e["type"].as_str(), e["code"].as_str()), (Some("error"), Some("full")));
    assert!(e["message"].as_str().unwrap().contains("sessions_per_gpu"), "{e}");
    drop(a);
    wait_for("the dropped session ends", || seen.ended.lock().unwrap().len() == 1);
    let mut c = socket(&rig, "/v1/duplex", None, None);
    send(&mut c, json!({"type": "start"}));
    until(&mut c, "ready");
    assert!(rig.metrics().contains("inference_duplex_refused_total{reason=\"full\"} 1"));
}

#[test]
fn the_route_needs_the_api_key_like_chat() {
    let (rig, _) = rig(1, "api_keys = [\"sk-1\"]\n", false);
    let refused = {
        use tungstenite::client::IntoClientRequest;
        let r = duplex_client::ws_url(&rig.url, "/v1/duplex").into_client_request().unwrap();
        tungstenite::connect(r).err().unwrap()
    };
    assert!(matches!(refused, tungstenite::Error::Http(ref r) if r.status() == 401), "{refused:?}");
    let mut s = socket(&rig, "/v1/duplex", Some("sk-1"), None);
    send(&mut s, json!({"type": "start"}));
    until(&mut s, "ready");
}

#[test]
fn a_dropped_session_is_parked_when_keep_idle_is_set_and_resumes_by_token() {
    let (rig, seen) = rig(1, "[duplex]\nkeep_idle_secs = 60\n", true);
    let mut s = socket(&rig, "/v1/duplex", None, None);
    send(&mut s, json!({"type": "start"}));
    let token = until(&mut s, "ready")["resume"].as_str().unwrap().to_string();
    drop(s);
    wait_for("parked", || seen.parked.lock().unwrap().len() == 1);
    assert!(seen.ended.lock().unwrap().is_empty(), "a parked session is not ended");
    let mut r = socket(&rig, "/v1/duplex", None, None);
    send(&mut r, json!({"type": "start", "resume": token}));
    until(&mut r, "ready");
    assert_eq!(seen.resumed.lock().unwrap().as_slice(), ["key-1"]);
    drop(r);
    wait_for("the resumed session is parked again", || seen.parked.lock().unwrap().len() == 2);
    let mut again = socket(&rig, "/v1/duplex", None, None);
    send(&mut again, json!({"type": "start", "resume": token}));
    let e = next_json(&mut again);
    assert_eq!(e["code"], "resume", "{e}");
}

#[test]
fn a_start_error_and_max_frames_end_the_session_with_events() {
    let (rig, seen) = rig(1, "", false);
    let mut s = socket(&rig, "/v1/duplex", None, None);
    send(&mut s, json!({"type": "start", "card": "boom"}));
    let e = next_json(&mut s);
    assert!(e["message"].as_str().unwrap().contains("does not fit"), "{e}");
    seen.max_frames.store(2, Ordering::SeqCst);
    let mut t = socket(&rig, "/v1/duplex", None, None);
    send(&mut t, json!({"type": "start"}));
    until(&mut t, "ready");
    t.send(tungstenite::Message::Binary(vec![0u8; SAMPLES * 2].into())).unwrap();
    assert!(until(&mut t, "end")["reason"].as_str().unwrap().contains("frame limit"));
}

#[test]
fn the_realtime_subset_takes_instructions_and_appended_audio_and_sends_deltas() {
    let (rig, seen) = rig(1, "", false);
    let mut s = socket(&rig, "/v1/realtime?model=haru", None, Some("realtime"));
    assert_eq!(next_json(&mut s)["type"], "session.created");
    send(&mut s, json!({"type": "session.update", "session": {"instructions": "You are Haru."}}));
    assert_eq!(next_json(&mut s)["session"]["instructions"], "You are Haru.");
    send(&mut s, json!({"type": "response.create"}));
    let e = next_json(&mut s);
    assert!(e["error"]["message"].as_str().unwrap().contains("never take turns"), "{e}");
    let pcm = base64::engine::general_purpose::STANDARD.encode(wav::f32_to_pcm16(&vec![0.5; SAMPLES]));
    send(&mut s, json!({"type": "input_audio_buffer.append", "audio": pcm}));
    let mut audio = 0;
    let mut transcript = String::new();
    for _ in 0..80 {
        let v = next_json(&mut s);
        match v["type"].as_str().unwrap() {
            "response.output_audio.delta" => {
                let b = base64::engine::general_purpose::STANDARD.decode(v["delta"].as_str().unwrap()).unwrap();
                assert_eq!(b.len(), SAMPLES * 2);
                audio += 1;
            }
            "response.output_audio_transcript.delta" => transcript.push_str(v["delta"].as_str().unwrap()),
            _ => {}
        }
        if audio >= 3 {
            break;
        }
    }
    assert!(audio >= 3 && transcript.starts_with("hi"), "{audio} {transcript}");
    assert_eq!(seen.started.lock().unwrap()[0].options.card, "You are Haru.");
    send(&mut s, json!({"type": "session.update", "session": {"instructions": "Someone else."}}));
    assert_eq!(until(&mut s, "error")["error"]["code"], "instructions_fixed");
    drop(s);
    wait_for("the realtime session ends", || !seen.ended.lock().unwrap().is_empty());
    wait_for("frames out are counted for realtime deltas", || !rig.metrics().contains("inference_duplex_frames_out_total 0"));
}

#[test]
fn capabilities_list_the_duplex_routes_and_the_realtime_limits() {
    let (rig, _) = rig(1, "", false);
    let (_, body, _) = rig.get("/v1/capabilities", None);
    let v: Value = serde_json::from_str(&body).unwrap();
    let routes = v["routes"].as_array().unwrap();
    let duplex = routes.iter().find(|r| r["path"] == "/v1/duplex").unwrap();
    assert!(duplex["fields"].as_array().unwrap().iter().any(|f| f["name"] == "speak_first"));
    let rt = routes.iter().find(|r| r["path"] == "/v1/realtime").unwrap();
    assert!(rt["unsupported"].as_str().unwrap().contains("response.create"));
    assert!(routes.iter().any(|r| r["path"] == "/v1/duplex/voices"));
}

#[test]
fn an_uploaded_voice_is_used_by_id() {
    let (rig, seen) = rig(1, "", false);
    let pcm = base64::engine::general_purpose::STANDARD.encode(wav::f32_to_pcm16(&[0.1; 1600]));
    let (status, body, _) = rig.post("/v1/duplex/voices", None, json!({"audio": pcm, "sample_rate": 16000}));
    assert_eq!(status, 200, "{body}");
    let id = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let mut s = socket(&rig, "/v1/duplex", None, None);
    send(&mut s, json!({"type": "start", "voice": {"id": id}}));
    until(&mut s, "ready");
    assert_eq!(seen.started.lock().unwrap()[0].options.reference.as_ref().map(Vec::len), Some(1600));
}
