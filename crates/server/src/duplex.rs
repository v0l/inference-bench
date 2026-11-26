use crate::app::{AppState, RequestId};
use crate::error::ApiError;
use axum::Json;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State as AxState};
use axum::response::Response;
use base64::Engine as _;
use inference_bench_core::specrt_duplex::clock::{Control, Out, Reply, StartRequest};
use inference_bench_core::specrt_duplex::prompt::Prompt;
use inference_bench_core::specrt_duplex::service::{Service, Session, SessionHandle};
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_duplex::{Mode, RecordedNoise, StartOptions};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

const START_FIELDS: &[&str] = &["type", "model", "card", "voice", "speak_first", "guidance", "temperature", "mode", "seed", "resume", "replay"];

struct ModelGate {
    running: Arc<Semaphore>,
    admitted: AtomicUsize,
    limit: usize,
}

struct Voice {
    samples: Vec<f32>,
    rate: u32,
}

#[derive(Default)]
pub struct State {
    gates: Mutex<HashMap<String, Arc<ModelGate>>>,
    voices: Mutex<VecDeque<(String, Arc<Voice>)>>,
    parked: Mutex<HashMap<String, String>>,
    open: AtomicUsize,
}

impl State {
    pub fn open(&self) -> usize {
        self.open.load(Ordering::SeqCst)
    }

    fn gate(&self, model: &str, slots: usize, queue: usize) -> Arc<ModelGate> {
        let mut g = self.gates.lock().unwrap();
        g.entry(model.to_string()).or_insert_with(|| Arc::new(ModelGate { running: Arc::new(Semaphore::new(slots.max(1))), admitted: AtomicUsize::new(0), limit: slots.max(1) + queue })).clone()
    }

    fn voice(&self, id: &str) -> Option<Arc<Voice>> {
        self.voices.lock().unwrap().iter().find(|(k, _)| k == id).map(|(_, v)| v.clone())
    }
}

struct Ticket {
    gate: Arc<ModelGate>,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.gate.admitted.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Opened {
    state: Arc<AppState>,
}

impl Drop for Opened {
    fn drop(&mut self) {
        self.state.duplex.open.fetch_sub(1, Ordering::SeqCst);
    }
}

fn admit(gate: &Arc<ModelGate>) -> Result<Ticket, String> {
    let mut current = gate.admitted.load(Ordering::SeqCst);
    loop {
        if current >= gate.limit {
            return Err(format!("every duplex session on this model is in use ({current} open or waiting, {} allowed by duplex.sessions_per_gpu and duplex.queue_depth): retry later", gate.limit));
        }
        match gate.admitted.compare_exchange(current, current + 1, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return Ok(Ticket { gate: gate.clone() }),
            Err(c) => current = c,
        }
    }
}

fn decode_b64(s: &str, what: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD.decode(s.trim()).map_err(|e| format!("{what} is not base64: {e}"))
}

fn pcm_of(bytes: &[u8], what: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("{what} has an odd byte count; PCM16 is two bytes a sample"));
    }
    Ok(wav::pcm16_to_f32(bytes))
}

fn read_voice(state: &State, v: &Value, rate: usize) -> Result<Option<Vec<f32>>, String> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => pcm_of(&decode_b64(s, "voice")?, "voice").map(Some),
        Value::Object(o) => {
            if let Some(id) = o.get("id").and_then(Value::as_str) {
                let voice = state.voice(id).ok_or_else(|| format!("no uploaded voice {id}; POST it to /v1/duplex/voices first"))?;
                if voice.rate as usize != rate {
                    return Err(format!("voice {id} is {} Hz and this model runs at {rate} Hz", voice.rate));
                }
                return Ok(Some(voice.samples.clone()));
            }
            if let Some(w) = o.get("wav").and_then(Value::as_str) {
                let (samples, r) = wav::read_wav(&decode_b64(w, "voice.wav")?).map_err(|e| format!("voice.wav: {e}"))?;
                if r as usize != rate {
                    return Err(format!("voice.wav is {r} Hz and this model runs at {rate} Hz"));
                }
                return Ok(Some(samples));
            }
            Err("voice is base64 PCM16, {\"id\": uploaded id} or {\"wav\": base64 WAV}".into())
        }
        _ => Err("voice is base64 PCM16, {\"id\": uploaded id} or {\"wav\": base64 WAV}".into()),
    }
}

pub struct Parsed {
    pub model: Option<String>,
    pub request: StartRequest,
}

pub fn parse_start(state: &State, body: &Value, rate: usize, allow_replay: bool) -> Result<Parsed, String> {
    let obj = body.as_object().ok_or("the start message is a JSON object")?;
    if body["type"].as_str() != Some("start") {
        return Err(format!("the first message must be {{\"type\": \"start\", ...}}, not {}", body["type"]));
    }
    if let Some(k) = obj.keys().find(|k| !START_FIELDS.contains(&k.as_str())) {
        return Err(format!("unknown start field {k}; start reads {}", START_FIELDS.join(", ")));
    }
    let mut options = StartOptions::default();
    let mut card_fields = BTreeMap::new();
    match &body["card"] {
        Value::Null => {}
        Value::String(s) => options.card = s.trim().to_string(),
        Value::Object(o) => {
            for (k, v) in o {
                card_fields.insert(k.clone(), v.as_str().ok_or_else(|| format!("card field {k} must be a string"))?.to_string());
            }
        }
        _ => return Err("card is text or an object of card fields".into()),
    }
    options.reference = read_voice(state, &body["voice"], rate)?;
    options.speak_first = body["speak_first"].as_bool().unwrap_or(false);
    options.guidance = match &body["guidance"] {
        Value::Null => None,
        v => Some(v.as_f64().ok_or("guidance must be a number")? as f32),
    };
    options.temperature = match &body["temperature"] {
        Value::Null => None,
        v => {
            let t = v.as_f64().ok_or("temperature must be a number")?;
            if !(0.0..=2.0).contains(&t) {
                return Err(format!("temperature {t} is outside 0 to 2"));
            }
            Some(t as f32)
        }
    };
    options.mode = parse_mode(&body["mode"])?.unwrap_or_default();
    options.seed = match &body["seed"] {
        Value::Null => uuid::Uuid::new_v4().as_u64_pair().0 >> 12,
        v => v.as_u64().ok_or("seed must be a whole number")?,
    };
    let mut prompt = None;
    if !body["replay"].is_null() {
        if !allow_replay {
            return Err("replay is a test hook; it needs duplex.allow_replay in the server config".into());
        }
        let r = &body["replay"];
        prompt = Some(serde_json::from_value::<Prompt>(r["prompt"].clone()).map_err(|e| format!("replay.prompt: {e}"))?);
        options.noise = Some(serde_json::from_value::<RecordedNoise>(r["noise"].clone()).map_err(|e| format!("replay.noise: {e}"))?);
        options.greedy = r["greedy"].as_bool().unwrap_or(true);
    }
    let resume = body["resume"].as_str().map(String::from);
    Ok(Parsed { model: body["model"].as_str().filter(|m| !m.is_empty()).map(String::from), request: StartRequest { options, card_fields, prompt, resume } })
}

fn parse_mode(v: &Value) -> Result<Option<Mode>, String> {
    match v.as_str() {
        None if v.is_null() => Ok(None),
        Some("free") => Ok(Some(Mode::Free)),
        Some("relay") => Ok(Some(Mode::Relay)),
        _ => Err(format!("mode is free or relay, not {v}")),
    }
}

async fn send_json(ws: &mut WebSocket, v: Value) -> bool {
    ws.send(Message::Text(v.to_string().into())).await.is_ok()
}

async fn refuse(ws: &mut WebSocket, state: &AppState, reason: &str, message: String) {
    state.metrics.duplex_refused(reason);
    tracing::warn!(reason, %message, "duplex session refused");
    let _ = send_json(ws, json!({"type": "error", "code": reason, "message": message})).await;
    let _ = ws.send(Message::Close(None)).await;
}

async fn first_text(ws: &mut WebSocket) -> Option<Result<Value, String>> {
    loop {
        match ws.recv().await? {
            Ok(Message::Text(t)) => return Some(serde_json::from_str(&t).map_err(|e| format!("the start message is not JSON: {e}"))),
            Ok(Message::Binary(_)) => return Some(Err("send the JSON start message before any audio".into())),
            Ok(Message::Close(_)) | Err(_) => return None,
            Ok(_) => {}
        }
    }
}

struct Live {
    model: String,
    service: Arc<Service>,
    session: Session,
    frames: mpsc::UnboundedReceiver<Out>,
    _permit: OwnedSemaphorePermit,
    _ticket: Ticket,
    _held: crate::manager::InUse,
    _opened: Opened,
    resume_token: Option<String>,
}

async fn open_session(ws: &mut WebSocket, state: &Arc<AppState>, model: Option<&str>, build: impl FnOnce(&Service) -> Result<StartRequest, String>) -> Option<Live> {
    let named = match state.manager.requested_model(model) {
        Ok(m) => m,
        Err(e) => {
            refuse(ws, state, "model", e.message).await;
            return None;
        }
    };
    let (id, service, held) = match state.manager.duplex(&named).await {
        Ok(x) => x,
        Err(e) => {
            refuse(ws, state, "model", e.message).await;
            return None;
        }
    };
    tracing::Span::current().record("model", id.as_str());
    let gate = state.duplex.gate(&id, service.info.slots, state.config.duplex.queue_depth);
    let ticket = match admit(&gate) {
        Ok(t) => t,
        Err(m) => {
            refuse(ws, state, "full", m).await;
            return None;
        }
    };
    let permit = match gate.running.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            if !send_json(ws, json!({"type": "queued", "waiting": gate.admitted.load(Ordering::SeqCst).saturating_sub(service.info.slots)})).await {
                return None;
            }
            gate.running.clone().acquire_owned().await.ok()?
        }
    };
    let request = match build(&service) {
        Ok(r) => r,
        Err(m) => {
            refuse(ws, state, "invalid", m).await;
            return None;
        }
    };
    let resuming = request.resume.clone();
    let request = match resuming.as_deref() {
        Some(token) => {
            let key = state.duplex.parked.lock().unwrap().remove(token);
            match key {
                Some(key) => StartRequest { resume: Some(key), ..request },
                None => {
                    drop(permit);
                    refuse(ws, state, "resume", format!("no parked session {token}: it expired, was resumed already, or this server keeps none")).await;
                    return None;
                }
            }
        }
        None => request,
    };
    let svc = service.clone();
    let started = tokio::task::spawn_blocking(move || svc.start(request)).await;
    let mut session = match started {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            refuse(ws, state, "start", format!("{e:#}")).await;
            return None;
        }
        Err(e) => {
            refuse(ws, state, "start", format!("start task failed: {e}")).await;
            return None;
        }
    };
    let (tx, frames) = mpsc::unbounded_channel();
    if let Some(rx) = session.frames() {
        std::thread::Builder::new()
            .name("duplex-frames".into())
            .spawn(move || {
                while let Ok(o) = rx.recv() {
                    if tx.send(o).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
    }
    state.duplex.open.fetch_add(1, Ordering::SeqCst);
    state.metrics.duplex_session(&id);
    let resume_token = service.info.snapshots.then(|| format!("dpx-{}", uuid::Uuid::new_v4().simple()));
    Some(Live { model: id, service, session, frames, _permit: permit, _ticket: ticket, _held: held, _opened: Opened { state: state.clone() }, resume_token })
}

async fn control(handle: &SessionHandle, c: Control) -> Result<Reply, String> {
    let h = handle.clone();
    match tokio::task::spawn_blocking(move || h.control(c)).await {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(e) => Err(format!("control task failed: {e}")),
    }
}

#[derive(Default)]
struct Tally {
    frames_in: u64,
    frames_out: u64,
    late: u64,
    bytes_in: u64,
}

enum Ending {
    Dropped,
    Stopped,
    Done(String),
    Failed(String),
}

fn finish(state: &AppState, live: Live, tally: &Tally, ending: Ending, request_id: &str, path: &str, began: Instant) {
    state.metrics.duplex_frames(tally.frames_in, tally.frames_out, tally.late);
    let reason = match &ending {
        Ending::Dropped => "dropped",
        Ending::Stopped => "stopped",
        Ending::Done(_) => "done",
        Ending::Failed(_) => "failed",
    };
    let mut parked = None;
    let Live { session, resume_token, model, .. } = live;
    match (&ending, resume_token) {
        (Ending::Dropped, Some(token)) => match session.park() {
            Ok(key) => {
                state.duplex.parked.lock().unwrap().insert(token.clone(), key);
                parked = Some(token);
            }
            Err(e) => tracing::warn!(error = %format!("{e:#}"), "the dropped duplex session was not snapshotted"),
        },
        _ => session.close(),
    }
    let detail = match &ending {
        Ending::Done(d) | Ending::Failed(d) => d.as_str(),
        _ => "",
    };
    tracing::info!(model = %model, reason, detail, frames_in = tally.frames_in, frames_out = tally.frames_out, late = tally.late, parked = parked.is_some(), "duplex session ended");
    state.write_access(json!({
        "time": chrono::Utc::now().to_rfc3339(),
        "request_id": request_id,
        "method": "GET",
        "path": path,
        "status": 101,
        "model": model,
        "duplex": {"frames_in": tally.frames_in, "frames_out": tally.frames_out, "late_frames": tally.late, "bytes_in": tally.bytes_in, "ended": reason, "parked": parked.is_some()},
        "elapsed_ms": began.elapsed().as_secs_f64() * 1000.0,
    }));
}

trait Dialect {
    fn ready(&self, live: &Live) -> Vec<Value>;
    fn frame(&mut self, o: &Out) -> (Option<Message>, Vec<Value>);
    async fn message(&mut self, live: &Live, text: &str) -> Result<Vec<Value>, Ending>;
    fn audio(&self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        pcm_of(bytes, "an audio frame")
    }

    fn text_audio(&self, _text: &str) -> Option<Result<Vec<f32>, String>> {
        None
    }
}

struct NativeDialect;

impl Dialect for NativeDialect {
    fn ready(&self, live: &Live) -> Vec<Value> {
        let i = &live.service.info;
        vec![json!({"type": "ready", "session": live.session.id, "model": live.model, "sample_rate": i.sample_rate, "frame_samples": i.frame_samples, "frame_bytes": i.frame_samples * 2, "format": "pcm16le mono", "resume": live.resume_token})]
    }

    fn frame(&mut self, o: &Out) -> (Option<Message>, Vec<Value>) {
        let mut events = Vec::new();
        if let Some(i) = o.input {
            events.push(json!({"type": "frame", "frame": o.index, "input": i}));
        }
        if !o.text.is_empty() {
            events.push(json!({"type": "text", "frame": o.index, "text": o.text}));
        }
        if o.late {
            events.push(json!({"type": "late", "frame": o.index}));
        }
        let audio = (!o.audio.is_empty()).then(|| Message::Binary(wav::f32_to_pcm16(&o.audio).into()));
        (audio, events)
    }

    async fn message(&mut self, live: &Live, text: &str) -> Result<Vec<Value>, Ending> {
        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => return Ok(vec![json!({"type": "error", "message": format!("not JSON: {e}")})]),
        };
        let handle = live.session.handle();
        let reply = match v["type"].as_str() {
            Some("say") => match v["text"].as_str().filter(|t| !t.trim().is_empty()) {
                Some(t) => control(&handle, Control::Say(format!(" {}", t.trim()))).await,
                None => Err("say needs text".into()),
            },
            Some("mode") => match parse_mode(&v["mode"]) {
                Ok(Some(m)) => control(&handle, Control::Mode(m)).await,
                Ok(None) => Err("mode needs mode: free or relay".into()),
                Err(e) => Err(e),
            },
            Some("interrupt") => control(&handle, Control::Interrupt).await,
            Some("stop") => return Err(Ending::Stopped),
            other => Err(format!("unknown control {other:?}; use say, mode, interrupt or stop")),
        };
        Ok(vec![match reply {
            Ok(Reply::Queued(n)) => json!({"type": "queued", "tokens": n}),
            Ok(Reply::Said { said, unsaid }) => json!({"type": "said", "text": said, "unsaid_tokens": unsaid}),
            Ok(Reply::Done) => json!({"type": "mode", "mode": v["mode"]}),
            Err(m) => json!({"type": "error", "message": m}),
        }])
    }
}

async fn pump<D: Dialect>(ws: &mut WebSocket, state: &AppState, mut live: Live, first: Out, dialect: &mut D, request_id: &str, path: &str) {
    let began = Instant::now();
    let mut tally = Tally::default();
    let handle = live.session.handle();
    for e in dialect.ready(&live) {
        if !send_json(ws, e).await {
            return finish(state, live, &tally, Ending::Dropped, request_id, path, began);
        }
    }
    let mut pending = Some(first);
    let ending = loop {
        if let Some(o) = pending.take() {
            if o.late {
                tally.late += 1;
            }
            if let Some(e) = &o.error {
                let _ = send_json(ws, json!({"type": "error", "message": e})).await;
                break Ending::Failed(e.clone());
            }
            let (audio, events) = dialect.frame(&o);
            if !o.audio.is_empty() {
                tally.frames_out += 1;
            }
            if let Some(a) = audio {
                if ws.send(a).await.is_err() {
                    break Ending::Dropped;
                }
            }
            let mut ok = true;
            for e in events {
                ok &= send_json(ws, e).await;
            }
            if !ok {
                break Ending::Dropped;
            }
            if o.done {
                let _ = send_json(ws, json!({"type": "end", "reason": "the session reached its frame limit", "frame": o.index})).await;
                break Ending::Done("max_frames".into());
            }
            continue;
        }
        tokio::select! {
            o = live.frames.recv() => match o {
                Some(o) => pending = Some(o),
                None => {
                    let _ = send_json(ws, json!({"type": "error", "message": "the duplex engine stopped"})).await;
                    break Ending::Failed("engine stopped".into());
                }
            },
            m = ws.recv() => match m {
                Some(Ok(Message::Binary(b))) => {
                    tally.bytes_in += b.len() as u64;
                    match dialect.audio(&b) {
                        Ok(pcm) => {
                            tally.frames_in += 1;
                            if handle.audio(pcm).is_err() {
                                break Ending::Failed("engine stopped".into());
                            }
                        }
                        Err(e) => {
                            if !send_json(ws, json!({"type": "error", "message": e})).await {
                                break Ending::Dropped;
                            }
                        }
                    }
                }
                Some(Ok(Message::Text(t))) => {
                    if let Some(pcm) = dialect.text_audio(&t) {
                        match pcm {
                            Ok(p) => {
                                tally.frames_in += 1;
                                tally.bytes_in += p.len() as u64 * 2;
                                if handle.audio(p).is_err() {
                                    break Ending::Failed("engine stopped".into());
                                }
                            }
                            Err(e) => {
                                let _ = send_json(ws, json!({"type": "error", "error": {"type": "invalid_request_error", "message": e}})).await;
                            }
                        }
                        continue;
                    }
                    match dialect.message(&live, &t).await {
                        Ok(events) => {
                            let mut ok = true;
                            for e in events {
                                ok &= send_json(ws, e).await;
                            }
                            if !ok {
                                break Ending::Dropped;
                            }
                        }
                        Err(end) => {
                            if matches!(end, Ending::Stopped) {
                                let _ = send_json(ws, json!({"type": "stopped"})).await;
                                let _ = ws.send(Message::Close(None)).await;
                            }
                            break end;
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break Ending::Dropped,
                Some(Ok(_)) => {}
            },
        }
    };
    finish(state, live, &tally, ending, request_id, path, began);
}

fn realtime_append(text: &str) -> Option<Result<Vec<f32>, String>> {
    if !text.contains("input_audio_buffer.append") {
        return None;
    }
    let v: Value = serde_json::from_str(text).ok()?;
    if v["type"].as_str() != Some("input_audio_buffer.append") {
        return None;
    }
    Some(v["audio"].as_str().ok_or_else(|| "input_audio_buffer.append needs audio, base64 PCM16".to_string()).and_then(|a| pcm_of(&decode_b64(a, "audio")?, "audio")))
}

pub async fn native(AxState(state): AxState<Arc<AppState>>, Extension(rid): Extension<RequestId>, ws: WebSocketUpgrade) -> Response {
    ws.max_message_size(state.config.media.request_body_limit()).on_upgrade(move |mut socket| async move {
        let Some(start) = first_text(&mut socket).await else { return };
        let body = match start {
            Ok(v) => v,
            Err(m) => return refuse(&mut socket, &state, "invalid", m).await,
        };
        let model = body["model"].as_str().map(String::from);
        let allow = state.config.duplex.allow_replay;
        let st = state.clone();
        let Some(mut live) = open_session(&mut socket, &state, model.as_deref(), |svc| parse_start(&st.duplex, &body, svc.info.sample_rate, allow).map(|p| p.request)).await else { return };
        let first = std::mem::replace(&mut live.session.first, Out { audio: Vec::new(), token: None, text: String::new(), index: 0, late: false, done: false, error: None, input: None });
        pump(&mut socket, &state, live, first, &mut NativeDialect, &rid.0, "/v1/duplex").await;
    })
}

#[derive(serde::Deserialize)]
pub struct RealtimeQuery {
    model: Option<String>,
}

struct RealtimeDialect {
    response: String,
    item: String,
    counter: u64,
    instructions: String,
}

impl RealtimeDialect {
    fn event(&mut self, mut v: Value) -> Value {
        self.counter += 1;
        v["event_id"] = json!(format!("event_{}", self.counter));
        v
    }
}

pub const REALTIME_UNSUPPORTED: &str = "Haru style duplex models never take turns: response.create, response.cancel, conversation.item.* and input_audio_buffer.commit or clear do not apply. Stream input_audio_buffer.append; audio and transcript deltas come back every frame.";

impl Dialect for RealtimeDialect {
    fn text_audio(&self, text: &str) -> Option<Result<Vec<f32>, String>> {
        realtime_append(text)
    }

    fn ready(&self, live: &Live) -> Vec<Value> {
        let i = &live.service.info;
        vec![
            json!({"type": "session.updated", "session": {"object": "realtime.session", "model": live.model, "instructions": self.instructions, "output_modalities": ["audio"], "audio": {"input": {"format": {"type": "audio/pcm", "rate": i.sample_rate}}, "output": {"format": {"type": "audio/pcm", "rate": i.sample_rate}}}}}),
            json!({"type": "response.created", "response": {"id": self.response, "object": "realtime.response", "status": "in_progress"}}),
        ]
    }

    fn frame(&mut self, o: &Out) -> (Option<Message>, Vec<Value>) {
        let mut events = Vec::new();
        if !o.audio.is_empty() {
            let delta = base64::engine::general_purpose::STANDARD.encode(wav::f32_to_pcm16(&o.audio));
            let mut e = json!({"type": "response.output_audio.delta", "response_id": self.response, "item_id": self.item, "output_index": 0, "content_index": 0, "delta": delta});
            if let Some(i) = o.input {
                e["input_frame"] = json!(i);
            }
            events.push(self.event(e));
        }
        if !o.text.is_empty() {
            let e = json!({"type": "response.output_audio_transcript.delta", "response_id": self.response, "item_id": self.item, "output_index": 0, "content_index": 0, "delta": o.text});
            events.push(self.event(e));
        }
        (None, events)
    }

    async fn message(&mut self, _live: &Live, text: &str) -> Result<Vec<Value>, Ending> {
        let v: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        let e = match v["type"].as_str() {
            Some("session.update") => {
                let asked = v["session"]["instructions"].as_str();
                match asked {
                    Some(i) if i.trim() != self.instructions.trim() => json!({"type": "error", "error": {"type": "invalid_request_error", "code": "instructions_fixed", "message": "instructions are the character card at position 0 of the session; they cannot change once audio has started. Open a new session"}}),
                    _ => json!({"type": "session.updated", "session": {"instructions": self.instructions}}),
                }
            }
            Some(t) => json!({"type": "error", "error": {"type": "invalid_request_error", "code": "unsupported_event", "message": format!("{t}: {REALTIME_UNSUPPORTED}")}}),
            None => json!({"type": "error", "error": {"type": "invalid_request_error", "message": "every client event is a JSON object with a type"}}),
        };
        Ok(vec![self.event(e)])
    }
}

pub async fn realtime(AxState(state): AxState<Arc<AppState>>, Extension(rid): Extension<RequestId>, Query(q): Query<RealtimeQuery>, ws: WebSocketUpgrade) -> Response {
    ws.protocols(["realtime"]).max_message_size(state.config.media.request_body_limit()).on_upgrade(move |mut socket| async move {
        let mut dialect = RealtimeDialect { response: format!("resp_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]), item: format!("item_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]), counter: 0, instructions: String::new() };
        let created = dialect.event(json!({"type": "session.created", "session": {"object": "realtime.session", "model": q.model, "output_modalities": ["audio"], "note": REALTIME_UNSUPPORTED}}));
        if !send_json(&mut socket, created).await {
            return;
        }
        let mut early_audio: Vec<f32> = Vec::new();
        loop {
            let Some(Ok(m)) = socket.recv().await else { return };
            let Message::Text(t) = m else { continue };
            if let Some(pcm) = realtime_append(&t) {
                match pcm {
                    Ok(p) => {
                        early_audio.extend(p);
                        break;
                    }
                    Err(e) => {
                        let ev = dialect.event(json!({"type": "error", "error": {"type": "invalid_request_error", "message": e}}));
                        let _ = send_json(&mut socket, ev).await;
                        continue;
                    }
                }
            }
            let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
            let ev = match v["type"].as_str() {
                Some("session.update") => {
                    if let Some(i) = v["session"]["instructions"].as_str() {
                        dialect.instructions = i.trim().to_string();
                    }
                    json!({"type": "session.updated", "session": {"instructions": dialect.instructions}})
                }
                Some(other) => json!({"type": "error", "error": {"type": "invalid_request_error", "code": "unsupported_event", "message": format!("{other}: {REALTIME_UNSUPPORTED}")}}),
                None => json!({"type": "error", "error": {"type": "invalid_request_error", "message": "every client event is a JSON object with a type"}}),
            };
            let ev = dialect.event(ev);
            if !send_json(&mut socket, ev).await {
                return;
            }
        }
        let card = dialect.instructions.clone();
        let Some(mut live) = open_session(&mut socket, &state, q.model.as_deref(), |_| Ok(StartRequest { options: StartOptions { card, seed: uuid::Uuid::new_v4().as_u64_pair().0 >> 12, ..StartOptions::default() }, card_fields: BTreeMap::new(), prompt: None, resume: None })).await else { return };
        let _ = live.session.handle().audio(early_audio);
        let first = std::mem::replace(&mut live.session.first, Out { audio: Vec::new(), token: None, text: String::new(), index: 0, late: false, done: false, error: None, input: None });
        pump(&mut socket, &state, live, first, &mut dialect, &rid.0, "/v1/realtime").await;
    })
}

pub async fn upload_voice(AxState(state): AxState<Arc<AppState>>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let audio = body["audio"].as_str().ok_or_else(|| ApiError::bad_request("audio is required: base64 PCM16 mono or a base64 WAV").with_code("invalid_value", "audio"))?;
    let bytes = decode_b64(audio, "audio").map_err(ApiError::bad_request)?;
    let (samples, rate) = match body["format"].as_str().unwrap_or("pcm16") {
        "pcm16" => (pcm_of(&bytes, "audio").map_err(ApiError::bad_request)?, body["sample_rate"].as_u64().unwrap_or(24000) as u32),
        "wav" => wav::read_wav(&bytes).map_err(|e| ApiError::bad_request(format!("audio: {e}")))?,
        other => return Err(ApiError::bad_request(format!("format {other:?} is pcm16 or wav"))),
    };
    if samples.is_empty() {
        return Err(ApiError::bad_request("the voice reference is empty"));
    }
    let id = format!("voice_{}", &uuid::Uuid::new_v4().simple().to_string()[..20]);
    let seconds = samples.len() as f64 / rate.max(1) as f64;
    let mut voices = state.duplex.voices.lock().unwrap();
    voices.push_back((id.clone(), Arc::new(Voice { samples, rate })));
    while voices.len() > state.config.duplex.max_voices.max(1) {
        voices.pop_front();
    }
    Ok(Json(json!({"id": id, "object": "duplex.voice", "sample_rate": rate, "seconds": seconds})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_start_message_reads_card_text_or_fields_voice_and_sampling() {
        let s = State::default();
        let pcm = base64::engine::general_purpose::STANDARD.encode(wav::f32_to_pcm16(&[0.5, -0.5]));
        let p = parse_start(&s, &json!({"type": "start", "model": "haru", "card": "Be calm.", "voice": pcm, "speak_first": true, "guidance": 1.5, "temperature": 0.5, "mode": "relay", "seed": 7}), 24000, false).unwrap();
        let o = &p.request.options;
        assert_eq!((p.model.as_deref(), o.card.as_str(), o.speak_first, o.guidance, o.temperature, o.mode, o.seed), (Some("haru"), "Be calm.", true, Some(1.5), Some(0.5), Mode::Relay, 7));
        assert_eq!(o.reference.as_ref().unwrap().len(), 2);
        let f = parse_start(&s, &json!({"type": "start", "card": {"name": "Haru", "backchannel": "mid"}}), 24000, false).unwrap();
        assert_eq!(f.request.card_fields.get("name").map(String::as_str), Some("Haru"));
        assert!(f.request.options.card.is_empty());
    }

    #[test]
    fn a_bad_start_is_refused_with_the_reason() {
        let s = State::default();
        for (body, word) in [
            (json!({"type": "hello"}), "first message"),
            (json!({"type": "start", "wat": 1}), "unknown start field wat"),
            (json!({"type": "start", "mode": "loud"}), "free or relay"),
            (json!({"type": "start", "temperature": 9}), "outside 0 to 2"),
            (json!({"type": "start", "voice": {"id": "voice_nope"}}), "no uploaded voice"),
            (json!({"type": "start", "voice": "AAAA"}), "odd byte count"),
            (json!({"type": "start", "replay": {}}), "allow_replay"),
        ] {
            let e = parse_start(&s, &body, 24000, false).err().unwrap_or_else(|| panic!("{body} was accepted"));
            assert!(e.contains(word), "{body}: {e}");
        }
    }

    #[test]
    fn an_uploaded_voice_at_another_rate_is_refused() {
        let s = State::default();
        s.voices.lock().unwrap().push_back(("voice_a".into(), Arc::new(Voice { samples: vec![0.1; 8], rate: 16000 })));
        let e = parse_start(&s, &json!({"type": "start", "voice": {"id": "voice_a"}}), 24000, false).err().unwrap();
        assert!(e.contains("16000 Hz"), "{e}");
        assert_eq!(parse_start(&s, &json!({"type": "start", "voice": {"id": "voice_a"}}), 16000, false).unwrap().request.options.reference.unwrap().len(), 8);
    }

    #[test]
    fn the_gate_admits_sessions_plus_queue_and_frees_on_drop() {
        let s = State::default();
        let g = s.gate("m", 2, 1);
        let held: Vec<Ticket> = (0..3).map(|_| admit(&g).unwrap()).collect();
        assert!(admit(&g).err().unwrap().contains("sessions_per_gpu"));
        drop(held);
        assert!(admit(&g).is_ok());
    }
}
