use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use serde_json::{Value, json};
use specrt::media::{MediaInfo, parse_size};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

const KEEP_CLIPS: usize = 6;
const KEEP_ROWS: usize = 20;
const THUMB_HEIGHT: usize = 72;
const POLL: Duration = Duration::from_millis(100);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub state: String,
    pub step: usize,
    pub steps: usize,
    pub frames_done: usize,
    pub frames: usize,
    pub queue: Option<usize>,
    pub error: Option<String>,
}

impl Status {
    pub fn parse(v: &Value) -> Status {
        let n = |k: &str| v[k].as_u64().unwrap_or(0) as usize;
        Status { state: v["status"].as_str().unwrap_or("").to_string(), step: n("step"), steps: n("steps"), frames_done: n("frames_decoded"), frames: n("frames"), queue: v["queue_position"].as_u64().map(|q| q as usize), error: v["error"]["message"].as_str().map(String::from) }
    }
}

pub struct Clip {
    pub id: String,
    pub base: String,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
    pub frames: Vec<Vec<u8>>,
    thumbs: Vec<(usize, usize, Vec<u8>)>,
    pub strip: Vec<egui::TextureHandle>,
    pub main: Option<egui::TextureHandle>,
    pub main_frame: Option<usize>,
}

pub enum Update {
    Status(Status),
    Busy(u64),
    Done(Clip),
    Failed(String),
    Cancelled,
}

pub struct VideoRow {
    pub prompt: String,
    pub size: String,
    pub seconds: f64,
    pub fps: u32,
    pub frames: usize,
    pub total_secs: f64,
    pub outcome: String,
    pub seed: u64,
    pub thumb: Option<egui::TextureHandle>,
    pub clip: Option<Clip>,
}

pub struct Running {
    rx: Receiver<Update>,
    cancel: Arc<AtomicBool>,
    pub status: Status,
    started: Instant,
    prompt: String,
    size: String,
    seconds: f64,
    fps: u32,
    seed: u64,
}

pub struct Playback {
    pub frame: usize,
    pub playing: bool,
    acc: f64,
    at: Option<Instant>,
}

impl Playback {
    pub fn new() -> Playback {
        Playback { frame: 0, playing: false, acc: 0.0, at: None }
    }

    pub fn advance(&mut self, dt: f64, fps: u32, count: usize) {
        if !self.playing || count == 0 {
            return;
        }
        self.acc += dt * f64::from(fps.max(1));
        let whole = self.acc.floor();
        self.acc -= whole;
        let next = self.frame + whole as usize;
        if next >= count {
            self.frame = count - 1;
            self.playing = false;
            self.acc = 0.0;
        } else {
            self.frame = next;
        }
    }
}

pub struct VideoState {
    pub prompt: String,
    pub negative: String,
    pub size: String,
    pub seconds: f64,
    pub fps: u32,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    pub lock_seed: bool,
    pub sequential_cfg: bool,
    pub tiled_vae: bool,
    pub running: Option<Running>,
    pub rows: Vec<VideoRow>,
    pub shown: Option<usize>,
    pub reveal: bool,
    pub play: Playback,
    pub error: Option<String>,
    pub for_model: String,
    saving: Option<Receiver<Result<PathBuf, String>>>,
}

impl Default for VideoState {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            size: String::new(),
            seconds: 5.0,
            fps: 16,
            steps: 20,
            guidance: 5.0,
            seed: 1,
            lock_seed: false,
            sequential_cfg: false,
            tiled_vae: false,
            running: None,
            rows: Vec::new(),
            shown: None,
            reveal: false,
            play: Playback { frame: 0, playing: false, acc: 0.0, at: None },
            error: None,
            for_model: String::new(),
            saving: None,
        }
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(600))).build().into()
}

struct Reply {
    status: u16,
    retry_after: Option<u64>,
    body: Vec<u8>,
}

fn call(method: &str, url: &str, body: Option<&Value>) -> Result<Reply, String> {
    let a = agent();
    let result = match (method, body) {
        ("POST", Some(b)) => a.post(url).send_json(b),
        ("DELETE", _) => a.delete(url).call(),
        _ => a.get(url).call(),
    };
    let mut resp = result.map_err(|e| format!("cannot reach the server: {e}"))?;
    let status = resp.status().as_u16();
    let retry_after = resp.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok());
    let body = resp.body_mut().with_config().limit(1 << 30).read_to_vec().map_err(|e| format!("read the reply: {e}"))?;
    Ok(Reply { status, retry_after, body })
}

fn error_text(r: &Reply) -> String {
    let v: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
    v["error"]["message"].as_str().map(String::from).unwrap_or_else(|| format!("the server answered {}", r.status))
}

pub fn busy_text(secs: u64) -> String {
    format!("the server is busy, try again in {secs} {}", if secs == 1 { "second" } else { "seconds" })
}

pub fn download(base: &str, id: &str) -> Result<Vec<u8>, String> {
    let r = call("GET", &format!("{base}/videos/{id}/content"), None)?;
    if r.status == 200 { Ok(r.body) } else { Err(error_text(&r)) }
}

fn thumb(rgb: &[u8], w: usize, h: usize) -> (usize, usize, Vec<u8>) {
    let th = THUMB_HEIGHT.min(h).max(1);
    let tw = (w * th / h.max(1)).max(1);
    let mut out = Vec::with_capacity(tw * th * 3);
    for y in 0..th {
        for x in 0..tw {
            let at = ((y * h / th) * w + x * w / tw) * 3;
            out.extend_from_slice(&rgb[at..at + 3]);
        }
    }
    (tw, th, out)
}

fn finish(base: &str, id: &str, cancel: &AtomicBool, tx: &Sender<Update>) {
    let _ = cancel;
    let result = download(base, id).and_then(|mp4| specrt::mp4::decode(&mp4).map_err(|e| format!("decode the video: {e:#}")));
    let _ = tx.send(match result {
        Ok((head, frames)) => {
            let (w, h) = (head.width as usize, head.height as usize);
            let thumbs = frames.iter().map(|f| thumb(f, w, h)).collect();
            Update::Done(Clip { id: id.to_string(), base: base.to_string(), width: w, height: h, fps: head.fps().round() as u32, frames, thumbs, strip: Vec::new(), main: None, main_frame: None })
        }
        Err(e) => Update::Failed(e),
    });
}

fn work(base: String, body: Value, cancel: Arc<AtomicBool>, tx: Sender<Update>) {
    let created = match call("POST", &format!("{base}/videos"), Some(&body)) {
        Ok(r) if r.status == 429 => {
            let _ = tx.send(Update::Busy(r.retry_after.unwrap_or(1)));
            return;
        }
        Ok(r) if r.status == 200 => serde_json::from_slice::<Value>(&r.body).unwrap_or(Value::Null),
        Ok(r) => {
            let _ = tx.send(Update::Failed(error_text(&r)));
            return;
        }
        Err(e) => {
            let _ = tx.send(Update::Failed(e));
            return;
        }
    };
    let Some(id) = created["id"].as_str().map(String::from) else {
        let _ = tx.send(Update::Failed("the server did not name the video job".into()));
        return;
    };
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = call("DELETE", &format!("{base}/videos/{id}"), None);
            let _ = tx.send(Update::Cancelled);
            return;
        }
        match call("GET", &format!("{base}/videos/{id}"), None) {
            Ok(r) if r.status == 200 => {
                let status = Status::parse(&serde_json::from_slice(&r.body).unwrap_or(Value::Null));
                let state = status.state.clone();
                let error = status.error.clone();
                let _ = tx.send(Update::Status(status));
                match state.as_str() {
                    "completed" => return finish(&base, &id, &cancel, &tx),
                    "failed" => {
                        let _ = tx.send(Update::Failed(error.unwrap_or_else(|| "the video job failed".into())));
                        return;
                    }
                    "cancelled" => {
                        let _ = tx.send(Update::Cancelled);
                        return;
                    }
                    _ => {}
                }
            }
            Ok(r) => {
                let _ = tx.send(Update::Failed(error_text(&r)));
                return;
            }
            Err(e) => {
                let _ = tx.send(Update::Failed(e));
                return;
            }
        }
        std::thread::sleep(POLL);
    }
}

impl VideoState {
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    pub fn is_saving(&self) -> bool {
        self.saving.is_some()
    }

    pub fn follow(&mut self, display: &str, info: &MediaInfo, defaults: &inference_bench_core::settings::ImageDefaults) {
        if self.for_model == display {
            return;
        }
        self.for_model = display.to_string();
        self.size = if info.sizes.contains(&defaults.size) { defaults.size.clone() } else { info.default_size.clone() };
        self.steps = defaults.steps.max(1);
        self.guidance = defaults.guidance;
        self.fps = info.fps.max(1);
        self.seconds = 5.0;
    }

    pub fn frames(&self, info: &MediaInfo) -> usize {
        inference_bench_server::videos::frames_for(self.seconds, self.fps, info.frame_step)
    }

    pub fn body(&self, model: &str) -> Result<Value, String> {
        parse_size(&self.size).ok_or_else(|| format!("size {:?} is not WIDTHxHEIGHT", self.size))?;
        let mut body = json!({"model": model, "prompt": self.prompt, "negative_prompt": self.negative, "seconds": self.seconds, "fps": self.fps, "size": self.size, "steps": self.steps, "guidance": self.guidance, "seed": self.seed});
        if self.sequential_cfg {
            body["sequential_cfg"] = json!(true);
        }
        if self.tiled_vae {
            body["tiled_vae"] = json!(true);
        }
        Ok(body)
    }

    pub fn start(&mut self, base: String, body: Value) {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let c = cancel.clone();
        std::thread::spawn(move || work(base, body, c, tx));
        self.attach(rx, cancel);
    }

    pub fn attach(&mut self, rx: Receiver<Update>, cancel: Arc<AtomicBool>) {
        self.error = None;
        self.running = Some(Running { rx, cancel, status: Status::default(), started: Instant::now(), prompt: self.prompt.clone(), size: self.size.clone(), seconds: self.seconds, fps: self.fps, seed: self.seed });
    }

    pub fn stop(&mut self) {
        if let Some(r) = &self.running {
            r.cancel.store(true, Ordering::SeqCst);
        }
    }

    pub fn pump(&mut self, ctx: &egui::Context) -> Option<(bool, String)> {
        let mut notice = None;
        if let Some(saving) = &self.saving
            && let Ok(done) = saving.try_recv()
        {
            self.saving = None;
            notice = Some(match done {
                Ok(path) => (true, format!("saved {}", path.display())),
                Err(e) => (false, format!("save: {e}")),
            });
        }
        let Some(r) = self.running.as_mut() else { return notice };
        let mut finished: Option<Result<Clip, String>> = None;
        let mut cancelled = false;
        while let Ok(u) = r.rx.try_recv() {
            match u {
                Update::Status(s) => r.status = s,
                Update::Busy(secs) => finished = Some(Err(busy_text(secs))),
                Update::Done(c) => finished = Some(Ok(c)),
                Update::Failed(e) => finished = Some(Err(e)),
                Update::Cancelled => cancelled = true,
            }
        }
        if finished.is_none() && !cancelled {
            return notice;
        }
        let r = self.running.take().unwrap();
        let mut row = VideoRow { prompt: r.prompt, size: r.size, seconds: r.seconds, fps: r.fps, frames: r.status.frames, total_secs: r.started.elapsed().as_secs_f64(), outcome: "stopped".into(), seed: r.seed, thumb: None, clip: None };
        match finished {
            Some(Ok(mut clip)) => {
                row.outcome = "done".into();
                row.frames = clip.frames.len();
                row.fps = clip.fps;
                row.seconds = clip.frames.len() as f64 / f64::from(clip.fps.max(1));
                clip.strip = clip.thumbs.iter().enumerate().map(|(i, (w, h, rgb))| ctx.load_texture(format!("vid-{}-{i}", clip.id), egui::ColorImage::from_rgb([*w, *h], rgb), egui::TextureOptions::LINEAR)).collect();
                row.thumb = clip.strip.first().cloned();
                row.clip = Some(clip);
                if !self.lock_seed {
                    self.seed = self.seed.wrapping_add(1);
                }
            }
            Some(Err(e)) => {
                row.outcome = "failed".into();
                self.error = Some(e);
            }
            None => {}
        }
        let has_clip = row.clip.is_some();
        if has_clip || row.outcome == "stopped" || self.error.is_some() {
            self.rows.push(row);
        }
        if has_clip {
            self.shown = Some(self.rows.len() - 1);
            self.reveal = true;
            self.play = Playback { frame: 0, playing: false, acc: 0.0, at: None };
        }
        let n = self.rows.len();
        if n > KEEP_ROWS {
            self.rows.drain(..n - KEEP_ROWS);
            self.shown = self.shown.and_then(|s| s.checked_sub(n - KEEP_ROWS));
        }
        let n = self.rows.len();
        for row in self.rows.iter_mut().take(n.saturating_sub(KEEP_CLIPS)) {
            row.clip = None;
        }
        notice
    }

    pub fn save(&mut self, dir: &std::path::Path) {
        let Some(row) = self.shown.and_then(|i| self.rows.get(i)) else { return };
        let Some(clip) = &row.clip else { return };
        let (base, id, seed) = (clip.base.clone(), clip.id.clone(), row.seed);
        let dir = dir.join("videos");
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let path = dir.join(format!("video-{}-{seed}.mp4", chrono::Utc::now().format("%Y%m%d-%H%M%S")));
            let _ = tx.send(std::fs::create_dir_all(&dir).map_err(|e| e.to_string()).and_then(|_| download(&base, &id)).and_then(|mp4| std::fs::write(&path, mp4).map(|_| path.clone()).map_err(|e| e.to_string())));
        });
        self.saving = Some(rx);
    }
}

pub fn controls(app: &mut App, ui: &mut egui::Ui, info: &MediaInfo, model: &str) {
    let busy = app.generate_video.busy();
    let g = &mut app.generate_video;
    section(ui, "prompt", "what to film", |ui| {
        prose(ui, &mut g.prompt, "a red fox running through snow", 4);
        ui.label(legend("negative prompt"));
        prose(ui, &mut g.negative, "what to keep out of the clip", 2);
        hint(ui, "Used when guidance is above 1 on a model that runs classifier free guidance.");
    });
    ui.add_space(8.0);
    section(ui, "settings", &format!("{} family", info.family), |ui| {
        row(ui, "size", |ui| {
            let mut sizes = info.sizes.clone();
            if !sizes.contains(&g.size) {
                sizes.insert(0, g.size.clone());
            }
            ui.allocate_ui(egui::vec2(150.0, KEY_H), |ui| {
                choice(ui, "vid-size", &mut g.size, sizes.into_iter().map(|s| (s.clone(), s)));
            });
        });
        row(ui, "seconds", |ui| {
            ui.add(egui::DragValue::new(&mut g.seconds).range(0.5..=60.0).speed(0.1).max_decimals(1));
        });
        row(ui, "fps", |ui| {
            ui.add(egui::DragValue::new(&mut g.fps).range(1..=120).speed(0.2));
        });
        let frames = g.frames(info);
        Line::new().legend("clip").gap(8.0).value(format!("{frames} frames, {:.1} s", frames as f64 / f64::from(g.fps.max(1)))).size(10.5).elided(ui);
        row(ui, "seed", |ui| {
            ui.add(egui::DragValue::new(&mut g.seed).speed(1.0));
            if ui.add(Key::new("dice").outline(VALUE).id_salt("vid-dice")).clicked() {
                g.seed = crate::imagegen::dice();
            }
            ui.checkbox(&mut g.lock_seed, "lock");
        });
        row(ui, "steps", |ui| {
            ui.add(egui::DragValue::new(&mut g.steps).range(1..=200).speed(0.2));
        });
        row(ui, "guidance", |ui| {
            ui.add(egui::DragValue::new(&mut g.guidance).range(0.0..=30.0).speed(0.05).max_decimals(2));
        });
    });
    ui.add_space(8.0);
    section(ui, "memory", "off lets the engine decide from free memory", |ui| {
        switch(ui, "sequential guidance", &mut g.sequential_cfg, "on", "Runs the guided and unguided passes one after the other. Saves about half the activation memory, costs about twice the time per step.");
        switch(ui, "tiled vae", &mut g.tiled_vae, "on", "Decodes each frame in overlapping tiles. Saves decoder memory on large sizes, costs extra decode time and can leave faint seams.");
    });
    ui.add_space(8.0);
    let ready = !g.prompt.trim().is_empty() && !busy;
    let mut go = false;
    ui.horizontal(|ui| {
        if busy {
            if ui.add(Key::new("cancel").outline(FAULT).min_width(120.0).id_salt("vid-stop")).clicked() {
                g.stop();
            }
        } else if ui.add(Key::new("generate video").solid(READOUT).enabled(ready).min_width(140.0).id_salt("vid-go")).clicked() {
            go = true;
        }
    });
    if let Some(r) = &g.running {
        let s = &r.status;
        if s.state == "queued" || s.state.is_empty() {
            let said = match s.queue {
                Some(0) | None => "waiting for the server".to_string(),
                Some(n) => format!("{n} ahead in the queue"),
            };
            Line::new().legend("queue").gap(8.0).value(said).size(11.0).wrapped(ui);
        }
        progress(ui, "step", s.step as f32, Some(s.steps.max(1) as f32), &format!("step {} of {}", s.step, s.steps));
        progress(ui, "frames", s.frames_done as f32, Some(s.frames.max(1) as f32), &format!("frame {} of {}", s.frames_done, s.frames));
    }
    if let Some(e) = &g.error {
        status(ui, false, e);
    }
    if go {
        match g.body(model) {
            Ok(body) => {
                let handle = app.engine.handle();
                match app.server.ensure_agent_endpoint(handle) {
                    Ok(base) => app.generate_video.start(base, body),
                    Err(e) => app.generate_video.error = Some(e),
                }
            }
            Err(e) => app.generate_video.error = Some(e),
        }
    }
}

fn player(app: &mut App, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    let g = &mut app.generate_video;
    let Some(i) = g.shown else { return };
    let Some(row) = g.rows.get_mut(i) else { return };
    let Some(clip) = row.clip.as_mut() else { return };
    let count = clip.frames.len();
    let now = Instant::now();
    if let Some(at) = g.play.at.take().filter(|_| g.play.playing) {
        g.play.advance(now.duration_since(at).as_secs_f64(), clip.fps, count);
    }
    if g.play.playing {
        g.play.at = Some(now);
        ctx.request_repaint_after(Duration::from_millis(15));
    }
    let frame = g.play.frame.min(count.saturating_sub(1));
    if clip.main_frame != Some(frame) {
        let img = egui::ColorImage::from_rgb([clip.width, clip.height], &clip.frames[frame]);
        match &mut clip.main {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => clip.main = Some(ctx.load_texture(format!("vid-main-{}", clip.id), img, egui::TextureOptions::LINEAR)),
        }
        clip.main_frame = Some(frame);
    }
    let top = ui.cursor().min;
    let reveal = std::mem::take(&mut g.reveal);
    let aspect = clip.height as f32 / clip.width.max(1) as f32;
    let w = ui.available_width().min(960.0).min(380.0 / aspect.max(0.01));
    let h = w * aspect;
    if let Some(t) = &clip.main {
        ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(w, h)));
    }
    Line::new().legend("frame").set(format!("{} of {count}", frame + 1)).gap(8.0).value(format!("{}x{} at {} fps", clip.width, clip.height, clip.fps)).size(10.5).elided(ui);
    let mut save = false;
    let mut toggle = false;
    if reveal {
        ui.scroll_to_rect(egui::Rect::from_min_size(top, egui::vec2(10.0, 4.0)), Some(egui::Align::TOP));
    }
    ui.horizontal_wrapped(|ui| {
        if ui.add(Key::new(if g.play.playing { "pause" } else { "play" }).solid(READOUT).min_width(90.0).id_salt("vid-play")).clicked() {
            toggle = true;
        }
        if ui.add(Key::new("save as mp4").outline(VALUE).id_salt("vid-save")).clicked() {
            save = true;
        }
    });
    let mut pick = None;
    let strip_h = THUMB_HEIGHT as f32 + 6.0;
    egui::ScrollArea::horizontal().id_salt("vid-strip").max_height(strip_h + 14.0).auto_shrink([false, true]).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (j, t) in clip.strip.iter().enumerate() {
                let size = t.size_vec2();
                let r = ui.add(egui::Image::new(t).fit_to_exact_size(size).sense(egui::Sense::click()));
                if j == frame {
                    ui.painter().rect_stroke(r.rect, 1.0, egui::Stroke::new(2.0, READOUT), egui::StrokeKind::Inside);
                }
                if r.clicked() {
                    pick = Some(j);
                }
            }
        });
    });
    if toggle {
        if g.play.playing {
            g.play.playing = false;
        } else {
            if g.play.frame + 1 >= count {
                g.play.frame = 0;
            }
            g.play.playing = true;
            g.play.at = Some(Instant::now());
        }
    }
    if let Some(j) = pick {
        g.play.frame = j;
        g.play.playing = false;
    }
    if save {
        let dir = app.dir.clone();
        app.generate_video.save(&dir);
    }
}

fn history(app: &mut App, ui: &mut egui::Ui) {
    let g = &mut app.generate_video;
    if g.rows.is_empty() {
        note(ui, "Generated videos appear here. Each keeps its seed, so the same settings give the same clip.", LEGEND);
        return;
    }
    heading(ui, "recent videos");
    let mut pick = None;
    let shown = g.shown;
    for (i, r) in g.rows.iter().enumerate().rev().take(8) {
        ui.horizontal_top(|ui| {
            match &r.thumb {
                Some(t) => {
                    let resp = ui.add(egui::Image::new(t).fit_to_exact_size(t.size_vec2()).sense(egui::Sense::click()));
                    if resp.clicked() && r.clip.is_some() {
                        pick = Some(i);
                    }
                }
                None => {
                    ui.allocate_exact_size(egui::vec2(48.0, THUMB_HEIGHT as f32), egui::Sense::hover());
                }
            }
            ui.vertical(|ui| {
                let mark = if shown == Some(i) { "showing" } else { "" };
                Line::new().value(&r.prompt).size(11.0).wrapped(ui);
                Line::new().legend(&r.outcome).gap(8.0).value(format!("{} {:.1} s {} fps {:.0} s run {mark}", r.size, r.seconds, r.fps, r.total_secs)).size(10.5).wrapped(ui);
                if r.clip.is_some() && shown != Some(i) && ui.add(Key::new("show").outline(LEGEND).id_salt(("vid-show", i))).clicked() {
                    pick = Some(i);
                }
            });
        });
        ui.add_space(4.0);
    }
    if let Some(i) = pick {
        g.shown = Some(i);
        g.play = Playback { frame: 0, playing: false, acc: 0.0, at: None };
    }
}

pub fn results(app: &mut App, ui: &mut egui::Ui) {
    player(app, ui);
    ui.add_space(8.0);
    history(app, ui);
}
