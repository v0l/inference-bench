use crate::{App, Tab};
use eframe::egui;
use egui_bench::chat as chat_ui;
use egui_bench::prelude::*;
use inference_bench_core::chat::{ChatRequest, Message, Role, Stats};
use inference_bench_core::conversations::{self, Conversation, Turn};
use inference_bench_core::engine::ChatEvent;
use inference_bench_core::library::human_size;
use inference_bench_core::media::{Attachment, MediaKind};
use inference_bench_core::settings::Settings;
use inference_bench_core::target::NATIVE;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};

const READY_HINT: &str = "Send a message. Enter sends, Shift+Enter adds a line.";
const IDLE_HINT: &str = "Load a model to chat";

pub struct Streaming {
    rx: Receiver<ChatEvent>,
    pub text: String,
    pub opens: bool,
    pub folded: bool,
    cancel: Arc<AtomicBool>,
}

pub struct ChatState {
    pub conversations: Vec<Conversation>,
    pub current: usize,
    pub input: String,
    pub streaming: Option<Streaming>,
    pub last_stats: Option<Stats>,
    pub error: Option<String>,
    pub load_error: Option<String>,
    pub markdown: egui_bench::markdown::Cache,
    pub show_settings: bool,
    pub seed_text: String,
    pub max_tokens_text: String,
    pub pending: Vec<Attachment>,
    pub attach_error: Option<String>,
    pub picking: Option<Receiver<Vec<std::path::PathBuf>>>,
    pub thumbs: std::collections::HashMap<String, egui::TextureHandle>,
}

impl ChatState {
    pub fn load(dir: &Path, _settings: &Settings) -> Self {
        let mut conversations = conversations::load_all(dir);
        if conversations.is_empty() {
            conversations.push(Conversation::new());
        }
        Self {
            conversations,
            current: 0,
            input: String::new(),
            streaming: None,
            last_stats: None,
            error: None,
            load_error: None,
            markdown: egui_bench::markdown::Cache::default(),
            show_settings: false,
            seed_text: String::new(),
            max_tokens_text: String::new(),
            pending: Vec::new(),
            attach_error: None,
            picking: None,
            thumbs: Default::default(),
        }
    }

    pub fn streaming(&self) -> bool {
        self.streaming.is_some()
    }

    pub fn conv(&mut self) -> &mut Conversation {
        let last = self.conversations.len() - 1;
        &mut self.conversations[self.current.min(last)]
    }

    pub fn pick(&mut self, model: &str) {
        self.conv().model = model.to_string();
    }

    pub fn new_conversation(&mut self) {
        let prev = self.conv().clone();
        let mut c = Conversation::new();
        c.model = prev.model;
        c.sampling = prev.sampling;
        c.system = prev.system;
        c.use_template = prev.use_template;
        c.thinking = prev.thinking;
        self.conversations.insert(0, c);
        self.current = 0;
        self.error = None;
    }

    pub fn pump(&mut self, dir: &Path) {
        let Some(s) = &mut self.streaming else { return };
        let mut outcome = None;
        while let Ok(ev) = s.rx.try_recv() {
            match ev {
                ChatEvent::Text(t) => s.text.push_str(&t),
                ChatEvent::Done(g) => outcome = Some(Ok(g)),
                ChatEvent::Failed(e) => outcome = Some(Err(e)),
            }
        }
        let Some(outcome) = outcome else { return };
        let opens = s.opens;
        let text = inference_bench_core::reasoning::canonical(&std::mem::take(&mut s.text), opens, true);
        self.streaming = None;
        match outcome {
            Ok(g) => {
                let mut stats = Stats::of(&g);
                let reply = if g.text.is_empty() { text } else { inference_bench_core::reasoning::canonical(&g.text, opens, true) };
                stats.reasoning_tokens = inference_bench_core::reasoning::reasoning_of(&reply).map(|r| inference_bench_core::chat::estimate_tokens(&r) as u64).unwrap_or(0);
                self.last_stats = Some(stats);
                let c = self.conv();
                c.turns.push(Turn { message: Message::new(Role::Assistant, reply), stats: Some(stats), attachments: Vec::new() });
                c.updated = chrono::Utc::now();
                let _ = conversations::save(dir, c);
            }
            Err(e) => {
                let c = self.conv();
                if !text.is_empty() {
                    c.turns.push(Turn { message: Message::new(Role::Assistant, text), stats: None, attachments: Vec::new() });
                }
                let _ = conversations::save(dir, c);
                self.error = Some(e.to_string());
            }
        }
    }

    pub fn attach_bytes(&mut self, name: &str, bytes: &[u8]) {
        match Attachment::from_bytes(name, bytes) {
            Some(a) => {
                self.attach_error = None;
                self.pending.push(a);
            }
            None => self.attach_error = Some(format!("{name} is not an image or a video the app can send")),
        }
    }

    pub fn attach_path(&mut self, path: &Path) {
        match Attachment::from_path(path) {
            Ok(a) => {
                self.attach_error = None;
                self.pending.push(a);
            }
            Err(e) => self.attach_error = Some(e.to_string()),
        }
    }

    pub fn pick_files(&mut self) {
        if self.picking.is_some() {
            return;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let files = rfd::FileDialog::new()
                .set_title("Attach images or video")
                .add_filter("images and video", &["png", "jpg", "jpeg", "webp", "gif", "bmp", "mp4", "mov", "webm", "mkv"])
                .pick_files()
                .unwrap_or_default();
            let _ = tx.send(files);
        });
        self.picking = Some(rx);
    }

    pub fn poll_picker(&mut self) {
        let Some(rx) = &self.picking else { return };
        if let Ok(files) = rx.try_recv() {
            self.picking = None;
            for f in files {
                self.attach_path(&f);
            }
        }
    }

    pub fn send(&mut self, handle: inference_bench_core::engine::Handle, dir: &Path) {
        let text = self.input.trim().to_string();
        if (text.is_empty() && self.pending.is_empty()) || self.streaming.is_some() {
            return;
        }
        self.input.clear();
        let attachments = std::mem::take(&mut self.pending);
        let c = self.conv();
        c.turns.push(Turn { message: Message::new(Role::User, text), stats: None, attachments });
        c.retitle();
        let _ = conversations::save(dir, c);
        self.generate(handle);
    }

    pub fn generate(&mut self, handle: inference_bench_core::engine::Handle) {
        self.error = None;
        let thinking = handle.thinking();
        let c = self.conv();
        let mut request = c.chat_request(thinking);
        let opens = c.use_template && inference_bench_core::reasoning::opens_in_prompt(thinking, c.thinking);
        if let Some(list) = &request.openai_messages {
            request.media = inference_bench_core::media::extract(list, &Default::default()).unwrap_or_default();
        }
        let cancel = Arc::new(AtomicBool::new(false));
        request.cancel = Some(cancel.clone());
        let rx = handle.chat(request);
        self.streaming = Some(Streaming { rx, text: String::new(), opens, folded: false, cancel });
    }

    pub fn stop(&self) {
        if let Some(s) = &self.streaming {
            s.cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn regenerate(&mut self, handle: inference_bench_core::engine::Handle) {
        if self.streaming.is_some() {
            return;
        }
        let c = self.conv();
        if c.turns.last().is_some_and(|t| t.message.role == Role::Assistant) {
            c.turns.pop();
        }
        if !c.turns.is_empty() {
            self.generate(handle);
        }
    }
}

fn usable_models(app: &App) -> Vec<String> {
    app.library
        .iter()
        .filter(|m| m.support_for(NATIVE).is_some_and(|s| s.supported))
        .map(|m| m.name.clone())
        .collect()
}

fn header(app: &mut App, ui: &mut egui::Ui) {
    let busy = app.chat.streaming();
    let mut toggle_settings = false;
    let show = app.chat.show_settings;
    crate::modelbar::draw(app, ui, busy, |ui| {
        let label = if show { "hide settings" } else { "settings" };
        toggle_settings = ui.add(Key::new(label).outline(VALUE).min_width(100.0).id_salt("chat-settings")).clicked();
    });
    if toggle_settings {
        app.chat.show_settings = !app.chat.show_settings;
    }
}

fn context_limit(app: &App) -> usize {
    app.loaded().map_or_else(|| app.settings.default_ctx, |l| l.ctx)
}

fn side_list(app: &mut App, ui: &mut egui::Ui) {
    let entries: Vec<chat_ui::Entry> = app
        .chat
        .conversations
        .iter()
        .map(|c| chat_ui::Entry {
            title: c.title.clone(),
            subtitle: format!("{} messages  {}", c.turns.len(), if c.model.is_empty() { "no model" } else { &c.model }),
        })
        .collect();
    for event in chat_ui::conversations(ui, &entries, app.chat.current) {
        match event {
            chat_ui::Event::New => app.chat.new_conversation(),
            chat_ui::Event::Select(i) => {
                app.chat.current = i;
                app.chat.error = None;
            }
            chat_ui::Event::Delete(i) => {
                let id = app.chat.conversations[i].id.clone();
                conversations::remove(&app.dir, &id);
                app.chat.conversations.remove(i);
                if app.chat.conversations.is_empty() {
                    app.chat.new_conversation();
                }
                app.chat.current = app.chat.current.min(app.chat.conversations.len() - 1);
            }
            _ => {}
        }
    }
}

pub fn stats_line(s: &Stats) -> Line {
    let mut line = Line::new().legend("ttft").measured(format!("{:.0} ms", s.ttft_secs * 1000.0));
    if let Some(v) = s.prefill_tok_s {
        line = line.gap(12.0).legend("prefill").measured(format!("{v:.0} tok/s"));
    }
    if let Some(v) = s.decode_tok_s {
        line = line.gap(12.0).legend("decode").measured(format!("{v:.1} tok/s"));
    }
    if s.reasoning_tokens > 0 {
        line = line.gap(12.0).legend("thinking").measured(format!("~{} tok", s.reasoning_tokens));
    }
    line.gap(12.0).legend("tokens").measured(format!("{} + {}", s.prompt_tokens, s.completion_tokens)).size(10.5)
}

fn ui_role(role: Role) -> chat_ui::Role {
    match role {
        Role::System => chat_ui::Role::System,
        Role::User => chat_ui::Role::User,
        Role::Assistant => chat_ui::Role::Assistant,
    }
}

fn sampling_panel(app: &mut App, ui: &mut egui::Ui) {
    let limit = context_limit(app);
    let st = &mut app.chat;
    let seed_text = &mut st.seed_text;
    let max_text = &mut st.max_tokens_text;
    let last = st.conversations.len() - 1;
    let c = &mut st.conversations[st.current.min(last)];
    section(ui, "system prompt", "sent before every conversation", |ui| {
        prose(ui, &mut c.system, "You are a helpful assistant.", 4);
    });
    section(ui, "sampling", "what you set", |ui| {
        row(ui, "temperature", |ui| {
            ui.add(egui::Slider::new(&mut c.sampling.temperature, 0.0..=2.0));
        });
        row(ui, "top p", |ui| {
            let mut p = c.sampling.top_p.unwrap_or(1.0);
            if ui.add(egui::Slider::new(&mut p, 0.0..=1.0)).changed() {
                c.sampling.top_p = Some(p);
            }
        });
        row(ui, "max tokens", |ui| {
            if max_text.is_empty() && c.sampling.max_tokens != inference_bench_core::chat::UNTIL_FULL {
                *max_text = c.sampling.max_tokens.to_string();
            }
            if field(ui, max_text, "until the context is full").changed() {
                c.sampling.max_tokens = match max_text.trim().parse::<usize>() {
                    Ok(v) => v.clamp(1, limit),
                    Err(_) => inference_bench_core::chat::UNTIL_FULL,
                };
            }
        });
        row(ui, "seed", |ui| {
            if seed_text.is_empty()
                && let Some(s) = c.sampling.seed
            {
                *seed_text = s.to_string();
            }
            if field(ui, seed_text, "random").changed() {
                c.sampling.seed = seed_text.trim().parse().ok();
            }
        });
        switch(ui, "thinking", &mut c.thinking, "let the model think first", "Shown above the answer. Only models whose chat template supports it think.");
        switch(ui, "chat template", &mut c.use_template, "wrap messages in the model's chat template", "Off sends the text as a raw prompt.");
    });
}

fn empty_state(app: &mut App, ui: &mut egui::Ui) {
    let ready = app.loaded().is_some();
    if ready {
        chat_ui::welcome(ui, "Start a conversation", "Type below. Replies stream in with their speed underneath.");
        return;
    }
    if usable_models(app).is_empty() {
        chat_ui::welcome(
            ui,
            IDLE_HINT,
            "No model on disk runs here yet. Download one from the Models page.",
        );
        if ui.add(Key::new("go to models").solid(READOUT).min_width(140.0).id_salt("chat-to-models")).clicked() {
            app.tab = Tab::Models;
        }
    } else {
        chat_ui::welcome(ui, IDLE_HINT, "Select a model above, choose its context length and load it. It stays loaded for every chat until you unload it.");
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ThinkBlock {
    Keep,
    Reset,
    Fold,
}

impl ThinkBlock {
    fn apply(self, ui: &egui::Ui) {
        if self == ThinkBlock::Keep {
            return;
        }
        let id = ui.make_persistent_id(("thinking", usize::MAX));
        let Some(mut state) = egui::collapsing_header::CollapsingState::load(ui.ctx(), id) else { return };
        match self {
            ThinkBlock::Reset => state.remove(ui.ctx()),
            _ => {
                state.set_open(false);
                state.store(ui.ctx());
            }
        }
    }
}

fn conversation_view(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let streaming_text = app.chat.streaming.as_ref().map(|s| inference_bench_core::reasoning::canonical(&s.text, s.opens, false));
    let mut think_block = ThinkBlock::Keep;
    if let Some(s) = app.chat.streaming.as_mut() {
        if s.text.is_empty() {
            think_block = ThinkBlock::Reset;
        } else if !s.folded && inference_bench_core::reasoning::split(&s.text, s.opens, false).answering {
            s.folded = true;
            think_block = ThinkBlock::Fold;
        }
    }
    let last = app.chat.conversations.len() - 1;
    let cur = app.chat.current.min(last);
    let turns = app.chat.conversations[cur].turns.clone();
    let error = app.chat.error.clone();
    let load_error = app.chat.load_error.clone();
    if turns.is_empty() && streaming_text.is_none() {
        empty_state(app, ui);
        if let Some(e) = &load_error {
            chat_ui::fault(ui, "load failed", e);
        }
        return;
    }
    let messages: Vec<chat_ui::Message> =
        turns.iter().map(|t| chat_ui::Message::new(ui_role(t.message.role), t.message.content.clone())).collect();
    let mut strip = |ui: &mut egui::Ui, i: usize| {
        if let Some(s) = turns.get(i).and_then(|t| t.stats.as_ref()) {
            stats_line(s).show(ui);
        }
    };
    let below = |ui: &mut egui::Ui| {
        think_block.apply(ui);
        if let Some(e) = &error {
            chat_ui::fault(ui, "generation failed", e);
        }
        if let Some(e) = &load_error {
            chat_ui::fault(ui, "load failed", e);
        }
    };
    let events = chat_ui::transcript_with(
        ui,
        &mut app.chat.markdown,
        &messages,
        streaming_text.as_deref(),
        Some(&mut strip),
        below,
    );
    for event in events {
        if let chat_ui::Event::Regenerate(_) = event {
            app.chat.regenerate(handle.clone());
        }
    }
}

fn thumbnail(ui: &mut egui::Ui, thumbs: &mut std::collections::HashMap<String, egui::TextureHandle>, a: &Attachment) {
    let key = format!("{}:{}", a.name, a.data.len());
    if a.kind == MediaKind::Image && !thumbs.contains_key(&key) {
        if let Ok(img) = image::load_from_memory(&a.bytes()) {
            let small = img.thumbnail(160, 160).to_rgba8();
            let size = [small.width() as usize, small.height() as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, small.as_raw());
            thumbs.insert(key.clone(), ui.ctx().load_texture(format!("thumb-{key}"), color, Default::default()));
        }
    }
    match thumbs.get(&key) {
        Some(t) => {
            let r = ui.add(egui::Image::new(t).max_height(72.0).corner_radius(4));
            r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Image, true, format!("attachment {}", a.name)));
        }
        None => {
            let label = if a.kind == MediaKind::Video { format!("video {}", a.name) } else { a.name.clone() };
            ui.add(egui::Label::new(egui::RichText::new(label).color(LEGEND)));
        }
    }
}

fn attachment_row(app: &mut App, ui: &mut egui::Ui) {
    let mut remove = None;
    if let Some(e) = &app.chat.attach_error {
        chat_ui::fault(ui, "attachment", e);
    }
    if app.chat.pending.is_empty() {
        return;
    }
    let pending = app.chat.pending.clone();
    ui.horizontal_wrapped(|ui| {
        for (i, a) in pending.iter().enumerate() {
            ui.vertical(|ui| {
                thumbnail(ui, &mut app.chat.thumbs, a);
                if ui.add(Key::new("remove").outline(LEGEND).id_salt(("remove-attachment", i))).clicked() {
                    remove = Some(i);
                }
            });
        }
    });
    if let Some(i) = remove {
        app.chat.pending.remove(i);
    }
}

fn collect_input(app: &mut App, ui: &egui::Ui) {
    let (dropped, pasted) = ui.input(|i| {
        let dropped: Vec<_> = i.raw.dropped_files.clone();
        let pasted: Vec<_> = i
            .events
            .iter()
            .filter_map(|e| match e {
                egui::Event::PasteImage(img) => Some(img.clone()),
                _ => None,
            })
            .collect();
        (dropped, pasted)
    });
    for f in dropped {
        match f.bytes() {
            Ok(bytes) => {
                let name = f.path().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                app.chat.attach_bytes(&name, &bytes);
            }
            Err(e) => app.chat.attach_error = Some(e),
        }
    }
    for img in pasted {
        let mut png = Vec::new();
        let rgba: Vec<u8> = img.pixels.iter().flat_map(|p| p.to_array()).collect();
        if let Some(buf) = image::RgbaImage::from_raw(img.size[0] as u32, img.size[1] as u32, rgba) {
            let _ = image::DynamicImage::ImageRgba8(buf).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png);
            app.chat.attach_bytes("pasted.png", &png);
        }
    }
}

fn composer(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let ready = app.loaded().is_some();
    app.chat.poll_picker();
    collect_input(app, ui);
    attachment_row(app, ui);
    let vision = app.engine.vision();
    ui.horizontal(|ui| {
        let attach = ui.add_enabled(ready && vision, Key::new("attach").outline(VALUE).min_width(90.0).id_salt("attach"));
        if attach.clicked() {
            app.chat.pick_files();
        }
        if ready && !vision {
            ui.label(egui::RichText::new("this model takes text only").size(10.5).color(LEGEND));
        }
        if ready && app.engine.thinking() != specrt::chat::Thinking::Unsupported {
            ui.checkbox(&mut app.chat.conv().thinking, "thinking");
        }
    });
    let streaming = app.chat.streaming();
    let hint = if ready { READY_HINT } else { IDLE_HINT };
    let used = app.chat.conv().token_estimate();
    let limit = context_limit(app);
    chat_ui::context_bar(ui, used, limit);
    let events = chat_ui::ask_box(ui, &mut app.chat.input, hint, streaming, ready);
    let _ = &events;
    for event in events {
        match event {
            chat_ui::Event::Stop => app.chat.stop(),
            chat_ui::Event::Sent(text) => {
                app.chat.input = text;
                let dir = app.dir.clone();
                app.chat.send(handle.clone(), &dir);
            }
            _ => {}
        }
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    egui::Panel::left("chats")
        .exact_size(230.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(WELL).inner_margin(8))
        .show_inside(ui, |ui| side_list(app, ui));
    if app.chat.show_settings {
        egui::Panel::right("chat-settings")
            .exact_size(330.0)
            .resizable(false)
            .frame(egui::Frame::NONE.fill(WELL).inner_margin(8))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;
                    sampling_panel(app, ui);
                });
            });
    }
    let frame = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(16, 10));
    egui::Panel::top("chat-header").frame(frame).show_inside(ui, |ui| header(app, ui));
    egui::Panel::bottom("chat-composer").frame(frame).show_inside(ui, |ui| composer(app, ui));
    egui::CentralPanel::no_frame().show_inside(ui, |ui| {
        egui::Frame::NONE.inner_margin(egui::Margin::symmetric(16, 0)).show(ui, |ui| {
            ui.set_max_width(ui.available_width().min(900.0));
            conversation_view(app, ui);
        });
    });
}
