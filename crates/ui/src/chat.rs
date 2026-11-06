use crate::shell::page;
use crate::App;
use eframe::egui;
use egui_bench::chat as chat_ui;
use egui_bench::prelude::*;
use inference_bench_core::chat::{ChatRequest, Message, Role, Stats};
use inference_bench_core::conversations::{self, Conversation, Turn};
use inference_bench_core::engine::ChatEvent;
use inference_bench_core::library::human_size;
use inference_bench_core::settings::Settings;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

pub struct Streaming {
    rx: Receiver<ChatEvent>,
    pub text: String,
    cancel: Arc<AtomicBool>,
}

pub struct ChatState {
    pub conversations: Vec<Conversation>,
    pub current: usize,
    pub input: String,
    pub streaming: Option<Streaming>,
    pub last_stats: Option<Stats>,
    pub error: Option<String>,
    pub markdown: egui_bench::markdown::Cache,
    pub show_settings: bool,
    pub seed_text: String,
    pub max_tokens_text: String,
}

impl ChatState {
    pub fn load(dir: &Path, settings: &Settings) -> Self {
        let mut conversations = conversations::load_all(dir);
        if conversations.is_empty() {
            let mut c = Conversation::new();
            c.backend = settings.backend_names().first().cloned().unwrap_or_default();
            conversations.push(c);
        }
        Self {
            conversations,
            current: 0,
            input: String::new(),
            streaming: None,
            last_stats: None,
            error: None,
            markdown: egui_bench::markdown::Cache::default(),
            show_settings: true,
            seed_text: String::new(),
            max_tokens_text: String::new(),
        }
    }

    pub fn streaming(&self) -> bool {
        self.streaming.is_some()
    }

    pub fn conv(&mut self) -> &mut Conversation {
        let last = self.conversations.len() - 1;
        &mut self.conversations[self.current.min(last)]
    }

    pub fn pick(&mut self, model: &str, backend: &str) {
        let c = self.conv();
        c.model = model.to_string();
        c.backend = backend.to_string();
    }

    pub fn new_conversation(&mut self) {
        let prev = self.conv().clone();
        let mut c = Conversation::new();
        c.backend = prev.backend;
        c.model = prev.model;
        c.sampling = prev.sampling;
        c.system = prev.system;
        c.use_template = prev.use_template;
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
        let text = std::mem::take(&mut s.text);
        self.streaming = None;
        match outcome {
            Ok(g) => {
                let stats = Stats::of(&g);
                let reply = if g.text.is_empty() { text } else { g.text };
                self.last_stats = Some(stats);
                let c = self.conv();
                c.turns.push(Turn { message: Message::new(Role::Assistant, reply), stats: Some(stats) });
                c.updated = chrono::Utc::now();
                let _ = conversations::save(dir, c);
            }
            Err(e) => {
                let c = self.conv();
                if !text.is_empty() {
                    c.turns.push(Turn { message: Message::new(Role::Assistant, text), stats: None });
                }
                let _ = conversations::save(dir, c);
                self.error = Some(e.to_string());
            }
        }
    }

    pub fn send(&mut self, handle: inference_bench_core::engine::Handle, dir: &Path) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.streaming.is_some() {
            return;
        }
        self.input.clear();
        let c = self.conv();
        c.turns.push(Turn { message: Message::new(Role::User, text), stats: None });
        c.retitle();
        let _ = conversations::save(dir, c);
        self.generate(handle);
    }

    pub fn generate(&mut self, handle: inference_bench_core::engine::Handle) {
        self.error = None;
        let c = self.conv();
        let mut request = ChatRequest::new(c.messages(), c.sampling);
        request.use_template = c.use_template;
        let cancel = Arc::new(AtomicBool::new(false));
        request.cancel = Some(cancel.clone());
        let rx = handle.chat(request);
        self.streaming = Some(Streaming { rx, text: String::new(), cancel });
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

fn top_bar(app: &mut App, ui: &mut egui::Ui) {
    let backends = app.settings.backend_names();
    let (backend, model) = {
        let c = app.chat.conv();
        (c.backend.clone(), c.model.clone())
    };
    let library_backend = app.settings.library_backend(&backend);
    let usable: Vec<String> = app
        .library
        .iter()
        .filter(|m| m.support_for(&library_backend).is_some_and(|s| s.supported))
        .map(|m| m.name.clone())
        .collect();
    let loaded = app.loaded();
    ui.horizontal(|ui| {
        ui.allocate_ui(egui::vec2(130.0, KEY_H), |ui| {
            let mut b = backend.clone();
            if choice(ui, "chat-backend", &mut b, backends.iter().map(|x| (x.clone(), x.clone()))) {
                app.chat.conv().backend = b;
            }
        });
        ui.allocate_ui(egui::vec2(260.0, KEY_H), |ui| {
            let mut m = model.clone();
            let mut options: Vec<(String, String)> = usable.iter().map(|x| (x.clone(), x.clone())).collect();
            if !model.is_empty() && !usable.contains(&model) {
                options.insert(0, (model.clone(), model.clone()));
            }
            if choice(ui, "chat-model", &mut m, options) {
                app.chat.conv().model = m;
            }
        });
        let is_loaded = loaded.as_ref().is_some_and(|l| l.display == model && l.backend == backend);
        if app.loading.is_some() {
            ui.add(Key::new("loading").outline(READOUT).enabled(false).min_width(110.0));
        } else if is_loaded {
            if ui.add(Key::new("unload").outline(FAULT).min_width(100.0)).clicked() {
                app.unload();
            }
        } else {
            let found = app.library.iter().find(|m| m.name == model).cloned();
            if ui.add(Key::new("load").solid(READOUT).enabled(found.is_some()).min_width(100.0)).clicked()
                && let Some(m) = found
            {
                app.load_model(&backend, &m);
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(g) = app.gpu.first() {
                Line::new().legend("vram").measured(format!("{} / {}", human_size(g.used_bytes), human_size(g.total_bytes))).size(11.0).show(ui);
            }
        });
    });
}

fn context_bar(app: &mut App, ui: &mut egui::Ui) {
    let ctx_len = {
        let model = app.chat.conv().model.clone();
        let from_lib = app.library.iter().find(|m| m.name == model).and_then(|m| m.context_length).map(|c| c as usize);
        match app.chat.conv().backend.as_str() {
            "specrt" => app.settings.specrt_ctx,
            _ => from_lib.unwrap_or(8192),
        }
    };
    let used = app.chat.conv().token_estimate();
    chat_ui::context_bar(ui, used, ctx_len);
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

fn stats_strip(ui: &mut egui::Ui, s: &Stats) {
    let mut line = Line::new().legend("ttft").measured(format!("{:.0} ms", s.ttft_secs * 1000.0));
    if let Some(v) = s.prefill_tok_s {
        line = line.gap(12.0).legend("prefill").measured(format!("{v:.0} tok/s"));
    }
    if let Some(v) = s.decode_tok_s {
        line = line.gap(12.0).legend("decode").measured(format!("{v:.1} tok/s"));
    }
    line.gap(12.0).legend("tokens").measured(format!("{} + {}", s.prompt_tokens, s.completion_tokens)).size(10.5).show(ui);
}

fn ui_role(role: Role) -> chat_ui::Role {
    match role {
        Role::System => chat_ui::Role::System,
        Role::User => chat_ui::Role::User,
        Role::Assistant => chat_ui::Role::Assistant,
    }
}

fn sampling_panel(app: &mut App, ui: &mut egui::Ui) {
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
            ui.add(egui::Slider::new(&mut c.sampling.top_p, 0.0..=1.0));
        });
        row(ui, "max tokens", |ui| {
            if max_text.is_empty() {
                *max_text = c.sampling.max_tokens.to_string();
            }
            if field(ui, max_text, "512").changed() {
                if let Ok(v) = max_text.trim().parse::<usize>() {
                    c.sampling.max_tokens = v.clamp(1, 131072);
                }
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
        switch(ui, "chat template", &mut c.use_template, "wrap messages in the model's chat template", "Off sends the text as a raw prompt.");
    });
}

fn conversation_view(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let streaming_text = app.chat.streaming.as_ref().map(|s| s.text.clone());
    let last = app.chat.conversations.len() - 1;
    let cur = app.chat.current.min(last);
    let turns = app.chat.conversations[cur].turns.clone();
    if turns.is_empty() && streaming_text.is_none() {
        chat_ui::welcome(ui, "START A CONVERSATION", "pick a model above, load it, and type below");
    }
    let messages: Vec<chat_ui::Message> = turns.iter().map(|t| chat_ui::Message::new(ui_role(t.message.role), t.message.content.clone())).collect();
    let mut strip = |ui: &mut egui::Ui, i: usize| {
        if let Some(s) = turns.get(i).and_then(|t| t.stats.as_ref()) {
            stats_strip(ui, s);
        }
    };
    let streaming = streaming_text.as_deref().map(|t| if t.is_empty() { "..." } else { t });
    let events = chat_ui::transcript(ui, &mut app.chat.markdown, &messages, streaming, Some(&mut strip));
    if let Some(e) = &app.chat.error {
        note(ui, e.clone(), FAULT);
    }
    for event in events {
        if let chat_ui::Event::Regenerate(_) = event {
            app.chat.regenerate(handle.clone());
        }
    }
}

fn composer(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let ready = app.loaded().is_some();
    let streaming = app.chat.streaming();
    let hint = if ready { "Send a message. Enter sends, Shift+Enter adds a line." } else { "Load a model to chat" };
    for event in chat_ui::ask_box(ui, &mut app.chat.input, hint, streaming, ready) {
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
        .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(8))
        .show_inside(ui, |ui| side_list(app, ui));
    egui::Panel::right("chat-settings")
        .exact_size(330.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(8))
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 8.0;
                sampling_panel(app, ui);
            });
        });
    egui::CentralPanel::no_frame().show_inside(ui, |ui| {
        page(ui, "chat", |ui| {
            top_bar(app, ui);
            ui.add_space(4.0);
            context_bar(app, ui);
            ui.add_space(6.0);
            let h = ui.available_height() - 90.0;
            ui.allocate_ui(egui::vec2(ui.available_width(), h.max(100.0)), |ui| conversation_view(app, ui));
            composer(app, ui);
        });
    });
}
