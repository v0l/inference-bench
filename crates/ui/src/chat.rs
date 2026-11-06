use crate::shell::page;
use crate::App;
use eframe::egui;
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
        ui.allocate_ui(egui::vec2(150.0, KEY_H), |ui| {
            let mut b = backend.clone();
            if choice(ui, "chat-backend", &mut b, backends.iter().map(|x| (x.clone(), x.clone()))) {
                app.chat.conv().backend = b;
            }
        });
        ui.allocate_ui(egui::vec2(330.0, KEY_H), |ui| {
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
            if ui.add(Key::new("unload").outline(FAULT).min_width(110.0)).clicked() {
                app.unload();
            }
        } else {
            let found = app.library.iter().find(|m| m.name == model).cloned();
            if ui.add(Key::new("load").solid(READOUT).enabled(found.is_some()).min_width(110.0)).clicked()
                && let Some(m) = found
            {
                app.load_model(&backend, &m);
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(g) = app.gpu.first() {
                Line::new().legend("memory").measured(format!("{} / {}", human_size(g.used_bytes), human_size(g.total_bytes))).size(11.0).show(ui);
            }
            if let Some(l) = &loaded {
                ui.add_space(12.0);
                Line::new().legend("loaded in").measured(format!("{:.1} s", l.load_secs)).size(11.0).show(ui);
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
    let frac = used as f32 / ctx_len.max(1) as f32;
    ui.horizontal(|ui| {
        Line::new().legend("context").measured(format!("{used} / {ctx_len}")).size(11.0).show(ui);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 10.0), egui::Sense::hover());
        bar(ui.painter(), rect, frac, if frac > 0.85 { WARN } else { TRACE });
    });
}

fn side_list(app: &mut App, ui: &mut egui::Ui) {
    if ui.add(Key::new("new chat").solid(READOUT).min_width(ui.available_width())).clicked() {
        app.chat.new_conversation();
    }
    ui.add_space(6.0);
    let mut remove = None;
    let mut select = None;
    let current = app.chat.current;
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for (i, c) in app.chat.conversations.iter().enumerate() {
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 44.0), egui::Sense::click());
            let p = ui.painter();
            let on = i == current;
            if on {
                p.rect_filled(rect, 2.0, PANEL);
                p.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(3.0, rect.height())), 0.0, READOUT);
            } else if resp.hovered() {
                p.rect_filled(rect, 2.0, BAND);
            }
            p.text(rect.left_top() + egui::vec2(12.0, 12.0), egui::Align2::LEFT_CENTER, &c.title, legend_font(12.0), if on { VALUE } else { LEGEND });
            let sub = format!("{} messages  {}", c.turns.len(), if c.model.is_empty() { "no model" } else { &c.model });
            p.text(rect.left_top() + egui::vec2(12.0, 30.0), egui::Align2::LEFT_CENTER, sub, figure(10.0), LEGEND);
            if resp.clicked() {
                select = Some(i);
            }
            resp.context_menu(|ui| {
                if ui.add(Key::new("delete chat").outline(FAULT).min_width(ui.available_width())).clicked() {
                    remove = Some(i);
                    ui.close();
                }
            });
        }
    });
    if let Some(i) = select {
        app.chat.current = i;
        app.chat.error = None;
    }
    if let Some(i) = remove {
        let id = app.chat.conversations[i].id.clone();
        conversations::remove(&app.dir, &id);
        app.chat.conversations.remove(i);
        if app.chat.conversations.is_empty() {
            app.chat.new_conversation();
        }
        app.chat.current = app.chat.current.min(app.chat.conversations.len() - 1);
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

fn bubble(ui: &mut egui::Ui, cache: &mut egui_bench::markdown::Cache, role: Role, text: &str, salt: usize, stats: Option<&Stats>) -> (bool, bool) {
    let (mut copy, mut regen) = (false, false);
    let user = role == Role::User;
    let rail = if user { READOUT } else { TRACE };
    egui::Frame::NONE.fill(if user { PANEL } else { CHASSIS }).stroke(egui::Stroke::new(1.0, ETCH)).corner_radius(2).inner_margin(egui::Margin::symmetric(12, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(egui::RichText::new(if user { "YOU" } else { "ASSISTANT" }).font(legend_font(10.5)).extra_letter_spacing(1.5).color(rail));
        ui.add_space(2.0);
        if user {
            ui.label(egui::RichText::new(text).font(figure(13.0)).color(VALUE));
        } else {
            egui_bench::markdown::show(ui, cache, text, salt);
        }
        if !user {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if let Some(s) = stats {
                    stats_strip(ui, s);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    copy = ui.add(Key::new("copy").outline(VALUE).id_salt(("copy", salt))).clicked();
                    regen = ui.add(Key::new("regenerate").outline(VALUE).id_salt(("regen", salt))).clicked();
                });
            });
        }
    });
    (copy, regen)
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
    let mut regen = false;
    let mut copy_text = None;
    let streaming_text = app.chat.streaming.as_ref().map(|s| s.text.clone());
    let n = app.chat.conv().turns.len();
    let last = app.chat.conversations.len() - 1;
    let cur = app.chat.current.min(last);
    egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 10.0;
        if n == 0 && streaming_text.is_none() {
            ui.add_space(80.0);
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("START A CONVERSATION").font(legend_font(18.0)).extra_letter_spacing(3.0).color(LEGEND));
                note(ui, "pick a model above, load it, and type below", LEGEND);
            });
        }
        let turns = app.chat.conversations[cur].turns.clone();
        for (i, t) in turns.iter().enumerate() {
            let is_last = i + 1 == turns.len();
            let (c, r) = bubble(ui, &mut app.chat.markdown, t.message.role, &t.message.content, i, t.stats.as_ref());
            if c {
                copy_text = Some(t.message.content.clone());
            }
            if r && is_last {
                regen = true;
            }
        }
        if let Some(text) = &streaming_text {
            bubble(ui, &mut app.chat.markdown, Role::Assistant, if text.is_empty() { "..." } else { text }, 10_000, None);
        }
        if let Some(e) = &app.chat.error {
            note(ui, e.clone(), FAULT);
        }
    });
    if let Some(t) = copy_text {
        ui.ctx().copy_text(t);
    }
    if regen {
        app.chat.regenerate(handle);
    }
}

fn composer(app: &mut App, ui: &mut egui::Ui) {
    let handle = app.engine.handle();
    let ready = app.loaded().is_some();
    let streaming = app.chat.streaming();
    let mut send = false;
    ui.horizontal(|ui| {
        let w = ui.available_width() - 120.0;
        let r = ui.add_sized(
            [w, 56.0],
            egui::TextEdit::multiline(&mut app.chat.input)
                .hint_text(if ready { "Send a message. Enter sends, Shift+Enter adds a line." } else { "Load a model to chat" })
                .desired_rows(2)
                .font(figure(13.0)),
        );
        let enter = r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
        if enter {
            if let Some(i) = app.chat.input.rfind('\n') {
                if i + 1 == app.chat.input.len() {
                    app.chat.input.pop();
                }
            }
            send = true;
        }
        ui.vertical(|ui| {
            if streaming {
                if ui.add(Key::new("stop").solid(FAULT).min_width(110.0)).clicked() {
                    app.chat.stop();
                }
            } else if ui.add(Key::new("send").solid(READOUT).enabled(ready).min_width(110.0)).clicked() {
                send = true;
            }
        });
    });
    if send && ready {
        let dir = app.dir.clone();
        app.chat.send(handle, &dir);
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    egui::Panel::left("chats")
        .exact_size(230.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(8))
        .show_inside(ui, |ui| side_list(app, ui));
    egui::Panel::right("chat-settings")
        .exact_size(290.0)
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
