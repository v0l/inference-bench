use crate::{App, Tab};
use agent_core::config::Config;
use agent_core::skillset::{Skill, listing};
use agent_core::{Backend, Welcome, async_trait};
use agent_ui::{Agent, Handled, Host};
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::engine::Loaded;
use inference_bench_core::live::RequestRecord;
use inference_bench_core::target::NATIVE;
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};

const IDLE_HINT: &str = "Load a model to chat";

#[derive(Default)]
pub struct Bench {
    model: Arc<Mutex<String>>,
}

#[async_trait]
impl Backend for Bench {
    fn app(&self) -> &'static str {
        "inference-bench"
    }

    fn user_agent(&self) -> String {
        concat!("inference-bench/", env!("CARGO_PKG_VERSION")).to_string()
    }

    fn brief(&self, workspace: &Path, skills: &[Skill]) -> String {
        let model = self.model.lock().map(|m| m.clone()).unwrap_or_default();
        format!(
            "You are the coding agent inside inference-bench, a local LLM server and desktop app. \
You run on the model {model}, loaded in this app and served by its own engine on this machine.\n\n\
{}\n\nThe workspace is {}.\n\nSkills installed now, for load_skill:\n{}\n",
            agent_core::tools::GUIDE,
            workspace.display(),
            listing(skills)
        )
    }

    fn welcome(&self) -> Welcome {
        Welcome {
            kicker: "local agent".into(),
            title: "Ask the model to work".into(),
            text: "The agent runs on the model loaded in this app. It reads and writes files in its workspace, runs commands in a sandbox and asks before it touches any other folder.".into(),
            placeholder: "Ask for something. Enter sends, shift-enter breaks the line.".into(),
        }
    }
}

pub struct ChatState {
    pub agent: Agent<App>,
    pub load_error: Option<String>,
    pub attempted: String,
    served: Arc<Mutex<String>>,
}

impl ChatState {
    pub fn new(ctx: &egui::Context) -> Self {
        let backend = Bench::default();
        let served = backend.model.clone();
        Self { agent: Agent::new(ctx, Arc::new(backend), true), load_error: None, attempted: String::new(), served }
    }

    pub fn streaming(&self) -> bool {
        self.agent.live()
    }

    pub fn follow(&mut self, endpoint: &str, loaded: Option<&Loaded>) {
        follow(&mut self.agent.chat.config, endpoint, loaded);
        if let Ok(mut m) = self.served.lock() {
            *m = self.agent.chat.config.model.clone();
        }
    }
}

pub fn follow(config: &mut Config, endpoint: &str, loaded: Option<&Loaded>) {
    match loaded {
        Some(l) => {
            config.url = endpoint.to_string();
            config.model = l.display.clone();
        }
        None => config.model.clear(),
    }
}

pub fn last_request(app: &App) -> Option<RequestRecord> {
    app.engine.handle().recent_requests().last().cloned()
}

impl Host for App {
    fn agent(&mut self) -> &mut Agent<App> {
        &mut self.chat.agent
    }

    fn call(&mut self, _ctx: &egui::Context, name: &str, _args: &Value) -> Handled<App> {
        Handled::Now(Err(format!("no tool called {name}")))
    }
}

fn usable_models(app: &App) -> usize {
    app.library.iter().filter(|m| m.support_for(NATIVE).is_some_and(|s| s.supported)).count()
}

fn sync(app: &mut App) {
    let loaded = app.loaded();
    let wanted = loaded.is_some();
    let endpoint = match wanted {
        true => match app.server.ensure_agent_endpoint(app.engine.handle()) {
            Ok(e) => e,
            Err(e) => {
                app.chat.load_error = Some(e);
                String::new()
            }
        },
        false => String::new(),
    };
    app.chat.follow(&endpoint, loaded.as_ref());
}

fn header(app: &mut App, ui: &mut egui::Ui) {
    let busy = app.chat.streaming();
    crate::modelbar::draw(app, ui, busy, |_| {});
}

fn empty_state(app: &mut App, ui: &mut egui::Ui) {
    if let (Some(name), Some(p)) = (&app.loading, &app.load_progress) {
        welcome(ui, &format!("Loading {name}"), &crate::modelbar::progress_text(p));
        ui.vertical_centered(|ui| crate::modelbar::load_bar(ui, p, 420.0));
        return;
    }
    if usable_models(app) == 0 {
        welcome(ui, IDLE_HINT, "No model on disk runs here yet. Download one from the Models page.");
        if ui.add(Key::new("go to models").solid(READOUT).min_width(140.0).id_salt("chat-to-models")).clicked() {
            app.tab = Tab::Models;
        }
    } else {
        welcome(ui, IDLE_HINT, "Select a model above, choose its context length and load it. It stays loaded for every chat until you unload it.");
    }
    if let Some(e) = &app.chat.load_error {
        status(ui, false, e);
    }
}

fn welcome(ui: &mut egui::Ui, title: &str, text: &str) {
    ui.add_space(40.0);
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(title).font(legend_font(16.0)).color(VALUE));
        ui.add_space(6.0);
        note(ui, text, LEGEND);
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    sync(app);
    let frame = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(16, 10));
    egui::Panel::top("chat-header").frame(frame).show_inside(ui, |ui| header(app, ui));
    egui::CentralPanel::no_frame().show_inside(ui, |ui| {
        if app.loaded().is_none() {
            empty_state(app, ui);
            return;
        }
        app.chat.agent.show(ui);
    });
}
