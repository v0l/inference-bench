use crate::App;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::engine::Handle;
use inference_bench_core::logging::LogBuffer;
use inference_bench_server::config::{ModelConfig, ServerConfig};
use inference_bench_server::engine_backend::EngineBackend;
use inference_bench_server::{Server, ServerHandle};
use tracing::Level;

pub struct ServerState {
    pub running: Option<ServerHandle>,
    pub port_text: String,
    pub error: Option<String>,
    pub shown: usize,
    runtime: tokio::runtime::Runtime,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            running: None,
            port_text: String::new(),
            error: None,
            shown: 0,
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("tokio runtime"),
        }
    }
}

impl ServerState {
    pub fn start(&mut self, engine: Handle, port: u16, key: &str, lan: bool) {
        let mut config = ServerConfig {
            bind: format!("{}:{port}", if lan { "0.0.0.0" } else { "127.0.0.1" }),
            shutdown_grace_secs: 2,
            specs: inference_bench_server::config::SpecsSection { auto_update: false, ..Default::default() },
            ..ServerConfig::default()
        };
        if !key.trim().is_empty() {
            config.api_keys = vec![key.trim().to_string()];
        }
        if let Some(l) = engine.loaded() {
            config.models = vec![ModelConfig { id: l.display.clone(), ..ModelConfig::default() }];
            config.default_model = Some(l.display);
        }
        let factory = move |_: Option<usize>| {
            Ok(Box::new(EngineBackend::new(engine.clone())) as Box<dyn inference_bench_core::target::Target>)
        };
        let started = self.runtime.block_on(Server::new(config).with_factory(factory).with_gpu_probe(std::sync::Arc::new(|| None)).start());
        match started {
            Ok(h) => {
                self.running = Some(h);
                self.error = None;
            }
            Err(e) => self.error = Some(format!("cannot listen on port {port}: {e:#}")),
        }
    }

    pub fn stop(&mut self) {
        if let Some(h) = self.running.take() {
            h.shutdown();
        }
    }
}

struct RequestLine {
    time: String,
    request: String,
    status: u16,
    ms: String,
}

fn request_lines(logs: &LogBuffer) -> Vec<RequestLine> {
    logs.snapshot(Level::INFO, "request finished")
        .iter()
        .rev()
        .map(|e| RequestLine {
            time: e.time.with_timezone(&chrono::Local).format("%H:%M:%S").to_string(),
            request: format!("{} {}", e.field("method").unwrap_or("?"), e.field("path").unwrap_or("?")),
            status: e.field("status").and_then(|s| s.parse().ok()).unwrap_or(0),
            ms: e.field("elapsed_ms").and_then(|s| s.parse::<f64>().ok()).map(|v| format!("{v:.0}")).unwrap_or_default(),
        })
        .collect()
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    page(ui, "server", |ui| {
        let running = app.server.running.is_some();
        let loaded = app.loaded();
        if app.server.port_text.is_empty() {
            app.server.port_text = app.settings.server.port.to_string();
        }
        card(
            ui,
            Some(if running { OK } else { ETCH }),
            |ui| {
                Line::new().legend("local server").show(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    lamp(ui, if running { "running" } else { "stopped" }, running, false);
                });
            },
            |ui| {
                match &app.server.running {
                    Some(s) => {
                        Line::new().legend("address").measured(format!("{}/v1", s.url())).show(ui);
                    }
                    None => {
                        Line::new().legend("address").note("not listening").show(ui);
                    }
                }
                match &loaded {
                    Some(l) => Line::new().legend("serving").value(l.display.clone()).show(ui),
                    None => Line::new().legend("serving").note("nothing loaded: load a model in chat or models").show(ui),
                };
                ui.horizontal(|ui| {
                    if running {
                        if ui.add(Key::new("stop server").solid(FAULT).min_width(140.0)).clicked() {
                            app.server.stop();
                        }
                    } else {
                        let port = app.server.port_text.trim().parse::<u16>().ok();
                        if ui.add(Key::new("start server").solid(READOUT).enabled(port.is_some()).min_width(140.0)).clicked()
                            && let Some(p) = port
                        {
                            app.settings.server.port = p;
                            app.server.start(app.engine.handle(), p, &app.settings.server.api_key, app.settings.server.lan);
                        }
                    }
                });
                if let Some(e) = &app.server.error {
                    status(ui, false, e);
                }
            },
        );
        section(ui, "settings", "applied the next time the server starts", |ui| {
            row(ui, "port", |ui| {
                field(ui, &mut app.server.port_text, "1234");
            });
            row(ui, "api key", |ui| {
                secret(ui, &mut app.settings.server.api_key);
            });
            switch(ui, "allow other machines", &mut app.settings.server.lan, "listen on all interfaces", "Off keeps it on 127.0.0.1.");
        });
        if let Some(s) = &app.server.running {
            let url = s.url();
            section(ui, "use it", "any OpenAI-compatible client works", |ui| {
                note(ui, format!("curl {url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'"), LEGEND);
                note(ui, format!("Metrics at {url}/metrics. The model is the one loaded when the server started; restart it after loading another."), LEGEND);
            });
        }
        let log = request_lines(&app.logs);
        section(ui, "request log", "newest first, full detail in the logs tab", |ui| {
            if log.is_empty() {
                note(ui, "no requests yet", LEGEND);
                return;
            }
            let cols: [(&str, f32); 4] = [("time", 90.0), ("request", 280.0), ("status", 70.0), ("ms", 80.0)];
            Table::new(&cols, log.len()).show(ui, |i, p, rect, at| {
                let e = &log[i];
                let tint = if e.status < 400 { OK } else { FAULT };
                cell(p, rect, at(0), 90.0, &e.time, LEGEND);
                cell(p, rect, at(1), 280.0, &e.request, VALUE);
                cell(p, rect, at(2), 70.0, &e.status.to_string(), tint);
                cell(p, rect, at(3), 80.0, &e.ms, TRACE);
            });
        });
    });
}
