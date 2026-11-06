use crate::App;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::server::Server;

#[derive(Default)]
pub struct ServerState {
    pub running: Option<Server>,
    pub port_text: String,
    pub error: Option<String>,
    pub shown: usize,
}

impl ServerState {
    pub fn start(&mut self, app_handle: inference_bench_core::engine::Handle, port: u16, key: &str, lan: bool, wake: inference_bench_core::engine::Wake) {
        match Server::start(app_handle, port, key, lan, wake) {
            Ok(s) => {
                self.running = Some(s);
                self.error = None;
            }
            Err(e) => self.error = Some(format!("cannot listen on port {port}: {e}")),
        }
    }

    pub fn stop(&mut self) {
        self.running = None;
    }
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
                    Some(l) => Line::new().legend("serving").value(format!("{} on {}", l.display, l.backend)).show(ui),
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
                            app.server.start(app.engine.handle(), p, &app.settings.server.api_key, app.settings.server.lan, app.wake.clone());
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
                note(ui, format!("curl {url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"x\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'"), LEGEND);
                note(ui, "In the bench tab, add an endpoint in settings pointing at this address to benchmark it.", LEGEND);
            });
        }
        let log: Vec<_> = app.server.running.as_ref().map(|s| s.log.lock().unwrap().clone()).unwrap_or_default();
        section(ui, "request log", "newest first", |ui| {
            if log.is_empty() {
                note(ui, "no requests yet", LEGEND);
                return;
            }
            let cols: [(&str, f32); 6] = [("time", 90.0), ("request", 230.0), ("status", 70.0), ("ms", 80.0), ("ttft", 80.0), ("decode", 110.0)];
            let rows: Vec<_> = log.iter().rev().collect();
            Table::new(&cols, rows.len()).show(ui, |i, p, rect, at| {
                let e = rows[i];
                let tint = if e.status < 400 { OK } else { FAULT };
                cell(p, rect, at(0), 90.0, &e.when.format("%H:%M:%S").to_string(), LEGEND);
                cell(p, rect, at(1), 230.0, &format!("{} {}", e.method, e.path), VALUE);
                cell(p, rect, at(2), 70.0, &e.status.to_string(), tint);
                cell(p, rect, at(3), 80.0, &e.millis.to_string(), TRACE);
                cell(p, rect, at(4), 80.0, &e.stats.map(|s| format!("{:.0} ms", s.ttft_secs * 1000.0)).unwrap_or("-".into()), TRACE);
                cell(p, rect, at(5), 110.0, &e.stats.and_then(|s| s.decode_tok_s).map(|v| format!("{v:.1} t/s")).unwrap_or("-".into()), TRACE);
            });
        });
    });
}
