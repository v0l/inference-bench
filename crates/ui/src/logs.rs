use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::logging::LogEntry;
use tracing::Level;

pub struct LogsState {
    pub level: Level,
    pub search: String,
    pub follow: bool,
}

impl Default for LogsState {
    fn default() -> Self {
        Self { level: Level::INFO, search: String::new(), follow: true }
    }
}

const LEVELS: [(&str, Level); 5] =
    [("error", Level::ERROR), ("warn", Level::WARN), ("info", Level::INFO), ("debug", Level::DEBUG), ("trace", Level::TRACE)];

fn tint(level: Level) -> egui::Color32 {
    match level {
        Level::ERROR => FAULT,
        Level::WARN => WARN,
        Level::INFO => VALUE,
        _ => LEGEND,
    }
}

pub fn save_path(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join(format!("logs-{}.txt", chrono::Local::now().format("%Y%m%d-%H%M%S")))
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    {
        let mut copy = false;
        let mut save = false;
        ui.horizontal(|ui| {
            for (name, level) in LEVELS {
                let on = app.logs_ui.level == level;
                if chip(ui, name, if on { READOUT } else { LEGEND }, on).interact(egui::Sense::click()).clicked() {
                    app.logs_ui.level = level;
                }
            }
            ui.add_space(8.0);
            field_then(ui, &mut app.logs_ui.search, "search", 330.0, |ui| {
                copy = ui.add(Key::new("copy").outline(VALUE).min_width(70.0)).clicked();
                save = ui.add(Key::new("save").outline(VALUE).min_width(70.0)).clicked();
                if ui.add(Key::new("clear").outline(FAULT).min_width(70.0)).clicked() {
                    app.logs.clear();
                }
            });
        });
        let entries: Vec<LogEntry> = app.logs.snapshot(app.logs_ui.level, &app.logs_ui.search);
        if copy {
            let text = app.logs.text(app.logs_ui.level, &app.logs_ui.search);
            ui.ctx().copy_text(text);
            app.notice = Some((true, format!("copied {} log lines", entries.len())));
        }
        if save {
            let path = save_path(&app.dir);
            app.notice = Some(match std::fs::create_dir_all(&app.dir).map_err(anyhow::Error::from).and_then(|_| app.logs.save(&path, app.logs_ui.level, &app.logs_ui.search)) {
                Ok(()) => (true, format!("saved {}", path.display())),
                Err(e) => (false, format!("{e:#}")),
            });
        }
        note(ui, format!("{} lines shown of {} kept", entries.len(), app.logs.len()), LEGEND);
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(app.logs_ui.follow).show_rows(ui, 16.0, entries.len(), |ui, range| {
            for e in &entries[range] {
                ui.add(egui::Label::new(egui::RichText::new(e.line()).monospace().size(11.5).color(tint(e.level))).truncate());
            }
        });
    }
}
