mod chart;
mod views;

use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::backend::Request;
use inference_bench_core::matrix::{MatrixSpec, run_matrix};
use inference_bench_core::result::RunResult;
use inference_bench_core::report::Metric;
use inference_bench_core::store;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Matrix,
    History,
    Charts,
}

pub struct App {
    pub dir: PathBuf,
    pub runs: Vec<RunResult>,
    pub tab: Tab,
    pub metric: Metric,
    pub matrix_path: String,
    pub shown_run: usize,
    pub notice: Option<(bool, String)>,
    running: Option<Receiver<Progress>>,
    pub progress: Vec<RunResult>,
}

enum Progress {
    Cell(RunResult),
    Done,
    Failed(String),
}

impl App {
    pub fn new(dir: PathBuf, matrix_path: String) -> Self {
        let runs = store::load_all(&dir);
        Self {
            dir,
            shown_run: runs.len().saturating_sub(1),
            runs,
            tab: Tab::Matrix,
            metric: Metric::DecodeTokS,
            matrix_path,
            notice: None,
            running: None,
            progress: Vec::new(),
        }
    }

    pub fn reload(&mut self) {
        self.runs = store::load_all(&self.dir);
        self.shown_run = self.runs.len().saturating_sub(1);
    }

    pub fn running(&self) -> bool {
        self.running.is_some()
    }

    pub fn start_matrix(&mut self) {
        let path = PathBuf::from(self.matrix_path.trim());
        let dir = self.dir.clone();
        let (tx, rx) = channel();
        let loaded = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}", path.display()))
            .and_then(|t| MatrixSpec::parse(&t))
            .and_then(|spec| {
                let base = path.parent().map(PathBuf::from).unwrap_or_default();
                let prompt = spec.prompt_text(&base)?;
                Ok((spec, prompt))
            });
        let (spec, prompt) = match loaded {
            Ok(v) => v,
            Err(e) => {
                self.notice = Some((false, e));
                return;
            }
        };
        self.progress.clear();
        self.notice = Some((true, format!("running {} cells", spec.backends.len() * spec.models.len())));
        self.running = Some(rx);
        std::thread::spawn(move || {
            let request = Request { prompt, max_tokens: spec.max_tokens };
            run_matrix(&spec, &request, &mut |r| {
                if let Err(e) = store::save(&dir, &r) {
                    let _ = tx.send(Progress::Failed(e.to_string()));
                }
                let _ = tx.send(Progress::Cell(r));
            });
            let _ = tx.send(Progress::Done);
        });
    }

    fn drain(&mut self) {
        let Some(rx) = &self.running else { return };
        let mut finished = false;
        while let Ok(p) = rx.try_recv() {
            match p {
                Progress::Cell(r) => self.progress.push(r),
                Progress::Failed(e) => self.notice = Some((false, e)),
                Progress::Done => finished = true,
            }
        }
        if finished {
            self.running = None;
            self.reload();
            let failed = self.progress.iter().filter(|r| r.error.is_some()).count();
            self.notice = Some((failed == 0, format!("{} cells run, {failed} failed", self.progress.len())));
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain();
        if self.running() {
            root.ctx().request_repaint_after(Duration::from_millis(200));
        }
        egui::Panel::top("header")
            .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin {
                left: 12,
                right: 12,
                top: 8,
                bottom: 6,
            }))
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("INFERENCE BENCH")
                            .font(legend_font(16.0))
                            .extra_letter_spacing(2.0)
                            .color(VALUE),
                    );
                    ui.add_space(16.0);
                    tabs(
                        ui,
                        &mut self.tab,
                        &[(Tab::Matrix, "matrix"), (Tab::History, "history"), (Tab::Charts, "charts")],
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(Key::new("reload").outline(VALUE)).clicked() {
                            self.reload();
                        }
                    });
                });
            });
        egui::Panel::top("runner")
            .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show(root, |ui| views::runner(self, ui));
        egui::Panel::bottom("notice")
            .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(egui::Margin::symmetric(12, 4)))
            .show(root, |ui| match &self.notice {
                Some((ok, text)) => status(ui, *ok, text),
                None => status(ui, true, &format!("{} runs in {}", self.runs.len(), self.dir.display())),
            });
        egui::CentralPanel::no_frame().show(root, |ui| {
            let r = ui.max_rect();
            ui.painter().rect_filled(r, 0.0, CHASSIS);
            ui.painter().line_segment([r.left_top(), r.right_top()], egui::Stroke::new(1.0, ETCH));
            egui::Frame::NONE.inner_margin(egui::Margin::same(10)).show(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;
                    match self.tab {
                        Tab::Matrix => views::matrix(self, ui),
                        Tab::History => views::history(self, ui),
                        Tab::Charts => views::charts(self, ui),
                    }
                });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().map(PathBuf::from).unwrap_or_else(store::data_dir);
    let matrix = args.get(1).cloned().unwrap_or_default();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 640.0])
            .with_min_inner_size([760.0, 480.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon-256.png"))
                    .unwrap_or_default(),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "inference-bench",
        options,
        Box::new(|cc| {
            egui_bench::install(&cc.egui_ctx);
            Ok(Box::new(App::new(dir, matrix)))
        }),
    )
}

#[cfg(test)]
mod tests;
