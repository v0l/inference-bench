use eframe::egui::{self, Align2, Pos2, Rect, Vec2};
use egui_bench::prelude::*;
use inference_bench_core::report::{Bar, Metric};

const BAR_H: f32 = 14.0;
const GAP: f32 = 3.0;
const LABEL_W: f32 = 120.0;
const VALUE_W: f32 = 90.0;

fn show_value(metric: Metric, v: f64) -> String {
    match metric {
        Metric::TtftMs => format!("{v:.0} ms"),
        _ => format!("{v:.1} tok/s"),
    }
}

pub fn bars(ui: &mut egui::Ui, bars: &[Bar], metric: Metric, per_model: usize) {
    if bars.is_empty() || per_model == 0 {
        note(ui, "no runs yet", LEGEND);
        return;
    }
    let groups = bars.chunks(per_model);
    for group in groups {
        let model = &group[0].model;
        ui.add_space(4.0);
        ui.label(legend(model));
        let width = ui.available_width();
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(width, group.len() as f32 * (BAR_H + GAP)),
            egui::Sense::hover(),
        );
        let p = ui.painter_at(rect);
        let track = (width - LABEL_W - VALUE_W).max(40.0);
        let best = group.iter().filter_map(|b| b.value).fold(None, |acc: Option<f64>, v| {
            Some(match (acc, metric.higher_is_better()) {
                (None, _) => v,
                (Some(a), true) => a.max(v),
                (Some(a), false) => a.min(v),
            })
        });
        for (i, b) in group.iter().enumerate() {
            let top = rect.top() + i as f32 * (BAR_H + GAP);
            let row = Rect::from_min_size(Pos2::new(rect.left(), top), Vec2::new(width, BAR_H));
            p.text(
                Pos2::new(row.left(), row.center().y),
                Align2::LEFT_CENTER,
                &b.backend,
                figure(11.0),
                LEGEND,
            );
            let well = Rect::from_min_size(
                Pos2::new(row.left() + LABEL_W, top),
                Vec2::new(track, BAR_H),
            );
            p.rect_filled(well, 0.0, WELL);
            match b.value {
                Some(v) => {
                    let fill = Rect::from_min_size(well.min, Vec2::new(track * b.fraction, BAR_H));
                    let tint = if Some(v) == best { TRACE } else { TRACE.gamma_multiply(0.55) };
                    p.rect_filled(fill, 0.0, tint);
                    p.text(
                        Pos2::new(well.right() + 8.0, row.center().y),
                        Align2::LEFT_CENTER,
                        show_value(metric, v),
                        figure(11.0),
                        TRACE,
                    );
                }
                None => {
                    p.text(
                        Pos2::new(well.left() + 6.0, row.center().y),
                        Align2::LEFT_CENTER,
                        "no result",
                        figure(11.0),
                        LEGEND,
                    );
                }
            }
        }
    }
}
