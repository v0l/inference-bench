use crate::{App, Tab};
use eframe::egui::{self, Align2, Pos2, Rect, Sense, Stroke, Vec2};
use egui_bench::prelude::*;
use inference_bench_core::library::human_size;

const RAIL_W: f32 = 92.0;
const RAIL_ITEM_H: f32 = 64.0;

const TABS: [(Tab, &str); 5] = [
    (Tab::Chat, "chat"),
    (Tab::Models, "models"),
    (Tab::Server, "server"),
    (Tab::Hardware, "hardware"),
    (Tab::Settings, "settings"),
];

fn rail(app: &mut App, ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.y = 0.0;
    ui.add_space(6.0);
    for (tab, label) in TABS {
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(RAIL_W, RAIL_ITEM_H), Sense::click());
        let on = app.tab == tab;
        let p = ui.painter();
        if on {
            p.rect_filled(rect, 0.0, PANEL);
            p.rect_filled(Rect::from_min_size(rect.min, Vec2::new(3.0, rect.height())), 0.0, READOUT);
        } else if resp.hovered() {
            p.rect_filled(rect, 0.0, BAND);
        }
        glyph(p, tab, Pos2::new(rect.center().x, rect.top() + 22.0), if on { READOUT } else { LEGEND });
        p.text(
            Pos2::new(rect.center().x, rect.bottom() - 14.0),
            Align2::CENTER_CENTER,
            label.to_uppercase(),
            legend_font(11.0),
            if on { READOUT } else { LEGEND },
        );
        if resp.clicked() {
            app.load = None;
            app.tab = tab;
        }
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, label));
    }
}

fn glyph(p: &egui::Painter, tab: Tab, c: Pos2, tint: egui::Color32) {
    let s = Stroke::new(1.6, tint);
    match tab {
        Tab::Chat => {
            p.rect_stroke(Rect::from_center_size(c, Vec2::new(22.0, 15.0)), 2.0, s, egui::StrokeKind::Middle);
            p.line_segment([c + Vec2::new(-5.0, 7.5), c + Vec2::new(-8.0, 12.0)], s);
            p.line_segment([c + Vec2::new(-8.0, 12.0), c + Vec2::new(-1.0, 7.5)], s);
        }
        Tab::Models => {
            for i in 0..3 {
                let y = -7.0 + i as f32 * 7.0;
                p.rect_stroke(Rect::from_center_size(c + Vec2::new(0.0, y), Vec2::new(22.0, 5.0)), 1.0, s, egui::StrokeKind::Middle);
            }
        }
        Tab::Server => {
            p.rect_stroke(Rect::from_center_size(c + Vec2::new(0.0, -5.0), Vec2::new(22.0, 8.0)), 1.0, s, egui::StrokeKind::Middle);
            p.rect_stroke(Rect::from_center_size(c + Vec2::new(0.0, 6.0), Vec2::new(22.0, 8.0)), 1.0, s, egui::StrokeKind::Middle);
            p.circle_filled(c + Vec2::new(-7.0, -5.0), 1.5, tint);
            p.circle_filled(c + Vec2::new(-7.0, 6.0), 1.5, tint);
        }
        Tab::Hardware => {
            p.rect_stroke(Rect::from_center_size(c, Vec2::new(18.0, 18.0)), 2.0, s, egui::StrokeKind::Middle);
            p.rect_stroke(Rect::from_center_size(c, Vec2::new(8.0, 8.0)), 1.0, s, egui::StrokeKind::Middle);
            for d in [-5.0f32, 5.0] {
                p.line_segment([c + Vec2::new(d, -9.0), c + Vec2::new(d, -13.0)], s);
                p.line_segment([c + Vec2::new(d, 9.0), c + Vec2::new(d, 13.0)], s);
                p.line_segment([c + Vec2::new(-9.0, d), c + Vec2::new(-13.0, d)], s);
                p.line_segment([c + Vec2::new(9.0, d), c + Vec2::new(13.0, d)], s);
            }
        }
        Tab::Settings => {
            for (i, x) in [-0.4f32, 0.5, -0.1].iter().enumerate() {
                let y = -7.0 + i as f32 * 7.0;
                p.line_segment([c + Vec2::new(-11.0, y), c + Vec2::new(11.0, y)], s);
                p.circle_filled(c + Vec2::new(x * 20.0, y), 3.0, tint);
            }
        }
    }
}

fn status_bar(app: &mut App, ui: &mut egui::Ui) {
    if ui.available_width() < 560.0 {
        ui.vertical(|ui| {
            ui.horizontal(|ui| status_left(app, ui, 0.0));
            if let Some((ok, text)) = app.notice.clone() {
                crate::working::status_line(ui, ok, &text, 0.0);
            }
            ui.horizontal(|ui| {
                if let Some(s) = crate::chat::last_request(app) {
                    last_request_line(ui, &s);
                    ui.add_space(14.0);
                }
                vram_line(app, ui);
            });
        });
        return;
    }
    ui.horizontal(|ui| {
        status_left(app, ui, 460.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            vram_line(app, ui);
            if let Some(s) = crate::chat::last_request(app) {
                ui.add_space(14.0);
                last_request_line(ui, &s);
            }
        });
    });
}

fn last_request_line(ui: &mut egui::Ui, s: &inference_bench_core::live::RequestRecord) {
    let mut line = Line::new().legend("ttft").measured(format!("{:.0} ms", s.ttft_secs * 1000.0));
    if s.decode_tok_s > 0.0 {
        line = line.gap(10.0).legend("decode").measured(format!("{:.1} t/s", s.decode_tok_s));
    }
    line.size(11.0).show(ui);
}

fn vram_line(app: &App, ui: &mut egui::Ui) {
    if let Some((used, total)) = inference_bench_core::gpu::used_total(&app.gpu) {
        let legend = if app.gpu.len() > 1 { format!("vram x{}", app.gpu.len()) } else { "vram".into() };
        let mut line = Line::new().legend(&legend).measured(format!("{} / {}", human_size(used), human_size(total)));
        if let Some(u) = inference_bench_core::gpu::mean_util(&app.gpu) {
            line = line.gap(12.0).legend("gpu").measured(format!("{u}%"));
        }
        line.size(11.0).show(ui);
    }
}

fn status_left(app: &mut App, ui: &mut egui::Ui, reserve: f32) {
    {
        match (&app.loading, app.loaded()) {
            (Some(name), _) => {
                lamp(ui, &format!("loading {name}"), true, false);
                if let Some(p) = &app.load_progress {
                    status(ui, true, &crate::modelbar::progress_text(p));
                }
            }
            (None, Some(l)) => {
                lamp(ui, &l.display, true, false);
            }
            _ => {
                lamp(ui, "no model loaded", false, false);
            }
        }
        if let (true, Some((ok, text))) = (reserve > 0.0, &app.notice) {
            ui.add_space(10.0);
            crate::working::status_line(ui, *ok, text, reserve);
        }
    }
}

pub fn draw(app: &mut App, root: &mut egui::Ui) {
    egui::Panel::bottom("status")
        .frame(egui::Frame::NONE.fill(WELL).inner_margin(egui::Margin::symmetric(12, 5)))
        .show(root, |ui| status_bar(app, ui));
    egui::Panel::left("rail")
        .exact_size(RAIL_W)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(WELL))
        .show(root, |ui| rail(app, ui));
    egui::CentralPanel::no_frame().show(root, |ui| {
        let r = ui.max_rect();
        ui.painter().rect_filled(r, 0.0, CHASSIS);
        if app.load.is_some() {
            crate::load::draw(app, ui);
            return;
        }
        match app.tab {
            Tab::Chat => crate::chat::draw(app, ui),
            Tab::Models => crate::models::draw(app, ui),
            Tab::Server => crate::server::draw(app, ui),
            Tab::Hardware => crate::hardware::draw(app, ui),
            Tab::Settings => crate::settings::draw(app, ui),
        }
    });
    crate::models::modals(app, root.ctx());
}

pub fn page(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE.inner_margin(egui::Margin::same(14)).show(ui, |ui| {
        ui.label(
            egui::RichText::new(title.to_uppercase())
                .font(legend_font(16.0))
                .extra_letter_spacing(2.0)
                .color(VALUE),
        );
        ui.add_space(8.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            body(ui);
        });
    });
}
