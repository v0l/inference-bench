use eframe::egui;
use egui_bench::prelude::*;

pub fn working(ui: &mut egui::Ui, what: &str) {
    ui.add_space(2.0);
    progress(ui, "", 0.0, None, "");
    let r = Line::new().note(what).size(11.0).show(ui);
    r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, what));
    ui.add_space(2.0);
}

pub fn status_line(ui: &mut egui::Ui, ok: bool, text: &str, reserve: f32) {
    let room = (ui.available_width() - reserve).max(80.0);
    ui.scope(|ui| {
        ui.set_max_width(room);
        ui.horizontal(|ui| {
            let tint = if ok { OK } else { FAULT };
            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 18.0), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), 3.0, tint);
            ui.painter().circle_stroke(rect.center(), 4.5, egui::Stroke::new(1.0, tint.gamma_multiply(0.4)));
            let r = Line::new().value(text).tint(tint).size(11.0).elided(ui);
            r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, text));
            r.on_hover_text(text);
        });
    });
}
