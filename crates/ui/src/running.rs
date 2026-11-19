use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::engine::Loaded;
use specrt::options::{Group, schema};

pub fn rows(loaded: &Loaded) -> Vec<(String, String)> {
    let profile = if loaded.engine.is_custom() { "custom" } else { loaded.engine.profile.label() };
    let mut rows = vec![
        ("profile".to_string(), profile.to_string()),
        ("context".to_string(), format!("{} tokens", loaded.ctx)),
        ("parallel".to_string(), loaded.max_num_seqs.to_string()),
        ("mtp".to_string(), loaded.speculation.clone().unwrap_or_else(|| "not reported by this engine".to_string())),
    ];
    if let Some(n) = &loaded.parallel_note {
        rows.push(("fallback".to_string(), n.clone()));
    }
    let changed: Vec<_> = schema().into_iter().filter(|f| !loaded.engine.is_default(&f.key)).collect();
    for g in Group::ALL.into_iter().filter(|g| g.advanced()) {
        for f in changed.iter().filter(|f| f.group == g) {
            rows.push((f.label.to_string(), crate::load::display_value(&loaded.engine.value(&f.key))));
        }
    }
    rows
}

pub fn parser_rows(loaded: &Loaded) -> Vec<(String, String)> {
    use inference_bench_core::specrt_chat::ToolFormat;
    let config = inference_bench_server::config::ServerConfig::default().with_thinking(loaded.thinking).with_tools(loaded.tools);
    let tools = match (&config.tool_parser, loaded.tools) {
        (Some(p), _) => p.clone(),
        (None, ToolFormat::Unparsed) => "not parsed: the template uses a tool format this server cannot read".to_string(),
        (None, _) => "off: the template has no tools".to_string(),
    };
    let reasoning = config.reasoning_parser.unwrap_or_else(|| "off".to_string());
    vec![("tool parser".to_string(), tools), ("reasoning parser".to_string(), reasoning)]
}

pub fn draw(ui: &mut egui::Ui, loaded: &Loaded, several: bool) {
    let sub = if several { format!("what {} is using now", loaded.display) } else { "what the loaded model is using now".to_string() };
    section(ui, "running options", &sub, |ui| {
        let mut rows = rows(loaded);
        let defaults_only = rows.len() <= 4 && loaded.parallel_note.is_none();
        rows.extend(parser_rows(loaded));
        for (label, value) in &rows {
            let line = Line::new().legend(label).column(ui, 120.0);
            if label == "fallback" {
                line.tint(WARN).value(value.clone()).wrapped(ui);
            } else {
                line.value(value.clone()).show(ui);
            }
        }
        if defaults_only {
            hint(ui, "every advanced option is at its default");
        }
    });
}
