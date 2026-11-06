use super::*;
use chrono::Utc;
use inference_bench_core::result::{BackendError, Generation};

fn sample() -> Vec<RunResult> {
    let ok = RunResult::ok(
        "llama.cpp",
        "Qwen3-0.6B",
        Utc::now(),
        Generation {
            text: "hello".into(),
            prompt_tokens: 100,
            completion_tokens: 65,
            ttft_secs: 0.2,
            prefill_secs: 0.2,
            decode_secs: 0.4,
            peak_memory_bytes: Some(2 << 30),
        },
    );
    let bad = RunResult::failed(
        "vllm",
        "Qwen3-0.6B",
        Utc::now(),
        BackendError::unsupported("Model architectures ['Foo'] are not supported"),
    );
    vec![ok, bad]
}

fn app() -> App {
    let dir = std::env::temp_dir().join(format!("ib-ui-{}", std::process::id()));
    let mut app = App::new(dir, String::new());
    app.runs = sample();
    app
}

fn draw(app: &mut App) {
    let ctx = egui::Context::default();
    egui_bench::install(&ctx);
    for _ in 0..2 {
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            eframe::App::ui(app, ui, &mut frame());
        });
        out.textures_delta.clear();
    }
}

fn frame() -> eframe::Frame {
    eframe::Frame::_new_kittest()
}

#[test]
fn every_tab_draws_with_runs() {
    let mut app = app();
    for tab in [Tab::Matrix, Tab::History, Tab::Charts] {
        app.tab = tab;
        draw(&mut app);
    }
}

#[test]
fn every_tab_draws_with_no_runs() {
    let mut app = app();
    app.runs.clear();
    for tab in [Tab::Matrix, Tab::History, Tab::Charts] {
        app.tab = tab;
        draw(&mut app);
    }
}

#[test]
fn an_unsupported_architecture_is_labelled_in_the_matrix_cell() {
    let runs = sample();
    let (text, tint) = views::cell_text(&runs[1]);
    assert_eq!(text, "unsupported architecture");
    assert_eq!(tint, egui_bench::prelude::WARN);
    assert_eq!(views::cell_text(&runs[0]).0, "160.0 tok/s");
}

#[test]
fn a_missing_matrix_file_is_reported_not_run() {
    let mut app = app();
    app.matrix_path = "/nonexistent/matrix.toml".into();
    app.start_matrix();
    assert!(!app.running());
    assert!(!app.notice.as_ref().unwrap().0);
}
