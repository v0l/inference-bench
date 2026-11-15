mod common;

use common::*;
use inference_bench_server::connect::{self, App, Endpoint};

const AGENT: &str = "tool_parser = 'qwen3_coder'\nreasoning_parser = 'qwen3'\n[[models]]\nid='m'\n";

const CALL: [&str; 6] = [
    "Let me look.",
    "<tool_",
    "call>\n<function=get_weather>\n<parameter=city>\nDal",
    "las\n</parameter>\n</function>\n</tool_call>",
    "<|im_end|>",
    "",
];

fn rig(pieces: &[&str]) -> Rig {
    let shape = Shape { pieces: pieces.iter().map(|s| s.to_string()).collect(), token_delay_ms: 0, ..Shape::default() };
    start(config(AGENT), shape)
}

fn endpoint(rig: &Rig) -> Endpoint {
    Endpoint { url: rig.url.clone(), key: None, model: "m".into() }
}

#[test]
fn check_passes_for_every_app_against_a_running_server() {
    let rig = rig(&CALL);
    let url = rig.url.clone();
    let e = endpoint(&rig);
    for app in App::ALL {
        let report = connect::check(app, &e);
        assert!(report.passed, "{}", report.describe());
        assert_eq!(report.steps.len(), 2);
    }
    assert_eq!(connect::first_model(&url, None).unwrap(), "m");
    rig.stop();
}

#[test]
fn check_fails_when_the_model_never_calls_the_tool() {
    let shape = Shape { pieces: vec!["Sunny".into(), "".into()], token_delay_ms: 0, ..Shape::default() };
    let rig = start(config("tool_parser = 'qwen3_coder'\n[[models]]\nid='m'\n"), shape);
    let report = connect::check(App::Codex, &endpoint(&rig));
    assert!(!report.passed);
    assert!(report.steps[0].passed);
    assert!(!report.steps[1].passed, "{}", report.describe());
    assert!(report.describe().contains("FAIL"));
    rig.stop();
}

#[test]
fn check_fails_with_the_wrong_key() {
    let rig = start(config(&format!("api_keys = ['right']\n{AGENT}")), Shape::default());
    let e = Endpoint { url: rig.url.clone(), key: Some("wrong".into()), model: "m".into() };
    let report = connect::check(App::Pi, &e);
    assert!(!report.passed);
    assert!(report.steps[0].detail.contains("401"), "{}", report.describe());
    rig.stop();
}
