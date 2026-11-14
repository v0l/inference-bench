mod common;

use common::*;
use inference_bench_core::engine::{Command, Engine, EngineEvent, Handle};
use inference_bench_core::fake::{Script, ScriptedModel};
use inference_bench_core::target::Target;
use inference_bench_server::{EngineFactory, Server};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

fn script() -> Script {
    Script {
        models: vec!["org/one".into(), "org/two".into()],
        chunks: vec!["Hel".into(), "lo".into()],
        chunk_delay_ms: 1,
        ..Script::default()
    }
}

struct AppRig {
    engine: Engine,
    rig: Rig,
}

fn app_rig() -> AppRig {
    app_rig_with("", Engine::with_factory(Arc::new(|| {}), ScriptedModel::factory(script())))
}

fn app_rig_with(extra: &str, engine: Engine) -> AppRig {
    let handle = engine.handle();
    let source = handle.clone();
    let factory = move |_: Option<usize>| -> Result<Box<dyn Target>, inference_bench_core::result::BackendError> {
        Ok(Box::new(inference_bench_server::engine_backend::EngineBackend::new(handle.clone())))
    };
    let rig = start_app(config(extra), factory, source);
    AppRig { engine, rig }
}

fn start_app(config: inference_bench_server::ServerConfig, factory: impl EngineFactory, source: Handle) -> Rig {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
    let handle = rt
        .block_on(
            Server::new(config)
                .with_factory(factory)
                .with_model_source(source)
                .with_gpu_probe(Arc::new(|| None))
                .start(),
        )
        .unwrap();
    let url = handle.url();
    Rig { rt, handle: Some(handle), stats: Default::default(), url }
}

impl AppRig {
    fn load(&self, model: &str, display: &str) {
        self.engine.send(Command::Load { model_id: model.into(), display: display.into(), max_ctx: None, max_num_seqs: None });
        loop {
            match self.engine.events.recv_timeout(Duration::from_secs(10)).unwrap() {
                EngineEvent::Loaded(_) => return,
                EngineEvent::LoadFailed(e) => panic!("{e:?}"),
                _ => {}
            }
        }
    }

    fn unload(&self) {
        self.engine.send(Command::Unload);
        loop {
            if let EngineEvent::Unloaded = self.engine.events.recv_timeout(Duration::from_secs(10)).unwrap() {
                return;
            }
        }
    }

    fn model_ids(&self) -> Vec<String> {
        let (status, body, _) = self.rig.get("/v1/models", None);
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                assert_eq!(m["loaded"], true);
                m["id"].as_str().unwrap().to_string()
            })
            .collect()
    }

    fn health_ids(&self) -> Vec<String> {
        let (_, body, _) = self.rig.get("/health", None);
        let v: Value = serde_json::from_str(&body).unwrap();
        v["loaded_models"].as_array().unwrap().iter().map(|m| m.as_str().unwrap().to_string()).collect()
    }

    fn chat(&self, body: Value) -> (u16, Value) {
        let (status, body, _) = self.rig.post("/v1/chat/completions", None, body);
        (status, serde_json::from_str(&body).unwrap())
    }
}

fn say(model: Option<&str>) -> Value {
    let mut v = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4});
    if let Some(m) = model {
        v["model"] = json!(m);
    }
    v
}

#[test]
fn models_and_health_follow_the_engine_as_it_loads_switches_and_unloads() {
    let app = app_rig();
    assert!(app.model_ids().is_empty());
    assert!(app.health_ids().is_empty());
    app.load("org/one", "One");
    assert_eq!(app.model_ids(), ["One"]);
    assert_eq!(app.health_ids(), ["One"]);
    assert!(app.rig.metrics().contains("inference_models_loaded 1"));
    app.load("org/two", "Two");
    assert_eq!(app.model_ids(), ["Two"]);
    assert_eq!(app.health_ids(), ["Two"]);
    app.unload();
    assert!(app.model_ids().is_empty());
    assert!(app.health_ids().is_empty());
    assert!(app.rig.metrics().contains("inference_models_loaded 0"));
    app.rig.stop();
}

#[test]
fn a_model_that_is_not_loaded_is_a_404_model_not_found() {
    let app = app_rig();
    let (status, body) = app.chat(say(Some("One")));
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["type"], "not_found_error");
    app.load("org/one", "One");
    let (status, body) = app.chat(say(Some("x")));
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["param"], "model");
    let (status, _) = app.chat(say(Some("Two")));
    assert_eq!(status, 404);
    app.rig.stop();
}

#[test]
fn the_loaded_model_answers_by_id_by_repo_and_when_the_model_is_omitted() {
    let app = app_rig();
    app.load("org/one", "One");
    for named in [Some("One"), Some("org/one"), None] {
        let (status, body) = app.chat(say(named));
        assert_eq!(status, 200, "{named:?}: {body}");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["model"], "One");
    }
    app.rig.stop();
}

#[test]
fn an_omitted_model_with_nothing_loaded_is_a_404() {
    let app = app_rig();
    let (status, body) = app.chat(say(None));
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    app.rig.stop();
}

#[test]
fn a_switch_is_followed_by_chat_on_the_new_model() {
    let app = app_rig();
    app.load("org/one", "One");
    assert_eq!(app.chat(say(Some("One"))).0, 200);
    app.load("org/two", "Two");
    assert_eq!(app.chat(say(Some("One"))).0, 404);
    let (status, body) = app.chat(say(Some("Two")));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "Two");
    app.rig.stop();
}

#[test]
fn four_api_requests_run_together_on_an_engine_loaded_with_four_sequences() {
    let slow = Script { chunks: (0..30).map(|i| format!("t{i} ")).collect(), chunk_delay_ms: 10, first_delay_ms: 20, ..script() };
    let engine = inference_bench_core::fake::engine(Arc::new(|| {}), slow);
    let app = app_rig_with("max_num_seqs = 64\n", engine);
    app.engine.send(Command::Load { model_id: "org/one".into(), display: "One".into(), max_ctx: Some(4096), max_num_seqs: Some(4) });
    loop {
        match app.engine.events.recv_timeout(Duration::from_secs(10)).unwrap() {
            EngineEvent::Loaded(_) => break,
            EngineEvent::LoadFailed(e) => panic!("{e:?}"),
            _ => {}
        }
    }
    let handle = app.engine.handle();
    let rig = &app.rig;
    let peak = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(std::sync::atomic::Ordering::SeqCst) {
                peak.fetch_max(handle.reading().running, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let (status, body, _) = rig.post("/v1/chat/completions", None, json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30}));
                    assert_eq!(status, 200, "{body}");
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 4);
    assert!(app.rig.metrics().contains("inference_running_seqs"));
    app.rig.stop();
}
