use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use inference_bench_core::fake::{FakeServer, Script};
use inference_bench_core::library::{Format, Support};
use inference_bench_core::settings::Endpoint;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn model(name: &str, specrt: bool, reason: &str) -> LocalModel {
    LocalModel {
        name: name.into(),
        repo: format!("acme/{name}"),
        path: PathBuf::from("/tmp/none"),
        root: PathBuf::from("/tmp/none"),
        format: Format::Safetensors,
        size_bytes: 3 << 30,
        architecture: "Qwen3ForCausalLM".into(),
        quant: "BF16".into(),
        context_length: Some(40960),
        files: vec![],
        support: vec![
            Support { backend: "specrt", supported: specrt, reason: reason.into() },
            Support { backend: "llama.cpp", supported: false, reason: "needs a GGUF conversion".into() },
            Support { backend: "vllm", supported: true, reason: "registered".into() },
            Support { backend: "transformers", supported: true, reason: "config".into() },
        ],
    }
}

fn script() -> Script {
    Script {
        models: vec!["acme/Qwen3-0.6B".into()],
        chunks: vec!["Hello ".into(), "**world**".into(), "!".into()],
        chunk_delay_ms: 5,
        first_delay_ms: 20,
        usage: Some((9, 3)),
        fail_status: None,
    }
}

fn harness(name: &str, tab: Tab) -> Harness<'static, App> {
    let d = dir(name);
    Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_accessibility_check(false).allow_missing_glyphs().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx));
        app.tab = tab;
        app.logs = shared_logs();
        app.library = vec![model("Qwen3-0.6B", true, "dense decoder"), model("Weird-7B", false, "model_type weird is not supported")];
        app
    })
}

#[test]
fn every_tab_draws() {
    for tab in [Tab::Chat, Tab::Models, Tab::Bench, Tab::Server, Tab::Logs, Tab::Settings] {
        let mut h = harness(&format!("tab{tab:?}"), tab);
        h.run();
        assert_eq!(h.state().tab, tab);
    }
}

#[test]
fn the_rail_switches_tabs() {
    let mut h = harness("rail", Tab::Chat);
    h.run();
    h.get_by_label("models").click();
    h.run();
    assert_eq!(h.state().tab, Tab::Models);
}

#[test]
fn the_model_list_shows_an_unsupported_architecture_badge() {
    let mut h = harness("badge", Tab::Models);
    h.run();
    h.get_by_label("specrt unsupported architecture");
    h.get_by_label("specrt supported");
    assert_eq!(h.get_all_by_label("llama.cpp unsupported architecture").count(), 2);
}

#[test]
fn a_fake_streaming_backend_fills_a_chat_with_stats() {
    let server = FakeServer::start(script());
    let mut h = harness("chat", Tab::Chat);
    {
        let app = h.state_mut();
        app.settings.endpoints = vec![Endpoint { label: "fake".into(), backend: "vllm".into(), url: server.url.clone(), api_key: String::new() }];
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B", "fake");
        app.load_model("fake", &m);
    }
    let start = Instant::now();
    while h.state().loaded().is_none() {
        assert!(start.elapsed() < Duration::from_secs(10), "model never loaded: {:?}", h.state().notice);
        h.step();
        std::thread::sleep(Duration::from_millis(20));
    }
    {
        let app = h.state_mut();
        app.chat.input = "say hi".into();
        let handle = app.engine.handle();
        let d = app.dir.clone();
        app.chat.send(handle, &d);
    }
    let start = Instant::now();
    while h.state().chat.streaming() || h.state().chat.conversations[0].turns.len() < 2 {
        assert!(start.elapsed() < Duration::from_secs(10), "reply never finished");
        h.step();
        std::thread::sleep(Duration::from_millis(20));
    }
    h.run();
    let turns = h.state_mut().chat.conv().turns.clone();
    assert_eq!(turns[1].message.content, "Hello **world**!");
    let stats = turns[1].stats.unwrap();
    assert_eq!((stats.prompt_tokens, stats.completion_tokens), (9, 3));
    assert!(stats.ttft_secs >= 0.02);
    assert!(h.state().chat.last_stats.is_some());
}

#[test]
fn stop_cancels_a_stream_and_regenerate_replaces_the_last_reply() {
    let server = FakeServer::start(Script { chunks: (0..60).map(|i| format!("w{i} ")).collect(), chunk_delay_ms: 20, ..script() });
    let mut h = harness("regen", Tab::Chat);
    {
        let app = h.state_mut();
        app.settings.endpoints = vec![Endpoint { label: "fake".into(), backend: "vllm".into(), url: server.url.clone(), api_key: String::new() }];
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B", "fake");
        app.load_model("fake", &m);
    }
    while h.state().loaded().is_none() {
        h.step();
        std::thread::sleep(Duration::from_millis(20));
    }
    {
        let app = h.state_mut();
        app.chat.input = "go".into();
        let handle = app.engine.handle();
        let d = app.dir.clone();
        app.chat.send(handle, &d);
    }
    std::thread::sleep(Duration::from_millis(150));
    h.state().chat.stop();
    let start = Instant::now();
    while h.state().chat.streaming() {
        assert!(start.elapsed() < Duration::from_secs(10));
        h.step();
        std::thread::sleep(Duration::from_millis(20));
    }
    let n = h.state_mut().chat.conv().turns[1].message.content.split_whitespace().count();
    assert!(n < 60, "stream was not cut: {n} words");
    let handle = h.state().engine.handle();
    h.state_mut().chat.regenerate(handle);
    assert!(h.state().chat.streaming());
    assert_eq!(h.state_mut().chat.conv().turns.len(), 1);
}

#[test]
fn the_bench_grid_runs_from_the_ui_against_fake_http_backends() {
    let a = FakeServer::start(Script { usage: Some((50, 3)), ..script() });
    let b = FakeServer::start(Script { models: vec!["other".into()], ..script() });
    let mut h = harness("bench", Tab::Bench);
    {
        let app = h.state_mut();
        app.settings.endpoints = vec![
            Endpoint { label: "alpha".into(), backend: "vllm".into(), url: a.url.clone(), api_key: String::new() },
            Endpoint { label: "beta".into(), backend: "vllm".into(), url: b.url.clone(), api_key: String::new() },
        ];
        app.bench.models.insert("Qwen3-0.6B".into());
        app.bench.models.insert("Weird-7B".into());
        app.bench.backends.insert("alpha".into());
        app.bench.backends.insert("beta".into());
        app.bench.lengths = vec![(32, true), (64, true)];
        app.bench.max_tokens = 3;
    }
    h.run();
    let plan = crate::bench::build_plan(h.state()).expect("a plan");
    assert_eq!(plan.cells().len(), 8);
    let wake = h.state().wake.clone();
    h.state_mut().bench.start(plan, wake);
    let start = Instant::now();
    while h.state().bench.running() || h.state().bench.results.is_empty() {
        assert!(start.elapsed() < Duration::from_secs(20), "bench never finished");
        h.step();
        std::thread::sleep(Duration::from_millis(30));
    }
    h.run();
    let results = h.state().bench.results.clone();
    assert_eq!(results.len(), 8);
    let find = |b: &str, m: &str| results.iter().find(|r| r.backend == b && r.model == m).unwrap();
    assert!(find("alpha", "Qwen3-0.6B @32").error.is_none());
    assert!(find("beta", "Qwen3-0.6B @32").error.is_some());
    let skipped = find("alpha", "Weird-7B @64").error.as_ref().unwrap();
    assert_eq!(skipped.kind, inference_bench_core::result::ErrorKind::ModelNotFound);
    assert_eq!(h.state().bench.sessions.len(), 1);
    assert!(h.state().dir.join("sessions").is_dir());
}

#[test]
fn the_server_tab_starts_and_serves_the_loaded_model() {
    let fake = FakeServer::start(script());
    let mut h = harness("server", Tab::Server);
    {
        let app = h.state_mut();
        app.settings.endpoints = vec![Endpoint { label: "fake".into(), backend: "vllm".into(), url: fake.url.clone(), api_key: String::new() }];
        app.settings.server.port = 0;
        app.server.port_text = "0".into();
        let m = app.library[0].clone();
        app.load_model("fake", &m);
    }
    while h.state().loaded().is_none() {
        h.step();
        std::thread::sleep(Duration::from_millis(20));
    }
    h.run();
    {
        let app = h.state_mut();
        let handle = app.engine.handle();
        app.server.start(handle, 0, "", false);
    }
    h.run();
    let url = h.state().server.running.as_ref().expect("server running").url();
    let mut resp = ureq::get(format!("{url}/v1/models")).call().unwrap();
    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert_eq!(v["data"][0]["id"], "Qwen3-0.6B");
    h.run();
    let logged = h.state().logs.snapshot(tracing::Level::INFO, "request finished");
    assert!(logged.iter().any(|e| e.field("path") == Some("/v1/models") && e.field("status") == Some("200")), "{logged:?}");
    let mut resp = ureq::post(format!("{url}/v1/chat/completions"))
        .send_json(serde_json::json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":8}))
        .unwrap();
    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello **world**!");
    h.state_mut().server.stop();
}

fn shared_logs() -> inference_bench_core::logging::LogBuffer {
    use tracing_subscriber::layer::SubscriberExt;
    static BUF: std::sync::OnceLock<inference_bench_core::logging::LogBuffer> = std::sync::OnceLock::new();
    BUF.get_or_init(|| {
        let b = inference_bench_core::logging::LogBuffer::new(500);
        let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry().with(b.clone()));
        b
    })
    .clone()
}

#[test]
fn the_logs_tab_filters_searches_and_saves() {
    let mut h = harness("logs", Tab::Logs);
    {
        let app = h.state_mut();
        app.logs = inference_bench_core::logging::LogBuffer::new(50);
        for (level, text) in [(tracing::Level::ERROR, "disk failed"), (tracing::Level::INFO, "model loaded"), (tracing::Level::DEBUG, "token step")] {
            app.logs.push(inference_bench_core::logging::LogEntry {
                time: chrono::Utc::now(),
                level,
                target: "t".into(),
                message: text.into(),
                fields: vec![],
                spans: vec![],
            });
        }
    }
    h.run();
    assert!(h.query_by_label_contains("model loaded").is_some());
    assert!(h.query_by_label_contains("token step").is_none(), "debug hidden at info");
    h.state_mut().logs_ui.search = "disk".into();
    h.run();
    assert!(h.query_by_label_contains("model loaded").is_none());
    assert!(h.query_by_label_contains("disk failed").is_some());
    let path = crate::logs::save_path(&h.state().dir);
    h.state().logs.save(&path, tracing::Level::TRACE, "").unwrap();
    assert!(std::fs::read_to_string(path).unwrap().contains("token step"));
}

#[test]
fn settings_secrets_are_saved_private() {
    let mut h = harness("settings", Tab::Settings);
    h.state_mut().settings.endpoints[0].api_key = "sk-test".into();
    h.state_mut().save_settings();
    let saved = Settings::load(&h.state().dir);
    assert_eq!(saved.endpoints[0].api_key, "sk-test");
}

#[test]
fn no_model_loaded_disables_sending() {
    let mut h = harness("nomodel", Tab::Chat);
    h.run();
    assert!(h.state().loaded().is_none());
    assert!(!h.state().chat.streaming());
}

fn pump_until(h: &mut Harness<'static, App>, secs: u64, what: &str, done: impl Fn(&App) -> bool) {
    let start = Instant::now();
    while !done(h.state()) {
        assert!(start.elapsed() < Duration::from_secs(secs), "{what}: {:?}", h.state().notice);
        h.step();
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
#[ignore = "needs a GPU, the hub cache and a built specrt worker; run with SHOTS=docs/gui"]
fn screenshots_with_real_data() {
    let out = PathBuf::from(std::env::var("SHOTS").unwrap_or("docs/gui".into()));
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("shots");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::new(d, wake_for(&cc.egui_ctx));
        app.settings.specrt_worker = std::env::var("SPECRT_WORKER").unwrap_or_default();
        app
    });
    pump_until(&mut h, 60, "scan", |a| !a.scan.busy() && !a.library.is_empty());
    let shot = |h: &mut Harness<'static, App>, name: &str| {
        h.run();
        h.render().unwrap().save(out.join(format!("{name}.png"))).unwrap();
    };
    h.state_mut().tab = Tab::Models;
    shot(&mut h, "models");

    let qwen = h.state().library.iter().find(|m| m.name == "Qwen3-0.6B").cloned().expect("Qwen3-0.6B in the hub cache");
    {
        let app = h.state_mut();
        app.chat.pick("Qwen3-0.6B", "specrt");
        app.load_model("specrt", &qwen);
        app.tab = Tab::Chat;
    }
    pump_until(&mut h, 120, "load", |a| a.loaded().is_some());
    for q in ["Explain what a KV cache is in two sentences.", "Show a tiny Rust struct with one method."] {
        {
            let app = h.state_mut();
            app.chat.conv().sampling.max_tokens = 160;
            app.chat.conv().sampling.temperature = 0.0;
            app.chat.input = q.into();
            let (handle, d) = (app.engine.handle(), app.dir.clone());
            app.chat.send(handle, &d);
        }
        let n = h.state_mut().chat.conv().turns.len() + 1;
        pump_until(&mut h, 120, "reply", move |a| !a.chat.streaming() && a.chat.conversations[0].turns.len() >= n);
    }
    std::thread::sleep(Duration::from_millis(2500));
    pump_until(&mut h, 20, "gpu", |a| !a.gpu.is_empty());
    shot(&mut h, "chat");

    {
        let app = h.state_mut();
        let handle = app.engine.handle();
        app.server.start(handle, 0, "", false);
        let url = app.server.running.as_ref().unwrap().url();
        for _ in 0..2 {
            let _ = ureq::get(format!("{url}/v1/models")).call();
        }
        let _ = ureq::post(format!("{url}/v1/chat/completions")).send_json(serde_json::json!({"messages":[{"role":"user","content":"Say hi"}],"max_tokens":24,"temperature":0}));
        app.tab = Tab::Server;
    }
    shot(&mut h, "server");

    {
        let app = h.state_mut();
        app.unload();
        app.bench.models.insert("Qwen3-0.6B".into());
        app.bench.models.insert("SmolLM2-135M".into());
        app.bench.backends.insert("specrt".into());
        app.bench.backends.insert("transformers".into());
        app.bench.lengths = vec![(128, true), (512, true), (2048, false), (8192, false)];
        app.bench.max_tokens = 64;
        app.tab = Tab::Bench;
    }
    shot(&mut h, "bench-setup");
    let plan = crate::bench::build_plan(h.state()).expect("plan");
    let wake = h.state().wake.clone();
    h.state_mut().bench.start(plan, wake);
    pump_until(&mut h, 600, "bench", |a| !a.bench.running());
    shot(&mut h, "bench-results");
    {
        let app = h.state_mut();
        let first = app.bench.sessions[0].1.clone();
        let mut second = first.clone();
        for r in &mut second {
            r.decode_tok_s = r.decode_tok_s.map(|v| v * 1.07);
        }
        app.bench.sessions.insert(0, ("previous run".into(), first));
        app.bench.sessions[1].0 = "this run".into();
        app.bench.sessions[1].1 = second;
        app.bench.left = 0;
        app.bench.right = 1;
        app.bench.view = crate::bench::View::Compare;
    }
    shot(&mut h, "bench-compare");
    h.state_mut().tab = Tab::Settings;
    shot(&mut h, "settings");
    h.state_mut().tab = Tab::Models;
    h.state_mut().models.pane = crate::models::Pane::Discover;
    shot(&mut h, "models-discover");
}
