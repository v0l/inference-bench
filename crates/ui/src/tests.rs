use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use inference_bench_core::conversations::Turn;
use inference_bench_core::fake::{FakeServer, Script, ScriptedModel};
use inference_bench_core::library::{Format, Support};

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn shots_dir() -> PathBuf {
    match std::env::var_os("SHOTS") {
        Some(d) => PathBuf::from(d),
        None => dir("shots"),
    }
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
        ],
        origin: None,
        vision: false,
    }
}

fn scripted(app: &mut App, script: Script) {
    app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), ScriptedModel::factory(script));
}

fn script() -> Script {
    Script {
        models: vec!["acme/Qwen3-0.6B".into()],
        chunks: vec!["Hello ".into(), "**world**".into(), "!".into()],
        chunk_delay_ms: 5,
        first_delay_ms: 20,
        usage: Some((9, 3)),
        fail_status: None,
        vision: false,
        files: Default::default(),
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
    for tab in [Tab::Chat, Tab::Models, Tab::Server, Tab::Settings] {
        let mut h = harness(&format!("tab{tab:?}"), tab);
        h.run();
        assert_eq!(h.state().tab, tab);
    }
}

#[test]
fn the_server_page_shows_a_configured_placement_read_only() {
    let mut h = harness("placement", Tab::Server);
    std::fs::create_dir_all(&h.state().dir).unwrap();
    std::fs::write(
        crate::server::config_path(&h.state().dir),
        r#"
        [placement]
        context = 4096
        [[placement.devices]]
        name = "alpha"
        memory_gib = 24
        bandwidth_gbs = 900
        tflops = 80
        [[placement.devices]]
        name = "beta"
        memory_gib = 24
        bandwidth_gbs = 900
        tflops = 80
        address = "10.0.0.2:9000"
        [[placement.stages]]
        device = "alpha"
        layers = [0, 20]
        [[placement.stages]]
        device = "beta"
        layers = [20, 40]
        "#,
    )
    .unwrap();
    h.run();
    h.get_by_label_contains("alpha 24 GiB on this machine");
    h.get_by_label_contains("beta 24 GiB at 10.0.0.2:9000");
    h.get_by_label_contains("stage 1 beta layers 20 to 40");
    assert!(h.state().server.placement.as_ref().unwrap().is_ok());
    assert!(h.query_by_label_contains("bandwidth").is_none());
}

#[test]
fn the_server_page_has_no_placement_without_a_server_config() {
    let mut h = harness("noplacement", Tab::Server);
    h.run();
    assert!(h.state().server.placement.is_none());
    assert!(h.query_by_label_contains("stage 0").is_none());
    h.get_by_label("select a model");
}

#[test]
fn the_model_bar_picks_from_a_search_and_opens_the_load_dialog() {
    let mut h = harness("modelbar", Tab::Chat);
    h.run();
    assert!(h.state().loaded().is_none());
    h.get_by_label("select a model").click();
    h.run();
    h.get_by_label("search your models");
    assert!(h.query_by_label_contains("Weird-7B").is_none(), "unsupported models are not offered");
    h.get_by_label_contains("Qwen3-0.6B").click();
    h.run();
    assert_eq!(h.state().load.as_ref().unwrap().model.name, "Qwen3-0.6B");
    h.get_by_label("LOAD QWEN3-0.6B");
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
fn typing_in_the_library_search_filters_the_list() {
    let mut h = harness("search", Tab::Models);
    {
        let app = h.state_mut();
        app.models.show_all = true;
        let mut gguf = model("Llama-3.2-1B-Instruct", true, "dense decoder");
        (gguf.format, gguf.quant, gguf.architecture) = (Format::Gguf, "Q4_K_M".into(), "llama".into());
        app.library.push(gguf);
    }
    h.run();
    for name in ["Qwen3-0.6B", "Weird-7B", "Llama-3.2-1B-Instruct"] {
        h.get_by_label(name);
    }
    let search = h.get_by_label("search my models by name, architecture or quant");
    search.click();
    h.run();
    h.get_by_label("search my models by name, architecture or quant").type_text("weird");
    h.run();
    assert_eq!(h.state().models.library_query, "weird");
    h.get_by_label("Weird-7B");
    assert!(h.query_by_label("Qwen3-0.6B").is_none());
    assert!(h.query_by_label("Llama-3.2-1B-Instruct").is_none());
    h.state_mut().models.library_query = "q4_k llama".into();
    h.run();
    h.get_by_label("Llama-3.2-1B-Instruct");
    assert!(h.query_by_label("Weird-7B").is_none());
    h.state_mut().models.library_query = "nothing like this".into();
    h.run();
    h.get_by_label_contains("no model matches");
}

#[test]
fn the_search_matches_name_repo_architecture_quant_and_format() {
    let mut m = model("Qwen3-0.6B", true, "dense decoder");
    m.quant = "Q8_0".into();
    m.format = Format::Gguf;
    for q in ["", "qwen", "acme", "qwen3forcausal", "q8_0", "gguf", "QWEN gguf", "0.6b q8"] {
        assert!(crate::models::matches(&m, q), "{q}");
    }
    for q in ["llama", "qwen vllm", "bf16"] {
        assert!(!crate::models::matches(&m, q), "{q}");
    }
}

fn real_model(name: &str) -> LocalModel {
    let d = dir(&format!("real-{name}"));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3-0.6b.config.json"), d.join("config.json")).unwrap();
    let mut m = model(name, true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m
}

fn with_gpu(h: &mut Harness<'static, App>, used: u64, total: u64) {
    let app = h.state_mut();
    app.gpu = vec![inference_bench_core::gpu::GpuStats { name: "test".into(), used_bytes: used, total_bytes: total, util_percent: None }];
    app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
}

#[test]
fn the_context_slider_stops_at_the_models_own_maximum() {
    let mut h = harness("slider", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    let dialog = h.state().load.as_ref().unwrap();
    let spec_max = dialog.view.as_ref().unwrap().anatomy.max_context;
    assert_eq!(spec_max, 40960);
    assert_eq!(dialog.max, spec_max);
    h.get_by_label("LOAD QWEN3-0.6B");
}

#[test]
fn a_model_with_a_huge_context_is_capped_at_what_the_engine_allows() {
    let a = detail_view("big", Some("qwen3.8-27b-exact.toml")).anatomy;
    assert_eq!(a.max_context, 131072);
    assert_eq!(crate::load::max_context(&a), specrt::engine::MAX_CTX);
}

#[test]
fn the_estimate_turns_red_when_the_model_does_not_fit_free_memory() {
    let mut h = harness("estimate", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    h.get_by_label("fits in free memory");
    assert!(h.query_by_label_contains("does not fit").is_none());
    with_gpu(&mut h, 95 << 30, 96 << 30);
    h.state_mut().load.as_mut().unwrap().ctx = 40960;
    h.run();
    h.get_by_label_contains("does not fit in free memory");
    assert!(h.query_by_label("fits in free memory").is_none());
    let e = h.state().load.as_ref().unwrap().estimate(h.state().free_gpu_bytes()).unwrap();
    assert_eq!(e.fits(), Some(false));
}

#[test]
fn the_default_context_is_reduced_to_what_fits_and_a_remembered_one_wins() {
    let a = detail_view("27b", Some("qwen3.8-27b-exact.toml")).anatomy;
    let max = crate::load::max_context(&a);
    let roomy = Some(a.total_bytes + a.state_bytes_per_seq + a.kv_bytes_per_token * 131072);
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, roomy, max), 8192);
    let tight = Some(a.total_bytes + a.state_bytes_per_seq + a.kv_bytes_per_token * 3000);
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, tight, max), 2560);
    assert_eq!(crate::load::initial_ctx(Some(100_000), 8192, &a, tight, max), 100_000);
    assert_eq!(crate::load::initial_ctx(Some(999_999), 8192, &a, tight, max), max);
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, None, max), 8192);
}

#[test]
fn loading_from_the_dialog_uses_the_chosen_context_and_remembers_it() {
    let mut h = harness("load-dialog", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    scripted(h.state_mut(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.state_mut().load.as_mut().unwrap().ctx = 16384;
    h.run();
    h.get_all_by_label("load").last().unwrap().click();
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    assert!(h.state().load.is_none());
    assert_eq!(h.state().loaded().unwrap().ctx, 16384);
    assert_eq!(h.state().settings.ctx_for("Qwen3-0.6B"), 16384);
    assert_eq!(Settings::load(&h.state().dir).ctx_for("Qwen3-0.6B"), 16384);
    assert_eq!(h.state().tab, Tab::Chat);
    h.state_mut().open_load_dialog(&m);
    assert_eq!(h.state().load.as_ref().unwrap().ctx, 16384);
}

#[test]
fn the_model_list_shows_an_unsupported_architecture_badge() {
    let mut h = harness("badge", Tab::Models);
    h.state_mut().models.show_all = true;
    h.run();
    h.get_by_label("unsupported architecture");
    h.get_by_label("runs here");
    assert!(h.query_by_label_contains("llama.cpp").is_none() && h.query_by_label_contains("vllm").is_none());
}

#[test]
fn a_scripted_native_model_fills_a_chat_with_stats() {
    let mut h = harness("chat", Tab::Chat);
    {
        let app = h.state_mut();
        scripted(app, script());
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B");
        app.load_model(&m);
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
    h.get_by_label_contains("Hello");

    h.get_by_label("copy").click();
    let mut copied = None;
    for _ in 0..4 {
        h.step();
        for c in &h.output().platform_output.commands {
            if let egui::OutputCommand::CopyText(t) = c {
                copied = Some(t.clone());
            }
        }
    }
    assert_eq!(copied.as_deref(), Some("Hello **world**!"));

    h.get_by_label("regenerate").click();
    h.step();
    h.step();
    assert!(h.state().chat.streaming());
    let turns = h.state_mut().chat.conv().turns.clone();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].message.content, "say hi");
}

#[test]
fn stop_cancels_a_stream_and_regenerate_replaces_the_last_reply() {
    let mut h = harness("regen", Tab::Chat);
    {
        let app = h.state_mut();
        scripted(app, Script { chunks: (0..60).map(|i| format!("w{i} ")).collect(), chunk_delay_ms: 20, ..script() });
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B");
        app.load_model(&m);
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
fn the_server_tab_starts_and_serves_the_loaded_model() {
    let mut h = harness("server", Tab::Server);
    {
        let app = h.state_mut();
        scripted(app, script());
        app.settings.server.port = 0;
        app.server.port_text = "0".into();
        let m = app.library[0].clone();
        app.load_model(&m);
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
    let mut h = harness("logs", Tab::Server);
    h.state_mut().server.pane = crate::server::Pane::Logs;
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
    h.state_mut().settings.hf_token = "hf_test".into();
    h.state_mut().save_settings();
    let saved = Settings::load(&h.state().dir);
    assert_eq!(saved.hf_token, "hf_test");
}

#[test]
fn no_model_loaded_disables_sending() {
    let mut h = harness("nomodel", Tab::Chat);
    h.run();
    assert!(h.state().loaded().is_none());
    assert!(!h.state().chat.streaming());
    assert!(h.get_all_by_label_contains("Load a model to chat").count() >= 1);
    h.get_by_label("select a model");
}

#[test]
fn an_empty_library_sends_the_user_to_models() {
    let mut h = harness("tomodels", Tab::Chat);
    h.state_mut().library.clear();
    h.run();
    h.get_by_label("go to models").click();
    h.run();
    assert_eq!(h.state().tab, Tab::Models);
}

#[test]
fn a_failed_load_shows_a_fault_on_the_chat_page() {
    let mut h = harness("loadfail", Tab::Chat);
    {
        let app = h.state_mut();
        scripted(app, Script { models: vec![], ..script() });
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B");
        app.load_model(&m);
    }
    pump_until(&mut h, 20, "load failure", |a| a.chat.load_error.is_some());
    h.run();
    assert_eq!(h.get_all_by_label_contains("no config.json").count(), 2, "status bar and chat fault");
}

#[test]
fn a_failed_reply_shows_a_fault_under_the_transcript() {
    let mut h = harness("replyfail", Tab::Chat);
    {
        let app = h.state_mut();
        app.chat.conv().turns.push(Turn { message: inference_bench_core::chat::Message::new(inference_bench_core::chat::Role::User, "hi"), stats: None, attachments: Vec::new() });
        app.chat.error = Some("the engine has stopped".into());
    }
    h.run();
    h.get_by_label_contains("the engine has stopped");
}

#[test]
fn chat_has_no_backend_picker_and_settings_have_no_bench_or_engine_numbers() {
    let mut h = harness("nopicker", Tab::Chat);
    h.run();
    for name in ["llama.cpp", "vllm", "transformers", "specrt"] {
        assert!(h.query_by_label(name).is_none(), "chat offers {name}");
    }
    let tree = format!("{:?}", h.root());
    assert!(!tree.contains("specrt") && !tree.contains("vllm"), "chat shows an engine choice: {tree}");
    h.state_mut().tab = Tab::Settings;
    h.run();
    for gone in ["name for a new bench target", "gpu bandwidth", "context", "native engine"] {
        assert!(h.query_by_label_contains(gone).is_none(), "settings still has {gone}");
    }
    assert!(h.query_by_label_contains("python3").is_none());
    assert!(h.query_by_label_contains("inference-bench-specrt").is_none());
    h.get_by_label("/path/to/models");
}

#[test]
fn there_is_no_bench_tab() {
    let mut h = harness("nobench", Tab::Chat);
    h.run();
    assert!(h.query_by_label("bench").is_none());
    for label in ["chat", "models", "server"] {
        h.get_by_label(label);
    }
    assert!(h.query_all_by_label("settings").count() >= 1);
}

#[test]
fn the_model_list_refreshes_when_a_spec_update_lands() {
    use inference_bench_core::fake::{FakeFile, tarball};
    let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs/qwen3-0.6b.toml")).unwrap();
    let fresh = base.replace("name = \"qwen3-0.6b\"", "name = \"landed-from-github\"");
    let gh = Script::default();
    let server = FakeServer::start(gh.clone());
    gh.serve_file(FakeFile {
        path: "/v0l/inference-bench/tar.gz/refs/heads/master".into(),
        body: tarball("inference-bench-master", &[("crates/specrt/specs/landed.toml", fresh.as_bytes())], Some("feed42")),
        etag: Some("\"x\"".into()),
        status: None,
    });
    let mut h = harness("specs-landed", Tab::Models);
    h.state_mut().specs.base = Some(server.url.clone());
    h.state_mut().rescan();
    pump_until(&mut h, 20, "first scan", |a| !a.scan.busy() && a.library.iter().any(|m| m.name == "llama-3.2-1b"));
    assert!(!h.state().library.iter().any(|m| m.name == "landed-from-github"));
    h.state_mut().models.show_all = true;
    h.run();
    h.get_by_label("check specs now").click();
    pump_until(&mut h, 20, "update and rescan", |a| !a.specs.busy() && !a.scan.busy() && a.library.iter().any(|m| m.name == "landed-from-github"));
    h.run();
    let landed = h.state().library.iter().find(|m| m.name == "landed-from-github").unwrap().clone();
    assert!(landed.origin.as_ref().unwrap().label().starts_with("updated from GitHub v0l/inference-bench@master (feed42) at "));
    h.get_by_label("landed-from-github");
    assert_eq!(h.state().notice.as_ref().unwrap().1, "specs updated: added landed-from-github");
    assert!(h.state().specs.state.last_ok);
}

#[test]
fn spec_updates_off_in_settings_makes_no_request() {
    let gh = Script::default();
    let server = FakeServer::start(gh);
    let mut h = harness("specs-off", Tab::Settings);
    {
        let app = h.state_mut();
        app.specs.base = Some(server.url.clone());
        app.specs.auto = true;
        app.settings.specs_auto_update = false;
    }
    for _ in 0..5 {
        h.step();
    }
    assert!(!h.state().specs.busy());
    assert!(server.requests.lock().unwrap().is_empty());
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
#[ignore = "needs a GPU and the hub cache; run with SHOTS=docs/gui"]
fn screenshots_with_real_data() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("shots");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::new(d, wake_for(&cc.egui_ctx));
        app.settings.default_ctx = 4096;
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
        app.chat.pick("Qwen3-0.6B");
        app.load_model(&qwen);
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

    h.state_mut().unload();
    h.state_mut().tab = Tab::Settings;
    shot(&mut h, "settings");
    h.state_mut().tab = Tab::Models;
    h.state_mut().models.pane = crate::models::Pane::Discover;
    shot(&mut h, "models-discover");
}

#[test]
#[ignore = "needs a GPU and Qwen3-0.6B in the hub cache; run with SHOTS=dir"]
fn chat_screenshots_with_a_real_model() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("chat-shots");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        App::new(d, wake_for(&cc.egui_ctx))
    });
    pump_until(&mut h, 60, "scan", |a| !a.scan.busy() && !a.library.is_empty());
    let shot = |h: &mut Harness<'static, App>, name: &str| {
        h.run();
        h.render().unwrap().save(out.join(format!("{name}.png"))).unwrap();
    };
    let qwen = h.state().library.iter().find(|m| m.name == "Qwen3-0.6B").cloned().expect("Qwen3-0.6B in the hub cache");
    h.state_mut().chat.pick("Qwen3-0.6B");
    shot(&mut h, "chat-empty");
    h.state_mut().load_model(&qwen);
    pump_until(&mut h, 120, "load", |a| a.loaded().is_some());
    {
        let app = h.state_mut();
        app.chat.conv().sampling.max_tokens = 700;
        app.chat.conv().sampling.temperature = 0.0;
        app.chat.input = "Write a Rust struct for a ring buffer with push and pop methods, then explain each method in a short paragraph.".into();
        let (handle, d) = (app.engine.handle(), app.dir.clone());
        app.chat.send(handle, &d);
    }
    pump_until(&mut h, 60, "first tokens", |a| !a.chat.streaming() || a.chat.streaming.as_ref().is_some_and(|s| s.text.len() > 300));
    assert!(h.state().chat.error.is_none(), "{:?}", h.state().chat.error);
    h.step();
    h.render().unwrap().save(out.join("chat-streaming.png")).unwrap();
    pump_until(&mut h, 120, "reply", |a| !a.chat.streaming() && a.chat.conversations[0].turns.len() >= 2);
    std::thread::sleep(Duration::from_millis(2500));
    pump_until(&mut h, 20, "gpu", |a| !a.gpu.is_empty());
    shot(&mut h, "chat-reply");
    let reply = h.state_mut().chat.conv().turns[1].message.content.clone();
    assert!(reply.contains("struct"), "{reply}");
    {
        let app = h.state_mut();
        app.chat.conv().sampling.max_tokens = 120;
        app.chat.input = "Now name the struct you just wrote, in one word.".into();
        let (handle, d) = (app.engine.handle(), app.dir.clone());
        app.chat.send(handle, &d);
    }
    pump_until(&mut h, 120, "second reply", |a| !a.chat.streaming() && a.chat.conversations[0].turns.len() >= 4);
    assert!(h.state().chat.error.is_none(), "{:?}", h.state().chat.error);
    shot(&mut h, "chat-second-turn");
    let second = h.state_mut().chat.conv().turns[3].message.content.clone();
    assert!(second.to_lowercase().replace(' ', "").contains("ringbuffer"), "{second}");
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_of_spec_provenance_and_update_settings() {
    use inference_bench_core::fake::{FakeFile, tarball};
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let specs_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs");
    let read = |f: &str| std::fs::read_to_string(format!("{specs_dir}/{f}")).unwrap();
    let newer = read("qwen3-0.6b.toml").replace("name = \"qwen3-0.6b\"", "name = \"qwen3-0.6b-new\"");
    let future = format!("min_engine = 3\n{}", read("qwen2.5-0.5b.toml").replace("name = \"qwen2.5-0.5b\"", "name = \"qwen2.5-0.5b-moe\""));
    let gh = Script::default();
    let server = FakeServer::start(gh.clone());
    gh.serve_file(FakeFile {
        path: "/v0l/inference-bench/tar.gz/refs/heads/master".into(),
        body: tarball(
            "inference-bench-master",
            &[
                ("crates/specrt/specs/qwen3-0.6b.toml", read("qwen3-0.6b.toml").as_bytes()),
                ("crates/specrt/specs/qwen3-0.6b-new.toml", newer.as_bytes()),
                ("crates/specrt/specs/qwen2.5-0.5b-moe.toml", future.as_bytes()),
            ],
            Some("5f9d34982b950f61d8f0a238aff52c4676d8b9b1"),
        ),
        etag: Some("\"s\"".into()),
        status: None,
    });
    let d = dir("spec-shot");
    let mine = d.join("my-specs");
    std::fs::create_dir_all(&mine).unwrap();
    std::fs::write(mine.join("llama.toml"), read("llama-3.2-1b.toml")).unwrap();
    let size = egui::vec2(1280.0, 1000.0);
    let mut h = Harness::builder().with_size(size).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx));
        app.tab = Tab::Models;
        app.logs = shared_logs();
        app.specs.base = Some(server.url.clone());
        app.settings.spec_folders = vec![mine];
        app.models.show_all = true;
        app
    });
    h.run();
    h.get_by_label("check specs now").click();
    pump_until(&mut h, 30, "update", |a| !a.specs.busy() && a.specs.state.last_ok);
    pump_until(&mut h, 30, "scan", |a| !a.scan.busy() && a.library.iter().any(|m| m.name == "qwen3-0.6b-new"));
    h.state_mut().notice = None;
    h.state_mut().library.retain(|m| m.format == inference_bench_core::library::Format::Spec);
    h.run();
    let models = h.render().unwrap();
    h.state_mut().tab = Tab::Settings;
    h.run();
    let mut settings_shot = None;
    for _ in 0..3 {
        h.run();
        settings_shot = Some(h.render().unwrap());
    }
    let settings = settings_shot.unwrap();
    let mut both = image::RgbaImage::new(models.width() + settings.width(), models.height().max(settings.height()));
    image::imageops::overlay(&mut both, &models, 0, 0);
    image::imageops::overlay(&mut both, &settings, models.width() as i64, 0);
    both.save(out.join("spec-updates.png")).unwrap();
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshots_of_the_server_and_logs_tabs() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness("shots-logs", Tab::Server);
    {
        let app = h.state_mut();
        app.logs = shared_logs();
        app.logs.clear();
        scripted(app, script());
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    {
        let handle = h.state().engine.handle();
        let url = {
            let app = h.state_mut();
            app.server.start(handle, 0, "", false);
            let url = app.server.running.as_ref().unwrap().url();
            let _ = ureq::get(format!("{url}/v1/models")).call();
            let _ = ureq::get(format!("{url}/health")).call();
            let _ = ureq::post(format!("{url}/v1/chat/completions"))
                .send_json(serde_json::json!({"messages":[{"role":"user","content":"Say hi"}],"max_tokens":24}));
            let _ = ureq::get(format!("{url}/nothing")).call();
            tracing::warn!(target: "specrt", "jit cache miss, compiling kernels");
            std::thread::sleep(Duration::from_millis(200));
            url
        };
        assert!(url.starts_with("http://"));
    }
    let shot = |h: &mut Harness<'static, App>, name: &str| {
        h.run();
        h.render().unwrap().save(out.join(format!("{name}.png"))).unwrap();
    };
    shot(&mut h, "server");
    h.state_mut().server.pane = crate::server::Pane::Logs;
    h.state_mut().logs_ui.level = tracing::Level::DEBUG;
    shot(&mut h, "logs");
    h.state_mut().server.stop();
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures")
}

fn header_structure(name: &str) -> specrt::structure::Structure {
    let text = std::fs::read_to_string(fixtures().join("headers").join(format!("{name}.json"))).unwrap();
    specrt::structure::Structure::from_json(&text).unwrap()
}

fn detail_view(name: &str, toml: Option<&str>) -> detail::View {
    match toml {
        Some(t) => {
            let spec = specrt::spec::Spec::load(&fixtures().join(t)).unwrap();
            detail::View::new(name, spec, detail::Source::SpecToml, 1, Some(131072)).with_structure(header_structure("qwen3.8-27b-fp8"))
        }
        None => {
            let json = std::fs::read_to_string(fixtures().join("tiny-qwen3.5/config.json")).unwrap();
            let spec = specrt::spec::Spec::from_hf_config(&json).unwrap();
            detail::View::new(name, spec, detail::Source::HfConfig, 0, None)
        }
    }
}

fn detail_harness(name: &str, size: egui::Vec2, view: detail::View, other: Option<detail::View>) -> Harness<'static, App> {
    let d = dir(name);
    Harness::builder().with_size(size).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx));
        app.tab = Tab::Models;
        let mut detail = detail::Detail::from_view(view);
        if let Some(o) = other {
            detail.compare_with(o);
        }
        app.models.detail = Some(detail);
        app
    })
}

#[test]
fn the_detail_view_draws_the_tiny_fixture_and_hovering_a_cell_shows_its_tooltip() {
    let mut h = detail_harness("detail-tiny", egui::vec2(1280.0, 900.0), detail_view("tiny", None), None);
    h.run();
    let count = h.state().models.detail.as_ref().unwrap().view.as_ref().unwrap().anatomy.layers.len();
    assert!(count >= 2);
    h.get_by_label_contains("layer 0 ").hover();
    h.run();
    assert!(h.query_by_label_contains("layer 0: ").is_some());
    assert!(h.query_by_label_contains("parameters").is_some());
    assert!(h.query_by_label("memory against context").is_some());
}

#[test]
fn a_gguf_model_opens_a_detail_view_from_its_metadata() {
    use specrt::gguf::writer::{Kv, build};
    let d = dir("detail-gguf");
    std::fs::create_dir_all(&d).unwrap();
    let path = d.join("Tiny-Q8_0.gguf");
    let embd = vec![0u8; 2 * 256 * 4];
    let bytes = build(
        3,
        &[
            ("general.architecture", Kv::Str("llama")),
            ("llama.block_count", Kv::U32(3)),
            ("llama.embedding_length", Kv::U32(256)),
            ("llama.feed_forward_length", Kv::U32(512)),
            ("llama.attention.head_count", Kv::U32(4)),
            ("llama.attention.head_count_kv", Kv::U32(2)),
            ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5)),
        ],
        &[("token_embd.weight", &[256, 4], 1, &embd[..])],
    );
    std::fs::write(&path, bytes).unwrap();
    let mut m = model("Tiny Q8_0", true, "GGUF Q8_0");
    (m.format, m.path, m.files) = (Format::Gguf, path.clone(), vec![path]);
    let view = detail::View::load(&m).unwrap();
    assert_eq!(view.source, detail::Source::GgufMetadata);
    assert_eq!((view.spec.layers.count, view.spec.dims.hidden, view.spec.attention.kv_heads), (3, 256, 2));
    assert_eq!(view.anatomy.layers.len(), 3);
}

#[test]
fn the_detail_view_draws_the_27b_with_its_delta_net_rhythm_and_opens_a_block_diagram() {
    let mut h = detail_harness("detail-27b", egui::vec2(1280.0, 900.0), detail_view("qwen3.8-27b", Some("qwen3.8-27b-exact.toml")), None);
    h.run();
    h.get_by_label("layer 3 full attention").hover();
    h.run();
    assert!(h.query_by_label_contains("layer 3: full attention").is_some());
    h.get_by_label("layer 3 full attention").click();
    h.run();
    let d = h.state().models.detail.as_ref().unwrap();
    assert_eq!((d.layer, d.kind), (Some(3), specrt::anatomy::LayerKind::Attention));
    h.get_by_label("layer 2 gated delta-net").click();
    h.run();
    assert_eq!(h.state().models.detail.as_ref().unwrap().kind, specrt::anatomy::LayerKind::DeltaNet);
}

#[test]
fn compare_draws_a_second_strip_on_the_same_scale() {
    let mut h = detail_harness(
        "detail-compare",
        egui::vec2(1280.0, 900.0),
        detail_view("qwen3.8-27b", Some("qwen3.8-27b-exact.toml")),
        Some(detail_view("qwen3.8-27b-nvfp4", Some("qwen3.8-27b-nvfp4-exact.toml"))),
    );
    h.run();
    assert_eq!(h.query_all_by_label("layer 3 full attention").count(), 2);
}

#[test]
#[ignore = "renders pictures; run with SHOTS=/tmp/shots"]
fn screenshots_of_the_model_detail_view() {
    let out = PathBuf::from(std::env::var("SHOTS").unwrap_or("/tmp/shots".into()));
    std::fs::create_dir_all(&out).unwrap();
    let cases: [(&str, detail::View, Option<detail::View>); 7] = [
        ("moe-35b", {
            let json = std::fs::read_to_string(fixtures().join("qwen3.6-35b-a3b-fp8.config.json")).unwrap();
            let spec = specrt::spec::Spec::describe_hf_config(&json).unwrap();
            detail::View::new("Qwen3.6-35B-A3B-FP8", spec, detail::Source::HfConfig, 1, Some(262144)).with_structure(header_structure("qwen3.6-35b-a3b-fp8"))
        }, None),
        ("27b-nvfp4", detail_view("qwen3.8-27b-nvfp4", Some("qwen3.8-27b-nvfp4-exact.toml")), None),
        ("27b-fp8", detail_view("qwen3.8-27b", Some("qwen3.8-27b-exact.toml")), None),
        ("qwen3-0.6b", {
            let spec = specrt::spec::Spec::load(&fixtures().join("../../specs/qwen3-0.6b.toml")).unwrap();
            detail::View::new("Qwen3-0.6B", spec, detail::Source::SpecToml, 0, Some(40960)).with_structure(header_structure("qwen3-0.6b"))
        }, None),
        ("llama", {
            let json = std::fs::read_to_string(fixtures().join("llama-3.2-1b.config.json")).unwrap();
            let spec = specrt::spec::Spec::describe_hf_config(&json).unwrap();
            detail::View::new("Llama-3.2-1B", spec, detail::Source::HfConfig, 0, None).with_structure(header_structure("llama-3.2-1b"))
        }, None),
        ("gemma4", {
            let json = std::fs::read_to_string(fixtures().join("gemma4-31b.config.json")).unwrap();
            let spec = specrt::spec::Spec::describe_hf_config(&json).unwrap();
            detail::View::new("gemma-4-31B-it", spec, detail::Source::HfConfig, 0, None).with_structure(header_structure("gemma4-31b"))
        }, None),
        ("compare", detail_view("qwen3.8-27b", Some("qwen3.8-27b-exact.toml")), Some(detail_view("qwen3.8-27b-nvfp4", Some("qwen3.8-27b-nvfp4-exact.toml")))),
    ];
    for (name, view, other) in cases {
        for (w, h_px) in [(1400.0, 900.0), (1024.0, 700.0), (1400.0, 3000.0)] {
            let mut h = detail_harness(&format!("shot-{name}"), egui::vec2(w, h_px), view_clone(&view), other.as_ref().map(view_clone));
            h.run();
            h.query_all_by_label_contains("layer 3 ").next().unwrap().click();
            h.run();
            h.render().unwrap().save(out.join(format!("detail-{name}-{w}x{h_px}.png"))).unwrap();
        }
    }
}

fn view_clone(v: &detail::View) -> detail::View {
    {
        let mut c = detail::View::new(&v.name, v.spec.clone(), v.source, v.anatomy.mtp_layers, Some(v.anatomy.max_context));
        c.structure = v.structure.clone();
        c
    }
}

fn red_png() -> Vec<u8> {
    let mut out = Vec::new();
    let img = image::RgbaImage::from_pixel(64, 48, image::Rgba([200, 30, 30, 255]));
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).unwrap();
    out
}

fn load_vision(h: &mut Harness<'static, App>, vision: bool) {
    {
        let app = h.state_mut();
        scripted(app, Script { vision, ..script() });
        let m = app.library[0].clone();
        app.chat.pick("Qwen3-0.6B");
        app.load_model(&m);
    }
    pump_until(h, 10, "model never loaded", |a| a.loaded().is_some());
}

#[test]
fn a_pasted_image_shows_a_thumbnail_and_goes_out_as_an_attachment() {
    let mut h = harness("attach", Tab::Chat);
    load_vision(&mut h, true);
    assert!(h.state().engine.vision());
    let rgba = vec![egui::Color32::from_rgb(200, 30, 30); 64 * 48];
    let img = egui::ColorImage { size: [64, 48], source_size: egui::vec2(64.0, 48.0), pixels: rgba };
    h.event(egui::Event::PasteImage(std::sync::Arc::new(img)));
    h.run();
    assert_eq!(h.state().chat.pending.len(), 1);
    assert_eq!(h.state().chat.pending[0].mime, "image/png");
    h.get_by_label_contains("attachment pasted.png");
    h.get_by_label("remove");
    {
        let app = h.state_mut();
        app.chat.input = "what is this".into();
        let handle = app.engine.handle();
        let d = app.dir.clone();
        app.chat.send(handle, &d);
    }
    assert!(h.state().chat.pending.is_empty());
    let turn = h.state_mut().chat.conv().turns[0].clone();
    assert_eq!(turn.attachments.len(), 1);
    pump_until(&mut h, 10, "reply never finished", |a| !a.chat.streaming() && a.chat.conversations[0].turns.len() >= 2);
    let stats = h.state_mut().chat.conv().turns[1].stats.unwrap();
    assert_eq!(stats.vision_tokens, 100);
    let saved = inference_bench_core::conversations::load_all(&h.state().dir);
    assert_eq!(saved[0].turns[0].attachments.len(), 1);
}

#[test]
fn a_text_only_model_cannot_attach_and_a_dropped_text_file_is_refused() {
    let mut h = harness("attach-no", Tab::Chat);
    load_vision(&mut h, false);
    h.run();
    assert!(!h.state().engine.vision());
    h.get_by_label_contains("this model takes text only");
    h.state_mut().chat.attach_bytes("notes.txt", b"hello");
    assert!(h.state().chat.pending.is_empty());
    assert!(h.state().chat.attach_error.as_deref().unwrap().contains("notes.txt"));
    h.state_mut().chat.attach_bytes("red.png", &red_png());
    assert_eq!(h.state().chat.pending.len(), 1);
}

#[test]
fn a_dropped_image_file_becomes_a_pending_attachment() {
    #[derive(Debug)]
    struct File(std::path::PathBuf, Vec<u8>);
    impl egui::DroppedFile for File {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
        fn bytes(&self) -> Result<Vec<u8>, String> {
            Ok(self.1.clone())
        }
    }
    let mut h = harness("drop", Tab::Chat);
    load_vision(&mut h, true);
    h.input_mut().dropped_files.push(std::sync::Arc::new(File("/tmp/red.png".into(), red_png())));
    h.run();
    assert_eq!(h.state().chat.pending.len(), 1);
    assert_eq!(h.state().chat.pending[0].name, "red.png");
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_chat_attachment_thumbnail() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("shot-attach");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx));
        app.tab = Tab::Chat;
        app.logs = shared_logs();
        app.library = vec![model("Qwen3-0.6B", true, "dense decoder")];
        app
    });
    load_vision(&mut h, true);
    let mut img = image::RgbaImage::new(160, 120);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgba([(x * 255 / 160) as u8, (y * 255 / 120) as u8, 120, 255]);
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    h.state_mut().chat.attach_bytes("gradient.png", &png);
    h.run();
    h.render().unwrap().save(out.join("chat-attachment.png")).unwrap();
}

#[test]
fn the_models_page_says_vision_yes_only_for_models_with_a_vision_config() {
    let mut h = harness("vision-chip", Tab::Models);
    h.state_mut().models.show_all = true;
    h.state_mut().library[0].vision = true;
    h.run();
    h.get_by_label("vision: yes");
    assert_eq!(h.query_all_by_label("vision: no").count(), 1);
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_vision_tower_in_the_model_detail() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let json = std::fs::read_to_string(fixtures().join("qwen3.8-27b.multimodal.json")).unwrap();
    let mut spec = specrt::spec::Spec::from_hf_config(&json).unwrap();
    spec.name = "qwen3.8-27b".into();
    let view = detail::View::new("qwen3.8-27b", spec, detail::Source::HfConfig, 1, None);
    let mut h = detail_harness("shot-vision", egui::vec2(1400.0, 3200.0), view, None);
    h.run();
    h.render().unwrap().save(out.join("detail-vision-tower.png")).unwrap();
}
