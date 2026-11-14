use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
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
        lineage: None,
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
        let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
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
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = dir(&format!("real-{name}-{n}"));
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
fn the_27b_config_gives_the_load_dialog_a_262144_maximum() {
    let mut h = harness("slider-27b", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let d = dir("real-27b");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), d.join("config.json")).unwrap();
    let mut m = model("Qwen3.8-27B", true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    let dialog = h.state().load.as_ref().unwrap();
    assert_eq!(dialog.max, 262144);
    assert_eq!(crate::load::max_context(&dialog.view.as_ref().unwrap().anatomy), 262144);
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
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, roomy, 1, max), 8192);
    let tight = Some(a.total_bytes + a.state_bytes_per_seq + a.kv_bytes_per_token * 3000);
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, tight, 1, max), 2560);
    assert_eq!(crate::load::initial_ctx(Some(100_000), 8192, &a, tight, 1, max), 100_000);
    assert_eq!(crate::load::initial_ctx(Some(999_999), 8192, &a, tight, 1, max), max);
    assert_eq!(crate::load::initial_ctx(None, 8192, &a, None, 1, max), 8192);
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

fn slow_script() -> Script {
    Script { chunks: (0..40).map(|i| format!("t{i} ")).collect(), chunk_delay_ms: 10, first_delay_ms: 20, usage: Some((12, 40)), ..script() }
}

fn serving_harness(name: &str, seqs: usize) -> Harness<'static, App> {
    let mut h = harness(name, Tab::Server);
    {
        let app = h.state_mut();
        app.engine = inference_bench_core::fake::engine(app.wake.clone(), slow_script());
        let m = app.library[0].clone();
        app.engine.send(Command::Load { model_id: m.model_id(inference_bench_core::target::NATIVE), display: m.name.clone(), max_ctx: Some(4096), max_num_seqs: Some(seqs), served_names: Vec::new() });
    }
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    let handle = h.state().engine.handle();
    h.state_mut().server.start(handle, 0, "", false);
    h
}

fn chat_once(url: &str) {
    let mut resp = ureq::post(format!("{url}/v1/chat/completions"))
        .send_json(serde_json::json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":40}))
        .unwrap();
    let _: serde_json::Value = resp.body_mut().read_json().unwrap();
}

#[test]
fn the_server_tab_has_no_curl_line_and_one_compact_address_line() {
    let mut h = serving_harness("nocurl", 4);
    h.run();
    let url = h.state().server.running.as_ref().unwrap().url();
    assert!(h.query_by_label_contains("curl").is_none());
    assert!(h.query_by_label_contains("use it").is_none());
    assert!(h.query_by_label_contains("USE IT").is_none());
    h.get_by_label_contains(&format!("{url}/v1"));
    assert!(h.get_all_by_label_contains("Qwen3-0.6B").count() >= 1);
    h.get_by_label("copy");
    h.state_mut().server.stop();
}

#[test]
fn the_load_dialog_has_a_parallel_requests_slider_and_the_estimate_grows_with_it() {
    let mut h = harness("seqs-dialog", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    h.get_by_label_contains("PARALLEL REQUESTS");
    let free = h.state().free_gpu_bytes();
    let dialog = h.state_mut().load.as_mut().unwrap();
    dialog.ctx = 8192;
    dialog.seqs = 1;
    let one = dialog.estimate(free).unwrap();
    dialog.seqs = 4;
    let four = dialog.estimate(free).unwrap();
    assert_eq!(four.kv, one.kv * 4);
    assert!(four.total() > one.total());
    assert_eq!(four.weights, one.weights);
    let a = detail_view("seqs-27b", Some("qwen3.8-27b-exact.toml")).anatomy;
    let (a1, a4) = (crate::load::estimate(&a, 8192, 1, None), crate::load::estimate(&a, 8192, 4, None));
    assert_eq!(a4.state, a1.state * 4);
    assert!(a.state_bytes_per_seq > 100 << 20);
    assert_eq!(crate::load::largest_seqs(&a, 8192, Some(a.total_bytes)), 1);
    assert_eq!(crate::load::largest_seqs(&a, 8192, None), crate::load::MAX_SEQS);
}

#[test]
fn loading_from_the_dialog_passes_parallel_requests_to_the_engine_and_remembers_them() {
    let mut h = harness("seqs-load", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    assert_eq!(h.state().load.as_ref().unwrap().seqs, 4);
    h.state_mut().load.as_mut().unwrap().seqs = 6;
    h.run();
    h.get_all_by_label("load").last().unwrap().click();
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    assert_eq!(h.state().loaded().unwrap().max_num_seqs, 6);
    assert_eq!(h.state().settings.seqs_for("Qwen3-0.6B"), 6);
    assert_eq!(Settings::load(&h.state().dir).seqs_for("Qwen3-0.6B"), 6);
}

#[test]
fn served_names_from_the_load_dialog_reach_the_engine_and_are_remembered() {
    let mut h = harness("served-names", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    h.get_by_label_contains("SERVED NAMES");
    h.state_mut().load.as_mut().unwrap().served_names = " qwen3.8-27b, gpt-4o ,, qwen3.8-27b".into();
    h.run();
    h.get_all_by_label("load").last().unwrap().click();
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    let want = vec!["qwen3.8-27b".to_string(), "gpt-4o".to_string()];
    assert_eq!(h.state().loaded().unwrap().served_names, want);
    assert_eq!(Settings::load(&h.state().dir).served_names_for("Qwen3-0.6B"), want);
    h.state_mut().open_load_dialog(&m);
    assert_eq!(h.state().load.as_ref().unwrap().served_names, "qwen3.8-27b, gpt-4o");
}

#[test]
fn four_parallel_api_requests_reach_running_four_and_chat_keeps_working() {
    let mut h = serving_harness("parallel", 4);
    h.run();
    let url = h.state().server.running.as_ref().unwrap().url();
    let handle = h.state().engine.handle();
    let peak = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(std::sync::atomic::Ordering::SeqCst) {
                peak.fetch_max(handle.reading().running, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        let api: Vec<_> = (0..3).map(|_| scope.spawn(|| chat_once(&url))).collect();
        std::thread::sleep(Duration::from_millis(60));
        let rx = handle.chat(inference_bench_core::chat::ChatRequest::new(
            vec![inference_bench_core::chat::Message::new(inference_bench_core::chat::Role::User, "from the app chat")],
            inference_bench_core::chat::Sampling::default(),
        ));
        let mut chat_done = false;
        while let Ok(e) = rx.recv_timeout(Duration::from_secs(10)) {
            if matches!(e, inference_bench_core::engine::ChatEvent::Done(_)) {
                chat_done = true;
                break;
            }
        }
        assert!(chat_done);
        for t in api {
            t.join().unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 4);
    h.state_mut().server.stop();
}

#[test]
fn the_metrics_section_shows_throughput_and_ttft_after_a_request() {
    let mut h = serving_harness("metrics", 4);
    let url = h.state().server.running.as_ref().unwrap().url();
    let t = std::thread::spawn(move || chat_once(&url));
    let until = Instant::now() + Duration::from_secs(20);
    while Instant::now() < until {
        h.step();
        std::thread::sleep(Duration::from_millis(100));
        let live = h.state().engine.handle().history().lock().unwrap().points().any(|p| p.gen_tok_s > 0.0);
        if live && t.is_finished() {
            break;
        }
    }
    t.join().unwrap();
    let start = Instant::now();
    while !h.state().engine.handle().history().lock().unwrap().last().is_some_and(|p| p.ttft_p50 > 0.0) {
        let hist = h.state().engine.handle().history();
        let hist = hist.lock().unwrap();
        assert!(start.elapsed() < Duration::from_secs(10), "points {:?} reading {:?}", hist.points().collect::<Vec<_>>(), hist.latest().map(|r| (r.generated, r.requests, r.ttft.total)));
        drop(hist);
        h.step();
        std::thread::sleep(Duration::from_millis(100));
    }
    h.run();
    let snap = crate::serving::snapshot(h.state(), crate::serving::Span::Five);
    let last = snap.last.unwrap();
    assert!(snap.generation.iter().any(|v| *v > 0.0), "no throughput in {:?}", snap.generation);
    assert!(last.ttft_p50 > 0.0 && last.ttft_p95 >= last.ttft_p50);
    assert_eq!(snap.recent.len(), 1);
    assert_eq!((snap.recent[0].prompt_tokens, snap.recent[0].completion_tokens), (12, 40));
    assert!(snap.recent[0].ttft_secs > 0.0 && snap.recent[0].decode_tok_s > 0.0);
    h.get_by_label_contains("REQUESTS PER MIN");
    h.get_by_label_contains("KV CACHE");
    h.get_by_label_contains("PREFIX HITS");
    h.state_mut().server.stop();
}

fn burst(url: &str, n: usize, tokens: usize) -> (f64, u64) {
    let started = Instant::now();
    let total: u64 = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..n)
            .map(|i| {
                scope.spawn(move || {
                    let mut resp = ureq::post(format!("{url}/v1/chat/completions"))
                        .send_json(serde_json::json!({"messages":[{"role":"user","content":format!("Write a long story about robot number {i}.")}],"max_tokens":tokens,"temperature":0.0,"ignore_eos":true}))
                        .unwrap();
                    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
                    v["usage"]["completion_tokens"].as_u64().unwrap()
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).sum()
    });
    (total as f64 / started.elapsed().as_secs_f64(), total)
}

#[test]
#[ignore = "needs a GPU and Qwen3-0.6B in the hub cache; run with SHOTS=dir under flock /tmp/gpu27.lock"]
fn parallel_requests_with_a_real_model_run_together_and_aggregate_faster() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("parallel-real");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 1500.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::new(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
        app.tab = Tab::Server;
        app.logs = shared_logs();
        app
    });
    pump_until(&mut h, 60, "scan", |a| !a.scan.busy() && !a.library.is_empty());
    let qwen = h.state().library.iter().find(|m| m.name == "Qwen3-0.6B").cloned().expect("Qwen3-0.6B in the hub cache");
    let mut rates = Vec::new();
    for seqs in [1usize, 4] {
        h.state_mut().load_model_with(&qwen, 4096, seqs);
        pump_until(&mut h, 180, "load", |a| a.loaded().is_some_and(|l| l.max_num_seqs == seqs));
        if h.state().server.running.is_none() {
            let handle = h.state().engine.handle();
            h.state_mut().server.start(handle, 0, "", false);
        }
        let url = h.state().server.running.as_ref().unwrap().url();
        let _ = burst(&url, 1, 8);
        let handle = h.state().engine.handle();
        let peak = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::atomic::AtomicBool::new(false);
        let (rate, total) = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(std::sync::atomic::Ordering::SeqCst) {
                    peak.fetch_max(handle.reading().running, std::sync::atomic::Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(5));
                }
            });
            let r = burst(&url, 4, 400);
            done.store(true, std::sync::atomic::Ordering::SeqCst);
            r
        });
        let peak = peak.load(std::sync::atomic::Ordering::SeqCst);
        eprintln!("max_num_seqs {seqs}: 4 concurrent requests, {total} tokens, {rate:.1} tok/s aggregate, running peaked at {peak}");
        rates.push((rate, peak));
        if seqs == 4 {
            let until = Instant::now() + Duration::from_secs(36);
            while Instant::now() < until {
                let u = url.clone();
                let t = std::thread::spawn(move || burst(&u, 4, 300));
                while !t.is_finished() {
                    h.step();
                    std::thread::sleep(Duration::from_millis(250));
                }
                t.join().unwrap();
                for _ in 0..3 {
                    h.step();
                    std::thread::sleep(Duration::from_millis(1000));
                }
            }
            for _ in 0..3 {
                h.step();
                std::thread::sleep(Duration::from_millis(1000));
            }

            h.run();
            h.render().unwrap().save(out.join("server-live.png")).unwrap();
        }
    }
    assert_eq!(rates[0].1, 1);
    assert_eq!(rates[1].1, 4);
    assert!(rates[1].0 > rates[0].0 * 1.5, "{rates:?}");
    h.state_mut().server.stop();
}

#[test]
fn after_a_request_prompt_rate_kv_and_the_request_log_all_show_numbers() {
    let mut h = serving_harness("after-request", 4);
    let url = h.state().server.running.as_ref().unwrap().url();
    chat_once(&url);
    let start = Instant::now();
    while !h.state().engine.handle().history().lock().unwrap().points().any(|p| p.prefill_tok_s > 0.0) {
        assert!(start.elapsed() < Duration::from_secs(10), "prompt tok/s never rose above 0");
        h.step();
        std::thread::sleep(Duration::from_millis(100));
    }
    h.run();
    let snap = crate::serving::snapshot(h.state(), crate::serving::Span::Five);
    assert!(snap.last.as_ref().unwrap().prefill_tok_s > 0.0, "{:?}", snap.last);
    assert!(snap.prefill.iter().any(|v| *v > 0.0));
    let r = snap.reading.unwrap();
    let shown = crate::serving::readings(&r, 0.0);
    let kv = shown.iter().find(|(l, _, _)| *l == "kv cache").unwrap();
    assert!(kv.1.starts_with("0 MB / "), "{}", kv.1);
    assert!(!kv.1.contains('-'), "{}", kv.1);
    let lines = crate::server::request_lines(&h.state().logs);
    assert!(lines.iter().any(|l| l.request == "POST /v1/chat/completions" && l.status == 200), "request log has no API request");
    assert!(h.query_by_label("no requests yet").is_none());
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
        app.load_model(&m);
    }
    pump_until(&mut h, 20, "load failure", |a| a.chat.load_error.is_some());
    h.run();
    assert_eq!(h.get_all_by_label_contains("no config.json").count(), 2, "status bar and chat fault");
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
    use inference_bench_core::fake::publish_specs;
    let base = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs/qwen3-0.6b.toml")).unwrap();
    let fresh = base.replace("name = \"qwen3-0.6b\"", "name = \"landed-from-github\"");
    let gh = Script::default();
    let server = FakeServer::start(gh.clone());
    publish_specs(&gh, "v0l/inference-bench", "refs/heads/master", "feed42a", "x", &[("landed.toml", fresh.as_bytes())]);
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
    assert!(landed.origin.as_ref().unwrap().label().starts_with("updated from GitHub v0l/inference-bench@master (feed42a) at "));
    h.get_by_label("landed-from-github");
    assert_eq!(h.state().notice.as_ref().unwrap().1, "specs updated: added landed-from-github");
    assert!(h.state().specs.state.last_ok);
}

#[test]
fn a_300_character_spec_error_leaves_the_check_key_its_full_width() {
    let mut h = harness("specs-long-error", Tab::Models);
    h.run();
    let plain = h.get_by_label("check specs now").rect();
    assert!(plain.width() > 90.0, "{plain:?}");
    let error = format!("https://example.invalid/{}: {}", "x".repeat(60), "download over 67108864 bytes or broken ".repeat(8));
    assert!(error.len() >= 300);
    {
        let app = h.state_mut();
        app.specs.last = Some((false, format!("spec update failed: {error}")));
        app.specs.state.last_ok = false;
    }
    h.run();
    let key = h.get_by_label("check specs now").rect();
    assert!(key.width() >= plain.width() - 1.0, "key shrank from {} to {}", plain.width(), key.width());
    assert!(key.height() <= plain.height() + 1.0, "key wrapped to {} high", key.height());
    assert!(h.query_by_label_contains("download over").is_none());
}

#[test]
fn the_spec_verdict_is_one_short_word() {
    assert_eq!(crate::specs::verdict(true, ""), "not checked yet");
    assert_eq!(crate::specs::verdict(true, "specs are up to date"), "up to date");
    assert_eq!(crate::specs::verdict(true, "specs updated: added x"), "ok");
    assert_eq!(crate::specs::verdict(false, &"boom ".repeat(100)), "failed");
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
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_of_spec_provenance_and_update_settings() {
    use inference_bench_core::fake::publish_specs;
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let specs_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/specs");
    let read = |f: &str| std::fs::read_to_string(format!("{specs_dir}/{f}")).unwrap();
    let newer = read("qwen3-0.6b.toml").replace("name = \"qwen3-0.6b\"", "name = \"qwen3-0.6b-new\"");
    let future = format!("min_engine = 3\n{}", read("qwen2.5-0.5b.toml").replace("name = \"qwen2.5-0.5b\"", "name = \"qwen2.5-0.5b-moe\""));
    let gh = Script::default();
    let server = FakeServer::start(gh.clone());
    publish_specs(
        &gh,
        "v0l/inference-bench",
        "refs/heads/master",
        "5f9d34982b950f61d8f0a238aff52c4676d8b9b1",
        "s",
        &[("qwen3-0.6b.toml", read("qwen3-0.6b.toml").as_bytes()), ("qwen3-0.6b-new.toml", newer.as_bytes()), ("qwen2.5-0.5b-moe.toml", future.as_bytes())],
    );
    let d = dir("spec-shot");
    let mine = d.join("my-specs");
    std::fs::create_dir_all(&mine).unwrap();
    std::fs::write(mine.join("llama.toml"), read("llama-3.2-1b.toml")).unwrap();
    let size = egui::vec2(1280.0, 1000.0);
    let mut h = Harness::builder().with_size(size).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
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
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_of_a_failed_spec_check_line() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("spec-failed-shot");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 420.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
        app.tab = Tab::Models;
        app.logs = shared_logs();
        app.library = vec![model("Qwen3-0.6B", true, "dense decoder")];
        let error = format!("https://api.github.com/repos/v0l/inference-bench/commits/refs/heads/master: {}", "download over 67108864 bytes or broken ".repeat(8));
        app.specs.last = Some((false, format!("spec update failed: {error}")));
        app.specs.state.checked_at = Some(chrono::Utc::now());
        app.specs.state.last_ok = false;
        app
    });
    h.run();
    h.render().unwrap().save(out.join("spec-check-failed.png")).unwrap();
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
    shot(&mut h, "server-logs");
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
            detail::View::new(name, spec, detail::Source::SpecToml, 1, Some(262144)).with_structure(header_structure("qwen3.8-27b-fp8"))
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
        let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
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

fn tree_hub() -> String {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let own = base.clone();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let own = own.clone();
            std::thread::spawn(move || {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                let path = line.split_whitespace().nth(1).unwrap().to_string();
                loop {
                    line.clear();
                    r.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                }
                let ids = |n: usize| format!("[{}]", (0..n).map(|i| format!(r#"{{"id":"u/m{i}","downloads":9,"likes":1}}"#)).collect::<Vec<_>>().join(","));
                let next = |filter: &str| format!("link: <{own}/api/models?filter={filter}&limit=1000&cursor=2>; rel=\"next\"\r\n");
                let page_two = path.contains("cursor=2");
                let (extra, body) = if path.starts_with("/api/models/Qwen/Qwen3.8-27B") {
                    (String::new(), br#"{"cardData":{}}"#.to_vec())
                } else if path.contains("filter=base_model%3Aquantized") && page_two {
                    (String::new(), ids(2).into_bytes())
                } else if path.contains("filter=base_model%3Aquantized") {
                    (next("base_model%3Aquantized%3AQwen%2FQwen3.8-27B"), ids(3).into_bytes())
                } else if path.contains("filter=base_model%3Afinetune") {
                    (String::new(), ids(4).into_bytes())
                } else if path.contains("filter=base_model%3Aadapter") {
                    (String::new(), ids(1).into_bytes())
                } else if path.contains("filter=base_model%3Amerge") {
                    (String::new(), b"[]".to_vec())
                } else {
                    (String::new(), b"{}".to_vec())
                };
                let _ = write!(s, "HTTP/1.1 200 OK\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            });
        }
    });
    base
}

fn hub_model(repo: &str, base: Option<&str>) -> LocalModel {
    let mut m = model(repo.rsplit('/').next().unwrap(), true, "dense decoder");
    m.name = repo.to_string();
    m.repo = repo.to_string();
    m.lineage = base.map(|b| inference_bench_core::library::Lineage { bases: vec![b.to_string()], relation: inference_bench_core::library::Relation::Quantized });
    m
}

fn tree_library(app: &mut App) {
    app.library = vec![
        hub_model("Qwen/Qwen3.8-27B-FP8", Some("Qwen/Qwen3.8-27B")),
        hub_model("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", Some("Qwen/Qwen3.8-27B")),
        hub_model("community/Qwen3.8-27B-FP8", None),
        hub_model("acme/Solo", None),
    ];
}

#[test]
fn two_models_with_one_short_name_show_distinct_names_and_only_the_loaded_one_is_marked() {
    let mut h = harness("samename", Tab::Models);
    {
        let app = h.state_mut();
        app.library = vec![hub_model("Qwen/Qwen3.8-27B-FP8", None), hub_model("community/Qwen3.8-27B-FP8", None)];
        scripted(app, Script { models: vec!["Qwen/Qwen3.8-27B-FP8".into(), "community/Qwen3.8-27B-FP8".into()], ..script() });
        let m = app.library[1].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    h.run();
    let app = h.state();
    assert_ne!(app.library[0].name, app.library[1].name);
    let marked: Vec<bool> = app.library.iter().map(|m| models::is_loaded(app, m)).collect();
    assert_eq!(marked, [false, true]);
    assert_eq!(app.loaded().unwrap().display, "community/Qwen3.8-27B-FP8");
}

#[test]
fn the_library_tree_puts_two_quants_under_their_base_and_search_matches_the_base() {
    let mut h = harness("libtree", Tab::Models);
    tree_library(h.state_mut());
    h.state_mut().models.show_all = true;
    h.run();
    let app = h.state();
    let rows = inference_bench_core::library::tree(&app.library, &[0, 1, 2, 3]);
    let shape: Vec<(&str, usize)> = rows.iter().map(|r| (r.name.as_str(), r.depth)).collect();
    assert_eq!(shape[0], ("Qwen/Qwen3.8-27B", 0));
    assert_eq!(&shape[1..3], [("Qwen/Qwen3.8-27B-FP8", 1), ("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", 1)]);
    h.get_by_label("find on hugging face");
    assert!(models::matches(&h.state().library[1], "Qwen3.8-27B uncensored"));
    assert!(models::matches(&h.state().library[1], "org/none") == false);
    let mut quant = h.state().library[3].clone();
    quant.lineage = h.state().library[0].lineage.clone();
    assert!(models::matches(&quant, "Qwen/Qwen3.8-27B"));
}

#[test]
fn the_detail_tree_reads_the_counts_from_the_hub_and_a_count_opens_discover_filtered() {
    let hub = tree_hub();
    let mut h = harness("detailtree", Tab::Models);
    {
        let app = h.state_mut();
        tree_library(app);
        app.models.hub_base = hub;
        let m = app.library[0].clone();
        app.models.detail = Some(detail::Detail::open(&m));
    }
    pump_until(&mut h, 20, "tree", |a| a.models.detail.as_ref().is_some_and(|d| d.tree.is_some()));
    let tree = h.state().models.detail.as_ref().unwrap().tree.clone().unwrap().unwrap();
    assert_eq!(tree.chain, ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B"]);
    let counts: Vec<u64> = tree.counts.iter().map(|c| c.1).collect();
    assert_eq!(counts, [1, 4, 0, 5]);
    h.run();
    h.get_by_label("quantizations 5").click();
    pump_until(&mut h, 20, "discover", |a| a.models.pane == models::Pane::Discover && !a.models.results.is_empty());
    let app = h.state();
    assert_eq!(app.models.results[0].id, "u/m0");
    assert_eq!(app.models.relation.as_ref().map(|r| (r.0, r.1.as_str())), Some((inference_bench_core::library::Relation::Quantized, "Qwen/Qwen3.8-27B")));
    assert!(app.models.detail.is_none());
}

#[test]
#[ignore = "renders pictures; run with SHOTS=/tmp/shots"]
fn screenshots_of_the_model_tree() {
    let out = PathBuf::from(std::env::var("SHOTS").unwrap_or("/tmp/shots".into()));
    std::fs::create_dir_all(&out).unwrap();
    let hub = tree_hub();
    let mut h = harness("treeshot", Tab::Models);
    tree_library(h.state_mut());
    h.state_mut().models.hub_base = hub;
    h.run();
    h.render().unwrap().save(out.join("models-tree.png")).unwrap();
    h.state_mut().models.tree_view = false;
    h.run();
    h.render().unwrap().save(out.join("models-flat.png")).unwrap();
    let m = h.state().library[0].clone();
    h.state_mut().models.detail = Some(detail::Detail::open(&m));
    pump_until(&mut h, 20, "tree", |a| a.models.detail.as_ref().is_some_and(|d| d.tree.is_some()));
    h.run();
    h.render().unwrap().save(out.join("detail-tree.png")).unwrap();
}

#[test]
#[ignore = "needs the hub cache and huggingface.co; run with SHOTS=/tmp/shots"]
fn screenshots_of_the_model_tree_with_the_real_hub_cache() {
    let out = PathBuf::from(std::env::var("SHOTS").unwrap_or("/tmp/shots".into()));
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness("realtree", Tab::Models);
    h.state_mut().library = inference_bench_core::library::scan_hub(&inference_bench_core::library::hub_dir());
    h.state_mut().models.show_all = true;
    assert!(!h.state().library.is_empty());
    h.run();
    h.render().unwrap().save(out.join("real-models-tree.png")).unwrap();
    let pick = std::env::var("PICK").unwrap_or("orcarouter/Qwen3.8-27B-Uncensored-NVFP4".into());
    let m = h.state().library.iter().find(|m| m.name == pick).cloned().unwrap_or_else(|| h.state().library[0].clone());
    h.state_mut().models.detail = Some(detail::Detail::open(&m));
    pump_until(&mut h, 60, "tree", |a| a.models.detail.as_ref().is_some_and(|d| d.tree.is_some()));
    h.run();
    h.render().unwrap().save(out.join("real-detail-tree.png")).unwrap();
}

#[test]
fn the_agent_config_follows_the_loaded_model() {
    let mut h = harness("agent-follow", Tab::Chat);
    h.run();
    assert!(h.state().chat.agent.chat.config.model.is_empty());
    {
        let app = h.state_mut();
        scripted(app, script());
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let config = h.state().chat.agent.chat.config.clone();
    assert_eq!(config.model, "Qwen3-0.6B");
    let server = h.state().server.agent.as_ref().expect("agent endpoint").url();
    assert_eq!(config.url, format!("{server}/v1"));
    assert!(config.url.starts_with("http://127.0.0.1:"));
    assert!(config.fault().is_none());
    assert_eq!(ureq::get(format!("{}/models", config.url)).call().unwrap().status(), 200);
    h.get_by_label_contains("Ask the model to work");

    h.state_mut().unload();
    pump_until(&mut h, 10, "unload", |a| a.loaded().is_none());
    h.run();
    assert!(h.state().chat.agent.chat.config.model.is_empty());
    h.get_by_label_contains("Load a model to chat");
}

#[test]
fn the_brief_names_the_app_and_the_loaded_model() {
    let mut h = harness("agent-brief", Tab::Chat);
    {
        let app = h.state_mut();
        scripted(app, script());
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let backend = h.state().chat.agent.chat.backend.clone();
    let brief = backend.brief(std::path::Path::new("/tmp/ws"), &[]);
    assert!(brief.contains("inference-bench") && brief.contains("Qwen3-0.6B") && brief.contains("/tmp/ws"), "{brief}");
    assert_eq!(backend.app(), "inference-bench");
}
