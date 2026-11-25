use super::*;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use inference_bench_core::chat::{ChatRequest, Message, Role, Sampling};
use inference_bench_core::engine::ChatEvent;
use inference_bench_core::fake::{FakeServer, Script, ScriptedModel};
use inference_bench_core::library::{Format, Support};

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ib-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

pub(crate) fn shots_dir() -> PathBuf {
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
        spec_source: None,
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
        offload: None,
        vision: false,
        files: Default::default(),
        sessions: Default::default(),
    }
}

fn harness(name: &str, tab: Tab) -> Harness<'static, App> {
    harness_sized(name, tab, 800.0)
}

fn harness_sized(name: &str, tab: Tab, height: f32) -> Harness<'static, App> {
    harness_at(name, tab, egui::vec2(1280.0, height))
}

pub(crate) fn harness_at(name: &str, tab: Tab, size: egui::Vec2) -> Harness<'static, App> {
    let d = dir(name);
    Harness::builder().with_size(size).with_max_steps(100).with_accessibility_check(false).allow_missing_glyphs().build_eframe(move |cc| {
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
    for tab in [Tab::Chat, Tab::Models, Tab::Server, Tab::Hardware, Tab::Settings] {
        let mut h = harness(&format!("tab{tab:?}"), tab);
        h.step();
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

fn click_load(h: &mut Harness<'static, App>) {
    let mut last = None;
    for _ in 0..30 {
        h.step();
        let at = h.get_all_by_label("load model").last().unwrap().accesskit_node().bounding_box();
        if at == last {
            break;
        }
        last = at;
    }
    h.get_all_by_label("load model").last().unwrap().click();
}

fn click_settled(h: &mut Harness<'static, App>, label: &str) {
    let mut last = None;
    for _ in 0..30 {
        h.step();
        let at = h.get_by_label_contains(label).accesskit_node().bounding_box();
        if at == last {
            break;
        }
        last = at;
    }
    h.get_by_label_contains(label).click();
}

pub(crate) fn with_gpu(h: &mut Harness<'static, App>, used: u64, total: u64) {
    let app = h.state_mut();
    app.gpu = vec![inference_bench_core::gpu::GpuStats { name: "test".into(), used_bytes: used, total_bytes: total, util_percent: None, uuid: String::new(), own_bytes: 0, display: false, index: None }];
    app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
    app.gpu_poll = Default::default();
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
    assert_eq!(dialog.max, spec_max * 16);
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
    assert_eq!(dialog.max, 262144 * 16);
    assert_eq!(crate::load::max_context(&dialog.view.as_ref().unwrap().anatomy), 262144);
}

#[test]
fn the_load_dialog_judges_fit_across_every_gpu_like_discover_does() {
    let mut h = harness("load-three-gpus", Tab::Models);
    let gpu = |total: u64| inference_bench_core::gpu::GpuStats { name: "card".into(), used_bytes: 0, total_bytes: total << 30, util_percent: None, uuid: String::new(), own_bytes: 0, display: false, index: None };
    {
        let app = h.state_mut();
        app.gpu = vec![gpu(24), gpu(24), gpu(24)];
        app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
        app.gpu_poll = Default::default();
    }
    let d = dir("three-gpus-27b");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), d.join("config.json")).unwrap();
    let mut m = model("Qwen3.8-27B", true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    let list = h.state().free_gpu_list();
    let dialog = h.state().load.as_ref().unwrap();
    let weights = dialog.anatomy().unwrap().total_bytes;
    assert!(weights > list[0], "the model must be too big for GPU 0 alone");
    assert_eq!(h.state().free_gpu_bytes(), Some(list.iter().sum()));
    let devices = inference_bench_core::discover::pool_devices(&list);
    let r = dialog.planned(&devices);
    assert!(matches!(r, Some(Ok(_))), "{:?} {:?} {:?}", r.map(|r| r.map(|p| p.placement.render())), devices.iter().map(|d| d.memory_gib).collect::<Vec<_>>(), dialog.seqs);
    h.get_by_label_contains("free on 3 gpus");
    h.get_by_label_contains("load plan");
    assert!(!h.query_all_by_label_contains("layers 0-").collect::<Vec<_>>().is_empty());
}

#[test]
fn load_stays_available_when_the_planner_says_the_model_will_not_fit() {
    let mut h = harness("load-anyway", Tab::Models);
    with_gpu(&mut h, 95 << 30, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.state_mut().load.as_mut().unwrap().ctx = 40960;
    h.run();
    assert!(h.query_all_by_label_contains("still try to load").count() >= 1);
    assert!(h.query_by_label_contains("gpu 0").is_none());
    let load = click_load_node(&mut h);
    assert!(!load, "the load key must not be disabled");
}

fn click_load_node(h: &mut Harness<'static, App>) -> bool {
    h.query_all_by_label("load model").last().expect("a load key").accesskit_node().is_disabled()
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
    h.get_by_label_contains("planner says it is");
    assert!(!click_load_node(&mut h));
    assert!(h.query_by_label("fits in free memory").is_none());
    let e = h.state().load.as_ref().unwrap().estimate(h.state().free_gpu_bytes()).unwrap();
    assert_eq!(e.fits(), Some(false));
}

#[test]
fn the_default_context_is_the_default_and_a_remembered_one_wins_whatever_the_memory() {
    let a = detail_view("27b", Some("qwen3.8-27b-exact.toml")).anatomy;
    let max = crate::load::max_context(&a);
    assert_eq!(crate::load::initial_ctx(None, 8192, max), 8192);
    assert_eq!(crate::load::initial_ctx(Some(100_000), 8192, max), 100_000);
    assert_eq!(crate::load::initial_ctx(Some(999_999), 8192, max), max);
}

fn loading_harness(name: &str, gate: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Harness<'static, App> {
    use specrt::progress::Stage;
    let mut h = harness(name, Tab::Chat);
    let steps = vec![(Stage::Kernels, 1, 1), (Stage::Weights, 0, 30_000_000_000), (Stage::Weights, 12_000_000_000, 30_000_000_000)];
    h.state_mut().engine = inference_bench_core::engine::Engine::with_factory(h.state().wake.clone(), ScriptedModel::factory_loading(script(), steps, 20, Some(gate)));
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().load_model_with(&m, 4096, 1);
    h
}

#[test]
fn the_model_bar_and_the_chat_show_load_progress_with_the_stage_and_bytes() {
    let gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut h = loading_harness("load-progress", gate.clone());
    pump_until(&mut h, 10, "progress", |a| a.load_progress.as_ref().is_some_and(|p| p.done == 12_000_000_000));
    h.run();
    let p = h.state().load_progress.clone().unwrap();
    assert_eq!(p.stage, "reading weights");
    assert_eq!(p.total, 30_000_000_000);
    assert!(h.query_all_by_label_contains("reading weights").count() >= 1);
    assert!(h.query_all_by_label_contains("12.0 / 30.0 GB").count() >= 1, "bytes done are shown");
    h.get_by_label_contains("Loading Qwen3-0.6B");
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    h.render().unwrap().save(out.join("load-progress.png")).unwrap();
    gate.store(true, std::sync::atomic::Ordering::SeqCst);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    assert!(h.state().load_progress.is_none(), "the bar goes away once the model is loaded");
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
    click_load(&mut h);
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
        app.server.start(handle, 0, "", false, true);
    }
    h.run();
    let url = h.state().server.running.as_ref().expect("server running").url();
    let mut resp = ureq::get(format!("{url}/v1/models")).call().unwrap();
    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert_eq!(v["data"][0]["id"], "Qwen3-0.6B");
    h.run();
    let mut resp = ureq::post(format!("{url}/v1/chat/completions"))
        .send_json(serde_json::json!({"model":"yesterdays-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8}))
        .unwrap();
    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert_eq!(v["model"], "yesterdays-model");
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

const WEATHER_XML: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nDublin\n</parameter>\n</function>\n</tool_call>";

fn weather_tools() -> serde_json::Value {
    serde_json::json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
}

fn tool_server(h: &mut Harness<'static, App>, tools: inference_bench_core::specrt_chat::ToolFormat) -> String {
    {
        let app = h.state_mut();
        let script = Script { chunks: vec![WEATHER_XML.into()], ..script() };
        let (factory, _) = ScriptedModel::factory_with_tools(script, Default::default(), tools);
        app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), factory);
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(h, 20, "load", |a| a.loaded().is_some());
    let handle = h.state().engine.handle();
    h.state_mut().server.start(handle, 0, "", false, false);
    h.state().server.running.as_ref().expect("server running").url()
}

#[test]
fn the_app_server_parses_tool_calls_for_a_qwen_template() {
    use inference_bench_core::specrt_chat::ToolFormat;
    let mut h = harness("toolcalls", Tab::Server);
    let url = tool_server(&mut h, ToolFormat::XmlFunction);
    let body = |stream: bool| serde_json::json!({"messages":[{"role":"user","content":"weather?"}],"tools":weather_tools(),"stream":stream,"max_tokens":64});
    let mut resp = ureq::post(format!("{url}/v1/chat/completions")).send_json(body(false)).unwrap();
    let v: serde_json::Value = resp.body_mut().read_json().unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{v}");
    assert_eq!(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    let args: serde_json::Value = serde_json::from_str(choice["message"]["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"city": "Dublin"}));
    assert!(choice["message"]["content"].as_str().is_none_or(str::is_empty), "{v}");
    let mut resp = ureq::post(format!("{url}/v1/chat/completions")).send_json(body(true)).unwrap();
    let text = resp.body_mut().read_to_string().unwrap();
    assert!(text.contains("\"tool_calls\"") && text.contains("get_weather") && text.contains("Dublin"), "{text}");
    assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
    assert!(!text.contains("<function="), "{text}");
    let mut resp = ureq::get(format!("{url}/props")).call().unwrap();
    let props: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert_eq!(props["tool_parser"], "xml_function");
    h.state_mut().server.stop();
}

#[test]
fn the_app_server_reports_a_tool_format_it_cannot_parse() {
    use inference_bench_core::specrt_chat::ToolFormat;
    let mut h = harness("toolcalls-unparsed", Tab::Server);
    let url = tool_server(&mut h, ToolFormat::Unparsed);
    let mut resp = ureq::get(format!("{url}/props")).call().unwrap();
    let props: serde_json::Value = resp.body_mut().read_json().unwrap();
    assert!(props["tool_parser"].is_null());
    assert_eq!(props["tool_format"], "Unparsed");
    h.state_mut().server.stop();
}

struct XmlThenAnswer;

impl inference_bench_core::native::Model for XmlThenAnswer {
    fn generate(&mut self, _: &str, _: usize, _: specrt::runtime::Sampling) -> anyhow::Result<specrt::engine::Reply> {
        anyhow::bail!("chat only")
    }

    fn chat(&mut self, messages: &[serde_json::Value], _: Option<&serde_json::Value>, _: usize, _: specrt::runtime::Sampling, on_text: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<specrt::engine::Reply> {
        let text = if messages.iter().any(|m| m["role"] == "tool") { "Nothing more to check.\n</think>\n\nThe workspace is ready." } else { "The user wants the workspace checked, so I will call the workspace tool first.\n</think>\n\n<tool_call>\n<function=workspace>\n</function>\n</tool_call>" };
        on_text(text);
        Ok(specrt::engine::Reply { text: text.into(), prompt_tokens: 4, completion_tokens: 4, prefill_secs: 0.0, decode_secs: 0.0, peak_device_bytes: 0 })
    }

    fn logits(&mut self, _: &str) -> anyhow::Result<specrt::engine::PromptLogits> {
        anyhow::bail!("chat only")
    }

    fn tool_format(&self) -> specrt::chat::ToolFormat {
        specrt::chat::ToolFormat::XmlFunction
    }

    fn thinking(&self) -> specrt::chat::Thinking {
        specrt::chat::Thinking::OpensInPrompt
    }
}

#[test]
fn the_agent_chat_runs_a_tool_the_model_asked_for_in_xml() {
    use agent_ui::chat::Turn;
    let mut h = harness("agent-tools", Tab::Chat);
    let work = dir("agent-tools-ws");
    std::fs::create_dir_all(&work).unwrap();
    {
        let app = h.state_mut();
        let factory: inference_bench_core::native::Factory = std::sync::Arc::new(|_, _| Ok(Box::new(XmlThenAnswer) as Box<dyn inference_bench_core::native::Model>));
        app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), factory);
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let (desk, _asks) = agent_core::Desk::new(|| {});
    {
        let chat = &mut h.state_mut().chat.agent.chat;
        chat.config.workspace = work.display().to_string();
        chat.ask("check the workspace", desk);
    }
    pump_until(&mut h, 20, "agent turn", |a| !a.chat.agent.chat.busy());
    let turns = &h.state().chat.agent.chat.turns;
    let did = turns.iter().find_map(|t| match t {
        Turn::Did { name, answer, .. } => Some((name.clone(), answer.clone())),
        _ => None,
    });
    let (name, answer) = did.unwrap_or_else(|| panic!("no tool turn: {} turns", turns.len()));
    assert_eq!(name, "workspace");
    assert!(matches!(answer, Some(Ok(_))), "{answer:?}");
    assert!(turns.iter().any(|t| matches!(t, Turn::Said(s) if s.contains("workspace is ready"))));
    assert!(!turns.iter().any(|t| matches!(t, Turn::Said(s) if s.contains("<function="))));
}

#[test]
fn the_agent_chat_shows_the_reasoning_parser_output_as_thinking() {
    use agent_ui::chat::Turn;
    let mut h = harness("agent-reasoning", Tab::Chat);
    {
        let app = h.state_mut();
        let script = Script { chunks: vec!["weigh the options\n".into(), "</think>\n\n".into(), "The answer is 42.".into()], ..script() };
        let (factory, _) = ScriptedModel::factory_probed(script, inference_bench_core::specrt_chat::Thinking::OpensInPrompt);
        app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), factory);
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let (desk, _asks) = agent_core::Desk::new(|| {});
    h.state_mut().chat.agent.chat.ask("what is the answer", desk);
    pump_until(&mut h, 20, "agent turn", |a| !a.chat.agent.chat.busy());
    let turns = &h.state().chat.agent.chat.turns;
    assert!(turns.iter().any(|t| matches!(t, Turn::Thinking(s) if s.contains("weigh the options"))), "no thinking turn in {} turns", turns.len());
    assert!(turns.iter().any(|t| matches!(t, Turn::Said(s) if s.contains("The answer is 42.") && !s.contains("think>") && !s.contains("weigh"))));
    assert!(!turns.iter().any(|t| matches!(t, Turn::Fault(_))));
    let thoughts: Vec<String> = turns.iter().filter_map(|t| match t { Turn::Thinking(s) => Some(s.trim().to_string()), _ => None }).collect();
    assert_eq!(thoughts, ["weigh the options"]);
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_the_chat_with_thinking_and_a_tool_row() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness_sized("shots-chat", Tab::Chat, 800.0);
    let work = dir("shots-chat-ws");
    std::fs::create_dir_all(&work).unwrap();
    {
        let app = h.state_mut();
        let factory: inference_bench_core::native::Factory = std::sync::Arc::new(|_, _| Ok(Box::new(XmlThenAnswer) as Box<dyn inference_bench_core::native::Model>));
        app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), factory);
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let (desk, _asks) = agent_core::Desk::new(|| {});
    {
        let chat = &mut h.state_mut().chat.agent.chat;
        chat.config.workspace = work.display().to_string();
        chat.ask("check the workspace", desk);
    }
    pump_until(&mut h, 20, "agent turn", |a| !a.chat.agent.chat.busy());
    h.run();
    h.render().unwrap().save(out.join("chat-thinking-tool-composer.png")).unwrap();
}

fn shared_transcript_harness(name: &str) -> Harness<'static, App> {
    use agent_ui::chat::{Attached, Turn};
    let mut h = harness(name, Tab::Chat);
    {
        let app = h.state_mut();
        let factory: inference_bench_core::native::Factory = std::sync::Arc::new(|_, _| Ok(Box::new(XmlThenAnswer) as Box<dyn inference_bench_core::native::Model>));
        app.engine = inference_bench_core::engine::Engine::with_factory(app.wake.clone(), factory);
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 10, "model never loaded", |a| a.loaded().is_some());
    h.run();
    let png = Attached { name: "red.png".into(), mime: "image/png".into(), data: std::sync::Arc::new(red_png()) };
    let turns = &mut h.state_mut().chat.agent.chat.turns;
    turns.push(Turn::You { text: "what is in this picture".into(), files: vec![png] });
    turns.push(Turn::Thinking("A red rectangle, so I should say that.".into()));
    turns.push(Turn::Did { name: "workspace".into(), args: "{}".into(), answer: Some(Ok("The workspace is /tmp/ws.".into())) });
    turns.push(Turn::Said("It is a red rectangle.".into()));
    h.run();
    h
}

#[test]
fn the_chat_draws_reasoning_an_attachment_and_a_tool_row_with_the_shared_widgets() {
    let mut h = shared_transcript_harness("shared-transcript");
    h.run();
    h.get_by_label_contains("what is in this picture");
    h.get_by_label_contains("It is a red rectangle.");
    h.get_by_label_contains("A red rectangle, so I should say that.");
    h.get_by_role_and_label(egui::accesskit::Role::Button, "workspace");
    h.get_by_label_contains("The workspace is /tmp/ws.");
    let turns = &h.state().chat.agent.chat.turns;
    assert!(turns.iter().any(|t| matches!(t, agent_ui::chat::Turn::You { files, .. } if files.len() == 1 && files[0].is_image())));
    assert!(turns.iter().any(|t| matches!(t, agent_ui::chat::Turn::Thinking(_))));
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_the_chat_with_reasoning_attachment_and_tool_row() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = shared_transcript_harness("shots-shared");
    h.run();
    h.render().unwrap().save(out.join("chat-shared-widgets.png")).unwrap();
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
        app.engine.send(Command::Load { alongside: false, model_id: m.model_id(inference_bench_core::target::NATIVE), display: m.name.clone(), max_ctx: Some(4096), max_num_seqs: Some(seqs), kv_cache_bytes: None, served_names: Vec::new(), engine: Default::default() });
    }
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    let handle = h.state().engine.handle();
    h.state_mut().server.start(handle, 0, "", false, false);
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
    h.state_mut().load.as_mut().unwrap().advanced = true;
    h.run();
    h.get_by_label_contains("PARALLEL REQUESTS");
    let free = h.state().free_gpu_bytes();
    let dialog = h.state_mut().load.as_mut().unwrap();
    dialog.ctx = 8192;
    dialog.seqs = 1;
    let one = dialog.estimate(free).unwrap();
    dialog.seqs = 4;
    dialog.kv_gb = 1.0;
    let four = dialog.estimate(free).unwrap();
    assert_eq!(four.kv, 1 << 30);
    assert_eq!(dialog.max, crate::load::max_context(&dialog.view.as_ref().unwrap().anatomy) * 16);
    assert_eq!(four.weights, one.weights);
    let a = detail_view("seqs-27b", Some("qwen3.8-27b-exact.toml")).anatomy;
    let (a1, a4) = (crate::load::estimate(&a, 1 << 30, 1, None), crate::load::estimate(&a, 1 << 30, 4, None));
    assert_eq!(a4.kv, a1.kv);
    assert_eq!(a4.state, a1.state * 4);
    assert!(a.state_bytes_per_seq > 100 << 20);
    assert_eq!(crate::load::largest_seqs(&a, Some(a.total_bytes)), 1);
    assert_eq!(crate::load::largest_seqs(&a, None), crate::load::MAX_SEQS);
}

#[test]
fn the_kv_pool_is_shared_and_must_hold_one_full_length_request() {
    let a = detail_view("pool-27b", Some("qwen3.8-27b-exact.toml")).anatomy;
    let max = crate::load::max_context(&a);
    let pool = a.kv_bytes_per_token * max as u64 * 3 / 2;
    assert!(crate::load::pool_holds_one(&a, pool, max));
    assert!((crate::load::concurrency(&a, pool, max) - 1.5).abs() < 0.01);
    assert!(!crate::load::pool_holds_one(&a, a.kv_bytes_per_token * (max as u64 - 1), max));
    let free = Some(a.total_bytes + a.state_bytes_per_seq * 8 + (40u64 << 30));
    let gb = crate::load::default_kv_gb(&a, free, 8, max);
    assert!(gb * (1u64 << 30) as f64 <= 40.0 * (1u64 << 30) as f64);
    assert!(gb > 1.0);
}

#[test]
fn the_load_dialog_slider_runs_to_the_model_max_with_four_parallel_requests() {
    let mut h = harness("slider-max", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    let d = h.state_mut().load.as_mut().unwrap();
    d.seqs = 4;
    let a = &d.view.as_ref().unwrap().anatomy;
    assert_eq!(d.max, crate::load::max_context(a) * 16);
    d.ctx = crate::load::max_context(a);
    assert!(d.estimate(Some(96 << 30)).unwrap().kv <= (96u64 << 30));
}

#[test]
fn loading_from_the_dialog_passes_parallel_requests_to_the_engine_and_remembers_them() {
    let mut h = harness("seqs-load", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.state_mut().load.as_mut().unwrap().seqs = 6;
    h.state_mut().load.as_mut().unwrap().pin_seqs = true;
    h.run();
    click_load(&mut h);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    assert_eq!(h.state().loaded().unwrap().max_num_seqs, 6);
    assert_eq!(h.state().settings.seqs_for("Qwen3-0.6B"), 6);
    assert_eq!(Settings::load(&h.state().dir).seqs_for("Qwen3-0.6B"), 6);
}

fn offloaded_script() -> Script {
    use specrt::offload::{LayerResidency, OffloadPlan, OffloadStats, Tier};
    let layers = (0..4).map(|i| LayerResidency { layer: i, bytes: 2 << 30, tier: if i < 2 { Tier::Vram } else if i == 2 { Tier::Host } else { Tier::Nvme }, pinned: i < 2 }).collect();
    let plan = OffloadPlan { layers, staging_bytes: 1 << 30, vram_bytes: 5 << 30, host_bytes: 2 << 30, nvme_bytes: 2 << 30, host_staging_bytes: 0, swap_staging_bytes: 0 };
    let stats = OffloadStats { vram_hits: 3, host_hits: 1, nvme_reads: 0, host_to_vram_bytes: 10, ..Default::default() };
    Script { offload: Some((plan, stats)), ..script() }
}

#[test]
fn where_the_weights_live_and_the_hit_rate_show_for_an_offloaded_model_at_desktop_and_phone_width() {
    for (name, size) in [("tiers-desktop", egui::vec2(1280.0, 1400.0)), ("tiers-phone", egui::vec2(390.0, 2400.0))] {
        let mut h = harness_at(name, Tab::Server, size);
        with_gpu(&mut h, 0, 96 << 30);
        h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), offloaded_script());
        let m = real_model(&h.state().library[0].name.clone());
        h.state_mut().library[0] = m.clone();
        h.state_mut().load_model_with(&m, 2048, 2);
        pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
        pump_until(&mut h, 10, "serving reading", |a| a.engine.reading().serving.is_some());
        h.run();
        assert!(h.state().engine.handle().serving_stats_of(&m.name).and_then(|s| s.offload_plan).is_some());
        h.get_by_label_contains("2 of 4 layers stream in");
        let screen = h.ctx.content_rect();
        for text in ["2 of 4 layers stream in", m.name.as_str()] {
            for n in h.get_all_by_label_contains(text) {
                let r = n.rect();
                assert!(r.right() <= screen.right() + 0.5, "{text} at {size:?} ends at {} past {}", r.right(), screen.right());
            }
        }
        let r = h.state().engine.reading();
        let shown = crate::serving::offload_readings(&r, 2.5e9);
        assert_eq!(shown[0].1, "75%");
        assert_eq!(shown[1].1, "2.50 GB/s");
        let plan = h.state().engine.handle().serving_stats_of(&m.name).unwrap().offload_plan.unwrap();
        let cells = crate::tiers::tier_cells(&plan);
        assert_eq!(cells.iter().map(|c| c.0).collect::<Vec<_>>(), ["vram", "host ram", "nvme"]);
    }
}

static PAINTED_TEXT: std::sync::Mutex<Vec<(String, egui::Rect)>> = std::sync::Mutex::new(Vec::new());

fn capture_painted_text(ctx: &egui::Context) {
    ctx.on_end_pass(
        "painted-text",
        std::sync::Arc::new(|ui: &mut egui::Ui| {
            let ctx = ui.ctx().clone();
            let found: Vec<(String, egui::Rect)> = ctx.graphics(|g| {
                [egui::LayerId::background()]
                    .into_iter()
                    .filter_map(|id| g.get(id))
                    .flat_map(|list| list.all_entries())
                    .filter_map(|c| match &c.shape {
                        egui::Shape::Text(t) if !t.galley.text().trim().is_empty() => Some((t.galley.text().to_string(), c.shape.visual_bounding_rect())),
                        _ => None,
                    })
                    .collect()
            });
            *PAINTED_TEXT.lock().unwrap() = found;
        }),
    );
}

fn server_tab_painted_text(width: f32, name: &str, gpus: usize) -> Vec<(String, egui::Rect)> {
    let mut h = harness_at(&format!("clip-{width}-{gpus}"), Tab::Server, egui::vec2(width, 2400.0));
    capture_painted_text(&h.ctx);
    for i in 0..3 {
        tracing::info!(method = "POST", path = "/v1/chat/completions", status = 200, elapsed_ms = 1234 + i, "request finished");
    }
    with_gpu(&mut h, 0, 96 << 30);
    let one = h.state().gpu[0].clone();
    h.state_mut().gpu = vec![one; gpus];
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), offloaded_script());
    let mut m = real_model(&h.state().library[0].name.clone());
    m.name = name.to_string();
    h.state_mut().library[0] = m.clone();
    h.state_mut().load_model_with(&m, 2048, 2);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    h.step();
    let found = PAINTED_TEXT.lock().unwrap().clone();
    found
}

#[test]
fn server_tab_text_stays_inside_the_screen_from_360_to_600_wide() {
    let _guard = PAINTED_TEXT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut offenders = Vec::new();
    for width in [360.0f32, 390.0, 412.0, 600.0] {
        for (name, gpus) in [("tiny-model", 1), ("tiny-org/a-long-model-name-for-a-phone-instruct-build", 2)] {
            let found = server_tab_painted_text(width, name, gpus);
            assert!(found.len() > 20, "only {} text shapes at {width}", found.len());
            for (text, r) in found.iter() {
                if r.max.x > width + 0.5 {
                    offenders.push(format!("width {width}, {gpus} gpu, {name}: {text:?} ends at {} past the screen", r.max.x));
                }
            }
        }
    }
    assert!(offenders.is_empty(), "{}", offenders.join("\n"));
}

static PAINTED_TEXT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshot_of_the_server_tab_with_a_long_model_name_at_phone_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness_at("server-phone-shot", Tab::Server, egui::vec2(390.0, 844.0));
    with_gpu(&mut h, 0, 96 << 30);
    let one = h.state().gpu[0].clone();
    h.state_mut().gpu = vec![one; 2];
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), offloaded_script());
    let mut m = real_model(&h.state().library[0].name.clone());
    m.name = "tiny-org/a-long-model-name-for-a-phone-instruct-build".to_string();
    h.state_mut().library[0] = m.clone();
    h.state_mut().load_model_with(&m, 2048, 2);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    h.render().unwrap().save(out.join("server-phone-390x844.png")).unwrap();
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_status_bar_with_a_long_model_name_at_phone_and_desktop_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (name, size) in [("status-phone-390x844", egui::vec2(390.0, 844.0)), ("status-desktop-1280x800", egui::vec2(1280.0, 800.0))] {
        let mut h = harness_at(name, Tab::Server, size);
        with_gpu(&mut h, 0, 96 << 30);
        let one = h.state().gpu[0].clone();
        h.state_mut().gpu = vec![one; 2];
        h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), offloaded_script());
        let mut m = real_model(&h.state().library[0].name.clone());
        m.name = "tiny-org/a-long-model-name-for-a-phone-instruct-build".to_string();
        h.state_mut().library[0] = m.clone();
        h.state_mut().load_model_with(&m, 2048, 2);
        pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
        h.run();
        h.render().unwrap().save(out.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_weight_tiers_and_hit_rate_at_desktop_and_phone_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (name, size) in [("tiers-desktop", egui::vec2(1280.0, 1500.0)), ("tiers-phone", egui::vec2(390.0, 2600.0))] {
        let mut h = harness_at(name, Tab::Server, size);
        with_gpu(&mut h, 0, 96 << 30);
        h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), offloaded_script());
        let m = real_model(&h.state().library[0].name.clone());
        h.state_mut().library[0] = m.clone();
        h.state_mut().load_model_with(&m, 2048, 2);
        pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
        pump_until(&mut h, 10, "serving reading", |a| a.engine.reading().serving.is_some());
        h.run();
        h.render().unwrap().save(out.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
fn without_offload_no_weight_readings_show() {
    let r = inference_bench_core::live::Reading { generated: 0, prefilled: 0, requests: 0, running: 0, waiting: 0, max_seqs: 1, ttft: specrt::serve::Histogram::new(&[1.0]), itl: specrt::serve::Histogram::new(&[1.0]), serving: Some(specrt::serve::Stats::default()) };
    assert!(crate::serving::offload_readings(&r, 0.0).is_empty());
}

#[test]
fn the_server_tab_kv_total_is_the_shared_pool_chosen_in_the_dialog() {
    let mut h = harness("kv-total", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    let d = h.state_mut().load.as_mut().unwrap();
    d.seqs = 4;
    d.pin_seqs = true;
    d.kv_gb = 3.0;
    d.pin_kv = true;
    let pool = d.kv_bytes().unwrap();
    h.run();
    click_load(&mut h);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    let start = Instant::now();
    let reading = loop {
        let r = h.state().engine.reading();
        if r.serving.is_some() {
            break r;
        }
        assert!(start.elapsed() < Duration::from_secs(10), "no serving reading");
        h.step();
        std::thread::sleep(Duration::from_millis(50));
    };
    let s = reading.serving.as_ref().unwrap();
    assert_eq!(s.kv_blocks_total as u64 * s.kv_block_bytes as u64, pool);
    let shown = crate::serving::readings(&reading, 0.0);
    let kv = shown.iter().find(|(l, _, _)| *l == "kv cache").unwrap();
    assert_eq!(kv.1, "0 MB / 3.0 GB");
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
    click_load(&mut h);
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
    let qwen = h.state().library.iter().find(|m| m.name == "Qwen/Qwen3-0.6B").cloned().expect("Qwen3-0.6B in the hub cache");
    let mut rates = Vec::new();
    for seqs in [1usize, 4] {
        h.state_mut().load_model_with(&qwen, 4096, seqs);
        pump_until(&mut h, 180, "load", |a| a.loaded().is_some_and(|l| l.max_num_seqs == seqs));
        if h.state().server.running.is_none() {
            let handle = h.state().engine.handle();
            h.state_mut().server.start(handle, 0, "", false, false);
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
fn adding_a_model_folder_saves_it_and_rescans() {
    let mut h = harness("addfolder", Tab::Settings);
    h.state_mut().settings_ui.new_folder = "/tmp/more-models".into();
    h.run();
    h.get_all_by_label("add").next().unwrap().click();
    h.run();
    assert_eq!(Settings::load(&h.state().dir).model_folders, vec![PathBuf::from("/tmp/more-models")]);
    assert!(h.state().scan.busy() || h.state().notice.is_some());
    h.get_by_label("rescan models");
}

#[test]
fn downloading_the_cuda_compiler_shows_megabytes_in_the_load_bar() {
    let p = crate::LoadProgress { stage: "downloading the CUDA compiler".into(), done: 47_000_000, total: 95_436_000, weights_started: None };
    assert_eq!(crate::modelbar::progress_text(&p), "downloading the CUDA compiler  47 / 95 MB");
    assert!((crate::modelbar::progress_fraction(&p) - 0.49).abs() < 0.01);
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
    h.get_by_label("check specs").click();
    pump_until(&mut h, 20, "update and rescan", |a| !a.specs.busy() && !a.scan.busy() && a.library.iter().any(|m| m.name == "landed-from-github"));
    h.run();
    let landed = h.state().library.iter().find(|m| m.name == "landed-from-github").unwrap().clone();
    assert!(landed.origin.as_ref().unwrap().label().starts_with("updated from GitHub v0l/inference-bench@master (feed42a) at "));
    h.get_by_label("landed-from-github");
    assert_eq!(h.state().notice.as_ref().unwrap().1, "specs updated: 1 added");
    assert!(h.state().specs.state.last_ok);
}

#[test]
fn a_300_character_spec_error_leaves_the_check_key_its_full_width() {
    let mut h = harness("specs-long-error", Tab::Models);
    h.run();
    let plain = h.get_by_label("check specs").rect();
    assert!(plain.width() > 90.0, "{plain:?}");
    let error = format!("https://example.invalid/{}: {}", "x".repeat(60), "download over 67108864 bytes or broken ".repeat(8));
    assert!(error.len() >= 300);
    {
        let app = h.state_mut();
        app.specs.last = Some((false, format!("spec update failed: {error}")));
        app.specs.state.last_ok = false;
    }
    h.run();
    let key = h.get_by_label("check specs").rect();
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

pub(crate) fn pump_until(h: &mut Harness<'static, App>, secs: u64, what: &str, done: impl Fn(&App) -> bool) {
    let start = Instant::now();
    while !done(h.state()) {
        assert!(start.elapsed() < Duration::from_secs(secs.max(60)), "{what}: {:?}", h.state().notice);
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
    h.get_by_label("check specs").click();
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
            app.server.start(handle, 0, "", false, false);
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
    h.run_steps(12);
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

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_load_dialog_with_a_shared_kv_pool() {
    load_dialog_shot("load-dialog-shared-kv.png", 20 << 30);
    load_dialog_shot("load-dialog-shared-kv-over.png", 50 << 30);
}

fn load_dialog_shot(file: &str, used: u64) {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness("shot-load", Tab::Models);
    with_gpu(&mut h, used, 96 << 30);
    let d = dir(&format!("shot-load-{used}"));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), d.join("config.json")).unwrap();
    let mut m = model("Qwen3.8-27B", true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    let free = h.state().free_gpu_bytes();
    let dialog = h.state_mut().load.as_mut().unwrap();
    dialog.ctx = dialog.max;
    dialog.seqs = 8;
    let a = &dialog.view.as_ref().unwrap().anatomy;
    dialog.kv_gb = crate::load::default_kv_gb(a, free, 8, dialog.ctx);
    h.run();
    h.render().unwrap().save(out.join(file)).unwrap();
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
fn the_detail_header_chips_do_not_overlap_the_compare_label_at_1280() {
    let mut h = harness("headerfit", Tab::Models);
    let view = detail_view("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", Some("qwen3.8-27b-exact.toml"));
    let mut view = view;
    view.name = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4".into();
    let mut d = detail::Detail::from_view(view);
    d.model = "orcarouter/Qwen3.8-27B-Uncensored-NVFP4".into();
    h.state_mut().models.detail = Some(d);
    h.run();
    let compare = h.get_by_label_contains("compare with").rect();
    let chips: Vec<egui::Rect> = h.query_all_by_label_contains("header chip").map(|n| n.rect()).collect();
    assert!(chips.len() >= 2, "{}", chips.len());
    for (i, c) in chips.iter().enumerate() {
        assert!(!c.intersects(compare), "chip {i} {c:?} overlaps {compare:?}");
        for o in &chips[i + 1..] {
            assert!(!c.intersects(*o), "{c:?} {o:?}");
        }
    }
}

#[test]
fn the_library_list_is_flat_and_search_matches_the_base() {
    let mut h = harness("libflat", Tab::Models);
    tree_library(h.state_mut());
    h.state_mut().models.show_all = true;
    h.run();
    assert!(h.query_by_label("find on hugging face").is_none());
    assert!(h.query_by_label_contains("group by base model").is_none());
    assert!(h.query_by_label_contains("not downloaded").is_none());
    h.get_by_label_contains("Qwen/Qwen3.8-27B-FP8");
    h.get_by_label_contains("orcarouter/Qwen3.8-27B-Uncensored-NVFP4");
    assert!(models::matches(&h.state().library[1], "Qwen3.8-27B uncensored"));
    assert!(!models::matches(&h.state().library[1], "org/none"));
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
fn the_detail_tree_shows_the_chain_and_local_siblings_and_a_click_opens_one() {
    let hub = tree_hub();
    let mut h = harness("detailsib", Tab::Models);
    {
        let app = h.state_mut();
        tree_library(app);
        app.models.hub_base = hub;
        let m = app.library[0].clone();
        app.models.detail = Some(detail::Detail::open(&m));
    }
    pump_until(&mut h, 20, "tree", |a| a.models.detail.as_ref().is_some_and(|d| d.tree.is_some()));
    let t = h.state().models.detail.as_ref().unwrap().tree.clone().unwrap().unwrap();
    let rows = detail::lineage_rows(&t.chain, &t.relations, &h.state().library);
    let shape: Vec<(&str, usize, Option<usize>, bool)> = rows.iter().map(|r| (r.name.as_str(), r.depth, r.local, r.current)).collect();
    assert_eq!(
        shape,
        [
            ("Qwen/Qwen3.8-27B", 0, None, false),
            ("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", 1, Some(1), false),
            ("Qwen/Qwen3.8-27B-FP8", 1, Some(0), true),
        ]
    );
    for _ in 0..8 {
        h.step();
    }
    h.get_by_label("find on hugging face");
    h.get_by_label_contains("this model");
    h.get_by_label("open orcarouter/Qwen3.8-27B-Uncensored-NVFP4").click();
    pump_until(&mut h, 20, "opened", |a| a.models.detail.as_ref().is_some_and(|d| d.model == "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"));
    assert_eq!(h.state().models.detail.as_ref().unwrap().model, "orcarouter/Qwen3.8-27B-Uncensored-NVFP4");
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
    h.render().unwrap().save(out.join("real-models-flat.png")).unwrap();
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

fn open_dialog(name: &str) -> Harness<'static, App> {
    let mut h = harness_sized(name, Tab::Models, 2600.0);
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().engine = inference_bench_core::fake::engine(h.state().wake.clone(), script());
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h
}

fn token_counts_have_thousands_separators_and_long_names_shorten_in_the_middle() {
    assert_eq!(crate::load::tokens(0), "0");
    assert_eq!(crate::load::tokens(999), "999");
    assert_eq!(crate::load::tokens(8192), "8,192");
    assert_eq!(crate::load::tokens(524288), "524,288");
    assert_eq!(crate::load::tokens(1048576), "1,048,576");
    assert_eq!(crate::load::shorten_middle("short", 20), "short");
    let s = crate::load::shorten_middle("Load HuggingFaceTB/SmolLM2-135M", 16);
    assert_eq!(s.chars().count(), 16);
    assert!(s.contains('…') && s.starts_with("Load") && s.ends_with("135M"));
}

#[test]
fn concurrency_is_said_in_plain_words() {
    assert_eq!(crate::load::concurrency_text(32.4, 8192), "serves 32 requests at 8,192 tokens each");
    assert_eq!(crate::load::concurrency_text(1.0, 8192), "serves 1 request at 8,192 tokens each");
    assert_eq!(crate::load::concurrency_text(0.4, 8192), "cannot hold one request of 8,192 tokens");
}

#[test]
fn the_kv_type_costs_sit_inside_the_segments_with_no_repeated_list() {
    let mut h = open_dialog("page-kv-segments");
    h.run();
    assert!(h.query_all_by_label_contains("MB").count() >= 1);
    assert_eq!(h.query_all_by_label_contains("at 8192 tokens").count(), 0);
    h.get_by_label_contains("the size of fp16");
}

#[test]
fn the_result_card_leads_with_four_tiles_and_the_fit_line() {
    let mut h = open_dialog("page-tiles");
    h.run();
    h.get_by_label_contains("WHAT YOU WILL GET");
    let tile = h.get_all_by_label_contains("weights").next().unwrap().accesskit_node().bounding_box().unwrap();
    let fit = h.get_by_label_contains("fits in free memory").accesskit_node().bounding_box().unwrap();
    let weights = h.get_by_label_contains("weights").accesskit_node().bounding_box().unwrap();
    assert!(fit.y0 < weights.y0, "{tile:?} {fit:?} {weights:?}");
}

#[test]
fn the_common_rows_share_one_control_column_at_desktop_width() {
    let mut h = open_dialog("page-grid");
    h.run();
    let mut xs = Vec::new();
    for label in ["KV CACHE SIZE", "PARALLEL REQUESTS", "GPU OFFLOAD"] {
        let n = h.get_all_by_label_contains(label).last().unwrap().accesskit_node().bounding_box().unwrap();
        xs.push((label, n.x0));
    }
    assert!(xs.windows(2).all(|w| (w[0].1 - w[1].1).abs() < 1.0), "{xs:?}");
}

#[test]
fn the_phone_page_collapses_the_result_card_to_tiles_and_the_fit_line() {
    let mut h = harness_at("page-phone-compact", Tab::Models, egui::vec2(390.0, 844.0));
    with_gpu(&mut h, 0, 96 << 30);
    let m = real_model("Qwen3-0.6B");
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    assert!(h.query_by_label_contains("weights").is_none());
    h.get_by_label("details").click();
    h.run();
    h.get_by_label_contains("weights");
    assert!(h.state().load.as_ref().unwrap().details);
}

#[test]
fn loading_through_the_page_then_chatting_answers() {
    use agent_ui::chat::Turn;
    let mut h = open_dialog("page-chat");
    h.run();
    click_load(&mut h);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    let (desk, _asks) = agent_core::Desk::new(|| {});
    h.state_mut().chat.agent.chat.ask("say hello", desk);
    pump_until(&mut h, 20, "agent turn", |a| !a.chat.agent.chat.busy());
    let turns = &h.state().chat.agent.chat.turns;
    assert!(turns.iter().any(|t| matches!(t, Turn::Said(s) if s.contains("Hello"))), "{} turns", turns.len());
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_load_page_at_desktop_and_exactly_390_wide() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (w, hgt, tag) in [(1280.0, 800.0, "1280x800"), (390.0, 844.0, "390x844")] {
        let mut h = harness_at(&format!("page-shot-{tag}"), Tab::Models, egui::vec2(w, hgt));
        with_gpu(&mut h, 20 << 30, 96 << 30);
        let m = real_model("Qwen3-0.6B");
        h.state_mut().library[0] = m.clone();
        h.state_mut().open_load_dialog(&m);
        h.run();
        h.render().unwrap().save(out.join(format!("load-page-harness-{tag}.png"))).unwrap();
    }
}

#[test]
fn the_load_page_opens_in_place_of_the_models_page_and_back_returns() {
    let mut h = open_dialog("page-back");
    h.run();
    h.get_by_label_contains("WHAT YOU WILL GET");
    assert!(h.query_by_label("search your models").is_none());
    h.get_by_label("back").click();
    h.run();
    assert!(h.state().load.is_none());
    assert_eq!(h.state().tab, Tab::Models);
}

#[test]
fn escape_goes_back_and_enter_loads() {
    let mut h = open_dialog("page-keys");
    h.run();
    h.key_press(egui::Key::Escape);
    h.run();
    assert!(h.state().load.is_none());
    let mut h = open_dialog("page-enter");
    h.run();
    h.key_press(egui::Key::Enter);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    assert!(h.state().load.is_none());
}

#[test]
fn the_common_options_are_visible_without_any_toggle() {
    let mut h = open_dialog("page-common");
    h.run();
    for label in ["CONTEXT LENGTH", "KV CACHE TYPE", "KV CACHE SIZE", "PARALLEL REQUESTS", "GPU OFFLOAD", "GPU SELECTION"] {
        assert!(h.query_all_by_label_contains(label).count() >= 1, "{label}");
    }
    assert!(h.query_by_label_contains("FAST LOADER").is_none());
    for heading in specrt::options::SECTION_ORDER {
        assert!(h.query_all_by_label_contains(&heading.to_uppercase()).count() >= 1, "{heading}");
    }
}

#[test]
fn choosing_fp8_kv_changes_the_result_card() {
    let mut h = open_dialog("page-kv");
    h.run();
    let before = h.state().load.as_ref().unwrap().kv_bytes().unwrap();
    crate::load::set_kv_kind(&mut h.state_mut().load.as_mut().unwrap().choice, crate::load::KvKind::Fp8);
    h.run();
    let d = h.state().load.as_ref().unwrap();
    assert_eq!(crate::load::kv_kind_of(&d.choice), crate::load::KvKind::Fp8);
    assert!(d.kv_bytes().unwrap() < before || d.kv_gb < 1e9);
    assert!(d.choice.is_custom());
    h.get_by_label_contains("changed from the profile");
}

#[test]
fn searching_filters_every_option_by_name() {
    let mut h = open_dialog("page-search");
    h.run();
    h.state_mut().load.as_mut().unwrap().search = "readers".into();
    h.run();
    assert!(h.state().load.as_ref().unwrap().search == "readers");
    assert!(h.query_all_by_label_contains("READERS").count() >= 1);
    assert!(h.query_by_label_contains("CONTEXT LENGTH").is_none());
}

#[test]
fn the_load_page_fits_phone_width_with_every_section_open() {
    for (w, hgt) in [(390.0, 844.0), (412.0, 900.0)] {
        let mut h = harness_at("page-phone", Tab::Models, egui::vec2(w, hgt));
        with_gpu(&mut h, 0, 96 << 30);
        let m = real_model("Qwen3-0.6B");
        h.state_mut().library[0] = m.clone();
        h.state_mut().open_load_dialog(&m);
        h.state_mut().load.as_mut().unwrap().advanced = true;
        h.run();
        let screen = h.ctx.content_rect();
        for n in h.query_all_by(|_| true).filter(|n| n.accesskit_node().bounding_box().is_some()) {
            let r = n.rect();
            assert!(r.right() <= screen.right() + 0.5 && r.left() >= screen.left() - 0.5, "{:?} {:?} at {r:?} past {screen:?} {:?}", n.accesskit_node().role(), n.accesskit_node().label(), n.accesskit_node().value());
        }
        let b = h.get_all_by_label("load model").last().unwrap().accesskit_node().bounding_box().unwrap();
        assert!(b.y1 <= hgt as f64 && b.x1 <= w as f64, "{b:?}");
    }
}

#[test]
fn the_advanced_section_renders_every_option_the_engine_declares() {
    let mut h = open_dialog("adv-all");
    h.run();
    assert!(h.query_by_label_contains("FAST LOADER").is_none(), "advanced starts folded");
    h.state_mut().load.as_mut().unwrap().advanced = true;
    h.run();
    for f in specrt::options::schema().iter().filter(|f| f.group.advanced()) {
        let label = f.label.to_uppercase();
        assert!(h.query_all_by_label_contains(&label).count() >= 1, "{} ({}) is not in the dialog", f.key, f.label);
    }
}

#[test]
fn the_dialog_defaults_are_the_engine_defaults_and_max_performance() {
    let h = open_dialog("adv-defaults");
    let d = h.state().load.as_ref().unwrap();
    assert!(d.seqs >= specrt::options::DEFAULT_MAX_NUM_SEQS, "{}", d.seqs);
    assert!(d.solved.as_ref().is_some_and(|s| s.is_ok()));
    assert!(!d.choice.is_custom());
    for f in specrt::options::schema() {
        if f.key == "runtime.prefix_cache.enabled" {
            continue;
        }
        assert_eq!(d.choice.value(&f.key), f.default, "{}", f.key);
    }
    let o = d.serve_options().unwrap();
    let rt = o.runtime.unwrap();
    assert!(rt.graphs && rt.fuse && rt.loader.enabled && rt.prefix_cache.enabled);
    assert_eq!(rt.loader, specrt::loader::LoaderOptions::default());
}

#[test]
fn the_load_dialog_shows_the_fp8_kv_split_for_the_context_and_warns_on_a_short_override() {
    let mut h = open_dialog("fp8-warn");
    {
        let d = h.state_mut().load.as_mut().unwrap();
        d.choice.set("runtime.kv_dtype", serde_json::json!("fp8"));
        d.ctx = d.max;
        d.sync();
    }
    h.run();
    let d = h.state().load.as_ref().unwrap();
    let all = d.view.as_ref().unwrap().spec.attention_layers();
    let long = format!("{} of {all} attention layers keep f16 KV", (all * 3).div_ceil(4));
    assert!(d.ctx > specrt::spec::FP8_KV_VERIFIED_CONTEXT, "{}", d.ctx);
    assert!(h.query_by_label_contains(&long).is_some(), "{long}");
    assert!(h.query_by_label_contains("past 32768 tokens").is_some());
    {
        let d = h.state_mut().load.as_mut().unwrap();
        d.choice.set("runtime.kv_f16_layers", serde_json::json!(1));
        d.sync();
    }
    h.run();
    assert!(h.query_by_label_contains("drift from f16 and can derail").is_some());
    {
        let d = h.state_mut().load.as_mut().unwrap();
        d.choice.reset("runtime.kv_f16_layers");
        d.ctx = 8192;
        d.sync();
    }
    h.run();
    assert!(h.query_by_label_contains(&format!("{} of {all} attention layers keep f16 KV", all / 2)).is_some());
    h.state_mut().load.as_mut().unwrap().choice.set("runtime.kv_dtype", serde_json::json!("f16"));
    h.run();
    assert!(h.query_by_label_contains("attention layers keep f16 KV").is_none());
}

#[test]
fn changing_a_field_shows_custom_and_its_reset_key_restores_the_default() {
    let mut h = open_dialog("adv-custom");
    h.state_mut().load.as_mut().unwrap().advanced = true;
    h.run();
    assert!(h.query_by_label_contains("changed from the profile").is_none());
    assert!(h.query_by_label("reset").is_none());
    let d = h.state_mut().load.as_mut().unwrap();
    d.choice.set("serve.max_batched_tokens", serde_json::json!(1024));
    h.run();
    h.get_by_label_contains("1 advanced option(s) changed from the profile");
    h.get_by_label("reset");
    h.state_mut().load.as_mut().unwrap().reset("serve.max_batched_tokens");
    h.run();
    let d = h.state().load.as_ref().unwrap();
    assert!(!d.choice.is_custom());
    assert!(h.query_by_label_contains("changed from the profile").is_none());
}

#[test]
fn the_low_memory_profile_turns_off_what_it_says_and_reaches_the_engine() {
    let mut h = open_dialog("adv-low");
    h.state_mut().load.as_mut().unwrap().choice.select(specrt::options::Profile::LowMemory);
    let o = h.state().load.as_ref().unwrap().serve_options().unwrap();
    let rt = o.runtime.unwrap();
    assert_eq!(o.speculative.enabled, Some(false));
    assert!(!rt.prefix_cache.enabled);
    assert_eq!(rt.prefix_cache.max_snapshots, 0);
    assert!(rt.loader.host_budget_mb < specrt::loader::LoaderOptions::default().host_budget_mb);
    h.run();
    h.get_by_label_contains("fp8 KV cache, 2 parallel requests");
}

#[test]
fn engine_options_from_the_dialog_are_remembered_and_shown_on_the_server_tab() {
    let mut h = open_dialog("adv-remember");
    h.state_mut().load.as_mut().unwrap().choice.set("runtime.loader.readers", serde_json::json!(3));
    h.run();
    click_load(&mut h);
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    let l = h.state().loaded().unwrap();
    assert!(l.engine.is_custom());
    assert_eq!(l.engine.value("runtime.loader.readers"), serde_json::json!(3));
    assert_eq!(Settings::load(&h.state().dir).engine_for("Qwen3-0.6B").value("runtime.loader.readers"), serde_json::json!(3));
    h.state_mut().tab = Tab::Server;
    h.run();
    let rows = crate::running::rows(&l);
    assert!(rows.contains(&("profile".into(), "custom".into())));
    assert!(rows.contains(&("readers".into(), "3".into())));
    assert!(rows.iter().any(|(k, _)| k == "mtp"));
}

#[test]
fn screenshot_of_the_models_page_header() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness_sized("shot-models-header", Tab::Models, 520.0);
    h.run();
    h.render().unwrap().save(out.join("models-header.png")).unwrap();
}

#[test]
fn screenshots_of_the_load_dialog_with_advanced_folded_and_open() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (open, file) in [(false, "load-dialog-profile-folded.png"), (true, "load-dialog-profile-advanced.png")] {
        let mut h = harness_sized(&format!("shot-adv-{open}"), Tab::Models, 760.0);
        with_gpu(&mut h, 20 << 30, 96 << 30);
        let d = dir(&format!("shot-adv-model-{open}"));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), d.join("config.json")).unwrap();
        let mut m = model("Qwen3.8-27B", true, "dense decoder");
        (m.path, m.root) = (d.clone(), d);
        m.context_length = None;
        h.state_mut().library[0] = m.clone();
        h.state_mut().open_load_dialog(&m);
        let dialog = h.state_mut().load.as_mut().unwrap();
        dialog.ctx = 65536;
        dialog.advanced = open;
        h.run();
        with_gpu(&mut h, 20 << 30, 96 << 30);
        h.run();
        h.render().unwrap().save(out.join(file)).unwrap();
    }
}

#[test]
fn the_load_dialog_with_a_million_token_context_fits_desktop_and_phone_widths() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (w, h_px, tag) in [(1280.0, 800.0, "desktop"), (390.0, 844.0, "phone")] {
        let d = dir(&format!("yarn-shot-{tag}"));
        let mut h = Harness::builder().with_size(egui::vec2(w, h_px)).with_max_steps(100).with_accessibility_check(false).allow_missing_glyphs().build_eframe({
            let d = d.clone();
            move |cc| {
                egui_bench::install(&cc.egui_ctx);
                let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
                app.tab = Tab::Models;
                app.logs = shared_logs();
                app.library = vec![model("small-dense", true, "dense decoder")];
                app
            }
        });
        with_gpu(&mut h, 20 << 30, 96 << 30);
        let md = dir(&format!("yarn-shot-model-{tag}"));
        std::fs::create_dir_all(&md).unwrap();
        std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), md.join("config.json")).unwrap();
        let mut m = model("big-dense", true, "dense decoder");
        (m.path, m.root) = (md.clone(), md);
        m.context_length = None;
        h.state_mut().library[0] = m.clone();
        h.state_mut().open_load_dialog(&m);
        h.run();
        let dialog = h.state_mut().load.as_mut().unwrap();
        dialog.ctx = dialog.max;
        h.run();
        let dialog = h.state().load.as_ref().unwrap();
        assert_eq!(dialog.ctx, 262144 * 16);
        assert_eq!(dialog.extension_factor(), 16);
        assert_eq!(dialog.choice.value("runtime.rope_extension.type").as_str(), Some("yarn"));
        h.render().unwrap().save(out.join(format!("load-dialog-yarn-{tag}.png"))).unwrap();
        let dialog = h.state_mut().load.as_mut().unwrap();
        dialog.ctx = 262144;
        h.run();
        let dialog = h.state().load.as_ref().unwrap();
        assert_eq!(dialog.extension_factor(), 1);
        assert_eq!(dialog.choice.value("runtime.rope_extension.type").as_str(), None);
        h.render().unwrap().save(out.join(format!("load-dialog-native-{tag}.png"))).unwrap();
    }
}

#[test]
fn a_context_extension_lifts_the_slider_to_a_million_tokens_and_prices_the_kv_cache() {
    let mut h = harness("extension", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let d = dir("ext-27b");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3.8-27b.config.json"), d.join("config.json")).unwrap();
    let mut m = model("Qwen3.8-27B", true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    let dialog = h.state_mut().load.as_mut().unwrap();
    assert_eq!(dialog.max, 262144 * 16);
    assert_eq!(dialog.max, 262144 * 16);
    dialog.ctx = 1_048_576;
    dialog.sync();
    assert_eq!(dialog.choice.value("runtime.rope_extension.type").as_str(), Some("yarn"));
    assert_eq!(dialog.extension_factor(), 4);
    dialog.ctx = 262144 * 2 + 1;
    dialog.sync();
    assert_eq!(dialog.extension_factor(), 3);
    dialog.ctx = 262144;
    dialog.sync();
    assert_eq!(dialog.choice.value("runtime.rope_extension.type").as_str(), None);
    assert_eq!(dialog.extension_factor(), 1);
    dialog.choice.set("runtime.rope_extension.type", serde_json::json!("yarn"));
    dialog.sync();
    assert_eq!(dialog.extension_factor(), 4);
    dialog.choice.reset("runtime.rope_extension.type");
    dialog.choice.reset("runtime.rope_extension.factor");
    dialog.choice.set("runtime.rope_extension.factor", serde_json::json!(4));
    dialog.sync();
    assert_eq!(dialog.max, 262144 * 16);
    dialog.ctx = 1_048_576;
    dialog.seqs = 1;
    let kv = dialog.estimate(Some(96 << 30)).unwrap().kv;
    let gib = kv as f64 / (1u64 << 30) as f64;
    assert!((60.0..=70.0).contains(&gib), "bf16 kv at 1M is {gib} GiB");
    dialog.choice.set("runtime.kv_dtype", serde_json::json!("fp8"));
    dialog.sync();
    let fp8 = dialog.estimate(Some(96 << 30)).unwrap().kv as f64 / (1u64 << 30) as f64;
    assert!(fp8 < gib * 0.9 && fp8 > gib * 0.8, "mixed fp8 kv {fp8} GiB vs {gib}");
    let fits_big = dialog.estimate(Some(96 << 30)).is_none_or(|e| e.fits() != Some(false));
    let fits_small = dialog.estimate(Some(40 << 30)).is_none_or(|e| e.fits() != Some(false));
    assert!(fits_big && !fits_small);
    assert_eq!(dialog.serve_options().unwrap().runtime.unwrap().rope_extension.unwrap().factor, 4);
    dialog.choice.reset("runtime.rope_extension.factor");
    dialog.sync();
    assert_eq!(dialog.max, 262144 * 16);
    assert_eq!(dialog.ctx, 262144 * 4);
}

fn fixture_spec_model() -> LocalModel {
    let mut m = model("Qwen27", true, "dense decoder");
    m.format = Format::Spec;
    m.path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/qwen3.8-27b.toml");
    m
}

fn fake_peer(host: &str, gib: f64, busy: bool) -> specrt::cluster::Peer {
    let machine = specrt::hardware::fake_gpus(host, 1, true);
    let mut d = machine.devices(&[]).remove(0);
    d.memory_gib = gib;
    d.address = Some(format!("10.0.0.7:{}", 7600));
    specrt::cluster::Peer {
        version: specrt::cluster::ANNOUNCEMENT_VERSION,
        host: host.into(),
        app_version: Some("0.1".into()),
        busy,
        serving: busy.then(|| "Qwen3-0.6B".to_string()),
        control: Some("10.0.0.7:7599".into()),
        gpus: vec![specrt::cluster::PeerGpu { device: d, gpu_name: "NVIDIA GB10".into(), backend: specrt::hardware::GpuBackend::Cuda }],
    }
}

fn hardware_harness(name: &str, gpus: usize) -> Harness<'static, App> {
    let mut h = harness(name, Tab::Hardware);
    {
        let app = h.state_mut();
        app.hardware.auto = false;
        let mut m = specrt::hardware::fake_gpus("desk", gpus, true);
        for g in &mut m.gpus {
            g.free_bytes = 12 << 30;
        }
        m.interfaces = vec![specrt::hardware::NetInterface { name: "eth0".into(), ip: "192.168.1.5".parse().unwrap() }];
        app.hardware.machine = Some(m);
        app.library = vec![fixture_spec_model()];
    }
    h
}

#[test]
fn the_hardware_tab_lists_every_gpu_with_memory_bandwidth_and_the_links_between_them() {
    for n in [2usize, 4] {
        let mut h = hardware_harness(&format!("hw{n}"), n);
        h.run();
        for i in 0..n {
            assert!(h.get_all_by_label_contains(&format!("GPU {i}")).count() >= 1);
        }
        assert_eq!(h.get_all_by_label_contains("RTX PRO 6000 Blackwell Max-Q").count(), n);
        assert_eq!(h.get_all_by_label_contains("sm_120").count(), n);
        assert_eq!(h.get_all_by_label_contains("12.0 GB free").count(), n);
        assert_eq!(h.get_all_by_label_contains("1792 GB/s").count(), n);
        h.get_by_label_contains("GPU 0 and GPU 1: peer to peer");
        h.get_by_label_contains("eth0 192.168.1.5");
    }
}

#[test]
fn the_hardware_tab_marks_unlinked_gpus_as_host_staged() {
    let mut h = hardware_harness("hwstaged", 2);
    h.state_mut().hardware.machine = Some(specrt::hardware::fake_gpus("desk", 2, false));
    h.run();
    h.get_by_label_contains("GPU 0 and GPU 1: staged through host memory");
}

#[test]
fn peers_from_fake_announcements_show_name_gpus_memory_and_busy_state() {
    let mut h = hardware_harness("hwpeers", 1);
    {
        let app = h.state_mut();
        app.hardware.peers.insert("spark".into(), fake_peer("spark", 110.0, true));
        app.hardware.peers.insert("nuc".into(), fake_peer("nuc", 16.0, false));
        app.hardware.peers.insert("desk".into(), fake_peer("desk", 96.0, false));
    }
    h.run();
    h.get_by_label_contains("spark: GB10  118 GB");
    h.get_by_label_contains("busy");
    h.get_by_label_contains("serving Qwen3-0.6B");
    assert!(h.get_all_by_label_contains("10.0.0.7:7599").count() >= 1);
    h.get_by_label_contains("nuc: GB10  17 GB");
    assert!(h.get_all_by_label_contains("idle").count() >= 1);
    assert!(h.query_by_label_contains("desk: GB10").is_none());
}

#[test]
fn switching_a_gpu_off_in_the_hardware_tab_removes_it_from_every_budget() {
    let mut h = hardware_harness("hwswitch", 2);
    let gpu = |total: u64| inference_bench_core::gpu::GpuStats { name: "card".into(), used_bytes: 0, total_bytes: total << 30, util_percent: None, uuid: String::new(), own_bytes: 0, display: false, index: None };
    {
        let app = h.state_mut();
        app.gpu = vec![gpu(24), gpu(24)];
        app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
        app.gpu_poll = Default::default();
    }
    h.run();
    assert_eq!(h.state().free_gpu_list().len(), 2);
    assert_eq!(h.state().free_gpu_bytes(), Some(48 << 30));
    h.get_all_by_label("in use").next().unwrap().click();
    h.run();
    assert_eq!(h.state().free_gpu_list().len(), 1);
    assert_eq!(h.state().free_gpu_bytes(), Some(24 << 30));
    assert_eq!(Settings::load(&h.state().dir).cluster.disabled_gpus.len(), 1);
    h.get_by_label("switched off");
}

#[test]
fn the_hardware_tab_has_no_second_way_to_load_a_model() {
    let mut h = hardware_harness("hwnoload", 2);
    h.run();
    for gone in ["load across the cluster", "run a cluster", "stop cluster"] {
        assert!(h.query_by_label_contains(gone).is_none(), "{gone}");
    }
}

#[test]
fn an_unpaired_peer_cannot_be_selected_and_pairing_with_a_wrong_code_fails() {
    let mut h = hardware_harness("hwpair", 1);
    h.state_mut().hardware.peers.insert("spark".into(), fake_peer("spark", 14.0, false));
    h.run();
    h.get_by_label_contains("spark: GB10");
    assert!(h.query_by_label("not in use").is_none());
    assert!(h.state().settings.cluster.peers_in_use.is_empty());
}

#[test]
fn sharing_is_off_until_the_switch_is_turned_on() {
    let mut h = hardware_harness("hwshare", 1);
    h.run();
    assert!(!h.state().settings.cluster.share);
    assert!(h.state().hardware.host.is_none());
}

#[test]
fn the_pipeline_section_names_each_stage_and_shows_a_lost_stage() {
    use specrt::sched::{ChainMetrics, StageMetric};
    let chain = ChainMetrics {
        stages: vec![
            StageMetric { name: "desk/gpu0".into(), device: 0, start: 0, end: 32, busy_secs: 1.5, calls: 10, tokens: 40 },
            StageMetric { name: "spark/gpu0".into(), device: 1, start: 32, end: 64, busy_secs: 1.0, calls: 10, tokens: 40 },
        ],
        hops: 10,
        hop_secs: 0.02,
        hop_bytes: 4 << 20,
        lost: Some("stage spark/gpu0 was lost: connection reset".into()),
        ..ChainMetrics::default()
    };
    let rows = crate::hardware::pipeline_rows(&specrt::serve::Stats { chain: Some(chain), ..Default::default() });
    assert!(rows[0].starts_with("stage 0 desk / GPU 0  layers 0..32"));
    assert!(rows[1].contains("spark / GPU 0"));
    assert!(rows[2].contains("2.00 ms each"));
    assert!(rows[3].starts_with("lost: stage spark/gpu0 was lost"));
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_the_hardware_tab() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let d = dir("hw-shot");
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 1500.0)).with_accessibility_check(false).allow_missing_glyphs().wgpu().build_eframe(move |cc| {
        egui_bench::install(&cc.egui_ctx);
        let mut app = App::bare(d, wake_for(&cc.egui_ctx), &cc.egui_ctx);
        app.tab = Tab::Hardware;
        app.logs = shared_logs();
        let mut m = specrt::hardware::fake_gpus("desk", 2, true);
        for g in &mut m.gpus {
            g.free_bytes = 12 << 30;
        }
        m.interfaces = vec![specrt::hardware::NetInterface { name: "enp5s0".into(), ip: "192.168.1.5".parse().unwrap() }];
        app.hardware.machine = Some(m);
        app.hardware.peers.insert("spark".into(), fake_peer("spark", 14.0, false));
        app.hardware.peers.insert("nuc".into(), fake_peer("nuc", 16.0, true));
        app.settings.cluster.paired.insert("spark".into());
        app.settings.cluster.peers_in_use.insert("spark".into());
        app.library = vec![fixture_spec_model()];
        app
    });
    h.run();
    h.run();
    h.render().unwrap().save(out.join("hardware.png")).unwrap();
}

fn gguf_header(arch: &str, context: u32) -> Vec<u8> {
    use specrt::gguf::writer::{Kv as W, build};
    let key = |k: &str| format!("{arch}.{k}");
    let (a, b, c, d, e, f, g) = (key("block_count"), key("embedding_length"), key("feed_forward_length"), key("attention.head_count"), key("attention.head_count_kv"), key("attention.key_length"), key("context_length"));
    let eps = key("attention.layer_norm_rms_epsilon");
    build(
        3,
        &[
            ("general.architecture", W::Str(arch)),
            (&a, W::U32(36)),
            (&b, W::U32(4096)),
            (&c, W::U32(12288)),
            (&d, W::U32(32)),
            (&e, W::U32(8)),
            (&f, W::U32(128)),
            (&eps, W::F32(1e-6)),
            (&g, W::U32(context)),
            ("tokenizer.ggml.tokens", W::Strs(&["a", "b"])),
        ],
        &[],
    )
}

fn discover_hub() -> String {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let gb = |n: u64| n << 30;
    let tree = |files: &[(&str, u64)]| format!("[{}]", files.iter().map(|(p, s)| format!(r#"{{"type":"file","path":"{p}","size":{s}}}"#)).collect::<Vec<_>>().join(","));
    let big = tree(&[("Big-Q4_K_M.gguf", gb(5)), ("Big-Q5_K_M.gguf", gb(6)), ("Big-Q8_0.gguf", gb(9)), ("mmproj-Big-BF16.gguf", gb(1)), ("config.json", 100)]);
    let odd = tree(&[("Odd-Q4_K_M-00001-of-00002.gguf", gb(3)), ("Odd-Q4_K_M-00002-of-00002.gguf", gb(3))]);
    let huge = tree(&[("Huge-Q4_K_M.gguf", gb(200))]);
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let (big, odd, huge) = (big.clone(), odd.clone(), huge.clone());
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
                let (head, body): (&str, Vec<u8>) = if path.starts_with("/api/models?") {
                    ("200 OK", br#"[{"id":"acme/Huge-GGUF","downloads":900,"likes":1},{"id":"acme/Odd-GGUF","downloads":500,"likes":1},{"id":"acme/Big-GGUF","downloads":100,"likes":1}]"#.to_vec())
                } else if path.starts_with("/api/models/acme/Big-GGUF/tree") {
                    ("200 OK", big.into_bytes())
                } else if path.starts_with("/api/models/acme/Odd-GGUF/tree") {
                    ("200 OK", odd.into_bytes())
                } else if path.starts_with("/api/models/acme/Huge-GGUF/tree") {
                    ("200 OK", huge.into_bytes())
                } else if path.starts_with("/acme/Odd-GGUF/resolve") {
                    ("206 Partial Content", gguf_header("brandnew", 8192))
                } else if path.contains("/resolve/") {
                    ("206 Partial Content", gguf_header("qwen3", 131072))
                } else {
                    ("404 Not Found", Vec::new())
                };
                let _ = write!(s, "HTTP/1.1 {head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            });
        }
    });
    base
}

fn discover_harness(name: &str) -> Harness<'static, App> {
    discover_harness_on(name, 24 << 30)
}

fn discover_harness_on(name: &str, total: u64) -> Harness<'static, App> {
    let mut h = harness(name, Tab::Models);
    with_gpu(&mut h, 0, total);
    {
        let app = h.state_mut();
        app.models.hub_base = discover_hub();
        app.models.pane = models::Pane::Discover;
        app.models.query = "1017".into();
        app.settings.default_seqs = 1;
    }
    h.run();
    h.get_by_label("search").click();
    pump_until(&mut h, 20, "inspected", |a| a.models.infos.len() == 3);
    h.run();
    h
}

#[test]
fn discover_shows_one_row_per_repo_with_its_best_quant_and_no_file_rows() {
    let h = discover_harness("discover-rows");
    let plans = h.state().models.plans(&h.state().free_gpu_list(), 1);
    assert_eq!(plans.iter().map(|p| p.repo.as_str()).collect::<Vec<_>>(), ["acme/Big-GGUF", "acme/Huge-GGUF", "acme/Odd-GGUF"]);
    let line = models::headline(&plans[0]).unwrap();
    assert!(line.starts_with("Best fit: Q8_0, full 128k context, "), "{line}");
    h.get_by_label_contains("Best fit: Q8_0, full 128k context");
    assert!(h.query_by_label_contains("Big-Q4_K_M.gguf").is_none());
    assert!(h.query_by_label_contains("mmproj").is_none());
    assert!(h.query_by_label_contains("00001-of-00002").is_none());
    let buttons: Vec<bool> = h.query_all_by_label("download").map(|n| n.accesskit_node().is_disabled()).collect();
    assert_eq!(buttons.len(), 2, "the unsupported repo has no download button");
    assert_eq!(buttons.iter().filter(|d| !**d).count(), 1, "the repo that does not fit has a disabled download");
    h.get_by_label_contains("supported");
    assert!(h.query_by_label_contains("RUNS HERE").is_none() && h.query_by_label_contains("runs here").is_none());
}

#[test]
fn discover_says_when_nothing_fits_and_flags_unsupported_architectures() {
    let h = discover_harness("discover-nofit");
    let plans = h.state().models.plans(&h.state().free_gpu_list(), 1);
    assert_eq!(models::headline(&plans[1]).map(|l| l.starts_with("Does not fit: needs")), Some(true));
    h.get_by_label_contains("Does not fit: needs");
    assert!(h.query_by_label_contains("context 0").is_none() && h.query_by_label_contains("0 of").is_none());
    h.get_by_label_contains("not supported");
    assert!(plans[2].unsupported.as_deref().is_some_and(|w| w.contains("brandnew")), "{:?}", plans[2].unsupported);
}

#[test]
fn discover_other_quants_are_a_collapsed_picker_not_rows() {
    let mut h = discover_harness("discover-picker");
    h.state_mut().models.picks.insert("acme/Big-GGUF".into(), "Q4_K_M".into());
    h.run();
    assert_eq!(h.query_all_by_label_contains("Best fit: Q8_0").count(), 0, "a picked quant replaces the best-fit line");
    assert_eq!(h.query_all_by_label_contains("Big-Q8_0.gguf").count(), 0);
}

#[test]
fn a_finished_download_preselects_its_fitted_context_and_kv_in_the_load_dialog() {
    use inference_bench_core::discover::Kv;
    let mut h = harness("preselect", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    let m = hub_model("acme/Qwen3-0.6B", None);
    let app = h.state_mut();
    let m = LocalModel { path: dir("preselect-model"), ..m };
    std::fs::create_dir_all(&m.path).unwrap();
    std::fs::copy(fixtures().join("qwen3-0.6b.config.json"), m.path.join("config.json")).unwrap();
    app.models.preselect.push(models::Preselect { model: m.name.clone(), context: 16384, kv: Kv::MixedFp8 });
    app.open_load_dialog(&m);
    let d = app.load.as_ref().unwrap();
    assert_eq!(d.ctx, 16384);
    assert_eq!(d.kv_dtype(), specrt::spec::KvDtype::Fp8);
    assert!(d.choice.value("runtime.kv_f16_layers").is_null(), "mixed fp8 leaves the split to the context-aware default");
    let layers = d.kv_spec().unwrap().kv_f16_layers();
    assert!(layers > 0, "mixed fp8 keeps the early layers in f16");
    let a = d.anatomy().unwrap();
    let fp8_only = Kv::Fp8.bytes_per_token(&d.view.as_ref().unwrap().spec);
    assert!(a.kv_bytes_per_token > fp8_only, "{} {}", a.kv_bytes_per_token, fp8_only);
    assert!(app.models.preselect.is_empty());
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_discover() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut fixture = discover_harness("discover-shot");
    fixture.run();
    fixture.render().unwrap().save(out.join("discover-fixture.png")).unwrap();
    let mut gpus = inference_bench_core::gpu::sample();
    let apps = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=gpu_uuid,pid,used_memory,process_name", "--format=csv,noheader,nounits"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let running_app: Vec<u32> = apps.lines().filter(|l| l.contains("inference-bench-ui")).filter_map(|l| l.split(',').nth(1)?.trim().parse().ok()).collect();
    let parsed = inference_bench_core::gpu::parse_compute_apps(&apps.lines().map(|l| l.rsplitn(2, ',').last().unwrap_or("").to_string()).collect::<Vec<_>>().join("\n"));
    inference_bench_core::gpu::attribute(&mut gpus, &parsed, &running_app);
    assert!(!gpus.is_empty(), "no gpu reading on this machine");
    for (file, query) in [("discover-1017.png", "1017"), ("discover-qwen3.8-27b.png", "qwen3.8 27b"), ("discover-qwen3.8-27b-fp8.png", "Qwen3.8-27B-FP8"), ("discover-qwen3-8b-gguf.png", "Qwen3-8B-GGUF")] {
        let mut h = harness_sized("discover-live", Tab::Models, 1400.0);
        {
            let app = h.state_mut();
            app.gpu = gpus.clone();
            app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
            app.gpu_poll = Default::default();
            app.models.pane = models::Pane::Discover;
            app.models.query = query.into();
        }
        h.run();
        h.get_by_label("search").click();
        pump_until(&mut h, 180, "live inspect", |a| !a.models.results.is_empty() && a.models.infos.len() == a.models.results.len());
        h.run();
        h.render().unwrap().save(out.join(file)).unwrap();
    }
}

#[test]
fn the_budget_excludes_this_apps_own_memory_and_a_model_that_fits_then_shows_as_fitting() {
    use inference_bench_core::gpu::{GpuStats, attribute, parse_compute_apps};
    let mut gpu = vec![GpuStats { name: "RTX".into(), used_bytes: 87 << 30, total_bytes: 96 << 30, util_percent: None, uuid: "GPU-a".into(), own_bytes: 0, display: false, index: None }];
    attribute(&mut gpu, &parse_compute_apps("GPU-a, 4242, 61440\nGPU-a, 7, 12288\n"), &[4242]);
    let mut h = harness("budget", Tab::Models);
    h.state_mut().gpu = gpu;
    h.state_mut().gpu_at = Some(Instant::now() + Duration::from_secs(600));
    h.state_mut().gpu_poll = Default::default();
    assert_eq!(h.state().free_gpu_list(), [(96u64 << 30) - (27u64 << 30)]);
    assert_eq!(models::budget_line(&h.state().gpu, 8), "69.0 GB available for a model (27.0 GB used by other processes), 8 parallel requests");
    let g = specrt::gguf::Gguf::from_header(gguf_header("qwen3", 131072)).unwrap();
    let spec = g.spec().unwrap();
    let anatomy = specrt::anatomy::ModelAnatomy::new(&spec, 0, Some(131072));
    let raw_free = (96u64 << 30) - (87u64 << 30);
    let weights = 30u64 << 30;
    assert_eq!(inference_bench_core::discover::assess(&spec, &anatomy, weights, &[raw_free], 1).verdict, inference_bench_core::discover::Verdict::DoesNotFit);
    let budget = h.state().free_gpu_list();
    let a = inference_bench_core::discover::assess(&spec, &anatomy, weights, &budget, 1);
    assert_eq!((a.verdict, a.full()), (inference_bench_core::discover::Verdict::Fits, true));
}

fn request(ttft: f64, prompt: u64, cached: u64, completion: u64, prefill: f64, decode: f64) -> inference_bench_core::live::RequestRecord {
    inference_bench_core::live::RequestRecord {
        at: chrono::Utc::now(),
        ttft_secs: ttft,
        queue_secs: 0.0,
        prefill_secs: 0.0,
        prompt_tokens: prompt,
        cached_tokens: cached,
        completion_tokens: completion,
        prefill_tok_s: prefill,
        decode_tok_s: decode,
    }
}

#[test]
fn the_per_request_table_shows_clean_units_and_cache_columns() {
    let wide = crate::serving::request_columns(false, 1280.0);
    let plain = request(0.842, 625, 0, 1574, 742.3, 35.44);
    assert_eq!(crate::serving::request_cells(&plain, &wide), ["842 ms", "0 ms", "625", "625", "1.57k", "742 tok/s", "35.4 tok/s"]);
    let wide = crate::serving::request_columns(true, 1280.0);
    let warm = request(1.308, 1450, 1000, 40, 2310.4, 35.44);
    assert_eq!(crate::serving::request_cells(&warm, &wide), ["1.31 s", "0 ms", "1.45k", "69%", "1.00k", "450", "40", "2,310 tok/s", "35.4 tok/s"]);
    let full = request(0.19, 512, 507, 8, 0.0, 20.0);
    assert_eq!(crate::serving::request_cells(&full, &wide)[5..], ["5", "8", "-", "20.0 tok/s"]);
    let mut waited = request(3.71, 5000, 0, 8, 5376.0, 20.0);
    waited.queue_secs = 3.5;
    assert_eq!(crate::serving::request_cells(&waited, &wide)[..2], ["3.71 s", "3,500 ms"]);
    let cols = |c: bool, w: f32| crate::serving::request_columns(c, w).iter().map(|c| c.0).collect::<Vec<_>>();
    assert_eq!(cols(false, 800.0), ["time", "ttft", "queue", "prompt", "new", "prefill", "decode"]);
    assert_eq!(cols(true, 1280.0), ["time", "ttft", "queue", "prompt", "cache hit", "cache read", "new", "completion", "prefill", "decode"]);
}

#[test]
fn the_per_request_columns_fit_every_width_down_to_a_phone() {
    for width in [1280.0f32, 1000.0, 900.0, 780.0, 700.0, 500.0, 392.0, 340.0, 280.0] {
        for with_cache in [false, true] {
            let cols = crate::serving::request_columns(with_cache, width);
            let total: f32 = cols.iter().map(|c| c.1).sum();
            assert!(total + 6.0 <= width, "{width} {with_cache} {total} {:?}", cols.iter().map(|c| c.0).collect::<Vec<_>>());
            let names: Vec<_> = cols.iter().map(|c| c.0).collect();
            assert!(names.contains(&"new") && names.contains(&"prefill") && names.contains(&"ttft") && names.contains(&"time"), "{names:?}");
        }
    }
    let names = |w: f32| crate::serving::request_columns(true, w).iter().map(|c| c.0).collect::<Vec<_>>();
    assert!(names(1280.0).contains(&"queue"));
    assert!(!names(340.0).contains(&"queue"));
    assert!(names(392.0).contains(&"decode") && !names(280.0).contains(&"decode"));
}

#[test]
fn the_host_busy_reading_shows_only_above_the_threshold() {
    assert!(crate::serving::host_reading(4.0, 16).is_none());
    let (label, text, _) = crate::serving::host_reading(49.0, 16).unwrap();
    assert_eq!((label, text.as_str()), ("host busy", "load 49.0 on 16 cores"));
}

#[test]
fn the_per_request_table_renders_with_and_without_cache_columns() {
    let recent = vec![request(0.842, 625, 0, 1574, 742.3, 35.4), request(1.308, 1450, 1000, 40, 2310.4, 35.4)];
    let mut h = Harness::new_ui(|ui| {
        egui_bench::install(ui.ctx());
        crate::serving::per_request(ui, &recent);
    });
    h.run();
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_the_per_request_table() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let recent = vec![
        request(0.842, 625, 0, 1574, 742.3, 35.4),
        request(2.254, 14_500, 0, 912, 6431.0, 33.1),
        request(12.43, 145_000, 120_000, 300, 2310.4, 31.9),
        request(0.05, 512, 512, 8, 0.0, 41.2),
        request(65.0, 1_200_000, 1_000_000, 4096, 3100.0, 28.0),
    ];
    let mut h = Harness::builder().with_size(egui::vec2(1000.0, 220.0)).with_accessibility_check(false).allow_missing_glyphs().build_ui(|ui| {
        egui_bench::install(ui.ctx());
        crate::serving::per_request(ui, &recent);
    });
    h.run();
    h.render().unwrap().save(out.join("server-per-request.png")).unwrap();
    let mut phone = Harness::builder().with_size(egui::vec2(392.0, 220.0)).with_accessibility_check(false).allow_missing_glyphs().build_ui(|ui| {
        egui_bench::install(ui.ctx());
        crate::serving::per_request(ui, &recent);
    });
    phone.run();
    phone.render().unwrap().save(out.join("server-per-request-phone.png")).unwrap();
}

fn many_hub(repos: usize, sha: std::sync::Arc<std::sync::Mutex<String>>) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = hits.clone();
    let list = format!("[{}]", (0..repos).map(|i| format!(r#"{{"id":"acme/M{i}-GGUF","downloads":{},"likes":1}}"#, 1000 - i)).collect::<Vec<_>>().join(","));
    let gb = |n: u64| n << 30;
    let tree = format!(
        r#"[{{"type":"file","path":"M-Q4_K_M.gguf","size":{}}},{{"type":"file","path":"M-Q5_K_M.gguf","size":{}}},{{"type":"file","path":"M-Q8_0.gguf","size":{}}}]"#,
        gb(5),
        gb(6),
        gb(9)
    );
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let (list, tree, sha, counter) = (list.clone(), tree.clone(), sha.clone(), counter.clone());
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
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let (head, body): (&str, Vec<u8>) = if path.starts_with("/api/models?") {
                    ("200 OK", list.into_bytes())
                } else if path.contains("/tree/") {
                    ("200 OK", tree.into_bytes())
                } else if path.starts_with("/api/models/") {
                    ("200 OK", format!(r#"{{"sha":"{}"}}"#, sha.lock().unwrap()).into_bytes())
                } else if path.contains("/resolve/") {
                    ("206 Partial Content", gguf_header("qwen3", 131072))
                } else {
                    ("404 Not Found", Vec::new())
                };
                let _ = write!(s, "HTTP/1.1 {head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            });
        }
    });
    (base, hits)
}

fn many_harness(name: &str, repos: usize, sha: &std::sync::Arc<std::sync::Mutex<String>>) -> (Harness<'static, App>, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let mut h = harness_sized(name, Tab::Models, 800.0);
    with_gpu(&mut h, 0, 96 << 30);
    let (hub, hits) = many_hub(repos, sha.clone());
    {
        let app = h.state_mut();
        app.models.hub_base = hub;
        app.models.pane = models::Pane::Discover;
        app.models.query = "m".into();
        app.settings.default_seqs = 1;
    }
    h.run();
    h.get_by_label("search").click();
    pump_until(&mut h, 60, "inspected", |a| a.models.infos.len() == repos);
    h.run();
    (h, hits)
}

fn frame_ms(h: &mut Harness<'static, App>, frames: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..frames {
        h.step();
    }
    start.elapsed().as_secs_f64() * 1000.0 / frames as f64
}

#[test]
fn discover_plans_once_and_frames_stay_cheap_with_100_results() {
    let sha = std::sync::Arc::new(std::sync::Mutex::new("abc".to_string()));
    let (mut h, _) = many_harness("frame-time-100", 100, &sha);
    let runs = h.state().models.plan_runs;
    let ms = frame_ms(&mut h, 30);
    eprintln!("discover frame 100 results: {ms:.2} ms");
    assert_eq!(h.state().models.plan_runs, runs, "frames must not replan");
    assert!(ms < 6.0, "{ms} ms per frame");
    assert!(runs <= 3, "{runs} plan runs while loading");
}

#[test]
fn discover_replans_when_the_gpu_budget_moves_past_a_bucket_only() {
    let sha = std::sync::Arc::new(std::sync::Mutex::new("abc".to_string()));
    let (mut h, _) = many_harness("replan-budget", 5, &sha);
    h.state_mut().gpu[0].used_bytes = 100 << 20;
    h.run();
    let runs = h.state().models.plan_runs;
    h.state_mut().gpu[0].used_bytes += 1 << 20;
    h.run();
    assert_eq!(h.state().models.plan_runs, runs);
    h.state_mut().gpu[0].used_bytes += 40 << 30;
    h.run();
    assert_eq!(h.state().models.plan_runs, runs + 1);
}

#[test]
fn a_second_search_for_the_same_repos_reads_the_disk_cache_not_the_hub() {
    let sha = std::sync::Arc::new(std::sync::Mutex::new("abc".to_string()));
    let (mut h, hits) = many_harness("info-cache", 4, &sha);
    let before = hits.load(std::sync::atomic::Ordering::SeqCst);
    assert!(before > 4, "{before}");
    h.state_mut().models.infos.clear();
    h.get_by_label("search").click();
    pump_until(&mut h, 20, "reinspected", |a| a.models.infos.len() == 4);
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), before + 1, "only the search itself hits the hub");
}

#[test]
fn the_server_tab_lists_saved_sessions_with_restore_strip_and_delete() {
    use inference_bench_core::specrt_sessions::*;
    let mut h = harness_sized("sessions", Tab::Server, 2400.0);
    let dir = std::env::temp_dir().join(format!("ui-sessions-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let compat = Compat { model_hash: [1; 32], spec_hash: [2; 32], kv_dtype: 0, layers: 1, block_size: 16 };
    let shared = std::sync::Arc::new(Shared::new(SessionStore::open(&dir, compat).unwrap(), SessionOptions { enabled: true, ..Default::default() }));
    let now = now_seconds();
    let id = shared.store.write(&Session { scope: [0; 32], tokens: (0..5000).collect(), sections: vec![vec![0; 32], vec![1; 2 << 20], vec![2; 100], vec![]] }, now - 600).unwrap().id;
    let rows = crate::sessions::rows(&shared, now);
    assert_eq!((rows[0].tokens, rows[0].age_secs, rows[0].expires_in_secs, rows[0].stripped), (5000, 600, 8 * 3600 - 600, false));
    let mut s = script();
    s.sessions = std::sync::Arc::new(std::sync::Mutex::new(Some(shared.clone())));
    {
        let app = h.state_mut();
        app.engine = inference_bench_core::engine::Engine::with_factories_sampled(app.wake.clone(), ScriptedModel::factory(s.clone()), inference_bench_core::fake::ScriptedServing::factory(s), inference_bench_core::engine::no_gpu_sampler());
        app.settings.server.port = 0;
        let m = app.library[0].clone();
        app.load_model(&m);
    }
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    h.run();
    h.get_by_label_contains("5000 tokens");
    h.get_by_label_contains("10m ");
    h.get_by_label("restore").click();
    h.run();
    assert!(h.state().notice.as_ref().is_some_and(|n| n.1.contains("session verified")), "{:?}", h.state().notice);
    h.get_by_label("strip").click();
    h.step();
    h.step();
    assert!(shared.store.list()[0].stripped, "{:?}", h.state().notice);
    h.get_by_label_contains("stripped");
    h.get_by_label("delete").click();
    h.run();
    assert!(shared.store.list().is_empty());
    assert!(!id.is_empty());
    h.get_by_label("no saved sessions");
}

fn two_card_moe_dialog_at(name: &str, size: egui::Vec2) -> Harness<'static, App> {
    let mut h = harness_at(name, Tab::Models, size);
    let free: u64 = (92.7 * (1u64 << 30) as f64) as u64;
    let gpu = |uuid: &str| inference_bench_core::gpu::GpuStats { name: "card".into(), used_bytes: (96 << 30) - free, total_bytes: 96 << 30, util_percent: None, uuid: uuid.into(), own_bytes: 0, display: false, index: None };
    {
        let app = h.state_mut();
        app.gpu = vec![gpu("GPU-fake-0"), gpu("GPU-fake-1")];
        app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
        app.gpu_poll = Default::default();
        let mut m = specrt::hardware::fake_gpus("desk", 2, true);
        for g in &mut m.gpus {
            g.free_bytes = free;
        }
        app.hardware.machine = Some(m);
    }
    let d = dir(&format!("{name}-model"));
    std::fs::create_dir_all(&d).unwrap();
    let (repo, name, arch) = ("Piecrust/GLM-5.3-Flash-FP4-Abliterated", "glm5.3-flash", "Glm5NextForConditionalGeneration");
    std::fs::copy(fixtures().join(format!("{name}.config.json")), d.join("config.json")).unwrap();
    let mut m = model(repo.rsplit('/').next().unwrap(), true, "mixer moe");
    (m.path, m.root) = (d.clone(), d);
    (m.repo, m.architecture, m.quant, m.size_bytes) = (repo.into(), arch.into(), "MODELOPT".into(), 166_400_000_000);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.state_mut().load.as_mut().unwrap().ctx = 8192;
    h.run();
    h
}

#[test]
fn a_mixer_moe_too_big_for_one_card_plans_across_two_with_no_mtp_ticked() {
    let mut h = two_card_moe_dialog_at("moe-two", egui::vec2(1280.0, 800.0));
    let d = h.state().load.as_ref().unwrap();
    assert!(d.view.as_ref().unwrap().spec.speculative().is_none(), "the engine refuses MTP for this family");
    let solved = d.solved.clone().expect("a plan").unwrap_or_else(|e| panic!("{e}"));
    assert!(!solved.mtp, "the plan turns MTP on");
    assert_eq!(solved.placement.stages.len(), 2, "{}", solved.placement.render());
    assert!(h.query_by_label_contains("does not fit").is_none());
    assert!(h.query_by_label("mtp").is_none(), "the MTP row is shown in the common options");
    assert!(!click_load_node(&mut h));
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (shot, mut at) in [("1280x800", h), ("390x844", two_card_moe_dialog_at("moe-phone", egui::vec2(390.0, 844.0)))] {
        at.run();
        at.render().unwrap().save(out.join(format!("two-card-moe-load-{shot}.png"))).unwrap();
    }
}

fn two_gpu_27b_dialog(name: &str) -> Harness<'static, App> {
    two_gpu_27b_dialog_at(name, egui::vec2(1280.0, 800.0))
}

fn two_gpu_27b_dialog_at(name: &str, size: egui::Vec2) -> Harness<'static, App> {
    let mut h = harness_at(name, Tab::Models, size);
    let gpu = |total: u64| inference_bench_core::gpu::GpuStats { name: "card".into(), used_bytes: 0, total_bytes: total << 30, util_percent: None, uuid: String::new(), own_bytes: 0, display: false, index: None };
    {
        let app = h.state_mut();
        app.gpu = vec![gpu(96), gpu(96)];
        app.gpu_at = Some(Instant::now() + Duration::from_secs(600));
        app.gpu_poll = Default::default();
        let mut m = specrt::hardware::fake_gpus("desk", 2, true);
        for g in &mut m.gpus {
            g.free_bytes = 96 << 30;
        }
        app.hardware.machine = Some(m);
    }
    let d = dir(&format!("{name}-model"));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(fixtures().join("qwen3.8-27b-nvfp4.config.json"), d.join("config.json")).unwrap();
    let mut m = model("Qwen3.8-27B-NVFP4", true, "dense decoder");
    (m.path, m.root) = (d.clone(), d);
    m.context_length = None;
    h.state_mut().library[0] = m.clone();
    h.state_mut().open_load_dialog(&m);
    h.run();
    h
}

#[test]
fn the_user_settings_on_two_96gb_gpus_plan_across_both() {
    let mut h = two_gpu_27b_dialog("two96");
    {
        let d = h.state_mut().load.as_mut().unwrap();
        d.choice.set("runtime.kv_dtype", serde_json::json!("fp8"));
        d.choice.set("runtime.kv_f16_layers", serde_json::json!(0));
        d.seqs = 21;
        d.pin_seqs = true;
        d.ctx = 262144;
        d.sync();
    }
    h.run();
    let app = h.state();
    let devices = crate::hardware::load_devices(app);
    let d = app.load.as_ref().unwrap();
    let planned = d.planned(&devices).unwrap().unwrap();
    assert_eq!(planned.placement.stages.len(), 1);
    assert_eq!(planned.placement.stages[0].tp, 2);
    assert_eq!(planned.placement.slots, 21);
    assert_eq!(planned.placement.context.min(262144), planned.placement.context);
    h.get_by_label_contains("TP 2 across desk / GPU 0 + desk / GPU 1");
    h.get_by_label_contains("GPU 0 layers 0-");
    h.get_by_label_contains("GPU 1 layers 0-");
}

type Seen = std::sync::Arc<std::sync::Mutex<Vec<specrt::serve::ServeOptions>>>;

const SCRIPTED_REPO: &str = "acme/dense-27b-nvfp4";

fn recording_engine(app: &mut App) -> Seen {
    let seen: Seen = Default::default();
    let log = seen.clone();
    let script = Script { models: vec![SCRIPTED_REPO.into()], ..script() };
    let fake = inference_bench_core::fake::ScriptedServing::factory(script.clone());
    let serving: inference_bench_core::engine::ServingFactory = std::sync::Arc::new(move |opts| {
        log.lock().unwrap().push(opts.clone());
        fake(opts)
    });
    app.engine = inference_bench_core::engine::Engine::with_factories_sampled(app.wake.clone(), ScriptedModel::factory(script), serving, inference_bench_core::engine::no_gpu_sampler());
    seen
}

#[test]
fn the_engine_receives_exactly_the_solved_plan_the_dialog_showed() {
    let mut h = two_gpu_27b_dialog("two96load");
    let seen = recording_engine(h.state_mut());
    {
        let app = h.state_mut();
        app.library[0].repo = SCRIPTED_REPO.into();
        let d = app.load.as_mut().unwrap();
        d.model.repo = SCRIPTED_REPO.into();
        d.seqs = 21;
        d.pin_seqs = true;
        d.ctx = 262144;
        d.sync();
    }
    h.run();
    let shown = h.state().load.as_ref().unwrap().solved.clone().unwrap().unwrap();
    assert_eq!(shown.tp, 2);
    click_load(&mut h);
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    let loaded = h.state().loaded().unwrap();
    assert_eq!(loaded.max_num_seqs, 21);
    let calls = seen.lock().unwrap().clone();
    assert_eq!(calls.len(), 1, "{} calls", calls.len());
    let got = calls[0].solved.as_deref().expect("the engine is given the solved plan");
    assert_eq!(got, &shown);
    assert!(calls[0].placement.is_none());
    assert_eq!((calls[0].tp, calls[0].max_num_seqs, calls[0].kv_cache_bytes), (shown.tp, shown.slots, Some(shown.kv_pool_bytes)));
}

#[test]
fn unticking_gpu_0_in_the_dialog_plans_and_loads_on_gpu_1_only() {
    let mut h = two_gpu_27b_dialog("untick-gpu0");
    let seen = recording_engine(h.state_mut());
    {
        let app = h.state_mut();
        app.library[0].repo = SCRIPTED_REPO.into();
        app.load.as_mut().unwrap().model.repo = SCRIPTED_REPO.into();
    }
    h.get_by_label_contains("GPU 0 layers 0-");
    click_settled(&mut h, "GPU 0  ");
    h.run();
    assert!(h.state().settings.cluster.disabled_gpus.contains("desk/gpu0"));
    let shown = h.state().load.as_ref().unwrap().solved.clone().unwrap().unwrap();
    assert_eq!(shown.ordinals(), vec![1], "{}", shown.strategy);
    h.get_by_label_contains("fits on desk / GPU 1");
    h.get_by_label_contains("GPU 1 layers 0-");
    assert!(h.query_by_label_contains("GPU 0 layers").is_none());
    click_load(&mut h);
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    let calls = seen.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    let got = calls[0].solved.as_deref().expect("the engine is given the solved plan");
    assert_eq!(got.ordinals(), vec![1]);
    assert_eq!(calls[0].tp, 1);
}

#[test]
fn with_every_gpu_unticked_the_dialog_refuses_to_load_on_a_default_gpu() {
    let mut h = two_gpu_27b_dialog("untick-all");
    let seen = recording_engine(h.state_mut());
    click_settled(&mut h, "GPU 0  ");
    h.run();
    click_settled(&mut h, "GPU 1  ");
    h.run();
    assert_eq!(h.state().settings.cluster.disabled_gpus.len(), 2);
    click_load(&mut h);
    for _ in 0..5 {
        h.run();
    }
    assert!(seen.lock().unwrap().is_empty(), "nothing may load with no gpu selected");
    let notice = h.state().notice.clone().unwrap();
    assert!(!notice.0 && notice.1.contains("no gpu selected"), "{notice:?}");
}

#[test]
fn with_gpu_0_unticked_a_model_with_no_plan_is_not_loaded_on_gpu_0() {
    let mut h = two_gpu_27b_dialog("unplanned-gpu0-off");
    let seen = recording_engine(h.state_mut());
    click_settled(&mut h, "GPU 0  ");
    h.run();
    {
        let d = h.state_mut().load.as_mut().unwrap();
        d.view = Err("no spec for this architecture".into());
        d.solved = d.solve(&[]);
    }
    h.run();
    assert!(h.state().load.as_ref().unwrap().solved.is_none());
    click_load(&mut h);
    for _ in 0..5 {
        h.run();
    }
    assert!(seen.lock().unwrap().is_empty(), "an unplanned load would open GPU 0");
    let notice = h.state().notice.clone().unwrap();
    assert!(!notice.0 && notice.1.contains("GPU 0, which is switched off"), "{notice:?}");
}

#[test]
fn a_model_that_lands_on_the_wrong_gpu_is_reported_against_the_plan() {
    let mut h = two_gpu_27b_dialog("landing");
    recording_engine(h.state_mut());
    {
        let app = h.state_mut();
        app.library[0].repo = SCRIPTED_REPO.into();
        app.load.as_mut().unwrap().model.repo = SCRIPTED_REPO.into();
    }
    click_settled(&mut h, "GPU 0  ");
    h.run();
    click_load(&mut h);
    pump_until(&mut h, 20, "load", |a| a.loaded().is_some());
    let mut l = h.state().loaded().unwrap();
    l.landed.clear();
    assert!(h.state().landing(&l).is_none());
    l.landed = vec![("GPU-fake-1".into(), 40 << 30)];
    let ok = h.state().landing(&l).unwrap();
    assert!(ok.matches_plan && ok.text.starts_with("running on GPU 1"), "{ok:?}");
    l.landed = vec![("GPU-fake-0".into(), 40 << 30)];
    let wrong = h.state().landing(&l).unwrap();
    assert!(!wrong.matches_plan && wrong.text.starts_with("running on GPU 0"), "{wrong:?}");
}

#[test]
fn the_dialog_lists_every_gpu_and_marks_the_one_driving_a_display() {
    let mut h = two_gpu_27b_dialog("display-gpu");
    h.state_mut().gpu[0].display = true;
    h.run();
    let line = h.get_by_label_contains("GPU 0  ").accesskit_node().label().unwrap_or_default().to_string();
    assert!(line.contains("drives a display"), "{line}");
    let other = h.get_by_label_contains("GPU 1  ").accesskit_node().label().unwrap_or_default().to_string();
    assert!(!other.contains("drives a display"), "{other}");
    assert!(other.contains("free of"), "{other}");
}

#[test]
fn low_memory_on_two_gpus_plans_one_gpu_and_shows_one_plan_row() {
    let mut h = two_gpu_27b_dialog("low-memory-one-gpu");
    h.state_mut().load.as_mut().unwrap().choice.select(specrt::options::Profile::LowMemory);
    h.run();
    let rows = ["GPU 0 layers 0-", "GPU 1 layers 0-"].iter().map(|l| h.query_all_by_label_contains(l).count()).sum::<usize>();
    assert_eq!(rows, 1);
}

#[test]
fn the_load_dialog_fits_every_window_size() {
    for (w, hgt) in [(1024.0, 700.0), (1280.0, 800.0), (1920.0, 1080.0), (390.0, 844.0)] {
        let mut h = two_gpu_27b_dialog_at(&format!("dialog-fit-{w}"), egui::vec2(w, hgt));
        for key in ["load model", "cancel"] {
            let b = h.get_all_by_label(key).last().unwrap().accesskit_node().bounding_box().unwrap();
            assert!(b.x0 >= 0.0 && b.x1 <= w as f64 && b.y0 >= 0.0 && b.y1 <= hgt as f64, "{key} at {b:?} outside {w}x{hgt}");
        }
        let hw = h.get_by_label_contains("GPU 1  ").accesskit_node().bounding_box().unwrap();
        assert!(hw.x1 <= w as f64, "hardware row {hw:?} outside {w}");
        h.state_mut().load.as_mut().unwrap().advanced = false;
    }
}

#[test]
fn the_dialog_shows_the_placement_strategy_for_each_profile() {
    let mut h = two_gpu_27b_dialog("strategy-lines");
    h.get_by_label_contains("placement TP 2 across desk / GPU 0 + desk / GPU 1");
    h.state_mut().load.as_mut().unwrap().choice.select(specrt::options::Profile::LowMemory);
    h.run();
    h.get_by_label_contains("smallest that fits");
    let d = h.state().load.as_ref().unwrap();
    let s = d.solved.as_ref().unwrap().as_ref().unwrap();
    assert_eq!((s.tp, s.placement.stages.len(), s.slots), (1, 1, 2));
}

fn two_model_script() -> Script {
    Script { models: vec!["acme/Qwen3-0.6B".into(), "acme/Other-1B".into()], ..script() }
}

fn load_both(h: &mut Harness<'static, App>) -> (LocalModel, LocalModel) {
    with_gpu(h, 0, 96 << 30);
    scripted(h.state_mut(), two_model_script());
    h.state_mut().settings.server.max_loaded_models = 2;
    let (a, b) = (real_model("Qwen3-0.6B"), real_model("Other-1B"));
    h.state_mut().library = vec![a.clone(), b.clone()];
    h.state_mut().load_model_placed(&a, 4096, 1, None, Vec::new(), Default::default(), true);
    pump_until(h, 10, "first load", |app| app.loaded_all().len() == 1);
    h.state_mut().load_model_placed(&b, 4096, 1, None, Vec::new(), Default::default(), true);
    pump_until(h, 10, "second load", |app| app.loaded_all().len() == 2);
    (a, b)
}

#[test]
fn the_dialog_offers_load_alongside_and_both_models_stay_loaded() {
    let mut h = harness("alongside", Tab::Models);
    let (a, _) = load_both(&mut h);
    assert_eq!(h.state().loaded_all().len(), 2);
    h.state_mut().engine.send(Command::Unload { model: Some("Other-1B".into()) });
    pump_until(&mut h, 10, "unload one", |app| app.loaded_all().len() == 1);
    h.state_mut().open_load_dialog(&a);
    h.run();
    h.get_by_label("replace current");
    assert!(!h.get_by_label("load alongside").accesskit_node().is_disabled());
}

#[test]
fn load_alongside_is_disabled_with_a_reason_at_the_model_limit() {
    let mut h = harness("alongside-limit", Tab::Models);
    with_gpu(&mut h, 0, 96 << 30);
    scripted(h.state_mut(), two_model_script());
    let a = real_model("Qwen3-0.6B");
    h.state_mut().library = vec![a.clone()];
    h.state_mut().load_model(&a);
    pump_until(&mut h, 10, "load", |app| app.loaded().is_some());
    h.state_mut().open_load_dialog(&a);
    h.run();
    assert!(h.get_by_label("load alongside").accesskit_node().is_disabled());
    assert!(h.query_all_by_label_contains("cannot load alongside").count() >= 1);
}

#[test]
fn the_server_tab_lists_every_loaded_model_and_unloads_one() {
    let mut h = harness("loaded-list", Tab::Server);
    load_both(&mut h);
    h.run();
    assert_eq!(h.get_all_by_label("unload").count(), 3);
    h.get_all_by_label("unload").nth(1).unwrap().click();
    pump_until(&mut h, 10, "unload one", |app| app.loaded_all().len() == 1);
    assert_eq!(h.state().loaded_all()[0].display, "Other-1B");
}

#[test]
fn the_chat_follows_the_model_picked_among_the_loaded() {
    let mut h = harness("chat-pick", Tab::Chat);
    load_both(&mut h);
    h.state_mut().chat_model = Some("Qwen3-0.6B".into());
    h.run();
    assert_eq!(h.state().loaded().unwrap().display, "Qwen3-0.6B");
    h.state_mut().chat_model = Some("Other-1B".into());
    h.run();
    assert_eq!(h.state().loaded().unwrap().display, "Other-1B");
    assert_eq!(h.state().chat.agent.chat.config.model, "Other-1B");
}

fn strip_requests(h: &Harness<'static, App>) -> u64 {
    crate::serving::snapshot(h.state(), h.state().server.span).reading.map_or(0, |r| r.requests)
}

fn pick_in_bar(h: &mut Harness<'static, App>, display: &str) {
    h.get_all_by_label(display).next().unwrap().click();
    h.run();
}

#[test]
fn the_stats_strip_follows_the_model_picked_in_the_bar_and_names_it() {
    let mut h = harness("strip-per-model", Tab::Server);
    load_both(&mut h);
    let handle = h.state().engine.handle();
    for _ in 0..2 {
        let rx = handle.chat_on(Some("Qwen3-0.6B"), ChatRequest::new(vec![Message::new(Role::User, "hi")], Sampling::default()));
        while !matches!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), ChatEvent::Done(_) | ChatEvent::Failed(_)) {}
    }
    let rx = handle.chat_on(Some("Other-1B"), ChatRequest::new(vec![Message::new(Role::User, "hi")], Sampling::default()));
    while !matches!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), ChatEvent::Done(_) | ChatEvent::Failed(_)) {}
    h.state_mut().server.running = None;
    pump_until(&mut h, 10, "samples", |a| {
        let seen = |m: &str| a.engine.handle().history_of(Some(m)).lock().unwrap().latest().map_or(0, |r| r.requests);
        seen("Qwen3-0.6B") == 2 && seen("Other-1B") == 1
    });
    h.state_mut().chat_model = Some("Qwen3-0.6B".into());
    h.run();
    h.get_by_label_contains("stats for Qwen3-0.6B");
    assert!(h.query_all_by_label_contains("stats for Other-1B").next().is_none());
    assert_eq!(strip_requests(&h), 2);
    pick_in_bar(&mut h, "Other-1B");
    h.get_by_label_contains("stats for Other-1B");
    assert!(h.query_all_by_label_contains("stats for Qwen3-0.6B").next().is_none());
    assert_eq!(strip_requests(&h), 1);
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=docs/gui"]
fn screenshot_of_the_server_tab_with_two_models_loaded() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    let mut h = harness_sized("shot-two-models", Tab::Server, 1300.0);
    load_both(&mut h);
    let handle = h.state().engine.handle();
    for m in ["Qwen3-0.6B", "Other-1B"] {
        let rx = handle.chat_on(Some(m), ChatRequest::new(vec![Message::new(Role::User, "hi")], Sampling::default()));
        while !matches!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), ChatEvent::Done(_) | ChatEvent::Failed(_)) {}
    }
    h.state_mut().chat_model = Some("Qwen3-0.6B".into());
    pump_until(&mut h, 10, "samples", |a| a.engine.handle().history_of(Some("Other-1B")).lock().unwrap().latest().is_some_and(|r| r.requests == 1));
    h.run();
    h.render().unwrap().save(out.join("server-two-models.png")).unwrap();
}
