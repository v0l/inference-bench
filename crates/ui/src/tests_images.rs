use crate::tests::{harness_at, pump_until, shots_dir, with_gpu};
use crate::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use inference_bench_core::fake::{FakeImage, tiny_pipeline};
use inference_bench_core::library::LocalModel;
use std::sync::Arc;

fn shows(h: &Harness<'static, App>, text: &str) -> bool {
    h.query_all(egui_kittest::kittest::by().predicate(move |n| n.value().is_some_and(|v| v.contains(text)) || n.label().is_some_and(|l| l.contains(text)))).next().is_some()
}

fn image_model() -> LocalModel {
    inference_bench_core::library::local_model_at(&tiny_pipeline()).expect("the tiny pipeline scans as a model")
}

fn image_app(name: &str, size: egui::Vec2) -> Harness<'static, App> {
    let mut h = harness_at(name, Tab::Models, size);
    with_gpu(&mut h, 0, 96 << 30);
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wake = h.state().wake.clone();
    h.state_mut().engine = inference_bench_core::engine::Engine::with_factories_sampled(wake, FakeImage::factory(2, runs, None), Arc::new(|_| None), inference_bench_core::engine::no_gpu_sampler());
    let m = image_model();
    h.state_mut().library.push(m);
    h
}

fn load_through_the_page(h: &mut Harness<'static, App>) {
    let m = image_model();
    h.state_mut().open_load_dialog(&m);
    h.run();
    assert!(h.state().load.is_none(), "an image model does not open the text load dialog");
    assert!(h.state().image_load.is_some());
    h.get_by_label("load model").click();
    h.run();
    pump_until(h, 10, "image load", |a| a.loaded().is_some());
    h.run();
}

fn generate(h: &mut Harness<'static, App>, prompt: &str) {
    h.state_mut().generate.prompt = prompt.into();
    h.state_mut().generate.size = "32x32".into();
    h.run();
    h.get_by_label("generate image").click();
    h.run();
    pump_until(h, 10, "generation", |a| !a.generate.busy());
    h.run();
}

#[test]
fn an_image_model_has_its_own_load_page_with_the_common_options_and_components() {
    let mut h = image_app("img-load", egui::vec2(1280.0, 800.0));
    let m = image_model();
    h.state_mut().open_load_dialog(&m);
    h.run();
    for label in ["SIZE", "STEPS", "GUIDANCE", "TEXT ENCODER", "TRANSFORMER", "VAE", "release after encoding"] {
        assert!(shows(&h, label), "{label}");
    }
    assert!(!shows(&h, "ADVANCED"), "the common options are not behind Advanced");
    let d = h.state().image_load.as_ref().unwrap();
    assert_eq!(d.defaults.size, "1024x1024");
    assert_eq!(d.defaults.steps, 4);
    assert!(d.defaults.offload_text_encoder);
    h.get_by_label("load model").click();
    h.run();
    pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
    h.run();
    assert_eq!(h.state().settings.model_image[&m.name].steps, 4);
    assert_eq!(h.state().tab, Tab::Chat);
    h.get_by_label("generate image");
    h.get_by_label("generate image");
}

#[test]
fn the_generate_page_runs_a_job_shows_the_result_and_a_request_row() {
    let mut h = image_app("img-gen", egui::vec2(1280.0, 800.0));
    load_through_the_page(&mut h);
    assert_eq!(h.state().generate.size, "1024x1024");
    let seed = h.state().generate.seed;
    generate(&mut h, "a red fox");
    let g = &h.state().generate;
    assert_eq!(g.results.len(), 1, "{:?}", g.error);
    assert_eq!(g.results[0].picture.seed, seed);
    assert_eq!((g.results[0].picture.width, g.results[0].picture.height), (32, 32));
    assert_eq!(g.rows.len(), 1);
    assert_eq!(g.rows[0].steps, 4);
    assert!(g.rows[0].steps_per_sec > 0.0 && g.rows[0].total_secs > 0.0);
    assert_eq!(g.seed, seed + 1, "an unlocked seed moves on");
    h.get_by_label("save");
    h.get_by_label("copy");
    h.get_by_label("reuse").click();
    h.run();
    assert_eq!(h.state().generate.seed, seed);
    assert!(h.state().generate.lock_seed);
    h.get_by_label("save").click();
    h.run();
    let saved = h.state().dir.join("images");
    assert_eq!(std::fs::read_dir(&saved).unwrap().count(), 1, "{:?}", h.state().notice);
}

#[test]
fn the_step_bar_shows_while_a_job_runs_and_stop_cancels_it() {
    let mut h = image_app("img-stop", egui::vec2(1280.0, 800.0));
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wake = h.state().wake.clone();
    h.state_mut().engine = inference_bench_core::engine::Engine::with_factories_sampled(wake, FakeImage::factory(40, runs, None), Arc::new(|_| None), inference_bench_core::engine::no_gpu_sampler());
    load_through_the_page(&mut h);
    h.state_mut().generate.prompt = "slow fox".into();
    h.state_mut().generate.steps = 100;
    h.state_mut().generate.size = "32x32".into();
    h.run();
    h.get_by_label("generate image").click();
    pump_until(&mut h, 10, "a step", |a| a.generate.running.as_ref().is_some_and(|r| r.step >= 2));
    h.run();
    assert!(h.state().generate.running.as_ref().is_some_and(|r| r.steps == 100 && r.step >= 2));
    h.get_by_label("stop").click();
    h.run();
    pump_until(&mut h, 10, "stopped", |a| !a.generate.busy());
    let g = &h.state().generate;
    assert!(g.results.is_empty());
    assert_eq!(g.rows.last().unwrap().outcome, "stopped");
    assert!(g.error.is_none());
}

#[test]
fn the_generate_page_stacks_on_a_phone_and_nothing_spills_sideways() {
    let mut h = image_app("img-phone", egui::vec2(390.0, 844.0));
    load_through_the_page(&mut h);
    generate(&mut h, "a red fox");
    for label in ["generate image", "save", "copy"] {
        let node = h.get_by_label(label);
        let r = node.rect();
        assert!(r.right() <= 390.0 + 0.5 && r.left() >= 0.0, "{label} at {r:?}");
    }
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_image_load_page_and_generate_page_at_desktop_and_phone_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (name, size) in [("desktop-1280x800", egui::vec2(1280.0, 800.0)), ("phone-390x844", egui::vec2(390.0, 844.0))] {
        let mut h = image_app(&format!("img-shot-{name}"), size);
        let m = image_model();
        h.state_mut().open_load_dialog(&m);
        h.run();
        h.render().unwrap().save(out.join(format!("image-load-{name}.png"))).unwrap();
        h.get_by_label("load model").click();
        h.run();
        pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
        h.run();
        for p in ["a red fox in studio light", "a lighthouse at dusk"] {
            generate(&mut h, p);
        }
        h.state_mut().generate.prompt = "a slow one".into();
        h.run();
        h.render().unwrap().save(out.join(format!("image-generate-{name}.png"))).unwrap();
    }
}
