use crate::tests::{harness_at, pump_until, shots_dir, with_gpu};
use crate::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use inference_bench_core::fake::{FakeImage, tiny_video_pipeline};
use inference_bench_core::library::LocalModel;
use std::sync::Arc;

fn shows(h: &Harness<'static, App>, text: &str) -> bool {
    h.query_all(egui_kittest::kittest::by().predicate(move |n| n.value().is_some_and(|v| v.contains(text)) || n.label().is_some_and(|l| l.contains(text)))).next().is_some()
}

fn video_model() -> LocalModel {
    inference_bench_core::library::local_model_at(&tiny_video_pipeline()).expect("the tiny video pipeline scans as a model")
}

fn video_app(name: &str, size: egui::Vec2, step_ms: u64) -> Harness<'static, App> {
    let mut h = harness_at(name, Tab::Models, size);
    with_gpu(&mut h, 0, 96 << 30);
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wake = h.state().wake.clone();
    h.state_mut().engine = inference_bench_core::engine::Engine::with_factories_sampled(wake, FakeImage::factory(step_ms, runs, None), Arc::new(|_| None), inference_bench_core::engine::no_gpu_sampler());
    h.state_mut().library.push(video_model());
    h
}

fn load_through_the_page(h: &mut Harness<'static, App>) {
    let m = video_model();
    h.state_mut().open_load_dialog(&m);
    h.run();
    h.get_by_label("load model").click();
    h.run();
    pump_until(h, 10, "video load", |a| a.loaded().is_some());
    h.run();
}

#[test]
fn a_video_model_shows_the_video_controls_instead_of_the_image_ones() {
    let mut h = video_app("vid-controls", egui::vec2(1280.0, 800.0), 2);
    load_through_the_page(&mut h);
    for label in ["SECONDS", "FPS", "NEGATIVE PROMPT", "SEQUENTIAL GUIDANCE", "TILED VAE", "generate video"] {
        assert!(shows(&h, label), "{label}");
    }
    assert!(!shows(&h, "generate image"));
    assert!(!shows(&h, "IMAGES"));
}

fn start_generation(h: &mut Harness<'static, App>, prompt: &str) {
    let g = &mut h.state_mut().generate_video;
    g.prompt = prompt.into();
    g.size = "32x32".into();
    g.steps = 2;
    h.run();
    h.get_by_label("generate video").click();
    h.run();
}

fn finish_generation(h: &mut Harness<'static, App>) {
    pump_until(h, 20, "video generation", |a| !a.generate_video.busy());
    h.run();
}

#[test]
fn submit_posts_the_chosen_seconds_and_fps_and_the_result_has_that_many_frames() {
    let mut h = video_app("vid-submit", egui::vec2(1280.0, 800.0), 2);
    load_through_the_page(&mut h);
    {
        let g = &mut h.state_mut().generate_video;
        assert_eq!(g.seconds, 5.0);
        assert_eq!(g.fps, 16);
        g.seconds = 1.0;
        g.fps = 8;
    }
    start_generation(&mut h, "a red fox");
    finish_generation(&mut h);
    let g = &h.state().generate_video;
    assert!(g.error.is_none(), "{:?}", g.error);
    let row = g.rows.last().unwrap();
    assert_eq!((row.outcome.as_str(), row.fps, row.frames), ("done", 8, 9), "1 s at 8 fps is 9 frames for a temporal step of 2");
    let clip = row.clip.as_ref().unwrap();
    assert_eq!((clip.width, clip.height, clip.frames.len()), (32, 32, 9));
    assert!(row.thumb.is_some());
    assert_eq!(g.shown, Some(0));
    assert!(shows(&h, "play") && shows(&h, "save as mp4"));
}

#[test]
fn progress_rows_follow_the_step_and_the_decoded_frames() {
    let mut h = video_app("vid-progress", egui::vec2(1280.0, 800.0), 30);
    load_through_the_page(&mut h);
    h.state_mut().generate_video.prompt = "slow fox".into();
    h.state_mut().generate_video.size = "32x32".into();
    h.state_mut().generate_video.steps = 40;
    h.state_mut().generate_video.seconds = 1.0;
    h.run();
    h.get_by_label("generate video").click();
    h.run();
    pump_until(&mut h, 20, "a few steps", |a| a.generate_video.running.as_ref().is_some_and(|r| r.status.step >= 3));
    h.run();
    let s = h.state().generate_video.running.as_ref().unwrap().status.clone();
    assert_eq!(s.steps, 40);
    assert!(s.steps == 40 && s.step >= 3, "{s:?}");
    h.get_by_label("cancel").click();
    h.run();
    pump_until(&mut h, 20, "cancelled", |a| !a.generate_video.busy());
    let g = &h.state().generate_video;
    assert_eq!(g.rows.last().unwrap().outcome, "stopped");
    assert!(g.error.is_none());
}

#[test]
fn a_busy_server_shows_the_try_again_text() {
    let mut h = video_app("vid-busy", egui::vec2(1280.0, 800.0), 30);
    load_through_the_page(&mut h);
    let base = {
        let handle = h.state().engine.handle();
        h.state_mut().server.ensure_agent_endpoint(handle).unwrap()
    };
    let name = h.state().loaded().unwrap().display;
    let slow = serde_json::json!({"model": name, "prompt": "hold", "size": "32x32", "frames": 5, "steps": 200});
    let mut ids = Vec::new();
    for _ in 0..40 {
        let mut resp = ureq::post(format!("{base}/videos")).config().http_status_as_error(false).build().send_json(&slow).unwrap();
        if resp.status().as_u16() == 429 {
            break;
        }
        let v: serde_json::Value = resp.body_mut().read_json().unwrap();
        ids.push(v["id"].as_str().unwrap().to_string());
    }
    start_generation(&mut h, "a red fox");
    pump_until(&mut h, 20, "the busy answer", |a| !a.generate_video.busy());
    h.run();
    let text = h.state().generate_video.error.clone().unwrap_or_default();
    assert!(text.starts_with("the server is busy, try again in "), "{text}");
    assert!(shows(&h, "the server is busy, try again in"));
    for id in ids {
        let _ = ureq::delete(format!("{base}/videos/{id}")).call();
    }
}

#[test]
fn save_as_mp4_downloads_the_content_and_writes_a_file_with_the_clips_frames_and_fps() {
    let mut h = video_app("vid-save", egui::vec2(1280.0, 800.0), 2);
    load_through_the_page(&mut h);
    h.state_mut().generate_video.seconds = 1.0;
    h.state_mut().generate_video.fps = 8;
    start_generation(&mut h, "a red fox");
    finish_generation(&mut h);
    h.get_by_label("save as mp4").click();
    h.run();
    let start = std::time::Instant::now();
    while !h.state().dir.join("videos").is_dir() && start.elapsed() < std::time::Duration::from_secs(5) {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    eprintln!("saving: {} dir {}", h.state().generate_video.is_saving(), h.state().dir.join("videos").is_dir());
    pump_until(&mut h, 10, "the save notice", |a| a.notice.as_ref().is_some_and(|n| n.1.starts_with("saved ")));
    let dir = h.state().dir.join("videos");
    let file = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    let head = specrt::mp4::header(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!((head.width, head.height, head.frames), (32, 32, 9));
    assert!((head.fps() - 8.0).abs() < 1e-9);
}

#[test]
fn play_steps_frames_at_the_clips_fps_and_stops_on_the_last() {
    let mut p = crate::videogen::Playback::new();
    p.playing = true;
    p.advance(0.125, 8, 9);
    assert_eq!(p.frame, 1);
    p.advance(0.25, 8, 9);
    assert_eq!(p.frame, 3);
    p.advance(10.0, 8, 9);
    assert_eq!((p.frame, p.playing), (8, false));
}

#[test]
fn the_video_page_stacks_on_a_phone_with_a_sideways_strip_and_nothing_clipped() {
    let mut h = video_app("vid-phone", egui::vec2(390.0, 844.0), 2);
    load_through_the_page(&mut h);
    start_generation(&mut h, "a red fox");
    finish_generation(&mut h);
    for label in ["generate video", "play", "save as mp4"] {
        let r = h.get_by_label(label).rect();
        assert!(r.right() <= 390.5 && r.left() >= 0.0, "{label} at {r:?}");
    }
    for text in ["SECONDS", "FPS", "SEQUENTIAL GUIDANCE", "TILED VAE"] {
        let r = h.query_all(egui_kittest::kittest::by().predicate(move |n| n.value().is_some_and(|v| v.contains(text)) || n.label().is_some_and(|l| l.contains(text)))).next().expect(text).rect();
        assert!(r.right() <= 390.5 && r.left() >= 0.0, "{text} at {r:?}");
    }
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_video_page_at_desktop_and_phone_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (name, size) in [("desktop-1280x800", egui::vec2(1280.0, 800.0)), ("phone-390x844", egui::vec2(390.0, 844.0))] {
        let mut h = video_app(&format!("vid-shot-{name}"), size, 2);
        load_through_the_page(&mut h);
        h.render().unwrap().save(out.join(format!("video-controls-{name}.png"))).unwrap();
        h.state_mut().generate_video.seconds = 2.0;
        for p in ["a red fox in studio light", "a lighthouse at dusk"] {
            start_generation(&mut h, p);
            finish_generation(&mut h);
        }
        h.state_mut().generate_video.play.frame = 3;
        h.run();
        h.render().unwrap().save(out.join(format!("video-result-{name}.png"))).unwrap();
    }
}
