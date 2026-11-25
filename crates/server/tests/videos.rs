mod common;

use common::*;
use inference_bench_core::fake::{FakeImage, tiny_video_pipeline};
use inference_bench_core::native::Native;
use inference_bench_core::result::BackendError;
use inference_bench_core::target::Target;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn rig(step_ms: u64, extra: &str) -> (Rig, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let r = runs.clone();
    let factory = move |ctx: Option<usize>| -> Result<Box<dyn Target>, BackendError> { Ok(Box::new(Native::with_factory(ctx, FakeImage::factory(step_ms, r.clone(), None)))) };
    let dir = tiny_video_pipeline().display().to_string();
    let rig = start_with(config(&format!("{extra}\n[[models]]\nid='wan'\nmodel={dir:?}\n")), factory, Default::default());
    (rig, runs)
}

fn create(rig: &Rig, body: Value) -> (u16, Value, Vec<(String, String)>) {
    let (status, text, headers) = rig.post("/v1/videos", None, body);
    (status, serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text})), headers)
}

fn status_of(rig: &Rig, id: &str) -> Value {
    let (status, text, _) = rig.get(&format!("/v1/videos/{id}"), None);
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap()
}

fn wait_done(rig: &Rig, id: &str) -> Value {
    let start = Instant::now();
    loop {
        let v = status_of(rig, id);
        if matches!(v["status"].as_str(), Some("completed" | "failed" | "cancelled")) {
            return v;
        }
        assert!(start.elapsed() < Duration::from_secs(20), "{v}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn seconds_and_fps_reach_the_mp4_header_as_frame_count_and_rate() {
    let (rig, _) = rig(1, "");
    let (status, v, _) = create(&rig, json!({"model": "wan", "prompt": "a fox", "size": "32x32", "seconds": "1", "fps": 8, "steps": 2}));
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["object"], "video");
    let id = v["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("video_"));
    let done = wait_done(&rig, &id);
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!((done["frames"].as_u64(), done["fps"].as_u64(), done["progress"].as_u64()), (Some(9), Some(8), Some(100)), "1 s at 8 fps is 9 frames for a temporal step of 2: {done}");
    let (status, mp4, headers) = rig.get_bytes(&format!("/v1/videos/{id}/content"));
    assert_eq!(status, 200);
    assert_eq!(header(&headers, "content-type"), Some("video/mp4"));
    let h = inference_bench_core::specrt_mp4::header(&mp4).unwrap();
    assert_eq!((h.width, h.height, h.frames, h.codec.as_str()), (32, 32, 9, "avc1"));
    assert!((h.fps() - 8.0).abs() < 1e-9, "{}", h.fps());
    rig.stop();
}

#[test]
fn a_bad_frame_count_fails_the_job_naming_the_nearest_valid_count_and_runs_nothing() {
    let (rig, runs) = rig(1, "");
    let (status, v, _) = create(&rig, json!({"prompt": "a fox", "size": "32x32", "frames": 4}));
    assert_eq!(status, 200, "{v}");
    let done = wait_done(&rig, v["id"].as_str().unwrap());
    assert_eq!(done["status"], "failed", "{done}");
    assert!(done["error"]["message"].as_str().unwrap().contains("nearest valid count is 5"), "{done}");
    let (status, v, _) = create(&rig, json!({"prompt": "", "size": "32x32"}));
    assert_eq!(status, 400, "{v}");
    let (status, v, _) = create(&rig, json!({"prompt": "a fox", "n": 2}));
    assert_eq!(status, 400, "{v}");
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let (status, text, _) = rig.get("/v1/videos/video_nope/content", None);
    assert_eq!(status, 404, "{text}");
    rig.stop();
}

#[test]
fn a_full_media_queue_is_a_429_with_retry_after() {
    let (rig, _) = rig(40, "retry_after_secs = 3\n[images]\njobs_per_gpu = 1\nqueue_depth = 1\n");
    let body = json!({"prompt": "a fox", "size": "32x32", "frames": 5, "steps": 25});
    let (a, va, _) = create(&rig, body.clone());
    let (b, vb, _) = create(&rig, body.clone());
    assert_eq!((a, b), (200, 200), "{va} {vb}");
    let (status, v, headers) = create(&rig, body);
    assert_eq!(status, 429, "{v}");
    assert_eq!(header(&headers, "retry-after"), Some("3"));
    assert_eq!(wait_done(&rig, va["id"].as_str().unwrap())["status"], "completed");
    assert_eq!(wait_done(&rig, vb["id"].as_str().unwrap())["status"], "completed");
    let (status, _, _) = create(&rig, json!({"prompt": "a fox", "size": "32x32", "frames": 5, "steps": 1}));
    assert_eq!(status, 200, "the queue frees once the jobs finish");
    rig.stop();
}

#[test]
fn progress_rises_by_step_then_by_decoded_frame_and_delete_cancels() {
    let (rig, runs) = rig(30, "");
    let (_, v, _) = create(&rig, json!({"prompt": "a fox", "size": "32x32", "frames": 5, "steps": 200}));
    let id = v["id"].as_str().unwrap().to_string();
    wait_for("a few steps", || status_of(&rig, &id)["step"].as_u64().unwrap_or(0) >= 3);
    let mid = status_of(&rig, &id);
    assert_eq!(mid["status"], "in_progress", "{mid}");
    assert!(mid["progress"].as_u64().unwrap() < 10, "{mid}");
    let (status, text, _) = rig.get(&format!("/v1/videos/{id}/content"), None);
    assert_eq!(status, 404, "{text}");
    assert!(text.contains("in_progress"), "{text}");
    let started = Instant::now();
    let (status, text, _) = rig.delete(&format!("/v1/videos/{id}"));
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("\"cancelled\""), "{text}");
    wait_for("the cancel to count", || rig.metrics().contains("inference_requests_cancelled_total 1"));
    assert!(started.elapsed() < Duration::from_secs(5), "200 steps of 30 ms did not run");
    let (status, _, _) = rig.get(&format!("/v1/videos/{id}"), None);
    assert_eq!(status, 404);
    let (status, v, _) = create(&rig, json!({"prompt": "a fox", "size": "32x32", "frames": 5, "steps": 1}));
    assert_eq!(status, 200, "{v}");
    assert_eq!(wait_done(&rig, v["id"].as_str().unwrap())["status"], "completed");
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    rig.stop();
}

#[test]
fn memory_options_are_accepted_and_a_queued_job_reports_its_position() {
    let (rig, _) = rig(40, "[images]\njobs_per_gpu = 1\nqueue_depth = 2\n");
    let body = json!({"prompt": "a fox", "size": "32x32", "frames": 5, "steps": 10, "sequential_cfg": true, "tiled_vae": false});
    let (a, va, _) = create(&rig, body.clone());
    let (b, vb, _) = create(&rig, body);
    assert_eq!((a, b), (200, 200), "{va} {vb}");
    let queued = status_of(&rig, vb["id"].as_str().unwrap());
    assert_eq!(queued["status"], "queued", "{queued}");
    assert_eq!(queued["queue_position"], 1, "{queued}");
    assert_eq!(wait_done(&rig, vb["id"].as_str().unwrap())["status"], "completed");
    rig.stop();
}

#[test]
fn the_models_list_and_capabilities_name_the_video_route() {
    let (rig, _) = rig(1, "");
    let (_, models, _) = rig.get("/v1/models", None);
    let (_, caps, _) = rig.get("/v1/capabilities", None);
    assert!(caps.contains("\"/v1/videos\"") && caps.contains("\"seconds\""), "{caps}");
    let _ = models;
    rig.stop();
}
