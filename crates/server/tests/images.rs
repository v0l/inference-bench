mod common;

use base64::Engine as _;
use common::*;
use inference_bench_core::fake::{FakeImage, tiny_pipeline};
use inference_bench_core::native::Native;
use inference_bench_core::result::BackendError;
use inference_bench_core::target::Target;
use serde_json::{Value, json};
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn rig(step_ms: u64, extra: &str) -> (Rig, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let r = runs.clone();
    let factory = move |ctx: Option<usize>| -> Result<Box<dyn Target>, BackendError> { Ok(Box::new(Native::with_factory(ctx, FakeImage::factory(step_ms, r.clone(), None)))) };
    let dir = tiny_pipeline().display().to_string();
    let rig = start_with(config(&format!("{extra}\n[[models]]\nid='klein'\nmodel={dir:?}\n")), factory, Default::default());
    (rig, runs)
}

fn decode(b64: &str) -> image::RgbImage {
    let png = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
    image::load_from_memory(&png).unwrap().to_rgb8()
}

fn generate(rig: &Rig, body: Value) -> (u16, Value, Vec<(String, String)>) {
    let (status, text, headers) = rig.post("/v1/images/generations", None, body);
    (status, serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text})), headers)
}

#[test]
fn the_response_decodes_to_an_image_of_the_requested_size_and_the_seed_is_repeatable() {
    let (rig, _) = rig(1, "");
    let (status, v, _) = generate(&rig, json!({"model": "klein", "prompt": "a fox", "size": "48x32", "seed": 3, "steps": 2}));
    assert_eq!(status, 200, "{v}");
    let img = decode(v["data"][0]["b64_json"].as_str().unwrap());
    assert_eq!((img.width(), img.height()), (48, 32));
    assert_eq!(v["data"][0]["seed"], 3);
    assert_eq!(v["usage"]["steps"], 2);
    assert!(v["created"].as_i64().unwrap() > 0);
    let (_, again, _) = generate(&rig, json!({"model": "klein", "prompt": "a fox", "size": "48x32", "seed": 3, "steps": 2}));
    assert_eq!(again["data"][0]["b64_json"], v["data"][0]["b64_json"], "the same seed gives the same bytes");
    rig.stop();
}

#[test]
fn n_two_gives_two_different_images_with_consecutive_seeds() {
    let (rig, _) = rig(1, "");
    let (status, v, _) = generate(&rig, json!({"prompt": "a fox", "size": "32x32", "seed": 10, "n": 2}));
    assert_eq!(status, 200, "{v}");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_ne!(data[0]["b64_json"], data[1]["b64_json"]);
    assert_eq!((data[0]["seed"].as_u64(), data[1]["seed"].as_u64()), (Some(10), Some(11)));
    rig.stop();
}

#[test]
fn an_invalid_size_is_a_400_naming_the_nearest_valid_size_and_runs_nothing() {
    let (rig, runs) = rig(1, "");
    let (status, v, _) = generate(&rig, json!({"prompt": "a fox", "size": "30x32"}));
    assert_eq!(status, 400, "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("not a multiple of 4") && msg.contains("32x32"), "{msg}");
    let (status, v, _) = generate(&rig, json!({"prompt": "a fox", "size": "square"}));
    assert_eq!(status, 400, "{v}");
    let (status, _, _) = generate(&rig, json!({"prompt": "a fox", "n": 9, "size": "32x32"}));
    assert_eq!(status, 400);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    rig.stop();
}

#[test]
fn a_full_image_queue_is_a_429_with_retry_after() {
    let (rig, _) = rig(40, "retry_after_secs = 3\n[images]\njobs_per_gpu = 1\nqueue_depth = 1\n");
    let url = format!("{}/v1/images/generations", rig.url);
    let body = json!({"prompt": "a fox", "size": "32x32", "steps": 25});
    let spawn = |url: String, body: Value| std::thread::spawn(move || agent().post(&url).send_json(&body).unwrap().status().as_u16());
    let a = spawn(url.clone(), body.clone());
    wait_for("the first job to run", || metric_of(&rig, "inference_image_jobs_running") >= 1.0);
    let b = spawn(url.clone(), body.clone());
    wait_for("the second job to queue", || metric_of(&rig, "inference_image_queue_depth") >= 1.0);
    let (status, v, headers) = generate(&rig, body);
    assert_eq!(status, 429, "{v}");
    assert_eq!(header(&headers, "retry-after"), Some("3"));
    assert!(v["error"]["message"].as_str().unwrap().contains("image queue is full"));
    assert_eq!((a.join().unwrap(), b.join().unwrap()), (200, 200));
    let m = rig.metrics();
    assert!(m.contains("inference_image_jobs_total{model=\"klein\"} 2"), "{m}");
    assert!(m.contains("inference_requests_total{endpoint=\"images_generations\",status=\"429\"} 1"), "{m}");
    rig.stop();
}

fn metric_of(rig: &Rig, name: &str) -> f64 {
    rig.metrics().lines().find_map(|l| l.strip_prefix(&format!("{name} ")).map(|v| v.parse().unwrap())).unwrap_or(0.0)
}

#[test]
fn stream_sends_step_progress_previews_then_the_image() {
    let (rig, _) = rig(1, "");
    let mut s = rig.raw_post("/v1/images/generations", &json!({"prompt": "a fox", "size": "32x32", "steps": 4, "stream": true, "partial_images": 2}).to_string());
    let text = read_until(&mut s, "image_generation.completed\ndata:");
    let text = text + &read_until(&mut s, "\n\n");
    let progress: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).filter_map(|d| serde_json::from_str::<Value>(d).ok()).filter(|v| v["type"] == "image_generation.progress").collect();
    assert_eq!(progress.iter().map(|v| v["step"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert!(progress.iter().all(|v| v["steps"] == 4));
    assert_eq!(text.matches("event: image_generation.partial_image").count(), 2, "{text}");
    let done = text.lines().filter_map(|l| l.strip_prefix("data: ")).filter_map(|d| serde_json::from_str::<Value>(d).ok()).find(|v| v["type"] == "image_generation.completed").expect(&text);
    let img = decode(done["b64_json"].as_str().unwrap());
    assert_eq!((img.width(), img.height()), (32, 32));
    rig.stop();
}

#[test]
fn a_dropped_client_cancels_the_job_between_steps() {
    let (rig, _) = rig(30, "");
    let started = Instant::now();
    {
        let mut s = rig.raw_post("/v1/images/generations", &json!({"prompt": "a fox", "size": "32x32", "steps": 200, "stream": true}).to_string());
        read_until(&mut s, "\"step\":2");
    }
    wait_for("the cancel to count", || rig.metrics().contains("inference_requests_cancelled_total 1"));
    assert!(started.elapsed() < Duration::from_secs(5), "200 steps of 30 ms did not run");
    wait_for("the gate to free", || metric_of(&rig, "inference_image_jobs_running") == 0.0);
    let (status, _, _) = generate(&rig, json!({"prompt": "a fox", "size": "32x32", "steps": 1}));
    assert_eq!(status, 200, "the GPU is free for the next job");
    rig.stop();
}

#[test]
fn url_responses_are_served_from_a_short_lived_file_behind_auth() {
    let (rig, _) = rig(1, "api_keys = ['k']\n[images]\nfile_ttl_secs = 60\n");
    let (status, text, _) = rig.post("/v1/images/generations", Some("k"), json!({"prompt": "a fox", "size": "32x32", "response_format": "url"}));
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let url = v["data"][0]["url"].as_str().unwrap();
    let path = &url[url.find("/v1/").unwrap()..];
    assert!(path.starts_with("/v1/images/files/") && path.ends_with(".png"), "{url}");
    assert_eq!(rig.get(path, None).0, 401);
    let resp = agent().get(format!("{}{path}", rig.url)).header("Authorization", "Bearer k").call().unwrap();
    assert_eq!(resp.headers().get("content-type").unwrap(), "image/png");
    let mut bytes = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut bytes).unwrap();
    assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 32);
    assert_eq!(rig.get("/v1/images/files/../../etc/passwd", Some("k")).0, 404);
    assert_eq!(rig.get("/v1/images/files/abc.png", Some("k")).0, 404);
    rig.stop();
}

#[test]
fn models_mark_their_type_and_text_routes_refuse_an_image_model() {
    let (rig, _) = rig(1, "");
    let (_, body, _) = rig.get("/v1/models", None);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["type"], "image", "{v}");
    let (status, _, _) = generate(&rig, json!({"prompt": "a fox", "size": "32x32", "steps": 1}));
    assert_eq!(status, 200);
    let (status, body, _) = rig.post("/v1/chat/completions", None, json!({"model": "klein", "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("/v1/images/generations"), "{body}");
    let (_, body, _) = rig.get("/v1/capabilities", None);
    let caps: Value = serde_json::from_str(&body).unwrap();
    let route = caps["routes"].as_array().unwrap().iter().find(|r| r["path"] == "/v1/images/generations").unwrap();
    let names: Vec<&str> = route["fields"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
    for f in ["prompt", "size", "seed", "n", "steps", "guidance", "negative_prompt", "sampler", "stream", "response_format"] {
        assert!(names.contains(&f), "{f} in {names:?}");
    }
    assert_eq!(caps["models"][0]["type"], "image");
    rig.stop();
}

#[test]
fn a_text_model_lists_as_text_and_refuses_an_image_job_with_a_400() {
    let rig = start(config("[[models]]\nid='m'"), Shape::default());
    let (_, body, _) = rig.get("/v1/models", None);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["data"][0]["type"], "text");
    let (status, v, _) = generate(&rig, json!({"model": "m", "prompt": "a fox", "size": "32x32"}));
    assert_eq!(status, 400, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("text model"), "{v}");
    rig.stop();
}
