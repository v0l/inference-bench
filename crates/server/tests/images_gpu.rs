mod common;

use base64::Engine as _;
use common::*;
use serde_json::{Value, json};
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/tiny-flux2-klein")
}

fn reference(key: &str) -> Vec<f32> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-flux2-klein.json")).unwrap()).unwrap();
    let file = meta["files"][key]["file"].as_str().unwrap();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn recorded_prompt(key: &str) -> String {
    let ids = reference(&format!("{key}.ids"));
    let mask = reference(&format!("{key}.mask"));
    ids.iter().zip(&mask).filter(|(_, m)| **m == 1.0).map(|(i, _)| format!("w{}", *i as u32)).collect::<Vec<_>>().join(" ")
}

fn rig() -> Rig {
    let dir = fixture().display().to_string();
    start_with(config(&format!("[[models]]\nid='tiny-klein'\nmodel={dir:?}\n")), inference_bench_server::NativeFactory, Default::default())
}

fn rgb(b64: &str) -> image::RgbImage {
    image::load_from_memory(&base64::engine::general_purpose::STANDARD.decode(b64).unwrap()).unwrap().to_rgb8()
}

fn post(rig: &Rig, body: Value) -> (u16, Value) {
    let (status, text, _) = rig.post("/v1/images/generations", None, body);
    (status, serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text})))
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test -p inference-bench-server --test images_gpu -- --ignored --test-threads=1"]
fn the_tiny_klein_pipeline_is_served_end_to_end_and_matches_the_reference_image() {
    let rig = rig();
    let body = json!({"model": "tiny-klein", "prompt": recorded_prompt("text"), "size": "32x32", "seed": 0, "steps": 3, "guidance": 1.0});
    let (status, v) = post(&rig, body.clone());
    assert_eq!(status, 200, "{v}");
    let img = rgb(v["data"][0]["b64_json"].as_str().unwrap());
    assert_eq!((img.width(), img.height()), (32, 32));
    let want = reference("bf16.image");
    let plane = 32 * 32;
    let mut worst = 0i32;
    for (i, px) in img.pixels().enumerate() {
        for c in 0..3 {
            let w = (want[c * plane + i].clamp(0.0, 1.0) * 255.0).round() as i32;
            worst = worst.max((px[c] as i32 - w).abs());
        }
    }
    eprintln!("served PNG against the diffusers bf16 weight image: worst {worst} levels of 255; usage {}", v["usage"]);
    assert!(worst <= 1, "{worst} levels");
    let (_, again) = post(&rig, body.clone());
    assert_eq!(again["data"][0]["b64_json"], v["data"][0]["b64_json"], "the same seed gives the same bytes");
    let (status, two) = post(&rig, json!({"model": "tiny-klein", "prompt": recorded_prompt("text"), "size": "32x32", "seed": 0, "steps": 3, "n": 2}));
    assert_eq!(status, 200, "{two}");
    let d = two["data"].as_array().unwrap();
    assert_eq!(d[0]["b64_json"], v["data"][0]["b64_json"], "image 0 of n uses the seed itself");
    assert_ne!(d[0]["b64_json"], d[1]["b64_json"]);
    let (status, wide) = post(&rig, json!({"model": "tiny-klein", "prompt": "w7 w9", "size": "48x32", "seed": 1, "steps": 2}));
    assert_eq!(status, 200, "{wide}");
    let w = rgb(wide["data"][0]["b64_json"].as_str().unwrap());
    assert_eq!((w.width(), w.height()), (48, 32));
    let (status, bad) = post(&rig, json!({"model": "tiny-klein", "prompt": "w7", "size": "30x32"}));
    assert_eq!(status, 400, "{bad}");
    let (status, guided) = post(&rig, json!({"model": "tiny-klein", "prompt": recorded_prompt("text"), "negative_prompt": recorded_prompt("negative"), "size": "32x32", "seed": 0, "steps": 3, "guidance": 4.0}));
    assert_eq!(status, 200, "{guided}");
    let (_, models, _) = rig.get("/v1/models", None);
    assert!(models.contains("\"type\":\"image\""), "{models}");
    let mut s = rig.raw_post("/v1/images/generations", &json!({"model": "tiny-klein", "prompt": "w7 w9", "size": "32x32", "steps": 3, "stream": true, "partial_images": 1}).to_string());
    let text = read_until(&mut s, "event: image_generation.completed");
    assert_eq!(text.matches("event: image_generation.progress").count(), 3, "{text}");
    assert_eq!(text.matches("event: image_generation.partial_image").count(), 3, "{text}");
    rig.stop();
}
