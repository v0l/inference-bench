mod common;

use common::*;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/tiny-wan21")
}

fn reference(key: &str) -> Vec<f32> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("reference/tiny-wan21.json")).unwrap()).unwrap();
    let file = meta["files"][key]["file"].as_str().unwrap();
    std::fs::read(fixture().join("reference").join(file)).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn recorded_prompt(key: &str) -> String {
    let ids = reference(&format!("{key}.ids"));
    let mask = reference(&format!("{key}.mask"));
    ids.iter().zip(&mask).filter(|(_, m)| **m == 1.0).map(|(i, _)| format!("w{}", *i as u32)).collect::<Vec<_>>().join(" ")
}

fn wait_done(rig: &Rig, id: &str) -> (Value, Vec<u32>) {
    let start = Instant::now();
    let mut seen = Vec::new();
    loop {
        let (status, text, _) = rig.get(&format!("/v1/videos/{id}"), None);
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        seen.push(v["progress"].as_u64().unwrap() as u32);
        if matches!(v["status"].as_str(), Some("completed" | "failed" | "cancelled")) {
            return (v, seen);
        }
        assert!(start.elapsed() < Duration::from_secs(120), "{v}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
#[ignore = "needs a GPU: ~/.agents/bin/gpu-run 1 -- cargo test -p inference-bench-server --test videos_gpu -- --ignored --test-threads=1"]
fn the_tiny_wan_pipeline_is_served_as_an_async_video_job_with_an_mp4() {
    let dir = fixture().display().to_string();
    let rig = start_with(config(&format!("[[models]]\nid='tiny-wan'\nmodel={dir:?}\n")), inference_bench_server::NativeFactory, Default::default());
    let body = json!({"model": "tiny-wan", "prompt": recorded_prompt("text"), "negative_prompt": recorded_prompt("negative"), "size": "32x32", "seconds": "0.3", "fps": 16, "seed": 1, "steps": 3, "guidance": 5.0});
    let (status, text, _) = rig.post("/v1/videos", None, body);
    assert_eq!(status, 200, "{text}");
    let created: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(created["object"], "video");
    assert!(matches!(created["status"].as_str(), Some("queued" | "in_progress")), "{created}");
    let id = created["id"].as_str().unwrap().to_string();
    let (done, seen) = wait_done(&rig, &id);
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!((done["progress"].as_u64(), done["frames"].as_u64(), done["fps"].as_u64(), done["size"].as_str()), (Some(100), Some(5), Some(16), Some("32x32")), "{done}");
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "progress only rises: {seen:?}");
    let (status, mp4, headers) = rig.get_bytes(&format!("/v1/videos/{id}/content"));
    assert_eq!(status, 200);
    assert!(headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-type") && v == "video/mp4"), "{headers:?}");
    let h = inference_bench_core::specrt_mp4::header(&mp4).unwrap();
    eprintln!("served MP4: {} bytes, {h:?}, fps {}, usage {}", mp4.len(), h.fps(), done["usage"]);
    assert_eq!((h.width, h.height, h.frames, h.codec.as_str()), (32, 32, 5, "avc1"));
    assert!((h.fps() - 16.0).abs() < 1e-9);
    if std::process::Command::new("ffprobe").arg("-version").output().is_ok() {
        let file = std::env::temp_dir().join(format!("{id}.mp4"));
        std::fs::write(&file, &mp4).unwrap();
        let out = std::process::Command::new("ffprobe").args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames", "-of", "csv=p=0"]).arg(&file).output().unwrap();
        let raw = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&file).args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output().unwrap();
        let _ = std::fs::remove_file(&file);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "h264,32,32,16/1,5");
        let want = reference("bf16.video");
        let plane = 32 * 32;
        let mut sq = 0f64;
        for f in 0..5 {
            for p in 0..plane {
                for c in 0..3 {
                    let w = (want[(f * 3 + c) * plane + p].clamp(0.0, 1.0) * 255.0).round() as f64;
                    let g = raw.stdout[(f * plane + p) * 3 + c] as f64;
                    sq += (w - g).powi(2);
                }
            }
        }
        let db = 10.0 * (255.0f64.powi(2) / (sq / (5 * plane * 3) as f64).max(1e-12)).log10();
        eprintln!("ffmpeg decode of the served MP4 against the diffusers bf16 weight video: PSNR {db:.1} dB");
        assert!(db > 30.0, "{db} dB through H.264");
    }
    let (status, text, _) = rig.post("/v1/videos", None, json!({"model": "tiny-wan", "prompt": "w7", "size": "32x32", "frames": 4}));
    assert_eq!(status, 200, "{text}");
    let bad: Value = serde_json::from_str(&text).unwrap();
    let (v, _) = wait_done(&rig, bad["id"].as_str().unwrap());
    assert_eq!(v["status"], "failed", "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("nearest valid count is 5"), "{v}");
    let (status, text, _) = rig.post("/v1/videos", None, json!({"model": "tiny-wan", "prompt": "w7", "size": "32x32", "frames": 5, "steps": 200}));
    assert_eq!(status, 200, "{text}");
    let long: Value = serde_json::from_str(&text).unwrap();
    let lid = long["id"].as_str().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let (status, text, _) = rig.delete(&format!("/v1/videos/{lid}"));
    assert_eq!(status, 200, "{text}");
    let (status, _, _) = rig.get(&format!("/v1/videos/{lid}"), None);
    assert_eq!(status, 404);
    let (status, text, _) = rig.get(&format!("/v1/videos/{id}"), None);
    assert_eq!(status, 200, "{text}");
    let (_, models, _) = rig.get("/v1/models", None);
    assert!(models.contains("\"type\":\"video\""), "{models}");
    rig.stop();
}
