mod common;

use common::*;
use inference_bench_core::duplex_client::{self, Options};
use inference_bench_core::specrt_duplex::prompt::Prompt;
use inference_bench_core::specrt_duplex::wav;
use inference_bench_core::specrt_duplex::{Engine, EngineOptions, RecordedNoise, StartOptions};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/haru/tiny")
}

fn reference(key: &str) -> Vec<f32> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("reference.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir().join("reference").join(meta["files"][key]["file"].as_str().unwrap())).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn prompt() -> Prompt {
    Prompt {
        tokens: reference("prompt.tokens").iter().map(|&t| t as u32).collect(),
        model_latents: reference("prompt.model_latents"),
        user_latents: reference("prompt.user_latents"),
        model_mask: reference("prompt.model_mask"),
        user_mask: reference("prompt.user_mask"),
    }
}

fn noise() -> RecordedNoise {
    RecordedNoise { head: reference("session.head_noise"), user: reference("session.user_noise"), user_scale: reference("session.user_scale") }
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

#[test]
#[ignore = "needs a CUDA GPU: ~/.agents/bin/gpu-run 1 --min-free 2000 -- <duplex_gpu binary> --ignored"]
fn the_tiny_fixture_served_through_the_websocket_replays_the_fixture_session() {
    let user = wav::pcm16_round(&reference("session.user_audio"));
    let frames: Vec<Vec<f32>> = user.chunks(3200).map(<[f32]>::to_vec).collect();

    let mut e = Engine::load(&dir(), EngineOptions { slots: 1, max_ctx: 256, ..EngineOptions::default() }).unwrap_or_else(|err| panic!("{err:#}"));
    let (id, first) = e.start_prompt(&prompt(), StartOptions { greedy: true, noise: Some(noise()), ..StartOptions::default() }).unwrap();
    let mut direct = vec![first.audio.clone()];
    let mut direct_tokens = vec![first.token];
    for f in &frames {
        let o = e.step(&[(id, f.clone())]).unwrap().remove(0);
        direct.push(o.audio);
        direct_tokens.push(o.token);
    }
    drop(e);

    let rig = start_with(config(&format!("[duplex]\nallow_replay = true\nsessions_per_gpu = 1\nmax_ctx = 256\n[[models]]\nid='tiny-haru'\nmodel={:?}\n", dir().display().to_string())), inference_bench_server::NativeFactory, Default::default());
    let start = json!({"type": "start", "model": "tiny-haru", "replay": {"prompt": prompt(), "noise": noise(), "greedy": true}});
    let o = Options { url: duplex_client::ws_url(&rig.url, "/v1/duplex"), key: None, start, realtime: false, tail_frames: 0, paced: false };
    let run = duplex_client::stream(&o, &user).unwrap_or_else(|err| panic!("{err:#}"));
    assert_eq!(run.ready["frame_bytes"], 6400);
    assert_eq!(run.frames.len(), 7, "the prompt frame and one reply per user frame");
    let mut worst = 0f32;
    for (i, (got, want)) in run.frames.iter().zip(&direct).enumerate() {
        let r = max_rel(got, &wav::pcm16_round(want));
        eprintln!("frame {i}: max rel {r:.2e} against the engine fed the same PCM16 frames");
        worst = worst.max(r);
    }
    let want_tokens: Vec<u32> = reference("session.token").iter().map(|&t| t as u32).collect();
    eprintln!("tokens direct {direct_tokens:?} fixture {want_tokens:?} text {:?}", run.text);
    assert!(worst < 1e-4, "websocket frames against the engine, {worst}");
    assert_eq!(direct_tokens, want_tokens, "the PCM16 rounded session gives the fixture's greedy tokens");
    let fixture_audio: Vec<f32> = reference("session.audio").iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let served: Vec<f32> = run.frames.concat();
    eprintln!("served audio against the fixture's float session: max rel {:.2e}", max_rel(&served, &wav::pcm16_round(&fixture_audio)));
    assert!(rig.metrics().contains("inference_duplex_frames_out_total 7"));
}
