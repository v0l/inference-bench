mod common;

use common::*;
use serde_json::{Value, json};
use std::path::PathBuf;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

#[test]
#[ignore = "needs a CUDA GPU, PHONON_DIR (the f2 model folder with voices.jsonl), F2_SENTENCES and F2_ENGINE (the #1484 run)"]
fn real_phonon_through_the_route_gives_the_f2_engine_audio_byte_for_byte() {
    let dir = env_path("PHONON_DIR");
    let count: usize = std::env::var("SPEECH_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let rig = start_with(config(&format!("[[models]]\nid = 'phonon'\nmodel = {:?}\npreload = true\n", dir.display().to_string())), |max_ctx: Option<usize>| -> Result<Box<dyn inference_bench_core::target::Target>, inference_bench_core::result::BackendError> {
        Ok(Box::new(inference_bench_core::native::Native::new(max_ctx)))
    }, Default::default());
    wait_for_load(&rig);
    let engine: Vec<Value> = std::fs::read_to_string(env_path("F2_ENGINE").join("engine.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let rows: Vec<Value> = std::fs::read_to_string(env_path("F2_SENTENCES")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let picked: Vec<&Value> = ["Harper", "Hadley"].iter().flat_map(|v| rows.iter().filter(move |r| r["voice_name"] == *v).take(count.div_ceil(2))).collect();
    let mut equal = 0;
    for row in &picked {
        let id = row["id"].as_str().unwrap();
        let f2 = engine.iter().find(|e| e["id"] == id).unwrap();
        let want = std::fs::read(f2["wav"].as_str().unwrap()).unwrap();
        let started = std::time::Instant::now();
        let mut resp = agent()
            .post(format!("{}/v1/audio/speech", rig.url))
            .header("content-type", "application/json")
            .header("accept-encoding", "identity")
            .send(json!({"model": "kokoro", "input": row["text"], "voice": row["voice_name"].as_str().unwrap().to_lowercase(), "response_format": "pcm", "seed": 1484}).to_string())
            .unwrap();
        let first = started.elapsed().as_secs_f64();
        assert_eq!(resp.status().as_u16(), 200);
        let got = resp.body_mut().with_config().limit(64 << 20).read_to_vec().unwrap();
        let total = started.elapsed().as_secs_f64();
        let same = got[..] == want[44..];
        println!("{id}: {} samples, f2 {} samples, headers after {first:.2}s, done in {total:.2}s, byte equal {same}", got.len() / 2, (want.len() - 44) / 2);
        equal += usize::from(same);
        let wav = agent().post(format!("{}/v1/audio/speech", rig.url)).header("content-type", "application/json").send(json!({"model": "kokoro", "input": row["text"], "voice": row["voice_name"], "seed": 1484}).to_string()).unwrap().body_mut().with_config().limit(64 << 20).read_to_vec().unwrap();
        assert_eq!(wav, want, "{id}: the WAV answer equals the f2 file");
    }
    assert_eq!(equal, picked.len(), "streamed PCM equals the f2 WAV data for every clip");
    let paragraph: String = rows.iter().filter(|r| r["voice_name"] == "Harper").take(8).map(|r| r["text"].as_str().unwrap()).collect::<Vec<_>>().join(" ");
    let started = std::time::Instant::now();
    let resp = agent().post(format!("{}/v1/audio/speech", rig.url)).header("content-type", "application/json").header("accept-encoding", "identity").send(json!({"model": "kokoro", "input": paragraph, "voice": "Harper", "response_format": "pcm"}).to_string()).unwrap();
    let mut body = resp.into_body().into_reader();
    let mut frame = vec![0u8; 2 * 1920];
    std::io::Read::read_exact(&mut body, &mut frame).unwrap();
    let first = started.elapsed().as_secs_f64();
    let mut rest = Vec::new();
    std::io::Read::read_to_end(&mut body, &mut rest).unwrap();
    let audio = (rest.len() + frame.len()) as f64 / 2.0 / 24_000.0;
    let total = started.elapsed().as_secs_f64();
    println!("{} characters in 8 sentences: first frame after {first:.2}s, {audio:.1}s of audio in {total:.1}s", paragraph.len());
    assert!(first < total / 4.0, "the first frame comes well before the paragraph is done");
    let status: Value = serde_json::from_str(&rig.get("/v1/audio/status", None).1).unwrap();
    assert_eq!(status["speaking_on"], "GPU");
    rig.stop();
}

fn wait_for_load(rig: &Rig) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    while std::time::Instant::now() < deadline {
        if rig.get("/health", None).1.contains("\"phonon\"") {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    panic!("phonon did not load");
}
