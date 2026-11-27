mod common;

use common::*;
use serde_json::Value;
use std::path::PathBuf;

fn hub_snapshot(repo: &str) -> Option<PathBuf> {
    let snaps = dirs_home().join(".cache/huggingface/hub").join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).find(|p| p.join("model.safetensors").exists())
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

fn lines(path: &str) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")).lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn multipart(wav: &[u8]) -> (String, Vec<u8>) {
    let boundary = "----agent-voice-speech";
    let mut body = Vec::new();
    for (k, v) in [("model", "whisper-1"), ("response_format", "json")] {
        body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend(wav);
    body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[test]
#[ignore = "needs a CUDA GPU, openai/whisper-base.en in the hub cache and the #1480 clip set (SPEECH_CLIPS, SPEECH_ENGINE)"]
fn real_whisper_through_the_route_gives_the_engine_transcripts() {
    let dir = hub_snapshot(&std::env::var("SPEECH_REPO").unwrap_or_else(|_| "openai/whisper-base.en".into())).expect("the model is in the hub cache");
    let clips = std::env::var("SPEECH_CLIPS").unwrap_or_else(|_| "/mnt/data/tmp/speech-1480/clips/manifest.jsonl".into());
    let engine = std::env::var("SPEECH_ENGINE").unwrap_or_else(|_| "/mnt/data/tmp/speech-1480/engine-whisper-base.en.jsonl".into());
    let count: usize = std::env::var("SPEECH_COUNT").ok().and_then(|n| n.parse().ok()).unwrap_or(8);
    let text = inference_bench_core::specrt_speech::service::Text::open(&dir).unwrap();
    let eot = serde_json::from_str::<Value>(&std::fs::read_to_string(dir.join("generation_config.json")).unwrap()).unwrap()["eos_token_id"].as_u64().unwrap() as u32;
    let want: std::collections::BTreeMap<String, Vec<u32>> = lines(&engine).into_iter().map(|r| (r["id"].as_str().unwrap().to_string(), r["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect())).collect();
    let rig = start_with(config(&format!("[[models]]\nid = 'whisper-base.en'\nmodel = {:?}\n", dir.display().to_string())), inference_bench_server::NativeFactory, Default::default());
    let models: Value = serde_json::from_str(&rig.get("/v1/models", None).1).unwrap();
    assert_eq!(models["data"][0]["type"], "transcription", "from the folder on disk before it loads");
    let mut same = 0;
    let rows: Vec<Value> = lines(&clips).into_iter().filter(|r| r["parts"].as_array().is_some_and(|p| p.len() == 1)).take(count).chain(lines(&clips).into_iter().filter(|r| r["parts"].as_array().is_some_and(|p| p.len() > 1)).take(5)).collect();
    for row in &rows {
        let id = row["id"].as_str().unwrap();
        let wav = std::fs::read(row["wav"].as_str().unwrap()).unwrap();
        let (ct, body) = multipart(&wav);
        let started = std::time::Instant::now();
        let mut resp = agent().post(format!("{}/v1/audio/transcriptions", rig.url)).header("Content-Type", &ct).send(&body[..]).unwrap();
        let code = resp.status().as_u16();
        let body_text = resp.body_mut().read_to_string().unwrap();
        assert_eq!(code, 200, "{id}: {body_text}");
        let got = serde_json::from_str::<Value>(&body_text).unwrap()["text"].as_str().unwrap().to_string();
        let expected = text.decode(&want[id], eot).unwrap().trim().to_string();
        println!("{id} {:.1}s audio in {:.2}s: {got}", row["seconds"].as_f64().unwrap(), started.elapsed().as_secs_f64());
        same += usize::from(got == expected);
        assert_eq!(got, expected, "{id}");
    }
    println!("{same}/{} transcripts through /v1/audio/transcriptions equal the #1480 engine run", rows.len());
    let status: Value = serde_json::from_str(&rig.get("/v1/audio/status", None).1).unwrap();
    assert_eq!(status["reading_on"], "GPU", "{status}");
    rig.stop();
}
