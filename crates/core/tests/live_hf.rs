use inference_bench_core::hf::{HF_BASE, Hub, Progress};
use std::sync::atomic::Ordering;

#[test]
#[ignore = "talks to huggingface.co"]
fn live_search_and_a_resumed_download() {
    let hub = Hub::new(HF_BASE, "");
    let repos = hub.search("SmolLM2-135M", 5).unwrap();
    println!("search: {:?}", repos.iter().map(|r| &r.id).collect::<Vec<_>>());
    assert!(repos.iter().any(|r| r.id == "HuggingFaceTB/SmolLM2-135M"));
    let files = hub.files("HuggingFaceTB/SmolLM2-135M").unwrap();
    let tok = files.iter().find(|f| f.path == "tokenizer.json").unwrap();
    println!("tokenizer.json is {} bytes", tok.size);
    let dir = std::env::temp_dir().join(format!("ib-live-hf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = Progress::default();
    let stopper = p.clone();
    let watcher = std::thread::spawn(move || {
        while stopper.done.load(Ordering::Relaxed) < 300_000 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        stopper.cancel.store(true, Ordering::Relaxed);
    });
    let first = hub.download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json", &dir, &p);
    watcher.join().unwrap();
    assert_eq!(first.unwrap_err(), "paused");
    let part = dir.join("models--HuggingFaceTB--SmolLM2-135M/snapshots/main/tokenizer.json.part");
    let partial = std::fs::metadata(&part).unwrap().len();
    println!("paused with {partial} bytes in the .part file");
    assert!(partial > 0 && partial < tok.size);
    let again = Progress::default();
    let path = hub.download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json", &dir, &again).unwrap();
    let full = std::fs::read(&path).unwrap();
    println!(
        "resumed: total {} bytes, {} fetched after the pause",
        full.len(),
        again.done.load(Ordering::Relaxed) - partial
    );
    assert_eq!(full.len() as u64, tok.size);
    let snapshots = inference_bench_core::library::hub_dir().join("models--HuggingFaceTB--SmolLM2-135M/snapshots");
    let cached = snapshots
        .read_dir()
        .unwrap()
        .flatten()
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.exists())
        .map(|p| std::fs::read(p).unwrap());
    if let Some(cached) = cached {
        assert_eq!(full, cached, "resumed file differs from the cached copy");
        println!("matches the copy already in the hub cache byte for byte");
    }
}
