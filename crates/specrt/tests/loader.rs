use specrt::progress::{Progress, Sink, Stage, with_sink};
use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn spec(dir: &str, enabled: bool, direct: bool) -> Spec {
    spec_with(dir, enabled, direct, false)
}

fn spec_with(dir: &str, enabled: bool, direct: bool, pinned: bool) -> Spec {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir);
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    s.name = dir.into();
    s.repo = d.display().to_string();
    s.runtime.loader.enabled = enabled;
    s.runtime.loader.direct = direct;
    s.runtime.loader.chunk_mb = 1;
    s.runtime.loader.skip_cached = false;
    s.runtime.loader.pinned = pinned;
    s.runtime.loader.workers = 3;
    s
}

fn logits(s: &Spec) -> (Vec<f32>, Vec<Progress>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let mut rt = with_sink(Some(Sink::new(move |p| sink.lock().unwrap().push(p))), || Runtime::load(s, 64)).unwrap();
    let prompt: Vec<u32> = (0..24u32).map(|i| (i * 37 + 5) % 1000).collect();
    let out = rt.prompt_logits(&prompt).unwrap();
    let events = seen.lock().unwrap().clone();
    (out, events)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_fast_loader_uploads_the_same_weights_as_the_mmap_path() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-fp8"] {
        let (mmap, _) = logits(&spec(dir, false, false));
        for (direct, pinned) in [(true, false), (false, false), (true, true)] {
            let (fast, _) = logits(&spec_with(dir, true, direct, pinned));
            assert_eq!(mmap.len(), fast.len());
            assert!(mmap.iter().zip(&fast).all(|(a, b)| a.to_bits() == b.to_bits()), "{dir} direct={direct} pinned={pinned}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn load_progress_is_monotonic_and_the_weights_end_at_their_total() {
    let (_, events) = logits(&spec("tiny-qwen3.5", true, true));
    let weights: Vec<&Progress> = events.iter().filter(|e| e.stage == Stage::Weights).collect();
    assert!(weights.len() >= 3);
    assert_eq!(weights[0].done, 0);
    assert!(weights.windows(2).all(|w| w[0].done <= w[1].done && w[0].total == w[1].total));
    let last = weights.last().unwrap();
    assert!(last.total > 0);
    assert_eq!(last.done, last.total);
    let stages: Vec<Stage> = events.iter().map(|e| e.stage).fold(Vec::new(), |mut v, s| {
        if v.last() != Some(&s) {
            v.push(s);
        }
        v
    });
    assert_eq!(stages, vec![Stage::Kernels, Stage::Weights, Stage::KvCache]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_model_already_in_the_page_cache_loads_through_the_mmap_path_with_the_same_weights() {
    let mut cached = spec("tiny-qwen3.5", true, true);
    cached.runtime.loader.skip_cached = true;
    let ck = specrt::weights::Checkpoint::for_repo(&cached.repo).unwrap();
    for f in &ck.files {
        std::fs::read(f).unwrap();
    }
    assert!(ck.cached(&cached.runtime.loader));
    let (a, _) = logits(&cached);
    let (b, _) = logits(&spec("tiny-qwen3.5", false, false));
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_tight_host_budget_and_the_split_head_and_mtp_groups_upload_the_same_weights() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-mtp"] {
        let (mmap, _) = logits(&spec(dir, false, false));
        let mut tight = spec(dir, true, true);
        tight.runtime.loader.host_budget_mb = 64;
        tight.runtime.loader.workers = 4;
        let (fast, events) = logits(&tight);
        assert!(mmap.iter().zip(&fast).all(|(a, b)| a.to_bits() == b.to_bits()), "{dir}");
        let last = events.iter().rfind(|e| e.stage == Stage::Weights).unwrap();
        assert_eq!(last.done, last.total, "{dir}");
    }
}
