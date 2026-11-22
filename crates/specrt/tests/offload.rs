use specrt::offload::Tier;
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, vram_budget: u64) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.split_activations = Some(true);
    spec.runtime.offload.vram_budget = vram_budget;
    spec.runtime.offload.host_budget = 1 << 30;
    spec
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 7 + 3) % 1024).collect()
}

fn budget_keeping(dir: &str, resident: usize) -> (u64, usize) {
    let rt = Runtime::load(&spec_for(dir, u64::MAX / 4), 256).unwrap();
    let plan = rt.offload_plan().unwrap().clone();
    let n = plan.layers.len();
    let sizes: Vec<u64> = plan.layers.iter().map(|l| l.bytes).collect();
    let always = plan.vram_bytes - sizes.iter().sum::<u64>() - plan.staging_bytes;
    let mut order = vec![0, n - 1];
    order.extend(1..n - 1);
    let kept: u64 = order.iter().take(resident).map(|&l| sizes[l]).sum();
    (always + plan.staging_bytes + kept, n - resident)
}

fn assert_identical(dir: &str, resident: usize) {
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let p = prompt(40);
    let want_prefill = full.trace_prefill(&p).unwrap();
    let want_decode = full.trace_decode(&prompt(6)).unwrap();
    let want_tokens = full.generate(&p, 16, Sampling::GREEDY).unwrap().tokens;
    drop(full);
    let (budget, streamed) = budget_keeping(dir, resident);
    let mut off = Runtime::load(&spec_for(dir, budget), 256).unwrap();
    assert_eq!(off.streamed_layers(), streamed, "{dir} with {resident} resident");
    let plan = off.offload_plan().unwrap();
    assert_eq!(plan.layers.iter().filter(|l| l.tier == Tier::Vram).count(), resident);
    let got_prefill = off.trace_prefill(&p).unwrap();
    assert_eq!(got_prefill.logits, want_prefill.logits, "prefill logits");
    assert_eq!(got_prefill.layers, want_prefill.layers, "prefill layer outputs");
    let got_decode = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(got_decode.logits, want_decode.logits, "decode logits");
    assert_eq!(got_decode.layers, want_decode.layers, "decode layer outputs");
    assert_eq!(off.generate(&p, 16, Sampling::GREEDY).unwrap().tokens, want_tokens);
    let stats = off.offload_stats().unwrap();
    assert!(stats.host_hits > 0 && stats.host_to_vram_bytes > 0, "{stats:?}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn one_resident_layer_is_bit_identical_to_fully_resident() {
    assert_identical("tiny-qwen3.5", 1);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_resident_layers_are_bit_identical_to_fully_resident() {
    assert_identical("tiny-qwen3.5", 2);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_layers_stream_bit_identically() {
    assert_identical("tiny-qwen3.5-fp8", 1);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn slots_are_reused_when_more_layers_stream_than_buffers_exist() {
    assert_identical("tiny-qwen3.5-27b", 1);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_budget_that_holds_everything_streams_nothing() {
    let rt = Runtime::load(&spec_for("tiny-qwen3.5", u64::MAX / 4), 256).unwrap();
    assert_eq!(rt.streamed_layers(), 0);
    assert!(rt.offload_stats().is_none());
}
