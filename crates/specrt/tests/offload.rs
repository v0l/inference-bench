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

fn moe_spec(dir: &str, slots: usize, hotlist: Option<String>) -> Spec {
    let mut spec = spec_for(dir, u64::MAX / 4);
    spec.runtime.offload.expert_slots = slots;
    spec.runtime.offload.hotlist = hotlist;
    spec
}

fn assert_experts_identical(dir: &str, slots: usize) {
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let p = prompt(40);
    let want_prefill = full.trace_prefill(&p).unwrap();
    let want_decode = full.trace_decode(&prompt(6)).unwrap();
    let want_tokens = full.generate(&p, 16, Sampling::GREEDY).unwrap().tokens;
    drop(full);
    let mut off = Runtime::load(&moe_spec(dir, slots, None), 256).unwrap();
    assert_eq!(off.expert_slots(), slots);
    assert_eq!(off.trace_prefill(&p).unwrap().logits, want_prefill.logits, "prefill logits");
    let got = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(got.logits, want_decode.logits, "decode logits");
    assert_eq!(got.layers, want_decode.layers, "decode layer outputs");
    assert_eq!(off.generate(&p, 16, Sampling::GREEDY).unwrap().tokens, want_tokens);
    let stats = off.offload_stats().unwrap();
    assert!(stats.host_to_vram_bytes > 0, "{stats:?}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_cache_with_every_expert_slot_is_bit_identical() {
    assert_experts_identical("tiny-qwen3.5-moe", 8);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_cache_with_few_slots_is_bit_identical() {
    assert_experts_identical("tiny-qwen3.5-moe", 2);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_cache_with_fp8_experts_is_bit_identical() {
    assert_experts_identical("tiny-qwen3.5-moe-fp8", 3);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_cache_with_nvfp4_experts_is_bit_identical() {
    assert_experts_identical("tiny-qwen3.5-moe-nvfp4", 2);
    assert_experts_identical("tiny-qwen3.5-moe-nvfp4", 8);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn calibration_hotlist_round_trips_and_warms_the_cache() {
    let first = prompt(1);
    let mut cold = Runtime::load(&moe_spec("tiny-qwen3.5-moe", 6, None), 256).unwrap();
    assert_eq!(cold.offload_stats().unwrap().host_hits, 0);
    let want = cold.trace_decode(&first).unwrap().logits;
    let cold_misses = cold.offload_stats().unwrap().host_hits;
    let path = std::env::temp_dir().join(format!("hotlist-{}.json", std::process::id()));
    cold.save_expert_hotlist(&path).unwrap();
    drop(cold);
    let mut warm = Runtime::load(&moe_spec("tiny-qwen3.5-moe", 6, Some(path.display().to_string())), 256).unwrap();
    assert!(warm.offload_stats().unwrap().host_to_vram_bytes > 0, "hotlist warm up copies experts before the first token");
    assert_eq!(warm.trace_decode(&first).unwrap().logits, want);
    let warm_misses = warm.offload_stats().unwrap().host_hits;
    assert!(cold_misses >= 6, "{cold_misses}");
    assert_eq!(warm_misses, 0, "first token misses: cold {cold_misses}, warm {warm_misses}");
    std::fs::remove_file(path).unwrap();
}
