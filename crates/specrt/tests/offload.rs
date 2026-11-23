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
fn layers_on_nvme_stream_bit_identically() {
    let dir = "tiny-qwen3.5";
    let (budget, _) = budget_keeping(dir, 1);
    let tmp = std::env::temp_dir().join(format!("specrt-layers-nvme-{}", std::process::id()));
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let p = prompt(40);
    let want_prefill = full.trace_prefill(&p).unwrap();
    let want_tokens = full.generate(&p, 16, Sampling::GREEDY).unwrap().tokens;
    drop(full);
    let mut spec = spec_for(dir, budget);
    spec.runtime.offload.host_budget = 1;
    spec.runtime.offload.nvme_path = Some(tmp.to_string_lossy().into_owned());
    let mut off = Runtime::load(&spec, 256).unwrap();
    assert!(off.offload_plan().unwrap().nvme_bytes > 0);
    assert_eq!(off.trace_prefill(&p).unwrap().logits, want_prefill.logits);
    assert_eq!(off.generate(&p, 16, Sampling::GREEDY).unwrap().tokens, want_tokens);
    let stats = off.offload_stats().unwrap();
    assert!(stats.nvme_read_bytes > 0 && stats.host_to_vram_bytes > 0, "{stats:?}");
    drop(off);
    assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(&tmp);
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
    assert_experts_identical_with(dir, slots, false);
}

fn assert_experts_identical_with(dir: &str, slots: usize, pointer_table: bool) -> u64 {
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let p = prompt(40);
    let want_prefill = full.trace_prefill(&p).unwrap();
    let want_decode = full.trace_decode(&prompt(6)).unwrap();
    let want_tokens = full.generate(&p, 16, Sampling::GREEDY).unwrap().tokens;
    drop(full);
    let mut spec = moe_spec(dir, slots, None);
    spec.runtime.offload.pointer_table = pointer_table;
    let mut off = Runtime::load(&spec, 256).unwrap();
    assert_eq!(off.expert_slots(), slots);
    let got_prefill = off.trace_prefill(&p).unwrap();
    if pointer_table {
        let worst = got_prefill.logits.iter().zip(&want_prefill.logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "prefill logits differ by {worst}");
    } else {
        assert_eq!(got_prefill.logits, want_prefill.logits, "prefill logits");
    }
    let got = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(got.logits, want_decode.logits, "decode logits");
    assert_eq!(got.layers, want_decode.layers, "decode layer outputs");
    assert_eq!(off.generate(&p, 16, Sampling::GREEDY).unwrap().tokens, want_tokens);
    let stats = off.offload_stats().unwrap();
    assert!(stats.host_to_vram_bytes > 0, "{stats:?}");
    off.expert_table_launches()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_cache_with_every_expert_slot_is_bit_identical() {
    assert_experts_identical("tiny-qwen3.5-moe", 8);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pointer_table_with_every_expert_resident_is_bit_identical() {
    assert!(assert_experts_identical_with("tiny-qwen3.5-moe", 24, true) > 0);
    assert!(assert_experts_identical_with("tiny-qwen3.5-moe-nvfp4", 24, true) > 0);
    assert!(assert_experts_identical_with("tiny-qwen3.5-moe-fp8", 24, true) > 0);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pointer_table_with_half_the_experts_resident_falls_back_bit_identically() {
    assert_experts_identical_with("tiny-qwen3.5-moe", 12, true);
    assert_experts_identical_with("tiny-qwen3.5-moe-nvfp4", 6, true);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pointer_table_on_glm_is_bit_identical() {
    assert_experts_identical_with("tiny-glm4-moe", 24, true);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn evicting_then_promoting_an_expert_switches_paths_and_stays_bit_identical() {
    let dir = "tiny-qwen3.5-moe";
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let want = full.trace_decode(&prompt(6)).unwrap();
    drop(full);
    let mut spec = moe_spec(dir, 24, None);
    spec.runtime.offload.pointer_table = true;
    let mut off = Runtime::load(&spec, 256).unwrap();
    let first = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(first.logits, want.logits);
    let on_table = off.expert_table_launches();
    assert!(on_table > 0);
    for layer in 0..3 {
        for e in 0..8 {
            assert!(off.evict_expert(layer, e));
        }
    }
    let second = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(second.logits, want.logits, "after evicting every expert");
    assert_eq!(second.layers, want.layers);
    assert!(off.expert_table_launches() < on_table * 2, "a layer with a missing expert must not use the table");
    for layer in 0..3 {
        for e in 0..8 {
            off.promote_expert(layer, e).unwrap();
        }
    }
    let third = off.trace_decode(&prompt(6)).unwrap();
    assert_eq!(third.logits, want.logits, "after promoting every expert");
    assert_eq!(third.layers, want.layers);
    assert!(off.expert_table_launches() > on_table, "promoted layers must use the table again");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pointer_table_matches_fully_resident_decode_on_gpt_oss() {
    let dir = "tiny-gpt-oss";
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let want_decode = full.trace_decode(&prompt(6)).unwrap();
    drop(full);
    let mut spec = moe_spec(dir, 32, None);
    spec.runtime.offload.pointer_table = true;
    let mut table = Runtime::load(&spec, 256).unwrap();
    let got = table.trace_decode(&prompt(6)).unwrap();
    assert_eq!(got.logits, want_decode.logits);
    assert_eq!(got.layers, want_decode.layers);
    assert!(table.expert_table_launches() > 0);
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
fn expert_cache_with_glm4_moe_experts_is_bit_identical() {
    assert_experts_identical("tiny-glm4-moe", 3);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn expert_pages_on_nvme_are_bit_identical() {
    let dir = "tiny-qwen3.5-moe-nvfp4";
    let mut full = Runtime::load(&spec_for(dir, 0), 256).unwrap();
    let p = prompt(40);
    let want = full.generate(&p, 16, Sampling::GREEDY).unwrap().tokens;
    drop(full);
    let tmp = std::env::temp_dir().join(format!("specrt-nvme-{}", std::process::id()));
    let mut spec = moe_spec(dir, 2, None);
    spec.runtime.offload.host_budget = 1;
    spec.runtime.offload.nvme_path = Some(tmp.to_string_lossy().into_owned());
    let mut off = Runtime::load(&spec, 256).unwrap();
    assert_eq!(off.generate(&p, 16, Sampling::GREEDY).unwrap().tokens, want);
    let stats = off.offload_stats().unwrap();
    assert!(stats.nvme_read_bytes > 0, "{stats:?}");
    drop(off);
    assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(&tmp);
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
