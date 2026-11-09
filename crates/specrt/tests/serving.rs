use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec
}

fn shuffled(n: u32, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n).collect();
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    for i in (1..v.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.swap(i, (x % (i as u64 + 1)) as usize);
    }
    v
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn shuffled_blocks_match_contiguous(dir: &str, kv_planes: usize) {
    let s = spec_for(dir);
    let max_ctx: usize = 512;
    let pages = max_ctx.div_ceil(specrt::codegen::PAGE) as u32;
    let prompt: Vec<u32> = (0..300u32).map(|i| (i * 37 + 11) % 1024).collect();
    let contiguous = Layout { kv_planes, prefill_chunk: 128, ..Layout::contiguous(max_ctx, 1) };
    let mut plain = Runtime::load_layout(&s, contiguous).unwrap();
    plain.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
    let logits = plain.prompt_logits(&prompt).unwrap();
    let greedy = plain.generate(&prompt, 40, Sampling::GREEDY).unwrap().tokens;
    drop(plain);
    let layout = Layout { max_ctx, slots: 3, kv_blocks: 3 * pages as usize, prefill_chunk: 128, kv_planes, kv_headroom: 0, state_on_demand: false };
    let mut paged = Runtime::load_layout(&s, layout).unwrap();
    let pool = shuffled(3 * pages, 7);
    paged.set_block_table(0, &pool[pages as usize..2 * pages as usize]).unwrap();
    let got = paged.prompt_logits(&prompt).unwrap();
    assert!(max_abs(&got, &logits) <= 1e-5, "{}", max_abs(&got, &logits));
    assert_eq!(paged.generate(&prompt, 40, Sampling::GREEDY).unwrap().tokens, greedy);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_shuffled_kv_blocks_match_the_contiguous_layout() {
    shuffled_blocks_match_contiguous("tiny-qwen3.5", 1);
    shuffled_blocks_match_contiguous("tiny-qwen3.5", 2);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_shuffled_kv_blocks_match_the_contiguous_layout() {
    shuffled_blocks_match_contiguous("tiny-qwen3.5-hd256", 1);
    shuffled_blocks_match_contiguous("tiny-qwen3.5-hd256", 2);
}

fn single_greedy(s: &Spec, prompts: &[Vec<u32>], new: usize) -> Vec<Vec<u32>> {
    let mut rt = Runtime::load(s, 512).unwrap();
    prompts.iter().map(|p| rt.generate(p, new, Sampling::GREEDY).unwrap().tokens).collect()
}

fn stepped_greedy(s: &Spec, prompts: &[Vec<u32>], new: usize, chunk: usize) -> Vec<Vec<u32>> {
    let slots = prompts.len() + 1;
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    let layout = Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(s, layout).unwrap();
    let pool = shuffled((slots * pages) as u32, 3);
    let mut st = rt.stepper().unwrap();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for (i, p) in prompts.iter().enumerate() {
        let slot = slots - 1 - i;
        rt.set_block_table(slot, &pool[i * pages..(i + 1) * pages]).unwrap();
        let mut start = 0;
        while start < p.len() {
            let end = (start + chunk).min(p.len());
            let last = end == p.len();
            let t = rt.prefill_slot(&mut st, slot, &p[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap();
            if let Some(t) = t {
                out[i].push(t);
            }
            start = end;
        }
    }
    for _ in 1..new {
        let rows: Vec<DecodeRow> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| DecodeRow {
                slot: slots - 1 - i,
                token: *out[i].last().unwrap(),
                pos: p.len() + out[i].len() - 1,
                sampling: Sampling::GREEDY,
                step: out[i].len() as u32,
            })
            .collect();
        for (i, t) in rt.decode(&mut st, &rows).unwrap().into_iter().enumerate() {
            out[i].push(t);
        }
    }
    out
}

fn mixed_prompts() -> Vec<Vec<u32>> {
    [5usize, 40, 129, 300]
        .iter()
        .enumerate()
        .map(|(q, &n)| (0..n as u32).map(|i| (i * 7 + q as u32 * 131 + 3) % 1024).collect())
        .collect()
}

fn stepped_matches_single(dir: &str) {
    stepped_spec_matches_single(&spec_for(dir));
}

fn stepped_spec_matches_single(s: &Spec) {
    let prompts = mixed_prompts();
    let expected = single_greedy(s, &prompts, 24);
    assert_eq!(stepped_greedy(s, &prompts, 24, 37), expected);
    assert_eq!(stepped_greedy(s, &prompts, 24, 128), expected);
}

fn head_quant_spec(dir: &str, head: specrt::spec::Quant) -> Spec {
    let mut s = spec_for(dir);
    s.head_quant = Some(head);
    s
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn quantised_head_step_decode_and_scheduler_match_single_sequence_greedy() {
    use specrt::spec::Quant;
    for (dir, head) in [("tiny-qwen3.5", Quant::Fp8Channel), ("tiny-qwen3.5", Quant::Nvfp4G16), ("tiny-qwen3.5-nvfp4", Quant::Nvfp4G16)] {
        let s = head_quant_spec(dir, head);
        stepped_spec_matches_single(&s);
        scheduled_spec_matches_single(&s);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_mixed_length_step_decode_matches_single_sequence_greedy() {
    stepped_matches_single("tiny-qwen3.5");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn rows_padded_to_a_captured_graph_size_match_single_sequence_greedy() {
    let s = spec_for("tiny-qwen3.5-hd256");
    let prompts: Vec<Vec<u32>> = mixed_prompts().into_iter().take(3).collect();
    let expected = single_greedy(&s, &prompts, 12);
    assert_eq!(stepped_greedy(&s, &prompts, 12, 64), expected);
}

#[test]
fn decode_batches_pad_to_the_next_graph_size() {
    let padded: Vec<usize> = [1, 2, 3, 5, 8, 9, 17, 32].iter().map(|&n| specrt::runtime::pad_rows(n)).collect();
    assert_eq!(padded, [1, 2, 4, 8, 8, 16, 32, 32]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_mixed_length_step_decode_matches_single_sequence_greedy() {
    stepped_matches_single("tiny-qwen3.5-hd256");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_mixed_length_step_decode_matches_single_sequence_greedy() {
    stepped_matches_single("tiny-qwen3.5-fp8");
}

fn scheduled_matches_single(dir: &str) {
    scheduled_spec_matches_single(&spec_for(dir));
}

fn scheduled_spec_matches_single(s: &Spec) {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let s = s.clone();
    let lens = [3usize, 17, 64, 65, 130, 250, 400];
    let prompts: Vec<Vec<u32>> = (0..64u32)
        .map(|q| (0..lens[q as usize % lens.len()] as u32).map(|i| (i * 13 + q * 97 + 1) % 1024).collect())
        .collect();
    let new: Vec<usize> = (0..64).map(|q| 8 + (q * 7) % 40).collect();
    let mut single = Runtime::load(&s, 512).unwrap();
    let want: Vec<Vec<u32>> =
        prompts.iter().zip(&new).map(|(p, &n)| single.generate(p, n, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let layout = Layout { max_ctx: 512, slots: 16, kv_blocks: 120, prefill_chunk: 256, kv_planes: 1, kv_headroom: 0, state_on_demand: true };
    let exec = GpuExecutor::new(Runtime::load_layout(&s, layout).unwrap()).unwrap();
    let cfg = Config { max_num_seqs: 16, max_batched_tokens: 300, reserve_blocks: Some(1), ..Config::default() };
    let mut sched = Scheduler::new(exec, cfg);
    let mut ids = Vec::new();
    for (p, &n) in prompts.iter().zip(&new) {
        ids.push(sched.submit(Request { prompt: p.clone(), max_tokens: n, sampling: Sampling::GREEDY, stop: vec![] }).unwrap());
    }
    let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
    let mut most = 0;
    while !sched.idle() {
        let mut events = Vec::new();
        let r = sched.step(&mut events).unwrap();
        most = most.max(r.decode_rows);
        for e in events {
            if let Event::Token { id, token } = e {
                got.entry(id).or_default().push(token);
            }
        }
    }
    assert!(sched.preemptions() > 0, "the pool of 120 blocks was meant to run out");
    assert!(most > 8, "decode batches stayed at {most} rows");
    assert_eq!(sched.blocks_used(), 0);
    assert_eq!(sched.exec.rt.state_live_slots(), 1);
    assert_eq!(sched.exec.rt.state_mapped_bytes() > 0, true);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(got[id], want[i], "sequence {i}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_sixty_four_scheduled_sequences_match_single_sequence_greedy_with_preemption() {
    scheduled_matches_single("tiny-qwen3.5");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_sixty_four_scheduled_sequences_match_single_sequence_greedy_with_preemption() {
    scheduled_matches_single("tiny-qwen3.5-hd256");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_serving_core_streams_concurrent_jobs_and_cancels_within_a_step() {
    use specrt::sched::{Finish, Request};
    use specrt::serve::{Core, Job, SeqEvent, ServeOptions};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let s = spec_for("tiny-qwen3.5");
    let prompts = mixed_prompts();
    let want = single_greedy(&s, &prompts, 30);
    let opts = ServeOptions { max_model_len: 512, max_num_seqs: 8, kv_cache_bytes: Some(4 << 20), ..ServeOptions::default() };
    let core = Core::start(&s, opts).unwrap();
    let mut rxs = Vec::new();
    for p in &prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        let request = Request { prompt: p.clone(), max_tokens: 30, sampling: Sampling::GREEDY, stop: vec![] };
        core.submit(Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }).unwrap();
        rxs.push(rx);
    }
    let (tx, long_rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let request = Request { prompt: prompts[0].clone(), max_tokens: 500, sampling: Sampling::GREEDY, stop: vec![] };
    core.submit(Job { request, events: tx, cancel: cancel.clone() }).unwrap();
    let mut seen = 0;
    let generated = loop {
        match long_rx.recv().unwrap() {
            SeqEvent::Token(_) => {
                seen += 1;
                if seen == 5 {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            SeqEvent::Finished { finish, completion_tokens, .. } => {
                assert_eq!(finish, Finish::Cancelled);
                break completion_tokens;
            }
            _ => {}
        }
    };
    assert!(generated < 100, "{generated} of 500 tokens were generated before the cancel took");
    for (rx, want) in rxs.iter().zip(&want) {
        let mut got = Vec::new();
        loop {
            match rx.recv().unwrap() {
                SeqEvent::Token(t) => got.push(t),
                SeqEvent::Finished { finish, completion_tokens, .. } => {
                    assert_eq!((finish, completion_tokens), (Finish::Length, 30));
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(&got, want);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while core.stats().kv_blocks_used > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let st = core.stats();
    assert_eq!((st.kv_blocks_used, st.running, st.waiting), (0, 0, 0));
    assert_eq!(st.kv_blocks_total, (4 << 20) / specrt::runtime::kv_block_bytes(&s, 1));
    assert!(st.inter_token_secs.total > 100 && st.decode_batch.total > 0);
}

#[test]
fn serving_kv_is_one_f16_copy_per_token() {
    let s = spec_for("tiny-qwen3.5-hd256");
    let per_token = specrt::runtime::kv_block_bytes(&s, 1) / specrt::codegen::PAGE;
    assert_eq!(per_token, s.attention_layers() * s.attention.kv_heads * s.attention.head_dim * 2 * 2);
    let real = Spec::load(&fixture_dir("qwen3.8-27b.toml")).unwrap();
    assert_eq!(specrt::runtime::kv_block_bytes(&real, 1) / specrt::codegen::PAGE, 64 * 1024);
    let layout = Layout::serving(&real, 131072, 32, Some(7 << 30));
    assert_eq!((layout.kv_planes, layout.kv_blocks), (1, (7 << 30) / (64 * 1024 * 16)));
    assert_eq!(Layout::serving(&real, 131072, 32, None).kv_blocks, 0, "0 sizes the pool from free memory");
}
