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
fn fp8_kv_step_decode_matches_single_sequence_greedy() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-hd256"] {
        let mut s = spec_for(dir);
        s.runtime.kv_dtype = specrt::spec::KvDtype::Fp8;
        stepped_spec_matches_single(&s);
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
#[ignore = "needs a CUDA GPU"]
fn plan_driven_decode_captures_a_graph_per_padded_size() {
    let s = spec_for("tiny-qwen3.5");
    let layout = Layout { max_ctx: 512, slots: 4, kv_blocks: 4 * 512usize.div_ceil(specrt::codegen::PAGE), prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    let mut st = rt.stepper().unwrap();
    st.set_graphs(true);
    for slot in 0..3 {
        rt.set_block_table(slot, &((slot * pages) as u32..((slot + 1) * pages) as u32).collect::<Vec<_>>()).unwrap();
        rt.prefill_slot(&mut st, slot, &[5, 6, 7, 8], 0, None).unwrap();
    }
    let row = |slot: usize| DecodeRow { slot, token: 9, pos: 4, sampling: Sampling::GREEDY, step: 0 };
    rt.decode(&mut st, &[row(0)]).unwrap();
    rt.decode(&mut st, &[row(0), row(1), row(2)]).unwrap();
    assert_eq!(st.captured_sizes(), vec![1, 4]);
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
        ids.push(sched.submit(Request { prompt: p.clone(), max_tokens: n, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap());
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
    let opts = ServeOptions { max_model_len: Some(512), max_num_seqs: 8, kv_cache_bytes: Some(4 << 20), ..ServeOptions::default() };
    let core = Core::start(&s, opts).unwrap();
    let mut rxs = Vec::new();
    for p in &prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        let request = Request { prompt: p.clone(), max_tokens: 30, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None };
        core.submit(Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }).unwrap();
        rxs.push(rx);
    }
    let (tx, long_rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let request = Request { prompt: prompts[0].clone(), max_tokens: 500, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None };
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
    assert_eq!(per_token, s.kv_layers() * s.attention.kv_heads * s.attention.head_dim * 2 * 2);
    let real = Spec::load(&fixture_dir("qwen3.8-27b-exact.toml")).unwrap();
    assert_eq!(specrt::runtime::kv_block_bytes(&real, 1) / specrt::codegen::PAGE, 68 * 1024);
    let mut plain = real.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    assert_eq!(specrt::runtime::kv_block_bytes(&plain, 1) / specrt::codegen::PAGE, 64 * 1024);
    let layout = Layout::serving(&real, 131072, 32, Some(7 << 30));
    assert_eq!((layout.kv_planes, layout.kv_blocks), (1, (7 << 30) / (68 * 1024 * 16)));
    assert_eq!(Layout::serving(&real, 131072, 32, None).kv_blocks, 0, "0 sizes the pool from free memory");
    let state = specrt::serve::slot_state_bytes(&real) as usize;
    let few = Layout::serving(&real, 131072, 8, None).kv_headroom;
    let many = Layout::serving(&real, 131072, 12, None).kv_headroom;
    assert_eq!(many - few, 4 * state, "auto-sized pool leaves recurrent state for every slot");
}

fn prefix_cached_scheduler_matches_single(dir: &str) {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    use specrt::spec::PrefixCacheOptions;
    let s = spec_for(dir);
    let shared: Vec<u32> = (0..160u32).map(|i| (i * 29 + 7) % 1024).collect();
    let prompts: Vec<Vec<u32>> = (0..24u32)
        .map(|q| [shared.clone(), (0..(5 + q * 11 % 90)).map(|i| (i * 13 + q * 97 + 1) % 1024).collect::<Vec<u32>>()].concat())
        .collect();
    let mut single = Runtime::load(&s, 512).unwrap();
    let want: Vec<Vec<u32>> = prompts.iter().map(|p| single.generate(p, 20, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let layout = Layout { max_ctx: 512, slots: 6, kv_blocks: 90, prefill_chunk: 256, kv_planes: 1, kv_headroom: 0, state_on_demand: true };
    let exec = GpuExecutor::new(Runtime::load_layout(&s, layout).unwrap()).unwrap();
    let prefix_cache = PrefixCacheOptions { enabled: true, snapshot_interval: 64, max_snapshots: 4, max_blocks: 0, async_snapshots: true };
    let cfg = Config { max_num_seqs: 6, max_batched_tokens: 300, reserve_blocks: Some(1), prefix_cache, ..Config::default() };
    let mut sched = Scheduler::new(exec, cfg);
    let ids: Vec<u64> = prompts
        .iter()
        .map(|p| sched.submit(Request { prompt: p.clone(), max_tokens: 20, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap())
        .collect();
    let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
    let mut cached = 0;
    while !sched.idle() {
        let mut events = Vec::new();
        sched.step(&mut events).unwrap();
        for e in events {
            match e {
                Event::Token { id, token } => got.entry(id).or_default().push(token),
                Event::Finished { cached_tokens, .. } => cached += cached_tokens,
                _ => {}
            }
        }
    }
    assert!(cached > 0 && sched.prefix_stats().hits > 0, "the shared prefix was never hit");
    assert_eq!(sched.blocks_used(), sched.prefix_stats().blocks);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(got[id], want[i], "sequence {i}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_prefix_cached_scheduling_matches_single_sequence_greedy() {
    prefix_cached_scheduler_matches_single("tiny-qwen3.5");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn serving_logprobs_match_a_log_softmax_of_the_full_prompt_logits() {
    use specrt::sched::Request;
    use specrt::serve::{Core, Job, SeqEvent, ServeOptions};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    let s = spec_for("tiny-qwen3.5");
    let prompts = mixed_prompts();
    let new = 12;
    let opts = ServeOptions { max_model_len: Some(512), max_num_seqs: 8, kv_cache_bytes: Some(4 << 20), ..ServeOptions::default() };
    let core = Core::start(&s, opts).unwrap();
    let mut rxs = Vec::new();
    for p in &prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        let request = Request { prompt: p.clone(), max_tokens: new, sampling: Sampling::GREEDY.with_logprobs(Some(3)), stop: vec![], salt: String::new(), media: Vec::new(), grammar: None };
        core.submit(Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }).unwrap();
        rxs.push(rx);
    }
    let got: Vec<(Vec<u32>, Vec<specrt::logprobs::TokenLogprob>)> = rxs
        .iter()
        .map(|rx| {
            let (mut tokens, mut lps) = (Vec::new(), Vec::new());
            loop {
                match rx.recv().unwrap() {
                    SeqEvent::Token(t) => tokens.push(t),
                    SeqEvent::Logprob(l) => lps.push(l),
                    SeqEvent::Finished { .. } => break (tokens, lps),
                    SeqEvent::Failed(e) => panic!("{e}"),
                    _ => {}
                }
            }
        })
        .collect();
    drop(core);
    let mut rt = Runtime::load(&s, 512).unwrap();
    let vocab = s.dims.vocab;
    for (p, (tokens, lps)) in prompts.iter().zip(&got) {
        assert_eq!(tokens.len(), new);
        assert_eq!(lps.iter().map(|l| l.token).collect::<Vec<_>>(), *tokens);
        let whole: Vec<u32> = p.iter().chain(tokens).copied().collect();
        let logits = rt.prompt_logits(&whole).unwrap();
        for (j, lp) in lps.iter().enumerate() {
            let row = &logits[(p.len() - 1 + j) * vocab..(p.len() + j) * vocab];
            let want = specrt::logprobs::token_logprob(row, lp.token, 3);
            assert!((lp.logprob - want.logprob).abs() < 2e-2, "prompt {} token {j}: {} vs {}", p.len(), lp.logprob, want.logprob);
            assert_eq!(lp.top.len(), 3);
            assert_eq!(lp.top[0].0, lp.token, "greedy picks the top token");
            assert!(lp.top.windows(2).all(|w| w[0].1 >= w[1].1));
        }
    }
}

#[test]
fn a_pool_smaller_than_one_max_model_len_is_refused() {
    use specrt::serve::{Core, ServeOptions};
    let s = spec_for("tiny-qwen3.5");
    let block = specrt::runtime::kv_block_bytes(&s, 1) as u64;
    let small = ServeOptions { max_model_len: Some(512), max_num_seqs: 4, kv_cache_bytes: Some(block * 31), ..ServeOptions::default() };
    let e = Core::start(&s, small).err().expect("refused").to_string();
    assert!(e.contains("KV cache holds 496 tokens") && e.contains("512"), "{e}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn four_requests_share_a_pool_of_one_and_a_half_max_model_lens_and_match_single_sequence() {
    use specrt::sched::{Finish, Request};
    use specrt::serve::{Core, Job, SeqEvent, ServeOptions};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    let s = spec_for("tiny-qwen3.5");
    let prompts: Vec<Vec<u32>> = (0..4u32).map(|q| (0..300u32).map(|i| (i * 7 + q * 131 + 3) % 1024).collect()).collect();
    let want = single_greedy(&s, &prompts, 150);
    let block = specrt::runtime::kv_block_bytes(&s, 1) as u64;
    let blocks = 512 * 3 / 2 / specrt::codegen::PAGE as u64;
    let opts = ServeOptions { max_model_len: Some(512), max_num_seqs: 4, kv_cache_bytes: Some(block * blocks), ..ServeOptions::default() };
    let core = Core::start(&s, opts).unwrap();
    let rxs: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (tx, rx) = std::sync::mpsc::channel();
            let request = Request { prompt: p.clone(), max_tokens: 150, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None };
            core.submit(Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }).unwrap();
            rx
        })
        .collect();
    for (rx, want) in rxs.iter().zip(&want) {
        let mut got = Vec::new();
        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(60)).unwrap() {
                SeqEvent::Token(t) => got.push(t),
                SeqEvent::Finished { finish, .. } => {
                    assert_eq!(finish, Finish::Length);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(&got, want);
    }
    assert!(core.stats().preemptions > 0, "four 350 token requests cannot grow in a 768 token pool without preemption");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_json_object_grammar_always_yields_an_object_and_leaves_other_rows_alone() {
    use specrt::grammar::{Grammar, TokenTable};
    use specrt::sched::{Config, Event, Finish, GpuExecutor, Request, Scheduler};
    let s = spec_for("tiny-qwen3.5");
    let vocab = s.dims.vocab;
    let mut tokens = vec![Vec::new(); vocab];
    for b in 0..=255u8 {
        tokens[b as usize + 1] = vec![b];
    }
    let grammar = Grammar::json_object(std::sync::Arc::new(TokenTable::new(tokens)), vocab, vec![0]);
    let layout = Layout { max_ctx: 1024, slots: 8, kv_blocks: 256, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let exec = GpuExecutor::new(Runtime::load_layout(&s, layout).unwrap()).unwrap();
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: 8, ..Config::default() });
    let mut checked = 0;
    for round in 0..8u32 {
        let mut ids = Vec::new();
        for i in 0..6u32 {
            let seed = round * 16 + i;
            let sampling = Sampling::new(1.0, seed);
            let prompt: Vec<u32> = (0..20u32).map(|j| (j * 37 + seed * 11) % 700 + 300).collect();
            let g = (i != 5).then(|| grammar.clone());
            ids.push((i, sched.submit(Request { prompt, max_tokens: 700, sampling, stop: vec![], salt: String::new(), media: Vec::new(), grammar: g }).unwrap()));
        }
        let mut out: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        let mut finish: std::collections::HashMap<u64, Finish> = Default::default();
        while !sched.idle() {
            let mut events = Vec::new();
            sched.step(&mut events).unwrap();
            for e in events {
                match e {
                    Event::Token { id, token } => out.entry(id).or_default().push(token),
                    Event::Finished { id, finish: f, .. } => {
                        finish.insert(id, f);
                    }
                    _ => {}
                }
            }
        }
        for (i, id) in ids {
            if i == 5 {
                assert_eq!(out[&id].len(), 700, "the unconstrained row is untouched");
                continue;
            }
            let bytes: Vec<u8> = out[&id].iter().map(|&t| (t - 1) as u8).collect();
            if finish[&id] == Finish::Stop {
                match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(v) => assert!(v.is_object()),
                    Err(e) => assert!(e.to_string().contains("number out of range"), "{e}: {}", String::from_utf8_lossy(&bytes)),
                }
                checked += 1;
            } else {
                assert!(out[&id].iter().all(|&t| (1..=256).contains(&t)), "only byte tokens are allowed");
            }
        }
    }
    assert!(checked >= 8, "only {checked} constrained requests finished");
}

fn schema_conforms(schema: &serde_json::Value, v: &serde_json::Value) -> bool {
    let Some(o) = v.as_object() else { return false };
    let name_ok = o.get("name").is_some_and(|x| x.is_string());
    let age_ok = o.get("age").is_some_and(|x| x.is_i64() || x.is_u64() || x.as_f64().is_some_and(|f| f.fract() == 0.0));
    let kind_ok = o.get("kind").is_some_and(|x| ["red", "green", "blue"].iter().any(|k| x == k));
    let items_ok = o.get("items").is_some_and(|x| {
        x.as_array().is_some_and(|a| {
            a.iter().all(|e| {
                e.as_object().is_some_and(|e| {
                    e.len() == 2 && e.get("id").is_some_and(|x| x.is_number()) && e.get("label").is_some_and(|x| x.is_string())
                })
            })
        })
    });
    let allowed = schema["properties"].as_object().unwrap();
    name_ok && age_ok && kind_ok && items_ok && o.keys().all(|k| allowed.contains_key(k))
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_json_schema_grammar_output_validates_over_many_seeds() {
    use specrt::grammar::{Schema, TokenTable};
    use specrt::sched::{Config, Event, Finish, GpuExecutor, Request, Scheduler};
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "kind": {"enum": ["red", "green", "blue"]},
            "items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "number"}, "label": {"type": "string"}}, "required": ["id", "label"]}}
        },
        "required": ["name", "age", "kind", "items"]
    });
    let s = spec_for("tiny-qwen3.5");
    let vocab = s.dims.vocab;
    let mut tokens = vec![Vec::new(); vocab];
    for b in 0..=255u8 {
        tokens[b as usize + 1] = vec![b];
    }
    let grammar = Schema::Json(schema.clone()).build(std::sync::Arc::new(TokenTable::new(tokens)), vocab, vec![0]);
    let layout = Layout { max_ctx: 1024, slots: 8, kv_blocks: 256, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let exec = GpuExecutor::new(Runtime::load_layout(&s, layout).unwrap()).unwrap();
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: 8, ..Config::default() });
    let mut checked = 0;
    for round in 0..12u32 {
        let mut ids = Vec::new();
        for i in 0..8u32 {
            let seed = round * 16 + i;
            let prompt: Vec<u32> = (0..20u32).map(|j| (j * 37 + seed * 11) % 700 + 300).collect();
            ids.push(sched.submit(Request { prompt, max_tokens: 900, sampling: Sampling::new(1.0, seed), stop: vec![], salt: String::new(), media: Vec::new(), grammar: Some(grammar.clone()) }).unwrap());
        }
        let mut out: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        let mut finish: std::collections::HashMap<u64, Finish> = Default::default();
        while !sched.idle() {
            let mut events = Vec::new();
            sched.step(&mut events).unwrap();
            for e in events {
                match e {
                    Event::Token { id, token } => out.entry(id).or_default().push(token),
                    Event::Finished { id, finish: f, .. } => {
                        finish.insert(id, f);
                    }
                    _ => {}
                }
            }
        }
        for id in ids {
            let bytes: Vec<u8> = out[&id].iter().map(|&t| (t - 1) as u8).collect();
            if finish[&id] != Finish::Stop {
                assert!(out[&id].iter().all(|&t| (1..=256).contains(&t)), "only byte tokens are allowed");
                continue;
            }
            match serde_json::from_slice::<serde_json::Value>(&bytes) {
                Ok(v) => assert!(schema_conforms(&schema, &v), "{v}"),
                Err(e) => assert!(e.to_string().contains("number out of range"), "{e}: {}", String::from_utf8_lossy(&bytes)),
            }
            checked += 1;
        }
    }
    assert!(checked >= 8, "only {checked} constrained requests finished");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn auto_sized_pool_admits_and_decodes_more_slots_than_the_old_resident_cap() {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let mut s = spec_for("tiny-qwen3.5-27b");
    s.mtp = None;
    let slots = 12;
    let layout = Layout::serving(&s, 512, slots, None);
    assert_eq!(layout.kv_blocks, 0);
    let exec = GpuExecutor::new(Runtime::load_layout(&s, layout).unwrap()).unwrap();
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: slots, ..Config::default() });
    for q in 0..slots as u32 {
        let prompt = (0..40u32).map(|i| (i * 13 + q * 97 + 1) % 1024).collect();
        sched.submit(Request { prompt, max_tokens: 24, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap();
    }
    let (mut most_running, mut most_rows, mut finished) = (0, 0, 0);
    while !sched.idle() {
        let mut events = Vec::new();
        let r = sched.step(&mut events).unwrap();
        most_running = most_running.max(sched.running());
        most_rows = most_rows.max(r.decode_rows);
        finished += events.iter().filter(|e| matches!(e, Event::Finished { .. })).count();
    }
    assert_eq!(most_running, slots, "only {most_running} of {slots} sequences were admitted at once");
    assert_eq!(most_rows, slots, "decode batches peaked at {most_rows} rows");
    assert_eq!(finished, slots);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_contiguous_layout_without_set_block_table_maps_every_page_to_its_own_block() {
    let s = spec_for("tiny-qwen3.5");
    let prompt: Vec<u32> = (0..70u32).map(|i| (i * 37 + 11) % 1024).collect();
    let mut loaded = Runtime::load(&s, 512).unwrap();
    let want = loaded.prompt_logits(&prompt).unwrap();
    drop(loaded);
    let mut bare = Runtime::load_layout(&s, Layout::contiguous(512, 1)).unwrap();
    let got = bare.prompt_logits(&prompt).unwrap();
    assert!(max_abs(&got, &want) <= 1e-5, "{}", max_abs(&got, &want));
}
