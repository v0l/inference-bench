use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, SpecRow};
use specrt::spec::Spec;
use std::path::Path;

fn spec(k: usize) -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-mtp");
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    s.name = "tiny-qwen3.5-mtp".into();
    s.repo = dir.display().to_string();
    s.mtp.as_mut().unwrap().draft_tokens = k;
    s
}

fn plain(k: usize) -> Spec {
    let mut s = spec(k);
    s.mtp.as_mut().unwrap().speculate = false;
    s
}

fn layout(slots: usize) -> Layout {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn tables(rt: &mut Runtime, slots: usize) {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
    for s in 0..slots {
        let blocks: Vec<u32> = (0..pages).map(|p| s as u32 * pages + p).collect();
        rt.set_block_table(s, &blocks).unwrap();
    }
}

fn prompt(i: usize) -> Vec<u32> {
    let len = 5 + (i * 37) % 120;
    (0..len as u32).map(|j| (j * 131 + i as u32 * 977 + 7) % 1024).collect()
}

fn truncated(row: &[f32], t: f32, top_k: u32, top_p: f32, min_p: f32) -> Vec<f64> {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let e: Vec<f64> = row.iter().map(|&v| (((v - m) / t) as f64).exp()).collect();
    let z: f64 = e.iter().sum();
    let p: Vec<f64> = e.iter().map(|v| v / z).collect();
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    let mut keep = vec![false; p.len()];
    let limit = if top_k == 0 { p.len() } else { top_k as usize };
    let kept_mass: f64 = order.iter().take(limit).map(|&i| p[i]).sum();
    let mut cum = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        if rank >= limit || (top_p < 1.0 && cum >= top_p as f64 * kept_mass) || p[i] < min_p as f64 * p[order[0]] {
            break;
        }
        keep[i] = true;
        cum += p[i];
    }
    let total: f64 = (0..p.len()).filter(|&i| keep[i]).map(|i| p[i]).sum();
    (0..p.len()).map(|i| if keep[i] { p[i] / total } else { 0.0 }).collect()
}

fn chi_square(counts: &[u64], probs: &[f64]) -> (f64, usize) {
    let total: u64 = counts.iter().sum();
    assert!(counts.iter().zip(probs).all(|(&c, &p)| p > 0.0 || c == 0), "a token outside the kept set was sampled");
    let (mut chi, mut bins, mut rest_o, mut rest_e) = (0.0, 0usize, 0.0, 0.0);
    for (&c, &p) in counts.iter().zip(probs) {
        let e = p * total as f64;
        if e >= 5.0 {
            chi += (c as f64 - e).powi(2) / e;
            bins += 1;
        } else {
            rest_o += c as f64;
            rest_e += e;
        }
    }
    if rest_e > 0.0 {
        chi += (rest_o - rest_e).powi(2) / rest_e;
        bins += 1;
    }
    (chi, bins.saturating_sub(1))
}

fn assert_follows(name: &str, counts: &[u64], probs: &[f64]) {
    let (chi, dof) = chi_square(counts, probs);
    let z = (chi - dof as f64) / (2.0 * dof.max(1) as f64).sqrt();
    let kept = probs.iter().filter(|&&p| p > 0.0).count();
    eprintln!("{name}: kept {kept}, chi-square {chi:.1} on {dof} dof (z {z:.2})");
    assert!(dof >= 3, "{name}: too few bins");
    assert!(z < 4.0, "{name}: samples do not follow the truncated distribution");
}

fn argmax(row: &[f64]) -> u32 {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) as u32
}

fn last_logits(rt: &mut Runtime, vocab: usize, prefix: &[u32]) -> Vec<f32> {
    let l = rt.prompt_logits(prefix).unwrap();
    l[(prefix.len() - 1) * vocab..prefix.len() * vocab].to_vec()
}

const CASES: [(u32, f32, f32); 5] = [(20, 1.0, 0.0), (0, 0.5, 0.0), (0, 0.95, 0.0), (20, 0.95, 0.0), (0, 1.0, 0.1)];
const RUNS: u64 = 6000;

fn seed_of(i: u64) -> u32 {
    (i as u32).wrapping_mul(2654435761).wrapping_add(17)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn plain_sampling_follows_the_truncated_distribution() {
    let s = plain(2);
    let vocab = s.dims.vocab;
    let p = prompt(3);
    let mut rt = Runtime::load_layout(&s, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    let logits = last_logits(&mut Runtime::load(&s, 512).unwrap(), vocab, &p);
    let t = 2.0f32;
    for (top_k, top_p, min_p) in CASES {
        let target = truncated(&logits, t, top_k, top_p, min_p);
        let mut counts = vec![0u64; vocab];
        for i in 0..RUNS {
            let sampling = Sampling::new(t, seed_of(i)).truncated(top_k, top_p, min_p);
            let tok = rt.prefill_slot(&mut st, 0, &p, 0, Some((sampling, 0))).unwrap().unwrap();
            counts[tok as usize] += 1;
        }
        assert_follows(&format!("plain top_k {top_k} top_p {top_p} min_p {min_p}"), &counts, &target);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn top_k_one_equals_greedy_and_batched_rows_equal_solo_rows() {
    let s = plain(2);
    let p = prompt(5);
    let mut rt = Runtime::load_layout(&s, layout(4)).unwrap();
    tables(&mut rt, 4);
    let mut st = rt.stepper().unwrap();
    let greedy = rt.prefill_slot(&mut st, 0, &p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
    for i in 0..50 {
        let one = Sampling::new(1.5, seed_of(i)).truncated(1, 1.0, 0.0);
        assert_eq!(rt.prefill_slot(&mut st, 0, &p, 0, Some((one, 0))).unwrap().unwrap(), greedy);
    }
    let params = [
        Sampling::new(1.0, 11).truncated(20, 0.95, 0.0),
        Sampling::new(1.5, 12).truncated(1, 1.0, 0.0),
        Sampling::new(1.0, 13).truncated(0, 0.5, 0.0),
        Sampling::new(2.0, 14).truncated(5, 1.0, 0.05),
    ];
    let mut firsts = Vec::new();
    for (slot, sampling) in params.iter().enumerate() {
        firsts.push(rt.prefill_slot(&mut st, slot, &p, 0, Some((*sampling, 0))).unwrap().unwrap());
    }
    let mut rows: Vec<DecodeRow> = params
        .iter()
        .enumerate()
        .map(|(slot, &sampling)| DecodeRow { slot, token: firsts[slot], pos: p.len(), sampling, step: 1 })
        .collect();
    let mut batched = vec![Vec::new(); 4];
    for _ in 0..24 {
        let out = rt.decode(&mut st, &rows).unwrap();
        for (i, r) in rows.iter_mut().enumerate() {
            batched[i].push(out[i]);
            r.token = out[i];
            r.pos += 1;
            r.step += 1;
        }
    }
    for (slot, sampling) in params.iter().enumerate() {
        let mut row = DecodeRow { slot: 0, token: firsts[slot], pos: p.len(), sampling: *sampling, step: 1 };
        rt.prefill_slot(&mut st, 0, &p, 0, None).unwrap();
        let mut solo = Vec::new();
        for _ in 0..24 {
            let t = rt.decode(&mut st, &[row]).unwrap()[0];
            solo.push(t);
            row.token = t;
            row.pos += 1;
            row.step += 1;
        }
        assert_eq!(solo, batched[slot], "row {slot} differs from its solo run");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn speculative_sampling_follows_the_truncated_distribution() {
    let s = spec(2);
    let vocab = s.dims.vocab;
    let p = prompt(3);
    let plain_spec = plain(2);
    let t = 2.0f32;
    let mut reference = Runtime::load(&plain_spec, 512).unwrap();
    let first = argmax(&truncated(&last_logits(&mut reference, vocab, &p), 1.0, 0, 1.0, 0.0));
    let ext: Vec<u32> = p.iter().copied().chain([first]).collect();
    let logits0 = last_logits(&mut reference, vocab, &ext);
    drop(reference);
    let mut rt = Runtime::load_layout(&s, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    for (top_k, top_p, min_p) in CASES {
        let target = truncated(&logits0, t, top_k, top_p, min_p);
        let mode = argmax(&target);
        let outside = (0..vocab as u32).find(|&i| target[i as usize] == 0.0).unwrap();
        for draft in [mode, outside] {
            let mut counts = vec![0u64; vocab];
            let mut accepted = 0u64;
            for i in 0..RUNS {
                rt.prefill_slot(&mut st, 0, &p, 0, Some((Sampling::GREEDY, 0))).unwrap();
                rt.set_drafts(0, &[draft, mode]).unwrap();
                let sampling = Sampling::new(t, seed_of(i)).truncated(top_k, top_p, min_p);
                let row = SpecRow { slot: 0, token: first, pos: p.len(), sampling, step: 1, max_accept: 2 };
                let out = rt.speculate(&[row]).unwrap().remove(0);
                counts[out[0] as usize] += 1;
                accepted += u64::from(out.len() >= 2);
            }
            assert_follows(&format!("speculative top_k {top_k} top_p {top_p} min_p {min_p} draft {draft}"), &counts, &target);
            let want = target[draft as usize];
            let rate = accepted as f64 / RUNS as f64;
            assert!((rate - want).abs() < 4.0 * (want * (1.0 - want) / RUNS as f64).sqrt() + 1e-3, "accept {rate} want {want}");
        }
    }
}

fn kept_set(logits: &[f32], t: f32, top_k: u32, top_p: f32, min_p: f32) -> Vec<bool> {
    truncated(logits, t, top_k, top_p, min_p).iter().map(|&p| p > 0.0).collect()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn single_sequence_generation_applies_top_k_top_p_and_min_p() {
    let s = plain(2);
    let vocab = s.dims.vocab;
    let p = prompt(3);
    let logits = last_logits(&mut Runtime::load(&s, 512).unwrap(), vocab, &p);
    let mut rt = Runtime::load(&s, 512).unwrap();
    let greedy = rt.generate(&p, 64, Sampling::GREEDY).unwrap().tokens;
    for seed in 0..4 {
        let one = Sampling::new(1.0, seed_of(seed)).truncated(1, 1.0, 0.0);
        assert_eq!(rt.generate(&p, 64, one).unwrap().tokens, greedy, "top_k 1 at temperature 1.0, seed {seed}");
    }
    for (name, top_k, top_p, min_p) in [("top_k 5", 5, 1.0, 0.0), ("top_p 0.5", 0, 0.5, 0.0), ("min_p 0.2", 0, 1.0, 0.2)] {
        let kept = kept_set(&logits, 1.5, top_k, top_p, min_p);
        assert!(kept.iter().filter(|&&k| k).count() < vocab / 2, "{name}: the case must cut the vocabulary");
        let mut distinct = std::collections::HashSet::new();
        for i in 0..300 {
            let sampling = Sampling::new(1.5, seed_of(i)).truncated(top_k, top_p, min_p);
            let tok = rt.generate(&p, 1, sampling).unwrap().tokens[0];
            assert!(kept[tok as usize], "{name}: token {tok} is outside the kept set (seed {i})");
            distinct.insert(tok);
        }
        assert!(distinct.len() > 1, "{name}: sampling collapsed to one token");
    }
}

fn scheduled_tokens(s: &Spec, sampling: Sampling, new: usize) -> Vec<u32> {
    use specrt::sched::Request;
    use specrt::serve::{Core, Job, SeqEvent, ServeOptions};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    let opts = ServeOptions { max_model_len: 512, max_num_seqs: 2, kv_cache_bytes: Some(4 << 20), ..ServeOptions::default() };
    let core = Core::start(s, opts).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let request = Request { prompt: prompt(5), max_tokens: new, sampling, stop: vec![], salt: String::new(), media: Vec::new() };
    core.submit(Job { request, events: tx, cancel: Arc::new(AtomicBool::new(false)) }).unwrap();
    let mut out = Vec::new();
    loop {
        match rx.recv().unwrap() {
            SeqEvent::Token(t) => out.push(t),
            SeqEvent::Finished { .. } => break,
            SeqEvent::Failed(e) => panic!("{e}"),
            _ => {}
        }
    }
    out
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn scheduled_top_k_one_equals_greedy_with_and_without_speculation() {
    for (name, s) in [("mtp", spec(2)), ("plain", plain(2))] {
        let greedy = scheduled_tokens(&s, Sampling::GREEDY, 64);
        assert_eq!(greedy.len(), 64);
        for seed in 0..3 {
            let one = Sampling::new(1.0, seed_of(seed)).truncated(1, 1.0, 0.0);
            assert_eq!(scheduled_tokens(&s, one, 64), greedy, "{name}: top_k 1 at temperature 1.0, seed {seed}");
        }
    }
}
