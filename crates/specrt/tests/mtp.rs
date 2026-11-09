use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, SpecRow};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"))
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(ref_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(ref_dir().join(name)).unwrap()).unwrap()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn spec(k: usize) -> Spec {
    let dir = root().join("tests/fixtures/tiny-qwen3.5-mtp");
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    s.name = "tiny-qwen3.5-mtp".into();
    s.repo = dir.display().to_string();
    s.mtp.as_mut().unwrap().draft_tokens = k;
    s
}

#[test]
fn tiny_mtp_fixture_derives_an_mtp_block() {
    let s = spec(2);
    let m = s.mtp.clone().unwrap();
    assert_eq!((m.layers, m.draft_tokens, m.speculate), (1, 2, true));
    assert_eq!(s.kv_layers(), s.attention_layers() + 1);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-mtp output"]
fn tiny_mtp_draft_logits_match_the_torch_reference() {
    let s = spec(2);
    let m = meta("tinymtp-prompt.json");
    let prompt = ids(&m["tokens"]);
    let mut rt = Runtime::load(&s, 512).unwrap();
    let (logits, drafts) = rt.mtp_prompt_logits(&prompt).unwrap();
    let want = floats("tinymtp-prompt.chain.bin");
    let err = max_abs(&logits, &want);
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    eprintln!("mtp chained draft logits max abs {err:e} (scale {scale})");
    assert!(err <= 1e-3 * scale.max(1.0), "{err}");
    assert_eq!(drafts[0] as u64, m["draft"].as_u64().unwrap());
    assert_eq!(drafts[1] as u64, m["chain_draft"].as_u64().unwrap());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn drafts_after_speculative_steps_equal_drafts_from_a_fresh_prefill_of_the_same_prefix() {
    let s = spec(3);
    let ps = prompts();
    let want = plain_greedy(&s, &ps[..4]);
    let mut rt = Runtime::load_layout(&s, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    let mut fresh = Runtime::load_layout(&s, layout(1)).unwrap();
    tables(&mut fresh, 1);
    for (p, w) in ps.iter().zip(&want) {
        let mut out = vec![rt.prefill_slot(&mut st, 0, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
        for step in 0..6 {
            if step % 2 == 1 {
                let k = 3;
                let have = out.len();
                let mut d: Vec<u32> = (0..k).map(|j| w[have + j]).collect();
                d[step % k] = (d[step % k] + 1) % 1024;
                rt.set_drafts(0, &d).unwrap();
            }
            let row = SpecRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32, max_accept: 3 };
            out.extend(rt.speculate(&[row]).unwrap().remove(0));
            let prefix: Vec<u32> = p.iter().chain(&out[..out.len() - 1]).copied().collect();
            let (_, again) = fresh.mtp_prompt_logits(&prefix).unwrap();
            assert_eq!(rt.drafts(0), again, "drafts after step {step} with {} tokens", out.len());
        }
    }
}

fn chi_square_of(counts: &[u64], probs: &[f64], total: u64) -> (f64, usize) {
    let mut chi = 0.0;
    let mut bins = 0usize;
    let (mut rest_obs, mut rest_exp) = (0.0, 0.0);
    for (c, p) in counts.iter().zip(probs) {
        let e = p * total as f64;
        if e >= 5.0 {
            chi += (*c as f64 - e).powi(2) / e;
            bins += 1;
        } else {
            rest_obs += *c as f64;
            rest_exp += e;
        }
    }
    if rest_exp > 0.0 {
        chi += (rest_obs - rest_exp).powi(2) / rest_exp.max(1e-9);
        bins += 1;
    }
    (chi, bins.saturating_sub(1))
}

fn softmax(row: &[f32], t: f32) -> Vec<f64> {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let e: Vec<f64> = row.iter().map(|&v| (((v - m) / t) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn argmax_row(row: &[f64]) -> u32 {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b }) as u32
}

fn target_after(plain: &Spec, prefix: &[u32], t: f32) -> Vec<f64> {
    let mut rt = Runtime::load(plain, 512).unwrap();
    let l = rt.prompt_logits(prefix).unwrap();
    let vocab = plain.dims.vocab;
    softmax(&l[(prefix.len() - 1) * vocab..prefix.len() * vocab], t)
}

fn check_distribution(name: &str, counts: &[u64], target: &[f64], draft: u32, accepted: u64) {
    let total: u64 = counts.iter().sum();
    let (chi, dof) = chi_square_of(counts, target, total);
    let z = (chi - dof as f64) / (2.0 * dof.max(1) as f64).sqrt();
    let rate = accepted as f64 / total as f64;
    let p = target[draft as usize];
    eprintln!("{name}: {total} samples, chi-square {chi:.1} on {dof} dof (z {z:.2}), accept rate {rate:.3} for p(draft) {p:.3}");
    assert!(dof >= 3, "{name}: too few bins to test");
    assert!(z < 4.0, "{name}: the emitted token does not follow the target distribution");
    assert!((rate - p).abs() < 4.0 * (p * (1.0 - p) / total as f64).sqrt() + 1e-3, "{name}: accept rate {rate}, want {p}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn rejection_sampling_follows_the_target_distribution_at_both_draft_positions() {
    let s = spec(2);
    let t = 0.3f32;
    let p: Vec<u32> = prompt(3);
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let first = argmax_row(&target_after(&plain, &p, 1.0));
    let ext: Vec<u32> = p.iter().copied().chain([first]).collect();
    let target0 = target_after(&plain, &ext, t);
    let mode0 = argmax_row(&target0);
    let ext2: Vec<u32> = ext.iter().copied().chain([mode0]).collect();
    let target1 = target_after(&plain, &ext2, t);
    let mode1 = argmax_row(&target1);
    let runs = 4000u64;
    let vocab = s.dims.vocab;
    let mut rt = Runtime::load_layout(&s, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    let low = (mode0 + 1) % vocab as u32;
    for (d0, name) in [(mode0, "position 0, the mode drafted"), (low, "position 0, another token drafted")] {
        let mut c0 = vec![0u64; vocab];
        let mut c1 = vec![0u64; vocab];
        let (mut a0, mut a1) = (0u64, 0u64);
        for seed in 0..runs {
            rt.prefill_slot(&mut st, 0, &p, 0, None).unwrap();
            rt.set_drafts(0, &[d0, mode1]).unwrap();
            let sampling = Sampling { temperature: t, seed: (seed as u32).wrapping_mul(2654435761).wrapping_add(17) };
            let row = SpecRow { slot: 0, token: first, pos: p.len(), sampling, step: 1, max_accept: 2 };
            let out = rt.speculate(&[row]).unwrap().remove(0);
            c0[out[0] as usize] += 1;
            a0 += u64::from(out.len() >= 2);
            if d0 == mode0 && out.len() >= 2 {
                c1[out[1] as usize] += 1;
                a1 += u64::from(out.len() == 3);
            }
        }
        check_distribution(name, &c0, &target0, d0, a0);
        if d0 == mode0 {
            check_distribution("position 1 after accepting position 0", &c1, &target1, mode1, a1);
        }
    }
}

const PROMPTS: usize = 20;
const NEW: usize = 128;

fn prompt(i: usize) -> Vec<u32> {
    let len = 5 + (i * 37) % 120;
    (0..len as u32).map(|j| (j * 131 + i as u32 * 977 + 7) % 1024).collect()
}

fn layout(slots: usize) -> Layout {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn tables(rt: &mut Runtime, slots: usize) {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
    for s in 0..slots {
        let blocks: Vec<u32> = (0..pages).map(|p| (s as u32 * pages + p * 7) % (slots as u32 * pages)).collect();
        rt.set_block_table(s, &blocks).unwrap();
    }
}

fn plain_greedy(s: &Spec, prompts: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let mut rt = Runtime::load_layout(&plain, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    prompts
        .iter()
        .map(|p| {
            let mut out = vec![rt.prefill_slot(&mut st, 0, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
            while out.len() < NEW {
                let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
                out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
            }
            out
        })
        .collect()
}

enum Drafts {
    Mtp,
    Oracle(usize),
    Wrong(usize),
    Random(u64),
}

fn speculative_greedy(s: &Spec, prompts: &[Vec<u32>], want: &[Vec<u32>], drafts: Drafts, batch: usize) -> (Vec<Vec<u32>>, f64) {
    let k = s.mtp.as_ref().unwrap().draft_tokens;
    let mut rt = Runtime::load_layout(s, layout(batch)).unwrap();
    tables(&mut rt, batch);
    let mut st = rt.stepper().unwrap();
    let mut outs = Vec::new();
    let mut rng = match drafts {
        Drafts::Random(seed) => seed,
        _ => 1,
    };
    for group in prompts.chunks(batch).zip(want.chunks(batch)) {
        let (ps, ws) = group;
        let mut out: Vec<Vec<u32>> = Vec::new();
        for (slot, p) in ps.iter().enumerate() {
            out.push(vec![rt.prefill_slot(&mut st, slot, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()]);
        }
        loop {
            let live: Vec<usize> = (0..ps.len()).filter(|&i| out[i].len() < NEW).collect();
            if live.is_empty() {
                break;
            }
            for &i in &live {
                let have = out[i].len();
                let forced: Option<Vec<u32>> = match drafts {
                    Drafts::Mtp => None,
                    Drafts::Oracle(wrong_at) | Drafts::Wrong(wrong_at) => {
                        let mut d: Vec<u32> = (0..k).map(|j| ws[i].get(have + j).copied().unwrap_or(0)).collect();
                        let at = if let Drafts::Wrong(_) = drafts { (have + i) % k } else { wrong_at };
                        if at < k {
                            d[at] = (d[at] + 1) % 1024;
                        }
                        Some(d)
                    }
                    Drafts::Random(_) => Some(
                        (0..k)
                            .map(|_| {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                (rng % 1024) as u32
                            })
                            .collect(),
                    ),
                };
                if let Some(d) = forced {
                    rt.set_drafts(i, &d).unwrap();
                }
            }
            let rows: Vec<SpecRow> = live
                .iter()
                .map(|&i| SpecRow {
                    slot: i,
                    token: *out[i].last().unwrap(),
                    pos: ps[i].len() + out[i].len() - 1,
                    sampling: Sampling::GREEDY,
                    step: out[i].len() as u32,
                    max_accept: NEW - out[i].len() - 1,
                })
                .collect();
            let emitted = rt.speculate(&rows).unwrap();
            for (&i, e) in live.iter().zip(emitted) {
                out[i].extend(e);
            }
        }
        for o in &mut out {
            o.truncate(NEW);
        }
        outs.extend(out);
    }
    let st = rt.spec_stats();
    let mean = 1.0 + st.accepted as f64 / st.rows.max(1) as f64;
    (outs, if st.drafted == 0 { 0.0 } else { mean })
}

fn check(name: &str, got: &[Vec<u32>], want: &[Vec<u32>]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let first = g.iter().zip(w).position(|(a, b)| a != b);
        assert_eq!(g, w, "{name}: prompt {i} differs first at token {first:?}");
    }
}

fn prompts() -> Vec<Vec<u32>> {
    (0..PROMPTS).map(prompt).collect()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn speculative_greedy_is_identical_to_plain_greedy_for_128_tokens_over_20_prompts() {
    for k in 1..=3 {
        let s = spec(k);
        let ps = prompts();
        let want = plain_greedy(&s, &ps);
        let (got, mean) = speculative_greedy(&s, &ps, &want, Drafts::Mtp, 1);
        eprintln!("k = {k}: mean acceptance length {mean:.3} with the MTP drafts");
        check(&format!("MTP drafts, k = {k}"), &got, &want);
        let (got, mean) = speculative_greedy(&s, &ps, &want, Drafts::Oracle(k), 1);
        assert!(mean > k as f64 + 0.8, "oracle drafts accept almost everything: {mean}");
        check(&format!("oracle drafts, k = {k}"), &got, &want);
        for at in 0..k {
            let (got, _) = speculative_greedy(&s, &ps, &want, Drafts::Oracle(at), 1);
            check(&format!("draft {at} forced wrong, k = {k}"), &got, &want);
        }
        let (got, _) = speculative_greedy(&s, &ps, &want, Drafts::Wrong(0), 1);
        check(&format!("rejection position rotating, k = {k}"), &got, &want);
        let (got, mean) = speculative_greedy(&s, &ps, &want, Drafts::Random(k as u64 * 77 + 1), 1);
        assert!(mean < 1.1, "random drafts are almost always rejected: {mean}");
        check(&format!("random drafts, k = {k}"), &got, &want);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn batched_speculation_with_different_accept_counts_matches_plain_greedy() {
    let s = spec(2);
    let ps = prompts();
    let want = plain_greedy(&s, &ps);
    for batch in [4, 5] {
        let (got, mean) = speculative_greedy(&s, &ps, &want, Drafts::Mtp, batch);
        eprintln!("batch {batch}: mean acceptance length {mean:.3}");
        check(&format!("batch {batch}"), &got, &want);
        let (got, _) = speculative_greedy(&s, &ps, &want, Drafts::Wrong(0), batch);
        check(&format!("batch {batch}, staggered rejections"), &got, &want);
    }
}
