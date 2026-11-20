use specrt::family::Registry as Families;
use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::PathBuf;

const NAME: &str = "granite-4.0-h-tiny";

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("GRANITE_H_TINY_DIR").expect("GRANITE_H_TINY_DIR names the ibm-granite/granite-4.0-h-tiny snapshot"))
}

fn spec() -> Spec {
    let dir = checkpoint();
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = dir.display().to_string();
    spec
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR from scripts/granite_h_tiny_ref.py (float32)"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{NAME}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{NAME}.ref.bin"))).unwrap();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[test]
#[ignore = "needs a CUDA GPU with 20 GB free, GRANITE_H_TINY_DIR and SPECRT_REF_DIR from scripts/granite_h_tiny_ref.py (float32)"]
fn granite_h_tiny_matches_transformers() {
    let (tokens, greedy, vocab, want) = reference();
    let spec = spec();
    let mut rt = Runtime::load_batched(&spec, 512, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let agree = (0..tokens.len()).filter(|r| top(&ours[r * vocab..(r + 1) * vocab]) == top(&want[r * vocab..(r + 1) * vocab])).count();
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{NAME} prefill max_abs {max_abs} argmax agree {agree}/{}", tokens.len());
    assert!(agree * 100 >= tokens.len() * 93, "argmax agreement {agree}/{}", tokens.len());
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "decode greedy against the fp32 reference");
}

#[test]
#[ignore = "needs a CUDA GPU with 20 GB free and GRANITE_H_TINY_DIR"]
fn granite_h_tiny_resumes_from_a_checkpointed_prefix_bit_identically() {
    use specrt::codegen::PAGE;
    use specrt::prefix::{PrefixCache, block_hashes, namespace};
    use specrt::runtime::{DecodeRow, Layout, Sampling};
    use specrt::sched::BlockPool;
    use specrt::spec::PrefixCacheOptions;
    use std::sync::Arc;
    let s = spec();
    let max_ctx: usize = 512;
    let vocab = s.dims.vocab as u32;
    let token = |i: u32, salt: u32| (i * 37 + salt * 11 + 5) % vocab;
    let a: Vec<u32> = (0..3 * PAGE as u32).map(|i| token(i, 0)).collect();
    let b: Vec<u32> = (0..37).map(|i| token(i, 4)).collect();
    let whole = [a.clone(), b.clone()].concat();
    let pages = max_ctx.div_ceil(PAGE);
    let layout = Layout { max_ctx, slots: 3, kv_blocks: 3 * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut pool = BlockPool::new(3 * pages);
    let mut cache = PrefixCache::new(&PrefixCacheOptions { enabled: true, snapshot_interval: 2048, max_snapshots: 4, max_blocks: 0, async_snapshots: true }, true);
    let root = namespace(&s, 1, "");
    let greedy = Some((Sampling::GREEDY, 0));
    let fill = |pool: &mut BlockPool| -> Vec<u32> { (0..pages).map(|_| pool.alloc().unwrap()).collect() };

    let table0 = fill(&mut pool);
    rt.set_block_table(0, &table0).unwrap();
    rt.prefill_slot(&mut st, 0, &a, 0, None).unwrap();
    let first0 = rt.prefill_slot(&mut st, 0, &b, a.len(), greedy).unwrap().unwrap();
    let logits0 = rt.last_logits().unwrap();

    let table1 = fill(&mut pool);
    rt.set_block_table(1, &table1).unwrap();
    rt.prefill_slot(&mut st, 1, &a, 0, None).unwrap();
    let hashes = block_hashes(&root, &a);
    cache.register(&hashes, &table1[..hashes.len()], &mut pool);
    assert!(cache.store_snapshot(*hashes.last().unwrap(), Arc::new(rt.snapshot_state(1).unwrap()), &mut pool));
    let mut released = table1.clone();
    pool.release(&mut released);

    let hit = cache.lookup(&block_hashes(&root, &whole), whole.len() - 1, &mut pool);
    assert_eq!(hit.tokens, a.len());
    let mut table2 = hit.blocks.clone();
    table2.extend(fill(&mut pool).into_iter().take(pages - table2.len()));
    rt.set_block_table(2, &table2).unwrap();
    rt.restore_state(2, &hit.snapshot.unwrap()).unwrap();
    let first2 = rt.prefill_slot(&mut st, 2, &whole[hit.tokens..], hit.tokens, greedy).unwrap().unwrap();
    let logits2 = rt.last_logits().unwrap();
    assert_eq!(first0, first2);
    assert!(logits0.iter().zip(&logits2).all(|(x, y)| x.to_bits() == y.to_bits()), "logits are not bit identical");
    let (mut t0, mut t2) = (first0, first2);
    for i in 0..8 {
        let row = |slot, token| DecodeRow { slot, token, pos: whole.len() + i, sampling: Sampling::GREEDY, step: i as u32 + 1 };
        let out = rt.decode(&mut st, &[row(0, t0), row(2, t2)]).unwrap();
        assert_eq!(out[0], out[1], "decode diverged at step {i}");
        (t0, t2) = (out[0], out[1]);
    }
}

#[test]
#[ignore = "needs a CUDA GPU with 20 GB free and GRANITE_H_TINY_DIR"]
fn granite_h_tiny_chunked_scan_prefill_speed_and_agreement() {
    use specrt::codegen::PAGE;
    use specrt::runtime::{Layout, Sampling};
    let base = spec();
    let vocab = base.dims.vocab as u32;
    let max_ctx: usize = 16384 + 64;
    let pages = max_ctx.div_ceil(PAGE);
    let mut result = Vec::new();
    for chunked in [false, true] {
        let mut s = base.clone();
        s.runtime.chunked_scan = chunked;
        let layout = Layout { max_ctx, slots: 1, kv_blocks: pages, prefill_chunk: 2048, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
        let mut rt = Runtime::load_layout(&s, layout).unwrap();
        let mut st = rt.stepper().unwrap();
        rt.set_block_table(0, &(0..pages as u32).collect::<Vec<_>>()).unwrap();
        let mut row = Vec::new();
        for len in [4096usize, 16384] {
            let prompt: Vec<u32> = (0..len as u32).map(|i| (i * 7919 + 13) % vocab).collect();
            let mut best = f64::MAX;
            let mut first = 0;
            for _ in 0..4 {
                rt.reset_slot(0).unwrap();
                let t = std::time::Instant::now();
                for at in (0..len).step_by(2048) {
                    let end = (at + 2048).min(len);
                    let sample = (end == len).then_some((Sampling::GREEDY, 0));
                    if let Some(tok) = rt.prefill_slot(&mut st, 0, &prompt[at..end], at, sample).unwrap() {
                        first = tok;
                    }
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            row.push((len, best, first));
        }
        result.push(row);
    }
    for (rec, ck) in result[0].iter().zip(&result[1]) {
        eprintln!("{NAME} prefill {} tokens: recurrent scan {:.0} tok/s, chunked scan {:.0} tok/s, {:.2}x", rec.0, rec.0 as f64 / rec.1, ck.0 as f64 / ck.1, rec.1 / ck.1);
        assert_eq!(rec.2, ck.2, "first greedy token after a {} token prompt", rec.0);
    }
}
