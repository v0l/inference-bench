use specrt::codegen::PAGE;
use specrt::prefix::{block_hashes, namespace, snapshot_points, PrefixCache, StateSnapshot};
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
use specrt::sched::BlockPool;
use specrt::spec::{PrefixCacheOptions, Spec};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

fn options(max_blocks: usize, max_snapshots: usize) -> PrefixCacheOptions {
    PrefixCacheOptions { enabled: true, snapshot_interval: 2048, max_snapshots, max_blocks }
}

fn tokens(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + salt * 11 + 5) % 1024).collect()
}

fn snapshot(tag: f32) -> Arc<StateSnapshot> {
    Arc::new(StateSnapshot { conv: vec![tag; 4], ssm: vec![tag; 8] })
}

fn root(salt: &str) -> specrt::prefix::Hash {
    namespace(&spec_for("tiny-qwen3.5"), 1, salt)
}

fn fill(pool: &mut BlockPool, n: usize) -> Vec<u32> {
    (0..n).map(|_| pool.alloc().unwrap()).collect()
}

#[test]
fn equal_prefixes_share_hashes_and_diverge_after_the_first_changed_token() {
    let r = root("");
    let a = tokens(5 * PAGE + 3, 0);
    let mut b = a.clone();
    b[2 * PAGE + 1] ^= 1;
    let ha = block_hashes(&r, &a);
    let hb = block_hashes(&r, &b);
    assert_eq!(ha.len(), 5, "a partial tail block is not hashed");
    assert_eq!(ha[..2], hb[..2]);
    assert!(ha[2..].iter().zip(&hb[2..]).all(|(x, y)| x != y), "a changed block changes every later hash");
    assert_eq!(block_hashes(&r, &a[..3 * PAGE]), ha[..3]);
}

#[test]
fn salt_model_and_kv_planes_separate_the_namespaces() {
    let s = spec_for("tiny-qwen3.5");
    let t = tokens(2 * PAGE, 0);
    let base = block_hashes(&namespace(&s, 1, ""), &t);
    assert_ne!(base, block_hashes(&namespace(&s, 1, "tenant-a"), &t));
    assert_ne!(block_hashes(&namespace(&s, 1, "tenant-a"), &t), block_hashes(&namespace(&s, 1, "tenant-b"), &t));
    assert_ne!(base, block_hashes(&namespace(&s, 2, ""), &t));
    let mut other = s.clone();
    other.name = "other".into();
    assert_ne!(base, block_hashes(&namespace(&other, 1, ""), &t));
    let mut runtime_only = s.clone();
    runtime_only.runtime.graphs = false;
    assert_eq!(base, block_hashes(&namespace(&runtime_only, 1, ""), &t), "runtime knobs do not change the key");
}

#[test]
fn block_refcounts_free_a_block_only_when_the_last_holder_lets_go() {
    let mut pool = BlockPool::new(4);
    let b = pool.alloc().unwrap();
    assert_eq!(pool.refs(b), 1);
    pool.retain(b);
    assert_eq!(pool.refs(b), 2);
    let mut one = vec![b];
    pool.release(&mut one);
    assert_eq!(pool.refs(b), 1);
    assert_eq!(pool.free_count(), 3);
    let mut again = vec![b];
    pool.release(&mut again);
    assert_eq!(pool.refs(b), 0);
    assert_eq!(pool.free_count(), 4);
    assert!(again.is_empty());
}

fn register_chain(cache: &mut PrefixCache, pool: &mut BlockPool, r: &specrt::prefix::Hash, t: &[u32]) -> (Vec<specrt::prefix::Hash>, Vec<u32>) {
    let hashes = block_hashes(r, t);
    let blocks = fill(pool, hashes.len());
    cache.register(&hashes, &blocks, pool);
    (hashes, blocks)
}

#[test]
fn a_hit_shares_the_cached_blocks_and_retains_them() {
    let mut pool = BlockPool::new(32);
    let mut cache = PrefixCache::new(&options(0, 4), false);
    let r = root("");
    let t = tokens(4 * PAGE, 0);
    let (_, blocks) = register_chain(&mut cache, &mut pool, &r, &t);
    pool.release(&mut blocks.clone());
    let longer = [t.clone(), tokens(PAGE + 5, 9)].concat();
    let hit = cache.lookup(&block_hashes(&r, &longer), longer.len() - 1, &mut pool);
    assert_eq!(hit.tokens, 4 * PAGE);
    assert_eq!(hit.blocks, blocks);
    assert!(hit.blocks.iter().all(|&b| pool.refs(b) == 2), "cache and the new sequence both hold it");
    assert!(hit.snapshot.is_none());
}

#[test]
fn the_last_prompt_token_is_never_served_from_the_cache() {
    let mut pool = BlockPool::new(32);
    let mut cache = PrefixCache::new(&options(0, 4), false);
    let r = root("");
    let t = tokens(3 * PAGE, 0);
    let (hashes, _) = register_chain(&mut cache, &mut pool, &r, &t);
    let hit = cache.lookup(&hashes, t.len() - 1, &mut pool);
    assert_eq!(hit.tokens, 2 * PAGE);
    assert_eq!(hit.blocks.len(), 2);
}

#[test]
fn a_hybrid_hit_stops_at_the_deepest_block_that_has_a_state_snapshot() {
    let mut pool = BlockPool::new(32);
    let mut cache = PrefixCache::new(&options(0, 4), true);
    let r = root("");
    let t = tokens(6 * PAGE, 0);
    let (hashes, _) = register_chain(&mut cache, &mut pool, &r, &t);
    let miss = cache.lookup(&hashes, 6 * PAGE - 1, &mut pool);
    assert_eq!((miss.tokens, miss.blocks.len()), (0, 0), "KV without a state is not a hit");
    cache.store_snapshot(hashes[1], snapshot(1.0));
    cache.store_snapshot(hashes[3], snapshot(3.0));
    let hit = cache.lookup(&hashes, 6 * PAGE - 1, &mut pool);
    assert_eq!(hit.tokens, 4 * PAGE);
    assert_eq!(hit.blocks.len(), 4);
    assert_eq!(hit.snapshot.unwrap().conv[0], 3.0);
    let shorter = cache.lookup(&hashes, 3 * PAGE + 2, &mut pool);
    assert_eq!(shorter.tokens, 2 * PAGE);
    assert_eq!(shorter.snapshot.unwrap().conv[0], 1.0);
}

#[test]
fn a_snapshot_for_an_uncached_block_is_refused() {
    let mut pool = BlockPool::new(8);
    let mut cache = PrefixCache::new(&options(0, 4), true);
    let r = root("");
    let hashes = block_hashes(&r, &tokens(2 * PAGE, 0));
    assert!(!cache.store_snapshot(hashes[0], snapshot(1.0)));
    let (hashes, _) = register_chain(&mut cache, &mut pool, &r, &tokens(2 * PAGE, 0));
    assert!(cache.store_snapshot(hashes[1], snapshot(1.0)));
    assert_eq!(cache.snapshots(), 1);
}

#[test]
fn eviction_takes_the_least_recently_used_unreferenced_block_tail_first() {
    let mut pool = BlockPool::new(16);
    let mut cache = PrefixCache::new(&options(0, 4), false);
    let r = root("");
    let (ha, mut ba) = register_chain(&mut cache, &mut pool, &r, &tokens(3 * PAGE, 0));
    let (hb, mut bb) = register_chain(&mut cache, &mut pool, &r, &tokens(3 * PAGE, 1));
    pool.release(&mut ba);
    pool.release(&mut bb);
    let touched = cache.lookup(&ha, 3 * PAGE, &mut pool);
    let mut held = touched.blocks.clone();
    pool.release(&mut held);
    assert!(cache.evict_one(&mut pool));
    assert!(cache.evict_one(&mut pool));
    assert!(cache.evict_one(&mut pool));
    let gone = cache.lookup(&hb, 3 * PAGE, &mut pool);
    assert_eq!(gone.tokens, 0, "the untouched chain went first");
    let kept = cache.lookup(&ha, 3 * PAGE, &mut pool);
    assert_eq!(kept.tokens, 3 * PAGE);
    let mut kb = kept.blocks.clone();
    pool.release(&mut kb);
    assert_eq!(cache.blocks(), 3);
}

#[test]
fn a_partly_evicted_chain_loses_its_tail_and_still_hits_its_head() {
    let mut pool = BlockPool::new(16);
    let mut cache = PrefixCache::new(&options(0, 4), false);
    let r = root("");
    let (hashes, mut blocks) = register_chain(&mut cache, &mut pool, &r, &tokens(4 * PAGE, 0));
    pool.release(&mut blocks);
    assert!(cache.evict_one(&mut pool));
    let hit = cache.lookup(&hashes, 4 * PAGE, &mut pool);
    assert_eq!(hit.tokens, 3 * PAGE);
}

#[test]
fn blocks_held_by_a_running_sequence_are_not_evicted() {
    let mut pool = BlockPool::new(8);
    let mut cache = PrefixCache::new(&options(0, 4), false);
    let r = root("");
    let (_, blocks) = register_chain(&mut cache, &mut pool, &r, &tokens(2 * PAGE, 0));
    assert!(!cache.evict_one(&mut pool));
    assert_eq!(cache.blocks(), 2);
    assert!(blocks.iter().all(|&b| pool.refs(b) == 2));
}

#[test]
fn the_block_budget_evicts_old_chains_when_registering_new_ones() {
    let mut pool = BlockPool::new(32);
    let mut cache = PrefixCache::new(&options(4, 4), false);
    let r = root("");
    let (ha, mut ba) = register_chain(&mut cache, &mut pool, &r, &tokens(3 * PAGE, 0));
    pool.release(&mut ba);
    let (hb, mut bb) = register_chain(&mut cache, &mut pool, &r, &tokens(3 * PAGE, 1));
    pool.release(&mut bb);
    assert_eq!(cache.blocks(), 4);
    assert_eq!(cache.lookup(&hb, 3 * PAGE, &mut pool).tokens, 3 * PAGE);
    assert!(cache.lookup(&ha, 3 * PAGE, &mut pool).tokens < 3 * PAGE);
}

#[test]
fn evicting_a_block_drops_its_snapshot() {
    let mut pool = BlockPool::new(8);
    let mut cache = PrefixCache::new(&options(0, 4), true);
    let r = root("");
    let (hashes, mut blocks) = register_chain(&mut cache, &mut pool, &r, &tokens(2 * PAGE, 0));
    cache.store_snapshot(hashes[1], snapshot(1.0));
    pool.release(&mut blocks);
    assert!(cache.evict_one(&mut pool));
    assert_eq!(cache.snapshots(), 0);
}

#[test]
fn the_snapshot_budget_drops_the_least_recently_used_state() {
    let mut pool = BlockPool::new(32);
    let mut cache = PrefixCache::new(&options(0, 2), true);
    let r = root("");
    let (hashes, mut blocks) = register_chain(&mut cache, &mut pool, &r, &tokens(6 * PAGE, 0));
    pool.release(&mut blocks);
    cache.store_snapshot(hashes[1], snapshot(1.0));
    cache.store_snapshot(hashes[3], snapshot(3.0));
    let hit = cache.lookup(&hashes[..2], 2 * PAGE, &mut pool);
    assert_eq!(hit.tokens, 2 * PAGE);
    let mut held = hit.blocks.clone();
    pool.release(&mut held);
    cache.store_snapshot(hashes[5], snapshot(5.0));
    assert_eq!(cache.snapshots(), 2);
    let deep = cache.lookup(&hashes, 6 * PAGE, &mut pool);
    assert_eq!(deep.snapshot.unwrap().conv[0], 5.0);
    let mid = cache.lookup(&hashes, 4 * PAGE, &mut pool);
    assert_eq!(mid.snapshot.unwrap().conv[0], 1.0, "the state touched by the earlier hit survived, the untouched one went");
}

#[test]
fn snapshot_points_fall_on_the_interval_and_on_the_last_full_block() {
    assert_eq!(snapshot_points(0, 100, 32), vec![32, 64, 96]);
    assert_eq!(snapshot_points(0, 90, 32), vec![32, 64, 80]);
    assert_eq!(snapshot_points(48, 200, 64), vec![64, 128, 192]);
    assert_eq!(snapshot_points(0, 10, 32), Vec::<usize>::new());
    assert_eq!(snapshot_points(64, 70, 32), Vec::<usize>::new());
}

#[test]
fn the_prefix_cache_table_parses_and_defaults_to_off() {
    let s = spec_for("tiny-qwen3.5");
    assert!(!s.runtime.prefix_cache.enabled);
    assert_eq!(s.runtime.prefix_cache.snapshot_interval, 2048);
    let toml = s.to_toml().unwrap();
    let with = format!("{toml}\n[runtime.prefix_cache]\nenabled = true\nsnapshot_interval = 512\nmax_snapshots = 7\nmax_blocks = 100\n");
    let p = Spec::parse(&with).unwrap().runtime.prefix_cache;
    assert_eq!(p, PrefixCacheOptions { enabled: true, snapshot_interval: 512, max_snapshots: 7, max_blocks: 100 });
    let bad = format!("{toml}\n[runtime.prefix_cache]\nsnapshot_interval = 100\n");
    assert!(format!("{:#}", Spec::parse(&bad).unwrap_err()).contains("snapshot_interval"));
}

fn max_ctx() -> usize {
    512
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn prefill_a_then_b_equals_prefill_a_snapshot_then_b_from_the_snapshot() {
    let s = spec_for("tiny-qwen3.5");
    let a = tokens(3 * PAGE, 0);
    let b = tokens(37, 4);
    let whole = [a.clone(), b.clone()].concat();
    let pages = max_ctx().div_ceil(PAGE);
    let layout = Layout { max_ctx: max_ctx(), slots: 4, kv_blocks: 4 * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut pool = BlockPool::new(4 * pages);
    let mut cache = PrefixCache::new(&options(0, 4), true);
    let root = namespace(&s, 1, "");
    let greedy = Some((Sampling::GREEDY, 0));

    let table0 = fill(&mut pool, pages);
    rt.set_block_table(0, &table0).unwrap();
    rt.prefill_slot(&mut st, 0, &a, 0, None).unwrap();
    let first0 = rt.prefill_slot(&mut st, 0, &b, a.len(), greedy).unwrap().unwrap();
    let logits0 = rt.last_logits().unwrap();

    let table1 = fill(&mut pool, pages);
    rt.set_block_table(1, &table1).unwrap();
    rt.prefill_slot(&mut st, 1, &a, 0, None).unwrap();
    let hashes = block_hashes(&root, &a);
    cache.register(&hashes, &table1[..hashes.len()], &mut pool);
    assert!(cache.store_snapshot(*hashes.last().unwrap(), Arc::new(rt.snapshot_state(1).unwrap())));
    let mut released = table1.clone();
    pool.release(&mut released);

    let hit = cache.lookup(&block_hashes(&root, &whole), whole.len() - 1, &mut pool);
    assert_eq!(hit.tokens, a.len());
    let mut table2 = hit.blocks.clone();
    table2.extend(fill(&mut pool, pages - table2.len()));
    rt.set_block_table(2, &table2).unwrap();
    rt.restore_state(2, &hit.snapshot.unwrap()).unwrap();
    let first2 = rt.prefill_slot(&mut st, 2, &whole[hit.tokens..], hit.tokens, greedy).unwrap().unwrap();
    let logits2 = rt.last_logits().unwrap();

    assert_eq!(first0, first2);
    assert_eq!(logits0.len(), logits2.len());
    assert!(logits0.iter().zip(&logits2).all(|(x, y)| x.to_bits() == y.to_bits()), "logits are not bit identical");

    let (mut t0, mut t2) = (first0, first2);
    for i in 0..16 {
        let row = |slot, token| DecodeRow { slot, token, pos: whole.len() + i, sampling: Sampling::GREEDY, step: i as u32 + 1 };
        let out = rt.decode(&mut st, &[row(0, t0), row(2, t2)]).unwrap();
        assert_eq!(out[0], out[1], "decode diverged at step {i}");
        (t0, t2) = (out[0], out[1]);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_restored_snapshot_is_independent_of_the_slot_it_came_from() {
    let s = spec_for("tiny-qwen3.5");
    let a = tokens(2 * PAGE, 2);
    let pages = max_ctx().div_ceil(PAGE);
    let layout = Layout { max_ctx: max_ctx(), slots: 3, kv_blocks: 3 * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    let mut st = rt.stepper().unwrap();
    let table: Vec<u32> = (0..pages as u32).collect();
    rt.set_block_table(0, &table).unwrap();
    rt.prefill_slot(&mut st, 0, &a, 0, None).unwrap();
    let snap = rt.snapshot_state(0).unwrap();
    rt.prefill_slot(&mut st, 0, &tokens(20, 8), a.len(), None).unwrap();
    assert_ne!(rt.snapshot_state(0).unwrap(), snap);
    rt.restore_state(1, &snap).unwrap();
    assert_eq!(rt.snapshot_state(1).unwrap(), snap);
}

fn thinking_template() -> specrt::chat::ChatTemplate {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    specrt::chat::ChatTemplate::from_files(&read("qwen3.8-27b.tokenizer_config.json"), Some(&read("qwen3.8-27b.chat_template.jinja")))
        .unwrap()
        .unwrap()
}

fn qwen_tokenizer() -> Option<tokenizers::Tokenizer> {
    ["Qwen/Qwen3-0.6B", "orcarouter/Qwen3.8-27B-Uncensored-NVFP4"]
        .iter()
        .find_map(|repo| specrt::weights::find_in_snapshot(repo, "tokenizer.json").ok())
        .and_then(|p| tokenizers::Tokenizer::from_file(p).ok())
}

fn ids(t: &specrt::chat::ChatTemplate, tok: &tokenizers::Tokenizer, messages: &[serde_json::Value], vars: &serde_json::Value) -> Vec<u32> {
    let text = t.render_with(messages, vars).unwrap();
    tok.encode(text.as_str(), false).unwrap().get_ids().to_vec()
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn shared_blocks(r: &specrt::prefix::Hash, a: &[u32], b: &[u32]) -> usize {
    block_hashes(r, a).iter().zip(block_hashes(r, b)).take_while(|(x, y)| *x == y).count()
}

#[test]
fn the_cache_key_follows_the_final_templated_ids_across_thinking_turns() {
    use serde_json::json;
    let Some(tok) = qwen_tokenizer() else { return };
    let t = thinking_template();
    let r = root("");
    let system = json!({"role": "system", "content": "You are a coding agent. Use the tools to answer. ".repeat(6)});
    let user1 = json!({"role": "user", "content": "List the files in src and explain what each one does in detail please."});
    let user2 = json!({"role": "user", "content": "Now read main.rs and summarise it."});
    let reasoning = "The user wants a listing, so I should call the tool and then describe each file carefully.";
    let answer = "src has main.rs and lib.rs. main.rs is the entry point and lib.rs holds the library code.";
    let assistant = json!({"role": "assistant", "content": answer, "reasoning_content": reasoning});

    for vars in [json!({"enable_thinking": true}), json!({"enable_thinking": true, "preserve_thinking": true})] {
        let turn1 = ids(&t, &tok, &[system.clone(), user1.clone()], &vars);
        let turn2 = ids(&t, &tok, &[system.clone(), user1.clone(), assistant.clone(), user2.clone()], &vars);
        let divergence = common_prefix(&turn1, &turn2);
        assert!(divergence >= turn1.len() - 1, "the generation prompt of turn 1 is a prefix of turn 2 when the reasoning is kept");
        assert_eq!(shared_blocks(&r, &turn1, &turn2), turn1.len().min(divergence) / PAGE);
    }

    let vars = json!({"enable_thinking": true, "preserve_thinking": false});
    let turn1 = ids(&t, &tok, &[system.clone(), user1.clone()], &vars);
    let turn2 = ids(&t, &tok, &[system.clone(), user1.clone(), assistant.clone(), user2.clone()], &vars);
    let divergence = common_prefix(&turn1, &turn2);
    assert!(divergence < turn1.len(), "stripping the reasoning changes the tail of the turn 1 prompt");
    assert_eq!(shared_blocks(&r, &turn1, &turn2), divergence / PAGE, "the chain forks at the first changed token block");

    let as_text = ids(&t, &tok, &[system.clone(), user1.clone()], &vars);
    let as_parts = ids(
        &t,
        &tok,
        &[system.clone(), json!({"role": "user", "content": [{"type": "text", "text": user1["content"].as_str().unwrap()}]})],
        &vars,
    );
    assert_eq!(block_hashes(&r, &as_text), block_hashes(&r, &as_parts), "equal templated ids give equal keys whatever the request shape");

    let off = ids(&t, &tok, &[system, user1], &json!({"enable_thinking": false}));
    assert_ne!(block_hashes(&r, &off).last(), block_hashes(&r, &as_text).last(), "thinking on and off are different prompts and never share a tail");
}
