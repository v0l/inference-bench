use specrt::pipeline::{Hop, Pipeline};
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn tiny(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec
}

fn named(file: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs").join(file)).unwrap()
}

const MAX_CTX: usize = 512;
const NEW_TOKENS: usize = 64;

fn layout() -> Layout {
    let pages = MAX_CTX.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: MAX_CTX, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn blocks() -> Vec<u32> {
    let pages = MAX_CTX.div_ceil(specrt::codegen::PAGE) as u32;
    (0..pages).collect()
}

fn prompt(vocab: u32) -> Vec<u32> {
    (0..150u32).map(|i| (i * 37 + 11) % vocab).collect()
}

fn single(spec: &Spec, prompt: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let mut rt = Runtime::load_layout(spec, layout()).unwrap();
    rt.set_block_table(0, &blocks()).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut first = None;
    let mut start = 0;
    let mut logits = Vec::new();
    while start < prompt.len() {
        let end = (start + 128).min(prompt.len());
        let last = end == prompt.len();
        first = rt.prefill_slot(&mut st, 0, &prompt[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap();
        start = end;
    }
    logits.extend(rt.last_logits().unwrap());
    let mut out = vec![first.unwrap()];
    for _ in 1..NEW_TOKENS {
        let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: prompt.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
        out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
    }
    (logits, out)
}

fn piped(spec: &Spec, ranges: &[(usize, usize, usize)], hop: Hop, prompt: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let mut p = Pipeline::load_ranges(spec, ranges, layout(), hop).unwrap();
    p.set_block_table(0, &blocks()).unwrap();
    let mut first = None;
    let mut start = 0;
    while start < prompt.len() {
        let end = (start + 128).min(prompt.len());
        let last = end == prompt.len();
        first = p.prefill_slot(0, &prompt[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap();
        start = end;
    }
    let logits = p.stage(ranges.len() - 1).last_logits().unwrap();
    let mut out = vec![first.unwrap()];
    for _ in 1..NEW_TOKENS {
        let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: prompt.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
        out.push(p.decode(&[row]).unwrap()[0]);
    }
    assert!(p.stats().hops > 0);
    (logits, out)
}

fn splits(layers: usize) -> Vec<(&'static str, Vec<(usize, usize, usize)>)> {
    let a = layers / 2;
    let b = layers / 3;
    let c = 2 * layers / 3;
    vec![("two", vec![(0, 0, a), (0, a, layers)]), ("three", vec![(0, 0, b), (0, b, c), (0, c, layers)])]
}

fn bit_identical(spec: &Spec) {
    let p = prompt(spec.dims.vocab as u32);
    let (logits, tokens) = single(spec, &p);
    for hop in [Hop::Direct, Hop::Host] {
        for (label, ranges) in splits(spec.layers.count) {
            let (l, t) = piped(spec, &ranges, hop, &p);
            assert_eq!(t, tokens, "{} {label} {hop:?}: tokens", spec.name);
            assert!(l.iter().zip(&logits).all(|(a, b)| a.to_bits() == b.to_bits()), "{} {label} {hop:?}: logits", spec.name);
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_hybrid_pipeline_is_bit_identical_to_the_single_runtime() {
    bit_identical(&tiny("tiny-qwen3.5"));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_hybrid_hd256_pipeline_is_bit_identical_to_the_single_runtime() {
    bit_identical(&tiny("tiny-qwen3.5-hd256"));
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn qwen3_dense_pipeline_is_bit_identical_to_the_single_runtime() {
    bit_identical(&named("qwen3-0.6b.toml"));
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn qwen2_5_dense_pipeline_with_biases_is_bit_identical_to_the_single_runtime() {
    bit_identical(&named("qwen2.5-0.5b.toml"));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_planner_placement_runs_as_a_pipeline() {
    use specrt::placement::PlacementConfig;
    let spec = tiny("tiny-qwen3.5");
    let cfg: PlacementConfig = toml::from_str(
        r#"
        context = 512
        [[devices]]
        name = "a"
        memory_gib = 1
        bandwidth_gbs = 100
        tflops = 10
        reserve_gib = 0
        [[stages]]
        device = "a"
        layers = [0, 3]
        "#,
    )
    .unwrap();
    let placement = cfg.place(&spec).unwrap();
    let p = prompt(spec.dims.vocab as u32);
    let (_, expected) = single(&spec, &p[..100]);
    let mut pipe = Pipeline::load(&spec, &placement, layout(), Hop::Direct).unwrap();
    pipe.set_block_table(0, &blocks()).unwrap();
    let first = pipe.prefill_slot(0, &p[..100], 0, Some((Sampling::GREEDY, 0))).unwrap();
    assert_eq!(first, Some(expected[0]));
}

fn scheduled_prompts(vocab: u32) -> Vec<Vec<u32>> {
    let lens = [3usize, 17, 64, 130, 250, 300];
    (0..12u32).map(|q| (0..lens[q as usize % lens.len()] as u32).map(|i| (i * 13 + q * 97 + 1) % vocab).collect()).collect()
}

fn scheduled_through_pipeline(spec: &Spec, micro_batch: usize) -> (Vec<Vec<u32>>, specrt::sched::ChainMetrics) {
    use specrt::sched::{Config, Event, PipelineExecutor, Request, Scheduler};
    let layout = Layout { max_ctx: 512, slots: 6, kv_blocks: 6 * 8, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: true };
    let layers = spec.layers.count;
    let mut pipe = Pipeline::load_ranges(spec, &[(0, 0, layers / 2), (0, layers / 2, layers)], layout, Hop::Direct).unwrap();
    pipe.set_micro_batch(micro_batch);
    let cfg = Config { max_num_seqs: 6, max_batched_tokens: 256, reserve_blocks: Some(1), ..Config::default() };
    let mut sched = Scheduler::new(PipelineExecutor::new(pipe), cfg);
    let prompts = scheduled_prompts(spec.dims.vocab as u32);
    let ids: Vec<u64> = prompts
        .iter()
        .map(|p| sched.submit(Request { prompt: p.clone(), max_tokens: 16, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new() }).unwrap())
        .collect();
    let mut got: std::collections::HashMap<u64, Vec<u32>> = Default::default();
    while !sched.idle() {
        let mut events = Vec::new();
        sched.step(&mut events).unwrap();
        for e in events {
            if let Event::Token { id, token } = e {
                got.entry(id).or_default().push(token);
            }
        }
    }
    (ids.iter().map(|id| got[id].clone()).collect(), sched.chain_metrics().unwrap())
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_scheduler_drives_a_pipeline_with_micro_batched_prefill_and_matches_the_single_runtime() {
    let spec = tiny("tiny-qwen3.5");
    let mut single = Runtime::load(&spec, 512).unwrap();
    let want: Vec<Vec<u32>> = scheduled_prompts(spec.dims.vocab as u32).iter().map(|p| single.generate(p, 16, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let (plain, plain_chain) = scheduled_through_pipeline(&spec, 0);
    assert_eq!(plain, want);
    assert_eq!(plain_chain.micro_batches, 0);
    let (micro, chain) = scheduled_through_pipeline(&spec, 32);
    assert_eq!(micro, want);
    assert!(chain.micro_batches > 0 && chain.overlapped_prefills > 0, "{chain:?}");
    assert_eq!(chain.stages.len(), 2);
    assert!(chain.stages.iter().all(|s| s.calls > 0 && s.tokens > 0 && s.busy_secs > 0.0));
    assert!(chain.hops > 0 && chain.hop_bytes > 0);
}
