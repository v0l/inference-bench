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
