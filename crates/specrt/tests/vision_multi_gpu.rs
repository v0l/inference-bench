use serde_json::Value;
use specrt::pipeline::{Hop, Pipeline};
use specrt::runtime::{Layout, Runtime};
use specrt::sched::{Executor, MediaInput};
use specrt::spec::Spec;
use specrt::tp_exec::TpExecutor;
use specrt::vision::{Patches, Processed};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-27b")
}

fn reference_dir() -> PathBuf {
    fixture().join("reference")
}

fn device() -> usize {
    std::env::var("SPECRT_TP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn spec() -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut s = Spec::from_hf_config(&json).unwrap();
    s.name = "tiny-qwen3.5-27b".into();
    s.repo = fixture().display().to_string();
    s.runtime.vision.precision = specrt::vision_encoder::VisionPrecision::Exact;
    s.runtime.graphs = false;
    s
}

fn floats(name: &str) -> Vec<f32> {
    std::fs::read(reference_dir().join(name)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(reference_dir().join(name)).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn mean_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows)
        .map(|r| {
            let (a, b) = (log_softmax(&p[r * vocab..(r + 1) * vocab]), log_softmax(&q[r * vocab..(r + 1) * vocab]));
            a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn layout() -> Layout {
    let pages = 256usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 256, slots: 2, kv_blocks: 2 * pages + 2, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn blocks() -> Vec<u32> {
    (0..256usize.div_ceil(specrt::codegen::PAGE) as u32).collect()
}

struct Case {
    prompt: Vec<u32>,
    media: Vec<MediaInput>,
}

fn case(name: &str) -> Case {
    let m = meta(&format!("tiny27b-vision-{name}.json"));
    let patch_dim = m["patch_dim"].as_u64().unwrap() as usize;
    let all = floats(&format!("tiny27b-vision-{name}.patches.bin"));
    let mut at = 0;
    let media = m["grids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let grid = [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize];
            let n = grid.iter().product::<usize>() * patch_dim;
            let data = all[at..at + n].to_vec();
            at += n;
            MediaInput { processed: Processed::Image(Patches { grid, data, layout: [0, 0] }), hash: [(name.len() * 8 + i) as u8 + 1; 32] }
        })
        .collect();
    Case { prompt: ids(&m["ids"]), media }
}

#[test]
fn every_tensor_parallel_rank_spec_keeps_the_vision_tower_and_mrope() {
    let s = spec();
    assert!(s.vision.is_some() && s.rope.mrope.is_some());
    for rank in 0..2 {
        let r = specrt::tp::rank_spec_of(&s, 2, rank).unwrap();
        assert_eq!(r.vision, s.vision, "rank {rank}");
        assert_eq!(r.rope.mrope, s.rope.mrope, "rank {rank}");
    }
}

#[test]
fn every_pipeline_stage_spec_keeps_the_vision_tower_and_mrope() {
    let s = spec();
    for (a, b) in [(0, 2), (2, 4)] {
        let stage = s.stage(a, b);
        assert_eq!(stage.vision, s.vision, "stage {a}..{b}");
        assert_eq!(stage.rope.mrope, s.rope.mrope, "stage {a}..{b}");
    }
}

fn single_logits(c: &Case) -> Vec<f32> {
    let mut rt = Runtime::load_layout(&spec(), layout()).unwrap();
    let plan = rt.plan_media(&c.prompt, &c.media).unwrap();
    rt.set_media(0, Some(&plan)).unwrap();
    rt.prompt_logits(&c.prompt).unwrap()
}

fn check(name: &str, got: &[f32], want: &[f32], vocab: usize) {
    let kl = mean_kl(want, got, vocab);
    eprintln!("{name}: logits KL against the single GPU path {kl:e}");
    assert!(kl <= 1e-4, "{name}: KL {kl}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tp2_virtual_ranks_report_vision_and_match_the_single_gpu_image_logits() {
    let s = spec();
    let vocab = s.dims.vocab;
    let mut lay = layout();
    lay.kv_blocks = 2 * 256usize.div_ceil(specrt::codegen::PAGE) + 2;
    let mut exec = TpExecutor::load(&s, lay, &[device(), device()]).unwrap();
    assert!(exec.media_encoder().is_some(), "tp=2 must report a vision encoder");
    for name in ["image", "images2"] {
        let c = case(name);
        let want = single_logits(&c);
        let plan = exec.plan_media(&c.prompt, &c.media).unwrap();
        exec.set_media(0, Some(&plan)).unwrap();
        let got = exec.prompt_logits(&c.prompt).unwrap();
        check(&format!("tp2 {name}"), &got, &want, vocab);
        exec.set_media(0, None).unwrap();
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_pipeline_stages_report_vision_and_match_the_single_gpu_image_logits() {
    let s = spec();
    let vocab = s.dims.vocab;
    let n = s.layers.count;
    let mut pipe = Pipeline::load_ranges(&s, &[(device(), 0, n / 2), (device(), n / 2, n)], layout(), Hop::Direct).unwrap();
    pipe.set_block_table(0, &blocks()).unwrap();
    assert!(pipe.media_encoder().is_some(), "two stages must report a vision encoder");
    for name in ["image", "images2"] {
        let c = case(name);
        let want = single_logits(&c);
        let plan = pipe.plan_media(&c.prompt, &c.media).unwrap();
        pipe.set_media(0, Some(&plan)).unwrap();
        pipe.acquire_state(0).unwrap();
        let got = pipe.prompt_logits(0, &c.prompt).unwrap();
        pipe.release_state(0);
        assert_eq!(got.len(), vocab);
        check(&format!("2 stages {name}"), &got, &want[want.len() - vocab..], vocab);
        pipe.set_media(0, None).unwrap();
    }
}
