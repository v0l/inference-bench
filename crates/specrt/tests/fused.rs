use specrt::runtime::{DecodeRow, Layout, PrefillPart, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const BIG: Option<specrt::spec::TileChoice> = Some(specrt::spec::TileChoice::Big);

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
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.swap(i, (x >> 33) as usize % (i + 1));
    }
    v
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn mixed_prompts() -> Vec<Vec<u32>> {
    [5usize, 40, 129, 300]
        .iter()
        .enumerate()
        .map(|(q, &n)| (0..n as u32).map(|i| (i * 7 + q as u32 * 131 + 3) % 1024).collect())
        .collect()
}

fn fused_pair(dir: &str, tile: Option<specrt::spec::TileChoice>, bound: f32) {
    let mut s = spec_for(dir);
    s.runtime.tile = tile.unwrap_or_default();
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    let slots = 4;
    let layout = Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 256, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let prompts = mixed_prompts();
    let long: Vec<u32> = (0..200u32).map(|i| (i * 13 + 5) % 1024).collect();
    let setup = |rt: &mut Runtime, st: &mut specrt::runtime::Stepper| -> Vec<DecodeRow> {
        let pool = shuffled((slots * pages) as u32, 5);
        for slot in 0..slots {
            rt.set_block_table(slot, &pool[slot * pages..(slot + 1) * pages]).unwrap();
        }
        (0..3)
            .map(|i| {
                let p = &prompts[i][..prompts[i].len().min(250)];
                let t = rt.prefill_slot(st, i, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
                DecodeRow { slot: i, token: t, pos: p.len(), sampling: Sampling::GREEDY, step: 1 }
            })
            .collect()
    };
    let mut apart = Runtime::load_layout(&s, layout).unwrap();
    let mut ast = apart.stepper().unwrap();
    let rows = setup(&mut apart, &mut ast);
    let next = apart.decode(&mut ast, &rows).unwrap();
    let decode_logits = apart.step_logits(&ast, 0, 3).unwrap();
    let first = apart.prefill_slot(&mut ast, 3, &long, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
    let prefill_logits = apart.prefill_logits().unwrap();
    drop(ast);
    drop(apart);
    let mut fused = Runtime::load_layout(&s, layout).unwrap();
    let mut fst = fused.stepper().unwrap();
    let frows = setup(&mut fused, &mut fst);
    assert_eq!(frows, rows);
    let part = PrefillPart { slot: 3, tokens: &long, start: 0, sample: Some((Sampling::GREEDY, 0)) };
    let (fnext, ffirst) = fused.fused_step(&mut fst, &rows, part).unwrap();
    assert_eq!(fnext, next);
    assert_eq!(ffirst, Some(first));
    let got = fused.step_logits(&fst, 0, 3).unwrap();
    assert!(max_abs(&got, &decode_logits) <= bound, "decode {}", max_abs(&got, &decode_logits));
    let got = fused.step_logits(&fst, 3, 1).unwrap();
    assert!(max_abs(&got, &prefill_logits) <= bound, "prefill {}", max_abs(&got, &prefill_logits));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_fused_step_of_three_decode_rows_and_a_200_token_chunk_matches_separate_launches_at_one_tile() {
    fused_pair("tiny-qwen3.5", BIG, 1e-5);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_fused_step_matches_separate_launches_within_tile_rounding() {
    fused_pair("tiny-qwen3.5", None, 1e-4);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_fused_step_of_three_decode_rows_and_a_200_token_chunk_matches_separate_launches_at_one_tile() {
    fused_pair("tiny-qwen3.5-hd256", BIG, 1e-5);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_fused_step_matches_separate_launches_within_tile_rounding() {
    fused_pair("tiny-qwen3.5-hd256", None, 1e-4);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_fused_step_of_three_decode_rows_and_a_200_token_chunk_matches_separate_launches_at_one_tile() {
    fused_pair("tiny-qwen3.5-fp8", BIG, 1e-5);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_fused_step_matches_separate_launches_within_tile_rounding() {
    fused_pair("tiny-qwen3.5-fp8", None, 1e-4);
}
