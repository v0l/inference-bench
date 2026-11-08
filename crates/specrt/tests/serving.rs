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
    let layout = Layout { max_ctx, slots: 3, kv_blocks: 3 * pages as usize, prefill_chunk: 128, kv_planes };
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
    let layout = Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 128, kv_planes: 1 };
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
    let s = spec_for(dir);
    let prompts = mixed_prompts();
    let expected = single_greedy(&s, &prompts, 24);
    assert_eq!(stepped_greedy(&s, &prompts, 24, 37), expected);
    assert_eq!(stepped_greedy(&s, &prompts, 24, 128), expected);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_mixed_length_step_decode_matches_single_sequence_greedy() {
    stepped_matches_single("tiny-qwen3.5");
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

#[test]
fn serving_kv_is_one_f16_copy_per_token() {
    let s = spec_for("tiny-qwen3.5-hd256");
    let per_token = specrt::runtime::kv_block_bytes(&s, 1) / specrt::codegen::PAGE;
    assert_eq!(per_token, s.attention_layers() * s.attention.kv_heads * s.attention.head_dim * 2 * 2);
    let real = Spec::load(&fixture_dir("qwen3.8-27b.toml")).unwrap();
    assert_eq!(specrt::runtime::kv_block_bytes(&real, 1) / specrt::codegen::PAGE, 64 * 1024);
    let layout = Layout::serving(&real, 131072, 32, 7 << 30);
    assert_eq!((layout.kv_planes, layout.kv_blocks), (1, (7 << 30) / (64 * 1024 * 16)));
}
