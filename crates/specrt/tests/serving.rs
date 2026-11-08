use specrt::runtime::{Layout, Runtime, Sampling};
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
