use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::PathBuf;

const PROMPT: [u32; 18] = [785, 6722, 315, 9625, 374, 12095, 13, 9645, 264, 2805, 32794, 911, 279, 9396, 323, 279, 9759, 13];
const VOCAB: usize = 151936;

fn dir() -> PathBuf {
    std::env::var_os("SPECRT_GGUF_DIR").filter(|d| !d.is_empty()).map(PathBuf::from).unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/specrt-gguf"))
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn kl(reference: &[f32], ours: &[f32]) -> f64 {
    let (p, q) = (log_softmax(reference), log_softmax(ours));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

struct Outcome {
    mean_kl: f64,
    max_kl: f64,
    top1: usize,
    greedy_match: usize,
    greedy_len: usize,
}

fn run(file: &str) -> Outcome {
    let stem = file.trim_end_matches(".gguf").trim_start_matches("Qwen3-0.6B-");
    let reference: Vec<f32> = std::fs::read(dir().join(format!("q3-{stem}.logits.bin")))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let greedy: Vec<u32> = std::fs::read_to_string(dir().join(format!("q3-{stem}.greedy.txt"))).unwrap().lines().map(|l| l.parse().unwrap()).collect();
    let spec = Spec::from_gguf_file(&dir().join(file)).unwrap();
    let mut rt = Runtime::load(&spec, 256).unwrap();
    let ours = rt.prompt_logits(&PROMPT).unwrap();
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&reference[r * VOCAB..(r + 1) * VOCAB], &ours[r * VOCAB..(r + 1) * VOCAB])).collect();
    let top1 = (0..rows).filter(|&r| argmax(&reference[r * VOCAB..(r + 1) * VOCAB]) == argmax(&ours[r * VOCAB..(r + 1) * VOCAB])).count();
    let out = rt.generate(&PROMPT, greedy.len(), Sampling::GREEDY).unwrap();
    let greedy_match = out.tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    let o = Outcome { mean_kl: kls.iter().sum::<f64>() / rows as f64, max_kl: kls.iter().copied().fold(0.0, f64::max), top1, greedy_match, greedy_len: greedy.len() };
    eprintln!("{file}: mean KL {:.3e} max KL {:.3e} top1 {}/{rows} greedy prefix {}/{}", o.mean_kl, o.max_kl, o.top1, o.greedy_match, o.greedy_len);
    o
}

#[test]
#[ignore = "needs a CUDA GPU, the unsloth Qwen3-0.6B GGUF files and llama.cpp reference logits in ~/.cache/specrt-gguf (scripts/gguf_reference.c)"]
fn qwen3_q8_0_gguf_logits_stay_within_the_kl_bound_of_llama_cpp() {
    let o = run("Qwen3-0.6B-Q8_0.gguf");
    assert!(o.mean_kl <= 2e-3, "mean KL {}", o.mean_kl);
    assert!(o.max_kl <= 1e-2, "max KL {}", o.max_kl);
    assert_eq!(o.top1, PROMPT.len());
}

#[test]
#[ignore = "needs a CUDA GPU, the unsloth Qwen3-0.6B GGUF files and llama.cpp reference logits in ~/.cache/specrt-gguf (scripts/gguf_reference.c)"]
fn qwen3_q8_0_gguf_greedy_generation_matches_llama_cpp_for_64_tokens() {
    let o = run("Qwen3-0.6B-Q8_0.gguf");
    assert_eq!(o.greedy_len, 64);
    assert_eq!(o.greedy_match, 64);
}

#[test]
#[ignore = "needs a CUDA GPU, the unsloth Qwen3-0.6B GGUF files and llama.cpp reference logits in ~/.cache/specrt-gguf (scripts/gguf_reference.c)"]
fn qwen3_q4_k_m_gguf_logits_stay_within_the_kl_bound_of_llama_cpp() {
    let o = run("Qwen3-0.6B-Q4_K_M.gguf");
    assert!(o.mean_kl <= 2e-3, "mean KL {}", o.mean_kl);
    assert!(o.max_kl <= 1e-2, "max KL {}", o.max_kl);
    assert_eq!(o.top1, PROMPT.len());
}
