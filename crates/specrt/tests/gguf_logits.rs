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
    std::fs::write(dir().join(format!("q3-{stem}.specrt.bin")), ours.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&reference[r * VOCAB..(r + 1) * VOCAB], &ours[r * VOCAB..(r + 1) * VOCAB])).collect();
    let top1 = (0..rows).filter(|&r| argmax(&reference[r * VOCAB..(r + 1) * VOCAB]) == argmax(&ours[r * VOCAB..(r + 1) * VOCAB])).count();
    let out = rt.generate(&PROMPT, greedy.len(), Sampling::GREEDY).unwrap();
    let greedy_match = out.tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    let o = Outcome { mean_kl: kls.iter().sum::<f64>() / rows as f64, max_kl: kls.iter().copied().fold(0.0, f64::max), top1, greedy_match, greedy_len: greedy.len() };
    eprintln!("{file}: mean KL {:.3e} max KL {:.3e} top1 {}/{rows} greedy prefix {}/{} per position {:?}", o.mean_kl, o.max_kl, o.top1, o.greedy_match, o.greedy_len, kls.iter().map(|k| format!("{k:.1e}")).collect::<Vec<_>>());
    o
}

fn native_against_bf16(file: &str) -> (f64, f64, usize) {
    let native = Spec::from_gguf_file(&dir().join(file)).unwrap();
    let mut plain = native.clone();
    plain.quant = specrt::spec::Quant::Bf16;
    plain.head_quant = Some(specrt::spec::Quant::Bf16);
    let logits = |s: &Spec| -> (Vec<f32>, Vec<u32>) {
        let mut rt = Runtime::load(s, 256).unwrap();
        let l = rt.prompt_logits(&PROMPT).unwrap();
        let g = rt.generate(&PROMPT, 32, Sampling::GREEDY).unwrap().tokens;
        (l, g)
    };
    let ((a, ga), (b, gb)) = (logits(&plain), logits(&native));
    std::fs::write(dir().join(format!("{file}.bf16.bin")), a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&a[r * VOCAB..(r + 1) * VOCAB], &b[r * VOCAB..(r + 1) * VOCAB])).collect();
    let same = ga.iter().zip(&gb).take_while(|(x, y)| x == y).count();
    let (mean, max) = (kls.iter().sum::<f64>() / rows as f64, kls.iter().copied().fold(0.0, f64::max));
    eprintln!("{file}: native against dequantised bf16 mean KL {mean:.3e} max KL {max:.3e} greedy prefix {same}/32");
    (mean, max, same)
}

const ENGINE_KL_MEAN: f64 = 5e-2;
const ENGINE_KL_MAX: f64 = 0.5;
const ENGINE_TOP1: usize = 16;
const Q4_K_TOP1: usize = 14;

#[test]
#[ignore = "needs a CUDA GPU and the unsloth Qwen3-0.6B Q8_0 GGUF in ~/.cache/specrt-gguf"]
fn qwen3_q8_0_native_kernels_match_the_same_weights_dequantised_to_bf16() {
    let (mean, max, _) = native_against_bf16("Qwen3-0.6B-Q8_0.gguf");
    assert!(mean <= 2e-4 && max <= 1e-3, "mean {mean} max {max}");
}

#[test]
#[ignore = "needs a CUDA GPU and the unsloth Qwen3-0.6B Q4_K_M GGUF in ~/.cache/specrt-gguf"]
fn qwen3_q4_k_m_native_kernels_match_the_same_weights_dequantised_to_bf16() {
    let (mean, max, same) = native_against_bf16("Qwen3-0.6B-Q4_K_M.gguf");
    assert!(mean <= 1e-4 && max <= 1e-3, "mean {mean} max {max}");
    assert!(same >= 16, "greedy prefix {same}");
}

#[test]
#[ignore = "needs a CUDA GPU, Qwen3-0.6B-BF16.gguf (convert_hf_to_gguf.py --outtype bf16) and llama.cpp reference logits in ~/.cache/specrt-gguf"]
fn qwen3_bf16_gguf_logits_match_llama_cpp_closely() {
    let o = run("Qwen3-0.6B-BF16.gguf");
    assert!(o.mean_kl <= 1e-3 && o.max_kl <= 5e-3, "mean KL {} max KL {}", o.mean_kl, o.max_kl);
    assert_eq!(o.top1, PROMPT.len());
}

#[test]
#[ignore = "needs a CUDA GPU, the unsloth Qwen3-0.6B GGUF files and llama.cpp reference logits in ~/.cache/specrt-gguf (scripts/gguf_reference.c)"]
fn qwen3_q8_0_gguf_logits_stay_within_the_kl_bound_of_llama_cpp() {
    let o = run("Qwen3-0.6B-Q8_0.gguf");
    assert!(o.mean_kl <= ENGINE_KL_MEAN && o.max_kl <= ENGINE_KL_MAX, "mean KL {} max KL {}", o.mean_kl, o.max_kl);
    assert!(o.top1 >= ENGINE_TOP1, "top1 {}", o.top1);
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
    assert!(o.mean_kl <= ENGINE_KL_MEAN && o.max_kl <= ENGINE_KL_MAX, "mean KL {} max KL {}", o.mean_kl, o.max_kl);
    assert!(o.top1 >= Q4_K_TOP1, "top1 {}", o.top1);
}
