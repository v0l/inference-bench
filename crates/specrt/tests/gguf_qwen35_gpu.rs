use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const PROMPT: [u32; 18] = [785, 6722, 315, 9625, 374, 12095, 13, 9645, 264, 2805, 32794, 911, 279, 9396, 323, 279, 9759, 13];

fn file() -> PathBuf {
    PathBuf::from(std::env::var("SPECRT_GGUF_FILE").expect("SPECRT_GGUF_FILE"))
}

fn reference() -> Option<PathBuf> {
    std::env::var_os("SPECRT_GGUF_REF").map(PathBuf::from)
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (log_softmax(p), log_softmax(q));
    p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
#[ignore = "needs a CUDA GPU, SPECRT_GGUF_FILE and SPECRT_GGUF_REF (prefix of llama.cpp scripts/gguf_reference.c output)"]
fn a_qwen35_gguf_matches_llama_cpp_logits_and_greedy_text() {
    let spec = Spec::from_gguf_file(&file()).unwrap();
    let vocab = spec.dims.vocab;
    let mut rt = Runtime::load(&spec, 256).unwrap();
    let ours = rt.prompt_logits(&PROMPT).unwrap();
    let prefix = reference().expect("SPECRT_GGUF_REF");
    let want = floats(&PathBuf::from(format!("{}.logits.bin", prefix.display())));
    let greedy: Vec<u32> = std::fs::read_to_string(format!("{}.greedy.txt", prefix.display())).unwrap().lines().map(|l| l.parse().unwrap()).collect();
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&want[r * vocab..(r + 1) * vocab], &ours[r * vocab..(r + 1) * vocab])).collect();
    let top1 = (0..rows).filter(|&r| argmax(&want[r * vocab..(r + 1) * vocab]) == argmax(&ours[r * vocab..(r + 1) * vocab])).count();
    let out = rt.generate(&PROMPT, greedy.len(), Sampling::GREEDY).unwrap();
    let prefix_len = out.tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    let mean = kls.iter().sum::<f64>() / rows as f64;
    let max = kls.iter().copied().fold(0.0, f64::max);
    eprintln!("{}: mean KL {mean:.3e} max KL {max:.3e} top1 {top1}/{rows} greedy prefix {prefix_len}/{}", file().display(), greedy.len());
    let (mean_bound, max_bound, top1_min, greedy_min): (f64, f64, usize, usize) = std::env::var("SPECRT_GGUF_BOUNDS")
        .ok()
        .and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 4).then(|| (v[0], v[1], v[2] as usize, v[3] as usize))
        })
        .unwrap_or((5e-2, 0.5, rows * 8 / 10, 16));
    assert!(mean <= mean_bound && max <= max_bound, "mean KL {mean} max KL {max}");
    assert!(top1 >= top1_min, "top1 {top1}/{rows}");
    assert!(prefix_len >= greedy_min.min(greedy.len()), "greedy prefix {prefix_len}/{}", greedy.len());
}

#[test]
#[ignore = "needs a CUDA GPU and SPECRT_GGUF_FILE"]
fn a_qwen35_gguf_native_kernels_match_the_same_weights_dequantised_to_bf16() {
    let native = Spec::from_gguf_file(&file()).unwrap();
    let mut plain = native.clone();
    plain.quant = specrt::spec::Quant::Bf16;
    plain.head_quant = Some(specrt::spec::Quant::Bf16);
    plain.runtime.unsplit.clear();
    plain.runtime.big_tile = None;
    let logits = |s: &Spec| {
        let mut rt = Runtime::load(s, 256).unwrap();
        (rt.prompt_logits(&PROMPT).unwrap(), rt.generate(&PROMPT, 32, Sampling::GREEDY).unwrap().tokens)
    };
    let ((a, ga), (b, gb)) = (logits(&plain), logits(&native));
    let vocab = native.dims.vocab;
    let rows = PROMPT.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&a[r * vocab..(r + 1) * vocab], &b[r * vocab..(r + 1) * vocab])).collect();
    let (mean, max) = (kls.iter().sum::<f64>() / rows as f64, kls.iter().copied().fold(0.0, f64::max));
    let same = ga.iter().zip(&gb).take_while(|(x, y)| x == y).count();
    eprintln!("native against dequantised bf16: mean KL {mean:.3e} max KL {max:.3e} greedy prefix {same}/32");
    assert!(mean <= 1e-3 && max <= 1e-2, "mean {mean} max {max}");
    assert!(same >= 16, "greedy prefix {same}");
}

#[test]
#[ignore = "needs a CUDA GPU and the tiny fixtures; compares the GGUF path against the safetensors path of the same weights"]
fn a_tiny_qwen35_gguf_q8_0_matches_the_safetensors_logits_within_the_quant_bound() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut hf = Spec::from_hf_config(&std::fs::read_to_string(fixtures.join("tiny-qwen3.5-hd256/config.json")).unwrap()).unwrap();
    hf.repo = fixtures.join("tiny-qwen3.5-hd256").display().to_string();
    let gg = Spec::from_gguf_file(&fixtures.join("qwen35-gguf/tiny-mtp-Q8_0.gguf")).unwrap();
    let prompt: Vec<u32> = (0..48u32).map(|i| (i * 37 + 11) % 1000).collect();
    let a = Runtime::load(&hf, 128).unwrap().prompt_logits(&prompt).unwrap();
    let b = Runtime::load(&gg, 128).unwrap().prompt_logits(&prompt).unwrap();
    let vocab = hf.dims.vocab;
    let rows = prompt.len();
    let kls: Vec<f64> = (0..rows).map(|r| kl(&a[r * vocab..(r + 1) * vocab], &b[r * vocab..(r + 1) * vocab])).collect();
    let (mean, max) = (kls.iter().sum::<f64>() / rows as f64, kls.iter().copied().fold(0.0, f64::max));
    let top1 = (0..rows).filter(|&r| argmax(&a[r * vocab..(r + 1) * vocab]) == argmax(&b[r * vocab..(r + 1) * vocab])).count();
    eprintln!("tiny qwen35 Q8_0 gguf against safetensors: mean KL {mean:.3e} max KL {max:.3e} top1 {top1}/{rows}");
    assert!(mean <= 2e-3 && max <= 2e-2, "mean {mean} max {max}");
}
