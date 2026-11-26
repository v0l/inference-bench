use specrt::runtime::Runtime;
use specrt::spec::{QuantizeChoice, Spec};
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, quantize: QuantizeChoice) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.quantize = quantize;
    spec.quant_keep.first_layers = 0;
    spec.quant_keep.last_layers = 0;
    spec
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn mean_kl(a: &[f32], b: &[f32], vocab: usize) -> f64 {
    let rows = a.len() / vocab;
    (0..rows)
        .map(|r| {
            let (lp, lq) = (log_softmax(&a[r * vocab..(r + 1) * vocab]), log_softmax(&b[r * vocab..(r + 1) * vocab]));
            lp.iter().zip(&lq).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn prompts() -> Vec<Vec<u32>> {
    (0..4u32).map(|q| (0..96u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect()
}

fn check(dir: &str, quantize: QuantizeChoice, bound: f64) {
    let mut plain = Runtime::load(&spec_for(dir, QuantizeChoice::Off), 256).unwrap();
    let mut quant = Runtime::load(&spec_for(dir, quantize), 256).unwrap();
    assert_eq!(plain.quantized.tensors, 0);
    assert!(quant.quantized.tensors > 0 && quant.quantized.bf16_bytes > 0);
    assert!(quant.weight_bytes < plain.weight_bytes, "{} against {}", quant.weight_bytes, plain.weight_bytes);
    let vocab = spec_for(dir, QuantizeChoice::Off).dims.vocab;
    let kl = prompts().iter().map(|p| mean_kl(&plain.prompt_logits(p).unwrap(), &quant.prompt_logits(p).unwrap(), vocab)).sum::<f64>() / 4.0;
    eprintln!("{dir} {quantize:?}: kl {kl:.5}, weights {} against {} bytes, quantised {} tensors in {:.3}s", quant.weight_bytes, plain.weight_bytes, quant.quantized.tensors, quant.quantized.secs);
    assert!(kl.is_finite() && kl < bound, "kl {kl} over {bound}");
    let single: Vec<Vec<u32>> = prompts().iter().map(|p| quant.generate_batch(std::slice::from_ref(p), 8).unwrap().tokens[0].clone()).collect();
    drop(plain);
    drop(quant);
    let mut batched = Runtime::load_batched(&spec_for(dir, quantize), 256, 4).unwrap();
    let batch = batched.generate_batch(&prompts(), 8).unwrap().tokens;
    assert_eq!(batch, single);
}

#[test]
#[ignore = "needs a GPU"]
fn nvfp4_quantize_at_load_serves_the_tiny_fixture_close_to_bf16() {
    check("tiny-qwen3.5", QuantizeChoice::Nvfp4G16, 0.05);
}

#[test]
#[ignore = "needs a GPU"]
fn fp8_quantize_at_load_serves_the_tiny_fixture_close_to_bf16() {
    check("tiny-qwen3.5", QuantizeChoice::Fp8Channel, 0.01);
}

#[test]
#[ignore = "needs a GPU"]
fn nvfp4_quantize_at_load_serves_the_tiny_moe_fixture_close_to_bf16() {
    check("tiny-qwen3.5-moe", QuantizeChoice::Nvfp4G16, 0.05);
}
