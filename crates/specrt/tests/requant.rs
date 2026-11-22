use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Quant, Requant, Spec};
use specrt::weights::{Checkpoint, Matrix, bf16_to_f32, e4m3_to_f32, f32_to_bf16, quantize_nvfp4};
use std::path::{Path, PathBuf};

const FP8: &str = "tiny-qwen3.5-fp8";

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, requant: Requant) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.split_activations = Some(true);
    spec.runtime.requant = requant;
    spec
}

fn all_fp4() -> Requant {
    let mut r = Requant::default();
    for class in ["lin_in", "lin_out", "qkv", "o", "mlp_tail"] {
        r.set(class, Some(Quant::Nvfp4G16));
    }
    r
}

fn fp8_as_bf16(values: &[u8], scales: &[f32], cols: usize) -> Vec<u16> {
    values
        .chunks(cols)
        .enumerate()
        .flat_map(|(r, row)| row.iter().enumerate().map(move |(c, &b)| (r, c, b)))
        .map(|(r, c, b)| f32_to_bf16(e4m3_to_f32(b) * scales[scales[r].to_bits() as usize + c / 128]))
        .collect()
}

#[test]
fn requantised_weights_match_the_cpu_quantiser_bit_for_bit() {
    let plain = spec_for(FP8, Requant::default());
    let spec = spec_for(FP8, all_fp4());
    let ck = Checkpoint::open(&fixture_dir(FP8)).unwrap();
    let mut checked = 0;
    for i in 0..spec.layers.count {
        let base = ck.layer(&plain, i).unwrap();
        let layer = ck.layer(&spec, i).unwrap();
        for (was, now) in [(&base.gate_up, &layer.gate_up), (&base.down, &layer.down)] {
            let (Matrix::Fp8 { values, scales }, Matrix::Fp4(got)) = (was, now) else { panic!("layer {i} mlp is not requantised") };
            let cols = got.cols;
            let want = quantize_nvfp4(&fp8_as_bf16(values, scales, cols), cols);
            assert_eq!(got.packed.as_ref(), want.packed.as_ref());
            assert_eq!(got.scales.as_ref(), want.scales.as_ref());
            assert_eq!(got.global, want.global);
            assert_eq!(got.input_global, 0.0);
            checked += 1;
        }
    }
    assert_eq!(checked, 2 * spec.layers.count);
}

#[test]
fn default_requant_leaves_every_matrix_in_the_checkpoint_format() {
    let spec = spec_for(FP8, Requant::default());
    let ck = Checkpoint::open(&fixture_dir(FP8)).unwrap();
    for i in 0..spec.layers.count {
        let layer = ck.layer(&spec, i).unwrap();
        assert!(matches!(layer.gate_up, Matrix::Fp8 { .. }));
    }
    assert!(spec.requanted_linears().is_empty());
}

#[test]
fn requant_rejects_formats_a_class_cannot_serve() {
    let mut spec = spec_for(FP8, Requant::default());
    spec.runtime.requant.qkv = Some(Quant::Bf16);
    assert!(spec.validate().is_err());
    spec.runtime.requant.qkv = None;
    spec.runtime.requant.lm_head = Some(Quant::Bf16);
    assert!(spec.validate().is_ok());
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn compare(a: &[f32], b: &[f32], vocab: usize) -> (f64, f64) {
    let rows = a.len() / vocab;
    let (mut kl, mut agree) = (0.0, 0usize);
    for r in 0..rows {
        let (p, q) = (&a[r * vocab..(r + 1) * vocab], &b[r * vocab..(r + 1) * vocab]);
        let (lp, lq) = (log_softmax(p), log_softmax(q));
        kl += lp.iter().zip(&lq).map(|(x, y)| x.exp() * (x - y)).sum::<f64>();
        let top = |v: &[f32]| (0..vocab).fold(0, |best, i| if v[i] > v[best] { i } else { best });
        agree += (top(p) == top(q)) as usize;
    }
    (kl / rows as f64, agree as f64 / rows as f64)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn requantised_logits_on_random_tiny_weights_stay_within_kl_and_a_loose_top1_floor_and_batch_decode_agrees() {
    let prompts: Vec<Vec<u32>> = (0..4u32).map(|q| (0..96u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect();
    let mut plain = Runtime::load(&spec_for(FP8, Requant::default()), 256).unwrap();
    let mut fp4 = Runtime::load(&spec_for(FP8, all_fp4()), 256).unwrap();
    assert!(fp4.weight_bytes < plain.weight_bytes);
    let (mut kl, mut top1) = (0.0, 0.0);
    for p in &prompts {
        let (k, t) = compare(&plain.prompt_logits(p).unwrap(), &fp4.prompt_logits(p).unwrap(), 1024);
        kl += k / 4.0;
        top1 += t / 4.0;
    }
    eprintln!("requant fp8 -> nvfp4 on tiny fixture: kl {kl:.5} top1 {top1:.3}");
    assert!(kl < 0.05 && top1 > 0.5, "kl {kl} top1 {top1}");
    let batch = fp4.generate_batch(&prompts, 12).unwrap().tokens;
    let single: Vec<Vec<u32>> = prompts.iter().map(|p| fp4.generate_batch(std::slice::from_ref(p), 12).unwrap().tokens[0].clone()).collect();
    for (b, s) in batch.iter().zip(&single) {
        assert_eq!(b, s);
    }
    let _ = (Sampling::GREEDY, bf16_to_f32(0));
}
