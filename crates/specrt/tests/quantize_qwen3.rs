use specrt::quantize::{LogitStats, logit_stats, prompt_set};
use specrt::runtime::Runtime;
use specrt::spec::{QuantizeChoice, Spec};
use std::path::Path;

fn spec(quantize: QuantizeChoice) -> Spec {
    let mut spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
    spec.runtime.quantize = quantize;
    spec.validate().unwrap();
    spec
}

fn prompts(spec: &Spec) -> Vec<(String, Vec<u32>)> {
    let path = specrt::weights::find_in_snapshot(&spec.repo, "tokenizer.json").unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    prompt_set().into_iter().map(|p| (p.name, tokenizer.encode(p.text.as_str(), false).unwrap().get_ids().to_vec())).collect()
}

struct Load {
    logits: Vec<Vec<f32>>,
    secs: f64,
    bytes_read: u64,
    weight_bytes: u64,
    tensors: u64,
    quantize_secs: f64,
}

fn load(spec: &Spec, prompts: &[(String, Vec<u32>)]) -> Load {
    let mut rt = Runtime::load(spec, 512).unwrap();
    let logits = prompts.iter().map(|(_, ids)| rt.prompt_logits(ids).unwrap()).collect();
    Load { logits, secs: rt.load_secs, bytes_read: rt.bytes_read, weight_bytes: rt.weight_bytes as u64, tensors: rt.quantized.tensors, quantize_secs: rt.quantized.secs }
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn quantize_at_load_stays_inside_its_kl_bound_on_qwen3_0_6b() {
    let base = spec(QuantizeChoice::Off);
    let vocab = base.dims.vocab;
    let prompts = prompts(&base);
    let lens: Vec<usize> = prompts.iter().map(|(_, ids)| ids.len()).collect();
    assert!(lens.iter().min().unwrap() * 8 < *lens.iter().max().unwrap(), "prompt lengths {lens:?}");
    load(&base, &prompts);
    let plain = load(&base, &prompts);
    eprintln!("off: load {:.2}s, read {} bytes, weights {} bytes", plain.secs, plain.bytes_read, plain.weight_bytes);
    let mut failures = Vec::new();
    for (choice, bound, floor) in [
        (QuantizeChoice::Fp8Channel, specrt::quantize::FP8_CHANNEL_MEAN_KL_BOUND, specrt::quantize::FP8_CHANNEL_TOP1_FLOOR),
        (QuantizeChoice::Nvfp4G16, specrt::quantize::NVFP4_G16_MEAN_KL_BOUND, specrt::quantize::NVFP4_G16_TOP1_FLOOR),
    ] {
        let q = load(&spec(choice), &prompts);
        assert!(q.tensors > 0 && q.weight_bytes < plain.weight_bytes);
        eprintln!("{choice:?}: load {:.2}s (search {:.2}s over {} tensors), read {} bytes, weights {} bytes", q.secs, q.quantize_secs, q.tensors, q.bytes_read, q.weight_bytes);
        for ((name, ids), (a, b)) in prompts.iter().zip(plain.logits.iter().zip(&q.logits)) {
            let s = logit_stats(&[a], &[b], vocab);
            eprintln!("{choice:?} {name} ({} tokens): mean KL {:.4}, max KL {:.4}, top-1 {:.1}%", ids.len(), s.mean_kl, s.max_kl, 100.0 * s.top1);
        }
        let all: LogitStats = logit_stats(&plain.logits.iter().map(Vec::as_slice).collect::<Vec<_>>(), &q.logits.iter().map(Vec::as_slice).collect::<Vec<_>>(), vocab);
        eprintln!("{choice:?} all {} positions: mean KL {:.4}, max KL {:.4}, top-1 {:.1}%", all.positions, all.mean_kl, all.max_kl, 100.0 * all.top1);
        if !(all.mean_kl.is_finite() && all.mean_kl <= bound && all.top1 >= floor) {
            failures.push(format!("{choice:?}: mean KL {} (bound {bound}), top-1 {} (floor {floor})", all.mean_kl, all.top1));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
