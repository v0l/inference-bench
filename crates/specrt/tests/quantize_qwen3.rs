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
fn a_quantcheck_recipe_beats_blanket_nvfp4_on_top1_at_a_similar_size_on_qwen3_0_6b() {
    use specrt::quantcheck::{Budget, Options, run};
    use specrt::spec::{Quant, QuantPlan};
    let base = spec(QuantizeChoice::Off);
    let opts = Options { formats: vec![Quant::Nvfp4G16, Quant::Fp8Channel], budgets: vec![Budget { kl: 0.1, bytes: None }], max_ctx: 512, tokens: None };
    let report = run(&base, &opts, &mut |l| eprintln!("{l}")).unwrap();
    let recipe = &report.recipes[0];
    let check = recipe.check.unwrap();
    let (_, blanket, blanket_bytes) = *report.blanket.iter().find(|(q, _, _)| *q == Quant::Nvfp4G16).unwrap();
    eprintln!("recipe: {} bytes, mean KL {:.4}, max KL {:.3}, top-1 {:.1}%; blanket nvfp4: {blanket_bytes} bytes, mean KL {:.4}, max KL {:.3}, top-1 {:.1}%", recipe.bytes, check.mean_kl, check.max_kl, 100.0 * check.top1, blanket.mean_kl, blanket.max_kl, 100.0 * blanket.top1);
    assert!(recipe.bytes <= blanket_bytes + blanket_bytes / 20, "the recipe is within 5% of the blanket size");
    assert!(check.top1 >= blanket.top1 + 0.03, "top-1 {} against {}", check.top1, blanket.top1);
    assert!(check.mean_kl <= 0.1 && check.mean_kl < blanket.mean_kl && check.max_kl < blanket.max_kl);
    let text = recipe.to_toml().unwrap();
    assert!(text.len() < 8192);
    let mut planned = base.clone();
    planned.quant_plan = QuantPlan { layers: recipe.plan().layers, kernels: Vec::new() };
    let prompts = prompts(&planned);
    let fresh = load(&planned, &prompts);
    assert_eq!(fresh.weight_bytes, recipe.bytes, "a fresh load of the recipe places the bytes quantcheck counted");
    let reference = load(&base, &prompts);
    let a: Vec<&[f32]> = reference.logits.iter().map(Vec::as_slice).collect();
    let b: Vec<&[f32]> = fresh.logits.iter().map(Vec::as_slice).collect();
    let s = logit_stats(&a, &b, base.dims.vocab);
    assert_eq!((s.mean_kl, s.top1), (check.mean_kl, check.top1), "loading the recipe reproduces the check");
    eprintln!("recipe load {:.2}s (search and plain rounding {:.2}s over {} tensors)", fresh.secs, fresh.quantize_secs, fresh.tensors);
    let shortest = prompts.iter().map(|(_, ids)| ids.len()).min().unwrap();
    let ids: Vec<Vec<u32>> = prompts.iter().map(|(_, ids)| ids[..shortest].to_vec()).collect();
    let mut single = Runtime::load(&planned, 512).unwrap();
    let one: Vec<Vec<u32>> = ids.iter().map(|p| single.generate_batch(std::slice::from_ref(p), 8).unwrap().tokens[0].clone()).collect();
    drop(single);
    let mut batched = Runtime::load_batched(&planned, 512, ids.len()).unwrap();
    assert_eq!(batched.generate_batch(&ids, 8).unwrap().tokens, one, "batched decode of a mixed fp8 and nvfp4 model equals single stream");
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
