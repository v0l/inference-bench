use specrt::quantcheck::{Budget, Options, Session, blanket_plan, run};
use specrt::quantize::logit_stats;
use specrt::runtime::Runtime;
use specrt::spec::{LayerQuant, Quant, QuantPlan, Spec};
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

fn prompts() -> Vec<Vec<u32>> {
    (0..3u32).map(|q| (0..(24 + 40 * q)).map(|i| (i * 7 + q * 131) % 1024).collect()).collect()
}

fn fresh_logits(spec: &Spec, plan: QuantPlan) -> (Vec<Vec<f32>>, u64) {
    let mut s = spec.clone();
    s.quant_plan = plan;
    let mut rt = Runtime::load(&s, 256).unwrap();
    (prompts().iter().map(|p| rt.prompt_logits(p).unwrap()).collect(), rt.weight_bytes)
}

#[test]
#[ignore = "needs a GPU"]
fn swapping_a_layer_in_place_gives_the_same_logits_as_loading_that_plan() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-moe"] {
        let spec = spec_for(dir);
        let formats: &[Quant] = if spec.moe.is_some() { &[Quant::Nvfp4G16] } else { &[Quant::Nvfp4G16, Quant::Fp8Channel] };
        let mut session = Session::open(&spec, formats, 256, Some(prompts())).unwrap();
        let n = spec.layers.count;
        let mut layers = vec![LayerQuant::BF16; n];
        layers[1] = LayerQuant::new(Quant::Nvfp4G16, false);
        layers[n - 1] = LayerQuant::new(*formats.last().unwrap(), true);
        let plan = QuantPlan { layers: layers.clone(), kernels: formats.to_vec() };
        let bytes = session.apply(&plan).unwrap();
        let swapped = session.logits().unwrap();
        let (fresh, fresh_bytes) = fresh_logits(&spec, plan);
        assert_eq!(bytes, fresh_bytes, "{dir}: weight bytes after swaps");
        assert_eq!(swapped, fresh, "{dir}: swapped layers must give the logits of a fresh load, bit for bit");
        let (planned_only, _) = fresh_logits(&spec, QuantPlan { layers, kernels: Vec::new() });
        assert_eq!(swapped, planned_only, "{dir}: the extra planned kernels change no logits");
        let back = session.apply(&QuantPlan::uniform(n, LayerQuant::BF16)).unwrap();
        assert!(back > bytes);
        let m = session.measure().unwrap();
        assert_eq!((m.mean_kl, m.max_kl, m.top1), (0.0, 0.0, 1.0), "{dir}: every layer back in bf16 is the reference");
        eprintln!("{dir}: swapped and fresh agree, {bytes} bytes quantised against {back} bf16");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn quantcheck_on_the_tiny_fixture_writes_a_recipe_that_meets_its_budget_when_checked_whole() {
    let spec = spec_for("tiny-qwen3.5");
    let opts = Options { formats: vec![Quant::Nvfp4G16, Quant::Fp8Channel], budgets: vec![Budget { kl: 0.004, bytes: None }, Budget { kl: 1.0, bytes: None }], max_ctx: 256, tokens: Some(prompts()) };
    let report = run(&spec, &opts, &mut |l| eprintln!("{l}")).unwrap();
    assert_eq!(report.probes.len(), spec.layers.count * 2 * 2);
    let (tight, loose) = (&report.recipes[0], &report.recipes[1]);
    let check = tight.check.unwrap();
    assert!(check.mean_kl <= 0.004, "whole model mean KL {} over the budget", check.mean_kl);
    assert!(tight.bytes < report.bf16_bytes);
    assert!(loose.layers.iter().all(|l| l.format == Quant::Nvfp4G16), "a loose budget leaves every layer at the lowest bits");
    assert!(loose.bytes <= tight.bytes);
    let mut s = spec.clone();
    s.quant_plan = tight.plan();
    let mut rt = Runtime::load(&s, 256).unwrap();
    let mut reference = Runtime::load(&spec, 256).unwrap();
    let a: Vec<Vec<f32>> = prompts().iter().map(|p| reference.prompt_logits(p).unwrap()).collect();
    let b: Vec<Vec<f32>> = prompts().iter().map(|p| rt.prompt_logits(p).unwrap()).collect();
    let fresh = logit_stats(&a.iter().map(Vec::as_slice).collect::<Vec<_>>(), &b.iter().map(Vec::as_slice).collect::<Vec<_>>(), spec.dims.vocab);
    assert_eq!(fresh.mean_kl, check.mean_kl, "loading the recipe reproduces the check quantcheck recorded");
    let blanket = blanket_plan(&spec, Quant::Nvfp4G16);
    assert_eq!(blanket.layers.iter().filter(|l| l.format == Quant::Bf16).count(), spec.layers.count.min(4));
}
