use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"))
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(ref_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn rms(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt() as f32
}

fn tokens(name: &str) -> Vec<u32> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir().join(name)).unwrap()).unwrap();
    meta["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore = "needs an sm_120 GPU and scripts/reference.py tiny-nvfp4 output"]
fn native_fp4_prefill_matches_the_activation_quantised_reference() {
    let dir = root().join("tests/fixtures/tiny-qwen3.5-nvfp4");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec.mlp.act_quant = specrt::spec::ActQuant::Fp4;
    spec.validate().unwrap();
    let ids = tokens("tinynvfp4act-prefill.json");
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    assert_eq!(rt.arch, "compute_120");
    let ours = rt.prompt_logits(&ids).unwrap();
    let trace = rt.trace_prefill(&ids).unwrap();
    let theirs = floats("tinynvfp4act-prefill.layers.bin");
    let weight_layers = floats("tinynvfp4-prefill.layers.bin");
    let n = ids.len() * spec.dims.hidden;
    let per_layer: Vec<(f32, f32)> = (0..spec.layers.count)
        .map(|l| {
            let r = l * n..(l + 1) * n;
            (rms(&trace.layers[r.clone()], &theirs[r.clone()]), rms(&weight_layers[r.clone()], &theirs[r]))
        })
        .collect();
    eprintln!("per layer rms (ours vs act reference, weight-only vs act reference): {per_layer:?}");
    let weight_only = floats("tinynvfp4-prefill.logits.bin");
    let act_quantised = floats("tinynvfp4act-prefill.logits.bin");
    let envelope = max_abs(&act_quantised, &weight_only);
    let to_act = max_abs(&ours, &act_quantised);
    let to_weight = max_abs(&ours, &weight_only);
    let logits_rms = (rms(&ours, &act_quantised), rms(&weight_only, &act_quantised));
    eprintln!("native fp4, {} tokens: max abs vs act-quantised {to_act}, vs weight-only {to_weight}, envelope {envelope}; logits rms vs act-quantised {logits_rms:?}", ids.len());
    assert!(per_layer[0].0 <= 0.1 * per_layer[0].1, "layer 0 rms: {:?}", per_layer[0]);
    assert!(logits_rms.0 <= 0.5 * logits_rms.1, "logits rms {logits_rms:?}");
    assert!(to_weight <= 1.25 * envelope, "{to_weight} from weight-only, envelope {envelope}");
}
