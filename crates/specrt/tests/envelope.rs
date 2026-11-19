use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const FAMILIES: [(&str, &str, &str); 5] = [
    ("llama-3.2-1b", "llama-3.2-1b.config.json", "llama-3.2-1b"),
    ("gemma-3-1b-it", "gemma3-1b.config.json", "gemma-3-1b-it"),
    ("granite-3.3-2b-base", "granite-3.3-2b.config.json", "granite-3.3-2b-base"),
    ("SmolLM3-3B", "smollm3-3b.config.json", "SmolLM3-3B"),
    ("Phi-3-mini-4k-instruct", "phi3-mini.config.json", "Phi-3-mini-4k-instruct"),
];

const MARGIN: f64 = 1e-4;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn recorded() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("tests/golden/envelope.json")).unwrap()).unwrap()
}

fn config_spec(config: &str) -> Spec {
    Spec::from_hf_config(&std::fs::read_to_string(root().join("tests/fixtures").join(config)).unwrap()).unwrap()
}

#[test]
fn recorded_references_put_single_plane_inside_vllms_distance_from_fp32() {
    let rec = recorded();
    for (name, _, _) in FAMILIES {
        let r = &rec[name];
        let ours = r["unsplit"]["mean_kl"].as_f64().unwrap();
        let vllm = r["vllm"]["mean_kl"].as_f64().unwrap();
        assert!(ours <= vllm + MARGIN, "{name}: single plane KL {ours:e} against vllm {vllm:e}");
        let top1 = |k: &str| r[k]["top1"].as_u64().unwrap();
        assert!(top1("unsplit") >= top1("vllm"), "{name}: top1 {} against vllm {}", top1("unsplit"), top1("vllm"));
    }
}

#[test]
fn single_plane_is_the_default_for_the_measured_families_and_the_exact_path_stays_a_runtime_option() {
    for (name, config, _) in FAMILIES {
        let mut spec = config_spec(config);
        assert!(!spec.split_activations(), "{name}");
        spec.runtime.split_activations = Some(true);
        assert!(spec.split_activations(), "{name}");
    }
    let explicit = Spec::load(&root().join("specs/llama-3.2-1b.toml")).unwrap();
    assert!(explicit.split_activations());
}

fn kl_against(reference: &[f32], ours: &[f32], vocab: usize, skip: &[usize]) -> f64 {
    let rows = reference.len() / vocab;
    let logsm = |row: &[f32]| -> Vec<f64> {
        let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
        row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
    };
    let (mut total, mut n) = (0f64, 0usize);
    for r in (0..rows).filter(|r| !skip.contains(r)) {
        let span = r * vocab..(r + 1) * vocab;
        let (p, q) = (logsm(&reference[span.clone()]), logsm(&ours[span]));
        total += p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
        n += 1;
    }
    total / n as f64
}

fn envelope_dir() -> PathBuf {
    std::env::var_os("SPECRT_ENVELOPE_DIR").map(PathBuf::from).expect("SPECRT_ENVELOPE_DIR holds scripts/envelope.sh output")
}

fn models_dir() -> PathBuf {
    std::env::var_os("SPECRT_ENVELOPE_MODELS").map(PathBuf::from).expect("SPECRT_ENVELOPE_MODELS holds the checkpoints")
}

fn measure(name: &str, dir: &str, split: Option<bool>) -> f64 {
    let env = envelope_dir().join(name);
    let prompts: Vec<Vec<u32>> = serde_json::from_str(&std::fs::read_to_string(env.join("prompts.json")).unwrap()).unwrap();
    let model = models_dir().join(dir);
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(model.join("config.json")).unwrap()).unwrap();
    spec.name = name.into();
    spec.repo = model.display().to_string();
    spec.runtime.split_activations = split;
    let mut rt = Runtime::load(&spec, prompts.iter().map(Vec::len).max().unwrap() + 8).unwrap();
    let mut ours = Vec::new();
    for p in &prompts {
        ours.extend(rt.prompt_logits(p).unwrap());
    }
    let bytes = std::fs::read(env.join("fp32.f32")).unwrap();
    let reference: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(reference.len(), ours.len());
    let vocab = spec.dims.vocab;
    let mut first = Vec::new();
    let mut at = 0;
    for p in &prompts {
        first.push(at);
        at += p.len();
    }
    kl_against(&reference, &ours, vocab, &first)
}

#[test]
#[ignore = "needs a CUDA GPU, the checkpoints and scripts/envelope.sh output"]
fn single_plane_kl_against_fp32_is_no_worse_than_vllms_for_every_measured_family() {
    let rec = recorded();
    for (name, _, dir) in FAMILIES {
        let kl = measure(name, dir, None);
        let vllm = rec[name]["vllm"]["mean_kl"].as_f64().unwrap();
        eprintln!("{name}: single plane KL {kl:e}, vllm {vllm:e}");
        assert!(kl <= vllm + MARGIN, "{name}: {kl:e} against vllm {vllm:e}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU, the checkpoints and scripts/envelope.sh output"]
fn split_activations_option_keeps_the_exact_path_at_fp32_distance() {
    for (name, _, dir) in FAMILIES {
        let kl = measure(name, dir, Some(true));
        eprintln!("{name}: exact KL {kl:e}");
        assert!(kl < 1e-8, "{name}: {kl:e}");
    }
}

#[test]
fn qwen3_5_runs_single_plane_activations_by_default() {
    let spec = config_spec("qwen3.5-4b-base.config.json");
    assert!(!spec.split_activations());
}
