use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

struct Reference {
    tokens: Vec<u32>,
    greedy: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn reference(name: &str) -> Reference {
    let dir =
        std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap(),
    )
    .unwrap();
    let ids = |key: &str| -> Vec<u32> {
        meta[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    Reference {
        tokens: ids("tokens"),
        greedy: ids("greedy"),
        vocab: meta["vocab"].as_u64().unwrap() as usize,
        logits: bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

struct Diff {
    max_abs: f32,
    mean_abs: f32,
    top1_agreement: f32,
}

fn diff(ours: &[f32], reference: &Reference) -> Diff {
    assert_eq!(ours.len(), reference.logits.len());
    let mut max_abs = 0f32;
    let mut sum = 0f64;
    for (a, b) in ours.iter().zip(&reference.logits) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum += d as f64;
    }
    let rows = reference.tokens.len();
    let agree = (0..rows)
        .filter(|&r| {
            let span = r * reference.vocab..(r + 1) * reference.vocab;
            argmax(&ours[span.clone()]) == argmax(&reference.logits[span])
        })
        .count();
    let d = Diff {
        max_abs,
        mean_abs: (sum / ours.len() as f64) as f32,
        top1_agreement: agree as f32 / rows as f32,
    };
    eprintln!("max_abs {:.5} mean_abs {:.6} top1 {:.3}", d.max_abs, d.mean_abs, d.top1_agreement);
    d
}

fn check_bf16(spec_file: &str) {
    let spec = Spec::load(&root().join("specs").join(spec_file)).unwrap();
    let reference = reference(&spec.name);
    let mut rt = Runtime::load(&spec, 256).unwrap();
    let ours = rt.prompt_logits(&reference.tokens).unwrap();
    let d = diff(&ours, &reference);
    assert!(d.max_abs <= 1e-3, "max abs diff {}", d.max_abs);
    assert_eq!(d.top1_agreement, 1.0);
    let greedy = rt.generate(&reference.tokens, reference.greedy.len()).unwrap();
    assert_eq!(greedy.tokens, reference.greedy);
}

#[test]
#[ignore = "needs a CUDA GPU, the HF weights and scripts/reference.py output"]
fn qwen3_bf16_logits_match_transformers() {
    check_bf16("qwen3-0.6b.toml");
}

#[test]
#[ignore = "needs a CUDA GPU, the HF weights and scripts/reference.py output"]
fn qwen25_bf16_logits_match_transformers() {
    check_bf16("qwen2.5-0.5b.toml");
}

#[test]
#[ignore = "needs a CUDA GPU, the HF weights and scripts/reference.py output"]
fn qwen3_q8_logits_stay_close_to_transformers() {
    let spec = Spec::load(&root().join("specs/qwen3-0.6b-q8.toml")).unwrap();
    let reference = reference("qwen3-0.6b");
    let mut rt = Runtime::load(&spec, 256).unwrap();
    let d = diff(&rt.prompt_logits(&reference.tokens).unwrap(), &reference);
    assert!(d.mean_abs <= 0.1, "mean abs diff {}", d.mean_abs);
    assert!(d.top1_agreement >= 0.95, "top1 agreement {}", d.top1_agreement);
}
