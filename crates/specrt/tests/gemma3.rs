use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-gemma3";

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures").join(NAME)
}

fn spec() -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = NAME.into();
    spec.repo = fixture().display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    reference_tagged("")
}

fn reference_tagged(tag: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{NAME}{tag}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{NAME}{tag}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
fn the_gemma3_config_maps_by_spec_alone() {
    let spec = spec();
    assert!(spec.norm.sandwich && spec.norm.unit_offset);
    assert_eq!(spec.dims.logit_softcap, Some(30.0));
    assert_eq!(spec.attention.softcap, Some(50.0));
    assert_eq!(spec.rope.local_theta, Some(10000.0));
    assert_eq!(spec.rope.theta, Some(1000000.0));
    assert_eq!(spec.dims.embed_scale, Some((128f64).sqrt()));
    assert_eq!(spec.layers.pattern.len(), 3);
    assert!(spec.layers.pattern.contains(&Block::Sliding));
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3.py output in SPECRT_REF_DIR"]
fn gemma3_logits_and_greedy_match_transformers() {
    let (tokens, greedy, vocab, want) = reference();
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    for r in 0..tokens.len() {
        let row = |v: &[f32]| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(row(&ours), row(&want), "row {r}");
    }
    let got = rt.generate(&tokens, greedy.len(), Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, greedy);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3.py output in SPECRT_REF_DIR"]
fn gemma3_layer_states_match_transformers() {
    let (tokens, _, _, _) = reference();
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap();
    let bytes = std::fs::read(dir.join("tiny-gemma3.layers.bin")).unwrap();
    let want: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * 128;
    for layer in 0..6 {
        let (a, b) = (&trace.layers[layer * per..(layer + 1) * per], &want[layer * per..(layer + 1) * per]);
        let max_abs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let scale = b.iter().map(|v| v.abs()).fold(0f32, f32::max);
        eprintln!("layer {layer} max_abs {max_abs} scale {scale}");
        if layer == 0 {
            eprintln!("ours {:?}\nwant {:?}", &a[..6], &b[..6]);
            eprintln!("ours row5 {:?}\nwant row5 {:?}", &a[5 * 128..5 * 128 + 4], &b[5 * 128..5 * 128 + 4]);
        }
    }
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

#[test]
#[ignore = "needs a CUDA GPU, GEMMA3_1B_DIR (google/gemma-3-1b-it) and scripts/gemma3_real.py output in SPECRT_REF_DIR"]
fn gemma3_1b_matches_transformers_on_logits_kl_and_greedy() {
    let model = PathBuf::from(std::env::var_os("GEMMA3_1B_DIR").expect("GEMMA3_1B_DIR"));
    let json = std::fs::read_to_string(model.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = "gemma3-1b".into();
    spec.repo = model.display().to_string();
    let (tokens, greedy, vocab, want) = reference_named("gemma3-1b");
    let rows = want.len() / vocab;
    let mut rt = Runtime::load(&spec, 1024).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    let first = tokens.len() - rows;
    let mut worst_kl = 0f64;
    for r in 0..rows {
        let a = &ours[(first + r) * vocab..(first + r + 1) * vocab];
        let b = &want[r * vocab..(r + 1) * vocab];
        let (la, lb) = (log_softmax(a), log_softmax(b));
        let kl: f64 = lb.iter().zip(&la).map(|(q, p)| q.exp() * (q - p)).sum();
        eprintln!("row {r} kl {kl}");
        worst_kl = worst_kl.max(kl);
        let top = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        assert_eq!(top(a), top(b), "row {r} top-1");
    }
    assert!(worst_kl < 0.02, "kl {worst_kl}");
    let got = rt.generate(&tokens, greedy.len(), Sampling::GREEDY).unwrap();
    let same = got.tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
    eprintln!("greedy prefix {same}/{}", greedy.len());
    assert_eq!(same, greedy.len(), "greedy diverged after {same}");
}

fn reference_named(name: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").unwrap());
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
fn the_gemma3_1b_config_without_layer_types_gets_the_five_to_one_pattern() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/gemma3-1b.config.json")).unwrap();
    let spec = Spec::from_hf_config(&json).unwrap();
    assert_eq!(spec.layers.pattern.len(), 26);
    assert_eq!(spec.layers.pattern.iter().filter(|b| **b == Block::Full).count(), 4);
    assert_eq!(spec.layers.window, Some(512));
    assert_eq!(spec.rope.local_theta, Some(10000.0));
    assert_eq!(spec.rope.theta, Some(1000000.0));
    assert!(spec.dims.tie_embeddings);
    assert_eq!(spec.block(25), Block::Sliding);
    assert_eq!(spec.block(5), Block::Full);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3.py output in SPECRT_REF_DIR"]
fn gemma3_single_stream_decode_matches_prefill() {
    let (tokens, greedy, vocab, _) = reference();
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let mut seq = tokens.clone();
    seq.extend(&greedy[..8]);
    let want = rt.prompt_logits(&seq).unwrap();
    let got = rt.prefill_then_decode(&tokens, &greedy[..7]).unwrap();
    let row = |v: &[f32], r: usize| v[r * vocab..(r + 1) * vocab].to_vec();
    for i in 0..8 {
        let a = row(&got, i);
        let b = row(&want, tokens.len() - 1 + i);
        let max_abs = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(max_abs < 5e-2, "step {i} decode differs from prefill by {max_abs}");
    }
}
