use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-qwen3-sliding";

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
    spec
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{NAME}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{NAME}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
fn the_sliding_config_maps_to_sliding_and_full_blocks() {
    let spec = spec();
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Full]);
    assert_eq!(spec.layers.window, Some(16));
    assert_eq!(specrt::plan::Plan::build(&spec, specrt::plan::Phase::Decode, true).layer(Block::Sliding).len(), 7);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_sliding.py output in SPECRT_REF_DIR"]
fn sliding_window_logits_and_greedy_match_transformers() {
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
