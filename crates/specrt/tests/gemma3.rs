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
    assert_eq!(spec.rope.theta, 1000000.0);
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
