use specrt::family::Registry as Families;
use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::PathBuf;

const NAME: &str = "nemotron-nano-9b-v2";

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("NEMOTRON_NANO_9B_DIR").expect("NEMOTRON_NANO_9B_DIR names the nvidia/NVIDIA-Nemotron-Nano-9B-v2 snapshot"))
}

fn spec() -> Spec {
    let dir = checkpoint();
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = dir.display().to_string();
    spec
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR from scripts/nemotron_nano_ref.py (float32)"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{NAME}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{NAME}.ref.bin"))).unwrap();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[test]
#[ignore = "needs a CUDA GPU with 24 GB free, NEMOTRON_NANO_9B_DIR and SPECRT_REF_DIR from scripts/nemotron_nano_ref.py (float32)"]
fn nemotron_nano_9b_matches_transformers() {
    let (tokens, greedy, vocab, want) = reference();
    let spec = spec();
    let mut rt = Runtime::load_batched(&spec, 512, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let agree = (0..tokens.len()).filter(|r| top(&ours[r * vocab..(r + 1) * vocab]) == top(&want[r * vocab..(r + 1) * vocab])).count();
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{NAME} prefill max_abs {max_abs} argmax agree {agree}/{}", tokens.len());
    assert!(agree * 100 >= tokens.len() * 93, "argmax agreement {agree}/{}", tokens.len());
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    eprintln!("{NAME} greedy ours {:?} reference {greedy:?}", got.tokens[0]);
    assert_eq!(got.tokens[0], greedy, "decode greedy against the fp32 reference");
}
