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

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn tokens(name: &str) -> Vec<u32> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir().join(name)).unwrap()).unwrap();
    meta["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore = "needs a CUDA GPU with FP8 mma and scripts/reference.py tiny-fp8 output"]
fn native_fp8_prefill_stays_inside_the_activation_quantisation_envelope() {
    unsafe { std::env::set_var("SPECRT_FP8_ACT", "1") };
    let dir = root().join("tests/fixtures/tiny-qwen3.5-fp8");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    let ids = tokens("tinyfp8-greedy.json");
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let ours = rt.prompt_logits(&ids).unwrap();
    let weight_only = floats("tinyfp8-greedy.logits.bin");
    let act_quantised = floats("tinyfp8act-greedy.logits.bin");
    let envelope = max_abs(&act_quantised, &weight_only);
    let error = max_abs(&ours, &weight_only);
    let vocab = spec.dims.vocab;
    let rows = ids.len();
    let top1 = (0..rows)
        .filter(|&i| argmax(&ours[i * vocab..(i + 1) * vocab]) == argmax(&weight_only[i * vocab..(i + 1) * vocab]))
        .count();
    let ref_top1 = (0..rows)
        .filter(|&i| {
            argmax(&act_quantised[i * vocab..(i + 1) * vocab]) == argmax(&weight_only[i * vocab..(i + 1) * vocab])
        })
        .count();
    eprintln!("native fp8 logits vs weight-only: {error} (envelope {envelope}), top1 {top1}/{rows} (reference {ref_top1}/{rows})");
    assert!(error <= 1.5 * envelope, "{error} outside the {envelope} envelope");
    assert!(top1 + 2 >= ref_top1, "{top1} against {ref_top1}");
}
