use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const PREFILL: usize = 1024;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures/tiny-qwen3.5")
}

fn spec() -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = "tiny-qwen3.5".into();
    spec.repo = fixture().display().to_string();
    spec
}

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"))
}

struct Reference {
    tokens: Vec<u32>,
    greedy: Vec<u32>,
    meta: serde_json::Value,
}

fn reference(name: &str) -> Reference {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir().join(format!("{name}.json"))).unwrap())
            .unwrap();
    let ids = |key: &str| -> Vec<u32> {
        meta[key].as_array().map_or(Vec::new(), |a| a.iter().map(|v| v.as_u64().unwrap() as u32).collect())
    };
    Reference { tokens: ids("tokens"), greedy: ids("greedy"), meta }
}

fn floats(name: &str, part: &str) -> Vec<f32> {
    let bytes = std::fs::read(ref_dir().join(format!("{name}.{part}.bin"))).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn per_layer(ours: &[f32], theirs: &[f32], layers: usize, rows: usize, hidden: usize, row_major: bool) -> Vec<f32> {
    (0..layers)
        .map(|l| {
            let pick = |v: &[f32]| -> Vec<f32> {
                (0..rows)
                    .flat_map(|r| {
                        let at = if row_major { (r * layers + l) * hidden } else { (l * rows + r) * hidden };
                        v[at..at + hidden].to_vec()
                    })
                    .collect()
            };
            max_abs(&pick(ours), &pick(theirs))
        })
        .collect()
}

fn last_taps(conv: &[f32], kernel: usize) -> Vec<f32> {
    conv.chunks_exact(kernel).flat_map(|c| c[1..].to_vec()).collect()
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
fn tiny_fixture_spec_is_hybrid() {
    let s = spec();
    assert_eq!(s.layers.pattern.len(), 3);
    assert_eq!(s.linear_layers(), 2);
    assert_eq!(s.rotary_dim(), 32);
    assert!(s.attention.output_gate);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_decode_matches_the_recurrent_reference_per_layer() {
    let s = spec();
    let r = reference("tiny-decode");
    let mut rt = Runtime::load(&s, 256).unwrap();
    let trace = rt.trace_decode(&r.tokens).unwrap();
    let steps = r.tokens.len();
    let layers = per_layer(&trace.layers, &floats("tiny-decode", "layers"), 3, steps, 256, true);
    let logits = max_abs(&trace.logits, &floats("tiny-decode", "logits"));
    let states = rt.linear_states().unwrap();
    let state = max_abs(&states.state, &floats("tiny-decode", "state"));
    let la = s.linear_attention.as_ref().unwrap();
    let conv = max_abs(&states.conv, &last_taps(&floats("tiny-decode", "conv"), la.conv_kernel));
    eprintln!("decode {steps} tokens: layers {layers:?} logits {logits} state {state} conv {conv}");
    assert!(layers.iter().all(|&d| d <= 1e-3), "{layers:?}");
    assert!(logits <= 1e-3, "{logits}");
    assert!(state <= 1e-4, "{state}");
    assert!(conv <= 1e-4, "{conv}");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_prefill_matches_the_chunked_reference_per_layer() {
    let s = spec();
    let r = reference("tiny-prefill");
    assert_eq!(r.tokens.len(), PREFILL);
    let mut rt = Runtime::load(&s, 2048).unwrap();
    let trace = rt.trace_prefill(&r.tokens).unwrap();
    let layers = per_layer(&trace.layers, &floats("tiny-prefill", "layers"), 3, PREFILL, 256, false);
    let logits = max_abs(&trace.logits, &floats("tiny-prefill", "logits"));
    let states = rt.linear_states().unwrap();
    let state = max_abs(&states.state, &floats("tiny-prefill", "state"));
    let la = s.linear_attention.as_ref().unwrap();
    let conv = max_abs(&states.conv, &last_taps(&floats("tiny-prefill", "conv"), la.conv_kernel));
    eprintln!("prefill {PREFILL} tokens: layers {layers:?} logits {logits} state {state} conv {conv}");
    assert!(layers.iter().all(|&d| d <= 1e-3), "{layers:?}");
    assert!(logits <= 1e-3, "{logits}");
    assert!(state <= 1e-4, "{state}");
    assert!(conv <= 1e-4, "{conv}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn chunked_prefill_hands_off_the_same_state_as_recurrent_decode() {
    let s = spec();
    let tokens: Vec<u32> = (0..PREFILL as u32).map(|i| (i.wrapping_mul(2654435761) >> 7) % 1024).collect();
    let mut rt = Runtime::load(&s, 2048).unwrap();
    rt.prompt_logits(&tokens).unwrap();
    let chunked = rt.linear_states().unwrap();
    rt.trace_decode(&tokens).unwrap();
    let recurrent = rt.linear_states().unwrap();
    let conv = max_abs(&chunked.conv, &recurrent.conv);
    let scale = recurrent.state.iter().fold(0f32, |m, v| m.max(v.abs()));
    let state = max_abs(&chunked.state, &recurrent.state);
    eprintln!("hand-off after {PREFILL} tokens: conv {conv} state {state} (state max {scale})");
    assert!(conv <= 1e-4, "{conv}");
    assert!(state <= 1e-4, "{state}");
    let tail: Vec<u32> = (0..16u32).map(|i| (i * 97 + 5) % 1024).collect();
    let split = rt.prefill_then_decode(&tokens, &tail).unwrap();
    let mut all = tokens.clone();
    all.extend(&tail);
    let whole = rt.prompt_logits(&all).unwrap();
    let vocab = s.dims.vocab;
    let from = (PREFILL - 1) * vocab;
    let d = max_abs(&split, &whole[from..from + split.len()]);
    eprintln!("prefill then decode vs one prefill: logits {d}");
    assert!(d <= 1e-3, "{d}");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_model_logits_and_greedy_match_transformers() {
    let s = spec();
    let r = reference("tiny-greedy");
    assert_eq!(r.greedy.len(), 64, "{}", r.meta);
    let mut rt = Runtime::load(&s, 256).unwrap();
    let ours = rt.prompt_logits(&r.tokens).unwrap();
    let theirs = floats("tiny-greedy", "logits");
    let d = max_abs(&ours, &theirs);
    let rows = r.tokens.len();
    let vocab = s.dims.vocab;
    let top1 = (0..rows)
        .filter(|&i| argmax(&ours[i * vocab..(i + 1) * vocab]) == argmax(&theirs[i * vocab..(i + 1) * vocab]))
        .count();
    eprintln!("tiny logits max_abs {d} top1 {top1}/{rows}");
    assert!(d <= 1e-3, "{d}");
    assert_eq!(top1, rows);
    let g = rt.generate(&r.tokens, r.greedy.len(), Sampling::GREEDY).unwrap();
    assert_eq!(g.tokens, r.greedy);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_batch_decode_matches_single_sequence_greedy() {
    let s = spec();
    let prompts: Vec<Vec<u32>> =
        (0..4u32).map(|q| (0..40u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect();
    let mut single = Runtime::load(&s, 256).unwrap();
    let expected: Vec<Vec<u32>> =
        prompts.iter().map(|p| single.generate(p, 24, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let mut batch = Runtime::load_batched(&s, 256, 4).unwrap();
    assert_eq!(batch.generate_batch(&prompts, 24).unwrap().tokens, expected);
}
