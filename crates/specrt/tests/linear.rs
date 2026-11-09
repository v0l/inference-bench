use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

const PREFILL: usize = 1024;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_dir(dir: &str) -> PathBuf {
    root().join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec
}

fn spec() -> Spec {
    spec_for("tiny-qwen3.5")
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

fn decode_matches_reference(dir: &str, prefix: &str) {
    let s = spec_for(dir);
    let r = reference(&format!("{prefix}-decode"));
    let mut rt = Runtime::load(&s, 256).unwrap();
    let trace = rt.trace_decode(&r.tokens).unwrap();
    let steps = r.tokens.len();
    let layers = per_layer(&trace.layers, &floats(&format!("{prefix}-decode"), "layers"), 3, steps, 256, true);
    let logits = max_abs(&trace.logits, &floats(&format!("{prefix}-decode"), "logits"));
    let states = rt.linear_states().unwrap();
    let state = max_abs(&states.state, &floats(&format!("{prefix}-decode"), "state"));
    let la = s.linear_attention.as_ref().unwrap();
    let conv = max_abs(&states.conv, &last_taps(&floats(&format!("{prefix}-decode"), "conv"), la.conv_kernel));
    eprintln!("decode {steps} tokens: layers {layers:?} logits {logits} state {state} conv {conv}");
    assert!(layers.iter().all(|&d| d <= 1e-3), "{layers:?}");
    assert!(logits <= 1e-3, "{logits}");
    assert!(state <= 1e-4, "{state}");
    assert!(conv <= 1e-4, "{conv}");
}

fn prefill_matches_reference(dir: &str, prefix: &str) {
    let s = spec_for(dir);
    let r = reference(&format!("{prefix}-prefill"));
    assert_eq!(r.tokens.len(), PREFILL);
    let mut rt = Runtime::load(&s, 2048).unwrap();
    let trace = rt.trace_prefill(&r.tokens).unwrap();
    let layers = per_layer(&trace.layers, &floats(&format!("{prefix}-prefill"), "layers"), 3, PREFILL, 256, false);
    let logits = max_abs(&trace.logits, &floats(&format!("{prefix}-prefill"), "logits"));
    let states = rt.linear_states().unwrap();
    let state = max_abs(&states.state, &floats(&format!("{prefix}-prefill"), "state"));
    let la = s.linear_attention.as_ref().unwrap();
    let conv = max_abs(&states.conv, &last_taps(&floats(&format!("{prefix}-prefill"), "conv"), la.conv_kernel));
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

fn logits_and_greedy_match_reference(dir: &str, prefix: &str) {
    let s = spec_for(dir);
    let r = reference(&format!("{prefix}-greedy"));
    assert_eq!(r.greedy.len(), 64, "{}", r.meta);
    let mut rt = Runtime::load(&s, 256).unwrap();
    let ours = rt.prompt_logits(&r.tokens).unwrap();
    let theirs = floats(&format!("{prefix}-greedy"), "logits");
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
    for batch in [2, 4, 8, 16] {
        batch_decode_matches_single_sequence_greedy("tiny-qwen3.5", batch);
    }
}

const WIDE: &str = "tiny-qwen3.5-hd256";

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_decode_matches_the_recurrent_reference_per_layer() {
    decode_matches_reference("tiny-qwen3.5", "tiny");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_prefill_matches_the_chunked_reference_per_layer() {
    prefill_matches_reference("tiny-qwen3.5", "tiny");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn tiny_model_logits_and_greedy_match_transformers() {
    logits_and_greedy_match_reference("tiny-qwen3.5", "tiny");
}

#[test]
fn head_dim_256_fixture_spec_is_hybrid() {
    let s = spec_for(WIDE);
    assert_eq!(s.attention.head_dim, 256);
    assert_eq!(s.linear_layers(), 2);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn head_dim_256_decode_matches_the_reference_per_layer() {
    decode_matches_reference(WIDE, "tiny256");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn head_dim_256_prefill_matches_the_reference_per_layer() {
    prefill_matches_reference(WIDE, "tiny256");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny output"]
fn head_dim_256_logits_and_greedy_match_transformers() {
    logits_and_greedy_match_reference(WIDE, "tiny256");
}

const FP8: &str = "tiny-qwen3.5-fp8";

#[test]
fn fp8_fixture_is_sharded_with_block_scales_and_the_spec_says_so() {
    let s = spec_for(FP8);
    assert_eq!(s.quant, specrt::spec::Quant::Fp8Block128);
    let ck = specrt::weights::Checkpoint::open(&fixture_dir(FP8)).unwrap();
    assert_eq!(ck.files.len(), 4);
    let w = ck.view("model.layers.2.mlp.down_proj.weight").unwrap();
    assert_eq!(w.dtype, safetensors::Dtype::F8_E4M3);
    assert_eq!(w.shape, [256, 512]);
    let scale = ck.view("model.layers.2.mlp.down_proj.weight_scale_inv").unwrap();
    assert_eq!(scale.shape, [2, 4]);
    assert_eq!(ck.view("model.layers.0.linear_attn.in_proj_b.weight").unwrap().dtype, safetensors::Dtype::BF16);
    let layer = ck.layer(&s, 0).unwrap();
    assert!(matches!(layer.gate_up, specrt::weights::Matrix::Fp8 { .. }));
    let mut bf16 = s.clone();
    bf16.quant = specrt::spec::Quant::Bf16;
    assert!(ck.layer(&bf16, 0).is_err(), "an FP8 checkpoint under a bf16 spec is refused");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-fp8 output"]
fn fp8_decode_matches_the_dequantised_reference_per_layer() {
    decode_matches_reference(FP8, "tinyfp8");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-fp8 output"]
fn fp8_prefill_matches_the_dequantised_reference_per_layer() {
    prefill_matches_reference(FP8, "tinyfp8");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-fp8 output"]
fn fp8_logits_and_greedy_match_transformers() {
    logits_and_greedy_match_reference(FP8, "tinyfp8");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fp8_batch_decode_matches_single_sequence_greedy() {
    let s = spec_for(FP8);
    let single = Runtime::load(&s, 256).unwrap();
    assert!(single.peak_host_rss_bytes > 0 && single.weight_bytes > 0);
    drop(single);
    for batch in [2, 4, 8, 16, 17, 24, 32] {
        batch_decode_matches_single_sequence_greedy(FP8, batch);
    }
}

const NVFP4: &str = "tiny-qwen3.5-nvfp4";

#[test]
fn nvfp4_fixture_has_nvfp4_mlps_then_fp8_channel_and_the_spec_says_so() {
    let s = spec_for(NVFP4);
    assert_eq!(s.quant, specrt::spec::Quant::Fp8Channel);
    assert_eq!((s.mlp.quant, s.mlp.quant_until), (Some(specrt::spec::Quant::Nvfp4G16), Some(2)));
    let ck = specrt::weights::Checkpoint::open(&fixture_dir(NVFP4)).unwrap();
    let packed = ck.view("model.language_model.layers.0.mlp.down_proj.weight_packed").unwrap();
    assert_eq!((packed.dtype, packed.shape.as_slice()), (safetensors::Dtype::U8, &[256usize, 256][..]));
    for i in 0..2 {
        let layer = ck.layer(&s, i).unwrap();
        assert!(matches!(layer.gate_up, specrt::weights::Matrix::Fp4(_)));
        assert!(matches!(layer.down, specrt::weights::Matrix::Fp4(_)));
    }
    let last = ck.layer(&s, 2).unwrap();
    assert!(matches!(last.gate_up, specrt::weights::Matrix::Fp8 { .. }));
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-nvfp4 output"]
fn nvfp4_decode_matches_the_dequantised_reference_per_layer() {
    decode_matches_reference(NVFP4, "tinynvfp4");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-nvfp4 output"]
fn nvfp4_prefill_matches_the_dequantised_reference_per_layer() {
    prefill_matches_reference(NVFP4, "tinynvfp4");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/reference.py tiny-nvfp4 output"]
fn nvfp4_logits_and_greedy_match_transformers() {
    logits_and_greedy_match_reference(NVFP4, "tinynvfp4");
}

fn batch_decode_matches_single_sequence_greedy(dir: &str, batch: u32) {
    let s = spec_for(dir);
    let prompts: Vec<Vec<u32>> =
        (0..batch).map(|q| (0..40u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect();
    let mut single = Runtime::load(&s, 256).unwrap();
    let expected: Vec<Vec<u32>> =
        prompts.iter().map(|p| single.generate(p, 24, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let mut batched = Runtime::load_batched(&s, 256, batch as usize).unwrap();
    assert_eq!(batched.generate_batch(&prompts, 24).unwrap().tokens, expected, "batch {batch}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn nvfp4_batch_decode_matches_single_sequence_greedy() {
    batch_decode_matches_single_sequence_greedy(NVFP4, 4);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn nvfp4_batch_2_8_16_decode_matches_single_sequence_greedy() {
    for batch in [2, 8, 16, 17, 24, 32] {
        batch_decode_matches_single_sequence_greedy(NVFP4, batch);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_batch_decode_matches_single_sequence_greedy() {
    let s = spec_for(WIDE);
    let prompts: Vec<Vec<u32>> =
        (0..4u32).map(|q| (0..40u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect();
    let mut single = Runtime::load(&s, 256).unwrap();
    let expected: Vec<Vec<u32>> =
        prompts.iter().map(|p| single.generate(p, 24, Sampling::GREEDY).unwrap().tokens).collect();
    drop(single);
    let mut batch = Runtime::load_batched(&s, 256, 4).unwrap();
    assert_eq!(batch.generate_batch(&prompts, 24).unwrap().tokens, expected);
}

fn expected_fp8_weight_bytes() -> u64 {
    let mut total = 0u64;
    for entry in std::fs::read_dir(fixture_dir(FP8)).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        for (name, view) in st.tensors() {
            let elems = view.shape().iter().product::<usize>() as u64;
            total += match view.dtype() {
                safetensors::Dtype::F8_E4M3 => elems + 4 * view.shape()[0] as u64,
                _ if name.ends_with("in_proj_a.weight")
                    || name.ends_with("in_proj_b.weight")
                    || name == "lm_head.weight"
                    || name == "model.embed_tokens.weight" =>
                {
                    elems * 2
                }
                _ => elems * 4,
            };
        }
    }
    total
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn weight_bytes_is_the_sum_of_the_uploaded_buffers() {
    let rt = Runtime::load(&spec_for(FP8), 256).unwrap();
    assert_eq!(rt.weight_bytes, expected_fp8_weight_bytes());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn weight_bytes_ignore_device_memory_freed_by_another_thread() {
    use cudarc::driver::CudaContext;
    let expected = expected_fp8_weight_bytes();
    let churn = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let worker = {
        let churn = churn.clone();
        std::thread::spawn(move || {
            let ctx = CudaContext::new(0).unwrap();
            let stream = ctx.new_stream().unwrap();
            while churn.load(std::sync::atomic::Ordering::Relaxed) {
                let big = stream.alloc_zeros::<u8>(512 << 20).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(3));
                drop(big);
                std::thread::sleep(std::time::Duration::from_millis(3));
            }
        })
    };
    let s = spec_for(FP8);
    for _ in 0..5 {
        assert_eq!(Runtime::load(&s, 256).unwrap().weight_bytes, expected);
    }
    churn.store(false, std::sync::atomic::Ordering::Relaxed);
    worker.join().unwrap();
}
