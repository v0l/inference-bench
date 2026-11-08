use specrt::runtime::Runtime;
use specrt::spec::{Quant, Spec};
use std::path::{Path, PathBuf};

fn out_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("out"))
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(out_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn relative(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs()));
    ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max) / scale
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

const NEAR_TIE: f32 = 0.25;

fn full_spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn write_floats(name: &str, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(out_dir().join(name), bytes).unwrap();
}

fn mode() -> &'static str {
    if std::env::var_os("SPECRT_FP8_ACT").is_some() { "act" } else { "weight" }
}

#[test]
#[ignore = "needs about 31 GB of free GPU memory and scripts/reference.py real-prompts output"]
fn real_27b_generates_greedy_and_dumps_logits() {
    let spec = full_spec();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir().join("real27b-prompts.json")).unwrap()).unwrap();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let mut rt = Runtime::load(&spec, 8192).unwrap();
    eprintln!(
        "loaded ({}): weights {:.2} GB in {:.1} s, device {:.2} GB, total load {:.1} s, peak host RSS {:.2} GB",
        mode(),
        rt.weight_bytes as f64 / 1e9,
        rt.weight_secs,
        rt.device_bytes as f64 / 1e9,
        rt.load_secs,
        rt.peak_host_rss_bytes as f64 / 1e9
    );
    let mut sequences = Vec::new();
    let mut logits = Vec::new();
    for p in meta["prompts"].as_array().unwrap() {
        let prompt = ids(p);
        let g = rt.generate(&prompt, new, specrt::runtime::Sampling::GREEDY).unwrap();
        assert_eq!(g.tokens.len(), new);
        let mut seq = prompt.clone();
        seq.extend(&g.tokens);
        sequences.push(seq);
    }
    let forced = std::fs::read_to_string(out_dir().join("real27b-specrt-weight.json"))
        .ok()
        .map(|t| serde_json::from_str::<serde_json::Value>(&t).unwrap())
        .map(|v| v["sequences"].as_array().unwrap().iter().map(ids).collect::<Vec<_>>());
    for seq in forced.as_ref().filter(|_| mode() == "act").unwrap_or(&sequences) {
        logits.extend(rt.prompt_logits(seq).unwrap());
    }
    write_floats(&format!("real27b-specrt-{}.logits.bin", mode()), &logits);
    let out = serde_json::json!({
        "sequences": sequences,
        "prompt_len": ids(&meta["prompts"][0]).len(),
        "load_secs": rt.load_secs,
        "weight_secs": rt.weight_secs,
        "weight_bytes": rt.weight_bytes,
        "device_bytes": rt.device_bytes,
        "peak_host_rss_bytes": rt.peak_host_rss_bytes,
    });
    std::fs::write(out_dir().join(format!("real27b-specrt-{}.json", mode())), out.to_string()).unwrap();
}

#[test]
#[ignore = "needs real27b_generates_greedy_and_dumps_logits and scripts/reference.py real-full output (both act and weight-only)"]
fn real_27b_greedy_and_logits_match_the_fp8_reference() {
    let vocab = full_spec().dims.vocab;
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(out_dir().join(name)).unwrap()).unwrap()
    };
    let weight_meta = read("real27b-specrt-weight.json");
    let prompt_len = weight_meta["prompt_len"].as_u64().unwrap() as usize;
    let forced: Vec<Vec<u32>> = weight_meta["sequences"].as_array().unwrap().iter().map(ids).collect();
    let weight_only = floats("real27b-ref.logits.bin");
    let act_ref = floats("real27b-ref-act.logits.bin");
    let envelope = relative(&act_ref, &weight_only);
    let len = forced[0].len();
    let first_miss = |seqs: &[Vec<u32>], reference: &[f32]| -> Vec<Result<usize, (usize, f32)>> {
        seqs.iter()
            .zip(&forced)
            .enumerate()
            .map(|(s, (seq, f))| {
                for p in prompt_len..len {
                    let at = (s * len + p - 1) * vocab;
                    let row = &reference[at..at + vocab];
                    let best = argmax(row);
                    if best != seq[p] as usize {
                        return Err((p - prompt_len, row[best] - row[seq[p] as usize]));
                    }
                    if seq[p] != f[p] {
                        return Ok(p + 1 - prompt_len);
                    }
                }
                Ok(len - prompt_len)
            })
            .collect()
    };
    let mut checked = 0;
    for (mode, reference) in [("weight", &weight_only), ("act", &act_ref)] {
        let Ok(text) = std::fs::read_to_string(out_dir().join(format!("real27b-specrt-{mode}.json"))) else { continue };
        let meta: serde_json::Value = serde_json::from_str(&text).unwrap();
        let seqs: Vec<Vec<u32>> = meta["sequences"].as_array().unwrap().iter().map(ids).collect();
        let ours = floats(&format!("real27b-specrt-{mode}.logits.bin"));
        let error = relative(&ours, reference);
        let to_weight_only = relative(&ours, &weight_only);
        let misses = first_miss(&seqs, reference);
        eprintln!(
            "{mode}: logits max error / max |logit| vs its reference {error}, vs weight-only {to_weight_only}; envelope {envelope}; greedy per prompt: Ok(n) = n tokens agree, Err((step, reference margin)) = first disagreement {misses:?}"
        );
        assert!(error <= envelope, "{mode}: {error} outside the {envelope} envelope");
        assert!(to_weight_only <= 1.5 * envelope, "{mode}: {to_weight_only} from the weight-only reference");
        if mode == "weight" {
            assert!(misses.iter().all(|m| *m == Ok(len - prompt_len)), "{misses:?}");
        } else {
            assert!(misses.iter().all(|m| m.as_ref().err().is_none_or(|e| e.1 <= NEAR_TIE)), "{misses:?}");
        }
        checked += 1;
    }
    assert!(checked > 0, "no specrt dumps");
}

#[test]
#[ignore = "needs a CUDA GPU, the Qwen3.8 27B FP8 checkpoint and scripts/reference.py real-slice output"]
fn real_27b_fp8_first_layers_match_transformers() {
    first_layers_match("qwen3.8-27b-slice", "real27b-slice", Quant::Fp8Block128);
}

#[test]
#[ignore = "needs a CUDA GPU, the NVFP4 27B checkpoint and scripts/reference.py real-slice output for it"]
fn real_27b_nvfp4_first_layers_match_the_dequantised_checkpoint() {
    first_layers_match("qwen3.8-27b-nvfp4-slice", "real27b-slice-nvfp4", Quant::Fp8Channel);
}

fn first_layers_match(fixture: &str, reference: &str, quant: Quant) {
    let dir = out_dir().join(fixture);
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    assert_eq!(spec.quant, quant);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir().join(format!("{reference}.json"))).unwrap()).unwrap();
    let tokens: Vec<u32> = meta["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mut rt = Runtime::load(&spec, 256).unwrap();
    eprintln!(
        "loaded {} layers: weights {:.2} GB in {:.2} s, device {:.2} GB, peak host RSS {:.2} GB",
        spec.layers.count,
        rt.weight_bytes as f64 / 1e9,
        rt.weight_secs,
        rt.device_bytes as f64 / 1e9,
        rt.peak_host_rss_bytes as f64 / 1e9
    );
    assert!(rt.device_bytes < 3_000_000_000, "{} bytes on the device", rt.device_bytes);
    let trace = rt.trace_prefill(&tokens).unwrap();
    let (n, hidden, vocab) = (tokens.len(), spec.dims.hidden, spec.dims.vocab);
    let theirs = floats(&format!("{reference}.layers.bin"));
    let per_layer: Vec<f32> = (0..spec.layers.count)
        .map(|l| relative(&trace.layers[l * n * hidden..(l + 1) * n * hidden], &theirs[l * n * hidden..(l + 1) * n * hidden]))
        .collect();
    let logits_ref = floats(&format!("{reference}.logits.bin"));
    let logits = relative(&trace.logits, &logits_ref);
    let top1 = (0..n)
        .filter(|&i| argmax(&trace.logits[i * vocab..(i + 1) * vocab]) == argmax(&logits_ref[i * vocab..(i + 1) * vocab]))
        .count();
    eprintln!("prefill {n} tokens: per-layer max error / max |x| {per_layer:?}, logits {logits}, top1 {top1}/{n}");
    assert!(per_layer.iter().all(|&e| e <= 2e-3), "{per_layer:?}");
    assert!(logits <= 2e-3, "{logits}");
    assert!(top1 >= n - 1, "{top1}/{n}");
    let decode = rt.trace_decode(&tokens).unwrap();
    let step_logits = relative(&decode.logits, &logits_ref);
    eprintln!("decode {n} steps: logits {step_logits}");
    assert!(step_logits <= 2e-3, "{step_logits}");
}
