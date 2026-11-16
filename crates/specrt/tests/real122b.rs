use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;

fn spec() -> Spec {
    let dir = std::env::var("SPECRT_122B_DIR").expect("SPECRT_122B_DIR points at the Qwen3.5-122B-A10B-NVFP4 snapshot");
    let json = std::fs::read_to_string(format!("{dir}/config.json")).unwrap();
    let mut spec = if std::env::var("SPECRT_122B_ACT").is_ok_and(|v| v == "0") { Spec::from_hf_config(&json) } else { Spec::from_hf_config_serving(&json) }.unwrap();
    spec.name = "qwen3.5-122b-a10b-nvfp4".into();
    spec.repo = dir;
    spec
}

fn prompt() -> Vec<u32> {
    serde_json::from_str(&std::fs::read_to_string(std::env::var("SPECRT_122B_PROMPT").unwrap()).unwrap()).unwrap()
}

#[test]
#[ignore = "needs a 96 GB GPU, the Qwen3.5-122B-A10B-NVFP4 snapshot and SPECRT_122B_DIR"]
fn real_122b_generates_and_reports_speed() {
    let s = spec();
    let mut rt = Runtime::load(&s, 4608).unwrap();
    let ids = prompt();
    let g = rt.generate(&ids, 24, Sampling::GREEDY).unwrap();
    eprintln!("tokens {:?}", g.tokens);
    eprintln!("prefill {:.3}s decode {:.3}s for {} tokens", g.prefill_secs, g.decode_secs, g.tokens.len());
    let long: Vec<u32> = (0..4096u32).map(|i| ids[(i as usize) % ids.len()] + (i / ids.len() as u32) % 7).collect();
    let first = rt.generate(&long, 1, Sampling::GREEDY).unwrap();
    eprintln!("prefill 4096 tokens {:.3}s", first.prefill_secs);
    std::fs::write("/tmp/g122.json", serde_json::to_string(&g.tokens).unwrap()).unwrap();
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
    let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
    row.iter().map(|&v| v as f64 - m - z.ln()).collect()
}

#[test]
#[ignore = "needs a 96 GB GPU, the 122B snapshot, SPECRT_122B_DIR and SPECRT_122B_REF (vllm122.json from the vLLM reference script)"]
fn real_122b_top1_and_kl_match_vllm() {
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("SPECRT_122B_REF").unwrap()).unwrap()).unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
    let greedy: Vec<u32> = serde_json::from_value(reference["greedy"].clone()).unwrap();
    let rows = reference["logprobs"].as_array().unwrap();
    let full: Vec<u32> = ids.iter().chain(&greedy).copied().collect();
    let mut rt = Runtime::load(&spec(), 4608).unwrap();
    let logits = rt.prompt_logits(&full).unwrap();
    let vocab = logits.len() / full.len();
    let (mut agree, mut kl_sum, mut kl_max) = (0usize, 0f64, 0f64);
    for (n, row) in rows.iter().enumerate() {
        let at = ids.len() - 1 + n;
        let ours = log_softmax(&logits[at * vocab..(at + 1) * vocab]);
        let theirs: Vec<(usize, f64)> = row.as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_f64().unwrap())).collect();
        let top_theirs = theirs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        let top_ours = ours.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        agree += usize::from(top_ours == top_theirs);
        let mass: f64 = theirs.iter().map(|t| t.1.exp()).sum();
        let kl: f64 = theirs.iter().map(|&(t, lp)| (lp.exp() / mass) * (lp - mass.ln() - (ours[t] - ours_mass(&ours, &theirs)))).sum();
        eprintln!("pos {n} ours {top_ours} theirs {top_theirs} kl {kl:.4}");
        kl_sum += kl;
        kl_max = kl_max.max(kl);
    }
    let mean = kl_sum / rows.len() as f64;
    eprintln!("top1 agreement {agree}/{} mean KL {mean:.5} max KL {kl_max:.5}", rows.len());
    assert!(agree * 10 >= rows.len() * 9, "top-1 {agree}/{}", rows.len());
    assert!(mean <= 0.05, "mean KL {mean}");
}

fn ours_mass(ours: &[f64], theirs: &[(usize, f64)]) -> f64 {
    theirs.iter().map(|&(t, _)| ours[t].exp()).sum::<f64>().ln()
}

#[test]
#[ignore = "needs a 96 GB GPU, the 122B snapshot and SPECRT_122B_DIR"]
fn real_122b_batch_8_decode_speed() {
    let ids = prompt();
    let prompts: Vec<Vec<u32>> = (0..8u32).map(|j| ids[..12.min(ids.len())].iter().copied().chain([j]).collect()).collect();
    let mut rt = Runtime::load_batched(&spec(), 1024, 8).unwrap();
    rt.generate_batch(&prompts, 4).unwrap();
    let g = rt.generate_batch(&prompts, 64).unwrap();
    eprintln!("batch 8: prefill {:.3}s decode {:.3}s, {:.1} tok/s over {} tokens", g.prefill_secs, g.decode_secs, 8.0 * 64.0 / g.decode_secs, 8 * 64);
}

fn read_floats(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
#[ignore = "needs a 96 GB GPU, the 122B snapshot, SPECRT_122B_DIR and SPECRT_122B_TRACE (scripts/moe_trace.py output dir)"]
fn real_122b_per_layer_trace_matches_the_transformers_oracle() {
    let dir = std::env::var("SPECRT_122B_TRACE").unwrap();
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/moe122-ref.json")).unwrap()).unwrap();
    let full_ids: Vec<u32> = serde_json::from_value(meta["ids"].clone()).unwrap();
    let ids: Vec<u32> = full_ids[..std::env::var("SPECRT_122B_TRACE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(full_ids.len())].to_vec();
    let mut s = spec();
    s.mlp.act_quant = specrt::spec::ActQuant::None;
    let mut rt = Runtime::load(&s, 4608).unwrap();
    let trace = rt.trace_prefill(&ids).unwrap();
    let theirs = read_floats(&format!("{dir}/moe122-ref.layers.bin"));
    if let Ok(path) = std::env::var("SPECRT_122B_DUMP") {
        let bytes: Vec<u8> = trace.layers.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }
    let (n, h) = (ids.len(), s.dims.hidden);
    let full_n = full_ids.len();
    let theirs: Vec<f32> = (0..s.layers.count).flat_map(|l| theirs[l * full_n * h..l * full_n * h + n * h].to_vec()).collect();
    let relative = |a: &[f32], b: &[f32]| {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
    };
    let per_layer: Vec<f32> = (0..s.layers.count).map(|l| relative(&trace.layers[l * n * h..(l + 1) * n * h], &theirs[l * n * h..(l + 1) * n * h])).collect();
    eprintln!("per-layer max error / max |x|: {per_layer:?}");
    if let Some(l) = per_layer.iter().position(|&e| e > 1e-2) {
        let per_token: Vec<f32> = (0..n)
            .map(|t| {
                let (a, b) = (&trace.layers[(l * n + t) * h..(l * n + t + 1) * h], &theirs[(l * n + t) * h..(l * n + t + 1) * h]);
                relative(a, b)
            })
            .collect();
        eprintln!("layer {l} per-token error: {per_token:?}");
    }
    let typical: Vec<f32> = (0..s.layers.count)
        .map(|l| {
            let scale = theirs[l * n * h..(l + 1) * n * h].iter().fold(0f32, |m, v| m.max(v.abs()));
            let mut per_token: Vec<f32> = (0..n)
                .map(|t| {
                    let (a, b) = (&trace.layers[(l * n + t) * h..(l * n + t + 1) * h], &theirs[(l * n + t) * h..(l * n + t + 1) * h]);
                    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
                })
                .collect();
            per_token.sort_by(f32::total_cmp);
            per_token[n * 9 / 10]
        })
        .collect();
    eprintln!("per-layer 90th percentile token error: {typical:?}");
    let first_bad = typical.iter().position(|&e| e > 1e-2);
    assert!(first_bad.is_none(), "first layer with a typical token error above 1e-2: {first_bad:?}");
}
