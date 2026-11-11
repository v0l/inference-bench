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

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn mean_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows)
        .map(|r| {
            let (a, b) = (log_softmax(&p[r * vocab..(r + 1) * vocab]), log_softmax(&q[r * vocab..(r + 1) * vocab]));
            a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn top1_agreement(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows).filter(|&r| argmax(&p[r * vocab..(r + 1) * vocab]) == argmax(&q[r * vocab..(r + 1) * vocab])).count() as f64
        / rows as f64
}

fn full_spec() -> Spec {
    named_spec("qwen3.8-27b-exact")
}

fn named_spec(default: &str) -> Spec {
    let name = std::env::var("SPECRT_27B_SPEC").unwrap_or_else(|_| default.into());
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml"))).unwrap().with_env_overrides()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn write_floats(name: &str, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(out_dir().join(name), bytes).unwrap();
}

fn mode(spec: &Spec) -> String {
    use specrt::spec::ActQuant;
    let base = if spec.act_quant != ActQuant::None || spec.mlp.act_quant != ActQuant::None { "act" } else { "weight" };
    match spec.head_quant {
        Some(h) if h != Quant::Bf16 => format!("{base}-head-{}", head_name(h)),
        _ => base.into(),
    }
}

fn head_name(q: Quant) -> &'static str {
    match q {
        Quant::Fp8Channel => "fp8_channel",
        Quant::Nvfp4G16 => "nvfp4_g16",
        _ => "bf16",
    }
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
        mode(&spec),
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
    for seq in forced.as_ref().filter(|_| mode(&spec) != "weight").unwrap_or(&sequences) {
        logits.extend(rt.prompt_logits(seq).unwrap());
    }
    write_floats(&format!("real27b-specrt-{}.logits.bin", mode(&spec)), &logits);
    let out = serde_json::json!({
        "sequences": sequences,
        "prompt_len": ids(&meta["prompts"][0]).len(),
        "load_secs": rt.load_secs,
        "weight_secs": rt.weight_secs,
        "weight_bytes": rt.weight_bytes,
        "device_bytes": rt.device_bytes,
        "peak_host_rss_bytes": rt.peak_host_rss_bytes,
    });
    std::fs::write(out_dir().join(format!("real27b-specrt-{}.json", mode(&spec))), out.to_string()).unwrap();
}

#[test]
#[ignore = "needs about 25 GB of free GPU memory, the NVFP4 checkpoint and scripts/vllm_logprobs.py output"]
fn real_27b_nvfp4_agrees_with_vllm_greedy_and_logprobs() {
    let spec = named_spec("qwen3.8-27b-nvfp4");
    assert_eq!(spec.mlp.quant, Some(Quant::Nvfp4G16), "SPECRT_27B_SPEC must name an NVFP4 spec for the vLLM comparison");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir().join("vllm-nvfp4.json")).unwrap()).unwrap();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let records = meta["records"].as_array().unwrap();
    let mut rt = Runtime::load(&spec, 8192).unwrap();
    eprintln!(
        "loaded ({}): weights {:.2} GB in {:.1} s, device {:.2} GB, load {:.1} s, peak host RSS {:.2} GB",
        mode(&spec),
        rt.weight_bytes as f64 / 1e9,
        rt.weight_secs,
        rt.device_bytes as f64 / 1e9,
        rt.load_secs,
        rt.peak_host_rss_bytes as f64 / 1e9
    );
    let vocab = spec.dims.vocab;
    let mut agree_prefix = Vec::new();
    let mut agree_positions = 0usize;
    let mut kls = Vec::new();
    let mut top1 = 0usize;
    let mut margins: Vec<f32> = Vec::new();
    for r in records {
        let prompt = ids(&r["prompt"]);
        let theirs = ids(&r["greedy"]);
        let ours = rt.generate(&prompt, new, specrt::runtime::Sampling::GREEDY).unwrap().tokens;
        let prefix = ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
        agree_prefix.push(prefix);
        if prefix < new {
            let row = r["logprobs"][prefix].as_object().unwrap();
            let lp = |t: u32| row.get(&t.to_string()).and_then(|v| v.as_f64());
            margins.push((lp(theirs[prefix]).unwrap_or(f64::NAN) - lp(ours[prefix]).unwrap_or(f64::NEG_INFINITY)) as f32);
        }
        let mut seq = prompt.clone();
        seq.extend(&theirs);
        let logits = rt.prompt_logits(&seq).unwrap();
        for (step, row) in r["logprobs"].as_array().unwrap().iter().enumerate() {
            let at = (prompt.len() + step - 1) * vocab;
            let l = &logits[at..at + vocab];
            let max = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = max + l.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
            let top: Vec<(usize, f64)> =
                row.as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_f64().unwrap())).collect();
            let p_mass: f64 = top.iter().map(|(_, lp)| lp.exp()).sum();
            let mut kl = 0.0;
            for &(t, lp) in &top {
                let ours_lp = (l[t] - lse) as f64;
                kl += lp.exp() * (lp - ours_lp);
            }
            let rest_p = (1.0 - p_mass).max(1e-12);
            let rest_q = (1.0 - top.iter().map(|&(t, _)| ((l[t] - lse) as f64).exp()).sum::<f64>()).max(1e-12);
            kl += rest_p * (rest_p / rest_q).ln();
            kls.push(kl);
            if argmax(l) == theirs[step] as usize {
                top1 += 1;
            }
            if ours.get(step) == Some(&theirs[step]) {
                agree_positions += 1;
            }
        }
    }
    let total = records.len() * new;
    let mut sorted = kls.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let first5: usize = agree_prefix.iter().take(5).sum();
    let report = serde_json::json!({
        "mode": mode(&spec),
        "prompts": records.len(),
        "new_tokens": new,
        "greedy_prefix": agree_prefix,
        "greedy_prefix_first5": first5,
        "vllm_margin_at_first_divergence": margins,
        "greedy_positions_equal": agree_positions,
        "teacher_forced_top1": top1,
        "positions": total,
        "kl_mean": mean,
        "kl_median": sorted[sorted.len() / 2],
        "kl_p99": sorted[sorted.len() * 99 / 100],
        "kl_max": sorted[sorted.len() - 1],
    });
    eprintln!("{report}");
    std::fs::write(out_dir().join(format!("vllm-compare-{}.json", mode(&spec))), report.to_string()).unwrap();
    assert!(top1 * 10 >= total * 9, "teacher-forced top-1 {top1}/{total}");
    assert!(mean <= 0.05, "mean KL {mean}");
}

#[test]
#[ignore = "needs the NVFP4 and FP8 27B checkpoints in the HF cache"]
fn real_27b_mtp_weights_load_from_both_checkpoints_with_fp8_dequantised_to_bf16() {
    use specrt::weights::{Checkpoint, Matrix, Mixer};
    for name in ["qwen3.8-27b-nvfp4-exact", "qwen3.8-27b-exact"] {
        let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml"))).unwrap();
        let ck = Checkpoint::for_repo(&spec.repo).unwrap();
        let m = ck.mtp(&spec).unwrap();
        let h = spec.dims.hidden;
        let len = |m: &Matrix| match m {
            Matrix::Bf16(w) => w.len(),
            _ => panic!("MTP matrices are bf16"),
        };
        assert_eq!(len(&m.fc), h * 2 * h);
        assert_eq!(len(&m.layer.gate_up), 2 * spec.dims.intermediate * h);
        assert_eq!(len(&m.layer.down), h * spec.dims.intermediate);
        let Mixer::Attention(a) = &m.layer.mixer else { panic!("the MTP layer is full attention") };
        assert_eq!(len(&a.qkv), spec.qkv_rows() * h);
        assert_eq!(len(&a.o), h * spec.q_dim());
        assert!(m.norm.iter().all(|v| v.is_finite()) && m.norm_embed.len() == h && m.norm_hidden.len() == h);
        assert!((m.norm.iter().sum::<f32>() / h as f32 - 2.25).abs() < 0.2, "norms carry the unit offset");
        if name == "qwen3.8-27b-exact" {
            let raw = ck.view("mtp.layers.0.self_attn.k_proj.weight").unwrap();
            let scale = ck.view("mtp.layers.0.self_attn.k_proj.weight_scale_inv").unwrap();
            let s = specrt::weights::bf16_to_f32(u16::from_le_bytes([scale.data[0], scale.data[1]]));
            let Matrix::Bf16(w) = &a.qkv else { unreachable!() };
            let k0 = spec.q_dim() * h;
            for c in 0..64 {
                let want = specrt::weights::f32_to_bf16(specrt::weights::e4m3_to_f32(raw.data[c]) * s);
                assert_eq!(w[k0 + c], want, "k_proj row 0 column {c}");
            }
        }
    }
}

#[test]
#[ignore = "needs about 26 GB of free GPU memory, the NVFP4 27B checkpoint and vllm-spec-greedy.json; then scripts/reference.py real-mtp"]
fn real_27b_mtp_dumps_hiddens_and_draft_logits_for_the_cpu_oracle() {
    let spec = named_spec("qwen3.8-27b-nvfp4-exact");
    let recs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out_dir().join("vllm-spec-greedy.json")).unwrap()).unwrap();
    let mut rt = Runtime::load(&spec, 1024).unwrap();
    let mut records = Vec::new();
    for (i, r) in recs.as_array().unwrap().iter().take(5).enumerate() {
        let ids: Vec<u32> = ids(&r["prompt"]).into_iter().chain(ids(&r["vllm"])).collect();
        let (logits, drafts, hidden) = rt.mtp_prompt_trace(&ids).unwrap();
        let name = format!("real27b-mtp-{i}");
        let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(out_dir().join(format!("{name}.hidden.bin")), bytes(&hidden)).unwrap();
        std::fs::write(out_dir().join(format!("{name}.chain.bin")), bytes(&logits)).unwrap();
        records.push(serde_json::json!({"name": name, "tokens": ids, "prompt": r["prompt"].as_array().unwrap().len(), "drafts": drafts}));
    }
    std::fs::write(out_dir().join("real27b-mtp.json"), serde_json::to_string(&records).unwrap()).unwrap();
}

#[test]
#[ignore = "needs about 26 GB of free GPU memory, the NVFP4 27B checkpoint and vllm-spec-greedy.json"]
fn real_27b_mtp_acceptance_on_vllms_own_greedy_tokens() {
    teacher_forced("vllm-spec-greedy.json");
}

#[test]
#[ignore = "needs about 26 GB of free GPU memory, the NVFP4 27B checkpoint and vllm-isolated-greedy.json from scripts/vllm_mtp_dump.py"]
fn real_27b_mtp_acceptance_on_an_isolated_vllms_greedy_tokens() {
    teacher_forced("vllm-isolated-greedy.json");
}

fn teacher_forced(file: &str) {
    let spec = named_spec("qwen3.8-27b-nvfp4-exact");
    let k = spec.mtp.as_ref().unwrap().draft_tokens;
    let recs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out_dir().join(file)).unwrap()).unwrap();
    let mut rt = Runtime::load(&spec, 1024).unwrap();
    let (mut steps, mut accepted, mut first, mut second) = (0u64, 0u64, 0u64, 0u64);
    for r in recs.as_array().unwrap() {
        let prompt = ids(&r["prompt"]);
        let vgen = ids(&r["vllm"]);
        let mut at = 0;
        while at + k < vgen.len() {
            let prefix: Vec<u32> = prompt.iter().chain(&vgen[..at]).copied().collect();
            let (_, drafts) = rt.mtp_prompt_logits(&prefix).unwrap();
            let a = drafts.iter().zip(&vgen[at + 1..]).take_while(|(d, g)| d == g).count();
            steps += 1;
            accepted += a as u64;
            first += u64::from(a >= 1);
            second += u64::from(a >= 2);
            at += a + 1;
        }
    }
    eprintln!(
        "teacher-forced on {file}: {steps} steps, mean acceptance length {:.3}, position 0 {:.3}, position 1 {:.3}",
        1.0 + accepted as f64 / steps as f64,
        first as f64 / steps as f64,
        second as f64 / steps as f64,
    );
}

#[test]
#[ignore = "needs about 25 GB of free GPU memory and scripts/reference.py real-full output"]
fn real_27b_per_layer_trace_matches_the_cpu_oracle() {
    let spec = full_spec();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir().join("real27b-ref.json")).unwrap()).unwrap();
    let seq = ids(&meta["sequences"][0]);
    let n = 64;
    let mut rt = Runtime::load(&spec, 8192).unwrap();
    let trace = rt.trace_prefill(&seq[..n]).unwrap();
    let theirs = floats("real27b-ref.layers.bin");
    let h = spec.dims.hidden;
    let per_layer: Vec<f32> = (0..spec.layers.count)
        .map(|l| relative(&trace.layers[l * n * h..(l + 1) * n * h], &theirs[l * n * h..(l + 1) * n * h]))
        .collect();
    let first_bad = per_layer.iter().position(|&e| e > 1e-3);
    eprintln!("per-layer max error / max |x|: {per_layer:?}; first above 1e-3: {first_bad:?}");
    assert!(first_bad.is_none(), "layer {first_bad:?}");
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
    let ref_kl = mean_kl(&weight_only, &act_ref, vocab);
    let ref_top1 = top1_agreement(&weight_only, &act_ref, vocab);
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
    let heads = [Quant::Fp8Channel, Quant::Nvfp4G16].map(|h| format!("weight-head-{}", head_name(h)));
    let modes = [("weight".to_string(), &weight_only), ("act".to_string(), &act_ref)].into_iter().chain(heads.into_iter().map(|m| (m, &weight_only)));
    for (mode, reference) in modes {
        let Ok(text) = std::fs::read_to_string(out_dir().join(format!("real27b-specrt-{mode}.json"))) else { continue };
        let meta: serde_json::Value = serde_json::from_str(&text).unwrap();
        let seqs: Vec<Vec<u32>> = meta["sequences"].as_array().unwrap().iter().map(ids).collect();
        let ours = floats(&format!("real27b-specrt-{mode}.logits.bin"));
        let error = relative(&ours, reference);
        let to_weight_only = relative(&ours, &weight_only);
        let misses = first_miss(&seqs, reference);
        let kl = mean_kl(&weight_only, &ours, vocab);
        let top1 = top1_agreement(&weight_only, &ours, vocab);
        eprintln!(
            "{mode}: logits max error / max |logit| vs its reference {error}, vs weight-only {to_weight_only}; envelope {envelope}; KL(weight-only || ours) {kl} against {ref_kl} for the act reference; top-1 vs weight-only {top1} against {ref_top1}; greedy per prompt: Ok(n) = n tokens agree, Err((step, reference margin)) = first disagreement {misses:?}"
        );
        if mode == "weight" {
            assert!(error <= envelope, "{mode}: {error} outside the {envelope} envelope");
            assert!(misses.iter().all(|m| *m == Ok(len - prompt_len)), "{misses:?}");
        } else {
            assert!(kl <= 1.5 * ref_kl, "{mode}: KL {kl} against the act reference's {ref_kl}");
            assert!(top1 + 0.02 >= ref_top1, "{mode}: top-1 {top1} against {ref_top1}");
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
    spec.mtp = None;
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
