use specrt::runtime::Runtime;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

#[test]
#[ignore = "needs the Qwen3-Next NVFP4 checkpoint in SPECRT_NEXT_DIR, vllm-next.json in SPECRT_REF_DIR (scripts/vllm_logprobs_offline.py) and about 50 GB of free GPU memory"]
fn qwen3_next_nvfp4_agrees_with_vllm_logprobs() {
    let dir = PathBuf::from(std::env::var_os("SPECRT_NEXT_DIR").expect("SPECRT_NEXT_DIR"));
    let refs = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR"));
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving(&config).unwrap();
    spec.repo = dir.display().to_string();
    eprintln!("quant={:?} mlp.quant={:?} act_quant={:?} linear={:?}", spec.quant, spec.mlp.quant, spec.mlp.act_quant, spec.linear_attention);
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refs.join("vllm-next.json")).unwrap()).unwrap();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let records = meta["records"].as_array().unwrap();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let vocab = spec.dims.vocab;
    let (mut kls, mut top1, mut equal) = (Vec::new(), 0usize, 0usize);
    for r in records {
        let prompt = ids(&r["prompt"]);
        let theirs = ids(&r["greedy"]);
        let ours = rt.generate(&prompt, new, specrt::runtime::Sampling::GREEDY).unwrap().tokens;
        let mut seq = prompt.clone();
        seq.extend(&theirs);
        let logits = rt.prompt_logits(&seq).unwrap();
        for (step, row) in r["logprobs"].as_array().unwrap().iter().enumerate() {
            let at = (prompt.len() + step - 1) * vocab;
            let l = &logits[at..at + vocab];
            let max = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = max + l.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
            let top: Vec<(usize, f64)> = row.as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_f64().unwrap())).collect();
            let p_mass: f64 = top.iter().map(|(_, lp)| lp.exp()).sum();
            let mut kl = 0.0;
            for &(t, lp) in &top {
                kl += lp.exp() * (lp - (l[t] - lse) as f64);
            }
            let rest_p = (1.0 - p_mass).max(1e-12);
            let rest_q = (1.0 - top.iter().map(|&(t, _)| ((l[t] - lse) as f64).exp()).sum::<f64>()).max(1e-12);
            kl += rest_p * (rest_p / rest_q).ln();
            kls.push(kl);
            top1 += (argmax(l) == theirs[step] as usize) as usize;
            equal += (ours.get(step) == Some(&theirs[step])) as usize;
        }
    }
    let total = records.len() * new;
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().cloned().fold(0.0, f64::max);
    let report = serde_json::json!({"prompts": records.len(), "positions": total, "teacher_forced_top1": top1, "greedy_positions_equal": equal, "kl_mean": mean, "kl_max": max});
    eprintln!("{report}");
    std::fs::write(Path::new(&refs).join("next-compare.json"), report.to_string()).unwrap();
    assert!(top1 * 10 >= total * 9, "teacher-forced top-1 {top1}/{total}");
    assert!(mean <= 0.05, "mean KL {mean}");
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

#[test]
#[ignore = "needs the Qwen3-Next NVFP4 checkpoint in SPECRT_NEXT_DIR, qwen3-next-{N}l.ref.* in SPECRT_REF_DIR (scripts/next_slice.py N) and about 20 GB of free GPU memory"]
fn qwen3_next_nvfp4_slice_matches_transformers() {
    let dir = PathBuf::from(std::env::var_os("SPECRT_NEXT_DIR").expect("SPECRT_NEXT_DIR"));
    let refs = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR"));
    let n: usize = std::env::var("SPECRT_NEXT_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refs.join(format!("qwen3-next-{n}l.ref.json"))).unwrap()).unwrap();
    let tokens = ids(&meta["tokens"]);
    let greedy = ids(&meta["greedy"]);
    let vocab = meta["vocab"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(refs.join(format!("qwen3-next-{n}l.ref.bin"))).unwrap();
    let want: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let rows = want.len() / vocab;
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec.layers.count = n;
    let mut rt = Runtime::load(&spec, 512).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    let first = tokens.len() - rows;
    let (mut worst, mut top1) = (0f64, 0usize);
    for r in 0..rows {
        let a = &ours[(first + r) * vocab..(first + r + 1) * vocab];
        let b = &want[r * vocab..(r + 1) * vocab];
        let (la, lb) = (log_softmax(a), log_softmax(b));
        let kl: f64 = lb.iter().zip(&la).map(|(q, p)| q.exp() * (q - p)).sum();
        worst = worst.max(kl);
        top1 += (argmax(a) == argmax(b)) as usize;
        eprintln!("qwen3-next-{n}l row {r} kl {kl}");
    }
    let same = if greedy.is_empty() { 0 } else { rt.generate(&tokens, greedy.len(), specrt::runtime::Sampling::GREEDY).unwrap().tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count() };
    eprintln!("qwen3-next-{n}l rows {rows} worst kl {worst} top1 {top1}/{rows} greedy prefix {same}/{}", greedy.len());
    assert!(worst < 0.001, "kl {worst}");
    assert!(top1 * 4 >= rows * 3, "top-1 {top1}/{rows}");
}
