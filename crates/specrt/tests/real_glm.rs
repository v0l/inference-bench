use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::PathBuf;

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_GLM_DIR").expect("SPECRT_GLM_DIR points at the Firworks/GLM-4.5-Air-nvfp4 snapshot"))
}

fn refs() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR"))
}

fn spec() -> Spec {
    let dir = checkpoint();
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    spec
}

fn reference() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(refs().join("vllm-glm-air.json")).unwrap()).unwrap()
}

#[test]
#[ignore = "needs the GLM-4.5-Air NVFP4 checkpoint in SPECRT_GLM_DIR, vllm-glm-air.json in SPECRT_REF_DIR (scripts/vllm_logprobs_offline.py) and about 70 GB of free GPU memory"]
fn glm_air_nvfp4_agrees_with_vllm_logprobs() {
    let spec = spec();
    eprintln!("quant={:?} mlp.quant={:?} act_quant={:?} mtp={}", spec.quant, spec.mlp.quant, spec.mlp.act_quant, spec.mtp.is_some());
    let meta = reference();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let records = meta["records"].as_array().unwrap();
    let mut rt = Runtime::load(&spec, 2048).unwrap();
    let vocab = spec.dims.vocab;
    let (mut kls, mut top1, mut equal) = (Vec::new(), 0usize, 0usize);
    for r in records {
        let prompt = ids(&r["prompt"]);
        let theirs = ids(&r["greedy"]);
        let ours = rt.generate(&prompt, new, Sampling::GREEDY).unwrap().tokens;
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
    std::fs::write(refs().join("glm-air-compare.json"), report.to_string()).unwrap();
    assert!(top1 * 10 >= total * 9, "teacher-forced top-1 {top1}/{total}");
    assert!(mean <= 0.05, "mean KL {mean}");
}

#[test]
#[ignore = "needs the GLM-4.5-Air NVFP4 checkpoint in SPECRT_GLM_DIR"]
fn glm_air_nvfp4_checkpoint_has_no_nextn_layer_so_it_serves_without_mtp() {
    let spec = spec();
    assert!(spec.mtp.is_some(), "num_nextn_predict_layers = 1 in config.json");
    let resolved = specrt::engine::resolve_spec(&spec.repo).unwrap();
    assert!(resolved.mtp.is_none(), "model.layers.46 is not in the Firworks export");
}
