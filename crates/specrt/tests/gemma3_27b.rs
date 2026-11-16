use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{ActQuant, Quant, Spec};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var_os("GEMMA3_27B_DIR").expect("GEMMA3_27B_DIR holds an FP8 or NVFP4 Gemma 3 27B checkpoint"))
}

#[test]
#[ignore = "needs GEMMA3_27B_DIR (RedHatAI/gemma-3-27b-it-FP8-dynamic)"]
fn the_gemma3_27b_fp8_config_maps_through_the_family_spec() {
    let json = std::fs::read_to_string(dir().join("config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.layers.count, 62);
    assert_eq!((spec.attention.heads, spec.attention.kv_heads, spec.attention.head_dim), (32, 16, 128));
    assert_eq!(spec.attention.query_scalar, Some(168.0));
    assert_eq!(spec.layers.window, Some(1024));
    assert_eq!(spec.layers.pattern.len(), 62);
    assert_eq!(spec.quant, Quant::Fp8Channel);
    assert_eq!(spec.act_quant, ActQuant::Fp8);
    assert!(spec.dims.tie_embeddings && spec.norm.sandwich && spec.norm.unit_offset);
    let v = spec.vision.unwrap();
    assert_eq!((v.config.tokens_per_tile(), v.config.out_hidden_size), (256, 5376));
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore = "needs GEMMA3_27B_DIR, SPECRT_REF_DIR with vllm-gemma3-27b.json (scripts/vllm_gemma3.sh) and about 36 GB of free GPU memory"]
fn gemma3_27b_fp8_agrees_with_vllm_on_top1_kl_and_greedy() {
    let json = std::fs::read_to_string(dir().join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving(&json).unwrap();
    spec.name = "gemma3-27b".into();
    spec.repo = dir().display().to_string();
    let reference = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").unwrap()).join("vllm-gemma3-27b.json");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let mut rt = Runtime::load(&spec, 4096).unwrap();
    let vocab = spec.dims.vocab;
    let (mut kls, mut top1, mut total, mut prefixes) = (Vec::new(), 0usize, 0usize, Vec::new());
    for r in meta["records"].as_array().unwrap() {
        let prompt = ids(&r["prompt"]);
        let theirs = ids(&r["greedy"]);
        let ours = rt.generate(&prompt, new, Sampling::GREEDY).unwrap().tokens;
        prefixes.push(ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count());
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
            let best = (0..vocab).fold(0, |b, i| if l[i] > l[b] { i } else { b });
            if best == theirs[step] as usize {
                top1 += 1;
            }
            total += 1;
        }
    }
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    eprintln!("gemma3 27b fp8 against vLLM: top-1 {top1}/{total}, mean KL {mean:.5}, max KL {:.4}, greedy prefixes {prefixes:?}", kls.iter().cloned().fold(0.0, f64::max));
    assert!(top1 as f64 >= 0.9 * total as f64, "top-1 {top1}/{total}");
    assert!(mean <= 0.05, "mean KL {mean}");
}
