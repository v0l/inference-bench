use specrt::runtime::Runtime;
use specrt::spec::{Quant, Spec};
use std::path::{Path, PathBuf};

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("out"))
}

fn work_dir() -> PathBuf {
    std::env::var_os("REQUANT_WORK").map(PathBuf::from).expect("REQUANT_WORK names a scratch folder for the checkpoint-format logits")
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn parse_quant(s: &str) -> Quant {
    match s {
        "nvfp4_g16" => Quant::Nvfp4G16,
        "fp8_channel" => Quant::Fp8Channel,
        "bf16" => Quant::Bf16,
        other => panic!("unknown quant {other}"),
    }
}

fn spec_with(sets: &str) -> Spec {
    let name = std::env::var("SPECRT_27B_NVFP4_SPEC").unwrap_or_else(|_| "qwen3.8-27b-nvfp4".into());
    let path = if name.ends_with(".toml") { PathBuf::from(&name) } else { Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml")) };
    let mut spec = Spec::load(&path).unwrap();
    for kv in sets.split(',').filter(|s| !s.is_empty()) {
        let (class, q) = kv.split_once('=').expect("class=quant");
        spec.runtime.requant.set(class, Some(parse_quant(q)));
    }
    spec.validate().unwrap();
    spec
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

#[test]
#[ignore = "needs the 27B NVFP4 checkpoint, a GPU and REQUANT_WORK"]
fn real_27b_requant_class_against_the_checkpoint_format() {
    let sets = std::env::var("REQUANT_SET").unwrap_or_default();
    let spec = spec_with(&sets);
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("vllm-nvfp4.json")).unwrap()).unwrap();
    let records = meta["records"].as_array().unwrap();
    let new = meta["new_tokens"].as_u64().unwrap() as usize;
    let vocab = spec.dims.vocab;
    let mut rt = Runtime::load(&spec, 4096).unwrap();
    eprintln!("set {sets:?}: weights {:.3} GB device {:.2} GB load {:.1} s", rt.weight_bytes as f64 / 1e9, rt.device_bytes as f64 / 1e9, rt.load_secs);
    let work = work_dir();
    std::fs::create_dir_all(&work).unwrap();
    let (mut kl_sum, mut rows, mut top1, mut vllm_top1) = (0.0f64, 0usize, 0usize, 0usize);
    let mut greedy_prefix = 0usize;
    for (n, r) in records.iter().enumerate() {
        let prompt = ids(&r["prompt"]);
        let theirs = ids(&r["greedy"]);
        let mut seq = prompt.clone();
        seq.extend(&theirs);
        let logits = rt.prompt_logits(&seq).unwrap();
        let file = work.join(format!("base-{n}.f32"));
        if sets.is_empty() {
            let bytes: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(&file, bytes).unwrap();
            continue;
        }
        let bytes = std::fs::read(&file).expect("run the empty REQUANT_SET first to write the checkpoint-format logits");
        let base: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        for pos in prompt.len() - 1..seq.len() - 1 {
            let (p, q) = (&base[pos * vocab..(pos + 1) * vocab], &logits[pos * vocab..(pos + 1) * vocab]);
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            kl_sum += lp.iter().zip(&lq).map(|(x, y)| x.exp() * (x - y)).sum::<f64>();
            top1 += (argmax(p) == argmax(q)) as usize;
            vllm_top1 += (argmax(q) == seq[pos + 1] as usize) as usize;
            rows += 1;
        }
        let ours = rt.generate_batch(std::slice::from_ref(&prompt), new.min(32)).unwrap().tokens.remove(0);
        greedy_prefix += ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
    }
    if sets.is_empty() {
        eprintln!("wrote checkpoint-format logits for {} sequences", records.len());
        return;
    }
    let report = serde_json::json!({
        "set": sets,
        "rows": rows,
        "mean_kl_vs_checkpoint_format": kl_sum / rows as f64,
        "top1_vs_checkpoint_format": top1 as f64 / rows as f64,
        "top1_vs_vllm_greedy": vllm_top1 as f64 / rows as f64,
        "greedy_prefix_total": greedy_prefix,
        "weight_gb": rt.weight_bytes as f64 / 1e9,
    });
    eprintln!("REQUANT_REPORT {report}");
}
