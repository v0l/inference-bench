use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::spec::Spec;

fn spec() -> Spec {
    let dir = std::env::var("SPECRT_122B_DIR").expect("SPECRT_122B_DIR points at the Qwen3.5-122B-A10B-NVFP4 snapshot");
    let json = std::fs::read_to_string(format!("{dir}/config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
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
    let mut rt = Runtime::load_layout(&s, Layout::contiguous(4608, 1)).unwrap();
    let ids = prompt();
    let g = rt.generate(&ids, 24, Sampling::GREEDY).unwrap();
    eprintln!("tokens {:?}", g.tokens);
    eprintln!("prefill {:.3}s decode {:.3}s for {} tokens", g.prefill_secs, g.decode_secs, g.tokens.len());
    let long: Vec<u32> = (0..4096u32).map(|i| ids[(i as usize) % ids.len()] + (i / ids.len() as u32) % 7).collect();
    let started = std::time::Instant::now();
    rt.prompt_logits(&long).unwrap();
    eprintln!("prefill 4096 tokens {:.3}s", started.elapsed().as_secs_f64());
    std::fs::write("/tmp/g122.json", serde_json::to_string(&g.tokens).unwrap()).unwrap();
}
