use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(dir: &str) -> Spec {
    let path: PathBuf = root().join("tests/fixtures").join(dir);
    let json = std::fs::read_to_string(path.join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = path.display().to_string();
    spec
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 3 + i * 7 % 50).collect()
}

fn run(spec: &Spec, pdl: bool, graphs: bool, tokens: &[u32]) -> (Vec<f32>, Vec<u32>) {
    run_hoisted(spec, pdl, false, graphs, tokens)
}

fn run_hoisted(spec: &Spec, pdl: bool, hoist: bool, graphs: bool, tokens: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let mut spec = spec.clone();
    spec.runtime.pdl = pdl;
    spec.runtime.pdl_hoist = hoist;
    spec.runtime.graphs = graphs;
    let mut rt = Runtime::load(&spec, 256).unwrap();
    let logits = rt.prompt_logits(tokens).unwrap();
    let greedy = rt.generate(tokens, 24, Sampling::GREEDY).unwrap().tokens;
    (logits, greedy)
}

fn same_with_and_without_pdl(spec: &Spec, tokens: &[u32]) {
    let on = run(spec, true, true, tokens);
    let off = run(spec, false, true, tokens);
    assert_eq!(on.1, off.1, "{}: greedy differs", spec.name);
    assert!(on.0.iter().zip(&off.0).all(|(a, b)| a.to_bits() == b.to_bits()), "{}: logits differ", spec.name);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pdl_on_and_off_are_bit_identical_on_tiny_fixtures() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-nvfp4", "tiny-qwen3.5-fp8", "tiny-mistral", "tiny-phi3", "tiny-granite", "tiny-olmo2", "tiny-gemma3", "tiny-smollm3", "tiny-qwen3.5-27b"] {
        let spec = fixture(dir);
        same_with_and_without_pdl(&spec, &prompt(40));
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pdl_hoist_on_and_off_are_bit_identical_on_tiny_fixtures() {
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-nvfp4", "tiny-qwen3.5-fp8", "tiny-mistral", "tiny-granite", "tiny-olmo2", "tiny-gemma3", "tiny-qwen3.5-27b"] {
        let spec = fixture(dir);
        let tokens = prompt(40);
        let off = run(&spec, false, true, &tokens);
        for graphs in [true, false] {
            let hoisted = run_hoisted(&spec, true, true, graphs, &tokens);
            assert_eq!(hoisted.1, off.1, "{dir}: greedy differs with pdl_hoist, graphs {graphs}");
            assert!(hoisted.0.iter().zip(&off.0).all(|(a, b)| a.to_bits() == b.to_bits()), "{dir}: logits differ with pdl_hoist, graphs {graphs}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn pdl_on_and_off_are_bit_identical_on_qwen3_0_6b() {
    let spec = Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap();
    same_with_and_without_pdl(&spec, &(0..64u32).map(|i| 1000 + i * 7 % 5000).collect::<Vec<_>>());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn pdl_graph_and_eager_batches_agree() {
    let spec = fixture("tiny-qwen3.5");
    let prompts: Vec<Vec<u32>> = (0..4u32).map(|s| (0..30u32).map(|i| 3 + (i * 7 + s * 13) % 50).collect()).collect();
    let mut outs = Vec::new();
    for (pdl, graphs) in [(true, true), (true, false), (false, true), (false, false)] {
        let mut s = spec.clone();
        s.runtime.pdl = pdl;
        s.runtime.graphs = graphs;
        let mut rt = Runtime::load_batched(&s, 256, 4).unwrap();
        outs.push(rt.generate_batch(&prompts, 16).unwrap().tokens);
    }
    assert!(outs.windows(2).all(|w| w[0] == w[1]));
}
