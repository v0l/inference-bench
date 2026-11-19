use specrt::family::Registry as Families;
use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::Runtime;
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const V4: &str = "tiny-deepseek-v4-attention";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn spec() -> Spec {
    let json = std::fs::read_to_string(fixture(V4).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = V4.into();
    spec.repo = fixture(V4).display().to_string();
    spec
}

#[test]
fn the_layer_types_become_blocks_and_the_ratios_come_from_the_config() {
    let spec = spec();
    assert_eq!(spec.layers.pattern, vec![Block::Compressed, Block::Indexed, Block::Sliding]);
    assert_eq!(spec.attention.params["csa_ratio"], 4.0);
    assert_eq!(spec.attention.params["hca_ratio"], 8.0);
    assert_eq!(spec.layers.window, Some(16));
}

#[test]
fn each_layer_kind_runs_only_its_own_kernels() {
    let spec = spec();
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let names = |b: Block| -> Vec<String> {
        plan.layer(b)
            .iter()
            .filter_map(|op| match op {
                Op::Kernel { node } => spec.attention.node[*node as usize].kernel.clone(),
                _ => None,
            })
            .collect()
    };
    assert!(names(Block::Compressed).contains(&"v4_hca_pool".to_string()) && !names(Block::Compressed).contains(&"v4_idx_topk".to_string()));
    assert!(names(Block::Indexed).contains(&"v4_idx_topk".to_string()) && names(Block::Indexed).contains(&"v4_csa_attn".to_string()));
    assert!(names(Block::Sliding).contains(&"v4_swa_attn".to_string()) && !names(Block::Sliding).iter().any(|n| n.contains("pool")));
    for b in [Block::Compressed, Block::Indexed, Block::Sliding] {
        assert!(names(b).contains(&"v4_o_group".to_string()));
    }
}

#[test]
fn the_module_builds_every_v4_kernel_from_descriptions() {
    let module = specrt::codegen::Module::generate_kv(&spec(), 256, 1, &kir::Target::sm_120()).unwrap();
    for k in ["v4_q_a_norm", "v4_kv_norm", "v4_q_matvec", "v4_q_rms", "v4_q_rope", "v4_kv_rope", "v4_kv_write", "v4_hca_pool", "v4_csa_pool", "v4_idx_pool", "v4_idx_topk", "v4_hca_attn", "v4_csa_attn", "v4_swa_attn", "v4_o_rope", "v4_o_group"] {
        assert!(module.kernels.contains(&k), "{k}");
        assert!(module.source.contains(&format!("{k}(")), "{k}");
    }
}

fn reference() -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{V4}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{V4}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn compressed_sparse_attention_matches_transformers() {
    let (tokens, greedy, vocab, want) = reference();
    let mut rt = Runtime::load_batched(&spec(), 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{V4} prefill max_abs {max_abs}");
    assert!(max_abs < 5e-2, "max abs diff {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    for r in 0..tokens.len() {
        let span = r * vocab..(r + 1) * vocab;
        assert_eq!(top(&ours[span.clone()]), top(&want[span]), "row {r}");
    }
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "decode greedy");
}
