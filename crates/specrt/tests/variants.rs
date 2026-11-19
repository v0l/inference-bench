use specrt::family::Registry as Families;
use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const SINKS: &str = "tiny-sinks";
const CHUNKED: &str = "tiny-chunked";
const MLA: &str = "tiny-deepseek-v2";

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}

fn spec(name: &str) -> Spec {
    let json = std::fs::read_to_string(fixture(name).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap();
    spec.name = name.into();
    spec.repo = fixture(name).display().to_string();
    spec
}

fn reference(name: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

fn kernel_nodes(spec: &Spec, phase: Phase) -> Vec<Op> {
    let plan = Plan::build(spec, phase, true);
    [Block::Full, Block::Sliding].iter().flat_map(|b| plan.layer(*b).iter().copied()).filter(|op| matches!(op, Op::Kernel { .. })).collect()
}

#[test]
fn the_sinks_config_becomes_kernel_nodes_from_the_family_toml() {
    let spec = spec(SINKS);
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Full]);
    assert_eq!(spec.attention.params["softcap"], 8.0);
    let prefill = Plan::build(&spec, Phase::Prefill, true);
    assert!(!prefill.layer(Block::Full).contains(&Op::Flash));
    assert!(!kernel_nodes(&spec, Phase::Prefill).is_empty() && !kernel_nodes(&spec, Phase::Decode).is_empty());
    let node = &spec.attention.node[1];
    assert_eq!(node.kernel.as_deref(), Some("attn_sinks"));
}

#[test]
fn the_chunked_config_maps_chunks_nope_and_temperature_from_the_config() {
    let spec = spec(CHUNKED);
    assert_eq!(spec.layers.pattern, vec![Block::Sliding, Block::Full]);
    assert_eq!(spec.layers.window, Some(16));
    assert_eq!(spec.rope.nope_layers, vec![1, 3]);
    assert_eq!(spec.attention.params["attn_scale"], 0.5);
    assert_eq!(spec.attention.params["floor_scale"], 16.0);
    assert_eq!(spec.attention.params["tuning"], 1.0);
}

#[test]
fn both_variant_modules_build_with_no_binary_support() {
    for name in [SINKS, CHUNKED] {
        let module = specrt::codegen::Module::generate_kv(&spec(name), 256, 1, &kir::Target::sm_120()).unwrap();
        let kernel = if name == SINKS { "attn_sinks" } else { "attn_chunked" };
        assert!(module.kernels.contains(&kernel), "{name}");
        assert!(module.source.contains(&format!("{kernel}(")), "{name}");
    }
}

fn close(name: &str, tolerance: f32) {
    let (tokens, greedy, vocab, want) = reference(name);
    let spec = spec(name);
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{name} prefill max_abs {max_abs}");
    assert!(max_abs < tolerance, "{name} max abs diff {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    for r in 0..tokens.len() {
        let span = r * vocab..(r + 1) * vocab;
        assert_eq!(top(&ours[span.clone()]), top(&want[span]), "{name} row {r}");
    }
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "{name} decode greedy");
    let _ = Sampling::GREEDY;
}

#[test]
fn the_deepseek_config_becomes_a_latent_cache_and_mla_nodes_from_the_family_toml() {
    let spec = spec(MLA);
    assert_eq!(spec.state.len(), 1);
    let latent = specrt::state::named(&spec, "latent").unwrap();
    assert_eq!(latent.dims(&spec).unwrap(), vec![1, 96 + 32]);
    assert_eq!(specrt::runtime::kv_block_bytes(&spec, 1), 4 * 128 * 2 * 16);
    assert_eq!(specrt::anatomy::ModelAnatomy::new(&spec, 0, None).kv_bytes_per_token, 4 * 128 * 2);
    assert_eq!(spec.qkv_rows(), 4 * 64 + 96 + 32);
    assert_eq!(spec.o_dim(), 4 * 64);
    assert!((spec.attention.params["softmax_scale"] - 1.0 / 64f64.sqrt()).abs() < 1e-12);
    let prefill = Plan::build(&spec, Phase::Prefill, true);
    assert!(!prefill.layer(Block::Dense).contains(&Op::Flash));
    assert!(!kernel_nodes(&spec, Phase::Decode).is_empty());
    let ops = Plan::build(&spec, Phase::Decode, true);
    assert_eq!(ops.layer(Block::Dense).iter().filter(|o| matches!(o, Op::Kernel { .. })).count(), 7);
}

#[test]
fn the_mla_module_builds_with_no_binary_support() {
    let module = specrt::codegen::Module::generate_kv(&spec(MLA), 256, 1, &kir::Target::sm_120()).unwrap();
    for k in ["mla_latent_norm", "mla_rope_q", "mla_rope_k", "mla_latent_write", "mla_absorb_q", "mla_attention", "mla_expand_v"] {
        assert!(module.kernels.contains(&k), "{k}");
        assert!(module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(module.source.contains("float a = v[2 * i];"), "the rope is pair style");
    assert!(module.source.contains("constexpr int QK = 128;") && module.source.contains("constexpr int VD = 96;"), "split QK and V widths");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn mla_absorbed_over_the_latent_cache_matches_transformers() {
    close(MLA, 5e-2);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn sinks_with_soft_cap_and_window_match_transformers() {
    close(SINKS, 5e-2);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn chunked_nope_with_temperature_matches_transformers() {
    close(CHUNKED, 5e-2);
}
