use specrt::family::Registry as Families;
use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const SINKS: &str = "tiny-sinks";
const CHUNKED: &str = "tiny-chunked";
const MLA: &str = "tiny-deepseek-v2";
const MLA_MOE: &str = "tiny-deepseek-v2-moe";
const GLM: &str = "tiny-glm4-moe";
const OSS: &str = "tiny-gpt-oss";
const DSA: &str = "tiny-glm-dsa";

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
fn the_deepseek_moe_config_reads_wide_shared_experts_and_the_routed_scale_from_the_family_toml() {
    let spec = spec(MLA_MOE);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k(), moe.expert_intermediate, moe.shared_intermediate), (8, 2, 256, 512));
    assert_eq!((moe.shared_slots(), moe.pairs(), moe.shared_gate, moe.norm_topk, moe.routed_scale), (2, 4, false, false, 1.0));
    assert_eq!(moe.shared_module, "shared_experts");
    assert_eq!(moe.dense_layers, vec![0]);
    spec.runnable().unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.source.contains("constexpr int MOE_SHARED = 2;") && module.source.contains("constexpr int MOE_PAIRS = 4;"));
}

#[test]
fn the_deepseek_moe_checkpoint_loads_the_wide_shared_mlp_as_slots() {
    let spec = spec(MLA_MOE);
    let ck = specrt::weights::Checkpoint::open(&fixture(MLA_MOE)).unwrap();
    let w = ck.layer(&spec, 1).unwrap().moe.unwrap();
    let (h, inter) = (spec.dims.hidden, 256);
    let specrt::weights::Stacked::Bf16(gate) = &w.gate else { panic!("bf16") };
    let specrt::weights::Stacked::Bf16(down) = &w.down else { panic!("bf16") };
    assert_eq!((gate.len(), down.len()), (10 * inter * h, 10 * h * inter));
    assert!(w.shared_gate.is_none());
    let whole = ck.bf16("model.layers.1.mlp.shared_experts.down_proj.weight", &[h, 2 * inter]).unwrap();
    let second = &down[9 * h * inter..];
    assert_eq!(second[0], whole[inter]);
    assert_eq!(second[inter], whole[2 * inter + inter]);
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
fn deepseek_moe_with_wide_shared_experts_matches_transformers() {
    close(MLA_MOE, 5e-2);
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

#[test]
fn the_glm_config_reads_groups_scaling_bias_and_sigmoid_from_the_family_toml() {
    let spec = spec(GLM);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k(), moe.groups, moe.topk_group), (8, 2, 2, 1));
    assert_eq!((moe.router, moe.score_bias, moe.norm_topk, moe.routed_scale), (specrt::spec::RouterAct::Sigmoid, true, true, 1.5));
    assert_eq!((moe.shared_slots(), moe.shared_gate, moe.dense_layers.clone()), (1, false, vec![0]));
    assert!(spec.attention.qk_norm);
    spec.runnable().unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    for c in ["constexpr int MOE_GROUPS = 2;", "constexpr int MOE_TOPK_GROUP = 1;", "constexpr bool MOE_BIAS = true;", "constexpr bool MOE_SIGMOID = true;"] {
        assert!(module.source.contains(c), "{c}");
    }
}

#[test]
fn the_glm_checkpoint_loads_the_score_bias() {
    let spec = spec(GLM);
    let ck = specrt::weights::Checkpoint::open(&fixture(GLM)).unwrap();
    let w = ck.layer(&spec, 1).unwrap().moe.unwrap();
    assert_eq!(w.score_bias.unwrap().len(), 8);
}

#[test]
fn a_dense_moe_config_has_no_groups_or_bias() {
    let spec = spec(MLA_MOE);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.groups, moe.topk_group, moe.score_bias), (0, 0, false));
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn group_limited_sigmoid_routing_with_score_bias_matches_transformers() {
    close(GLM, 5e-2);
}

#[test]
fn the_gpt_oss_config_reads_biases_clamp_and_mxfp4_experts_from_the_family_toml() {
    let spec = spec(OSS);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.top_k(), moe.router_bias, moe.expert_bias, moe.norm_topk), (8, 2, true, true, true));
    assert_eq!((moe.router_module.as_str(), moe.expert_block, moe.shared_slots()), ("router", Some(specrt::dequant::MXFP4), 0));
    assert!(spec.attention.qkv_bias && spec.attention.o_bias);
    let gated = spec.mlp.gated.as_ref().unwrap();
    assert_eq!((gated.params["alpha"], gated.params["limit"]), (1.702, 2.5));
    assert_eq!(spec.layers.window, Some(16));
    spec.runnable().unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    for c in ["constexpr bool MOE_ROUTER_BIAS = true;", "constexpr bool MOE_EBIAS = true;", "fminf(g, 2.5e0f)"] {
        assert!(module.source.contains(c), "{c}");
    }
}

#[test]
fn the_gpt_oss_checkpoint_decodes_mxfp4_experts_and_loads_every_bias() {
    let spec = spec(OSS);
    let ck = specrt::weights::Checkpoint::open(&fixture(OSS)).unwrap();
    let w = ck.layer(&spec, 0).unwrap();
    let m = w.moe.unwrap();
    let (h, inter, e) = (256, 256, 8);
    let specrt::weights::Stacked::Block { packed, scales } = &m.gate else { panic!("packed") };
    assert_eq!((packed.len(), scales.len()), (e * inter * h / 2, e * inter * h / 32));
    assert_eq!((m.router_bias.unwrap().len(), m.gate_up_bias.unwrap().len(), m.down_bias.unwrap().len()), (e, e * 2 * inter, e * h));
    let specrt::weights::Mixer::Attention(a) = &w.mixer else { panic!("attention") };
    assert_eq!(a.o_bias.as_ref().unwrap().len(), h);
    assert!(a.extra.iter().any(|(n, v)| n == "sinks" && v.len() == 4));
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn gpt_oss_sinks_clamped_swiglu_and_mxfp4_experts_match_transformers() {
    close(OSS, 5e-2);
}

#[test]
fn the_dsa_config_reads_indexer_layers_states_and_nodes_from_the_family_toml() {
    let spec = spec(DSA);
    assert_eq!(spec.layers.pattern, vec![Block::Full, Block::Shared]);
    assert_eq!(spec.state.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["latent", "index_key"]);
    assert_eq!(specrt::state::named(&spec, "index_key").unwrap().dims(&spec).unwrap(), vec![1, 64]);
    assert_eq!(spec.attention.params["index_topk"], 8.0);
    assert!((spec.attention.params["index_scale"] - 0.125).abs() < 1e-12);
    assert!((spec.attention.params["index_w_scale"] - 0.5).abs() < 1e-12);
    assert_eq!(spec.qkv_rows(), 64 + 128 + 64 + 4);
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let kernels = |block: Block| plan.layer(block).iter().filter(|o| matches!(o, Op::Kernel { .. })).count();
    assert_eq!((kernels(Block::Full), kernels(Block::Shared)), (15, 9));
}

#[test]
fn the_dsa_module_builds_with_no_binary_support() {
    let module = specrt::codegen::Module::generate_kv(&spec(DSA), 256, 1, &kir::Target::sm_120()).unwrap();
    for k in ["dsa_q_norm", "dsa_latent_norm", "dsa_q_b", "dsa_rope_q", "dsa_latent_write", "dsa_absorb_q", "dsa_index_q", "dsa_index_rope_q", "dsa_index_k_norm", "dsa_index_rope_k", "dsa_index_write", "dsa_index_topk", "dsa_attention"] {
        assert!(module.kernels.contains(&k), "{k}");
        assert!(module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(module.source.contains("sel_hit"), "the scan keeps only indexed keys");
    assert!(module.source.contains("rsqrtf(kir_block_sum(sq, scratch)"), "the indexer key uses a layernorm");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn dsa_indexer_with_a_shared_layer_matches_transformers() {
    close(DSA, 5e-2);
}

#[test]
#[ignore = "needs an sm_120 GPU and scripts/tiny_variants.py output in SPECRT_REF_DIR"]
fn gpt_oss_block_scaled_mma_experts_match_the_packed_gemv_experts() {
    let (tokens, _, vocab, _) = reference(OSS);
    let mut mma = spec(OSS);
    mma.mlp.act_quant = specrt::spec::ActQuant::Fp4;
    let gemv = spec(OSS);
    let mut rt_mma = Runtime::load_batched(&mma, 256, 1).unwrap();
    let mut rt_gemv = Runtime::load_batched(&gemv, 256, 1).unwrap();
    let a = rt_mma.prompt_logits(&tokens).unwrap();
    let b = rt_gemv.prompt_logits(&tokens).unwrap();
    let time = |rt: &mut Runtime| {
        let start = std::time::Instant::now();
        for _ in 0..50 {
            rt.prompt_logits(&tokens).unwrap();
        }
        start.elapsed().as_secs_f64() * 20.0
    };
    eprintln!("gpt-oss prefill of {} tokens: mma {:.2} ms, gemv {:.2} ms", tokens.len(), time(&mut rt_mma), time(&mut rt_gemv));
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
    let cosine = dot(&a, &b) / (dot(&a, &a).sqrt() * dot(&b, &b).sqrt());
    eprintln!("gpt-oss mma vs gemv cosine {cosine} over {} rows of {vocab}", tokens.len());
    assert!(a.iter().all(|v| v.is_finite()) && cosine > 0.8, "cosine {cosine}");
}
