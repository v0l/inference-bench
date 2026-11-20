use specrt::family::Registry as Families;
use specrt::plan::{Buf, Op, Phase, Plan, Weight};
use specrt::runtime::Runtime;
use specrt::spec::{Activation, Block, Spec};
use std::path::{Path, PathBuf};

const TINY: &str = "tiny-nemotronh";
const TINY_MOE: &str = "tiny-nemotronh-moe";

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}

fn config(file: &str) -> String {
    std::fs::read_to_string(fixture(file)).unwrap()
}

fn serving(json: &str) -> anyhow::Result<Spec> {
    Spec::from_hf_config_serving_in(Families::embedded(), json)
}

fn tiny_named(name: &str) -> Spec {
    let mut spec = serving(&config(&format!("{name}/config.json"))).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = name.into();
    spec.repo = fixture(name).display().to_string();
    spec
}

fn tiny() -> Spec {
    tiny_named(TINY)
}

#[test]
fn the_nano_9b_pattern_string_becomes_mixer_attention_and_mlp_layers_from_the_family_toml() {
    let spec = serving(&config("nemotron-nano-9b-v2.config.json")).unwrap_or_else(|e| panic!("{e:#}"));
    let pattern = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-";
    assert_eq!(spec.layers.count, 56);
    let kinds: String = (0..56).map(|i| match spec.block(i) { Block::Linear => 'M', Block::Full => '*', Block::Mlp => '-', other => panic!("layer {i} is {other:?}") }).collect();
    assert_eq!(kinds, pattern);
    assert!(spec.layers.split);
    assert_eq!((spec.linear_layers(), spec.attention_layers()), (27, 4));
    assert_eq!(spec.kv_layers(), 4);
    assert_eq!(spec.attention_before(31), 3);
    assert_eq!(spec.mlp.activation, Activation::Mapped);
    assert!(spec.mlp.mapped.as_ref().is_some_and(|m| m.expr.contains("max(x, 0.0)")));
    assert_eq!(spec.gate_up_rows(), 15680);
    assert_eq!(spec.rope.nope_layers.len(), 56);
    assert_eq!(spec.norm.eps, 1e-5);
    let m = spec.mixer.as_ref().unwrap();
    assert_eq!((m.params["ssm_inner"], m.params["ssm_conv"], m.params["ssm_in"]), (10240.0, 12288.0, 22656.0));
    assert!(spec.has_mixer(0) && !spec.has_mlp(0) && spec.has_mlp(1) && !spec.has_mixer(1));
    spec.runnable().unwrap();
}

#[test]
fn split_layers_plan_one_residual_branch_each_with_a_mapped_up_projection() {
    let spec = serving(&config("nemotron-nano-9b-v2.config.json")).unwrap();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        let mixer = plan.layer(Block::Linear);
        assert_eq!(mixer.iter().filter(|o| matches!(o, Op::MixKernel { .. })).count(), 3, "{phase:?}");
        assert!(!mixer.iter().any(|o| matches!(o, Op::Linear { weight: Weight::GateUp, .. })), "a mixer layer has no mlp");
        let attention = plan.layer(Block::Full);
        assert!(attention.iter().any(|o| matches!(o, Op::Flash | Op::DecodeAttention)));
        assert!(!attention.iter().any(|o| matches!(o, Op::Linear { weight: Weight::GateUp, .. })), "an attention layer has no mlp");
        let mlp = plan.layer(Block::Mlp);
        assert_eq!(mlp.len(), 3, "{mlp:?}");
        assert!(matches!(mlp[1], Op::Linear { weight: Weight::GateUp, output: Buf::ActPlanes, .. }));
        assert!(matches!(mlp[2], Op::Linear { weight: Weight::Down, output: Buf::X, .. }));
        for ops in [mixer, attention, mlp] {
            assert_eq!(ops.iter().filter(|o| matches!(o, Op::Norm { .. })).count(), 1, "one pre-norm per layer");
        }
    }
    assert_eq!(Weight::GateUp.tensors(&spec), vec!["mlp.up_proj.weight".to_string()]);
    assert_eq!(Weight::GateUp.rows(&spec), 15680);
}

#[test]
fn the_nano_9b_module_builds_a_relu2_mapped_up_projection() {
    let spec = serving(&config("nemotron-nano-9b-v2.config.json")).unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    let line = module.source.lines().find(|l| l.starts_with("#define MAPPED_BODY")).unwrap();
    eprintln!("{line}");
    assert!(line.contains("fmaxf(x, 0") && !line.contains("activation("), "{line}");
    assert!(module.source.contains("gemm_tc<15680, 4480, ") && module.source.contains("EPI_MAP"), "gate_up is the up projection alone with a mapped epilogue");
    for k in ["mamba2_conv", "mamba2_scan", "mamba2_gated_norm"] {
        assert!(module.kernels.contains(&k), "{k}");
    }
}

#[test]
fn the_nemotron_3_nano_moe_config_is_supported_with_non_gated_experts() {
    let spec = serving(&config("nemotron-3-nano-30b-a3b.config.json")).unwrap_or_else(|e| panic!("{e:#}"));
    let m = spec.moe.as_ref().unwrap();
    assert_eq!((m.experts, m.top_k(), m.expert_intermediate, m.shared_intermediate, m.routed_scale), (128, 6, 1856, 3712, 2.5));
    assert!(m.score_bias && !m.shared_gate && m.router == specrt::spec::RouterAct::Sigmoid);
    assert!(spec.mlp.mapped.is_some());
    let sparse = (0..spec.layers.count).filter(|&i| spec.block(i) == Block::Mlp).count();
    assert_eq!(sparse, 23);
    spec.runnable().unwrap();
}

#[test]
fn the_moe_module_runs_one_up_stack_through_the_mapped_activation() {
    let spec = serving(&config("nemotron-3-nano-30b-a3b.config.json")).unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.source.contains("constexpr bool MOE_MAPPED = true;"));
    assert!(module.source.contains("constexpr int MOE_UP_WEIGHTS = 1;"));
    assert!(module.source.contains("mapped_act(u)"));
}

#[test]
fn non_gated_experts_on_quantised_weights_are_refused() {
    let mut spec = serving(&config("nemotron-3-nano-30b-a3b.config.json")).unwrap();
    spec.mlp.quant = Some(specrt::spec::Quant::Nvfp4G16);
    assert!(spec.check_moe_runnable().is_err());
}

#[test]
fn the_mamba_mixer_layers_count_their_parameters_in_the_anatomy() {
    let spec = serving(&config("nemotron-nano-9b-v2.config.json")).unwrap();
    let a = specrt::anatomy::ModelAnatomy::new(&spec, 0, None);
    let mixer = a.layers.iter().find(|l| l.kind == specrt::anatomy::LayerKind::DeltaNet).unwrap();
    let h = 4480u64;
    assert!(mixer.mixer_params >= h * (22656 + 10240), "{}", mixer.mixer_params);
}

#[test]
fn a_pattern_that_disagrees_with_the_layer_count_is_refused() {
    let mut v: serde_json::Value = serde_json::from_str(&config("nemotron-nano-9b-v2.config.json")).unwrap();
    v["hybrid_override_pattern"] = "M-*".into();
    let err = serving(&v.to_string()).unwrap_err();
    assert!(format!("{err:#}").contains("names 3 layers but num_hidden_layers is 56"), "{err:#}");
    v["hybrid_override_pattern"] = "M-*X".repeat(14).into();
    assert!(format!("{:#}", serving(&v.to_string()).unwrap_err()).contains("layer type X is not supported"));
}

#[test]
fn the_tiny_nemotron_h_checkpoint_loads_through_the_family_tensor_names() {
    let spec = tiny();
    let ck = specrt::weights::Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    for i in 0..spec.layers.count {
        let layer = ck.layer(&spec, i).unwrap_or_else(|e| panic!("layer {i}: {e:#}"));
        let has_mixer = !matches!(layer.mixer, specrt::weights::Mixer::None);
        assert_eq!(has_mixer, spec.block(i) != Block::Mlp, "layer {i}");
        assert_eq!(layer.ln1, layer.ln2, "layer {i} has one norm");
    }
}

fn reference(name: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR from scripts/tiny_nemotronh.py"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn matches_transformers(name: &str) {
    let (tokens, greedy, vocab, want) = reference(name);
    let spec = tiny_named(name);
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let agree = (0..tokens.len()).filter(|r| top(&ours[r * vocab..(r + 1) * vocab]) == top(&want[r * vocab..(r + 1) * vocab])).count();
    eprintln!("{name} prefill max_abs {max_abs} argmax agree {agree}/{}", tokens.len());
    assert!(max_abs < 0.1, "{name} max abs diff {max_abs}");
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "{name} decode greedy");
}

#[test]
#[ignore = "needs a CUDA GPU and SPECRT_REF_DIR from scripts/tiny_nemotronh.py"]
fn tiny_nemotron_h_matches_transformers() {
    matches_transformers(TINY);
}

#[test]
#[ignore = "needs a CUDA GPU and SPECRT_REF_DIR from scripts/tiny_nemotronh.py"]
fn tiny_moe_nemotron_h_matches_transformers() {
    matches_transformers(TINY_MOE);
}

#[test]
#[ignore = "needs a CUDA GPU and SPECRT_REF_DIR from scripts/tiny_nemotronh.py"]
fn tiny_moe_nemotron_h_on_tensor_core_tiles_matches_transformers() {
    matches_transformers("tiny-nemotronh-moe-mma");
}
