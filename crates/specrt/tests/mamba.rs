use specrt::family::Registry as Families;
use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::Runtime;
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const BAMBA: &str = "tiny-bamba";

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}

fn spec(name: &str) -> Spec {
    let json = std::fs::read_to_string(fixture(name).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = name.into();
    spec.repo = fixture(name).display().to_string();
    spec
}

#[test]
fn the_bamba_config_becomes_a_described_mixer_from_the_family_toml() {
    let spec = spec(BAMBA);
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Full]);
    assert!(spec.linear_attention.is_none());
    let m = spec.mixer.as_ref().unwrap();
    assert_eq!((m.params["ssm_inner"], m.params["ssm_conv"], m.params["ssm_in"]), (512.0, 576.0, 1104.0));
    assert_eq!(spec.mix_in_rows(), 1104);
    assert_eq!(spec.mix_out_cols(), 512);
    assert_eq!(spec.rotary_dim(), 32);
    let ssm = specrt::state::named(&spec, "ssm").unwrap();
    assert_eq!(ssm.dims(&spec).unwrap(), vec![16, 16, 32]);
    assert_eq!(ssm.rollback, specrt::state::Rollback::Checkpoint);
    assert_eq!(specrt::state::named(&spec, "conv").unwrap().dims(&spec).unwrap(), vec![576, 3]);
    assert_eq!(specrt::state::sequence_bytes(&spec), 2 * (16 * 16 * 32 + 576 * 3) * 4);
    spec.runnable().unwrap();
}

#[test]
fn mixer_layers_plan_the_described_nodes_and_attention_layers_stay_stock() {
    let spec = spec(BAMBA);
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        let mix: Vec<&Op> = plan.layer(Block::Linear).iter().filter(|o| matches!(o, Op::MixKernel { .. })).collect();
        assert_eq!(mix.len(), 3, "{phase:?}");
        assert!(!plan.layer(Block::Linear).iter().any(|o| matches!(o, Op::GdnPrefill | Op::GdnDecode)));
        assert!(plan.layer(Block::Full).iter().any(|o| matches!(o, Op::Flash | Op::DecodeAttention)));
    }
}

#[test]
fn the_bamba_module_builds_its_mixer_kernels_from_spec_files() {
    let spec = spec(BAMBA);
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    for k in ["mamba2_conv", "mamba2_scan", "mamba2_gated_norm"] {
        assert!(module.kernels.contains(&k), "{k}");
        assert!(module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(!module.source.contains("gdn_chunk_scan("));
}

#[test]
#[ignore = "needs the NVRTC library (no GPU)"]
fn the_bamba_module_compiles_through_nvrtc() {
    let module = specrt::codegen::Module::generate_kv(&spec(BAMBA), 256, 1, &kir::Target::sm_120()).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&module.source, opts).unwrap_or_else(|e| panic!("{e:?}")).to_src();
    for k in ["mamba2_conv", "mamba2_scan", "mamba2_gated_norm"] {
        assert!(ptx.contains(&format!(".entry {k}(")), "{k}");
    }
}

#[test]
fn the_kda_description_builds_from_the_spec_file() {
    let text = std::fs::read_to_string(root().join("specs/kda_recurrence.toml")).unwrap();
    let d = kir::desc::Description::parse(&text).unwrap();
    let shape = kir::desc::Shape::new()
        .num("kda_head_dim", 128.0)
        .num("kda_heads", 64.0)
        .num("kda_row", 5.0 * 64.0 * 128.0 + 64.0)
        .num("kda_lower_bound", -5.0)
        .num("linear_layers", 34.0);
    let k = d.build(&kir::Target::sm_120(), &shape).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(k.threads, 512);
    let src = kir::cuda::emit_module(std::slice::from_ref(&k)).unwrap();
    assert!(src.contains("kda_recurrence("), "{src}");
}

fn reference(name: &str) -> (Vec<u32>, Vec<u32>, usize, Vec<f32>) {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    let logits = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (ids("tokens"), ids("greedy"), meta["vocab"].as_u64().unwrap() as usize, logits)
}

#[test]
#[ignore = "needs a CUDA GPU and SPECRT_REF_DIR from scripts/tiny_mamba.py"]
fn tiny_bamba_matches_transformers() {
    let (tokens, greedy, vocab, want) = reference(BAMBA);
    let spec = spec(BAMBA);
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{BAMBA} prefill max_abs {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let agree = (0..tokens.len()).filter(|r| top(&ours[r * vocab..(r + 1) * vocab]) == top(&want[r * vocab..(r + 1) * vocab])).count();
    eprintln!("{BAMBA} prefill argmax agree {agree}/{}", tokens.len());
    assert!(max_abs < 0.05, "{BAMBA} max abs diff {max_abs}");
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "{BAMBA} decode greedy");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_checkpointed_state_survives_a_reset_and_restore() {
    let spec = spec(BAMBA);
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let prompt: Vec<u32> = (0..40).map(|i| (i * 37 + 11) % 512).collect();
    rt.prompt_logits(&prompt).unwrap();
    let snap = rt.snapshot_state_blocking(0).unwrap();
    assert!(snap.ssm().iter().any(|v| *v != 0.0) && snap.conv().iter().any(|v| *v != 0.0));
    rt.reset_slot(0).unwrap();
    assert!(rt.snapshot_state_blocking(0).unwrap().ssm().iter().all(|v| *v == 0.0));
    rt.restore_state(0, &snap).unwrap();
    let back = rt.snapshot_state_blocking(0).unwrap();
    assert_eq!((back.conv(), back.ssm()), (snap.conv(), snap.ssm()));
}
