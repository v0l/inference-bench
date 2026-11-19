use specrt::family::Registry as Families;
use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::Runtime;
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

const BAMBA: &str = "tiny-bamba";
const GRANITE_HYBRID: &str = "tiny-granitehybrid";

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
#[ignore = "needs a CUDA GPU and SPECRT_REF_DIR from scripts/tiny_granitehybrid.py"]
fn tiny_granitehybrid_matches_transformers() {
    let (tokens, greedy, vocab, want) = reference(GRANITE_HYBRID);
    let spec = spec(GRANITE_HYBRID);
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap();
    let ours = rt.prompt_logits(&tokens).unwrap();
    assert_eq!(ours.len(), want.len());
    let max_abs = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{GRANITE_HYBRID} prefill max_abs {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let agree = (0..tokens.len()).filter(|r| top(&ours[r * vocab..(r + 1) * vocab]) == top(&want[r * vocab..(r + 1) * vocab])).count();
    eprintln!("{GRANITE_HYBRID} prefill argmax agree {agree}/{}", tokens.len());
    assert!(max_abs < 0.05, "{GRANITE_HYBRID} max abs diff {max_abs}");
    let got = rt.generate_batch(std::slice::from_ref(&tokens), greedy.len()).unwrap();
    assert_eq!(got.tokens[0], greedy, "{GRANITE_HYBRID} decode greedy");
}

fn kda_reference(mix: &[f32], dtb: &[f32], a_log: &[f32], heads: usize, d: usize, rows: usize, lower: f32) -> (Vec<f32>, Vec<f32>) {
    let row = 4 * heads * d + heads;
    let mut state = vec![0f32; heads * d * d];
    let mut out = vec![0f32; rows * heads * d];
    let norm = |v: &[f32]| {
        let n = (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        v.iter().map(|x| x / n).collect::<Vec<f32>>()
    };
    for t in 0..rows {
        let r = &mix[t * row..(t + 1) * row];
        for h in 0..heads {
            let seg = |o: usize| &r[o * heads * d + h * d..o * heads * d + (h + 1) * d];
            let q: Vec<f32> = norm(seg(0)).iter().map(|x| x / (d as f32).sqrt()).collect();
            let k = norm(seg(1));
            let v = seg(2);
            let f = seg(3);
            let beta = 1.0 / (1.0 + (-r[4 * heads * d + h]).exp());
            let s = &mut state[h * d * d..(h + 1) * d * d];
            for i in 0..d {
                let g = (lower / (1.0 + (-(a_log[h].exp() * (f[i] + dtb[h * d + i]))).exp())).exp();
                for j in 0..d {
                    s[i * d + j] *= g;
                }
            }
            for j in 0..d {
                let mem: f32 = (0..d).map(|i| s[i * d + j] * k[i]).sum();
                let delta = (v[j] - mem) * beta;
                for i in 0..d {
                    s[i * d + j] += k[i] * delta;
                }
                out[t * heads * d + h * d + j] = (0..d).map(|i| s[i * d + j] * q[i]).sum();
            }
        }
    }
    (out, state)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_kda_kernel_matches_the_kimi_delta_rule() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    let (heads, d, rows, lower) = (4usize, 64usize, 12usize, -5.0f32);
    let row = 4 * heads * d + heads;
    let text = std::fs::read_to_string(root().join("specs/kda_recurrence.toml")).unwrap();
    let shape = kir::desc::Shape::new().num("kda_head_dim", d as f64).num("kda_heads", heads as f64).num("kda_row", row as f64).num("kda_lower_bound", lower as f64).num("linear_layers", 1.0);
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let k = kir::desc::Description::parse(&text).unwrap().build(&target, &shape).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some(Box::leak(format!("compute_{}{}", ctx.compute_capability().unwrap().0, ctx.compute_capability().unwrap().1).into_boxed_str())), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(kir::cuda::emit_module(std::slice::from_ref(&k)).unwrap(), opts).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let f = module.load_function("kda_recurrence").unwrap();
    let mut seed = 808u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as u32) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    };
    let mix: Vec<f32> = (0..rows * row).map(|_| rand()).collect();
    let dtb: Vec<f32> = (0..heads * d).map(|_| rand()).collect();
    let a_log: Vec<f32> = (0..heads).map(|_| rand() * 0.5).collect();
    let (want, want_state) = kda_reference(&mix, &dtb, &a_log, heads, d, rows, lower);
    let stream = ctx.default_stream();
    let (dmix, ddtb, da) = (stream.clone_htod(&mix).unwrap(), stream.clone_htod(&dtb).unwrap(), stream.clone_htod(&a_log).unwrap());
    let y = stream.alloc_zeros::<f32>(rows * heads * d).unwrap();
    let st = stream.alloc_zeros::<f32>(heads * d * d).unwrap();
    let slots = stream.alloc_zeros::<i32>(1).unwrap();
    let (layer, base, step, per) = (0i32, 0i32, 0i32, rows as i32);
    let mut b = stream.launch_builder(&f);
    b.arg(&dmix).arg(&ddtb).arg(&da).arg(&y).arg(&st).arg(&slots).arg(&layer).arg(&base).arg(&step).arg(&per);
    unsafe { b.launch(LaunchConfig { grid_dim: (heads as u32, 1, 1), block_dim: (k.threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    let got = stream.clone_dtoh(&y).unwrap();
    let got_state = stream.clone_dtoh(&st).unwrap();
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!("kda out max_abs {} state max_abs {}", diff(&got, &want), diff(&got_state, &want_state));
    assert!(diff(&got, &want) < 1e-4 && diff(&got_state, &want_state) < 1e-4);
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

fn tiny_config(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap()
}

#[test]
fn granite_h_tiny_is_a_mamba_hybrid_with_moe_and_scaled_residuals_from_its_family_toml() {
    let json = tiny_config("granite-4.0-h-tiny.config.json");
    let spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(spec.layers.count, 40);
    assert_eq!(spec.linear_layers(), 36);
    assert_eq!(spec.block(5), Block::Full);
    assert_eq!(spec.block(4), Block::Linear);
    let m = spec.moe.as_ref().unwrap();
    assert_eq!((m.experts, m.top_k(), m.expert_intermediate, m.shared_intermediate, m.shared_slots()), (64, 6, 512, 1024, 2));
    assert!(m.fused_experts && !m.shared_gate && m.norm_topk);
    assert_eq!((m.router_module.as_str(), m.shared_module.as_str()), ("router", "shared_mlp"));
    assert_eq!(m.out_scale, 0.22);
    assert_eq!(spec.norm.residual_scale, Some(0.22));
    assert!(spec.rope.nope_layers.len() == 40);
    assert_eq!(spec.dims.embed_scale, Some(12.0));
    assert_eq!(spec.dims.logit_scale, Some(1.0 / 6.0));
    assert_eq!(spec.attention.query_scalar, Some(1.0 / (0.0078125f64 * 0.0078125)));
    let mixer = spec.mixer.as_ref().unwrap();
    assert_eq!((mixer.params["ssm_inner"], mixer.params["ssm_conv"], mixer.params["ssm_in"]), (3072.0, 3328.0, 6448.0));
    spec.validate().unwrap();
}

#[test]
fn granite_h_tiny_plans_scaled_mixer_and_attention_blocks_with_experts() {
    let json = tiny_config("granite-4.0-h-tiny.config.json");
    let spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        for block in [Block::Linear, Block::Full] {
            let ops = plan.layer_of(block, true);
            assert!(ops.contains(&Op::MoeRoute) && ops.contains(&Op::MoeExperts) && ops.contains(&Op::MoeCombine), "{block:?} {phase:?}");
            assert!(ops.iter().any(|o| matches!(o, Op::ScaledAdd { .. })), "{block:?} {phase:?}");
        }
        assert_eq!(plan.layer_of(Block::Linear, true).iter().filter(|o| matches!(o, Op::MixKernel { .. })).count(), 3);
    }
}

#[test]
fn granite_h_tiny_module_builds_the_scaled_moe_combine_and_mixer_kernels() {
    let json = tiny_config("granite-4.0-h-tiny.config.json");
    let spec = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap();
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.source.contains("MOE_OUT_SCALE = 2.2e-1f"), "the combine kernel carries the residual multiplier");
    for k in ["mamba2_conv", "mamba2_scan", "mamba2_gated_norm", "moe_combine", "residual_add"] {
        assert!(module.kernels.contains(&k), "{k}");
    }
}

#[test]
fn the_tiny_granite_hybrid_config_reads_experts_scales_and_nope_from_its_family_toml() {
    let spec = spec(GRANITE_HYBRID);
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Full]);
    let m = spec.moe.as_ref().unwrap();
    assert_eq!((m.experts, m.top_k(), m.expert_intermediate, m.shared_slots(), m.out_scale), (8, 2, 256, 2, 0.22));
    assert_eq!(spec.rope.nope_layers, vec![0, 1, 2, 3]);
    assert!(spec.dims.tie_embeddings);
    spec.runnable().unwrap();
}

#[test]
fn the_tiny_granite_hybrid_checkpoint_loads_through_the_family_tensor_names() {
    let spec = spec(GRANITE_HYBRID);
    let ck = specrt::weights::Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    for i in 0..4 {
        let layer = ck.layer(&spec, i).unwrap_or_else(|e| panic!("layer {i}: {e:#}"));
        let moe = layer.moe.as_ref().unwrap();
        assert_eq!(moe.router.len(), 8 * 256);
        let specrt::weights::Stacked::Bf16(down) = &moe.down else { panic!("bf16 experts") };
        assert_eq!(down.len(), (8 + 2) * 256 * 256);
    }
}
