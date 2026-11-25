use serde_json::Value;
use specrt::spec::{AttentionKind, Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-kimi-k3";
const HIDDEN: usize = 128;
const TOLERANCE: f32 = 3.8e-4;
const LOGIT_TOLERANCE: f32 = 1.0e-4;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures").join(NAME)
}

fn config() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap()
}

fn spec_of(json: &str) -> Spec {
    let mut spec = Spec::from_hf_config_serving(json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = fixture().display().to_string();
    spec
}

fn engine_spec() -> Spec {
    spec_of(&config().to_string())
}

fn floats(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("reference").join(file)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(format!("{NAME}.ref.json"))).unwrap()).unwrap()
}

fn weight(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor(&format!("language_model.{name}")).unwrap();
    t.data().as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect()
}

fn rms(row: &[f32]) -> f32 {
    (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + 1e-5).sqrt()
}

fn host_tail(tokens: &[u32], layers: &[f32]) -> Vec<f32> {
    let per = tokens.len() * HIDDEN;
    let embed = weight("model.embed_tokens.weight");
    let score: Vec<f32> = weight("model.output_attn_res_norm.weight").iter().zip(weight("model.output_attn_res_proj.weight")).map(|(a, b)| a * b).collect();
    let (norm, head) = (weight("model.norm.weight"), weight("lm_head.weight"));
    let mut out = Vec::new();
    for (t, &tok) in tokens.iter().enumerate() {
        let row = |l: usize| &layers[l * per + t * HIDDEN..l * per + (t + 1) * HIDDEN];
        let e = &embed[tok as usize * HIDDEN..(tok as usize + 1) * HIDDEN];
        let values = [e, row(1), row(3), row(5)];
        let scores: Vec<f32> = values.iter().map(|v| v.iter().zip(&score).map(|(a, b)| a * b).sum::<f32>() / rms(v)).collect();
        let m = scores.iter().fold(f32::MIN, |a, b| a.max(*b));
        let z: f32 = scores.iter().map(|s| (s - m).exp()).sum();
        let mixed: Vec<f32> = (0..HIDDEN).map(|i| values.iter().zip(&scores).map(|(v, s)| (s - m).exp() / z * v[i]).sum()).collect();
        let r = rms(&mixed);
        let x: Vec<f32> = mixed.iter().zip(&norm).map(|(v, w)| v / r * w).collect();
        out.extend(head.chunks(HIDDEN).map(|w| w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>()));
    }
    out
}

#[test]
fn situ_comes_from_the_family_as_a_gated_expression_with_the_config_betas() {
    let spec = engine_spec();
    let g = spec.mlp.gated.as_ref().expect("situ is a gated expression");
    assert_eq!((g.params["beta"], g.params["linear_beta"]), (4.0, 25.0));
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    assert!(module.source.contains("#define GATED_BODY(g, u) (") && module.source.contains("tanhf(g / 4e0f)") && module.source.contains("tanhf(u / 2.5e1f)"), "situ lowered into the gated body");
    let mut v = config();
    v["text_config"]["hidden_act"] = "silu".into();
    assert!(spec_of(&v.to_string()).mlp.gated.is_none(), "silu keeps the plain swiglu");
}

#[test]
fn mla_q_lora_and_the_output_gate_are_attention_nodes() {
    let spec = engine_spec();
    assert_eq!(spec.attention.kind, AttentionKind::Mla);
    assert_eq!(spec.qkv_rows(), 64 + 64 + 16 + 4 * 32, "q_a, kv_a and g_proj share the qkv GEMM");
    assert_eq!(spec.o_dim(), 4 * 32);
    let kernels: Vec<&str> = spec.attention.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert_eq!(kernels, ["mla_q_lora_norm", "mla_kv_lora_norm", "dsa_q_b", "dsa_latent_write", "dsa_absorb_q", "mla_lora_attention", "mla_expand_v_rows", "mla_out_gate"]);
    let mixer = spec.mixer.as_ref().unwrap();
    let projections: Vec<&str> = mixer.projection.iter().map(|p| p.tensor.as_str()).collect();
    assert_eq!(projections, ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_proj", "b_proj"]);
    assert_eq!(spec.mix_in_rows(), 4 * 128 + 64 + 2);
    let mixed: Vec<&str> = mixer.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert!(mixed.contains(&"kda_recurrence") && mixed.contains(&"kda_gated_norm_mix") && !mixed.contains(&"kda_out_gate_up"), "{mixed:?}");
    assert_eq!(mixer.params["kda_lower_bound"], -5.0);
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Linear, Block::Linear, Block::Full, Block::Linear, Block::Full]);
}

#[test]
fn latent_experts_and_block_residuals_come_from_the_family() {
    let spec = engine_spec();
    let moe = spec.moe.as_ref().unwrap();
    let latent = moe.latent.as_ref().expect("latent experts");
    assert_eq!((latent.width, latent.down.as_str(), latent.up.as_str(), latent.norm.as_deref()), (64, "routed_expert_down_proj", "routed_expert_up_proj", Some("routed_expert_norm")));
    assert_eq!((moe.shared_intermediate, moe.shared_slots(), moe.pairs()), (64, 0, 2), "the shared expert runs at full width outside the routed slots");
    let r = spec.block_residual.as_ref().expect("block residuals");
    assert_eq!((r.size, r.blocks(spec.layers.count)), (2, 3));
    let mut v = config();
    v["text_config"].as_object_mut().unwrap().remove("routed_expert_hidden_size");
    let err = format!("{:#}", Spec::from_hf_config_serving(&v.to_string()).unwrap_err());
    assert!(err.contains("[block_residual]") && err.contains("latent"), "{err}");
}

#[test]
fn the_k3_plan_validates_and_the_module_builds() {
    use specrt::plan::{Op, Phase, Plan, ResStage};
    let spec = engine_spec();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        for (block, sparse) in [(Block::Linear, false), (Block::Linear, true), (Block::Full, true)] {
            let ops = plan.layer_of(block, sparse);
            assert_eq!(ops.first(), Some(&Op::BlockRes { stage: ResStage::Enter }), "{phase:?} {block:?} {sparse}");
            assert_eq!(ops.last(), Some(&Op::BlockRes { stage: ResStage::Done }), "{phase:?} {block:?} {sparse}");
            assert_eq!(ops.contains(&Op::LatentNorm), sparse, "{phase:?} {block:?}");
        }
        assert!(plan.tail.contains(&Op::BlockRes { stage: ResStage::Tail }));
        specrt::signature::validate_plan(&plan, &spec).unwrap_or_else(|e| panic!("{phase:?}: {e:#}"));
    }
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    for k in ["block_res_mix", "block_res_push", "block_res_out", "latent_norm", "mla_out_gate", "kda_gated_norm_mix", "moe_combine"] {
        assert!(module.kernels.contains(&k) && module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(module.source.contains("constexpr int MOE_WIDTH = 64;") && module.source.contains("constexpr bool MOE_ASSIGN = true;"));
}

#[test]
fn the_fixture_checkpoint_uses_the_k3_tensor_names() {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let shape = |name: &str| st.tensor(&format!("language_model.model.{name}")).unwrap_or_else(|_| panic!("{name} missing")).shape().to_vec();
    assert_eq!(shape("layers.0.self_attn.g_proj.weight"), vec![128, 128]);
    assert_eq!(shape("layers.3.self_attn.q_a_proj.weight"), vec![64, 128]);
    assert_eq!(shape("layers.3.self_attn.g_proj.weight"), vec![128, 128]);
    assert_eq!(shape("layers.1.block_sparse_moe.routed_expert_down_proj.weight"), vec![64, 128]);
    assert_eq!(shape("layers.1.block_sparse_moe.experts.7.w1.weight"), vec![64, 64]);
    assert_eq!(shape("layers.1.block_sparse_moe.shared_experts.gate_proj.weight"), vec![64, 128]);
    assert_eq!(shape("layers.2.mlp_res_proj.weight"), vec![1, 128]);
    assert_eq!(shape("output_attn_res_norm.weight"), vec![128]);
    let size: u64 = std::fs::read_dir(fixture()).unwrap().flatten().chain(std::fs::read_dir(fixture().join("reference")).unwrap().flatten()).map(|e| e.metadata().unwrap().len()).sum();
    assert!(size < 5 * 1024 * 1024, "{size} bytes");
}

fn real(dir: &str) -> Option<Value> {
    let path = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/specrt-popular/files").join(dir).join("config.json");
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(serde_json::from_str(&s).unwrap()),
        Err(_) => {
            eprintln!("{} missing, skipped", path.display());
            None
        }
    }
}

#[test]
fn the_real_k3_config_serves_and_k3_1_names_its_sparse_attention() {
    if let Some(v) = real("moonshotai__Kimi-K3") {
        let spec = Spec::from_hf_config_serving(&v.to_string()).unwrap_or_else(|e| panic!("{e:#}"));
        spec.runnable().unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(spec.layers.count, 93);
        assert_eq!(spec.qkv_rows(), 1536 + 512 + 64 + 96 * 128);
        assert_eq!(spec.mix_in_rows(), 4 * 96 * 128 + 128 + 96);
        assert_eq!(spec.moe.as_ref().unwrap().width(spec.dims.hidden), 3584);
        assert_eq!(spec.block_residual.as_ref().unwrap().blocks(93), 8);
        for phase in [specrt::plan::Phase::Prefill, specrt::plan::Phase::Decode] {
            specrt::signature::validate_plan(&specrt::plan::Plan::build(&spec, phase, true), &spec).unwrap_or_else(|e| panic!("{phase:?}: {e:#}"));
        }
    }
    if let Some(v) = real("moonshotai__Kimi-K3.1-Pro") {
        let err = format!("{:#}", Spec::from_hf_config_serving(&v.to_string()).unwrap_err());
        assert!(err.contains("Kimi K3.1 sparse attention (dsa_config)"), "{err}");
        let mut v = v;
        v["text_config"].as_object_mut().unwrap().remove("dsa_config");
        Spec::from_hf_config_serving(&v.to_string()).unwrap_or_else(|e| panic!("K3.1 without DSA: {e:#}")).runnable().unwrap();
    }
}

#[test]
fn the_reference_has_finite_prefill_decode_and_hidden_outputs() {
    let m = meta();
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    let tokens = m["tokens"].as_array().unwrap().len();
    assert_eq!((vocab, tokens, m["greedy"].as_array().unwrap().len()), (512, 30, 5));
    assert!(m["stub_vs_transformers_on_tiny_kimi_linear"]["prefill"].as_f64().unwrap() < 1e-4);
    let prefill = floats(&format!("{NAME}.ref.bin"));
    assert_eq!(prefill.len(), tokens * vocab);
    assert!(prefill.iter().all(|x| x.is_finite()));
    assert_eq!(floats(&format!("{NAME}.decode.logits.bin")).len(), 4 * vocab);
    assert_eq!(floats(&format!("{NAME}.hidden.streams.bin")).len(), 6 * tokens * HIDDEN);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_matches_the_remote_code_on_hidden_states_prefill_and_four_decode_steps() {
    use specrt::runtime::{DecodeRow, Runtime, Sampling};
    let spec = engine_spec();
    let m = meta();
    let ids = |k: &str| -> Vec<u32> { m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (tokens, greedy) = (ids("tokens"), ids("greedy"));
    let vocab = 512;
    let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let scale = |a: &[f32]| a.iter().fold(0f32, |m, x| m.max(x.abs()));
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * HIDDEN;
    let saved = floats(&format!("{NAME}.hidden.streams.bin"));
    let mut failed = Vec::new();
    for layer in 0..6 {
        let want = &saved[layer * per..(layer + 1) * per];
        let g = gap(&trace.layers[layer * per..(layer + 1) * per], want);
        eprintln!("layer {layer} hidden max abs {g} (values up to {:.2})", scale(want));
        if g >= TOLERANCE {
            failed.push(format!("layer {layer} hidden {g}"));
        }
    }
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let p = gap(&trace.logits, &prefill) / scale(&prefill);
    eprintln!("engine prefill max abs relative to the largest logit {p} (logits up to {:.1})", scale(&prefill));
    eprintln!("f32 host tail on the reference layers against the reference max abs {}", gap(&host_tail(&tokens, &saved), &prefill));
    eprintln!("f32 host tail on the engine layers against the reference max abs {}", gap(&host_tail(&tokens, &trace.layers), &prefill));
    eprintln!("f32 host tail on the engine layers against the engine max abs {}", gap(&host_tail(&tokens, &trace.layers), &trace.logits));
    if p >= LOGIT_TOLERANCE {
        failed.push(format!("prefill relative {p}"));
    }
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let tops = |l: &[f32]| -> Vec<u32> { l.chunks(vocab).map(top).collect() };
    assert_eq!(tops(&trace.logits), tops(&prefill), "prefill greedy tokens");
    let decode = floats(&format!("{NAME}.decode.logits.bin"));
    let mut st = rt.stepper().unwrap();
    let mut st_prefill = rt.stepper().unwrap();
    rt.prefill_slot(&mut st_prefill, 0, &tokens, 0, None).unwrap();
    for s in 0..4 {
        let row = DecodeRow { slot: 0, token: greedy[s], pos: tokens.len() + s, sampling: Sampling::GREEDY, step: s as u32 };
        let next = rt.decode(&mut st, &[row]).unwrap();
        let got = rt.step_logits(&st, 0, 1).unwrap();
        let want = &decode[s * vocab..(s + 1) * vocab];
        let d = gap(&got, want) / scale(want);
        eprintln!("decode step {s} max abs relative to the largest logit {d}, greedy {} (reference {})", next[0], greedy[s + 1]);
        if d >= LOGIT_TOLERANCE {
            failed.push(format!("decode step {s} relative {d}"));
        }
        assert_eq!(next[0], greedy[s + 1], "decode step {s}");
    }
    drop(rt);
    let mut silu = engine_spec();
    silu.mlp.gated = None;
    let mut rt = Runtime::load_batched(&silu, 256, 1).unwrap();
    let wrong = rt.trace_prefill(&tokens).unwrap();
    let g = gap(&wrong.layers[..per], &saved[..per]);
    eprintln!("silu in place of situ: layer 0 hidden max abs {g}");
    assert!(g > 100.0 * TOLERANCE, "plain silu still matches the SiTU reference ({g})");
    assert!(failed.is_empty(), "{failed:?}");
}
