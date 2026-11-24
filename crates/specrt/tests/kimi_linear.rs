use serde_json::Value;
use specrt::spec::{AttentionKind, Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-kimi-linear";
const TOLERANCE: f32 = 3.8e-4;
const PREFILL_TOLERANCE: f32 = 7.0e-4;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures").join(NAME)
}

fn config() -> String {
    std::fs::read_to_string(fixture().join("config.json")).unwrap()
}

fn floats(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("reference").join(file)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(file)).unwrap()).unwrap()
}

fn engine_spec() -> Spec {
    let mut spec = Spec::from_hf_config_serving(&config()).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = fixture().display().to_string();
    spec
}

#[test]
fn the_tiny_kimi_linear_config_serves_kda_mla_and_sigmoid_experts() {
    let spec = engine_spec();
    spec.runnable().unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Linear, Block::Linear, Block::Full], "full_attn_layers and kda_layers are 1 indexed");
    assert_eq!(spec.attention.kind, AttentionKind::Mla);
    assert_eq!(spec.rope.nope_layers, vec![0, 1, 2, 3], "mla_use_nope: no layer rotates");
    assert_eq!(spec.attention.head_dim, 64, "MLA heads are nope plus rope wide, not the config's head_dim 72");
    assert!(!spec.norm.unit_offset);
    let mixer = spec.mixer.as_ref().expect("kda mixer");
    assert_eq!((mixer.params["kda_heads"], mixer.params["kda_head_dim"], mixer.params["kda_taps"]), (2.0, 64.0, 4.0));
    let kernels: Vec<&str> = mixer.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert!(kernels.contains(&"kda_softplus_recurrence") && !kernels.contains(&"kda_recurrence"), "{kernels:?}");
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (8, Some(2), 32, 32));
    assert_eq!((moe.dense_layers.clone(), moe.shared_gate, moe.norm_topk), (vec![0], false, true));
    assert!((moe.routed_scale - 2.446).abs() < 1e-9);
    assert!(spec.speculative().is_none());
}

#[test]
fn the_kimi_linear_plan_and_module_build() {
    use specrt::plan::{Op, Phase, Plan};
    let spec = engine_spec();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        for (block, sparse) in [(Block::Linear, false), (Block::Linear, true), (Block::Full, true)] {
            let ops = plan.layer_of(block, sparse);
            assert!(!ops.is_empty(), "{phase:?} {block:?} {sparse}");
            assert_eq!(ops.iter().any(|o| matches!(o, Op::MoeRoute)), sparse, "{phase:?} {block:?}");
        }
        specrt::signature::validate_plan(&plan, &spec).unwrap_or_else(|e| panic!("{phase:?}: {e:#}"));
    }
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    for k in ["kda_softplus_recurrence", "kda_conv_q", "kda_gated_norm", "mla_attention", "mla_absorb_q", "mla_expand_v"] {
        assert!(module.kernels.contains(&k) && module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(spec.attention.node.iter().filter_map(|n| n.kernel.as_deref()).all(|k| !k.contains("rope")));
}

#[test]
fn the_softplus_gate_decays_by_exp_minus_a_times_softplus() {
    let text = std::fs::read_to_string(root().join("specs/kda_softplus_recurrence.toml")).unwrap();
    let d = kir::desc::Description::parse(&text).unwrap();
    let shape = kir::desc::Shape::new()
        .num("kda_head_dim", 64.0)
        .num("kda_heads", 2.0)
        .num("kda_row", 384.0)
        .num("kda_gate_row", 128.0)
        .num("kda_gate_offset", 0.0)
        .num("kda_beta_row", 514.0)
        .num("kda_beta_offset", 512.0)
        .num("linear_layers", 3.0);
    let k = d.build(&kir::Target::sm_120(), &shape).unwrap_or_else(|e| panic!("{e}"));
    let src = kir::cuda::emit_module(std::slice::from_ref(&k)).unwrap();
    assert!(src.contains("kda_softplus_recurrence("), "{src}");
    assert!(!src.contains("lower"), "no gate lower bound on the Kimi gate");
}

#[test]
fn the_fixture_checkpoint_uses_the_kimi_tensor_names() {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let shape = |name: &str| st.tensor(name).unwrap_or_else(|_| panic!("{name} missing")).shape().to_vec();
    assert_eq!(shape("model.layers.0.self_attn.q_conv1d.weight"), vec![128, 1, 4]);
    assert_eq!(shape("model.layers.0.self_attn.A_log"), vec![1, 1, 2, 1]);
    assert_eq!(shape("model.layers.1.block_sparse_moe.experts.7.w1.weight"), vec![32, 128]);
    assert_eq!(shape("model.layers.1.block_sparse_moe.experts.7.w2.weight"), vec![128, 32]);
    assert_eq!(shape("model.layers.3.self_attn.q_proj.weight"), vec![256, 128]);
    assert_eq!(shape("model.layers.3.self_attn.kv_a_proj_with_mqa.weight"), vec![80, 128]);
    assert_eq!(shape("model.layers.3.block_sparse_moe.gate.e_score_correction_bias"), vec![8]);
    let size: u64 = std::fs::read_dir(fixture()).unwrap().flatten().chain(std::fs::read_dir(fixture().join("reference")).unwrap().flatten()).map(|e| e.metadata().unwrap().len()).sum();
    assert!(size < 5 * 1024 * 1024, "{size} bytes");
}

#[test]
fn the_reference_has_finite_prefill_decode_and_hidden_outputs() {
    let m = meta(&format!("{NAME}.ref.json"));
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    let tokens = m["tokens"].as_array().unwrap().len();
    assert_eq!((vocab, tokens, m["greedy"].as_array().unwrap().len()), (512, 30, 5));
    let prefill = floats(&format!("{NAME}.ref.bin"));
    assert_eq!(prefill.len(), tokens * vocab);
    assert!(prefill.iter().all(|x| x.is_finite()));
    assert_eq!(floats(&format!("{NAME}.decode.logits.bin")).len(), 4 * vocab);
    assert_eq!(floats(&format!("{NAME}.hidden.streams.bin")).len(), 3 * tokens * 128);
}

fn weight(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor(name).unwrap();
    t.data().as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect()
}

fn host_head(hidden: &[f32]) -> Vec<f32> {
    let (norm, head) = (weight("model.norm.weight"), weight("lm_head.weight"));
    hidden
        .chunks(128)
        .flat_map(|row| {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / 128.0 + 1e-5).sqrt();
            let x: Vec<f32> = row.iter().zip(&norm).map(|(v, w)| v / rms * w).collect();
            head.chunks(128).map(|w| w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>()).collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn the_real_kimi_linear_48b_config_serves() {
    let json = std::fs::read_to_string(root().join("tests/fixtures/kimi-linear-48b.config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.runnable().unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(spec.layers.count, 27);
    assert_eq!(spec.layers.pattern.len(), 27, "layer 27 breaks the 3:1 rhythm, so the pattern is the full list");
    assert_eq!((0..27).filter(|&l| spec.block(l) == Block::Full).collect::<Vec<_>>(), vec![3, 7, 11, 15, 19, 23, 26]);
    assert_eq!(spec.attention.head_dim, 192, "nope 128 plus rope 64, not the config's head_dim 72");
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (256, Some(8), 1024, 1024));
}

#[test]
fn kimi_k3_configs_name_each_missing_piece() {
    for (dir, model_type) in [("moonshotai__Kimi-K3", "kimi_k3"), ("moonshotai__Kimi-K3.1-Pro", "kimi_k31")] {
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/specrt-popular/files").join(dir).join("config.json");
        let Ok(json) = std::fs::read_to_string(&path) else {
            eprintln!("{} missing, skipped", path.display());
            continue;
        };
        let err = format!("{:#}", Spec::from_hf_config_serving(&json).unwrap_err());
        assert!(!err.contains(&format!("model_type {model_type} is not supported")), "{err}");
        assert!(err.contains("Kimi K3"), "{err}");
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let t = v["text_config"].as_object_mut().unwrap();
        for k in ["attn_res_block_size", "routed_expert_hidden_size", "mla_use_output_gate", "q_lora_rank", "activation_situ_beta"] {
            t.remove(k);
        }
        t.insert("hidden_act".into(), "silu".into());
        t["linear_attn_config"].as_object_mut().unwrap().remove("use_full_rank_gate");
        let stripped = Spec::from_hf_config_serving(&v.to_string()).unwrap_or_else(|e| panic!("{model_type} without the K3 pieces: {e:#}"));
        stripped.runnable().unwrap_or_else(|e| panic!("{e:#}"));
        assert_eq!(stripped.layers.count, 93);
        assert_eq!(stripped.attention.head_dim, 192);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_matches_transformers_on_hidden_states_prefill_and_four_decode_steps() {
    use specrt::runtime::{DecodeRow, Runtime, Sampling};
    let spec = engine_spec();
    let m = meta(&format!("{NAME}.ref.json"));
    let ids = |k: &str| -> Vec<u32> { m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (tokens, greedy) = (ids("tokens"), ids("greedy"));
    let vocab = 512;
    let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
    let trace = rt.trace_prefill(&tokens).unwrap();
    let per = tokens.len() * 128;
    let saved = floats(&format!("{NAME}.hidden.streams.bin"));
    let mut failed = Vec::new();
    for (slot, layer) in [(0, 0), (1, 1), (2, 3)] {
        let g = gap(&trace.layers[layer * per..(layer + 1) * per], &saved[slot * per..(slot + 1) * per]);
        eprintln!("layer {layer} hidden max abs {g}");
        if g >= TOLERANCE {
            failed.push(format!("layer {layer} hidden max abs {g}"));
        }
    }
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let p = gap(&trace.logits, &prefill);
    eprintln!("engine prefill max abs {p}");
    let last = &trace.layers[3 * per..4 * per];
    let f32_head = gap(&host_head(last), &prefill);
    eprintln!("f32 head on the engine's layer 3 hidden against the reference max abs {f32_head}");
    if p >= PREFILL_TOLERANCE {
        failed.push(format!("prefill max abs {p}"));
    }
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
        let d = gap(&got, &decode[s * vocab..(s + 1) * vocab]);
        eprintln!("decode step {s} max abs {d}, greedy {} (reference {})", next[0], greedy[s + 1]);
        if d >= TOLERANCE {
            failed.push(format!("decode step {s} max abs {d}"));
        }
        assert_eq!(next[0], greedy[s + 1], "decode step {s}");
    }
    drop(rt);
    let mut bounded = engine_spec();
    let mixer = bounded.mixer.as_mut().unwrap();
    mixer.params.insert("kda_lower_bound".into(), -5.0);
    for n in mixer.node.iter_mut().filter(|n| n.kernel.as_deref() == Some("kda_softplus_recurrence")) {
        n.kernel = Some("kda_recurrence".into());
    }
    let mut rt = Runtime::load_batched(&bounded, 256, 1).unwrap();
    let wrong = rt.trace_prefill(&tokens).unwrap();
    let g = gap(&wrong.layers[..per], &saved[..per]);
    eprintln!("lower bounded sigmoid gate in place of softplus: layer 0 hidden max abs {g}");
    assert!(g > TOLERANCE, "the GLM-5 gate still matches the Kimi reference ({g})");
    assert!(failed.is_empty(), "{failed:?}");
}
