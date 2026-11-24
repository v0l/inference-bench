mod glm5_host;

use glm5_host::{Config, Host, Knobs, Weights};
use serde_json::Value;
use specrt::spec::{AttentionKind, Block, LinearKind, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-glm5-next";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(NAME)
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

#[test]
fn the_tiny_glm5_next_config_describes_hyper_connections_nope_pools_kda_and_dsa() {
    let json = config();
    Spec::from_hf_config(&json).unwrap_or_else(|e| panic!("{e:#}")).runnable().unwrap_or_else(|e| panic!("{e:#}"));
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Linear, Block::Linear, Block::Full]);
    let hyper = spec.hyper.as_ref().expect("hc_mult names hyper-connections");
    assert_eq!((hyper.streams, hyper.sinkhorn_iters, hyper.eps), (4, 20, 1e-6));
    assert_eq!(spec.rope.nope_layers, vec![0, 1, 2, 3], "qk_rope_head_dim 0 means no layer rotates");
    assert_eq!(spec.rope.theta, None);
    assert_eq!(spec.attention.kind, AttentionKind::Dsa);
    let index = spec.attention.indexer.as_ref().expect("index_topk names an indexer");
    assert_eq!((index.heads, index.head_dim, index.topk, index.kpool), (4, 32, 8, 4));
    let la = spec.linear_attention.as_ref().unwrap();
    assert_eq!(la.kind, LinearKind::Kda);
    assert_eq!((la.key_heads, la.value_heads, la.key_head_dim, la.conv_kernel), (2, 2, 64, 4));
    assert!(!spec.norm.unit_offset, "plain weight * x");
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (8, Some(2), 32, 32));
    assert_eq!((moe.dense_layers.clone(), moe.shared_gate, moe.routed_scale), (vec![0], false, 2.5));
    assert_eq!(spec.speculative().map(|m| m.layers), Some(1));
    let text = serde_json::to_string(&spec).unwrap();
    for piece in ["\"hyper\":{\"streams\":4", "\"kind\":\"dsa\"", "\"kpool\":4", "\"kind\":\"kda\"", "\"nope_layers\":[0,1,2,3]"] {
        assert!(text.contains(piece), "{piece} missing from {text}");
    }
}

#[test]
fn the_real_glm5_3_flash_config_describes_the_same_pieces() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5.3-flash.config.json")).unwrap();
    let spec = Spec::describe_hf_config(&json).unwrap();
    assert_eq!(spec.hyper.as_ref().map(|h| h.streams), Some(4));
    assert_eq!(spec.attention.kind, AttentionKind::Dsa);
    assert_eq!(spec.attention.indexer.as_ref().map(|i| (i.heads, i.head_dim, i.topk, i.kpool)), Some((32, 128, 2048, 4)));
    assert_eq!(spec.linear_attention.as_ref().map(|l| l.kind), Some(LinearKind::Kda));
    assert_eq!(spec.rope.nope_layers.len(), 45);
    assert!(!spec.norm.unit_offset);
    assert!(!spec.moe.as_ref().unwrap().shared_gate);
}

#[test]
fn the_fixture_checkpoint_uses_the_real_tensor_names_and_dtypes() {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let shape = |name: &str| st.tensor(name).unwrap_or_else(|_| panic!("{name} missing")).shape().to_vec();
    let dtype = |name: &str| st.tensor(name).unwrap().dtype();
    let p = "model.language_model.";
    assert_eq!(shape(&format!("{p}layers.0.hc_attn_fn")), vec![24, 4 * 128]);
    assert_eq!(shape(&format!("{p}layers.0.self_attn.q_conv1d.weight")), vec![128, 1, 4]);
    assert_eq!(shape(&format!("{p}layers.2.self_attn.f_b_proj.weight")), vec![128, 64]);
    assert_eq!(shape(&format!("{p}layers.3.self_attn.indexer.index_kpool_compress_ape")), vec![4, 32]);
    assert_eq!(shape(&format!("{p}layers.3.mlp.experts.7.down_proj.weight")), vec![128, 32]);
    assert_eq!(shape(&format!("{p}layers.4.eh_proj.weight")), vec![128, 256]);
    assert_eq!(shape("lm_head.weight"), vec![512, 128]);
    for f32_name in [format!("{p}norm.weight"), format!("{p}layers.0.post_attention_layernorm.weight"), format!("{p}layers.1.self_attn.A_log"), format!("{p}layers.3.hc_ffn_scale")] {
        assert_eq!(dtype(&f32_name), safetensors::Dtype::F32, "{f32_name}");
    }
    assert_eq!(dtype(&format!("{p}layers.4.post_attention_layernorm.weight")), safetensors::Dtype::BF16);
    assert!(st.names().iter().all(|n| !n.contains("mlp.experts.gate_up_proj") && !n.contains("self_attn.conv1d")), "per expert and per conv tensors, as the checkpoint stores them");
    let size: u64 = std::fs::read_dir(fixture()).unwrap().flatten().chain(std::fs::read_dir(fixture().join("reference")).unwrap().flatten()).map(|e| e.metadata().unwrap().len()).sum();
    assert!(size < 5 * 1024 * 1024, "{size} bytes");
}

#[test]
fn the_reference_loads_with_finite_prefill_decode_hidden_and_mtp_outputs() {
    let m = meta(&format!("{NAME}.ref.json"));
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    let tokens = m["tokens"].as_array().unwrap().len();
    let greedy = m["greedy"].as_array().unwrap().len();
    let steps = m["decode_steps"].as_u64().unwrap() as usize;
    assert_eq!((vocab, tokens, steps, greedy), (512, 30, 4, 5));
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0);
    let prefill = floats(&format!("{NAME}.ref.bin"));
    assert_eq!(prefill.len(), tokens * vocab);
    assert!(finite(&prefill));
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;
    assert_eq!(top(&prefill[(tokens - 1) * vocab..]), m["greedy"][0].as_u64().unwrap());
    let decode = floats(&format!("{NAME}.decode.logits.bin"));
    assert_eq!(decode.len(), steps * vocab);
    assert!(finite(&decode));
    for s in 0..steps {
        assert_eq!(top(&decode[s * vocab..(s + 1) * vocab]), m["greedy"][s + 1].as_u64().unwrap(), "decode step {s}");
    }
    let h = meta(&format!("{NAME}.hidden.json"));
    assert_eq!(h["layers"], serde_json::json!([0, 1, 3]));
    let hidden = floats(&format!("{NAME}.hidden.streams.bin"));
    assert_eq!(hidden.len(), 3 * tokens * 4 * 128);
    assert!(finite(&hidden));
    let mtp = meta(&format!("{NAME}.mtp.json"));
    let logits = floats(&format!("{NAME}.mtp.logits.bin"));
    let chain = floats(&format!("{NAME}.mtp.chain.bin"));
    assert_eq!((logits.len(), chain.len()), (tokens * vocab, vocab));
    assert!(finite(&logits) && finite(&chain));
    assert_eq!(top(&logits[(tokens - 1) * vocab..]), mtp["draft"].as_u64().unwrap());
    assert_eq!(top(&chain), mtp["chain_draft"].as_u64().unwrap());
}

const TOLERANCE: f32 = 3.8e-4;
const BF16_SPLIT_TOLERANCE: f32 = 5.0e-4;

fn engine_spec() -> Spec {
    let mut spec = Spec::from_hf_config_serving(&config()).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = fixture().display().to_string();
    spec
}

#[test]
fn the_engine_family_plans_hyper_connections_and_no_rope_nodes() {
    use specrt::plan::{Op, Phase, Plan};
    let spec = engine_spec();
    spec.runnable().unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(spec.hyper.as_ref().map(|h| h.sites.len()), Some(2));
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&spec, phase, true);
        assert_eq!(plan.head.last(), Some(&Op::HyperExpand), "{phase:?}");
        assert_eq!(plan.tail.first(), Some(&Op::HyperMean), "{phase:?}");
        for (block, sparse) in [(Block::Linear, false), (Block::Linear, true), (Block::Full, true)] {
            let ops = plan.layer_of(block, sparse);
            assert_eq!(ops.iter().filter(|o| matches!(o, Op::HyperPre { .. })).count(), 2, "{block:?} {sparse}");
            assert_eq!(ops.iter().filter(|o| matches!(o, Op::HyperPost)).count(), 2, "{block:?} {sparse}");
            assert!(!ops.iter().any(|o| matches!(o, Op::KvRope | Op::QkRope | Op::DecodeRope)), "{block:?}");
        }
        specrt::signature::validate_plan(&plan, &spec).unwrap_or_else(|e| panic!("{phase:?}: {e:#}"));
    }
    let kernels: Vec<&str> = spec.attention.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert!(kernels.iter().all(|k| !k.contains("rope")), "{kernels:?}");
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    for k in ["hyper_pre", "hyper_post", "hyper_expand", "hyper_mean", "pool_index_topk", "dsa_latent_attention", "kda_recurrence", "kda_conv_q", "kda_gated_norm"] {
        assert!(module.kernels.contains(&k) && module.source.contains(&format!("{k}(")), "{k}");
    }
    let mixer = specrt::codegen::mixer_kernels(&spec, 1, 256, &kir::Target::sm_120()).unwrap();
    let kda = mixer.iter().find(|k| k.name == "kda_recurrence").unwrap();
    assert!(kda.chunk_form().is_none(), "KDA prefill runs the recurrent form: the chunked lowering takes one read stage and KDA reads the state twice");
}

#[test]
#[ignore = "needs the NVRTC library (no GPU)"]
fn the_engine_module_compiles_through_nvrtc() {
    let module = specrt::codegen::Module::generate_kv(&engine_spec(), 256, 1, &kir::Target::sm_120()).unwrap();
    let opts = cudarc::nvrtc::CompileOptions { arch: Some("compute_120"), options: vec!["--std=c++17".into()], ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(&module.source, opts).unwrap_or_else(|e| panic!("{e:?}")).to_src();
    for k in ["hyper_pre", "hyper_post", "pool_index_topk", "dsa_latent_attention", "kda_recurrence", "kda_gated_norm"] {
        assert!(ptx.contains(&format!(".entry {k}(")), "{k}");
    }
}

struct Run {
    logits: Vec<Vec<f32>>,
    trace: Vec<Vec<Vec<Vec<f32>>>>,
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_matches_the_reference_streams_prefill_and_four_decode_steps() {
    use specrt::runtime::{DecodeRow, Runtime, Sampling};
    let spec = engine_spec();
    let m = meta(&format!("{NAME}.ref.json"));
    let ids = |k: &str| -> Vec<u32> { m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let (tokens, greedy) = (ids("tokens"), ids("greedy"));
    let vocab = 512;
    let mut rt = Runtime::load_batched(&spec, 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
    let trace = rt.trace_prefill(&tokens).unwrap();
    let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let per = tokens.len() * 4 * 128;
    let saved = floats(&format!("{NAME}.hidden.streams.bin"));
    for (slot, layer) in [(0, 0), (1, 1), (2, 3)] {
        let g = gap(&trace.layers[layer * per..(layer + 1) * per], &saved[slot * per..(slot + 1) * per]);
        eprintln!("layer {layer} streams max abs {g}");
        assert!(g < TOLERANCE, "layer {layer} streams max abs {g}");
    }
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let tops = |logits: &[f32]| -> Vec<u32> { logits.chunks(vocab).map(top).collect() };
    let p = gap(&trace.logits, &prefill);
    eprintln!("engine prefill max abs {p}");
    let weights = Weights::load(&fixture().join("model.safetensors"));
    let c = Config::read(&config());
    let head_of = |streams: &[f32], round: fn(&[f32]) -> Vec<f32>| -> Vec<f32> {
        (0..tokens.len())
            .flat_map(|t| {
                let row = &streams[t * 4 * 128..(t + 1) * 4 * 128];
                let mean: Vec<f32> = (0..128).map(|d| (0..4).map(|s| row[s * 128 + d]).sum::<f32>() / 4.0).collect();
                let normed = glm5_host::rms(&mean, Some(weights.get("model.language_model.norm.weight")), c.eps);
                glm5_host::matvec(weights.get("lm_head.weight"), vocab, &round(&normed))
            })
            .collect()
    };
    let exact: fn(&[f32]) -> Vec<f32> = |x| x.to_vec();
    let one_plane: fn(&[f32]) -> Vec<f32> = |x| x.iter().map(|v| glm5_host::bf16(*v)).collect();
    let last = &trace.layers[3 * per..4 * per];
    let host_head = head_of(last, exact);
    let f32_head = gap(&host_head, &prefill);
    let planes_head = gap(&trace.logits, &head_of(last, glm5_host::planes));
    let one_plane_head = gap(&head_of(last, one_plane), &prefill);
    eprintln!("engine head against a host head with bf16 hi+lo planes on the engine's layer 3 streams max abs {planes_head}");
    eprintln!("f32 head on the engine's layer 3 streams against the reference max abs {f32_head}");
    eprintln!("one plane bf16 head input on the engine's layer 3 streams against the reference max abs {one_plane_head}");
    assert!(f32_head < TOLERANCE, "f32 head on the engine streams max abs {f32_head}");
    assert!(planes_head < 1e-4, "engine head against the two plane host head max abs {planes_head}");
    assert!(p < BF16_SPLIT_TOLERANCE, "engine prefill max abs {p}");
    assert!(one_plane_head > BF16_SPLIT_TOLERANCE, "a one plane bf16 head input stays inside the split bar ({one_plane_head})");
    assert_eq!(tops(&trace.logits), tops(&prefill), "prefill greedy tokens");
    assert_eq!(top(&trace.logits[(tokens.len() - 1) * vocab..]), greedy[0], "prefill greedy token");
    let mut failed = Vec::new();
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
        assert_eq!(top(&got), top(&decode[s * vocab..(s + 1) * vocab]), "decode step {s} logits argmax");
        assert_eq!(next[0], greedy[s + 1], "decode step {s}");
    }
    drop(rt);
    let wrong = |spec: &Spec, layer: usize, slot: usize| {
        let mut rt = Runtime::load_batched(spec, 256, 1).unwrap();
        let t = rt.trace_prefill(&tokens).unwrap();
        (gap(&t.layers[layer * per..(layer + 1) * per], &saved[slot * per..(slot + 1) * per]), gap(&t.logits, &prefill))
    };
    let mut short = engine_spec();
    short.hyper.as_mut().unwrap().sinkhorn_iters = 1;
    let (g, lp) = wrong(&short, 0, 0);
    eprintln!("one Sinkhorn iteration: layer 0 streams max abs {g}, prefill max abs {lp}");
    assert!(g > TOLERANCE && lp > BF16_SPLIT_TOLERANCE, "one Sinkhorn iteration still matches ({g}, {lp})");
    let mut tailless = engine_spec();
    tailless.attention.params.insert("index_tail".into(), 0.0);
    let (g, lp) = wrong(&tailless, 3, 2);
    eprintln!("tail pool dropped: layer 3 streams max abs {g}, prefill max abs {lp}");
    assert!(g > TOLERANCE && lp > BF16_SPLIT_TOLERANCE, "dropping the tail pool still matches ({g}, {lp})");
    assert!(failed.is_empty(), "{failed:?}");
}

#[test]
fn the_host_model_with_two_plane_bf16_inputs_shows_the_split_activation_rounding() {
    let m = meta(&format!("{NAME}.ref.json"));
    let greedy: Vec<u32> = m["greedy"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    glm5_host::PLANES.set(true);
    let run = host_run(|_| {}, &greedy[..4]);
    glm5_host::PLANES.set(false);
    let vocab = 512;
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let tokens = prefill.len() / vocab;
    let gap = |got: &[f32], want: &[f32]| got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let p = (0..tokens).map(|t| gap(&run.logits[t], &prefill[t * vocab..(t + 1) * vocab])).fold(0f32, f32::max);
    eprintln!("host with two plane bf16 matvec inputs: prefill max abs {p}");
    assert!(p > TOLERANCE, "the two plane rounding alone stays inside the f32 bar ({p})");
}

fn host_run(knobs: impl Fn(&mut Knobs), extra: &[u32]) -> Run {
    let weights = Weights::load(&fixture().join("model.safetensors"));
    let mut c = Config::read(&config());
    knobs(&mut c.knobs);
    let m = meta(&format!("{NAME}.ref.json"));
    let tokens: Vec<u32> = m["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mut host = Host::new(c, &weights);
    let mut run = Run { logits: vec![], trace: vec![] };
    for t in tokens.iter().chain(extra) {
        let mut trace = vec![];
        run.logits.push(host.step(*t, &mut trace));
        run.trace.push(trace);
    }
    run
}

fn saved_streams_gap(run: &Run, slot: usize, layer: usize) -> f32 {
    let saved = floats(&format!("{NAME}.hidden.streams.bin"));
    let per = 4 * 128;
    let tokens = saved.len() / 3 / per;
    (0..tokens)
        .flat_map(|t| {
            let want = &saved[(slot * tokens + t) * per..(slot * tokens + t + 1) * per];
            let got: Vec<f32> = run.trace[t][layer].iter().flatten().copied().collect();
            got.into_iter().zip(want.iter().copied()).collect::<Vec<_>>()
        })
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

#[test]
fn the_host_hyper_connections_match_the_saved_layer_0_streams_and_need_all_sinkhorn_iterations() {
    let gap = saved_streams_gap(&host_run(|_| {}, &[]), 0, 0);
    assert!(gap < TOLERANCE, "layer 0 streams max abs {gap}");
    for wrong in [1, 5] {
        let short = saved_streams_gap(&host_run(|k| k.sinkhorn_iters = wrong, &[]), 0, 0);
        assert!(short > TOLERANCE, "{wrong} Sinkhorn iterations still match ({short})");
    }
}

#[test]
fn the_host_kda_layers_match_the_saved_streams() {
    let run = host_run(|_| {}, &[]);
    for (slot, layer) in [(0, 0), (1, 1)] {
        let gap = saved_streams_gap(&run, slot, layer);
        assert!(gap < TOLERANCE, "layer {layer} streams max abs {gap}");
    }
}

#[test]
fn the_host_pooled_indexer_matches_layer_3_and_needs_the_open_tail_pool() {
    let gap = saved_streams_gap(&host_run(|_| {}, &[]), 2, 3);
    assert!(gap < TOLERANCE, "layer 3 streams max abs {gap}");
    let dropped = saved_streams_gap(&host_run(|k| k.tail = false, &[]), 2, 3);
    assert!(dropped > TOLERANCE, "dropping the tail pool still matches ({dropped})");
}

#[test]
fn the_host_model_matches_prefill_logits_and_four_decode_steps() {
    let m = meta(&format!("{NAME}.ref.json"));
    let greedy: Vec<u32> = m["greedy"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let run = host_run(|_| {}, &greedy[..4]);
    let vocab = 512;
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let decode = floats(&format!("{NAME}.decode.logits.bin"));
    let tokens = prefill.len() / vocab;
    let gap = |got: &[f32], want: &[f32]| got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let p = (0..tokens).map(|t| gap(&run.logits[t], &prefill[t * vocab..(t + 1) * vocab])).fold(0f32, f32::max);
    let d = (0..4).map(|s| gap(&run.logits[tokens + s], &decode[s * vocab..(s + 1) * vocab])).fold(0f32, f32::max);
    eprintln!("host prefill max abs {p}, decode max abs {d}");
    assert!(p < TOLERANCE && d < TOLERANCE, "prefill {p} decode {d}");
}
