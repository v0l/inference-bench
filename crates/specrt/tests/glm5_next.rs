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
    let refusal = Spec::from_hf_config(&json).unwrap_err().to_string();
    assert!(refusal.contains("model_type glm5_next is not supported"), "{refusal}");
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
    assert!(spec.runnable().is_err(), "the engine does not run glm5_next yet");
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

struct Run {
    logits: Vec<Vec<f32>>,
    trace: Vec<Vec<Vec<Vec<f32>>>>,
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
