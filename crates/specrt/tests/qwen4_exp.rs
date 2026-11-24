use serde_json::Value;
use specrt::spec::{AttentionKind, Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-qwen4-exp";

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
fn the_tiny_qwen4_exp_config_describes_hyper_connections_the_pooled_indexer_and_the_moe_hybrid() {
    let spec = Spec::describe_hf_config(&config()).unwrap();
    assert_eq!(spec.layers.pattern, vec![Block::Linear, Block::Linear, Block::Linear, Block::Full]);
    let hyper = spec.hyper.as_ref().expect("hc_count names hyper-connections");
    assert_eq!(hyper.streams, 4);
    let index = spec.attention.indexer.as_ref().expect("indexer_budget names an indexer");
    assert_eq!((index.heads, index.head_dim, index.topk, index.kpool), (4, 64, 16, 4));
    assert_ne!(spec.attention.kind, AttentionKind::Dsa);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (8, Some(2), 32, 32));
    assert!(spec.attention.output_gate);
}

#[test]
fn the_fixture_checkpoint_uses_the_real_tensor_names() {
    let bytes = std::fs::read(fixture().join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let shape = |name: &str| st.tensor(name).unwrap_or_else(|_| panic!("{name} missing")).shape().to_vec();
    let p = "model.language_model.";
    assert_eq!(shape(&format!("{p}layers.0.attn_hyper_connection.input_mix_weight_down.weight")), vec![16, 4 * 128]);
    assert_eq!(shape(&format!("{p}layers.0.attn_hyper_connection.block_inject_weight.weight")), vec![4, 4 * 128]);
    assert_eq!(shape(&format!("{p}hyper_connection_mixer.input_mix_weight_up.weight")), vec![4 * 128, 16]);
    assert_eq!(shape(&format!("{p}layers.3.self_attn.indexer.index_qk_proj.weight")), vec![5 * 64, 128]);
    assert_eq!(shape(&format!("{p}layers.1.ple.key_proj.weight")), vec![4 * 128, 128]);
    assert_eq!(shape(&format!("{p}layers.1.ple.conv1d.weight")), vec![4 * 128, 1, 4]);
    assert_eq!(shape(&format!("{p}layers.1.ple.ple_embedding.ngram_embedding.weight"))[1], 32);
    assert_eq!(shape(&format!("{p}layers.3.mlp.experts.7.down_proj.weight")), vec![128, 32]);
    assert_eq!(shape("lm_head.weight"), vec![512, 128]);
    let size: u64 = std::fs::read_dir(fixture()).unwrap().flatten().chain(std::fs::read_dir(fixture().join("reference")).unwrap().flatten()).map(|e| e.metadata().unwrap().len()).sum();
    assert!(size < 5 * 1024 * 1024, "{size} bytes");
}

#[test]
fn the_reference_loads_with_finite_prefill_decode_and_hidden_streams() {
    let m = meta(&format!("{NAME}.ref.json"));
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    let tokens = m["tokens"].as_array().unwrap().len();
    let steps = m["decode_steps"].as_u64().unwrap() as usize;
    assert_eq!((vocab, tokens, steps), (512, 30, 4));
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0);
    let prefill = floats(&format!("{NAME}.ref.bin"));
    assert_eq!(prefill.len(), tokens * vocab);
    assert!(finite(&prefill));
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;
    assert_eq!(top(&prefill[(tokens - 1) * vocab..]), m["greedy"][0].as_u64().unwrap());
    let decode = floats(&format!("{NAME}.decode.logits.bin"));
    assert_eq!(decode.len(), steps * vocab);
    for s in 0..steps {
        assert_eq!(top(&decode[s * vocab..(s + 1) * vocab]), m["greedy"][s + 1].as_u64().unwrap(), "decode step {s}");
    }
    let hidden = floats(&format!("{NAME}.hidden.streams.bin"));
    assert_eq!(hidden.len(), 3 * tokens * 4 * 128);
    assert!(finite(&hidden));
    assert!(m["decode_vs_recompute"].as_f64().unwrap() < 1e-3);
}

#[test]
fn the_engine_refuses_qwen4_exp_until_its_pieces_exist() {
    let refusal = Spec::from_hf_config(&config()).unwrap_err().to_string();
    assert!(refusal.contains("model_type qwen4_exp is not supported"), "{refusal}");
}
