use specrt::plan::{Buf, NormKind, Op, Phase, Plan};
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
use specrt::spec::{Block, Spec};
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}

fn spec(name: &str) -> Spec {
    let json = std::fs::read_to_string(fixture(name).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = name.into();
    spec.repo = fixture(name).display().to_string();
    spec
}

struct Reference {
    tokens: Vec<u32>,
    greedy: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn reference(name: &str) -> Reference {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    Reference {
        tokens: ids("tokens"),
        greedy: ids("greedy"),
        vocab: meta["vocab"].as_u64().unwrap() as usize,
        logits: bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
    }
}

fn check(name: &str, tolerance: f32) {
    let want = reference(name);
    let mut rt = Runtime::load(&spec(name), 256).unwrap();
    let ours = rt.prompt_logits(&want.tokens).unwrap();
    assert_eq!(ours.len(), want.logits.len());
    let max_abs = ours.iter().zip(&want.logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{name} max_abs {max_abs}");
    assert!(max_abs < tolerance, "{name} max abs diff {max_abs}");
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    for r in 0..want.tokens.len() {
        let span = r * want.vocab..(r + 1) * want.vocab;
        assert_eq!(top(&ours[span.clone()]), top(&want.logits[span]), "{name} row {r}");
    }
    let got = rt.generate(&want.tokens, want.greedy.len(), Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, want.greedy, "{name} greedy");
}

fn serve(name: &str) {
    let want = reference(name);
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    let layout = Layout { max_ctx, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&spec(name), layout).unwrap();
    rt.set_block_table(0, &(0..pages as u32).collect::<Vec<_>>()).unwrap();
    let mut st = rt.stepper().unwrap();
    let mut first = None;
    let mut start = 0;
    while start < want.tokens.len() {
        let end = (start + 64).min(want.tokens.len());
        let last = end == want.tokens.len();
        first = rt.prefill_slot(&mut st, 0, &want.tokens[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap();
        start = end;
    }
    let mut out = vec![first.unwrap()];
    while out.len() < want.greedy.len() {
        let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: want.tokens.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
        out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
    }
    assert_eq!(out, want.greedy, "{name} served greedy");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn every_family_serves_the_reference_greedy_tokens() {
    for name in ["tiny-mistral", "tiny-phi3", "tiny-smollm3", "tiny-olmo2", "tiny-granite"] {
        serve(name);
    }
}

#[test]
fn mistral_is_the_llama_family_with_sliding_window_off() {
    let s = spec("tiny-mistral");
    assert_eq!(s.layers.window, None);
    assert_eq!(s.layers.pattern, vec![Block::Dense]);
    assert!(!s.norm.sandwich && s.norm.pre && s.norm.residual_scale.is_none());
}

#[test]
fn phi3_reads_fused_tensors_and_partial_rotary_from_the_family() {
    let s = spec("tiny-phi3");
    assert!(s.attention.fused_qkv && s.mlp.fused_gate_up);
    assert_eq!(s.rotary_dim(), 16);
    let plan = Plan::build(&s, Phase::Prefill, true);
    let tensors: Vec<String> = plan.layer(Block::Dense).iter().flat_map(|o| o.tensors(&s)).collect();
    assert!(tensors.contains(&"self_attn.qkv_proj.weight".to_string()));
    assert!(tensors.contains(&"mlp.gate_up_proj.weight".to_string()));
    assert!(!tensors.iter().any(|t| t.contains("q_proj") || t.contains("gate_proj")));
}

#[test]
fn smollm3_marks_every_fourth_layer_without_rope() {
    let s = spec("tiny-smollm3");
    assert_eq!(s.rope.nope_layers, vec![1, 3]);
    assert!(s.nope(1) && !s.nope(0));
    let real = std::fs::read_to_string(root().join("tests/fixtures/smollm3-3b.config.json")).unwrap();
    let r = Spec::from_hf_config(&real).unwrap();
    assert_eq!(r.rope.nope_layers, (0..36).filter(|i| i % 4 == 3).collect::<Vec<_>>());
}

#[test]
fn olmo2_is_post_norm_with_full_width_qk_norm() {
    let s = spec("tiny-olmo2");
    assert!(s.norm.sandwich && !s.norm.pre && s.attention.qk_norm && s.attention.qk_norm_full);
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&s, phase, true);
        let ops = plan.layer(Block::Dense);
        assert!(!ops.iter().any(|o| matches!(o, Op::Norm { which: NormKind::Input | NormKind::PreMlp, .. })));
        assert!(ops.contains(&Op::QkNorm));
        assert_eq!(ops.iter().filter(|o| matches!(o, Op::NormAdd { .. })).count(), 2);
        assert_eq!(ops.iter().filter(|o| matches!(o, Op::Split { from: Buf::X, .. })).count(), 2);
    }
}

#[test]
fn granite_multipliers_come_from_the_config() {
    let s = spec("tiny-granite");
    assert_eq!(s.dims.embed_scale, Some(12.0));
    assert_eq!(s.dims.logit_scale, Some(0.125));
    assert_eq!(s.norm.residual_scale, Some(0.22));
    assert_eq!(s.attention.query_scalar, Some(64.0));
    let plan = Plan::build(&s, Phase::Decode, true);
    assert_eq!(plan.layer(Block::Dense).iter().filter(|o| matches!(o, Op::ScaledAdd { .. })).count(), 2);
    assert!(plan.tail.contains(&Op::LogitScale));
}

#[test]
fn no_family_spec_adds_an_architecture_to_the_engine() {
    let r = specrt::family::Registry::embedded();
    for model_type in ["mistral", "phi3", "smollm3", "olmo2", "granite"] {
        assert!(r.for_model_type(model_type).is_some(), "{model_type}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn tiny_mistral_matches_transformers() {
    check("tiny-mistral", 2e-3);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn tiny_phi3_matches_transformers() {
    check("tiny-phi3", 2e-3);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn tiny_smollm3_matches_transformers() {
    check("tiny-smollm3", 2e-3);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn tiny_olmo2_matches_transformers() {
    check("tiny-olmo2", 2e-3);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn tiny_granite_matches_transformers() {
    check("tiny-granite", 2e-3);
}
