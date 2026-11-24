use serde_json::Value;
use specrt::family::Registry;
use specrt::plan::{Buf, GATED_MODULE, Op, Phase, Plan};
use specrt::spec::{AttentionKind, Block, Spec, EMBEDDED};
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

fn hidden_streams() -> Vec<f32> {
    let m = meta(&format!("{NAME}.hidden.json"));
    let scales: Vec<f32> = m["scales"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let bytes = std::fs::read(fixture().join("reference").join(format!("{NAME}.hidden.streams.i16"))).unwrap();
    let q: Vec<i16> = bytes.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect();
    let each = q.len() / scales.len();
    q.iter().enumerate().map(|(i, v)| *v as f32 * scales[i / each]).collect()
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
    let hidden = hidden_streams();
    assert_eq!(hidden.len(), 3 * tokens * 4 * 128);
    assert!(finite(&hidden));
    let scales = meta(&format!("{NAME}.hidden.json"))["scales"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).fold(0.0f64, f64::max);
    assert!(scales / 2.0 < 1e-3 / 4.0, "the int16 rounding {} must stay well under the 1e-3 bar", scales / 2.0);
    assert!(m["decode_vs_recompute"].as_f64().unwrap() < 1e-3);
}

#[test]
fn the_engine_refuses_qwen4_exp_until_its_pieces_exist() {
    let refusal = Spec::from_hf_config(&config()).unwrap_err().to_string();
    assert!(refusal.contains("model_type qwen4_exp is not supported"), "{refusal}");
}

fn harness_family() -> String {
    std::fs::read_to_string(fixture().join("hyper-harness.toml")).unwrap()
}

fn harness_with(family: &str) -> anyhow::Result<Spec> {
    let registry = Registry::from_texts(EMBEDDED.iter().map(|(_, t)| *t).chain([family]))?;
    let mut s = Spec::from_hf_config_serving_in(&registry, &config())?;
    s.repo = fixture().display().to_string();
    s.runtime.split_activations = Some(true);
    Ok(s)
}

pub fn harness() -> Spec {
    harness_with(&harness_family()).unwrap()
}

#[test]
fn the_harness_reads_the_low_rank_gate_from_the_config_and_the_spec() {
    let s = harness();
    let h = s.gated_residual().unwrap();
    assert_eq!((h.streams, h.params["hc_lowrank"]), (4, 16.0));
    assert!(s.hyper.is_none(), "the glm5 [hyper] table is not read for the gated residual");
    assert_eq!(h.sites, vec!["attn_hyper_connection", "mlp_hyper_connection"]);
    assert_eq!(h.module, "hyper_connection_mixer");
    assert!(!s.norm.pre && !s.uses_branch());
    assert_eq!(s.layers.pattern, vec![Block::Linear, Block::Linear, Block::Linear, Block::Full]);
}

#[test]
fn a_gated_residual_is_checked_against_the_spec() {
    let bad = |from: &str, to: &str| format!("{:#}", harness_with(&harness_family().replacen(from, to, 1)).unwrap_err());
    assert!(bad("[norm]\npre = false", "[norm]\npre = true").contains("block norms"));
    let both = harness_family().replace("[norm]\npre = false", "[norm]\npre = false\n\n[hyper]\nstreams = \"hc_count\"");
    assert!(format!("{:#}", harness_with(&both).unwrap_err()).contains("[hyper]"));
    assert!(bad("op = \"post\"", "op = \"pre\"").contains("Post"));
    assert!(bad("b = { buffer = \"hc_normed\" }", "b = { buffer = \"nowhere\" }").contains("nowhere"));
    assert!(bad("hc_inject = \"hc_streams\"", "hc_inject = \"hc_nothing\"").contains("hc_nothing"));
    let missing = harness_with(&harness_family().replacen("kernel = \"hc_up\"", "kernel = \"hc_missing\"", 1)).unwrap();
    let e = specrt::codegen::gated_kernels(&missing, &kir::Target::sm_120()).unwrap_err().to_string();
    assert!(e.contains("hc_missing"), "{e}");
}

#[test]
fn the_plan_wraps_every_block_in_the_four_stream_residual() {
    let s = harness();
    let h = s.gated_residual().unwrap();
    let site_ops = |ops: &[Op], site: u8| ops.iter().filter(|o| matches!(o, Op::Gated { site: s, .. } if *s == site)).count();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&s, phase, true);
        assert_eq!(plan.head.iter().filter(|o| matches!(o, Op::Gated { site: GATED_MODULE, .. })).count(), 1);
        assert!(!plan.tail.iter().any(|o| matches!(o, Op::Norm { .. })), "{:?}", plan.tail);
        assert_eq!(site_ops(&plan.tail, GATED_MODULE), h.node.iter().filter(|n| n.op == "pre" && n.when.is_none()).count());
        for block in [Block::Linear, Block::Full] {
            let ops = plan.layer_of(block, true);
            assert!(!ops.iter().any(|o| matches!(o, Op::Norm { .. })), "{block:?} {ops:?}");
            assert_eq!(site_ops(ops, 0), site_ops(ops, 1));
            assert_eq!(site_ops(ops, 0), h.node.iter().filter(|n| n.op != "expand").count());
            assert!(ops.contains(&Op::Clear { buf: Buf::X }));
        }
        specrt::signature::validate_plan(&plan, &s).unwrap();
    }
}

#[test]
fn the_hyper_kernels_come_from_kir_descriptions_and_compile_into_the_module() {
    let s = harness();
    let names: Vec<&str> = specrt::codegen::gated_kernels(&s, &kir::Target::sm_120()).unwrap().iter().map(|k| k.name).collect();
    assert_eq!(names, ["hc_expand", "hc_norm", "hc_down", "hc_act", "hc_up", "hc_inject", "hc_mix", "hc_inject_add"]);
    let m = specrt::codegen::Module::generate_kv(&s, 256, 2, &kir::Target::sm_120()).unwrap();
    let src = m.source.as_str();
    for n in names {
        assert!(src.contains(&format!(" {n}(")), "{n} missing");
    }
    assert!(src.contains("constexpr bool MOE_INPUT_NORM = false;"));
    assert!(src.contains("o + av * 2.0f * ") || src.contains("(o + ((av * 2.0f) *"), "inject add text");
}

#[test]
fn the_loader_reads_each_site_and_the_mixer_but_no_block_norms() {
    let s = harness();
    let ck = specrt::weights::Checkpoint::open(&fixture()).unwrap();
    let l = ck.layer(&s, 0).unwrap();
    assert_eq!(l.gated.len(), 2);
    assert!(l.hyper.is_empty());
    let names: Vec<&str> = l.gated[1].iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight"]);
    assert_eq!(l.gated[1][1].1.len(), 16 * 512);
    let head: Vec<String> = ck.gated_head(&s).unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(head, ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"]);
    assert_eq!(l.ln2.len(), 1);
}

fn ple_inject(rows: usize) -> std::collections::BTreeMap<usize, Vec<f32>> {
    let m = meta(&format!("{NAME}.ple.json"));
    let layer = m["layer"].as_u64().unwrap() as usize;
    let width = m["shape"][1].as_u64().unwrap() as usize;
    let out = floats(&format!("{NAME}.ple.out.bin"));
    [(layer, out[..rows * width].to_vec())].into_iter().collect()
}

fn unselected_rows(tokens: usize) -> usize {
    let v: Value = serde_json::from_str(&config()).unwrap();
    let t = &v["text_config"];
    let (budget, ratio) = (t["indexer_budget"].as_u64().unwrap() as usize, t["indexer_compress_ratio"].as_u64().unwrap() as usize);
    (0..tokens).take_while(|p| (p + 1) / ratio <= budget / ratio).count()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_four_stream_residual_matches_the_reference_streams_of_layers_0_1_and_3() {
    let s = harness();
    let m = meta(&format!("{NAME}.ref.json"));
    let tokens: Vec<u32> = m["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let width = 4 * s.dims.hidden;
    let mut rt = specrt::runtime::Runtime::load(&s, 256).unwrap();
    let trace = rt.trace_streams(&tokens, &ple_inject(tokens.len())).unwrap();
    let want = hidden_streams();
    let n = tokens.len();
    let exact = unselected_rows(n);
    assert!(exact > 4 && exact < n, "{exact}");
    let mut worst = Vec::new();
    for (k, layer) in [0usize, 1, 3].into_iter().enumerate() {
        let rows = if layer == 3 { exact } else { n };
        let ours = &trace.layers[layer * n * width..layer * n * width + rows * width];
        let theirs = &want[k * n * width..k * n * width + rows * width];
        let scale = theirs.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = max_diff(ours, theirs);
        eprintln!("layer {layer}: {rows} rows, max abs diff {d:e}, max abs value {scale}");
        worst.push(d);
    }
    assert!(worst.iter().all(|&d| d <= 1e-3), "{worst:?}");
    let without = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let d1 = max_diff(&without.layers[n * width..2 * n * width], &want[n * width..2 * n * width]);
    assert!(d1 > 1e-2, "layer 1 should need the injected PLE delta, got {d1}");
}
