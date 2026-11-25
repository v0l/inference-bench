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
fn the_engine_serves_qwen4_exp_from_its_family_spec() {
    let s = Spec::from_hf_config(&config()).unwrap_or_else(|e| panic!("{e:#}"));
    s.runnable().unwrap();
    assert_eq!(s.name, "qwen4_exp");
    assert!(s.mtp.is_none() && s.vision.is_none(), "text only, the tiny fixture has no MTP layer");
    assert!(s.refused.is_empty());
    let p = s.prelude.as_ref().expect("ple_layer_ids names a prelude");
    assert_eq!((p.module.as_str(), p.layers.clone()), ("ple", vec![1]));
    assert_eq!((p.params["ngram_heads"], p.params["ngram_rows"], p.params["ple_head_dim"], p.params["ple_conv_state"]), (4.0, 4096.0, 32.0, 9.0));
    let a = &s.attention.params;
    assert_eq!((a["index_heads"], a["index_head_dim"], a["index_groups"], a["index_out"], a["index_scale"]), (4.0, 64.0, 4.0, 19.0, 0.125));
    let state = |n: &str| specrt::state::named(&s, n).unwrap().dims(&s).unwrap();
    assert_eq!((state("ngram_history"), state("ple_conv"), state("index_pool")), (vec![2], vec![512, 9], vec![1, 64]));
}

#[test]
fn the_real_qwen3_8_flash_next_config_serves_text_with_offload_sized_pieces() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular/Qwen__Qwen3.8-Flash-Next/config.json")).unwrap();
    let s = Spec::from_hf_config_serving(&json).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!((s.layers.count, s.dims.hidden, (0..s.layers.count).filter(|&i| s.block(i) == Block::Full).count()), (48, 2560, 12));
    let p = s.prelude.as_ref().unwrap();
    assert_eq!((p.layers.clone(), p.params["ngram_heads"], p.params["ngram_rows"], p.params["ple_head_dim"]), (vec![1], 16.0, 320_001_536.0, 160.0));
    let a = &s.attention.params;
    assert_eq!((a["index_groups"], a["index_out"]), (512.0, 2051.0));
    let moe = s.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.active, moe.expert_intermediate, moe.shared_intermediate), (512, Some(10), 640, 640));
    assert_eq!(s.gated_residual().map(|h| (h.streams, h.params["hc_lowrank"])), Some((4, 320.0)));
    assert!(s.mtp.is_none() && s.vision.is_none());
    let why = s.refused_part("mtp").expect("the MTP layer is refused by name");
    assert!(why.contains("#1350"), "{why}");
    let mut on = s.clone();
    let e = specrt::serve::SpeculativeOptions { enabled: Some(true), ..Default::default() }.apply(&mut on).unwrap_err().to_string();
    assert!(e.contains("refuses its MTP layer") && e.contains("#1350"), "{e}");
    specrt::serve::SpeculativeOptions::default().apply(&mut s.clone()).unwrap();
}

fn engine() -> Spec {
    let mut s = Spec::from_hf_config_serving(&config()).unwrap_or_else(|e| panic!("{e:#}"));
    s.repo = fixture().display().to_string();
    s.runtime.split_activations = Some(true);
    s
}

fn harness_family() -> String {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/family-qwen4_exp.toml")).unwrap();
    text.replacen("name = \"qwen4_exp\"", "name = \"edited\"", 1)
}

fn harness_with(family: &str) -> anyhow::Result<Spec> {
    let registry = Registry::from_texts(EMBEDDED.iter().filter(|(n, _)| *n != "family-qwen4_exp.toml").map(|(_, t)| *t).chain([family]))?;
    let mut s = Spec::from_hf_config_serving_in(&registry, &config())?;
    s.repo = fixture().display().to_string();
    s.runtime.split_activations = Some(true);
    Ok(s)
}

pub fn harness() -> Spec {
    engine()
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
    assert!(s.linear_attention.as_ref().unwrap().gate_sigmoid, "output_gate_type sigmoid gates the delta-net output");
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
    assert!(src.contains("* la_sigmoid("), "the delta-net output gate is a sigmoid");
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

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

const TOLERANCE: f32 = 1e-3;

fn reference() -> (Vec<u32>, Vec<u32>, usize) {
    let m = meta(&format!("{NAME}.ref.json"));
    let ids = |k: &str| -> Vec<u32> { m[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    (ids("tokens"), ids("greedy"), m["vocab"].as_u64().unwrap() as usize)
}

fn top(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

#[test]
fn the_plan_runs_the_ple_prelude_before_the_attn_residual_of_each_block() {
    let s = engine();
    let p = s.prelude.as_ref().unwrap();
    for phase in [Phase::Prefill, Phase::Decode] {
        let plan = Plan::build(&s, phase, true);
        for block in [Block::Linear, Block::Full] {
            let ops = plan.layer_of(block, true);
            let first: Vec<&Op> = ops.iter().take(p.node.len()).collect();
            assert!(first.iter().enumerate().all(|(i, o)| **o == Op::Prelude { node: i as u8 }), "{block:?} {ops:?}");
            assert!(matches!(ops[p.node.len()], Op::Gated { site: 0, .. }));
        }
        specrt::signature::validate_plan(&plan, &s).unwrap();
    }
    let names: Vec<&str> = specrt::codegen::prelude_kernels(&s, 256, &kir::Target::sm_120()).unwrap().iter().map(|k| k.name).collect();
    assert_eq!(names, ["ple_ngram_hash", "ple_embed_gather", "ple_key_proj", "ple_value_proj", "ple_gate", "ple_conv", "ple_add"]);
    let m = specrt::codegen::Module::generate_kv(&s, 256, 2, &kir::Target::sm_120()).unwrap();
    for k in names.iter().chain(&["qsa_index_raw_write", "qsa_index_pool", "qsa_index_select", "qsa_attention", "hc_mix"]) {
        assert!(m.kernels.contains(k) && m.source.contains(&format!(" {k}(")), "{k} missing from the module");
    }
    let ck = specrt::weights::Checkpoint::open(&fixture()).unwrap();
    let names = |i: usize| -> Vec<String> { ck.layer(&s, i).unwrap().prelude.into_iter().map(|(n, _)| n).collect() };
    assert!(names(0).is_empty() && names(3).is_empty());
    assert_eq!(names(1).len(), 7);
    assert!(names(1).contains(&"ple_embedding.ngram_embedding.weight".to_string()));
}

#[test]
fn a_prelude_is_checked_against_the_spec() {
    let bad = |from: &str, to: &str| format!("{:#}", harness_with(&harness_family().replacen(from, to, 1)).unwrap_err());
    assert!(bad("ple_ids = \"ngram_heads\"", "ple_ids = \"ngram_nothing\"").contains("ngram_nothing"));
    assert!(bad("idx = { buffer = \"ple_ids\" }", "idx = { buffer = \"nowhere\" }").contains("nowhere"));
    assert!(bad("state = { state = \"ngram_history\" }", "state = { state = \"ssm\" }").contains("prelude layers"));
    assert!(bad("layers_base = 1", "layers_base = 3").contains("ple_layer_ids"));
    let wrong = harness_with(&harness_family().replacen("idx = { buffer = \"ple_ids\" }", "idx = { buffer = \"ple_emb\" }", 1)).unwrap();
    let e = specrt::codegen::prelude_kernels(&wrong, 256, &kir::Target::sm_120()).unwrap_err().to_string();
    assert!(e.contains("wrong kind"), "{e}");
    let mut two = config().replace("\"ple_layer_ids\": [\n      2\n    ]", "\"ple_layer_ids\": [2, 3]");
    assert!(two.contains("[2, 3]"));
    two = Spec::from_hf_config_serving(&two).unwrap_err().to_string();
    assert!(two.contains("2 layers"), "{two}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_four_stream_residual_matches_the_reference_streams_of_layers_0_1_and_3() {
    let s = harness();
    let (tokens, _, _) = reference();
    let width = 4 * s.dims.hidden;
    let mut rt = specrt::runtime::Runtime::load(&s, 256).unwrap();
    let trace = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let want = hidden_streams();
    let n = tokens.len();
    let mut worst = Vec::new();
    for (k, layer) in [0usize, 1, 3].into_iter().enumerate() {
        let ours = &trace.layers[layer * n * width..(layer + 1) * n * width];
        let theirs = &want[k * n * width..(k + 1) * n * width];
        let scale = theirs.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = max_diff(ours, theirs);
        eprintln!("layer {layer}: {n} rows, max abs diff {d:e}, max abs value {scale}");
        worst.push(d);
    }
    assert!(worst.iter().all(|&d| d <= TOLERANCE), "{worst:?}");
    drop(rt);
    let mut without = harness();
    without.prelude = None;
    let mut rt = specrt::runtime::Runtime::load(&without, 256).unwrap();
    let t = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let d1 = max_diff(&t.layers[n * width..2 * n * width], &want[n * width..2 * n * width]);
    assert!(d1 > 1e-2, "layer 1 should need the PLE prelude, got {d1}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_engine_matches_transformers_prefill_logits_and_four_greedy_decode_steps() {
    use specrt::runtime::{DecodeRow, Runtime, Sampling};
    let s = engine();
    let (tokens, greedy, vocab) = reference();
    let mut rt = Runtime::load_batched(&s, 256, 1).unwrap_or_else(|e| panic!("{e:#}"));
    let trace = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let prefill = floats(&format!("{NAME}.ref.bin"));
    let p = max_diff(&trace.logits, &prefill);
    eprintln!("prefill logits max abs {p:e}");
    assert!(p <= TOLERANCE, "prefill logits max abs {p}");
    let tops: Vec<u32> = trace.logits.chunks(vocab).map(top).collect();
    assert_eq!(tops, prefill.chunks(vocab).map(top).collect::<Vec<_>>(), "prefill argmax per row");
    let decode = floats(&format!("{NAME}.decode.logits.bin"));
    let mut st = rt.stepper().unwrap();
    let first = rt.prefill_slot(&mut st, 0, &tokens, 0, Some((Sampling::GREEDY, 0))).unwrap();
    assert_eq!(first, Some(greedy[0]), "served prefill greedy token");
    let mut failed = Vec::new();
    for step in 0..4 {
        let row = DecodeRow { slot: 0, token: greedy[step], pos: tokens.len() + step, sampling: Sampling::GREEDY, step: step as u32 + 1 };
        let next = rt.decode(&mut st, &[row]).unwrap();
        let got = rt.step_logits(&st, 0, 1).unwrap();
        let d = max_diff(&got, &decode[step * vocab..(step + 1) * vocab]);
        eprintln!("decode step {} logits max abs {d:e}, greedy {} (reference {})", step + 1, next[0], greedy[step + 1]);
        if d > TOLERANCE {
            failed.push(format!("decode step {} max abs {d}", step + 1));
        }
        assert_eq!(next[0], greedy[step + 1], "decode step {}", step + 1);
    }
    assert!(failed.is_empty(), "{failed:?}");
}

fn served_greedy(rt: &mut specrt::runtime::Runtime, prompts: &[(usize, &[u32])], steps: usize) -> Vec<Vec<u32>> {
    use specrt::runtime::{DecodeRow, Sampling};
    let mut st = rt.stepper().unwrap();
    let mut out: Vec<Vec<u32>> = prompts.iter().map(|_| Vec::new()).collect();
    for (i, (slot, prompt)) in prompts.iter().enumerate() {
        let mut start = 0;
        while start < prompt.len() {
            let end = (start + 7).min(prompt.len());
            let last = end == prompt.len();
            if let Some(t) = rt.prefill_slot(&mut st, *slot, &prompt[start..end], start, last.then_some((Sampling::GREEDY, 0))).unwrap() {
                out[i].push(t);
            }
            start = end;
        }
    }
    for step in 0..steps {
        let rows: Vec<DecodeRow> = prompts.iter().zip(&out).map(|((slot, prompt), o)| DecodeRow { slot: *slot, token: *o.last().unwrap(), pos: prompt.len() + o.len() - 1, sampling: Sampling::GREEDY, step: step as u32 + 1 }).collect();
        let next = rt.decode(&mut st, &rows).unwrap();
        for (o, t) in out.iter_mut().zip(next) {
            o.push(t);
        }
    }
    out
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_served_slots_decode_side_by_side_and_match_greedy() {
    use specrt::runtime::{Layout, Runtime};
    let s = engine();
    let (tokens, greedy, _) = reference();
    let mut other: Vec<u32> = tokens.iter().rev().copied().collect();
    other[5] = 1;
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    let alone = {
        let mut rt = Runtime::load_batched(&s, max_ctx, 1).unwrap();
        served_greedy(&mut rt, &[(0, &other)], 4).remove(0)
    };
    let layout = Layout { max_ctx, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 2, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    rt.set_block_table(0, &(0..pages as u32).rev().collect::<Vec<_>>()).unwrap();
    rt.set_block_table(1, &(pages as u32..2 * pages as u32).collect::<Vec<_>>()).unwrap();
    let both = served_greedy(&mut rt, &[(0, &tokens), (1, &other)], 4);
    eprintln!("slot 0 {:?}, slot 1 {:?}, slot 1 alone {alone:?}", both[0], both[1]);
    assert_eq!(both[0], greedy, "slot 0 against the transformers greedy beside a second sequence");
    assert_eq!(both[1], alone, "slot 1 beside slot 0 against the same prompt served alone");
    let again = served_greedy(&mut rt, &[(1, &tokens), (0, &other)], 4);
    assert_eq!((&again[0], &again[1]), (&greedy, &alone), "slots swapped after reuse");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_state_snapshot_carries_the_ple_history_and_conv_window_to_another_slot() {
    use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
    let s = engine();
    let (tokens, greedy, _) = reference();
    let max_ctx = 256;
    let pages = max_ctx / specrt::codegen::PAGE;
    let layout = Layout { max_ctx, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 2, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&s, layout).unwrap();
    let table: Vec<u32> = (0..pages as u32).collect();
    rt.set_block_table(0, &table).unwrap();
    let mut st = rt.stepper().unwrap();
    let first = rt.prefill_slot(&mut st, 0, &tokens, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
    let snap = rt.snapshot_state_blocking(0).unwrap();
    assert!(snap.pending().len() >= 2 + 512 * 9, "the snapshot holds the n-gram history and the conv window");
    rt.reset_slot(0).unwrap();
    rt.restore_state(1, &snap).unwrap();
    rt.set_block_table(1, &table).unwrap();
    let mut out = vec![first];
    for step in 0..4 {
        let row = DecodeRow { slot: 1, token: *out.last().unwrap(), pos: tokens.len() + step, sampling: Sampling::GREEDY, step: step as u32 + 1 };
        out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
    }
    assert_eq!(out, greedy, "decode in slot 1 from slot 0's restored state");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_stubbed_ple_delta_matches_the_layer_1_streams_the_prelude_computes() {
    let s = harness();
    let (tokens, _, _) = reference();
    let n = tokens.len();
    let width = 4 * s.dims.hidden;
    let mut stubbed = harness();
    stubbed.prelude = None;
    let mut rt = specrt::runtime::Runtime::load(&stubbed, 256).unwrap();
    let injected = rt.trace_streams(&tokens, &ple_inject(n)).unwrap();
    drop(rt);
    let mut rt = specrt::runtime::Runtime::load(&s, 256).unwrap();
    let computed = rt.trace_streams(&tokens, &Default::default()).unwrap();
    let d = max_diff(&injected.layers[n * width..2 * n * width], &computed.layers[n * width..2 * n * width]);
    eprintln!("layer 1 streams, in engine PLE against the #1279 capture injected: max abs {d:e}");
    assert!(d <= TOLERANCE, "{d}");
}
