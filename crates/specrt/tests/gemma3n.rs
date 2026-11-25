use specrt::plan::{Op, Phase, Plan, Weight};
use specrt::spec::{Block, Spec};
use std::path::Path;

fn config(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/popular").join(name).join("config.json")).unwrap()
}

fn e2b() -> Spec {
    Spec::from_hf_config_serving(&config("google__gemma-3n-E2B-it")).unwrap()
}

fn tiny() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-gemma3n");
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-gemma3n".into();
    spec.repo = dir.display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

struct Reference {
    tokens: Vec<u32>,
    greedy: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
    streams: Vec<f32>,
}

fn reference() -> Reference {
    let dir = std::env::var_os("SPECRT_REF_DIR_3N").map(std::path::PathBuf::from).expect("SPECRT_REF_DIR_3N holds scripts/tiny_gemma3n.py output");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tiny-gemma3n.ref.json")).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let floats = |f: &str| -> Vec<f32> { std::fs::read(dir.join(f)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() };
    Reference { tokens: ids("tokens"), greedy: ids("greedy"), vocab: meta["vocab"].as_u64().unwrap() as usize, logits: floats("tiny-gemma3n.ref.bin"), streams: floats("tiny-gemma3n.streams.bin") }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn prompt_logits(spec: &Spec, tokens: &[u32]) -> Vec<f32> {
    specrt::runtime::Runtime::load(spec, 256).unwrap().prompt_logits(tokens).unwrap()
}

#[test]
fn the_tiny_fixture_maps_like_the_real_configs_and_builds_every_kernel() {
    let spec = tiny();
    let g = spec.gated_residual().unwrap();
    assert_eq!((g.streams, g.params["laurel_rank"], spec.layers.shared_tail, spec.per_layer().unwrap().dim), (4, 8.0, 2, 16));
    assert_eq!(g.layer_params["sparse_on"], vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(spec.per_layer().unwrap().params["ple_vocab"], 384.0);
    let states: Vec<(&str, usize)> = spec.state.iter().map(|s| (s.name.as_str(), s.layer_count(&spec))).collect();
    assert_eq!(states, vec![("k_s", 1), ("v_s", 1), ("k_g", 1), ("v_g", 1)]);
    specrt::codegen::Module::generate(&spec, 256).unwrap();
    for phase in [Phase::Prefill, Phase::Decode] {
        specrt::signature::validate_plan(&Plan::build(&spec, phase, true), &spec).unwrap();
    }
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3n.py output in SPECRT_REF_DIR_3N"]
fn gemma3n_streams_after_every_layer_match_transformers() {
    let r = reference();
    let mut rt = specrt::runtime::Runtime::load(&tiny(), 256).unwrap();
    let trace = rt.trace_streams(&r.tokens, &Default::default()).unwrap();
    eprintln!("trace logits max_abs {}", max_abs(&trace.logits, &r.logits));
    assert!(max_abs(&trace.logits, &r.logits) < 5e-2, "the collapsed streams go through the final norm");
    let per = r.tokens.len() * 4 * 128;
    for layer in 0..4 {
        let (a, b) = (&trace.layers[layer * per..(layer + 1) * per], &r.streams[layer * per..(layer + 1) * per]);
        let size = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
        let d = max_abs(a, b);
        eprintln!("tiny-gemma3n layer {layer} streams max_abs {d} of {size}");
        assert!(d < 2e-2 * size.max(1.0), "layer {layer} streams max abs diff {d}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3n.py output in SPECRT_REF_DIR_3N"]
fn gemma3n_logits_and_greedy_match_transformers() {
    let r = reference();
    let mut rt = specrt::runtime::Runtime::load(&tiny(), 256).unwrap();
    let ours = rt.prompt_logits(&r.tokens).unwrap();
    let d = max_abs(&ours, &r.logits);
    eprintln!("tiny-gemma3n logits max_abs {d}");
    assert!(d < 5e-2, "max abs diff {d}");
    for row in 0..r.tokens.len() {
        let arg = |v: &[f32]| v[row * r.vocab..(row + 1) * r.vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(arg(&ours), arg(&r.logits), "row {row}");
    }
    let got = rt.generate(&r.tokens, r.greedy.len(), specrt::runtime::Sampling::GREEDY).unwrap();
    assert_eq!(got.tokens, r.greedy);
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3n.py output in SPECRT_REF_DIR_3N"]
fn gemma3n_served_two_slots_match_transformers_greedy() {
    use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling};
    let r = reference();
    let pages = 256 / specrt::codegen::PAGE;
    let layout = Layout { max_ctx: 256, slots: 2, kv_blocks: 2 * pages, prefill_chunk: 64, kv_planes: 1, kv_headroom: 0, state_on_demand: false };
    let mut rt = Runtime::load_layout(&tiny(), layout).unwrap();
    rt.set_block_table(0, &(0..pages as u32).rev().collect::<Vec<_>>()).unwrap();
    rt.set_block_table(1, &(pages as u32..2 * pages as u32).collect::<Vec<_>>()).unwrap();
    let mut st = rt.stepper().unwrap();
    let short = &r.tokens[..40];
    let mut firsts = [None, None];
    for (slot, prompt) in [(0usize, &r.tokens[..]), (1, short)] {
        let mut start = 0;
        while start < prompt.len() {
            let end = (start + 64).min(prompt.len());
            firsts[slot] = rt.prefill_slot(&mut st, slot, &prompt[start..end], start, (end == prompt.len()).then_some((Sampling::GREEDY, 0))).unwrap();
            start = end;
        }
    }
    let mut out = vec![firsts[0].unwrap()];
    let mut other = vec![firsts[1].unwrap()];
    while out.len() < r.greedy.len() {
        let rows = [
            DecodeRow { slot: 0, token: *out.last().unwrap(), pos: r.tokens.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 },
            DecodeRow { slot: 1, token: *other.last().unwrap(), pos: short.len() + other.len() - 1, sampling: Sampling::GREEDY, step: other.len() as u32 },
        ];
        let next = rt.decode(&mut st, &rows).unwrap();
        out.push(next[0]);
        other.push(next[1]);
    }
    assert_eq!(out, r.greedy, "served greedy beside a second sequence");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3n.py output in SPECRT_REF_DIR_3N"]
fn gemma3n_without_activation_sparsity_or_laurel_misses_transformers() {
    let r = reference();
    let mut dense = tiny();
    dense.gated_residual.as_mut().unwrap().layer_params.insert("sparse_on".into(), vec![0.0; 4]);
    let d = max_abs(&prompt_logits(&dense, &r.tokens), &r.logits);
    eprintln!("tiny-gemma3n with a dense layer 0 mlp: max_abs {d}");
    assert!(d >= 5e-2, "the logit bound must catch a missing activation sparsity, max abs {d}");
    let mut plain = tiny();
    let g = plain.gated_residual.as_mut().unwrap();
    g.node.retain(|n| n.op != "laurel");
    let d = max_abs(&prompt_logits(&plain, &r.tokens), &r.logits);
    eprintln!("tiny-gemma3n without laurel: max_abs {d}");
    assert!(d >= 5e-2, "the logit bound must catch a missing laurel block, max abs {d}");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3n.py output in SPECRT_REF_DIR_3N"]
fn gemma3n_without_the_altup_correction_misses_transformers() {
    let r = reference();
    let mut stale = tiny();
    let g = stale.gated_residual.as_mut().unwrap();
    g.node.retain(|n| n.kernel.as_deref() != Some("g3n_router_x"));
    let d = max_abs(&prompt_logits(&stale, &r.tokens), &r.logits);
    eprintln!("tiny-gemma3n correcting with the prediction router: max_abs {d}");
    assert!(d >= 5e-2, "the logit bound must catch a correction that skips its own router, max abs {d}");
    let mut uncorrected = tiny();
    let g = uncorrected.gated_residual.as_mut().unwrap();
    for n in g.node.iter_mut().filter(|n| n.kernel.as_deref() == Some("g3n_correct")) {
        n.bind.insert("x".into(), specrt::graph::Bind::Buffer { buffer: "laurel_hi".into() });
    }
    let mut rt = specrt::runtime::Runtime::load(&uncorrected, 256).unwrap();
    let trace = rt.trace_streams(&r.tokens, &Default::default()).unwrap();
    let per = r.tokens.len() * 4 * 128;
    let d = max_abs(&trace.layers[..per], &r.streams[..per]);
    eprintln!("tiny-gemma3n correcting with the wrong layer output: layer 0 streams max_abs {d}");
    assert!(d >= 2e-2 * 25.0, "the stream bound must catch a correction fed the wrong innovation, max abs {d}");
    let texts: Vec<String> = specrt::spec::EMBEDDED.iter().map(|(n, t)| if *n == "g3n_correct.toml" { t.replace("1.0 + c0", "c0") } else { t.to_string() }).collect();
    assert!(texts.iter().any(|t| t.contains("expr = \"c0 * m0")), "the red edit hit g3n_correct");
    specrt::kir_kernel::Registry::install(specrt::kir_kernel::Registry::from_texts(texts.iter().map(String::as_str)).unwrap());
    let ran = std::panic::catch_unwind(|| prompt_logits(&tiny(), &r.tokens));
    specrt::kir_kernel::Registry::install(specrt::kir_kernel::Registry::embedded());
    let d = max_abs(&ran.unwrap(), &r.logits);
    eprintln!("tiny-gemma3n correction coefficients without the + 1: max_abs {d}");
    assert!(d >= 5e-2, "the logit bound must catch correction coefficients without the + 1, max abs {d}");
}

#[test]
fn the_e2b_and_e4b_configs_map_every_piece_from_the_config() {
    for (name, layers, shared, inter) in [("google__gemma-3n-E2B-it", 30, 10, 8192), ("google__gemma-3n-E4B-it", 35, 15, 16384)] {
        let spec = Spec::from_hf_config_serving(&config(name)).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let g = spec.gated_residual().expect("AltUp is a gated residual");
        assert_eq!((g.streams, g.params["laurel_rank"]), (4, 64.0), "{name}");
        assert_eq!((spec.layers.count, spec.layers.shared_tail, spec.dims.intermediate), (layers, shared, inter), "{name}");
        assert!(!spec.mlp.wide_shared_tail && spec.mlp.stored_gate_up, "{name}");
        assert_eq!(spec.per_layer().unwrap().dim, 256, "{name}");
        assert_eq!(spec.per_layer().unwrap().params["ple_vocab"], 262144.0, "{name}");
        assert_eq!(spec.dims.vocab, 262400, "{name}");
        assert_eq!(spec.dims.logit_softcap, Some(30.0), "{name}");
        assert!(spec.norm.sandwich && !spec.norm.unit_offset && spec.norm.layer_scale.is_none(), "{name}");
        assert!(spec.vision.is_none());
        let on = &g.layer_params["sparse_on"];
        assert_eq!(on.iter().filter(|v| **v == 1.0).count(), 10, "{name}");
        assert!(on[..10].iter().all(|v| *v == 1.0));
        let cut = &g.layer_params["sparse_cut"];
        assert!((cut[0] - 1.6448536269514722).abs() < 1e-8 && cut[10] == 0.0, "{name}: {cut:?}");
        assert_eq!(spec.attention.params["theta_s"], 10000.0);
        assert_eq!(spec.attention.params["theta_g"], 1000000.0);
        assert_eq!(spec.attention.params["kvh_g"], spec.attention.params["kvh_s"]);
        assert_eq!(spec.attention.params["angles_g"], 128.0, "full layers rotate every dimension");
    }
}

#[test]
fn the_e2b_attention_scores_are_not_scaled_by_the_head_dim() {
    let spec = e2b();
    let kernels: Vec<&str> = spec.attention.node.iter().filter_map(|n| n.kernel.as_deref()).filter(|k| k.contains("attn")).collect();
    assert_eq!(kernels, ["g4_attn_s", "g4_attn_g"]);
    for k in kernels {
        let text = specrt::spec::EMBEDDED.iter().find(|(n, _)| *n == format!("{k}.toml")).unwrap().1;
        let d = kir::desc::Description::parse(text).unwrap();
        let score = d.stage.iter().find(|s| s.name.as_deref() == Some("score")).and_then(|s| s.expr.clone());
        assert_eq!(score.as_deref(), Some("s"), "{k} scores q.k as is, like scaling = 1.0");
    }
}

#[test]
fn the_e2b_plan_runs_altup_laurel_and_the_stored_gate_up_in_order() {
    let spec = e2b();
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let g = spec.gated_residual().unwrap();
    let kernel = |o: &Op| match o {
        Op::Gated { node, .. } => g.node[*node as usize].kernel.clone().unwrap(),
        Op::Linear { weight, .. } => weight.name().to_string(),
        Op::Norm { which, .. } | Op::NormAdd { which, .. } => which.tensor().to_string(),
        other => other.label(),
    };
    for (layer, tail) in [(0, false), (29, true)] {
        let ops: Vec<String> = plan.layer_for(spec.block(layer), false, tail).iter().map(kernel).collect();
        let at = |k: &str| ops.iter().position(|o| o == k).unwrap_or_else(|| panic!("{k} not in {ops:?}"));
        assert!(at("g3n_router_s") < at("g3n_predict") && at("g3n_predict") < at("g3n_take"));
        assert!(at("g3n_take") < at("g3n_laurel_left") && at("g3n_laurel_right") < at("input_layernorm.weight"));
        assert!(at("post_attention_layernorm.weight") < at("g3n_laurel_add") && at("g3n_laurel_add") < at("pre_feedforward_layernorm.weight"));
        assert!(at("gate_up") < at("g3n_act") && at("g3n_act") < at("down"));
        assert!(at("post_feedforward_layernorm.weight") < at("g3n_router_x") && at("g3n_router_x") < at("g3n_correct"));
    }
    let gu = plan.layer_for(Block::Sliding, false, false).iter().find_map(|o| match o {
        Op::Linear { weight: Weight::GateUp, output, .. } => Some(*output),
        _ => None,
    });
    assert_eq!(gu, Some(specrt::plan::Buf::GateUp));
    let tail: Vec<String> = plan.tail.iter().map(kernel).collect();
    assert_eq!(&tail[..5], &["g3n_unembed_1", "g3n_unembed_2", "g3n_unembed_3", "g3n_collapse", "norm.weight"]);
    let head: Vec<String> = plan.head.iter().map(kernel).collect();
    assert!(head.ends_with(&["g3n_alt_proj".into(), "g3n_alt_proj".into(), "g3n_alt_proj".into(), "g3n_expand".into()]), "{head:?}");
    let per_layer: Vec<String> = plan.layer_tail().iter().map(|o| match o {
        Op::PerLayer { node, .. } => spec.per_layer().unwrap().node[*node as usize].kernel.clone().unwrap(),
        other => other.label(),
    }).collect();
    assert_eq!(per_layer, ["g3n_scaled", "g4_ple_gate", "g4_ple_act", "g4_ple_out", "g3n_ple_add"]);
    for phase in [Phase::Prefill, Phase::Decode] {
        specrt::signature::validate_plan(&Plan::build(&spec, phase, true), &spec).unwrap();
    }
}

#[test]
fn the_e2b_kernels_build() {
    let spec = e2b();
    let module = specrt::codegen::Module::generate(&spec, 256).unwrap();
    for k in ["g3n_predict", "g3n_correct", "g3n_act", "g3n_laurel_add", "g3n_collapse", "g3n_ple_mix"] {
        assert!(module.source.contains(&format!(" {k}(")), "{k}");
    }
    assert!(module.source.contains("(idx[row] < 262144 ? idx[row] : 0)"), "per layer ids past the per layer vocab read row 0");
}

#[test]
fn every_e2b_weight_the_plan_reads_is_named_like_the_checkpoint() {
    let spec = e2b();
    let plan = Plan::build(&spec, Phase::Prefill, true);
    let mut names: Vec<String> = plan.blocks.iter().flat_map(|b| b.ops.iter()).chain(plan.layer_tail()).flat_map(|o| o.tensors(&spec)).collect();
    names.sort();
    names.dedup();
    for want in ["altup.prediction_coefs.weight", "altup.correction_coefs.weight", "altup.modality_router.weight", "altup.router_norm.weight", "altup.correct_output_scale", "laurel.linear_left.weight", "laurel.linear_right.weight", "laurel.post_laurel_norm.weight", "per_layer_input_gate.weight"] {
        assert!(names.iter().any(|n| n == want), "{want} not in {names:?}");
    }
}
