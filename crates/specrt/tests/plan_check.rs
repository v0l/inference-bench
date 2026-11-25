use specrt::plan::{Buf, NormKind, Op, Phase, Plan, QuantSource, Weight};
use specrt::signature::validate_plan;
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn configs() -> Vec<PathBuf> {
    let dir = root().join("tests/fixtures");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path();
        if p.file_name().unwrap().to_string_lossy().ends_with(".config.json") {
            out.push(p);
        } else if p.is_dir() && p.join("config.json").exists() {
            out.push(p.join("config.json"));
        }
    }
    out.sort();
    out
}

fn spec_files() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(root().join("specs"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .chain(std::fs::read_dir(root().join("tests/fixtures")).unwrap().flatten().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "toml") && !p.file_name().unwrap().to_string_lossy().starts_with("family-"))
        .filter(|p| std::fs::read_to_string(p).is_ok_and(|t| !t.contains("kind = \"kir-kernel\"")))
        .collect();
    out.sort();
    out
}

fn specs() -> Vec<(String, Spec)> {
    let mut out = Vec::new();
    for path in configs() {
        let json = std::fs::read_to_string(&path).unwrap();
        if let Ok(spec) = Spec::from_hf_config_serving(&json).or_else(|_| Spec::describe_hf_config(&json)) {
            out.push((path.display().to_string(), spec));
        }
    }
    for path in spec_files() {
        if let Ok(spec) = Spec::load(&path) {
            out.push((path.display().to_string(), spec));
        }
    }
    out
}

fn fixture(name: &str) -> Spec {
    let json = std::fs::read_to_string(root().join("tests/fixtures").join(name).join("config.json")).unwrap();
    Spec::from_hf_config_serving(&json).unwrap()
}

fn runs_without_tensor_cores(spec: &Spec) -> bool {
    spec.attention.node.is_empty() && spec.linear_attention.is_none() && spec.mixer.is_none() && !spec.layers.split && spec.mlp.mapped.is_none()
}

#[test]
fn every_fixture_and_spec_plan_validates() {
    let all = specs();
    let mut failures = Vec::new();
    for (name, spec) in &all {
        for tensor_core in [true, false].into_iter().filter(|tc| *tc || runs_without_tensor_cores(spec)) {
            for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
                let plan = Plan::build(spec, phase, tensor_core);
                if let Err(e) = validate_plan(&plan, spec) {
                    failures.push(format!("{name} tc={tensor_core} {phase:?}: {e:#}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(all.len() >= 40, "only {} specs were checked", all.len());
}

#[test]
fn the_named_kinds_are_all_covered() {
    let all = specs();
    let has = |what: &str, f: &dyn Fn(&Spec) -> bool| assert!(all.iter().any(|(_, s)| f(s)), "no {what} spec was validated");
    has("delta-net", &|s| s.linear_attention.is_some());
    has("moe", &|s| s.moe.is_some());
    has("mtp", &|s| s.speculative().is_some());
    has("sandwich norm", &|s| s.norm.sandwich);
    has("scaled residual", &|s| !s.norm.sandwich && s.norm.residual_scale.is_some());
    has("described attention", &|s| !s.attention.node.is_empty());
    has("described mixer", &|s| s.mixer.is_some());
    has("split layers", &|s| s.layers.split);
    for name in ["tiny-qwen3.5", "tiny-qwen3.5-27b", "tiny-qwen3.5-moe-mtp", "tiny-sinks", "tiny-chunked", "tiny-mistral"] {
        let spec = fixture(name);
        for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
            validate_plan(&Plan::build(&spec, phase, true), &spec).unwrap_or_else(|e| panic!("{name} {phase:?}: {e:#}"));
        }
    }
}

#[test]
fn tensor_parallel_plans_validate() {
    let shards = specrt::graph::Graph::decoder().shards().unwrap();
    let mut checked = 0;
    for (name, spec) in specs().into_iter().filter(|(_, s)| !s.uses_branch()) {
        for fuse in [false, true] {
            for phase in [Phase::Prefill, Phase::Decode] {
                let plan = specrt::compile::compile(&spec, Plan::build(&spec, phase, true), &kir::Target::sm_120(), &specrt::compile::Options::ranked(shards.clone(), fuse, 4096)).unwrap();
                validate_plan(&plan, &spec).unwrap_or_else(|e| panic!("{name} tp fuse={fuse} {phase:?}: {e:#}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 40);
}

fn dense() -> Spec {
    fixture("tiny-mistral")
}

fn broken(edit: impl Fn(&mut Vec<Op>)) -> String {
    let spec = dense();
    let mut plan = Plan::build(&spec, Phase::Prefill, true);
    let block = plan.blocks.iter_mut().find(|b| b.layers.iter().any(|l| l == spec.block(0).name()) && !b.sparse).unwrap();
    edit(&mut block.ops);
    format!("{:#}", validate_plan(&plan, &spec).expect_err("a broken plan must fail"))
}

fn index_of(ops: &[Op], want: impl Fn(&Op) -> bool) -> usize {
    ops.iter().position(want).unwrap()
}

#[test]
fn a_read_before_write_names_the_node() {
    let e = broken(|ops| {
        let at = index_of(ops, |o| matches!(o, Op::Norm { which: NormKind::Input, .. }));
        ops.remove(at);
    });
    assert!(e.contains("layer 0 (") && e.contains(") node 0 (qkv") && e.contains("before any node writes it"), "{e}");
}

#[test]
fn a_shape_mismatch_names_the_node() {
    let e = broken(|ops| {
        let at = index_of(ops, |o| matches!(o, Op::Linear { weight: Weight::Down, .. }));
        ops.insert(at, Op::Linear { weight: Weight::Qkv, quant: QuantSource::Attn, input: Buf::XnPlanes, output: Buf::ActPlanes });
    });
    assert!(e.contains("shape mismatch") && e.contains("node 8 (down"), "{e}");
}

#[test]
fn a_plane_mismatch_names_the_node() {
    let e = broken(|ops| {
        let at = index_of(ops, |o| matches!(o, Op::Norm { which: NormKind::PostAttention, .. }));
        ops[at] = Op::Norm { which: NormKind::PostAttention, into: Buf::Xn };
    });
    assert!(e.contains("form mismatch") && e.contains("node 6 (gate_up") && e.contains("planes"), "{e}");
}

#[test]
fn the_plan_report_matches_hand_counted_bytes_on_the_tiny_dense_fixture() {
    use specrt::signature::{Class, Count, plan_report};
    let spec = dense();
    assert_eq!((spec.dims.hidden, spec.dims.intermediate, spec.dims.vocab, spec.qkv_rows(), spec.quant), (128, 256, 512, 512, specrt::spec::Quant::Bf16));
    let report = plan_report(&spec, Phase::Decode, Count::new(Phase::Decode, 1, 1), &kir::Target::sm_120()).unwrap();
    let layer = report.sections.iter().find(|s| s.name.starts_with("layer")).unwrap();
    assert_eq!((layer.name.as_str(), layer.repeats, report.sections.len()), ("layer dense", 4, 3), "only the layer kinds the spec runs");
    let (h, q, inter, qkv) = (128u64, 256u64, 256u64, 512u64);
    let x = 4 * h;
    let norm_w = 4 * h;
    let kv_row = 2 * 64 * 2;
    let want: Vec<(&str, Class, u64, u64, usize)> = vec![
        ("input norm", Class::RowReduce, x + norm_w, 4 * h, 1),
        ("qkv", Class::Gemm, 4 * h + 2 * qkv * h, 4 * qkv, 1),
        ("decode attention", Class::Attention, 4 * qkv + 2 * kv_row, 4 * qkv + 2 * kv_row + 4 * q, 3),
        ("o", Class::Gemm, 4 * q + 2 * h * q + x, x, 1),
        ("post attention norm", Class::RowReduce, x + norm_w, 4 * h, 1),
        ("gate_up", Class::Gemm, 4 * h + 2 * 2 * inter * h, 4 * inter, 1),
        ("down", Class::Gemm, 4 * inter + 2 * h * inter + x, x, 1),
    ];
    let got: Vec<(&str, Class, u64, u64, usize)> = layer.ops.iter().map(|o| (o.label.as_str(), o.class, o.read, o.written, o.launches)).collect();
    assert_eq!(got, want);
    assert_eq!((layer.read, layer.written, layer.launches), (401_920, 8_704, 9));
    let tail = report.section("tail").unwrap();
    assert_eq!((tail.read, tail.written, tail.launches), (x + norm_w + 4 * h + 2 * 512 * h, 4 * h + 4 * 512, 2));
    let head = report.section("head").unwrap();
    assert_eq!((head.read, head.written, head.launches), (4 + 2 * h, x, 1));
    assert_eq!(report.total(), (head.read + 4 * 401_920 + tail.read, head.written + 4 * 8_704 + tail.written, 1 + 4 * 9 + 2));
    let eight = plan_report(&spec, Phase::Decode, Count::new(Phase::Decode, 8, 100), &kir::Target::sm_120()).unwrap();
    let attention = &eight.sections.iter().find(|s| s.name.starts_with("layer")).unwrap().ops[2];
    assert_eq!(attention.read, 8 * 4 * qkv + 2 * 8 * 100 * kv_row, "eight sequences each read 100 cached rows");
    assert!(report.render().contains("decode attention"));
}

fn contract_break(phase: Phase, model: impl Fn(&mut Vec<Op>), pass: impl Fn(&mut Vec<Op>)) -> String {
    use specrt::signature::{Signer, check_contract, contract};
    let spec = dense();
    let mut before = Plan::build(&spec, phase, true);
    let layer = |p: &mut Plan| -> usize { p.blocks.iter().position(|b| b.layers.iter().any(|l| l == spec.block(0).name()) && !b.sparse).unwrap() };
    let at = layer(&mut before);
    model(&mut before.blocks[at].ops);
    let mut after = before.clone();
    pass(&mut after.blocks[at].ops);
    validate_plan(&before, &spec).unwrap();
    validate_plan(&after, &spec).expect("the broken plan still passes validate_plan");
    let signer = Signer::new(&spec).unwrap();
    let (a, b) = (contract(&signer, &spec, &before).unwrap(), contract(&signer, &spec, &after).unwrap());
    format!("{:#}", check_contract("test pass", &a, &b, specrt::signature::Declared::default()).expect_err("the pass breaks the contract"))
}

fn authored(_: &mut Vec<Op>) {}

#[test]
fn the_contract_refuses_a_bf16_tensor_turned_f32() {
    let e = contract_break(Phase::Prefill, authored, |ops| {
        for op in ops.iter_mut() {
            match op {
                Op::Norm { which: NormKind::Input, into } => *into = Buf::Xn,
                Op::Linear { weight: Weight::Qkv, input, .. } => *input = Buf::Xn,
                _ => {}
            }
        }
    });
    assert!(e.starts_with("pass test pass: layer dense, op input norm: tensor dtype") && e.contains("xn [Tokens x 128] f32") && e.contains("bf16 hi/lo planes"), "{e}");
}

#[test]
fn the_contract_refuses_a_dropped_cast() {
    let e = contract_break(Phase::Prefill, authored, |ops| ops.retain(|o| !matches!(o, Op::KvRope)));
    assert!(e.starts_with("pass test pass: layer dense, op norm and rope: cast") && e.contains("paged bf16") && e.contains("dropped"), "{e}");
}

#[test]
fn the_contract_refuses_a_changed_accumulator() {
    let e = contract_break(Phase::Decode, authored, |ops| {
        for op in ops.iter_mut() {
            if let Op::Linear { weight: Weight::Down, output, .. } = op {
                *output = Buf::XnPlanes;
            }
        }
    });
    assert!(e.starts_with("pass test pass: layer dense, op down: accumulator") && e.contains("bf16 hi/lo planes") && e.contains("x [Tokens x 128] f32"), "{e}");
}

fn parallel_residual(ops: &mut Vec<Op>) {
    let o = index_of(ops, |o| matches!(o, Op::Linear { weight: Weight::O, .. }));
    let Op::Linear { weight, quant, input, .. } = ops[o] else { unreachable!() };
    ops[o] = Op::Linear { weight, quant, input, output: Buf::Branch };
    ops.insert(o, Op::Clear { buf: Buf::Branch });
    ops.push(Op::ScaledAdd { from: Buf::Branch });
}

#[test]
fn the_contract_refuses_a_reordered_reduction() {
    let e = contract_break(Phase::Decode, parallel_residual, |ops| {
        let add = ops.pop().unwrap();
        let down = index_of(ops, |o| matches!(o, Op::Linear { weight: Weight::Down, .. }));
        ops.insert(down, add);
    });
    assert!(e.starts_with("pass test pass: layer dense, op scaled add: reduction order into x") && e.contains("the model adds down, scaled add, the plan adds scaled add, down"), "{e}");
}

#[test]
fn every_fixture_plan_keeps_its_contract_through_the_passes() {
    use specrt::compile::{Options, compile};
    let shards = specrt::graph::Graph::decoder().shards().unwrap();
    let mut checked = 0;
    for (name, spec) in specs() {
        for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
            let plan = Plan::build(&spec, phase, true);
            let mut options = vec![Options::planes(4096)];
            if !spec.uses_branch() {
                options.extend([Options::ranked(shards.clone(), false, 4096), Options::ranked(shards.clone(), true, 4096)]);
            }
            for opts in options {
                compile(&spec, plan.clone(), &kir::Target::sm_120(), &opts).unwrap_or_else(|e| panic!("{name} {phase:?}: {e:#}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 200, "{checked}");
}

#[test]
fn signatures_carry_the_class_and_forms() {
    use specrt::signature::{Class, Form, Signer};
    let spec = dense();
    let signer = Signer::new(&spec).unwrap();
    let norm = signer.signature(&Op::Norm { which: NormKind::Input, into: Buf::XnPlanes }, 0).unwrap();
    assert_eq!(norm.class, Class::RowReduce);
    assert_eq!(norm.writes[0].form, Form::Planes);
    let down = signer.signature(&Op::Linear { weight: Weight::Down, quant: QuantSource::Mlp, input: Buf::ActPlanes, output: Buf::X }, 0).unwrap();
    assert_eq!(down.class, Class::Gemm);
    assert!(down.reads.iter().any(|t| t.place == specrt::signature::Place::Buf(Buf::X)), "the residual epilogue reads x");
    assert_eq!(signer.signature(&Op::Flash, 0).unwrap().class, Class::Attention);
    assert_eq!(signer.signature(&Op::Embed, 0).unwrap().class, Class::Gather);
}
