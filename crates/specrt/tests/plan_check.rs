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
                let plan = Plan::build(&spec, phase, true).for_tp(&shards, fuse);
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
