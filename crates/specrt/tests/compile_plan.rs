use specrt::compile::{Options, compile, single};
use specrt::plan::{Op, Phase, Plan};
use specrt::signature::validate_plan;
use specrt::spec::{Block, Quant, Spec};
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn sm120() -> kir::Target {
    kir::Target::sm_120()
}

fn all_specs() -> Vec<(String, Spec)> {
    let mut out = Vec::new();
    let fixtures = root().join("tests/fixtures");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&fixtures).unwrap().flatten().map(|e| e.path()).collect();
    paths.extend(std::fs::read_dir(root().join("specs")).unwrap().flatten().map(|e| e.path()));
    paths.sort();
    for p in paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let spec = if name.ends_with(".config.json") {
            let json = std::fs::read_to_string(&p).unwrap();
            Spec::from_hf_config_serving(&json).or_else(|_| Spec::describe_hf_config(&json)).ok()
        } else if p.is_dir() && p.join("config.json").exists() {
            Spec::from_hf_config_serving(&std::fs::read_to_string(p.join("config.json")).unwrap()).ok()
        } else if name.ends_with(".toml") && !name.starts_with("family-") && std::fs::read_to_string(&p).is_ok_and(|t| !t.contains("kind = \"kir-kernel\"")) {
            Spec::load(&p).ok()
        } else {
            None
        };
        if let Some(s) = spec {
            out.push((name, s));
        }
    }
    out
}

fn fixture(name: &str) -> Spec {
    Spec::from_hf_config_serving(&std::fs::read_to_string(root().join("tests/fixtures").join(name).join("config.json")).unwrap()).unwrap()
}

fn dense() -> Spec {
    Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap()
}

fn hand_tp(ops: &[Op], shards: &std::collections::BTreeMap<String, specrt::tp::Shard>, fuse: bool) -> Vec<Op> {
    use specrt::plan::{Buf, Op};
    let mut out = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        match ops[i] {
            Op::Linear { weight, quant, input, output: Buf::X } if shards.get(weight.name()).is_some_and(|s| s.needs_allreduce()) => {
                out.push(Op::Linear { weight, quant, input, output: Buf::Partial });
                match ops.get(i + 1) {
                    Some(Op::Norm { which, into: Buf::XnPlanes }) if fuse => {
                        out.push(Op::AllReduceNorm { which: *which });
                        i += 1;
                    }
                    _ => out.push(Op::AllReduce),
                }
            }
            Op::MoeCombine if !shards.is_empty() => {
                out.push(Op::MoeCombine);
                out.push(Op::AllReduce);
            }
            op => out.push(op),
        }
        i += 1;
    }
    out
}

fn hand_tp_plan(mut plan: Plan, shards: &std::collections::BTreeMap<String, specrt::tp::Shard>, fuse: bool) -> Plan {
    for b in &mut plan.blocks {
        b.ops = hand_tp(&b.ops, shards, fuse);
    }
    plan.head = hand_tp(&plan.head, shards, fuse);
    plan.tail = hand_tp(&plan.tail, shards, fuse);
    plan.mtp = hand_tp(&plan.mtp, shards, fuse);
    plan.mtp_head = hand_tp(&plan.mtp_head, shards, fuse);
    plan
}

#[test]
fn tensor_parallel_passes_reproduce_the_hand_rewrite() {
    let shards = specrt::graph::Graph::decoder().shards().unwrap();
    let mut checked = 0;
    let mut failures = Vec::new();
    for (name, spec) in all_specs().into_iter().filter(|(_, s)| !s.uses_branch()) {
        for fuse in [false, true] {
            for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
                let plan = Plan::build(&spec, phase, true);
                let want = hand_tp_plan(plan.clone(), &shards, fuse);
                match compile(&spec, plan, &sm120(), &Options::ranked(shards.clone(), fuse, 4096)) {
                    Ok(got) if got == want => checked += 1,
                    Ok(got) => failures.push(format!("{name} fuse={fuse} {phase:?}:\n got {:?}\nwant {:?}", got.blocks, want.blocks)),
                    Err(e) => failures.push(format!("{name} fuse={fuse} {phase:?}: {e:#}")),
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(checked > 100, "{checked}");
}

#[test]
fn passes_leave_single_rank_plans_alone() {
    for (name, spec) in all_specs() {
        for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
            let plan = Plan::build(&spec, phase, true);
            assert_eq!(compile(&spec, plan.clone(), &sm120(), &Options::planes(4096)).unwrap(), plan, "{name} {phase:?}");
        }
    }
}

fn step_kernels(spec: &Spec) -> Vec<Vec<String>> {
    let fp4 = |i: usize| spec.mlp_quant(i) == Quant::Nvfp4G16;
    let mut out: Vec<Vec<String>> = Vec::new();
    for i in 0..spec.layers.count {
        let store = |name: &'static str, plain: &'static str| -> Vec<&'static str> { if spec.runtime.store_epilogue { vec![name] } else { vec!["clear_branch", plain] } };
        let mut k: Vec<&str> = Vec::new();
        if i == 0 {
            k.push("embed");
        }
        let mlp = |k: &mut Vec<&str>| {
            if spec.moe.as_ref().is_some_and(|m| m.is_sparse(i)) {
                k.extend(["moe_route_logits", "moe_route", "moe_up_bf16", "moe_down_bf16", "moe_combine"]);
                return;
            }
            let (act, down) = if fp4(i) { ("gate_up_act_f4", "down_proj_f4") } else { ("gate_up_act", "down_proj") };
            if spec.norm.pre {
                k.push(act);
            } else {
                k.extend(["gate_up_proj", "swiglu"]);
            }
            if spec.uses_branch() {
                if fp4(i) {
                    k.extend(["clear_branch", down]);
                } else {
                    k.extend(store("down_proj_store", down));
                }
                k.push(if spec.norm.sandwich { "norm_add" } else { "residual_add" });
            } else {
                k.push(down);
            }
        };
        if spec.block(i) == Block::Linear {
            k.extend(["lin_in_norm", "lin_ba_norm", "gdn_conv_step", "gdn_step", "lin_out_proj"]);
            mlp(&mut k);
        } else {
            k.push(if spec.norm.pre { "qkv_proj_norm" } else { "qkv_proj" });
            if spec.attention.qk_norm_full {
                k.extend(["q_norm_rows", "k_norm_rows"]);
            }
            k.extend(["qk_norm_rope", "attention_split", "attention_reduce"]);
            if spec.uses_branch() {
                k.extend(store("o_proj_store", "o_proj"));
                k.push(if spec.norm.sandwich { "norm_add" } else { "residual_add" });
            } else {
                k.push("o_proj");
            }
            mlp(&mut k);
        }
        out.push(k.into_iter().map(String::from).collect());
    }
    let mut tail = vec![if spec.lm_head_quant() == Quant::Nvfp4G16 { "lm_head_norm_f4" } else { "lm_head_norm" }];
    if spec.dims.logit_scale.is_some() {
        tail.push("logit_scale");
    }
    if spec.dims.logit_softcap.is_some() {
        tail.push("logit_softcap");
    }
    out.last_mut().unwrap().extend(tail.into_iter().map(String::from));
    out
}

fn plan_kernels(spec: &Spec, plan: &Plan) -> Vec<Vec<String>> {
    let t = sm120();
    let names = |ops: &[Op], layer: usize| -> Vec<String> {
        ops.iter().flat_map(|o| match o {
            Op::MoeExperts => vec!["moe_up_bf16".to_string(), "moe_down_bf16".to_string()],
            o => o.kernels(spec, layer, 1, &t),
        }).collect()
    };
    let mut out = Vec::new();
    for i in 0..spec.layers.count {
        let sparse = spec.moe.as_ref().is_some_and(|m| m.is_sparse(i) && spec.has_mlp(i));
        let mut k = if i == 0 { names(&plan.head, 0) } else { Vec::new() };
        k.extend(names(plan.layer_of(spec.block(i), sparse), i));
        out.push(k);
    }
    out.last_mut().unwrap().extend(names(&plan.tail, spec.layers.count - 1));
    out
}

fn runs_single_step(spec: &Spec) -> bool {
    spec.attention.node.is_empty() && spec.requanted_linears().is_empty() && spec.mixer.is_none() && spec.hyper.is_none() && !spec.layers.split && spec.mlp.mapped.is_none() && spec.dims.head_rows.is_none()
}

fn assert_single_matches(name: &str, spec: &Spec) {
    let plan = single(spec, &sm120(), 4096).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    validate_plan(&plan, spec).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    assert_eq!(plan_kernels(spec, &plan), step_kernels(spec), "{name}: {plan:?}");
}

#[test]
fn the_single_step_plan_is_the_hand_arrangement_for_qwen3_dense() {
    assert_single_matches("qwen3-0.6b", &dense());
}

#[test]
fn the_single_step_plan_is_the_hand_arrangement_for_tiny27b() {
    assert_single_matches("tiny-qwen3.5-27b", &fixture("tiny-qwen3.5-27b"));
}

#[test]
fn the_single_step_plan_is_the_hand_arrangement_for_every_spec_it_runs() {
    let mut checked = 0;
    for (name, spec) in all_specs().into_iter().filter(|(_, s)| runs_single_step(s)) {
        assert_single_matches(&name, &spec);
        checked += 1;
    }
    assert!(checked >= 20, "{checked}");
}

#[test]
fn the_store_epilogue_drops_the_branch_clear_from_every_branch_spec() {
    let mut checked = 0;
    for (name, mut spec) in all_specs().into_iter().filter(|(_, s)| runs_single_step(s) && s.uses_branch()) {
        let before: usize = plan_kernels(&spec, &single(&spec, &sm120(), 4096).unwrap()).iter().flatten().filter(|k| *k == "clear_branch").count();
        spec.runtime.store_epilogue = true;
        assert_single_matches(&name, &spec);
        let plan = single(&spec, &sm120(), 4096).unwrap();
        let kernels: Vec<String> = plan_kernels(&spec, &plan).into_iter().flatten().collect();
        assert_eq!(kernels.iter().filter(|k| *k == "clear_branch").count(), 0, "{name}");
        assert_eq!(before, 2 * spec.layers.count, "{name}");
        let module = specrt::codegen::Module::generate(&spec, 4096).unwrap();
        for k in kernels.iter().filter(|k| k.ends_with("_store")) {
            assert!(module.kernels.contains(&k.as_str()), "{name}: {k} is planned but not emitted");
        }
        checked += 1;
    }
    assert!(checked >= 4, "{checked}");
}

#[test]
fn the_store_epilogue_leaves_plans_without_a_branch_alone() {
    for (name, mut spec) in all_specs().into_iter().filter(|(_, s)| runs_single_step(s) && !s.uses_branch()) {
        let off = single(&spec, &sm120(), 4096).unwrap();
        spec.runtime.store_epilogue = true;
        assert_eq!(single(&spec, &sm120(), 4096).unwrap(), off, "{name}");
    }
}

fn launch_row(name: &str, spec: &Spec) -> (String, bool) {
    use specrt::signature::{Count, compiled_report};
    let count = Count::new(Phase::Decode, 1, 4096);
    let mut built = Plan::build(spec, Phase::Decode, true);
    built.mtp.clear();
    built.mtp_head.clear();
    let unfused = compiled_report(spec, &built, count, &sm120()).unwrap().total().2;
    let (plan, trace) = specrt::compile::single_traced(spec, &sm120(), 4096).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    let compiled = compiled_report(spec, &plan, count, &sm120()).unwrap().total().2;
    let before: usize = trace.sections.iter().map(|s| s.launches_before * s.repeats).sum();
    let predicted: usize = trace.sections.iter().map(|s| s.launches_after * s.repeats).sum();
    let ok = compiled < unfused && compiled == predicted;
    (format!("{name:<44} unfused {unfused:>5} units {before:>5} predicted {predicted:>5} compiled {compiled:>5}{}", if ok { "" } else { "  <-" }), ok)
}

fn assert_launches(specs: &[(String, Spec)]) {
    let rows: Vec<(String, bool)> = specs.iter().map(|(n, s)| launch_row(n, s)).collect();
    let table: Vec<&str> = rows.iter().map(|(r, _)| r.as_str()).collect();
    eprintln!("{}", table.join("\n"));
    assert!(rows.iter().all(|(_, ok)| *ok), "compiled single step launches against the unfused plan and the cost model:\n{}", table.join("\n"));
}

#[test]
fn the_compiled_step_launches_what_the_cost_model_predicts() {
    let specs: Vec<(String, Spec)> = all_specs().into_iter().filter(|(_, s)| runs_single_step(s)).collect();
    assert!(specs.len() >= 20, "{}", specs.len());
    assert_launches(&specs);
}

const MIXES: [&str; 3] = ["tiny-sandwich-scaled.toml", "tiny-sandwich-pre-qknorm.toml", "tiny-scaled-qknorm.toml"];

fn mix(name: &str, store: bool) -> Spec {
    let mut s = Spec::load(&root().join("tests/fixtures").join(name)).unwrap();
    s.runtime.store_epilogue = store;
    s
}

#[test]
fn spec_only_mixes_get_their_fusions_from_the_passes() {
    let mut specs = Vec::new();
    for name in MIXES {
        for store in [false, true] {
            let s = mix(name, store);
            assert!(runs_single_step(&s), "{name}");
            specs.push((format!("{name} store_epilogue={store}"), s));
        }
    }
    assert_launches(&specs);
    for (name, s) in &specs {
        let (_, trace) = specrt::compile::single_traced(s, &sm120(), 4096).unwrap();
        let layer = trace.section("layer dense").unwrap_or_else(|| panic!("{name}\n{}", trace.render()));
        let rules: Vec<specrt::fusion::Rule> = layer.fused.iter().map(|d| d.rule).collect();
        let count = |r| rules.iter().filter(|x| **x == r).count();
        use specrt::fusion::Rule;
        assert_eq!(count(Rule::Fill), if s.runtime.store_epilogue { 2 } else { 0 }, "{name}\n{}", trace.render());
        assert_eq!(count(Rule::Prologue), if s.norm.pre { 2 } else { 0 }, "{name}\n{}", trace.render());
        assert!(count(Rule::Epilogue) >= 3, "{name}\n{}", trace.render());
    }
}

#[test]
fn every_single_step_kernel_is_in_the_module() {
    for (name, spec) in all_specs().into_iter().filter(|(_, s)| runs_single_step(s)) {
        let Ok(module) = specrt::codegen::Module::generate(&spec, 4096) else { continue };
        let plan = single(&spec, &sm120(), 4096).unwrap();
        for k in plan_kernels(&spec, &plan).into_iter().flatten().filter(|k| !k.starts_with("moe_")) {
            assert!(module.kernels.contains(&k.as_str()), "{name}: {k} is planned but not emitted");
        }
    }
}
