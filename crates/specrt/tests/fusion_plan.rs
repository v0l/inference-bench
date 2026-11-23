use specrt::fusion::{Costs, FusionReport, Reason, SectionFusion, fusion_report};
use specrt::plan::{Phase, Plan};
use specrt::signature::Count;
use specrt::spec::Spec;
use std::path::Path;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn dense() -> Spec {
    Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap()
}

fn hybrid() -> Spec {
    Spec::from_hf_config_serving(&std::fs::read_to_string(root().join("tests/fixtures/tiny-qwen3.5-27b/config.json")).unwrap()).unwrap()
}

fn sm120() -> kir::Target {
    kir::Target::sm_120()
}

fn with_costs(spec: &Spec, plan: &Plan, tokens: usize, target: &kir::Target, costs: &Costs) -> FusionReport {
    fusion_report(spec, plan, Count::new(plan.phase, tokens, 1024), target, costs).unwrap()
}

fn report(spec: &Spec, phase: Phase, tokens: usize, target: &kir::Target) -> FusionReport {
    let plan = Plan::build(spec, phase, target.language == kir::Language::Cuda);
    with_costs(spec, &plan, tokens, target, &Costs::for_target(target))
}

fn section<'r>(r: &'r FusionReport, prefix: &str) -> &'r SectionFusion {
    r.sections.iter().find(|s| s.name.starts_with(prefix)).unwrap_or_else(|| panic!("no section {prefix}:\n{}", r.render()))
}

fn grouped(s: &SectionFusion, units: &[&str]) -> bool {
    s.groups.iter().any(|g| g.units.iter().map(String::as_str).eq(units.iter().copied()))
}

fn assert_groups(r: &FusionReport, prefix: &str, want: &[&[&str]]) {
    let s = section(r, prefix);
    for units in want {
        assert!(grouped(s, units), "{prefix}: no group {units:?}\n{}", r.render());
    }
}

fn kept_apart<'s>(s: &'s SectionFusion, producer: &str, consumer: &str) -> &'s Reason {
    let d = s.rejected_between(producer, consumer).unwrap_or_else(|| panic!("{producer} -> {consumer} was never considered"));
    assert!(s.group_of(producer).iter().all(|g| !g.units.iter().any(|u| u == consumer)), "{producer} and {consumer} ended in one group");
    d.verdict.as_ref().expect_err("a kept apart pair has a reason")
}

const ATTENTION: [&[&str]; 3] = [&["input norm", "qkv"], &["qk norm", "rope", "kv write"], &["o", "o residual"]];
const MLP: [&[&str]; 2] = [&["post attention norm", "gate_up", "gate_up act"], &["down", "down residual"]];

#[test]
fn dense_decode_selects_the_hand_fusions() {
    let r = report(&dense(), Phase::Decode, 1, &sm120());
    assert_groups(&r, "layer", &ATTENTION);
    assert_groups(&r, "layer", &MLP);
    assert_groups(&r, "tail", &[&["final norm", "lm_head"]]);
    let layer = section(&r, "layer");
    assert_eq!((layer.launches_before, layer.launches_after, layer.groups.len()), (14, 7, 7), "{}", r.render());
}

#[test]
fn hybrid_decode_selects_the_hand_fusions() {
    let r = report(&hybrid(), Phase::Decode, 1, &sm120());
    assert_groups(&r, "layer full", &ATTENTION);
    assert_groups(&r, "layer full", &MLP);
    assert_groups(&r, "layer linear", &[&["input norm", "lin_in"], &["input norm", "lin_ba"], &["lin_out", "lin_out residual"]]);
    assert_groups(&r, "layer linear", &MLP);
    assert_groups(&r, "tail", &[&["final norm", "lm_head"]]);
    assert_groups(&r, "mtp head", &[&["draft norm", "mtp_head"]]);
}

#[test]
fn batched_decode_puts_the_residual_and_act_in_the_split_reduce() {
    let r = report(&hybrid(), Phase::Decode, 8, &sm120());
    assert_groups(&r, "layer full", &[&["o reduce", "o residual"], &["qk norm", "rope", "kv write"]]);
    let full = section(&r, "layer full");
    assert!(matches!(kept_apart(full, "o", "o reduce"), Reason::Cost), "{}", r.render());
}

#[test]
fn prefill_selects_the_tile_epilogues_and_the_rope_chain() {
    let r = report(&hybrid(), Phase::Prefill, 512, &sm120());
    assert_groups(&r, "layer full", &[&["qk norm", "rope", "kv write"], &["o", "o residual"]]);
    assert!(section(&r, "layer full").group_of("gate_up act").iter().any(|g| g.units.iter().any(|u| u == "gate_up")), "{}", r.render());
    let full = section(&r, "layer full");
    assert!(full.groups.iter().all(|g| !(g.units.iter().any(|u| u == "kv write") && g.units.iter().any(|u| u == "attention"))), "attention stays its own kernel");
}

#[test]
fn tensor_parallel_decode_finds_the_allreduce_norm() {
    let spec = dense();
    let shards = specrt::graph::Graph::decoder().shards().unwrap();
    let plan = Plan::build(&spec, Phase::Decode, true).for_tp(&shards, false);
    let r = with_costs(&spec, &plan, 1, &sm120(), &Costs::for_target(&sm120()));
    assert_groups(&r, "layer", &[&["allreduce", "post attention norm"], &["o", "o residual"]]);
    let layer = section(&r, "layer");
    assert!(matches!(kept_apart(layer, "o residual", "allreduce"), Reason::Barrier), "{}", r.render());
}

#[test]
fn add_rmsnorm_is_rejected() {
    for (spec, tokens) in [(dense(), 1), (hybrid(), 1), (hybrid(), 8)] {
        let r = report(&spec, Phase::Decode, tokens, &sm120());
        let layer = section(&r, "layer");
        let residual = layer.groups.iter().flat_map(|g| &g.units).find(|u| u.ends_with("residual")).unwrap().clone();
        let norm = layer.group_of("post attention norm");
        assert!(norm.iter().all(|g| !g.units.contains(&residual)), "the residual add and the next norm fused\n{}", r.render());
        assert!(norm.iter().all(|g| g.units.iter().any(|u| u == "gate_up")), "{}", r.render());
    }
    let r = report(&dense(), Phase::Decode, 1, &sm120());
    let layer = section(&r, "layer");
    assert!(matches!(kept_apart(layer, "o residual", "post attention norm"), Reason::Outscored { .. }), "{}", r.render());
    assert!(matches!(kept_apart(layer, "o", "post attention norm"), Reason::Ownership { owned: 8, row: 1024 }), "{}", r.render());
}

#[test]
fn attention_reduce_stays_out_of_the_split() {
    for spec in [dense(), hybrid()] {
        let r = report(&spec, Phase::Decode, 1, &sm120());
        let layer = section(&r, "layer");
        let d = layer.rejected_between("attention split", "attention combine").unwrap();
        assert_eq!(d.verdict, Err(Reason::Cost), "{}", r.render());
        assert!(d.penalty >= d.saving);
        let free = Costs { round_trip_bytes: 0, ..Costs::for_target(&sm120()) };
        let plan = Plan::build(&spec, Phase::Decode, true);
        let r = with_costs(&spec, &plan, 1, &sm120(), &free);
        assert_groups(&r, "layer", &[&["attention split", "attention combine"]]);
    }
}

#[test]
fn the_wgpu_gemv_norm_fold_is_rejected_by_the_reread_term() {
    let wgpu = kir::Target::named("wgpu").unwrap();
    let r = report(&dense(), Phase::Decode, 1, &wgpu);
    let layer = section(&r, "layer");
    for (norm, gemv) in [("input norm", "qkv"), ("post attention norm", "gate_up")] {
        let d = layer.rejected_between(norm, gemv).unwrap_or_else(|| panic!("{norm} -> {gemv}\n{}", r.render()));
        assert_eq!(d.verdict, Err(Reason::Cost), "{}", r.render());
        assert!(d.penalty > 4 * d.saving, "{}", d.describe());
    }
    assert_groups(&r, "layer", &[&["gate_up", "swiglu"]]);
    let plan = Plan::build(&dense(), Phase::Decode, false);
    let free = Costs { reread_rate: 0.0, ..Costs::for_target(&wgpu) };
    assert_groups(&with_costs(&dense(), &plan, 1, &wgpu, &free), "layer", &[&["input norm", "qkv"]]);
}

#[test]
fn shared_memory_and_registers_bound_a_prologue() {
    let spec = dense();
    let plan = Plan::build(&spec, Phase::Decode, true);
    let base = Costs::for_target(&sm120());
    let small = with_costs(&spec, &plan, 1, &sm120(), &Costs { smem_limit: 2048, ..base });
    let layer = section(&small, "layer");
    assert!(!grouped(layer, &["input norm", "qkv"]), "{}", small.render());
    assert!(matches!(kept_apart(layer, "input norm", "qkv"), Reason::Smem { need: 4096, limit: 2048 }), "{}", small.render());
    let few = with_costs(&spec, &plan, 1, &sm120(), &Costs { max_regs: 19, ..base });
    let layer = section(&few, "layer");
    assert!(matches!(kept_apart(layer, "input norm", "qkv"), Reason::Regs { limit: 19, .. }), "{}", few.render());
    assert!(matches!(kept_apart(layer, "gate_up", "gate_up act"), Reason::Regs { limit: 19, .. }), "{}", few.render());
}

#[test]
fn the_dry_run_leaves_the_plan_alone() {
    let spec = hybrid();
    for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
        let plan = Plan::build(&spec, phase, true);
        let before = plan.clone();
        let r = with_costs(&spec, &plan, 4, &sm120(), &Costs::for_target(&sm120()));
        assert_eq!(plan, before);
        assert!(r.render().contains("dry run"));
        for s in &r.sections {
            assert!(s.launches_after <= s.launches_before, "{}: {} -> {}", s.name, s.launches_before, s.launches_after);
            let mut units: Vec<&String> = s.groups.iter().flat_map(|g| &g.units).collect();
            units.sort();
            let n = units.len();
            units.dedup();
            let duplicated = n - units.len();
            assert!(duplicated <= s.groups.len(), "{}: a unit sits in two groups only as a duplicated prologue", s.name);
        }
    }
}

#[test]
fn every_gemm_kernel_is_emitted_from_the_heavy_ir_and_checked_against_the_target() {
    for spec in [dense(), hybrid(), Spec::load(&root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap()] {
        let set = specrt::gemm::gemm_kernels(&spec, &sm120()).unwrap();
        let mut lifted = 0;
        for k in &set.kernels {
            if let Some(h) = kir::heavy::Heavy::lift(k).unwrap() {
                assert_eq!(&h.lower(&sm120()).unwrap(), k);
                assert!(h.budget(&sm120(), "test").unwrap().regs <= sm120().max_regs);
                lifted += 1;
            }
        }
        assert!(lifted > 0);
    }
    let tight = kir::Target { max_regs: 40, ..sm120() };
    let err = specrt::gemm::gemm_kernels(&hybrid(), &tight).err().expect("a 40 register target cannot hold the big tile accumulators").to_string();
    assert!(err.starts_with(&format!("pass {}: kernel ", specrt::gemm::KERNEL_SELECT)) && err.contains("registers per thread, the target has 40"), "{err}");
}

fn nvfp4() -> Spec {
    Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4-head-nvfp4.toml")).unwrap()
}

#[test]
fn the_decode_norm_gemvs_are_built_by_the_prologue_attach_pass() {
    use kir::gemv::Gemv;
    for spec in [dense(), hybrid(), nvfp4(), Spec::load(&root().join("tests/fixtures/qwen3.8-27b.toml")).unwrap()] {
        let ops = specrt::codegen::ops_shape(&spec);
        let built = specrt::codegen::gemv_kernels(&spec, &sm120(), &ops).unwrap();
        let op = |name: &str| built.iter().find(|k| k.name == name).unwrap_or_else(|| panic!("{}: no {name}", spec.name)).op.clone();
        assert_eq!(op("qkv_proj_norm"), Gemv::Norm { decoder: ops.weight_decoder.clone(), n: ops.qkv_row, bias: ops.qkv_bias });
        assert_eq!(op("gate_up_act"), Gemv::GateUpAct { decoder: ops.weight_decoder.clone() });
        assert_eq!(op("lm_head_norm"), Gemv::Norm { decoder: ops.head_decoder.clone(), n: ops.vocab, bias: false });
        if let Some(la) = spec.linear_attention.as_ref() {
            assert_eq!(op("lin_in_norm"), Gemv::Norm { decoder: ops.weight_decoder.clone(), n: la.in_proj_rows(), bias: false });
            assert_eq!(op("lin_ba_norm"), Gemv::Norm { decoder: "DecBf16".into(), n: la.ba_rows(), bias: false });
        }
        for k in &built {
            let h = kir::heavy::Heavy::lift_gemv(k, ops.hidden, ops.inter).unwrap();
            let normed = matches!(k.op, Gemv::Norm { .. } | Gemv::GateUpAct { .. });
            assert_eq!(h.is_some_and(|h| h.prologue == kir::heavy::norm_prologue()), normed, "{}", k.name);
        }
    }
    let set = specrt::gemm::gemm_kernels(&nvfp4(), &sm120()).unwrap();
    let fp4: Vec<&str> = set.kernels.iter().filter(|k| kir::heavy::Heavy::lift(k).unwrap().is_some_and(|h| !h.prologue.is_empty())).map(|k| k.name.as_str()).collect();
    assert_eq!(fp4, ["gate_up_act_f4", "lm_head_norm_f4"]);
    let small = kir::Target { smem_limit: 4 * dense().dims.hidden as u32 - 1, ..sm120() };
    let err = specrt::codegen::gemv_kernels(&dense(), &small, &specrt::codegen::ops_shape(&dense())).unwrap_err().to_string();
    assert!(err.starts_with(&format!("pass {}: kernel qkv_proj_norm needs", specrt::gemm::PROLOGUE_ATTACH)) && err.contains("bytes of shared memory"), "{err}");
}
