use specrt::compile::{Options, compile_traced};
use specrt::plan::{Phase, Plan};
use specrt::spec::Spec;
use std::path::Path;

fn hybrid() -> Spec {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-27b/config.json");
    Spec::from_hf_config_serving(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn single(spec: &Spec) -> Plan {
    let mut plan = Plan::build(spec, Phase::Decode, true);
    plan.mtp.clear();
    plan.mtp_head.clear();
    plan
}

#[test]
fn capture_groups_mark_the_independent_branches_and_nothing_with_a_dependency() {
    let spec = hybrid();
    let target = kir::Target::sm_120();
    let (_, trace) = specrt::compile::single_traced(&spec, &target, 4096).unwrap();
    let linear = trace.capture.iter().find(|c| c.name.contains("linear")).unwrap_or_else(|| panic!("{:?}", trace.capture));
    let labels: Vec<Vec<&str>> = linear.groups.iter().map(|g| g.iter().map(|(_, l)| l.as_str()).collect()).collect();
    assert_eq!(labels.iter().filter(|g| g.iter().any(|l| l.contains("lin_in")) && g.iter().any(|l| l.contains("lin_ba"))).count(), 1, "{labels:?}");
    let (plan, _) = compile_traced(&spec, single(&spec), &target, &Options::gemv(4096)).unwrap();
    let signer = specrt::signature::Signer::new(&spec).unwrap();
    for block in &plan.blocks {
        let groups = specrt::compile::capture_groups(&signer, &block.ops, 0).unwrap();
        for group in groups {
            assert!(group.len() > 1);
            for (a, &i) in group.iter().enumerate() {
                let si = signer.signature(&block.ops[i], 0).unwrap();
                for &j in &group[a + 1..] {
                    let sj = signer.signature(&block.ops[j], 0).unwrap();
                    let wi: Vec<_> = si.writes.iter().map(|t| &t.place).collect();
                    for t in sj.reads.iter().chain(&sj.writes) {
                        assert!(!wi.contains(&&t.place), "{} and {} depend on each other", block.ops[i].label(), block.ops[j].label());
                    }
                    for t in &si.reads {
                        assert!(!sj.writes.iter().any(|w| w.place == t.place), "{} and {} depend on each other", block.ops[i].label(), block.ops[j].label());
                    }
                }
            }
        }
    }
}

#[test]
fn capture_groups_are_off_unless_asked_for() {
    let spec = hybrid();
    let (_, off) = compile_traced(&spec, single(&spec), &kir::Target::sm_120(), &Options::gemv(4096)).unwrap();
    assert!(off.capture.is_empty());
}
