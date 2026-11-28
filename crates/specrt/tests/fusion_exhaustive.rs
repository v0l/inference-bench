use specrt::fusion::{Costs, Order, Probe, SectionFusion, probed_report};
use specrt::plan::{Phase, Plan};
use specrt::signature::Count;
use specrt::spec::Spec;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CAP: usize = 200_000;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn configs() -> Vec<(String, Spec)> {
    let dir = root().join("tests/fixtures");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
    paths.sort();
    let mut out = Vec::new();
    for p in paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let json = if name.ends_with(".config.json") {
            std::fs::read_to_string(&p).ok()
        } else if p.is_dir() && p.join("config.json").exists() {
            std::fs::read_to_string(p.join("config.json")).ok()
        } else {
            None
        };
        if name.ends_with(".toml") && !name.starts_with("family-") && std::fs::read_to_string(&p).is_ok_and(|t| !t.contains("kind = \"kir-kernel\"")) {
            if let Ok(spec) = Spec::load(&p) {
                out.push((name, spec));
            }
            continue;
        }
        if let Some(spec) = json.and_then(|j| Spec::from_hf_config_serving(&j).ok()) {
            out.push((name.trim_end_matches(".config.json").to_string(), spec));
        }
    }
    out
}

#[derive(Clone, Debug)]
struct Row {
    plan: String,
    greedy: u64,
    best: u64,
    capped: bool,
    split: bool,
    differs: Vec<String>,
}

impl Row {
    fn gap(&self) -> f64 {
        if self.best == 0 { 0.0 } else { self.best.saturating_sub(self.greedy) as f64 * 100.0 / self.best as f64 }
    }

    fn render(&self) -> String {
        let note = if self.capped { " capped" } else if self.split { " slot fit" } else { "" };
        format!("| {} | {} | {} | {:.2}{note} |", self.plan, self.greedy, self.best, self.gap())
    }
}

fn row(plan: String, sections: &[SectionFusion]) -> Row {
    let mut r = Row { plan, greedy: 0, best: 0, capped: false, split: false, differs: Vec::new() };
    for s in sections {
        let Some(found) = &s.search else { continue };
        r.greedy += found.greedy * s.repeats as u64;
        r.best += found.best.max(found.greedy) * s.repeats as u64;
        r.capped |= found.capped;
        r.split |= found.split;
        if found.best > found.greedy {
            let show = |g: &[Vec<String>]| g.iter().filter(|x| x.len() > 1).map(|x| format!("[{}]", x.join(" + "))).collect::<Vec<_>>().join(" ");
            r.differs.push(format!("{}: greedy {} best {} greedy groups {} best groups {}", s.name, found.greedy, found.best, show(&found.greedy_only), show(&found.best_only)));
        }
    }
    r
}

fn dry_rows(order: Order) -> Vec<Row> {
    let probe = Probe { cap: CAP, order };
    let mut out = Vec::new();
    for (name, spec) in configs() {
        for (tname, target) in [("sm_120", kir::Target::sm_120()), ("sm_80", kir::Target::sm_80())] {
            for (pname, phase, tokens) in [("decode", Phase::Decode, 1), ("prefill", Phase::Prefill, 512)] {
                let plan = Plan::build(&spec, phase, true);
                let Ok(r) = probed_report(&spec, &plan, Count::new(phase, tokens, 1024), &target, &Costs::for_target(&target), Some(probe)) else { continue };
                out.push(row(format!("{name} {pname} {tname}"), &r.sections));
            }
        }
    }
    out
}

fn denoiser_rows(order: Order) -> Vec<Row> {
    use specrt::denoiser::{self, DenoiserGraph, DenoiserSpec};
    let json = std::fs::read_to_string(root().join("tests/fixtures/tiny-flux2-klein/klein-4b-transformer.json")).unwrap();
    let spec = DenoiserSpec::from_transformer_config(&json, 2).unwrap();
    let graph = DenoiserGraph::embedded();
    let plan = graph.plan().unwrap();
    let costs = Costs::for_target(&kir::Target::sm_120()).with_compute(400.0, 1600.0);
    let mut out = Vec::new();
    for (name, image, text) in [("klein-4b denoiser 1024px", 4096, 512), ("klein-4b denoiser 512px", 1024, 512)] {
        let s = specrt::fusion::denoiser_probed(graph, &spec, &plan, denoiser::count(image, text), &costs, false, Some(Probe { cap: CAP, order })).unwrap();
        out.push(row(name.into(), &s));
    }
    out
}

fn compiled_rows(order: Order) -> Vec<Row> {
    let mut out = Vec::new();
    for (name, spec) in configs() {
        for (tname, target) in [("sm_120", kir::Target::sm_120()), ("sm_80", kir::Target::sm_80())] {
            let Ok((_, r)) = specrt::compile::single_probed(&spec, &target, 4096, Probe { cap: CAP, order }) else { continue };
            out.push(row(format!("{name} compiled step {tname}"), &r.sections));
        }
    }
    out
}

fn all_rows(order: Order) -> Vec<Row> {
    let mut rows = dry_rows(order);
    rows.extend(compiled_rows(order));
    rows.extend(denoiser_rows(order));
    rows
}

const SECTION: &str = "## Greedy gap table";

fn recorded() -> BTreeMap<(String, String), [f64; 2]> {
    let doc = std::fs::read_to_string(root().join("../../docs/design/kir-compiler.md")).unwrap();
    let start = doc.find(SECTION).unwrap_or_else(|| panic!("docs/design/kir-compiler.md has no {SECTION:?}"));
    let body = &doc[start + SECTION.len()..];
    let body = &body[..body.find("\n## ").unwrap_or(body.len())];
    let mut out = BTreeMap::new();
    for line in body.lines().filter(|l| l.starts_with("| ")) {
        let cols: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        let gap = |c: &str| c.split_whitespace().next().and_then(|n| n.parse::<f64>().ok());
        if let [plan, phase, a, b, ..] = cols[..]
            && let (Some(a), Some(b)) = (gap(a), gap(b))
        {
            out.insert((plan.to_string(), phase.to_string()), [a, b]);
        }
    }
    out
}

fn split_plan(plan: &str) -> (String, String, usize) {
    let (rest, target) = plan.rsplit_once(' ').unwrap();
    let at = if target == "sm_80" { 1 } else { 0 };
    if plan.contains(" denoiser ") {
        return (plan.to_string(), "denoiser".into(), 0);
    }
    for phase in ["compiled step", "decode", "prefill"] {
        if let Some(name) = rest.strip_suffix(&format!(" {phase}")) {
            return (name.to_string(), phase.to_string(), at);
        }
    }
    panic!("plan {plan:?} has no phase")
}

fn over_recorded(rows: &[Row], table: &BTreeMap<(String, String), [f64; 2]>) -> Vec<String> {
    let mut out = Vec::new();
    for r in rows {
        let (name, phase, at) = split_plan(&r.plan);
        let allowed = table.get(&(name, phase)).map_or(0.0, |g| g[at]);
        if r.capped {
            out.push(format!("{}: the search hit the cap of {CAP} states", r.plan));
        } else if r.gap() > allowed + 0.005 {
            out.push(format!("{}: gap {:.2} percent, the doc records {allowed:.2}\n    {}", r.plan, r.gap(), r.differs.join("\n    ")));
        }
    }
    out
}

fn doc_rows(rows: &[Row]) -> Vec<String> {
    let mut gaps: BTreeMap<(String, String), [f64; 2]> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.gap() > 0.0) {
        let (name, phase, at) = split_plan(&r.plan);
        gaps.entry((name, phase)).or_insert([0.0; 2])[at] = (r.gap() * 100.0).ceil() / 100.0;
    }
    gaps.into_iter().map(|((n, p), g)| format!("| {n} | {p} | {:.2} | {:.2} |", g[0], g[1])).collect()
}

#[test]
fn greedy_stays_within_the_recorded_gap_of_the_exhaustive_pick() {
    let rows = all_rows(Order::Best);
    assert!(rows.len() >= 400, "only {} plans were searched", rows.len());
    for name in ["tiny-qwen3.5-27b", "qwen3.5-4b-base", "qwen3.5-122b-a10b-nvfp4", "klein-4b denoiser 1024px"] {
        assert!(rows.iter().any(|r| r.plan.starts_with(name)), "{name} was not searched");
    }
    let bad = over_recorded(&rows, &recorded());
    assert!(bad.is_empty(), "greedy fell further behind the exhaustive pick than docs/design/kir-compiler.md records:\n{}", bad.join("\n"));
}

#[test]
fn a_worse_pair_order_breaks_the_recorded_gaps() {
    let rows: Vec<Row> = dry_rows(Order::Worst).into_iter().filter(|r| r.plan.starts_with("tiny-qwen3.5-27b ") || r.plan.starts_with("qwen3-0.6b ")).collect();
    assert_eq!(rows.len(), 8);
    let bad = over_recorded(&rows, &recorded());
    assert!(!bad.is_empty(), "merging the least saving pair first must fall behind the recorded gaps");
}

#[test]
fn print_doc_rows() {
    if let Ok(name) = std::env::var("FUSION_ONE") {
        let (_, spec) = configs().into_iter().find(|(n, _)| *n == name).unwrap();
        let t = kir::Target::sm_120();
        let plan = Plan::build(&spec, Phase::Decode, true);
        println!("{}", probed_report(&spec, &plan, Count::new(Phase::Decode, 1, 1024), &t, &Costs::for_target(&t), Some(Probe::exhaustive(CAP))).unwrap().render());
        return;
    }
    if std::env::var("FUSION_TABLE").is_err() {
        return;
    }
    for line in doc_rows(&all_rows(Order::Best)) {
        println!("{line}");
    }
    for r in all_rows(Order::Best).iter().filter(|r| r.gap() > 0.0) {
        println!("{}", r.render());
        for d in &r.differs {
            println!("    {d}");
        }
    }
}
