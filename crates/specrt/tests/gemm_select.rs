use specrt::gemm::WeightKind;
use specrt::spec::{GemmSelect, Spec};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
const TOKENS: [usize; 15] = [1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 24, 32, 48, 64, 96];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> Spec {
    Spec::load(&root().join(format!("tests/fixtures/{name}.toml"))).unwrap()
}

fn four_b(act: bool) -> Spec {
    let text = include_str!("../scripts/qwen3.5-4b-nvfp4.toml.in").replace("@CHECKPOINT@", "qwen3.5-4b-nvfp4");
    let text = if act { text } else { text.lines().filter(|l| !l.starts_with("act_quant")).collect::<Vec<_>>().join("\n") };
    Spec::parse(&text).unwrap()
}

fn specs() -> Vec<(&'static str, &'static str, Spec)> {
    vec![
        ("27b", "act", fixture("qwen3.8-27b-nvfp4")),
        ("27b", "exact", fixture("qwen3.8-27b-nvfp4-exact")),
        ("4b", "act", four_b(true)),
        ("4b", "exact", four_b(false)),
        ("0.6b", "exact", Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap()),
    ]
}

fn kinds(spec: &Spec, linear: &str) -> Vec<WeightKind> {
    let base = specrt::gemm::weight_format(spec.quant);
    let mut out = vec![match base {
        specrt::gemm::WeightFormat::Fp8 => WeightKind::Fp8,
        specrt::gemm::WeightFormat::Fp4 => WeightKind::Fp4,
        specrt::gemm::WeightFormat::Q8 => WeightKind::Q8,
        specrt::gemm::WeightFormat::Ggml => WeightKind::Ggml,
        specrt::gemm::WeightFormat::Bf16 => WeightKind::Bf16,
    }];
    if matches!(linear, "gate_up" | "down") && spec.mlp.quant == Some(specrt::spec::Quant::Nvfp4G16) {
        out.insert(0, WeightKind::Fp4);
    }
    out
}

fn table(t: &kir::Target) -> String {
    let mut out = String::new();
    for (model, label, spec) in specs() {
        let la = spec.linear_attention.is_some();
        for linear in LINEARS.into_iter().filter(|l| la || !l.starts_with("lin_")) {
            for kind in kinds(&spec, linear) {
                let row: Vec<String> = TOKENS.iter().map(|&n| specrt::gemm::select(&spec, t, linear, n, kind).map(|c| format!("{n}:{}", c.name.strip_prefix(linear).unwrap_or(&c.name))).unwrap_or_else(|e| format!("{n}:refused {e}"))).collect();
                let _ = writeln!(out, "{model} {label} {linear} {kind:?}: {}", row.join(" "));
            }
        }
    }
    out
}

#[test]
fn selection_on_sm120_is_pinned_for_the_27b_4b_and_0_6b() {
    let got = table(&kir::Target::sm_120());
    let path = root().join("tests/golden/gemm-select-sm120.txt");
    if std::env::var_os("GOLDEN_UPDATE").is_some() {
        std::fs::write(&path, &got).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is missing, run with GOLDEN_UPDATE=1", path.display()));
    assert_eq!(want, got, "the sm_120 gemm selection changed; check docs/perf/gemm-select.md and rerun with GOLDEN_UPDATE=1");
}

#[test]
fn exact_specs_never_select_an_act_quant_path() {
    let t = kir::Target::sm_120();
    for (model, label, spec) in specs().into_iter().filter(|s| s.1 == "exact") {
        for linear in LINEARS {
            for kind in kinds(&spec, linear) {
                for n in TOKENS {
                    let Ok(c) = specrt::gemm::select(&spec, &t, linear, n, kind) else { continue };
                    assert!(c.quant.is_none(), "{model} {label} {linear} {n}: {}", c.name);
                    assert!(specrt::gemm::eligible(&spec, &t, linear, n, kind, true).iter().all(|c| c.quant.is_none()), "{model} {label} {linear} {n}");
                }
            }
        }
    }
}

#[test]
fn single_row_decode_stays_weight_only_on_act_quant_specs() {
    let t = kir::Target::sm_120();
    for (model, _, spec) in specs().into_iter().filter(|s| s.1 == "act") {
        for linear in LINEARS {
            for kind in kinds(&spec, linear) {
                let c = specrt::gemm::select(&spec, &t, linear, 1, kind).unwrap();
                assert!(c.quant.is_none(), "{model} {linear} 1 row took {}", c.name);
                assert!(specrt::gemm::eligible(&spec, &t, linear, 1, kind, true).iter().all(|c| c.quant.is_none()), "{model} {linear}");
            }
        }
    }
}

#[test]
fn swap_rows_off_keeps_the_previous_candidates_and_the_nine_row_act_threshold() {
    let t = kir::Target::sm_120();
    let mut spec = fixture("qwen3.8-27b-nvfp4");
    spec.runtime.swap_rows = false;
    for linear in LINEARS {
        for kind in kinds(&spec, linear) {
            for n in TOKENS {
                let all = specrt::gemm::eligible(&spec, &t, linear, n, kind, true);
                assert!(all.iter().all(|c| !c.path.swapped()), "{linear} {n}");
                if n <= 8 {
                    assert!(all.iter().all(|c| c.quant.is_none()), "{linear} {n}");
                }
            }
        }
    }
    let kernels = specrt::gemm::gemm_kernels(&spec, &t).unwrap();
    assert!(kernels.kernels.iter().all(|k| k.swap_shape().is_none()));
}

#[test]
fn fixed_keeps_the_old_thresholds() {
    let t = kir::Target::sm_120();
    let mut spec = fixture("qwen3.8-27b-nvfp4");
    spec.runtime.gemm_select = GemmSelect::Fixed;
    let pick = |l: &str, n: usize, k: WeightKind| specrt::gemm::select(&spec, &t, l, n, k).unwrap().name;
    assert_eq!(pick("lin_in", 12, WeightKind::Fp8), "lin_in_f8b16");
    assert_eq!(pick("qkv", 24, WeightKind::Fp8), "qkv_f8_narrow");
    assert_eq!(pick("gate_up", 24, WeightKind::Fp4), "gate_up_f4a_skinnywide");
}

fn sweep() -> BTreeMap<(String, String, String, usize), BTreeMap<String, f64>> {
    let text = std::fs::read_to_string(root().join("../../docs/perf/gemm-select/sweep-sm120.tsv")).unwrap();
    let mut out: BTreeMap<_, BTreeMap<String, f64>> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let p: Vec<&str> = line.split('\t').collect();
        if p.len() != 6 {
            continue;
        }
        out.entry((p[0].to_string(), p[1].to_string(), p[2].to_string(), p[3].parse().unwrap())).or_default().insert(p[4].to_string(), p[5].parse().unwrap());
    }
    out
}

fn sweep_targets() -> PathBuf {
    root().join("../../docs/perf/gemm-select/sweep-sm120.tsv")
}

#[test]
fn measured_picks_from_the_sweep_are_within_five_percent_of_the_fastest_eligible_kernel() {
    assert!(sweep_targets().is_file());
    let t = kir::Target::sm_120();
    let table = sweep();
    let mut checked = 0;
    for (model, label, spec) in specs().into_iter().filter(|s| s.0 != "0.6b") {
        for linear in LINEARS {
            for kind in kinds(&spec, linear) {
                let w = if kind == WeightKind::Fp4 { "nvfp4" } else { "fp8" };
                for n in TOKENS {
                    let Some(times) = table.get(&(model.to_string(), linear.to_string(), w.to_string(), n)) else { continue };
                    let ok: Vec<String> = specrt::gemm::eligible(&spec, &t, linear, n, kind, true).into_iter().map(|c| c.name).collect();
                    let best = ok.iter().filter_map(|k| times.get(k)).fold(f64::INFINITY, |a, &b| a.min(b));
                    let bucket = kir::gemm::tune_bucket(n).unwrap();
                    let probe = table.get(&(model.to_string(), linear.to_string(), w.to_string(), bucket)).unwrap_or(times);
                    let at_bucket: Vec<String> = specrt::gemm::eligible(&spec, &t, linear, bucket, kind, true).into_iter().map(|c| c.name).collect();
                    let pick = at_bucket.iter().filter(|k| ok.contains(k)).min_by(|a, b| probe[a.as_str()].total_cmp(&probe[b.as_str()])).unwrap();
                    let ratio = times[pick.as_str()] / best;
                    assert!(ratio <= 1.05, "{model} {label} {linear} {w} {n} rows: measured pick {pick} is {ratio:.3}x the fastest");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 400, "{checked} cells");
}

#[test]
fn the_formula_cold_default_is_no_worse_than_the_old_thresholds_on_the_sweep() {
    let t = kir::Target::sm_120();
    let table = sweep();
    let (mut formula, mut fixed, mut cells) = (0.0, 0.0, 0);
    for (model, _, spec) in specs().into_iter().filter(|s| s.0 != "0.6b") {
        let mut old = spec.clone();
        old.runtime.gemm_select = GemmSelect::Fixed;
        let mut cold = spec.clone();
        cold.runtime.gemm_select = GemmSelect::Formula;
        for linear in LINEARS {
            for kind in kinds(&spec, linear) {
                let w = if kind == WeightKind::Fp4 { "nvfp4" } else { "fp8" };
                for n in TOKENS {
                    let Some(times) = table.get(&(model.to_string(), linear.to_string(), w.to_string(), n)) else { continue };
                    let best = specrt::gemm::eligible(&spec, &t, linear, n, kind, true).iter().filter_map(|c| times.get(&c.name)).fold(f64::INFINITY, |a, &b| a.min(b));
                    let at = |s: &Spec| times[&specrt::gemm::select(s, &t, linear, n, kind).unwrap().name] / best;
                    formula += at(&cold);
                    fixed += at(&old);
                    cells += 1;
                }
            }
        }
    }
    let (formula, fixed) = (formula / cells as f64, fixed / cells as f64);
    eprintln!("mean time over the fastest eligible kernel, {cells} cells: formula {formula:.4}, fixed thresholds {fixed:.4}");
    assert!(formula <= fixed, "formula {formula} against fixed {fixed}");
}
