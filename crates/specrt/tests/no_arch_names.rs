use std::path::{Path, PathBuf};

const NAMES: &[&str] = &[
    "deepseek", "glm", "gpt_oss", "gptoss", "mamba", "llama", "mixtral", "gemma", "phi3", "falcon",
    "olmo", "minimax", "kimi", "nemotron", "granite", "qwen",
];

#[derive(PartialEq)]
enum Kind {
    Permanent,
    Debt,
}

struct Entry {
    path: &'static str,
    lines: usize,
    kind: Kind,
    note: &'static str,
}

const ALLOWED: &[Entry] = &[
    Entry { path: "cli/src/hostcmd.rs", lines: 2, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "cli/src/launch.rs", lines: 3, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "cli/src/main.rs", lines: 1, kind: Kind::Permanent, note: "help text names other engines" },
    Entry { path: "core/src/cluster.rs", lines: 2, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "core/src/discover.rs", lines: 29, kind: Kind::Permanent, note: "discovery of model repos; tests name real repos" },
    Entry { path: "core/src/hf.rs", lines: 11, kind: Kind::Permanent, note: "Hugging Face client tests use real repo names" },
    Entry { path: "core/src/library.rs", lines: 92, kind: Kind::Permanent, note: "model catalogue and display names" },
    Entry { path: "core/src/matrix.rs", lines: 5, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "core/src/openai.rs", lines: 16, kind: Kind::Permanent, note: "OpenAI client tests use model ids" },
    Entry { path: "core/src/report.rs", lines: 3, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "core/src/result.rs", lines: 1, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "core/src/settings.rs", lines: 3, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "core/src/specs/tests.rs", lines: 16, kind: Kind::Permanent, note: "spec catalogue tests" },
    Entry { path: "core/src/store.rs", lines: 2, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "server/src/app.rs", lines: 2, kind: Kind::Permanent, note: "route docs name the llama.cpp dialect" },
    Entry { path: "server/src/config.rs", lines: 12, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "server/src/connect.rs", lines: 4, kind: Kind::Permanent, note: "tests use model ids" },
    Entry { path: "server/src/metrics.rs", lines: 10, kind: Kind::Permanent, note: "tests use model ids" },
    Entry { path: "server/src/serve.rs", lines: 2, kind: Kind::Permanent, note: "tests use model ids" },
    Entry { path: "server/src/tools_tests.rs", lines: 1, kind: Kind::Permanent, note: "tests use model ids" },
    Entry { path: "specrt/src/anatomy.rs", lines: 23, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/cluster.rs", lines: 2, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/engine.rs", lines: 5, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/gguf.rs", lines: 15, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/placement.rs", lines: 29, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/plan.rs", lines: 13, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/serve.rs", lines: 2, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/solve.rs", lines: 1, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/structure.rs", lines: 7, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/tp.rs", lines: 15, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "specrt/src/weights.rs", lines: 11, kind: Kind::Permanent, note: "tests only: fixtures and real model names" },
    Entry { path: "ui/src/block.rs", lines: 11, kind: Kind::Permanent, note: "rope scaling label and tests over real model fixtures" },
    Entry { path: "ui/src/detail.rs", lines: 12, kind: Kind::Permanent, note: "UI tests over real model fixtures" },
    Entry { path: "ui/src/engine.rs", lines: 6, kind: Kind::Permanent, note: "UI tests over real model fixtures" },
    Entry { path: "ui/src/load.rs", lines: 1, kind: Kind::Permanent, note: "example served name shown as a hint" },
    Entry { path: "ui/src/models.rs", lines: 1, kind: Kind::Permanent, note: "example model name shown as a hint" },
    Entry { path: "core/src/fit.rs", lines: 1, kind: Kind::Permanent, note: "fit tests over a real model fixture" },
    Entry { path: "ui/src/table.rs", lines: 3, kind: Kind::Permanent, note: "UI tests over real model fixtures" },
    Entry { path: "ui/src/tests.rs", lines: 153, kind: Kind::Permanent, note: "UI tests over real model fixtures" },
];

fn allowed(rel: &str) -> usize {
    ALLOWED.iter().find(|e| e.path == rel).map_or(0, |e| e.lines)
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn arch_lines(path: &Path) -> usize {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .filter(|line| {
            let lower = line.to_lowercase();
            NAMES.iter().any(|name| lower.contains(name))
        })
        .count()
}

#[test]
fn architecture_names_stay_inside_the_allowlist() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files = Vec::new();
    for entry in std::fs::read_dir(crates).unwrap() {
        let src = entry.unwrap().path().join("src");
        if src.is_dir() {
            sources(&src, &mut files);
        }
    }
    let mut failures = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(crates).unwrap().to_string_lossy().replace('\\', "/");
        let found = arch_lines(file);
        let limit = allowed(&rel);
        if found > limit {
            failures.push(format!("{rel}: {found} lines name an architecture, {limit} allowed"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const PASS_FILES: &[&str] = &["specrt/src/compile.rs", "specrt/src/fusion.rs", "specrt/src/signature.rs", "kir/src/heavy.rs"];

const PASS_FORBIDDEN: &[&str] = &["Block::", "spec.attention.qk_norm_full", "spec.norm.sandwich", "spec.moe", "spec.linear_attention", "DeviceMatrix"];

#[test]
fn pass_and_cost_files_name_no_architecture() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut failures = Vec::new();
    for rel in PASS_FILES {
        let text = std::fs::read_to_string(crates.join(rel)).unwrap();
        for (n, line) in text.lines().enumerate() {
            let lower = line.to_lowercase();
            if let Some(name) = NAMES.iter().find(|name| lower.contains(*name)) {
                failures.push(format!("{rel}:{}: names {name}", n + 1));
            }
        }
        if *rel != "specrt/src/signature.rs" {
            for f in PASS_FORBIDDEN.iter().filter(|f| text.contains(*f)) {
                failures.push(format!("{rel}: reads {f}, a pass reads signatures and the target only"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_pass_guard_sees_a_planted_name() {
    let lower = "if spec.name.contains(\"qwen\") {".to_lowercase();
    assert!(NAMES.iter().any(|n| lower.contains(n)));
    assert!(PASS_FORBIDDEN.iter().any(|f| "match spec.block(i) { Block::Linear => }".contains(f)));
}

#[test]
fn allowlist_has_no_stale_entries() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for entry in ALLOWED {
        let found = arch_lines(&crates.join(entry.path));
        assert!(found >= entry.lines, "{}: allowlist {} but only {found} lines, lower it", entry.path, entry.lines);
        assert!(!entry.note.is_empty(), "{} needs a reason", entry.path);
    }
}

fn runtime_lines(path: &Path) -> usize {
    let text = std::fs::read_to_string(path).unwrap();
    let name = path.file_name().unwrap().to_string_lossy();
    if name == "tests.rs" || name.ends_with("_tests.rs") {
        return 0;
    }
    let runtime: Vec<&str> = text
        .lines()
        .take_while(|l| !l.starts_with("#[cfg(test)]") && !l.starts_with("mod tests"))
        .collect();
    runtime
        .iter()
        .filter(|line| {
            let lower = line.to_lowercase();
            NAMES.iter().any(|name| lower.contains(name))
        })
        .count()
}

fn has_issue_number(note: &str) -> bool {
    note.split('#').skip(1).any(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

#[test]
fn engine_crates_name_no_architecture_outside_tests_unless_debt_with_an_issue() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut failures = Vec::new();
    for entry in ALLOWED {
        let engine = entry.path.starts_with("specrt/src/") || entry.path.starts_with("kir/src/");
        if entry.kind == Kind::Debt && !has_issue_number(entry.note) {
            failures.push(format!("{}: debt entry without an issue number", entry.path));
        }
        if engine && entry.kind == Kind::Permanent && runtime_lines(&crates.join(entry.path)) > 0 {
            failures.push(format!("{}: names an architecture outside tests, so it is debt and needs an issue", entry.path));
        }
    }
    let mut files = Vec::new();
    for dir in ["specrt/src", "kir/src"] {
        sources(&crates.join(dir), &mut files);
    }
    for file in files {
        let rel = file.strip_prefix(crates).unwrap().to_string_lossy().replace('\\', "/");
        let listed = ALLOWED.iter().any(|e| e.path == rel);
        if runtime_lines(&file) > 0 && !listed {
            failures.push(format!("{rel}: architecture name outside tests and not listed as debt"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
