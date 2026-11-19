use std::path::{Path, PathBuf};

const NAMES: &[&str] = &[
    "deepseek", "glm", "gpt_oss", "gptoss", "mamba", "llama", "mixtral", "gemma", "phi3", "falcon",
    "olmo", "minimax", "kimi", "nemotron", "granite", "qwen",
];

const ALLOWED_LINES: &[(&str, usize)] = &[
    ("cli/src/hostcmd.rs", 2),
    ("cli/src/launch.rs", 5),
    ("cli/src/main.rs", 1),
    ("core/src/cluster.rs", 2),
    ("core/src/discover.rs", 29),
    ("core/src/hf.rs", 11),
    ("core/src/library.rs", 93),
    ("core/src/matrix.rs", 5),
    ("core/src/openai.rs", 16),
    ("core/src/report.rs", 3),
    ("core/src/result.rs", 1),
    ("core/src/settings.rs", 3),
    ("core/src/specs/tests.rs", 16),
    ("core/src/store.rs", 2),
    ("server/src/app.rs", 2),
    ("server/src/config.rs", 15),
    ("server/src/connect.rs", 4),
    ("server/src/metrics.rs", 10),
    ("server/src/serve.rs", 2),
    ("server/src/tools_tests.rs", 1),
    ("specrt/src/anatomy.rs", 23),
    ("specrt/src/chat.rs", 3),
    ("specrt/src/cluster.rs", 2),
    ("specrt/src/engine.rs", 5),
    ("specrt/src/gguf.rs", 23),
    ("specrt/src/options.rs", 1),
    ("specrt/src/placement.rs", 27),
    ("specrt/src/plan.rs", 13),
    ("specrt/src/serve.rs", 1),
    ("specrt/src/spec.rs", 3),
    ("specrt/src/structure.rs", 7),
    ("specrt/src/tp.rs", 12),
    ("specrt/src/weights.rs", 11),
    ("ui/src/block.rs", 12),
    ("ui/src/detail.rs", 12),
    ("ui/src/engine.rs", 6),
    ("ui/src/load.rs", 1),
    ("ui/src/models.rs", 1),
    ("ui/src/table.rs", 3),
    ("ui/src/tests.rs", 155),
];

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
        let allowed = ALLOWED_LINES
            .iter()
            .find(|(path, _)| *path == rel)
            .map_or(0, |(_, n)| *n);
        if found > allowed {
            failures.push(format!("{rel}: {found} lines name an architecture, {allowed} allowed"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn allowlist_has_no_stale_entries() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    for (path, allowed) in ALLOWED_LINES {
        let found = arch_lines(&crates.join(path));
        assert!(found >= *allowed, "{path}: allowlist {allowed} but only {found} lines, lower it");
    }
}
