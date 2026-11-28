use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" || name == "out" || name == "node_modules" || name == "__pycache__" {
            continue;
        }
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    out
}

fn rel(p: &Path) -> String {
    p.strip_prefix(repo()).unwrap().to_string_lossy().replace('\\', "/")
}

fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("](") {
        rest = &rest[i + 2..];
        let end = rest.find(')').unwrap_or(rest.len());
        out.push(rest[..end].trim().to_string());
    }
    out
}

fn path_chars(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_./-".contains(c)
}

fn named_paths(text: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes: Vec<char> = text.chars().collect();
    let p: Vec<char> = prefix.chars().collect();
    let mut i = 0;
    while i + p.len() <= bytes.len() {
        let before_ok = i == 0 || !(path_chars(bytes[i - 1]) || bytes[i - 1] == '$' || bytes[i - 1] == '{' || bytes[i - 1] == '/');
        if before_ok && bytes[i..i + p.len()] == p[..] {
            let mut j = i + p.len();
            while j < bytes.len() && path_chars(bytes[j]) {
                j += 1;
            }
            let next = bytes.get(j).copied();
            let templated = matches!(next, Some('<' | '*' | '{' | '$'));
            let s: String = bytes[i..j].iter().collect();
            let s = s.trim_end_matches(['.', ',']).to_string();
            if !templated && s.len() > prefix.len() {
                out.push(s);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn docs() -> Vec<PathBuf> {
    files_under(&repo().join("docs")).into_iter().filter(|p| p.extension().is_some_and(|e| e == "md")).collect()
}

fn index_entries() -> BTreeSet<String> {
    let text = std::fs::read_to_string(repo().join("docs/README.md")).unwrap();
    links(&text).into_iter().filter(|l| !l.contains("://")).map(|l| l.split('#').next().unwrap().to_string()).collect()
}

#[test]
fn docs_top_level_holds_only_the_index_and_folders() {
    let mut loose = Vec::new();
    for e in std::fs::read_dir(repo().join("docs")).unwrap().flatten() {
        if e.path().is_file() && e.file_name() != "README.md" {
            loose.push(e.file_name().to_string_lossy().to_string());
        }
    }
    assert!(loose.is_empty(), "files at the top of docs/, move them into a folder and the index: {loose:?}");
}

#[test]
fn every_file_in_docs_is_in_the_index() {
    let index = index_entries();
    let mut missing = Vec::new();
    for f in files_under(&repo().join("docs")) {
        let r = rel(&f);
        let r = r.strip_prefix("docs/").unwrap();
        if r == "README.md" {
            continue;
        }
        let listed = index.iter().any(|e| e == r || (e.ends_with('/') && r.starts_with(e.as_str())));
        if !listed {
            missing.push(r.to_string());
        }
    }
    assert!(missing.is_empty(), "docs/README.md has no line for:\n{}", missing.join("\n"));
}

#[test]
fn every_relative_link_in_the_docs_resolves() {
    let mut files = docs();
    files.push(repo().join("README.md"));
    files.push(repo().join("AGENTS.md"));
    let mut broken = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        for l in links(&text) {
            let target = l.split('#').next().unwrap().split_whitespace().next().unwrap_or("");
            if target.is_empty() || target.contains("://") || target.starts_with("mailto:") || !target.contains(['.', '/']) {
                continue;
            }
            if !f.parent().unwrap().join(target).exists() {
                broken.push(format!("{}: {l}", rel(&f)));
            }
        }
    }
    assert!(broken.is_empty(), "broken links:\n{}", broken.join("\n"));
}

#[test]
fn every_path_a_doc_describes_exists() {
    let mut files = docs();
    files.push(repo().join("README.md"));
    files.push(repo().join("AGENTS.md"));
    let mut gone = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        for prefix in ["crates/", "docs/"] {
            for p in named_paths(&text, prefix) {
                if !repo().join(&p).exists() {
                    gone.push(format!("{}: {p}", rel(&f)));
                }
            }
        }
    }
    gone.sort();
    gone.dedup();
    assert!(gone.is_empty(), "docs name paths that do not exist:\n{}", gone.join("\n"));
}

#[test]
fn every_doc_path_named_in_code_exists() {
    let mut gone = Vec::new();
    for f in files_under(&repo().join("crates")).into_iter().chain([repo().join("CHANGELOG.md")]) {
        let r = rel(&f);
        if r.contains("/tests/fixtures/") || r.contains("/tests/golden/") {
            continue;
        }
        if !f.extension().is_some_and(|e| ["rs", "sh", "py", "toml", "md"].contains(&e.to_str().unwrap())) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        for p in named_paths(&text, "docs/") {
            if !repo().join(&p).exists() {
                gone.push(format!("{r}: {p}"));
            }
        }
    }
    for f in [".github/workflows/ci.yml", ".github/workflows/release.yml"] {
        let text = std::fs::read_to_string(repo().join(f)).unwrap_or_default();
        for p in named_paths(&text, "docs/") {
            if !repo().join(&p).exists() {
                gone.push(format!("{f}: {p}"));
            }
        }
    }
    assert!(gone.is_empty(), "code names doc paths that do not exist:\n{}", gone.join("\n"));
}

#[test]
fn the_path_scanner_sees_paths_and_skips_templates() {
    let t = "see `docs/design/kir.md`, docs/runs/<name>/ and $root/docs/x and crates/kir/src/lib.rs:12.";
    assert_eq!(named_paths(t, "docs/"), vec!["docs/design/kir.md".to_string()]);
    assert_eq!(named_paths(t, "crates/"), vec!["crates/kir/src/lib.rs".to_string()]);
    assert_eq!(links("a [b](c.md#d) e [f](g/)"), vec!["c.md#d".to_string(), "g/".to_string()]);
}
