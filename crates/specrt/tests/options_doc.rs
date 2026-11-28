use specrt::options::{Group, Kind, schema};
use std::path::{Path, PathBuf};

fn doc_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/generated/engine-options.md")
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn kind(k: &Kind) -> String {
    match k {
        Kind::Bool => "bool".into(),
        Kind::Int { min, max } if *max == u64::MAX => format!("int, from {min}"),
        Kind::Int { min, max } => format!("int, {min} to {max}"),
        Kind::Choice(c) => c.join(", "),
        Kind::Text => "text".into(),
        Kind::Json => "list".into(),
    }
}

fn render() -> String {
    let fields = schema();
    let mut out = vec![
        "# Engine options".to_string(),
        String::new(),
        "Generated from `specrt::options::schema()` by `crates/specrt/tests/options_doc.rs`; do not edit by hand. Rewrite with `OPTIONS_WRITE=1 cargo test -p inference-bench-specrt --test options_doc`.".to_string(),
        String::new(),
        "Every key can be set in the `[engine]` table of a server model, in the load dialog, or in a spec TOML under the section named by its prefix (`serve.`, `speculative.`, `runtime.`, `prefix_cache.`, `loader.`). An empty default means unset: the engine picks a value at load.".to_string(),
    ];
    for g in Group::ALL {
        let rows: Vec<_> = fields.iter().filter(|f| f.group == g).collect();
        if rows.is_empty() {
            continue;
        }
        out.push(String::new());
        out.push(format!("## {}", g.label()));
        out.push(String::new());
        out.push("| key | default | kind | what it does |".into());
        out.push("|---|---|---|---|".into());
        for f in rows {
            let default = if f.default.is_null() { String::new() } else { format!("`{}`", f.default) };
            out.push(format!("| `{}` | {} | {} | {} |", f.key, default, cell(&kind(&f.kind)), cell(f.help)));
        }
    }
    out.join("\n") + "\n"
}

#[test]
fn the_options_doc_matches_the_schema() {
    let text = render();
    if std::env::var_os("OPTIONS_WRITE").is_some() {
        std::fs::write(doc_path(), &text).unwrap();
    }
    let committed = std::fs::read_to_string(doc_path()).unwrap_or_default();
    assert!(committed == text, "docs/generated/engine-options.md is out of date: rerun with OPTIONS_WRITE=1");
}
