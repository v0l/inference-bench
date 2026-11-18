use specrt::codegen::Module;
use specrt::spec::Spec;
use std::collections::BTreeSet;
use std::path::Path;

fn blocks(source: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut cur: Vec<&str> = Vec::new();
    for l in source.split('\n') {
        cur.push(l);
        if l == "}" || (l.ends_with(';') && !l.starts_with(' ') && cur.len() == 1) {
            out.insert(cur.join(" ").split_whitespace().collect::<Vec<_>>().join(" "));
            cur.clear();
        } else if l.trim().is_empty() && cur.len() == 1 {
            cur.clear();
        }
    }
    out
}

fn linear_related(b: &str) -> bool {
    ["LA_", "gdn_", "mtp_", "spec_accept", "lin_"].iter().any(|p| b.contains(p))
}

#[test]
fn kir_linear_kernels_equal_the_pre_kir_template_text_for_the_27b_fixture() {
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    let module = Module::generate_kv(&spec, 4096, 1, &kir::Target::sm_120()).unwrap();
    let got: BTreeSet<String> = blocks(&module.source).into_iter().filter(|b| linear_related(b)).collect();
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/master_linear_27b.txt");
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::write(&golden, got.iter().map(|b| format!("{b}\n")).collect::<String>()).unwrap();
    }
    let want: BTreeSet<String> = std::fs::read_to_string(&golden)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    let missing: Vec<_> = want.difference(&got).map(|s| &s[..s.len().min(120)]).collect();
    let extra: Vec<_> = got.difference(&want).map(|s| &s[..s.len().min(120)]).collect();
    assert!(missing.is_empty() && extra.is_empty(), "missing {missing:?} extra {extra:?}");
    assert!(want.iter().any(|b| b.contains("gdn_verify")) && want.iter().any(|b| b.contains("spec_accept")));
}
