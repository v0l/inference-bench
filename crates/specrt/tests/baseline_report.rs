use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn report(dir: &Path) -> String {
    let out = Command::new("python3")
        .arg(root().join("crates/specrt/scripts/baseline_report.py"))
        .arg(dir)
        .output()
        .expect("python3 runs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn vllm_points(dir: &Path, suite: &str, mode: &str, key: &str) -> Vec<u64> {
    let text = std::fs::read_to_string(dir.join(format!("{suite}-{mode}.json"))).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    doc["points"].as_array().unwrap().iter().map(|p| p[key].as_u64().unwrap()).collect()
}

fn grouped(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[test]
fn the_report_prints_every_committed_vllm_point_for_both_modes() {
    let dir = root().join("docs/generated/qwen27-run");
    let text = report(&dir);
    for mode in ["mtp", "nomtp"] {
        let single = text.split(&format!("### Single stream, 128 output tokens, {mode}")).nth(1).unwrap();
        for n in vllm_points(&dir, "single", mode, "prompt_len") {
            assert!(single.contains(&format!("\n| {} | ", grouped(n))), "single {mode} {n} missing");
        }
        let conc = text.split(&format!("### Concurrency, 128 token prompts, aggregate decode tok/s, {mode}")).nth(1).unwrap();
        for n in vllm_points(&dir, "concurrent", mode, "concurrent") {
            assert!(conc.contains(&format!("\n| {n} | ")), "concurrent {mode} {n} missing");
        }
        let prefix = text.split(&format!("### Prefix cache, cold and warm TTFT s, {mode}")).nth(1).unwrap();
        for n in vllm_points(&dir, "prefix", mode, "prompt_len") {
            assert!(prefix.contains(&format!("| {} cold | ", grouped(n))) && prefix.contains(&format!("| {} warm | ", grouped(n))), "prefix {mode} {n} missing");
        }
        let held = text.split(&format!("### Decode of 512 tokens after a held prefix, tok/s, {mode}")).nth(1).unwrap();
        for n in vllm_points(&dir, "held", mode, "prompt_len") {
            assert!(held.contains(&format!("\n| {} | ", grouped(n))), "held {mode} {n} missing");
        }
    }
    assert!(text.contains("| 32 | 2,319 | "), "{text}");
    assert!(text.contains("vLLM `mtp` mean acceptance length"), "{text}");
}

#[test]
fn a_specrt_point_beside_vllm_gets_its_ratio() {
    let dir = std::env::temp_dir().join(format!("baseline-report-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root().join("docs/generated/qwen27-run/concurrent-mtp.json"), dir.join("concurrent-mtp.json")).unwrap();
    let ours = serde_json::json!({"label": "specrt-mtp", "suite": "concurrent", "points": [{"concurrent": 8, "aggregate_decode_tok_s": 409.3, "median_ttft_s": 0.5}]});
    std::fs::write(dir.join("specrt-concurrent-mtp.json"), ours.to_string()).unwrap();
    let text = report(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(text.contains("| 8 | 818.6 | 409.3 | 0.50x | not run |  |"), "{text}");
    assert!(text.contains("| 4 | 431.5 | not run |  | not run |  |"), "{text}");
}
