use specrt::runtime::{Layout, Runtime, kv_block_bytes};
use specrt::spec::{KvDtype, Spec};
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, kv: KvDtype) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.kv_dtype = kv;
    spec
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let z = x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln();
    x.iter().map(|v| *v as f64 - m - z).collect()
}

fn kl(reference: &[f32], got: &[f32]) -> f64 {
    let (p, q) = (log_softmax(reference), log_softmax(got));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

#[test]
fn fp8_kv_stores_one_byte_per_value_plus_a_scale_per_row() {
    let f16 = spec_for("tiny-qwen3.5-hd256", KvDtype::F16);
    let fp8 = spec_for("tiny-qwen3.5-hd256", KvDtype::Fp8);
    let rows = 2 * fp8.kv_layers() * fp8.attention.kv_heads * specrt::codegen::PAGE;
    assert_eq!(kv_block_bytes(&f16, 1), rows * fp8.attention.head_dim * 2);
    assert_eq!(kv_block_bytes(&fp8, 1), rows * (fp8.attention.head_dim + 4));
    let real = Spec::load(&fixture_dir("qwen3.8-27b-exact.toml")).unwrap();
    let mut real8 = real.clone();
    real8.runtime.kv_dtype = KvDtype::Fp8;
    let per_token = |s: &Spec| kv_block_bytes(s, 1) / specrt::codegen::PAGE;
    assert!(per_token(&real8) * 100 < per_token(&real) * 52, "{} vs {}", per_token(&real8), per_token(&real));
}

#[test]
fn kv_dtype_reads_from_the_runtime_table() {
    let t: specrt::spec::RuntimeOptions = toml::from_str("kv_dtype = \"fp8\"").unwrap();
    assert_eq!(t.kv_dtype, KvDtype::Fp8);
    assert_eq!(specrt::spec::RuntimeOptions::default().kv_dtype, KvDtype::F16);
}

fn fp8_logits_track_f16(dir: &str, len: usize) {
    let prompt: Vec<u32> = (0..len as u32).map(|i| (i * 37 + 11) % 1024).collect();
    let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
    let run = |kv: KvDtype| {
        let s = spec_for(dir, kv);
        let layout = Layout { prefill_chunk: 128, kv_planes: 1, ..Layout::contiguous(512, 1) };
        let mut rt = Runtime::load_layout(&s, layout).unwrap();
        rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
        rt.prompt_logits(&prompt).unwrap()
    };
    let base = run(KvDtype::F16);
    let got = run(KvDtype::Fp8);
    let d = kl(&base, &got);
    let top = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    eprintln!("{dir} len {len}: kl {d:.6}");
    assert!(d < 0.02, "kl {d}");
    assert_eq!(top(&base), top(&got));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny_fp8_kv_logits_stay_within_the_documented_kl_of_f16() {
    fp8_logits_track_f16("tiny-qwen3.5", 300);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_dim_256_fp8_kv_logits_stay_within_the_documented_kl_of_f16() {
    fp8_logits_track_f16("tiny-qwen3.5-hd256", 300);
}
