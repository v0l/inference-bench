use cudarc::driver::CudaContext;
use specrt::quantize::{fp8_channel_search, fp8_channel_search_on_device, nvfp4_search, nvfp4_search_on_device};
use specrt::weights::f32_to_bf16;
use std::path::Path;

fn rows_of(rows: usize, cols: usize) -> Vec<u16> {
    let mut x = 0x9e3779b97f4a7c15u64;
    (0..rows * cols)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let scale = [0.001, 0.7, 3.0, 0.0, 40.0][(i / cols) % 5];
            let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale * if i % 29 == 0 { 9.0 } else { 1.0 };
            f32_to_bf16(v)
        })
        .collect()
}

fn case() -> (Vec<u16>, usize) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quant-search/case.json");
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    (c["bf16"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect(), c["cols"].as_u64().unwrap() as usize)
}

fn tiny_fixture_weight() -> (Vec<u16>, usize) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-mtp");
    let mut spec = specrt::spec::Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    let ck = specrt::weights::Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    let (w, cols) = ck.draft_head_bf16(&spec).unwrap().expect("fixture head is bf16");
    (w.to_vec(), cols)
}

fn check(w: &[u16], cols: usize) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let dir = std::env::temp_dir().join("specrt-quant-search-gpu-test");
    let got = nvfp4_search_on_device(&stream, &dir, w, cols).unwrap();
    let want = nvfp4_search(w, cols);
    assert_eq!(stream.clone_dtoh(&got.packed).unwrap(), want.packed.as_ref(), "nvfp4 packed");
    assert_eq!(stream.clone_dtoh(&got.scales).unwrap(), want.device_scales(), "nvfp4 scales");
    let got = fp8_channel_search_on_device(&stream, &dir, w, cols).unwrap();
    let (values, scales) = fp8_channel_search(w, cols);
    let rows = w.len() / cols;
    let want_rows: Vec<u32> = (0..rows).map(|r| scales[scales[r].to_bits() as usize].to_bits()).collect();
    let got_rows: Vec<u32> = stream.clone_dtoh(&got.row_scales).unwrap().iter().map(|s| s.to_bits()).collect();
    assert_eq!(got_rows, want_rows, "fp8 row scales");
    assert_eq!(stream.clone_dtoh(&got.values).unwrap(), values, "fp8 values");
}

#[test]
fn device_scale_search_matches_the_cpu_search_on_the_reference_case() {
    let (w, cols) = case();
    check(&w, cols);
}

#[test]
fn device_scale_search_matches_the_cpu_search_on_mixed_rows() {
    check(&rows_of(37, 64), 64);
    check(&rows_of(4099, 128), 128);
    check(&vec![0u16; 4 * 64], 64);
}

#[test]
fn device_scale_search_matches_the_cpu_search_on_the_tiny_fixture() {
    let (w, cols) = tiny_fixture_weight();
    check(&w, cols);
}
