use cudarc::driver::CudaContext;
use specrt::nvfp4::quantize_on_device;
use specrt::weights::{f32_to_bf16, quantize_nvfp4};

fn head_rows(rows: usize, cols: usize) -> Vec<u16> {
    let mut x = 0x9e3779b97f4a7c15u64;
    (0..rows * cols)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let r = i / cols;
            let scale = [0.001, 0.7, 3.0, 0.0, 40.0][r % 5];
            let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale * if i % 29 == 0 { 9.0 } else { 1.0 };
            f32_to_bf16(v)
        })
        .collect()
}

fn check(rows: usize, cols: usize, w: Vec<u16>) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let dir = std::env::temp_dir().join("specrt-nvfp4-quant-test");
    let got = quantize_on_device(&stream, &dir, &w, cols).unwrap();
    let packed = stream.clone_dtoh(&got.packed).unwrap();
    let scales = stream.clone_dtoh(&got.scales).unwrap();
    let want = quantize_nvfp4(&w, cols);
    assert_eq!(want.rows(), rows);
    assert_eq!(packed, want.packed.as_ref());
    assert_eq!(scales, want.device_scales());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn device_quantiser_is_byte_identical_to_the_cpu_path() {
    check(37, 64, head_rows(37, 64));
    check(4099, 128, head_rows(4099, 128));
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn device_quantiser_matches_the_cpu_path_on_the_tiny_fixture_draft_head() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-mtp");
    let mut spec = specrt::spec::Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.repo = dir.display().to_string();
    let ck = specrt::weights::Checkpoint::for_repo(&spec.repo).unwrap().renamed(&spec.tensors);
    let (w, cols) = ck.draft_head_bf16(&spec).unwrap().expect("fixture head is bf16");
    assert_eq!(cols, spec.dims.hidden);
    check(spec.dims.vocab, cols, w.to_vec());
}

#[test]
#[ignore]
fn head_sized_timing() {
    let (rows, cols) = (248320, 5120);
    let w = head_rows(rows, cols);
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let dir = std::env::temp_dir().join("specrt-nvfp4-quant-test");
    let t = std::time::Instant::now();
    let cpu = quantize_nvfp4(&w, cols);
    let cpu_s = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let got = quantize_on_device(&stream, &dir, &w, cols).unwrap();
    stream.synchronize().unwrap();
    let gpu_s = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let up = stream.clone_htod(&w).unwrap();
    stream.synchronize().unwrap();
    let up_s = t.elapsed().as_secs_f64();
    drop(up);
    println!("cpu quantize {cpu_s:.3}s, device quantize incl upload {gpu_s:.3}s, raw bf16 upload {up_s:.3}s");
    assert_eq!(stream.clone_dtoh(&got.packed).unwrap(), cpu.packed.as_ref());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn device_quantiser_handles_an_all_zero_head() {
    check(4, 64, vec![0u16; 4 * 64]);
}
