use cudarc::driver::CudaContext;
use specrt::link::{DeviceLink, LinkMode};
use specrt::runtime::{Layout, Runtime, StageRange};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec
}

fn pattern(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.37 - 11.0).collect()
}

fn round_trip(mode: LinkMode) {
    let a = CudaContext::new(0).unwrap();
    let b = CudaContext::new(0).unwrap();
    let sa = a.new_stream().unwrap();
    let sb = b.new_stream().unwrap();
    let link = DeviceLink::with_mode(&a, &b, mode).unwrap();
    assert_eq!(link.mode(), mode);
    let host = pattern(100_003);
    let src = sa.memcpy_stod(&host).unwrap();
    let mut dst = sb.alloc_zeros::<f32>(host.len()).unwrap();
    link.copy(&src, &sa, &mut dst, &sb, host.len()).unwrap();
    let back = sb.memcpy_dtov(&dst).unwrap();
    assert_eq!(back, host);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn peer_copy_moves_a_buffer_between_two_contexts() {
    round_trip(LinkMode::Peer);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn staged_copy_moves_a_buffer_between_two_contexts() {
    round_trip(LinkMode::Staged);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_copy_longer_than_the_buffers_is_refused() {
    let a = CudaContext::new(0).unwrap();
    let sa = a.new_stream().unwrap();
    let link = DeviceLink::new(&a, &a).unwrap();
    let src = sa.alloc_zeros::<f32>(8).unwrap();
    let mut dst = sa.alloc_zeros::<f32>(8).unwrap();
    assert!(link.copy(&src, &sa, &mut dst, &sa, 9).is_err());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn two_runtimes_in_one_process_agree_on_logits() {
    let spec = spec_for("tiny-qwen3.5");
    let layout = Layout::contiguous(128, 1);
    let mut first = Runtime::load_stage(&spec, layout, StageRange::whole(&spec)).unwrap();
    let mut second = Runtime::load_stage(&spec, layout, StageRange::whole(&spec)).unwrap();
    for rt in [&mut first, &mut second] {
        let pages = rt.max_ctx().div_ceil(specrt::codegen::PAGE) as u32;
        let blocks: Vec<u32> = (0..pages).rev().collect();
        rt.set_block_table(0, &blocks).unwrap();
    }
    assert_eq!(first.device_ordinal(), 0);
    let tokens: Vec<u32> = (1..40).collect();
    let a = first.prompt_logits(&tokens).unwrap();
    let b = second.prompt_logits(&tokens).unwrap();
    let again = first.prompt_logits(&tokens).unwrap();
    let worst = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    assert!(worst(&a, &again) < 1e-6, "first runtime not repeatable: {}", worst(&a, &again));
    assert!(worst(&a, &b) < 1e-6, "runtimes differ by {}", worst(&a, &b));
}
