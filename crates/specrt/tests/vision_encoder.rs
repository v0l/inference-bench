use cudarc::driver::CudaContext;
use specrt::jit::{Jit, default_cache_dir};
use specrt::vision::{Patches, Vision};
use specrt::vision_encoder::VisionEncoder;
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn case(name: &str, enc: &VisionEncoder) -> (f32, f32) {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir().join(format!("{name}.json"))).unwrap()).unwrap();
    let g = meta["grid"].as_array().unwrap();
    let grid = [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize];
    let patches = Patches { grid, data: floats(dir().join(format!("{name}.f32"))) };
    let want = floats(dir().join(format!("{name}.vision.f32")));
    let got = enc.encode(&patches).unwrap();
    assert_eq!(got.embeds.len(), want.len(), "{name} output size");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max = got.embeds.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let rms = (got.embeds.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / want.len() as f32).sqrt();
    let rms_ref = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("{name}: max abs {max:e} of scale {scale}, rel rms {:e}", rms / rms_ref);
    (max / scale, rms / rms_ref)
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint in the HF cache and a GPU"]
fn encoder_matches_transformers_within_the_documented_bound() {
    let repo = "Qwen/Qwen3.8-27B-FP8";
    let ck = Checkpoint::for_repo(repo).unwrap();
    let cfg_path = specrt::weights::find_in_snapshot(repo, "config.json").unwrap();
    let vision = Vision::from_hf_config(&std::fs::read_to_string(cfg_path).unwrap()).unwrap().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let jit = Jit::new(ctx.clone(), default_cache_dir()).unwrap();
    let enc = VisionEncoder::load(&ctx, &stream, &jit, &ck, &vision.config).unwrap();
    for name in ["image_resize", "image_big", "video_resize"] {
        let (max_rel, rms_rel) = case(name, &enc);
        assert!(rms_rel < 1e-4, "{name}: relative rms {rms_rel}");
        assert!(max_rel < 5e-5, "{name}: max relative {max_rel}");
    }
}
