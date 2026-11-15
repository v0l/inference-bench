use cudarc::driver::CudaContext;
use specrt::jit::{Jit, default_cache_dir};
use specrt::vision::{Patches, Preprocessor, Vision, preprocess_image};
use specrt::vision_encoder::{VisionEncoder, VisionPrecision};
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

const REPO: &str = "Qwen/Qwen3.8-27B-FP8";
const FAST_RMS: f32 = 5e-3;
const FAST_MAX: f32 = 1e-2;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn fixture(name: &str) -> Patches {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir().join(format!("{name}.json"))).unwrap()).unwrap();
    let g = meta["grid"].as_array().unwrap();
    let grid = [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize];
    Patches { grid, data: floats(dir().join(format!("{name}.f32"))) }
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name} output size");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let rms = (got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / want.len() as f32).sqrt();
    let rms_ref = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("{name}: max abs {max:e} of scale {scale}, rel rms {:e}", rms / rms_ref);
    (max / scale, rms / rms_ref)
}

fn encoder(precision: VisionPrecision) -> VisionEncoder {
    let ck = Checkpoint::for_repo(REPO).unwrap();
    let cfg_path = specrt::weights::find_in_snapshot(REPO, "config.json").unwrap();
    let vision = Vision::from_hf_config(&std::fs::read_to_string(cfg_path).unwrap()).unwrap().unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let jit = Jit::new(ctx.clone(), default_cache_dir()).unwrap();
    VisionEncoder::load_with(&ctx, &stream, &jit, &ck, &vision.config, precision).unwrap()
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint in the HF cache and a GPU"]
fn exact_encoder_matches_transformers_within_the_documented_bound() {
    let enc = encoder(VisionPrecision::Exact);
    for name in ["image_resize", "image_big", "video_resize", "clip"] {
        let got = enc.encode(&fixture(name)).unwrap();
        let (max_rel, rms_rel) = compare(name, &got.embeds, &floats(dir().join(format!("{name}.vision.f32"))));
        assert!(rms_rel < 1e-4, "{name}: relative rms {rms_rel}");
        assert!(max_rel < 5e-5, "{name}: max relative {max_rel}");
    }
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint in the HF cache and a GPU"]
fn fast_encoder_matches_transformers_within_the_fast_bound() {
    let enc = encoder(VisionPrecision::Fast);
    for name in ["image_resize", "image_big", "video_resize", "clip"] {
        let got = enc.encode(&fixture(name)).unwrap();
        let (max_rel, rms_rel) = compare(name, &got.embeds, &floats(dir().join(format!("{name}.vision.f32"))));
        assert!(rms_rel < FAST_RMS, "{name}: relative rms {rms_rel}");
        assert!(max_rel < FAST_MAX, "{name}: max relative {max_rel}");
    }
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint in the HF cache and a GPU"]
fn a_batch_of_images_encodes_like_each_image_alone() {
    let enc = encoder(VisionPrecision::default());
    let items = [fixture("image_resize"), fixture("image_big"), fixture("video_resize")];
    let refs: Vec<&Patches> = items.iter().collect();
    let batch = enc.encode_batch(&refs).unwrap();
    for (name, (p, got)) in ["image_resize", "image_big", "video_resize"].iter().zip(items.iter().zip(&batch)) {
        let alone = enc.encode(p).unwrap().embeds;
        let (max_rel, rms_rel) = compare(&format!("{name} in batch"), got, &alone);
        assert!(rms_rel < FAST_RMS && max_rel < FAST_MAX, "{name}: batch differs from single, rms {rms_rel} max {max_rel}");
    }
}

#[test]
#[ignore = "needs the 27B FP8 checkpoint in the HF cache and a GPU"]
fn a_960x1280_classify_photo_encodes_in_under_150_ms() {
    let enc = encoder(VisionPrecision::default());
    let cfg = Preprocessor::from_json(&std::fs::read_to_string(specrt::weights::find_in_snapshot(REPO, "preprocessor_config.json").unwrap()).unwrap()).unwrap();
    let img = image::RgbImage::from_fn(960, 1280, |x, y| image::Rgb([(x * 7 % 251) as u8, (y * 3 % 241) as u8, ((x + y) % 239) as u8]));
    let mut jpeg = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img).write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
    let patches = preprocess_image(&cfg, jpeg.get_ref()).unwrap();
    assert_eq!(patches.grid, [1, 80, 60]);
    enc.encode(&patches).unwrap();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let out = enc.encode(&patches).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(out.tokens, 1200);
        assert!(out.embeds.iter().all(|v| v.is_finite()));
    }
    let four = [&patches, &patches, &patches, &patches];
    let t = std::time::Instant::now();
    enc.encode_batch(&four).unwrap();
    let batch_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!("960x1280 grid [1, 80, 60]: best of 5 {best:.1} ms, batch of 4 {batch_ms:.1} ms");
    assert!(best < 150.0, "encode took {best:.1} ms");
}
