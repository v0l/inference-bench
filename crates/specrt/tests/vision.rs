use specrt::vision::*;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn config(file: &str) -> Preprocessor {
    let mut cfg = Preprocessor::from_json(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file)).unwrap()).unwrap();
    cfg.size = SizeBounds { shortest_edge: 4096, longest_edge: 1048576 };
    cfg
}

fn reference(name: &str) -> (serde_json::Value, Vec<f32>) {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir().join(format!("{name}.json"))).unwrap()).unwrap();
    let raw = std::fs::read(dir().join(format!("{name}.f32"))).unwrap();
    (meta, raw.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn grid(meta: &serde_json::Value) -> [usize; 3] {
    let g = meta["grid"].as_array().unwrap();
    [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize]
}

fn worst(a: &[f32], b: &[f32]) -> (usize, f32) {
    let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
    (bad, a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max))
}

fn check_image(name: &str, file: &str, cfg: &Preprocessor) {
    let (meta, want) = reference(name);
    let got = preprocess_image(cfg, &std::fs::read(dir().join(file)).unwrap()).unwrap();
    assert_eq!(got.grid, grid(&meta), "{name} grid");
    assert_eq!(got.data.len(), want.len());
    let (bad, max) = worst(&got.data, &want);
    assert_eq!(bad, 0, "{name}: {bad} of {} differ, max {max}", want.len());
}

#[test]
fn image_without_resize_is_bit_identical_to_transformers() {
    check_image("image_exact", "image_exact.png", &config("qwen3.8-27b.image-preprocessor.json"));
}

#[test]
fn resized_image_is_bit_identical_to_transformers() {
    check_image("image_resize", "image_resize.png", &config("qwen3.8-27b.image-preprocessor.json"));
}

#[test]
fn large_image_with_config_bounds_is_bit_identical_to_transformers() {
    let cfg = Preprocessor::from_json(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.image-preprocessor.json")).unwrap()).unwrap();
    check_image("image_big", "image_big.png", &cfg);
}

fn check_video(name: &str) {
    let (meta, want) = reference(name);
    let mut cfg = config("qwen3.8-27b.video-preprocessor.json");
    if name == "clip" {
        cfg.size.longest_edge = 25165824;
    }
    let got = preprocess_video_file(&cfg, &dir().join(format!("{name}.mp4"))).unwrap();
    assert_eq!(got.patches.grid, grid(&meta), "{name} grid");
    let (bad, max) = worst(&got.patches.data, &want);
    assert_eq!(bad, 0, "{name}: {bad} of {} differ, max {max}", want.len());
}

#[test]
fn video_without_resize_is_bit_identical_to_transformers() {
    check_video("video_exact");
}

#[test]
fn resized_video_is_bit_identical_to_transformers() {
    check_video("video_resize");
}

#[test]
fn frame_sampling_matches_transformers_indices() {
    let (meta, _) = reference("video_resize");
    let cfg = config("qwen3.8-27b.video-preprocessor.json");
    let want: Vec<usize> = meta["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    assert_eq!(sample_indices(&cfg, meta["frames"].as_u64().unwrap() as usize, meta["fps"].as_f64().unwrap()), want);
}

#[test]
fn placeholders_expand_to_the_merged_token_count() {
    let tokens = MediaTokens::default();
    let image = Patches { grid: [1, 4, 6], data: vec![] };
    let prompt = "a<|vision_start|><|image_pad|><|vision_end|>b";
    let out = expand_placeholders(prompt, &tokens, 2, &[Media::Image(&image)]).unwrap();
    assert_eq!(out, format!("a<|vision_start|>{}<|vision_end|>b", "<|image_pad|>".repeat(6)));
    let video = VideoPatches { patches: Patches { grid: [2, 4, 4], data: vec![] }, timestamps: vec![0.25, 1.5] };
    let prompt = "<|vision_start|><|video_pad|><|vision_end|>";
    let out = expand_placeholders(prompt, &tokens, 2, &[Media::Video(&video)]).unwrap();
    let pad = "<|video_pad|>".repeat(4);
    assert_eq!(out, format!("<|vision_start|><0.2 seconds><|vision_start|>{pad}<|vision_end|><1.5 seconds><|vision_start|>{pad}<|vision_end|><|vision_end|>"));
    assert!(expand_placeholders("no slot", &tokens, 2, &[Media::Image(&image)]).is_err());
}

#[test]
fn the_27b_config_declares_a_vision_tower() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.multimodal.json")).unwrap();
    let v = Vision::from_hf_config(&json).unwrap().unwrap();
    assert_eq!((v.config.depth, v.config.hidden_size, v.config.num_heads, v.config.out_hidden_size), (27, 1152, 16, 5120));
    assert_eq!((v.tokens.image, v.tokens.video, v.tokens.start, v.tokens.end), (248056, 248057, 248053, 248054));
    assert!(v.config.deepstack_visual_indexes.is_empty());
}

#[test]
fn text_only_configs_have_no_vision() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap();
    assert!(Vision::from_hf_config(&json).unwrap().is_none());
}

#[test]
fn clip_with_the_config_bounds_is_bit_identical_to_transformers() {
    check_video("clip");
}
