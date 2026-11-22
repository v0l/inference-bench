use specrt::vision::*;
use specrt::vision_encoder::{plan, tower_params, tower_parts, VisionOp};
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
    if name == "clip" || name == "clip10" {
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
    let image = Patches { grid: [1, 4, 6], data: vec![], layout: [0, 0] };
    let prompt = "a<|vision_start|><|image_pad|><|vision_end|>b";
    let out = expand_placeholders(prompt, &tokens, 2, &[Media::Image(&image)]).unwrap();
    assert_eq!(out, format!("a<|vision_start|>{}<|vision_end|>b", "<|image_pad|>".repeat(6)));
    let video = VideoPatches { patches: Patches { grid: [2, 4, 4], data: vec![], layout: [0, 0] }, timestamps: vec![0.25, 1.5] };
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

fn tokens() -> VisionTokens {
    VisionTokens { image: 9, video: 8, start: 7, end: 6 }
}

fn image(h: usize, w: usize) -> Processed {
    Processed::Image(Patches { grid: [1, h, w], data: vec![], layout: [0, 0] })
}

#[test]
fn text_positions_advance_all_axes_and_vision_gets_a_grid() {
    let vis = tokens();
    let mut ids = vec![1, 2, 7];
    ids.extend([9; 6]);
    ids.extend([6, 3]);
    let plan = plan_media(&ids, &vis, 2, &[image(4, 6)], &[[1; 32]]).unwrap();
    let p = |i: usize| (plan.positions[i][0], plan.positions[i][1], plan.positions[i][2]);
    assert_eq!([p(0), p(1), p(2)], [(0, 0, 0), (1, 1, 1), (2, 2, 2)]);
    assert_eq!([p(3), p(4), p(5)], [(3, 3, 3), (3, 3, 4), (3, 3, 5)]);
    assert_eq!([p(6), p(7), p(8)], [(3, 4, 3), (3, 4, 4), (3, 4, 5)]);
    assert_eq!(p(9), (6, 6, 6));
    assert_eq!(p(10), (7, 7, 7));
    assert_eq!(plan.delta, 8 - 11);
    assert_eq!(plan.positions[3][3], 0);
    assert_eq!(plan.positions[8][3], 5);
    assert_eq!(plan.positions[0][3], -1);
}

#[test]
fn the_same_image_gives_the_same_prefix_tags_and_another_image_does_not() {
    let vis = tokens();
    let mut ids = vec![7];
    ids.extend([9; 4]);
    ids.push(6);
    let a = plan_media(&ids, &vis, 2, &[image(4, 4)], &[[1; 32]]).unwrap();
    let b = plan_media(&ids, &vis, 2, &[image(4, 4)], &[[1; 32]]).unwrap();
    let c = plan_media(&ids, &vis, 2, &[image(4, 4)], &[[2; 32]]).unwrap();
    assert_eq!(a.tags, b.tags);
    assert_ne!(a.tags[1..5], c.tags[1..5]);
    assert_eq!(a.tags[0], 0);
    assert!(a.tags[1..5].iter().all(|&t| t != 0));
    let distinct: std::collections::HashSet<_> = a.tags[1..5].iter().collect();
    assert_eq!(distinct.len(), 4);
}

#[test]
fn a_run_that_does_not_match_the_grid_or_the_media_count_is_an_error() {
    let vis = tokens();
    let ids = [7, 9, 9, 6];
    assert!(plan_media(&ids, &vis, 2, &[image(4, 4)], &[[1; 32]]).is_err());
    assert!(plan_media(&[1, 2], &vis, 2, &[image(2, 2)], &[[1; 32]]).is_err());
    assert!(plan_media(&[9], &vis, 2, &[], &[]).is_err());
}

#[test]
fn video_frames_each_advance_the_text_position_by_their_grid_side() {
    let vis = tokens();
    let video = Processed::Video(VideoPatches { patches: Patches { grid: [2, 4, 4], data: vec![], layout: [0, 0] }, timestamps: vec![0.0, 1.0] });
    let mut ids = vec![7, 5, 7];
    ids.extend([8; 4]);
    ids.extend([6, 5, 7]);
    ids.extend([8; 4]);
    ids.extend([6, 6]);
    let plan = plan_media(&ids, &vis, 2, &[video], &[[3; 32]]).unwrap();
    assert_eq!(plan.positions[3][..3], [3, 3, 3]);
    assert_eq!(plan.positions[6][..3], [3, 4, 4]);
    assert_eq!(plan.positions[10][..3], [8, 8, 8]);
    assert_eq!(plan.positions[14][..3], [10, 10, 10]);
        assert_eq!(plan.delta, plan.positions.last().unwrap()[0] + 1 - ids.len() as i32);
}

#[test]
fn the_tower_parameter_count_matches_the_checkpoint_tensors() {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.multimodal.json")).unwrap();
    let v = Vision::from_hf_config(&json).unwrap().unwrap();
    let repo = "Qwen/Qwen3.8-27B-FP8";
    if let Ok(dir) = specrt::weights::find_checkpoint(repo) {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let file = dir.join("outside.safetensors");
        let bytes = std::fs::read(&file).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        let mut total = 0u64;
        for (name, info) in header.as_object().unwrap() {
            if name.starts_with("model.visual.") && index["weight_map"].get(name).is_some() {
                total += info["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).product::<u64>();
            }
        }
        assert_eq!(tower_params(&v.config), total);
    }
    assert_eq!(tower_parts(&v.config).iter().find(|p| p.name == "attention (qkv, proj)").unwrap().count, 27);
}

#[test]
fn the_op_plan_has_a_merger_and_a_deepstack_merger_per_configured_block() {
    let mut cfg = Vision::from_hf_config(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.multimodal.json")).unwrap()).unwrap().unwrap().config;
    let ops = plan(&cfg);
    assert_eq!(ops.first(), Some(&VisionOp::PatchEmbed));
    assert_eq!(ops.last(), Some(&VisionOp::Merger));
    assert_eq!(ops.len(), 2 + 8 * 27);
    cfg.deepstack_visual_indexes = vec![5, 11];
    let ops = plan(&cfg);
    let deep: Vec<_> = ops.iter().filter(|o| matches!(o, VisionOp::DeepstackMerger { .. })).collect();
    assert_eq!(deep.len(), 2);
    assert_eq!(ops[ops.iter().position(|o| *o == VisionOp::DeepstackMerger { index: 0, after_block: 5 }).unwrap() - 1], VisionOp::MlpDown { block: 5 });
}

#[test]
fn the_ten_second_clip_has_the_reference_frames_grid_patches_and_timestamps() {
    check_video("clip10");
    let (meta, _) = reference("clip10");
    let mut cfg = config("qwen3.8-27b.video-preprocessor.json");
    cfg.size.longest_edge = 25165824;
    let want: Vec<usize> = meta["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let fps = meta["fps"].as_f64().unwrap();
    assert_eq!(sample_indices(&cfg, meta["frames"].as_u64().unwrap() as usize, fps), want);
    let got = preprocess_video_file(&cfg, &dir().join("clip10.mp4")).unwrap();
    let stamps: Vec<String> = got.timestamps.iter().map(|t| format!("{t:.1}")).collect();
    let want_stamps: Vec<String> = meta["timestamps"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(stamps, want_stamps);
    assert_eq!(got.patches.tokens(2), 800);
}

fn assets() -> MediaAssets {
    let base = |file: &str| {
        let mut c = config(file);
        c.size = SizeBounds { shortest_edge: 65536, longest_edge: 16777216 };
        c
    };
    MediaAssets { image: base("qwen3.8-27b.image-preprocessor.json"), video: base("qwen3.8-27b.video-preprocessor.json"), tokens: MediaTokens::default(), ids: tokens(), tiles: None }
}

fn png(w: u32, h: u32) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x % 251) as u8, (y % 241) as u8, 7]))).write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[test]
fn a_video_sent_as_an_image_is_refused_before_any_kernel_launch() {
    let mp4 = std::fs::read(dir().join("clip.mp4")).unwrap();
    let item = MediaItem::new(MediaKind::Image, mp4);
    let e = assets().process(&[item], MediaBudget::default()).unwrap_err().to_string();
    assert!(e.contains("video_url"), "{e}");
}

#[test]
fn a_huge_image_is_scaled_down_to_the_patch_budget() {
    let item = MediaItem::new(MediaKind::Image, png(4096, 4096));
    let unlimited = assets().process(std::slice::from_ref(&item), MediaBudget::default()).unwrap();
    let unlimited_patches: usize = unlimited[0].patches().grid.iter().product();
    assert!(unlimited_patches > 16384, "{unlimited_patches}");
    let budget = MediaBudget { max_patches: Some(4096), ..Default::default() };
    let limited = assets().process(&[item], budget).unwrap();
    let patches: usize = limited[0].patches().grid.iter().product();
    assert!(patches <= 4096, "{patches}");
}

#[test]
fn an_extreme_aspect_ratio_is_refused() {
    let item = MediaItem::new(MediaKind::Image, png(4000, 8));
    let e = assets().process(&[item], MediaBudget::default()).unwrap_err().to_string();
    assert!(e.contains("aspect ratio"), "{e}");
}

#[test]
fn a_video_over_the_frame_limit_is_refused() {
    let item = MediaItem::new(MediaKind::Video, std::fs::read(dir().join("clip10.mp4")).unwrap());
    let budget = MediaBudget { max_video_frames: Some(2), ..Default::default() };
    let e = assets().process(&[item], budget).unwrap_err().to_string();
    assert!(e.contains("sampled frames"), "{e}");
}
