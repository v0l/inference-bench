use specrt::runtime::Runtime;
use specrt::sched::MediaInput;
use specrt::spec::Spec;
use specrt::vision::{Patches, Processed};
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures/tiny-gemma3-vision")
}

fn ref_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR holds the output of scripts/tiny_gemma3_vision.py"))
}

fn spec() -> Spec {
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-gemma3-vision".into();
    spec.repo = fixture().display().to_string();
    spec
}

fn floats(name: &str) -> Vec<f32> {
    std::fs::read(ref_dir().join(name)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(ref_dir().join(name)).unwrap()).unwrap()
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    row.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

#[test]
fn the_tiny_config_reads_as_a_pooled_siglip_tower() {
    let s = spec();
    let v = s.vision.as_ref().unwrap();
    assert!(v.config.pooled() && v.config.bidirectional);
    assert_eq!((v.config.depth, v.config.hidden_size, v.config.num_heads, v.config.patch_size), (2, 64, 4, 14));
    assert_eq!((v.config.side(), v.config.tokens_per_tile(), v.config.pool_kernel(), v.config.out_hidden_size), (4, 4, 2, 128));
    assert_eq!((v.tokens.image, v.tokens.start, v.tokens.end), (592, 590, 591));
    let plan = specrt::vision_encoder::plan(&v.config);
    assert!(plan.contains(&specrt::vision_encoder::VisionOp::Pool));
    assert!(!plan.contains(&specrt::vision_encoder::VisionOp::Merger));
}

#[test]
fn pan_and_scan_boxes_follow_the_processor_rules() {
    use specrt::vision::{PanAndScan, pan_and_scan_boxes};
    let p = PanAndScan { min_crop: 256, max_crops: 4, min_ratio: 1.2 };
    assert!(pan_and_scan_boxes(512, 512, p).is_empty(), "a square image is not cropped");
    assert!(pan_and_scan_boxes(500, 560, p).is_empty(), "below the ratio");
    let wide = pan_and_scan_boxes(300, 900, p);
    assert_eq!(wide, vec![[0, 0, 300, 300], [0, 300, 300, 600], [0, 600, 300, 900]]);
    let tall = pan_and_scan_boxes(2000, 500, p);
    assert_eq!(tall.len(), 4);
    assert_eq!(tall[3], [1500, 0, 2000, 500]);
    assert!(pan_and_scan_boxes(200, 900, p).is_empty(), "crops under the minimum size are dropped");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3_vision.py output in SPECRT_REF_DIR"]
fn tiny_gemma3_vision_matches_transformers_on_every_row() {
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    assert!(rt.has_vision());
    let vocab = 600;
    for (case, seed) in [("square", 1u8), ("wide", 2u8)] {
        let m = meta(&format!("{case}.json"));
        let grid: Vec<usize> = m["grid"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let ids: Vec<u32> = m["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let data = floats(&format!("{case}.patches.bin"));
        let media = vec![MediaInput { processed: Processed::Image(Patches { grid: [grid[0], grid[1], grid[2]], data }), hash: [seed; 32] }];
        let plan = rt.plan_media(&ids, &media).unwrap();
        let want = floats(&format!("{case}.embeds.bin"));
        assert_eq!(plan.embeds.len(), want.len(), "{case}");
        let num: f64 = plan.embeds.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        eprintln!("{case}: image embeddings relative L2 {rel:e}");
        assert!(rel < 0.01, "{case}: embeddings differ by {rel}");
        assert!(plan.spans.iter().enumerate().any(|(i, &s)| s as usize > i));

        rt.set_media(0, Some(&plan)).unwrap();
        let ours = rt.prompt_logits(&ids).unwrap();
        let theirs = floats(&format!("{case}.logits.bin"));
        let mut worst = 0f64;
        for r in 0..ids.len() {
            let a = &ours[r * vocab..(r + 1) * vocab];
            let b = &theirs[r * vocab..(r + 1) * vocab];
            let (la, lb) = (log_softmax(a), log_softmax(b));
            let kl: f64 = lb.iter().zip(&la).map(|(q, p)| q.exp() * (q - p)).sum();
            worst = worst.max(kl);
        }
        eprintln!("{case}: worst logits KL over every row, image tokens included {worst:e}");
        assert!(worst < 1e-4, "{case}: logits KL {worst}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_gemma3_vision.py output in SPECRT_REF_DIR"]
fn image_tokens_attend_to_each_other_in_both_directions() {
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let m = meta("square.json");
    let ids: Vec<u32> = m["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let grid: Vec<usize> = m["grid"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let data = floats("square.patches.bin");
    let media = vec![MediaInput { processed: Processed::Image(Patches { grid: [grid[0], grid[1], grid[2]], data }), hash: [9; 32] }];
    let plan = rt.plan_media(&ids, &media).unwrap();
    let first = plan.spans.iter().enumerate().find(|(i, s)| **s as usize > *i).map(|(i, _)| i).unwrap();
    assert_eq!(plan.spans[first] as usize, first + 3);
    let split = plan.extend_chunk(0, first + 2, 64);
    assert_eq!(split, first, "a chunk ending inside an image block is cut back to the block start");
    let whole = plan.extend_chunk(first, first + 2, 64);
    assert_eq!(whole, first + 4, "a chunk that starts at the block takes all of it");
}
