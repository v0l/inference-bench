use specrt::runtime::Runtime;
use specrt::sched::MediaInput;
use specrt::spec::Spec;
use specrt::vision::{Patches, Processed, best_canvas, canvas_resolutions};
use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures/tiny-llama4-vision")
}

fn spec() -> Spec {
    let mut spec = Spec::from_hf_config_serving(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-llama4-vision".into();
    spec.repo = fixture().display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn floats(dir: &Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(name)).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
fn the_tiny_config_reads_as_a_tiled_tower() {
    let s = spec();
    let v = s.vision.as_ref().expect("the family declares a vision tower");
    assert!(v.config.tiled());
    assert_eq!((v.config.depth, v.config.hidden_size, v.config.num_heads, v.config.patch_size), (2, 64, 4, 14));
    assert_eq!((v.config.side(), v.config.tokens_per_tile(), v.config.out_hidden_size, v.config.num_position_embeddings), (4, 4, 128, 17));
    assert_eq!(v.config.eps(), 1e-5);
    assert_eq!((v.tokens.image, v.tokens.start, v.tokens.end), (592, 590, 591));
    let plan = specrt::vision_encoder::plan(&v.config);
    assert!(plan.contains(&specrt::vision_encoder::VisionOp::Shuffle) && plan.contains(&specrt::vision_encoder::VisionOp::Adapter));
    assert!(!plan.contains(&specrt::vision_encoder::VisionOp::Merger));
}

#[test]
fn canvas_choice_follows_the_processor_rules() {
    let options = canvas_resolutions(4, 336);
    assert!(options.contains(&(672, 672)) && options.contains(&(336, 1344)) && options.contains(&(336, 336)));
    assert_eq!(best_canvas(300, 300, &options, false), (336, 336));
    assert_eq!(best_canvas(336, 700, &options, false), (336, 1008));
    assert_eq!(best_canvas(1000, 1000, &options, false), (672, 672));
    assert_eq!(best_canvas(200, 900, &options, false), (336, 1008));
}

#[test]
fn a_wide_image_expands_to_tiles_a_separator_and_the_global_tile() {
    use specrt::vision::{GridTokens, Media, MediaTokens, expand_placeholders};
    let tokens = MediaTokens {
        image: "<p>".into(),
        video: String::new(),
        start: "<s>".into(),
        end: "<e>".into(),
        placeholder: Some("<|image|>".into()),
        wrap: String::new(),
        tile_tokens: 2,
        grid: Some(GridTokens { tile_x: "<x>".into(), tile_y: "<y>".into(), global: "<g>".into() }),
    };
    let wide = Patches { grid: [3, 4, 4], data: vec![], layout: [1, 2] };
    let single = Patches { grid: [1, 4, 4], data: vec![], layout: [1, 1] };
    let out = expand_placeholders("a<|image|>b", &tokens, 1, &[Media::Image(&wide)]).unwrap();
    assert_eq!(out, "a<s><p><p><x><p><p><y><g><p><p><e>b");
    let out = expand_placeholders("<|image|>", &tokens, 1, &[Media::Image(&single)]).unwrap();
    assert_eq!(out, "<s><g><p><p><e>");
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_llama4_vision.py output in SPECRT_REF_DIR"]
fn tiny_llama4_vision_matches_transformers() {
    let dir = PathBuf::from(std::env::var_os("SPECRT_REF_DIR").expect("SPECRT_REF_DIR holds the output of scripts/tiny_llama4_vision.py"));
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    assert!(rt.has_vision());
    for (case, seed) in [("single", 1u8), ("wide", 2u8)] {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{case}.json"))).unwrap()).unwrap();
        let grid: Vec<usize> = meta["grid"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let layout = if grid[0] > 1 { [1, grid[0] - 1] } else { [1, 1] };
        let data = floats(&dir, &format!("{case}.patches.bin"));
        let mut ids = vec![2u32, 7, 590];
        if layout[1] > 1 {
            for x in 0..layout[1] {
                ids.extend([592u32; 4]);
                ids.push(if x + 1 < layout[1] { 593 } else { 594 });
            }
        }
        ids.push(595);
        ids.extend([592u32; 4]);
        ids.extend([591, 9]);
        let media = vec![MediaInput { processed: Processed::Image(Patches { grid: [grid[0], grid[1], grid[2]], data, layout }), hash: [seed; 32] }];
        let plan = rt.plan_media(&ids, &media).unwrap();
        let want = floats(&dir, &format!("{case}.embeds.bin"));
        assert_eq!(plan.embeds.len(), want.len(), "{case}");
        let num: f64 = plan.embeds.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        eprintln!("{case}: image embeddings relative L2 {rel:e}");
        assert!(rel < 0.01, "{case}: embeddings differ by {rel}");
    }
}
