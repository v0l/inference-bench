use specrt::vision::{TileConfig, preprocess_canvas};
use std::path::{Path, PathBuf};

fn photos() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama4-vision/photos")
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn planar(data: &[f32], tiles: usize, tile: usize, patch: usize) -> Vec<f32> {
    let g = tile / patch;
    let mut out = vec![0f32; data.len()];
    let mut at = 0;
    for t in 0..tiles {
        for by in 0..g {
            for bx in 0..g {
                for c in 0..3 {
                    for py in 0..patch {
                        for px in 0..patch {
                            out[((t * 3 + c) * tile + by * patch + py) * tile + bx * patch + px] = data[at];
                            at += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn photo_tiles_match_the_transformers_image_processor() {
    let dir = photos();
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
    let mut worst = 0f32;
    for (key, case) in cases.as_object().unwrap() {
        let tile = case["tile"].as_u64().unwrap() as usize;
        let cfg = TileConfig::from_json(&format!(
            r#"{{"size":{{"height":{tile},"width":{tile}}},"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"max_patches":{},"resize_to_max_canvas":false}}"#,
            case["max_patches"]
        ))
        .unwrap();
        let bytes = std::fs::read(dir.join(case["image"].as_str().unwrap())).unwrap();
        let got = preprocess_canvas(&cfg, 14, &bytes).unwrap();
        let tiles = case["tiles"].as_u64().unwrap() as usize;
        let layout: Vec<usize> = case["layout"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(got.grid[0], tiles, "{key}: tile count");
        assert_eq!(got.layout.to_vec(), layout, "{key}: layout");
        let want = floats(&dir.join(format!("{key}.bin")));
        let have = planar(&got.data, tiles, tile, 14);
        assert_eq!(have.len(), want.len(), "{key}");
        let max = have.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("{key}: {tiles} tiles, max abs {max:e}");
        worst = worst.max(max);
        assert!(max <= 0.004, "{key}: pixels differ by {max}");
    }
    eprintln!("worst max abs {worst:e}");
}
