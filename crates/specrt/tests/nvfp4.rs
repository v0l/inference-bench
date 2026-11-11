use specrt::spec::Spec;
use specrt::weights::{Checkpoint, Matrix, Mixer, e4m3_to_f32};
use std::path::{Path, PathBuf};

fn ref_dir() -> PathBuf {
    std::env::var_os("SPECRT_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("out"))
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(ref_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap()
}

fn row(m: &Matrix, r: usize, cols: usize) -> Vec<f32> {
    match m {
        Matrix::Fp4(f) => f.dequant_row(r),
        Matrix::Fp8 { values, scales } => {
            let base = scales[r].to_bits() as usize;
            (0..cols).map(|c| e4m3_to_f32(values[r * cols + c]) * scales[base + c / 128]).collect()
        }
        _ => panic!("expected a quantised matrix"),
    }
}

#[test]
#[ignore = "needs the orcarouter/Qwen3.8-27B-Uncensored-NVFP4 checkpoint and scripts/reference.py nvfp4-dequant output"]
fn nvfp4_and_fp8_channel_rows_dequantise_bit_for_bit_like_compressed_tensors() {
    let spec = spec();
    let ck = Checkpoint::for_repo(&spec.repo).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_dir().join("nvfp4-dequant.json")).unwrap()).unwrap();
    let theirs = floats("nvfp4-dequant.f32.bin");
    let la = spec.linear_attention.clone().unwrap();
    let mut at = 0;
    let mut checked = 0;
    for e in meta["entries"].as_array().unwrap() {
        let layer = e["layer"].as_u64().unwrap() as usize;
        let proj = e["proj"].as_str().unwrap().trim_start_matches("mlp.");
        let cols = e["cols"].as_u64().unwrap() as usize;
        let l = ck.layer(&spec, layer).unwrap();
        for r in e["rows"].as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize) {
            let ours = match proj {
                "gate_proj" => row(&l.gate_up, 2 * r, cols),
                "up_proj" => row(&l.gate_up, 2 * r + 1, cols),
                "down_proj" => row(&l.down, r, cols),
                "linear_attn.in_proj_qkv" | "linear_attn.in_proj_z" => {
                    let Mixer::Linear(g) = &l.mixer else { panic!("layer {layer} is not linear") };
                    let off = if proj.ends_with("_z") { la.conv_dim() } else { 0 };
                    row(&g.in_proj, off + r, cols)
                }
                "self_attn.o_proj" => {
                    let Mixer::Attention(a) = &l.mixer else { panic!("layer {layer} is not attention") };
                    row(&a.o, r, cols)
                }
                other => panic!("{other}"),
            };
            let reference = &theirs[at..at + cols];
            let differ = ours.iter().zip(reference).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differ, 0, "layer {layer} {proj} row {r}: {differ} of {cols} values differ");
            at += cols;
            checked += cols;
        }
        drop(l);
        ck.release();
    }
    assert_eq!(at, theirs.len());
    eprintln!("{checked} values in {} tensors identical to compressed-tensors", meta["entries"].as_array().unwrap().len());
}
