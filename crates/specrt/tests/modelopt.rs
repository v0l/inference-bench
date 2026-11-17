use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use specrt::spec::Spec;
use specrt::weights::{Checkpoint, Matrix, Mixer};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-27b")
}

struct Owned {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn modelopt_name(name: &str) -> String {
    if let Some(m) = name.strip_suffix(".weight_packed") {
        format!("{m}.weight")
    } else if let Some(m) = name.strip_suffix(".weight_global_scale") {
        format!("{m}.weight_scale_2")
    } else if let Some(m) = name.strip_suffix(".input_global_scale") {
        format!("{m}.input_scale")
    } else {
        name.to_string()
    }
}

fn f32_bytes(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn convert(out: &Path) {
    std::fs::create_dir_all(out).unwrap();
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("model.safetensors.index.json")).unwrap()).unwrap();
    let mut map = serde_json::Map::new();
    let files: std::collections::BTreeSet<&str> = index["weight_map"].as_object().unwrap().values().map(|v| v.as_str().unwrap()).collect();
    for file in files {
        let bytes = std::fs::read(fixture().join(file)).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut tensors: BTreeMap<String, Owned> = BTreeMap::new();
        for (name, v) in st.tensors() {
            let new = modelopt_name(&name);
            let owned = if name.ends_with(".weight_scale") && v.shape().len() == 2 && v.shape()[1] == 1 {
                let first = half_or_f32(&v);
                Owned { dtype: Dtype::F32, shape: vec![], data: f32_bytes(first) }
            } else if name.ends_with(".weight_global_scale") {
                let g = f32::from_le_bytes(v.data()[..4].try_into().unwrap());
                Owned { dtype: Dtype::F32, shape: vec![], data: f32_bytes(1.0 / g) }
            } else if name.ends_with(".input_global_scale") {
                let g = f32::from_le_bytes(v.data()[..4].try_into().unwrap());
                Owned { dtype: Dtype::F32, shape: vec![], data: f32_bytes(1.0 / g) }
            } else {
                Owned { dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() }
            };
            map.insert(new.clone(), file.into());
            tensors.insert(new, owned);
        }
        let views: Vec<(String, TensorView)> = tensors.iter().map(|(k, o)| (k.clone(), TensorView::new(o.dtype, o.shape.clone(), &o.data).unwrap())).collect();
        safetensors::serialize_to_file(views, None, &out.join(file)).unwrap();
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(out.join("model.safetensors.index.json"), index.to_string()).unwrap();
}

fn half_or_f32(v: &safetensors::tensor::TensorView) -> f32 {
    match v.dtype() {
        Dtype::BF16 => f32::from_bits((u16::from_le_bytes([v.data()[0], v.data()[1]]) as u32) << 16),
        Dtype::F32 => f32::from_le_bytes(v.data()[..4].try_into().unwrap()),
        other => panic!("{other:?}"),
    }
}

fn fp8_parts(m: &Matrix) -> (Vec<u8>, Vec<f32>) {
    match m {
        Matrix::Fp8 { values, scales } => (values.to_vec(), scales.clone()),
        _ => panic!("expected fp8"),
    }
}

#[test]
fn modelopt_tensor_names_load_as_the_compressed_tensors_matrices() {
    let out = std::env::temp_dir().join(format!("specrt-modelopt-{}", std::process::id()));
    convert(&out);
    let spec = {
        let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
        Spec::from_hf_config(&json).unwrap()
    };
    let theirs = Checkpoint::open(&fixture()).unwrap();
    let ours = Checkpoint::open(&out).unwrap();
    assert!(ours.has("model.language_model.layers.0.mlp.gate_proj.weight_scale_2"));
    for i in 0..spec.layers.count {
        let (a, b) = (theirs.layer(&spec, i).unwrap(), ours.layer(&spec, i).unwrap());
        match (&a.gate_up, &b.gate_up) {
            (Matrix::Fp4(x), Matrix::Fp4(y)) => {
                assert_eq!(x.packed, y.packed, "layer {i} gate_up packed");
                assert_eq!(x.scales, y.scales);
                assert!((x.global / y.global - 1.0).abs() < 1e-6);
                assert!((x.input_global / y.input_global - 1.0).abs() < 1e-6);
            }
            _ => {}
        }
        if let (Mixer::Attention(x), Mixer::Attention(y)) = (&a.mixer, &b.mixer) {
            let (xv, xs) = fp8_parts(&x.qkv);
            let (yv, ys) = fp8_parts(&y.qkv);
            assert_eq!(xv, yv, "layer {i} qkv values");
            assert_eq!(xs.len(), ys.len());
            let rows = xv.len() / spec.dims.hidden;
            let grid = &ys[rows..];
            let mut distinct: Vec<u32> = grid.iter().map(|s| s.to_bits()).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(distinct.len() <= 3, "layer {i}: q, k and v each carry one per-tensor scale, got {}", distinct.len());
        }
    }
    std::fs::remove_dir_all(&out).ok();
}
