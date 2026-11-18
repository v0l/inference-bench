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

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Modelopt,
    ScalarCompressed,
}

fn convert(out: &Path, format: Format) {
    std::fs::create_dir_all(out).unwrap();
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("model.safetensors.index.json")).unwrap()).unwrap();
    let mut map = serde_json::Map::new();
    let files: std::collections::BTreeSet<&str> = index["weight_map"].as_object().unwrap().values().map(|v| v.as_str().unwrap()).collect();
    for file in files {
        let bytes = std::fs::read(fixture().join(file)).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut tensors: BTreeMap<String, Owned> = BTreeMap::new();
        for (name, v) in st.tensors() {
            let modelopt = format == Format::Modelopt;
            let new = if modelopt { modelopt_name(&name) } else { name.clone() };
            let owned = if name.ends_with(".weight_scale") && v.shape().len() == 2 && v.shape()[1] == 1 {
                let first = half_or_f32(&v);
                if modelopt {
                    Owned { dtype: Dtype::F32, shape: vec![], data: f32_bytes(first) }
                } else {
                    let bf16 = (first.to_bits() >> 16) as u16;
                    Owned { dtype: Dtype::BF16, shape: v.shape().to_vec(), data: (0..v.shape()[0]).flat_map(|_| bf16.to_le_bytes()).collect() }
                }
            } else if !modelopt {
                Owned { dtype: v.dtype(), shape: v.shape().to_vec(), data: v.data().to_vec() }
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
    convert(&out, Format::Modelopt);
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

fn scratch(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("specrt-modelopt-{tag}-{}", std::process::id()))
}

fn spec_at(dir: &Path, serving: bool) -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut s = if serving { Spec::from_hf_config_serving(&json).unwrap() } else { Spec::from_hf_config(&json).unwrap() };
    s.name = "tiny-qwen3.5-27b".into();
    s.repo = dir.display().to_string();
    s.runtime.vision.precision = specrt::vision_encoder::VisionPrecision::Exact;
    s.mtp.as_mut().unwrap().head_quant = Some(specrt::spec::Quant::Bf16);
    s
}

fn logits_of(dir: &Path, serving: bool) -> (Vec<f32>, Spec) {
    let s = spec_at(dir, serving);
    let seq: Vec<u32> = (0..48).map(|i| (i * 37 + 11) % 1000).collect();
    let mut rt = specrt::runtime::Runtime::load(&s, 256).unwrap();
    (rt.prompt_logits(&seq).unwrap(), s)
}

fn assert_same_logits(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale;
    eprintln!("{what}: max relative difference {err:e}");
    assert!(err <= 1e-5, "{what}: {err}");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn modelopt_fixture_gives_the_compressed_tensors_logits_weight_only() {
    let (modelopt, ct) = (scratch("w-mo"), scratch("w-ct"));
    convert(&modelopt, Format::Modelopt);
    convert(&ct, Format::ScalarCompressed);
    let (a, _) = logits_of(&ct, false);
    let (b, _) = logits_of(&modelopt, false);
    assert_same_logits(&a, &b, "weight-only");
    std::fs::remove_dir_all(&modelopt).ok();
    std::fs::remove_dir_all(&ct).ok();
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn modelopt_input_scale_gives_the_compressed_tensors_logits_with_fp4_activations() {
    let (modelopt, ct) = (scratch("a-mo"), scratch("a-ct"));
    convert(&modelopt, Format::Modelopt);
    convert(&ct, Format::ScalarCompressed);
    let (a, s) = logits_of(&ct, true);
    assert_eq!(s.mlp.act_quant, specrt::spec::ActQuant::Fp4);
    let (b, _) = logits_of(&modelopt, true);
    assert_same_logits(&a, &b, "fp4 activations");
    std::fs::remove_dir_all(&modelopt).ok();
    std::fs::remove_dir_all(&ct).ok();
}

fn weight_only_checkpoint(tag: &str, tensors: &[&str]) -> PathBuf {
    let dir = scratch(tag);
    std::fs::create_dir_all(&dir).unwrap();
    let data = 1f32.to_le_bytes();
    let views: Vec<(String, TensorView)> = tensors.iter().map(|n| (n.to_string(), TensorView::new(Dtype::F32, vec![], &data).unwrap())).collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    dir
}

fn darkstar_spec() -> Spec {
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/fixtures/hf/darkstar-config.json")).unwrap();
    Spec::from_hf_config_serving(&json).unwrap()
}

#[test]
fn mlp_fp4_activations_are_refused_on_a_checkpoint_without_input_scales() {
    let dir = weight_only_checkpoint("weight-only-mlp", &["model.language_model.embed_tokens.weight", "model.language_model.layers.0.mlp.gate_proj.weight_scale_2"]);
    let mut spec = darkstar_spec();
    spec.mlp.act_quant = specrt::spec::ActQuant::Fp4;
    spec.validate().unwrap();
    let err = Checkpoint::open(&dir).unwrap().layer(&spec, 0).err().expect("must refuse").to_string();
    assert!(err.contains("layer 0") && err.contains("mlp.gate_proj has no input_scale") && err.contains("weight-only"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn darkstar_with_both_activation_quants_forced_on_is_refused_at_load() {
    let dir = weight_only_checkpoint("weight-only-both", &["model.language_model.embed_tokens.weight"]);
    let mut spec = darkstar_spec();
    spec.act_quant = specrt::spec::ActQuant::Fp8;
    spec.mlp.act_quant = specrt::spec::ActQuant::Fp4;
    let err = Checkpoint::open(&dir).unwrap().layer(&spec, 0).err().expect("must refuse").to_string();
    assert!(err.contains("layer 0") && err.contains("weight-only"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn darkstar_as_derived_from_its_config_runs_without_activation_quantisation() {
    let spec = darkstar_spec();
    assert_eq!((spec.act_quant, spec.mlp.act_quant), (specrt::spec::ActQuant::None, specrt::spec::ActQuant::None));
}
