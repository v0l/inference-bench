use safetensors::SafeTensors;
use specrt::dequant::{BlockFormat, Elem, MXFP4, MXFP8, ScaleKind};
use std::path::Path;

struct Reference {
    rows: usize,
    cols: usize,
    weight: Vec<u8>,
    scales: Vec<u8>,
    expected: Vec<u16>,
}

fn reference() -> Reference {
    let bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mxfp8/reference.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let w = st.tensor("weight").unwrap();
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    let expected = st.tensor("expected").unwrap().data().chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    Reference { rows, cols, weight: w.data().to_vec(), scales: st.tensor("weight_scale").unwrap().data().to_vec(), expected }
}

fn mismatches(got: &[u16], want: &[u16]) -> usize {
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

#[test]
fn mxfp8_decodes_like_the_numpy_reference_and_an_exponent_off_by_one_does_not() {
    let r = reference();
    assert_eq!(MXFP8, BlockFormat { elem: Elem::E4m3, scale: ScaleKind::E8m0, group: 32 });
    assert_eq!(r.scales.len(), r.rows * r.cols / 32);
    let got = MXFP8.decode_bf16(&r.weight, &r.scales).unwrap();
    assert_eq!(mismatches(&got, &r.expected), 0);
    for delta in [1i16, -1] {
        let shifted: Vec<u8> = r.scales.iter().map(|&s| (s as i16 + delta) as u8).collect();
        let off = MXFP8.decode_bf16(&r.weight, &shifted).unwrap();
        assert!(mismatches(&off, &r.expected) > r.expected.len() / 2, "an exponent off by {delta} must not match");
    }
    assert_ne!(MXFP8.decode_bf16(&r.weight, &r.scales).unwrap(), MXFP4.decode_bf16(&r.weight[..r.weight.len() / 2], &r.scales).unwrap());
}

#[test]
fn e8m0_zero_is_two_to_the_minus_127_and_ff_is_nan() {
    let mut out = Vec::new();
    MXFP8.decode(&[0x38; 32], &[0], &mut out).unwrap();
    assert_eq!(out[0], f32::from_bits(0x0040_0000));
    out.clear();
    MXFP8.decode(&[0x38; 32], &[255], &mut out).unwrap();
    assert!(out[0].is_nan());
}

#[test]
fn the_dequant_block_comes_from_kir_and_follows_the_format() {
    let t = kir::Target::sm_120();
    let k = MXFP8.kernel(&t).unwrap();
    let src = kir::cuda::emit_block_dequant(&k).unwrap();
    assert!(src.contains("block_dequant_bf16") && src.contains("e4m3_f32(b) * sc") && !src.contains("e2m1_f32"));
    assert_eq!(k.group_bytes(), 32);
    let fp4 = MXFP4.kernel(&t).unwrap();
    assert_eq!(fp4.group_bytes(), 16);
    assert_ne!(k.hash(), fp4.hash());
    assert!(MXFP8.kernel(&kir::Target::wgpu_baseline()).is_err());
    assert!(BlockFormat { elem: Elem::Int8, scale: ScaleKind::E8m0, group: 32 }.kernel(&t).is_err());
}

fn tiny_checkpoint(tag: &str, tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("specrt-mxfp8-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let views: Vec<(String, safetensors::tensor::TensorView)> =
        tensors.iter().map(|(n, d, s, b)| (n.clone(), safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap())).collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn an_mxfp8_lm_head_loads_as_the_reference_bf16_rows() {
    use safetensors::Dtype;
    let r = reference();
    let json = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap();
    let mut spec = specrt::spec::Spec::from_hf_config(&json).unwrap();
    spec.dims.vocab = r.rows;
    spec.dims.hidden = r.cols;
    spec.dims.tie_embeddings = false;
    let dir = tiny_checkpoint(
        "head",
        vec![
            ("model.embed_tokens.weight".into(), Dtype::BF16, vec![r.rows, r.cols], vec![0; r.rows * r.cols * 2]),
            ("model.norm.weight".into(), Dtype::F32, vec![r.cols], vec![0; r.cols * 4]),
            ("lm_head.weight".into(), Dtype::F8_E4M3, vec![r.rows, r.cols], r.weight.clone()),
            ("lm_head.weight_scale".into(), Dtype::U8, vec![r.rows, r.cols / 32], r.scales.clone()),
        ],
    );
    let ck = specrt::weights::Checkpoint::open(&dir).unwrap();
    let (_, head) = ck.head(&spec).unwrap();
    match head {
        specrt::weights::Matrix::Bf16(w) => assert_eq!(mismatches(&w, &r.expected), 0),
        _ => panic!("a bf16 spec keeps the MXFP8 head as bf16"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn gpu_decode(format: BlockFormat, packed: &[u8], scales: &[u8], global: f32) -> Vec<u16> {
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let dir = std::env::temp_dir().join(format!("specrt-mxfp8-{}", std::process::id()));
    let out = format.decode_on_device(&stream, &dir, packed, scales, global).unwrap();
    let host = stream.clone_dtoh(&out).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    host
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_kir_mxfp8_block_matches_the_numpy_reference_on_the_gpu() {
    let r = reference();
    let got = gpu_decode(MXFP8, &r.weight, &r.scales, 1.0);
    assert_eq!(mismatches(&got, &r.expected), 0);
    let shifted: Vec<u8> = r.scales.iter().map(|&s| s + 1).collect();
    let off = gpu_decode(MXFP8, &r.weight, &shifted, 1.0);
    assert!(mismatches(&off, &r.expected) > r.expected.len() / 2);
    let fp4 = gpu_decode(MXFP4, &r.weight[..r.weight.len() / 2], &r.scales, 1.0);
    assert_eq!(fp4, MXFP4.decode_bf16(&r.weight[..r.weight.len() / 2], &r.scales).unwrap());
}
