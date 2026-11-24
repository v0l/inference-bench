use safetensors::SafeTensors;
use specrt::csf::CsfPlane;
use specrt::weights::Checkpoint;
use std::path::Path;

struct Owned {
    rows: usize,
    columns: usize,
    bases: Vec<u8>,
    codes: Vec<u8>,
    page_bases: Vec<u32>,
    offsets: Vec<u16>,
    exceptions: Vec<u16>,
    expected: Vec<u8>,
}

impl Owned {
    fn plane(&self) -> CsfPlane<'_> {
        CsfPlane { rows: self.rows, columns: self.columns, bases: &self.bases, codes: &self.codes, page_bases: &self.page_bases, offsets: &self.offsets, exceptions: &self.exceptions }
    }
}

fn fixture(name: &str) -> Owned {
    let bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/csf").join(name)).unwrap();
    let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
    let meta = meta.metadata().clone().unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let raw = |n: &str| st.tensor(n).unwrap().data().to_vec();
    let u16s = |n: &str| raw(n).chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Owned {
        rows: meta["rows"].parse().unwrap(),
        columns: meta["columns"].parse().unwrap(),
        bases: raw("bases"),
        codes: raw("codes"),
        page_bases: raw("page_bases").chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        offsets: u16s("offsets"),
        exceptions: u16s("exceptions"),
        expected: raw("expected"),
    }
}

#[test]
fn csf_slices_of_a_real_expert_scale_decode_like_vllm() {
    for name in ["slice_head.safetensors", "slice_tail.safetensors"] {
        let f = fixture(name);
        assert!(!f.exceptions.is_empty(), "{name} exercises exception records");
        assert_eq!(f.plane().decode().unwrap(), f.expected, "{name}");
    }
}

#[test]
fn a_dropped_exception_or_a_wrong_base_changes_or_refuses_the_decode() {
    let mut f = fixture("slice_tail.safetensors");
    f.bases[3] = f.bases[3].wrapping_add(1);
    assert_ne!(f.plane().decode().unwrap(), f.expected);
    let mut f = fixture("slice_tail.safetensors");
    f.exceptions.pop();
    let err = f.plane().decode().unwrap_err().to_string();
    assert!(err.contains("exception"), "{err}");
    let mut f = fixture("slice_head.safetensors");
    f.page_bases[0] = 2;
    assert!(f.plane().decode().is_err());
    let f = fixture("slice_head.safetensors");
    assert!(CsfPlane { rows: 16, ..f.plane() }.decode().is_err());
}

#[test]
#[ignore = "needs the GLM-5.3 Flash checkpoint on disk; reads one tensor"]
fn a_real_csf_expert_scale_decodes_to_the_vllm_hash_and_loads_as_nvfp4() {
    use sha2::Digest;
    let ck = Checkpoint::for_repo("Piecrust/GLM-5.3-Flash-FP4-Abliterated").unwrap();
    let module = "model.language_model.layers.3.mlp.experts.0.down_proj";
    let scales = ck.csf_scales(&format!("{module}.weight_scale"), 4096, 128).unwrap();
    let hash = sha2::Sha256::digest(&scales);
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, "788ca32403a7a546a766f73beef4df478f0681a9173cd35e92028b237dde244c", "vLLM's logical_csf_components gives this");
    let m = ck.fp4(module, 4096, 2048).unwrap();
    assert_eq!(&m.scales[..], &scales[..]);
    assert!(m.global > 0.0 && m.input_global > 0.0);
    assert!(m.dequant_row(0).iter().all(|v| v.is_finite()));
}
