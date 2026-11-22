use kir::attention::{AttentionKernel, AttentionShape, Family, MropeShape, RopeShape};
use kir::{Feature, Target};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/attention").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn rope(mrope: bool) -> RopeShape {
    RopeShape {
        rotary_dim: 64,
        theta: 10000000.0,
        kind: 0,
        factor: 1.0,
        low: 1.0,
        high: 1.0,
        original: 1.0,
        mscale: 1.0,
        freq_factors: vec![],
        mrope: mrope.then_some(MropeShape { section: [11, 11, 10], interleaved: true }),
        interleaved: false,
    }
}

fn shape(planes: usize, fp8: bool, mrope: bool) -> AttentionShape {
    AttentionShape {
        heads: 24,
        kv_heads: 4,
        head_dim: 256,
        qkv_row: 14336,
        output_gate: true,
        qk_norm: true,
        qk_norm_unit: false,
        eps: 1e-6,
        rope: rope(mrope),
        kv_layers: 16,
        kv_planes: planes,
        kv_fp8: fp8,
        kv_f16_layers: if fp8 { 8 } else { 0 },
        query_scale: 1.0 / 16.0,
        softcap: if mrope { 30.0 } else { 0.0 },
        local_theta: 10000.0,
        local_rope: vec![false, true, false, false],
        bidirectional: false,
        nope: vec![false, false, true, false],
    }
}

fn emit(t: &Target, s: AttentionShape, fam: &[Family]) -> String {
    let ks: Vec<_> = fam.iter().map(|&f| AttentionKernel::new(t, f, s.clone()).unwrap()).collect();
    let header: String = ks.iter().map(|k| format!("// {}\n", k.describe().replace('\n', "\n// "))).collect();
    format!("{header}{}", kir::cuda::emit_attention(&ks).unwrap())
}

#[test]
fn attention_goldens_per_family_and_variant() {
    let t = Target::sm_120();
    for (tag, s) in [("bf16x2", shape(2, false, false)), ("bf16x1_mrope", shape(1, false, true)), ("fp8", shape(1, true, false))] {
        for f in Family::ALL {
            golden(&format!("{}_{tag}_sm120.cu", f.block()), &emit(&t, s.clone(), &[f]));
        }
    }
    golden("all_sm80.cu", &emit(&Target::sm_80(), shape(1, false, false), &Family::ALL));
}

#[test]
fn interleaved_rope_and_unit_qk_norm_are_shape_data() {
    let t = Target::sm_120();
    let mut s = shape(1, false, false);
    let plain = emit(&t, s.clone(), &[Family::RopeKv, Family::FlashMma]);
    assert!(plain.contains("constexpr bool ROPE_PAIRS = false;") && plain.contains("constexpr bool QK_NORM_UNIT = false;"));
    s.rope.interleaved = true;
    s.qk_norm_unit = true;
    let paired = emit(&t, s.clone(), &[Family::RopeKv, Family::FlashMma]);
    assert!(paired.contains("constexpr bool ROPE_PAIRS = true;") && paired.contains("constexpr bool QK_NORM_UNIT = true;"));
    let hash = |s: AttentionShape| AttentionKernel::new(&t, Family::FlashMma, s).unwrap().hash();
    assert_ne!(hash(shape(1, false, false)), hash(s));
}

#[test]
fn source_has_no_context_constants() {
    let src = emit(&Target::sm_120(), shape(1, false, false), &Family::ALL);
    for banned in ["MAX_CTX", "PAGES", "BD_SPLITS", "{{"] {
        assert!(!src.contains(banned), "{banned} left in the source");
    }
}

#[test]
fn validate_refuses_bad_shapes_and_targets() {
    let t = Target::sm_120();
    let mut s = shape(2, true, false);
    assert!(AttentionKernel::new(&t, Family::FlashMma, s.clone()).unwrap_err().0.contains("fp8 KV needs a single plane"));
    s = shape(1, false, false);
    s.heads = 25;
    assert!(AttentionKernel::new(&t, Family::RopeKv, s.clone()).is_err());
    let mut small = t;
    small.smem_limit = 8 * 1024;
    assert!(AttentionKernel::new(&small, Family::FlashMma, shape(1, false, false)).unwrap_err().0.contains("shared memory"));
    let wgpu = Target::wgpu_baseline();
    assert!(AttentionKernel::new(&wgpu, Family::RopeKv, shape(1, false, false)).unwrap_err().0.contains("lowering"));
    let mut no_mma = Target::sm_80();
    no_mma = no_mma.without_feature(Feature::Mma16816);
    assert!(AttentionKernel::new(&no_mma, Family::GroupedDecode, shape(1, false, false)).unwrap_err().0.contains("Mma16816"));
}

#[test]
fn hash_changes_with_shape_and_target() {
    let a = AttentionKernel::new(&Target::sm_120(), Family::FlashMma, shape(1, false, false)).unwrap();
    let b = AttentionKernel::new(&Target::sm_120(), Family::FlashMma, shape(2, false, false)).unwrap();
    let c = AttentionKernel::new(&Target::sm_80(), Family::FlashMma, shape(1, false, false)).unwrap();
    assert_ne!(a.hash(), b.hash());
    assert_ne!(a.hash(), c.hash());
    assert_eq!(a.hash(), AttentionKernel::new(&Target::sm_120(), Family::FlashMma, shape(1, false, false)).unwrap().hash());
}
