use kir::linear::{Family, LinearKernel, LinearShape};
use kir::{Feature, Target};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/linear").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn shape() -> LinearShape {
    LinearShape { key_heads: 16, value_heads: 48, head_dim: 128, conv_kernel: 4, tiled_value_heads: false, gate_sigmoid: false }
}

fn emit(t: &Target, s: LinearShape, fam: &[Family]) -> String {
    let ks: Vec<_> = fam.iter().map(|&f| LinearKernel::new(t, f, s.clone()).unwrap()).collect();
    let header: String = ks.iter().map(|k| format!("// {}\n", k.describe().replace('\n', "\n// "))).collect();
    format!("{header}{}", kir::cuda::emit_linear(&ks).unwrap())
}

#[test]
fn linear_goldens_per_family_and_target() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        for f in Family::ALL {
            golden(&format!("{}_{tag}.cu", f.block()), &emit(&t, shape(), &[f]));
        }
        golden(&format!("all_{tag}.cu"), &emit(&t, shape(), &Family::ALL));
    }
}

#[test]
fn source_has_no_template_slots() {
    let src = emit(&Target::sm_120(), shape(), &Family::ALL);
    assert!(!src.contains("{{"));
    assert!(src.contains("constexpr int LA_CHUNK = 64;"));
}

#[test]
fn validate_refuses_bad_shapes_and_targets() {
    let t = Target::sm_120();
    let mut s = shape();
    s.head_dim = 136;
    assert!(LinearKernel::new(&t, Family::Step, s.clone()).is_err());
    s = shape();
    s.value_heads = 40;
    assert!(LinearKernel::new(&t, Family::Step, s).unwrap_err().0.contains("multiple of key heads"));
    assert!(LinearKernel::new(&Target::wgpu_baseline(), Family::Step, shape()).unwrap_err().0.contains("lowering"));
    let no_mma = Target::sm_80().without_feature(Feature::Mma16816);
    assert!(LinearKernel::new(&no_mma, Family::Chunk, shape()).unwrap_err().0.contains("Mma16816"));
    let mut small = t;
    small.smem_limit = 8 * 1024;
    assert!(LinearKernel::new(&small, Family::Chunk, shape()).unwrap_err().0.contains("shared memory"));
}

#[test]
fn hash_changes_with_shape_and_target() {
    let a = LinearKernel::new(&Target::sm_120(), Family::Step, shape()).unwrap();
    let mut s = shape();
    s.conv_kernel = 5;
    let b = LinearKernel::new(&Target::sm_120(), Family::Step, s).unwrap();
    let c = LinearKernel::new(&Target::sm_80(), Family::Step, shape()).unwrap();
    assert_ne!(a.hash(), b.hash());
    assert_ne!(a.hash(), c.hash());
}

#[test]
fn a_sigmoid_output_gate_replaces_silu_in_every_gated_norm() {
    let mut s = shape();
    s.gate_sigmoid = true;
    let kernels: Vec<LinearKernel> = [Family::Step, Family::Chunk, Family::Verify].into_iter().map(|f| LinearKernel::new(&Target::sm_120(), f, s.clone()).unwrap()).collect();
    let src = kir::cuda::emit_linear(&kernels).unwrap().into_string();
    assert_eq!(src.matches("* la_sigmoid(").count(), 4, "{src}");
    assert!(!src.contains("* silu(z[") && !src.contains("* silu(lin[") && !src.contains("* silu(l["));
    let plain: Vec<LinearKernel> = [Family::Step, Family::Chunk, Family::Verify].into_iter().map(|f| LinearKernel::new(&Target::sm_120(), f, shape()).unwrap()).collect();
    assert!(!kir::cuda::emit_linear(&plain).unwrap().into_string().contains("la_sigmoid"));
    assert_ne!(kernels[0].hash(), plain[0].hash());
}
