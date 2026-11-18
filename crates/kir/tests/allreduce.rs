use kir::Target;
use kir::allreduce::{AllReduceKernel, Dtype, Mode};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/allreduce").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn emit(t: &Target, ranks: u32, dtype: Dtype, mode: Mode) -> String {
    let k = AllReduceKernel::new(t, ranks, dtype, mode, 16).unwrap();
    format!("// {}\n{}", k.describe().replace('\n', "\n// "), kir::cuda::emit_allreduce(&[k]).unwrap())
}

#[test]
fn allreduce_goldens_per_mode_dtype_and_target() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        for dtype in [Dtype::F32, Dtype::Bf16] {
            for mode in [Mode::OneShot, Mode::TwoShot] {
                golden(&format!("{}_{}_{tag}.cu", mode.tag(), dtype.tag()), &emit(&t, 2, dtype, mode));
            }
        }
        golden(&format!("add_rmsnorm_f32_{tag}.cu"), &emit(&t, 2, Dtype::F32, Mode::AddRmsNorm { width: 5120, eps: 1e-6 }));
    }
}

#[test]
fn source_has_no_template_slots_and_names_its_rank_count() {
    let src = emit(&Target::sm_120(), 4, Dtype::Bf16, Mode::OneShot);
    assert!(!src.contains("{{") && !src.contains("}}"));
    assert!(src.contains("constexpr int AR_RANKS = 4;"));
}

#[test]
fn validate_refuses_bad_rank_counts_targets_and_widths() {
    let t = Target::sm_120();
    assert!(AllReduceKernel::new(&t, 1, Dtype::F32, Mode::OneShot, 8).is_err());
    assert!(AllReduceKernel::new(&t, 9, Dtype::F32, Mode::OneShot, 8).is_err());
    assert!(AllReduceKernel::new(&t, 2, Dtype::F32, Mode::OneShot, 0).is_err());
    assert!(AllReduceKernel::new(&Target::wgpu_baseline(), 2, Dtype::F32, Mode::OneShot, 8).unwrap_err().0.contains("lowering"));
    assert!(AllReduceKernel::new(&t, 2, Dtype::F32, Mode::AddRmsNorm { width: 0, eps: 1e-6 }, 8).is_err());
    assert!(AllReduceKernel::new(&t, 2, Dtype::F32, Mode::AddRmsNorm { width: 130, eps: 1e-6 }, 8).is_err());
    assert!(AllReduceKernel::new(&t, 2, Dtype::Bf16, Mode::AddRmsNorm { width: 128, eps: 1e-6 }, 8).unwrap_err().0.contains("f32"));
}

#[test]
fn hash_changes_with_ranks_mode_and_dtype() {
    let t = Target::sm_120();
    let h = |r, d, m| AllReduceKernel::new(&t, r, d, m, 8).unwrap().hash();
    let base = h(2, Dtype::F32, Mode::OneShot);
    assert_ne!(base, h(4, Dtype::F32, Mode::OneShot));
    assert_ne!(base, h(2, Dtype::Bf16, Mode::OneShot));
    assert_ne!(base, h(2, Dtype::F32, Mode::TwoShot));
    assert_eq!(base, h(2, Dtype::F32, Mode::OneShot));
}
