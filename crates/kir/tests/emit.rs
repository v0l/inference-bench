use kir::{Activation, BlockSum, Feature, Kernel, Language, Target};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn subgroups() -> Target {
    Target::wgpu_baseline().with_subgroups(32)
}

fn norm(t: &Target) -> Kernel {
    Kernel::rmsnorm(t, 1024, 1e-6, 1024).unwrap()
}

fn act(t: &Target, a: Activation) -> Kernel {
    Kernel::gated_activation(t, "swiglu", 3072, a, 256).unwrap()
}

#[test]
fn cuda_rmsnorm_golden() {
    golden("rmsnorm_sm120.cu", &kir::cuda::emit_module(&[norm(&Target::sm_120())]).unwrap());
    golden("rmsnorm_sm80.cu", &kir::cuda::emit_module(&[norm(&Target::sm_80())]).unwrap());
}

#[test]
fn cuda_activation_golden() {
    let t = Target::sm_120();
    golden("swiglu_silu.cu", &kir::cuda::emit_module(&[act(&t, Activation::Silu)]).unwrap());
    golden("swiglu_gelu.cu", &kir::cuda::emit_module(&[act(&t, Activation::GeluTanh)]).unwrap());
}

#[test]
fn wgsl_rmsnorm_golden_per_target() {
    golden("rmsnorm_tree.wgsl", &kir::wgsl::emit_kernel(&norm(&Target::wgpu_baseline())).unwrap());
    golden("rmsnorm_subgroup.wgsl", &kir::wgsl::emit_kernel(&norm(&subgroups())).unwrap());
}

#[test]
fn wgsl_activation_golden() {
    let t = Target::wgpu_baseline();
    golden("swiglu_silu.wgsl", &kir::wgsl::emit_kernel(&act(&t, Activation::Silu)).unwrap());
    golden("swiglu_gelu.wgsl", &kir::wgsl::emit_kernel(&act(&t, Activation::GeluTanh)).unwrap());
}

#[test]
fn block_sum_follows_the_target() {
    assert_eq!(BlockSum::for_target(&Target::sm_120()), BlockSum::WarpShuffle);
    assert_eq!(BlockSum::for_target(&Target::wgpu_baseline()), BlockSum::SharedTree);
    assert_eq!(BlockSum::for_target(&subgroups()), BlockSum::Subgroup);
}

#[test]
fn threads_are_clamped_to_the_target() {
    let t = Target::wgpu(128, 16 * 1024);
    assert_eq!(norm(&t).threads, 128);
    assert_eq!(act(&t, Activation::Silu).threads, 128);
    assert_eq!(norm(&Target::sm_80()).threads, 1024);
}

#[test]
fn wgsl_bindings_end_with_the_uniform() {
    use kir::wgsl::Binding::*;
    let t = Target::wgpu_baseline();
    assert_eq!(kir::wgsl::bindings(&norm(&t)), vec![Read, Read, Write, Uniform]);
    assert_eq!(kir::wgsl::bindings(&act(&t, Activation::Silu)), vec![Read, Write, Uniform]);
}

#[test]
fn emitters_refuse_the_wrong_language() {
    assert!(kir::wgsl::emit_kernel(&norm(&Target::sm_120())).is_err());
    assert!(kir::cuda::emit_module(&[norm(&Target::wgpu_baseline())]).is_err());
}

#[test]
fn a_block_the_target_cannot_run_is_refused() {
    let mut k = norm(&Target::wgpu_baseline());
    k.body = match k.body {
        kir::Body::RmsNorm { width, eps, offset, .. } => kir::Body::RmsNorm { width, eps, offset, sum: BlockSum::Subgroup },
        b => b,
    };
    let err = k.validate().unwrap_err();
    assert!(err.contains("Subgroups"), "{err}");
    assert!(!Target::wgpu_baseline().has(Feature::Subgroups));
    let mut cuda = norm(&Target::sm_120());
    cuda.body = match cuda.body {
        kir::Body::RmsNorm { width, eps, offset, .. } => kir::Body::RmsNorm { width, eps, offset, sum: BlockSum::SharedTree },
        b => b,
    };
    assert!(cuda.validate().unwrap_err().contains("cannot run"));
}

#[test]
fn shared_memory_over_the_target_limit_is_rejected() {
    let mut k = norm(&Target::sm_120());
    k.smem_bytes = Target::sm_120().smem_limit + 1;
    assert!(k.validate().unwrap_err().contains("shared memory"));
    let tiny = Target::wgpu(1024, 1024);
    assert!(Kernel::rmsnorm(&tiny, 1024, 1e-6, 1024).unwrap_err().contains("shared memory"));
}

#[test]
fn thread_count_must_be_a_power_of_two_within_the_limit() {
    let mut k = norm(&Target::sm_120());
    k.threads = 96;
    assert!(k.validate().is_err());
    k.threads = 2048;
    assert!(k.validate().is_err());
}

#[test]
fn param_roles_must_match_the_body() {
    let mut k = norm(&Target::sm_120());
    k.params.pop();
    assert!(k.validate().unwrap_err().contains("params"));
    let mut k = norm(&Target::sm_120());
    k.params[1].role = kir::Role::Scratch;
    assert!(k.validate().is_err());
}

#[test]
fn zero_width_is_rejected() {
    let t = Target::sm_120();
    assert!(Kernel::rmsnorm(&t, 0, 1e-6, 256).is_err());
    assert!(Kernel::gated_activation(&t, "swiglu", 0, Activation::Silu, 256).is_err());
}

#[test]
fn hash_is_stable_and_tracks_ir_and_target() {
    let t = Target::sm_120();
    assert_eq!(norm(&t).hash(), norm(&t).hash());
    assert_eq!(norm(&t).hash().len(), 16);
    assert_ne!(norm(&t).hash(), Kernel::rmsnorm(&t, 2048, 1e-6, 1024).unwrap().hash());
    assert_ne!(norm(&t).hash(), Kernel::rmsnorm(&t, 1024, 1e-5, 1024).unwrap().hash());
    assert_ne!(act(&t, Activation::Silu).hash(), act(&t, Activation::GeluTanh).hash());
    assert_ne!(norm(&Target::sm_120()).hash(), norm(&Target::sm_80()).hash());
    assert_ne!(norm(&Target::wgpu_baseline()).hash(), norm(&subgroups()).hash());
    let mut more_sms = Target::sm_120();
    more_sms.sm_count += 1;
    assert_ne!(norm(&t).hash(), norm(&more_sms).hash());
}

#[test]
fn module_hash_orders_kernels() {
    let t = Target::sm_120();
    let a = norm(&t);
    let b = act(&t, Activation::Silu);
    assert_eq!(kir::module_hash(&[a.clone(), b.clone()]), kir::module_hash(&[a.clone(), b.clone()]));
    assert_ne!(kir::module_hash(&[a.clone(), b.clone()]), kir::module_hash(&[b, a]));
}

#[test]
fn hash_value_is_pinned() {
    assert_eq!(norm(&Target::sm_120()).hash(), PINNED_SM120_NORM);
}

#[test]
fn language_matches_target() {
    assert_eq!(Target::sm_80().language, Language::Cuda);
    assert_eq!(Target::wgpu_baseline().language, Language::Wgsl);
}

const PINNED_SM120_NORM: &str = "58cb4e607bffc152";

mod layouts {
    use kir::layout::{Dialect, Layout, leaf, tuple};

    #[test]
    fn row_and_col_major_offsets() {
        let r = Layout::row_major(&[4, 8]).unwrap();
        assert_eq!(r.to_string(), "(4,8):(8,1)");
        assert_eq!(r.offset(1), 8);
        assert_eq!(r.offset(4), 1);
        let c = Layout::col_major(&[4, 8]).unwrap();
        assert_eq!(c.offset(1), 1);
        assert_eq!(c.offset(4), 4);
        assert_eq!(r.size(), 32);
        assert_eq!(r.cosize(), 32);
    }

    #[test]
    fn nested_layouts_expand_modes() {
        let l = Layout::new(tuple(vec![tuple(vec![leaf(2), leaf(2)]), leaf(4)]), tuple(vec![tuple(vec![leaf(1), leaf(8)]), leaf(2)])).unwrap();
        assert_eq!(l.rank(), 2);
        assert_eq!(l.size(), 16);
        assert_eq!(l.offset(0), 0);
        assert_eq!(l.offset(1), 1);
        assert_eq!(l.offset(2), 8);
        assert_eq!(l.offset(4), 2);
    }

    #[test]
    fn incongruent_shape_and_stride_are_rejected() {
        assert!(Layout::new(tuple(vec![leaf(2), leaf(3)]), leaf(1)).is_err());
        assert!(Layout::flat(&[0, 2], &[1, 1]).is_err());
    }

    #[test]
    fn coalesce_merges_contiguous_modes() {
        assert_eq!(Layout::col_major(&[4, 8]).unwrap().coalesce().to_string(), "32:1");
        assert_eq!(Layout::flat(&[4, 8], &[16, 1]).unwrap().coalesce().to_string(), "(4,8):(16,1)");
        assert_eq!(Layout::flat(&[1, 8], &[5, 1]).unwrap().coalesce().to_string(), "8:1");
    }

    #[test]
    fn composition_applies_the_outer_stride() {
        let outer = Layout::flat(&[64], &[2]).unwrap();
        let inner = Layout::flat(&[4, 8], &[8, 1]).unwrap();
        let c = outer.compose(&inner).unwrap();
        for i in 0..32 {
            assert_eq!(c.offset(i), outer.offset(inner.offset(i)));
        }
        assert!(Layout::flat(&[8], &[1]).unwrap().compose(&inner).is_err());
    }

    #[test]
    fn index_expressions_agree_with_offsets_in_both_dialects() {
        let l = Layout::flat(&[16, 2], &[2, 1]).unwrap();
        assert_eq!(l.expr(&["i", "k"], Dialect::Cuda).unwrap(), "i * 2 + k");
        assert_eq!(l.expr(&["i", "1"], Dialect::Wgsl).unwrap(), "i * 2u + 1u");
        assert_eq!(l.expr(&["i", "0"], Dialect::Cuda).unwrap(), "i * 2");
        assert!(l.expr(&["i"], Dialect::Cuda).is_err());
        assert!(l.expr(&["i", "2"], Dialect::Cuda).is_err());
    }

    #[test]
    fn nested_mode_expression_splits_the_coordinate() {
        let l = Layout::new(tuple(vec![leaf(4), tuple(vec![leaf(2), leaf(3)])]), tuple(vec![leaf(1), tuple(vec![leaf(12), leaf(4)])])).unwrap();
        assert_eq!(l.expr(&["a", "b"], Dialect::Cuda).unwrap(), "a + (b % 2) * 12 + (b / 2) * 4");
    }
}

#[test]
fn offset_norm_adds_the_offset_to_the_weight() {
    let k = Kernel::rmsnorm_offset(&Target::sm_120(), 128, 1e-6, 1.0, 256).unwrap();
    let src = kir::cuda::emit_kernel(&k).unwrap();
    assert!(src.contains("(weight[i] + 1e0f)"), "{src}");
    let plain = kir::cuda::emit_kernel(&norm(&Target::sm_120())).unwrap();
    assert!(!plain.contains("+ 1e0f"));
}

#[test]
fn residual_add_and_norm_fuse_into_one_kernel() {
    let k = Kernel::add_rmsnorm(&Target::sm_120(), 128, 1e-6, 0.0, 256).unwrap();
    let src = kir::cuda::emit_module(&[k]).unwrap();
    assert!(src.contains("add_rmsnorm"));
    assert!(src.contains("x[base + i] + delta[base + i]"));
    assert!(src.contains("sum[base + i] = v"));
}

#[test]
fn soft_cap_and_scale_are_generic_pointwise_primitives() {
    let t = Target::sm_120();
    let cap = Kernel::pointwise(&t, "softcap", 64, kir::Pointwise::SoftCap(30.0), 256).unwrap();
    assert!(kir::cuda::emit_kernel(&cap).unwrap().contains("3e1f * tanhf(v / 3e1f)"));
    let scale = Kernel::pointwise(&t, "embed_scale", 64, kir::Pointwise::Scale(8.0), 256).unwrap();
    assert!(kir::cuda::emit_kernel(&scale).unwrap().contains("v * 8e0f"));
    assert!(Kernel::pointwise(&t, "bad", 64, kir::Pointwise::SoftCap(0.0), 256).is_err());
}

#[test]
fn add_is_a_generic_elementwise_primitive() {
    let k = Kernel::add(&Target::sm_120(), "residual_add", 128, 256).unwrap();
    assert!(kir::cuda::emit_kernel(&k).unwrap().contains("out[base + i] = x[base + i] + delta[base + i]"));
}

#[test]
fn the_new_light_primitives_are_refused_on_wgpu_with_a_clear_error() {
    let t = Target::wgpu_baseline();
    for k in [
        Kernel::add(&t, "residual_add", 128, 256).unwrap(),
        Kernel::add_rmsnorm(&t, 128, 1e-6, 0.0, 256).unwrap(),
        Kernel::pointwise(&t, "softcap", 128, kir::Pointwise::SoftCap(30.0), 256).unwrap(),
    ] {
        let err = kir::wgsl::emit_kernel(&k).unwrap_err();
        assert!(err.contains("no WGSL emitter"), "{err}");
    }
}
