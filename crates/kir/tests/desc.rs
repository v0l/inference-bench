use kir::desc::{BUILTIN, Description, Shape, build_builtin};
use kir::{Activation, Kernel, Pointwise, Target};

fn shape() -> Shape {
    Shape::new()
        .num("hidden", 1024.0)
        .num("inter", 3072.0)
        .num("vocab", 4096.0)
        .num("norm_eps", 1e-6)
        .num("q_dim", 64.0)
        .num("kv_dim", 32.0)
        .num("qkv_row", 128.0)
        .num("residual_scale", 0.22)
        .num("logit_scale", 0.5)
        .num("logit_softcap", 30.0)
        .word("mlp_act", "silu")
}

fn targets() -> Vec<Target> {
    vec![Target::sm_120(), Target::sm_80(), Target::wgpu_baseline(), Target::wgpu_baseline().with_subgroups(32)]
}

#[test]
fn descriptions_build_the_same_kernels_as_the_rust_builders() {
    let s = shape();
    for t in targets() {
        let want = [
            ("rmsnorm", Kernel::rmsnorm(&t, 1024, 1e-6, 1024)),
            ("swiglu", Kernel::gated_activation(&t, "swiglu", 3072, Activation::Silu, 256)),
            ("norm_add", Kernel::norm_add(&t, 1024, 1e-6, 0.0, 1024)),
            ("add_rmsnorm", Kernel::add_rmsnorm(&t, 1024, 1e-6, 0.0, 1024)),
            ("clear_branch", Kernel::pointwise(&t, "clear_branch", 1024, Pointwise::Fill(0.0), 256)),
            ("residual_add", Kernel::add_scaled(&t, "residual_add", 1024, 0.22, 256)),
            ("q_norm_rows", Kernel::rmsnorm_segment(&t, "q_norm_rows", 64, 128, 0, 1e-6, 1024)),
            ("k_norm_rows", Kernel::rmsnorm_segment(&t, "k_norm_rows", 32, 128, 64, 1e-6, 1024)),
            ("logit_scale", Kernel::pointwise(&t, "logit_scale", 4096, Pointwise::Scale(0.5), 256)),
            ("logit_softcap", Kernel::pointwise(&t, "logit_softcap", 4096, Pointwise::SoftCap(30.0), 256)),
        ];
        assert_eq!(want.len(), BUILTIN.len());
        for (name, rust) in want {
            let rust = rust.unwrap();
            let data = build_builtin(name, &t, &s).unwrap();
            assert_eq!(data, rust, "{name}");
            assert_eq!(data.hash(), rust.hash(), "{name}");
        }
    }
}

#[test]
fn geglu_description_selects_the_gelu_activation() {
    let t = Target::sm_120();
    let s = shape().word("mlp_act", "gelu_tanh");
    let k = build_builtin("swiglu", &t, &s).unwrap();
    assert_eq!(k, Kernel::gated_activation(&t, "swiglu", 3072, Activation::GeluTanh, 256).unwrap());
}

#[test]
fn a_description_in_the_user_text_builds_without_a_builtin() {
    let d = Description::parse("kind = \"kir-kernel\"\nfamily = \"pointwise\"\nname = \"half\"\nthreads = 128\n[args]\nwidth = 64\nop = \"scale\"\nvalue = 0.5\n").unwrap();
    let k = d.build(&Target::sm_120(), &Shape::new()).unwrap();
    assert_eq!(k, Kernel::pointwise(&Target::sm_120(), "half", 64, Pointwise::Scale(0.5), 128).unwrap());
}

#[test]
fn bad_descriptions_are_refused_with_a_reason() {
    let t = Target::sm_120();
    let parse = |s: &str| Description::parse(s).unwrap().build(&t, &Shape::new());
    assert!(Description::parse("kind = \"graph\"\nfamily = \"rmsnorm\"\nname = \"x\"\nthreads = 1\n").unwrap_err().contains("kind"));
    assert!(Description::parse("kind = \"kir-kernel\"\nfamily = \"x\"\nname = \"x\"\nthreads = 1\nextra = 1\n").is_err());
    let e = parse("kind = \"kir-kernel\"\nfamily = \"mystery\"\nname = \"x\"\nthreads = 64\n").unwrap_err();
    assert!(e.contains("unknown kernel family"), "{e}");
    let e = parse("kind = \"kir-kernel\"\nfamily = \"rmsnorm\"\nname = \"x\"\nthreads = 64\n[args]\nwidth = \"hidden\"\neps = 1e-6\noffset = 0\n").unwrap_err();
    assert!(e.contains("hidden"), "{e}");
    let e = parse("kind = \"kir-kernel\"\nfamily = \"rmsnorm\"\nname = \"x\"\nthreads = 64\n[args]\nwidth = 64\neps = 1e-6\noffset = 0\nbogus = 1\n").unwrap_err();
    assert!(e.contains("bogus"), "{e}");
    let e = parse("kind = \"kir-kernel\"\nfamily = \"rmsnorm\"\nname = \"x\"\nthreads = 64\n[args]\nwidth = 0\neps = 1e-6\noffset = 0\n").unwrap_err();
    assert!(e.contains("width"), "{e}");
}

#[test]
fn target_checks_still_run_on_description_kernels() {
    let tiny = Target::wgpu(1024, 1024);
    let e = build_builtin("rmsnorm", &tiny, &shape()).unwrap_err();
    assert!(e.contains("shared memory"), "{e}");
}
