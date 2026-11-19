use kir::attention::{AttentionShape, Family, MropeShape, RopeShape};
use kir::desc::{ATTENTION, BUILTIN, Description, Shape, build_attention, build_builtin};
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

fn attention_shape() -> (Shape, AttentionShape) {
    let rust = AttentionShape {
        heads: 24,
        kv_heads: 4,
        head_dim: 256,
        qkv_row: 14336,
        output_gate: true,
        qk_norm: true,
        eps: 1e-6,
        rope: RopeShape {
            rotary_dim: 64,
            theta: 10000000.0,
            kind: 3,
            factor: 16.0,
            low: 1.0,
            high: 4.0,
            original: 8192.0,
            mscale: 0.7,
            freq_factors: vec![2.0, 3.0],
            mrope: Some(MropeShape { section: [11, 11, 10], interleaved: true }),
        },
        kv_layers: 16,
        kv_planes: 1,
        kv_fp8: true,
        kv_f16_layers: 8,
        query_scale: 0.0625,
        softcap: 30.0,
        local_theta: 10000.0,
        local_rope: vec![false, true, false, false],
        bidirectional: false,
        nope: vec![false, false, true, false],
    };
    let s = Shape::new()
        .num("heads", 24.0)
        .num("kv_heads", 4.0)
        .num("head_dim", 256.0)
        .num("qkv_row", 14336.0)
        .num("norm_eps", 1e-6)
        .num("rotary_dim", 64.0)
        .num("rope_theta", 10000000.0)
        .num("rope_kind", 3.0)
        .num("rope_factor", 16.0)
        .num("rope_low", 1.0)
        .num("rope_high", 4.0)
        .num("rope_original", 8192.0)
        .num("rope_mscale", 0.7)
        .nums("rope_freq_factors", &[2.0, 3.0])
        .nums("mrope_section", &[11.0, 11.0, 10.0])
        .flag("mrope_interleaved", true)
        .num("kv_layers", 16.0)        .num("kv_planes", 1.0)
        .num("kv_f16_layers", 8.0)
        .num("query_scale", 0.0625)
        .num("softcap", 30.0)
        .num("local_theta", 10000.0)
        .flag("output_gate", true)
        .flag("qk_norm", true)
        .flag("kv_fp8", true)
        .flag("bidirectional", false)
        .flags("nope", &[false, false, true, false])
        .flags("local_rope", &[false, true, false, false]);
    (s, rust)
}

#[test]
fn attention_descriptions_build_the_same_kernels_as_the_rust_builders() {
    let (s, want_shape) = attention_shape();
    for t in [Target::sm_120(), Target::sm_80()] {
        let want = [
            ("qk_norm_rope", Family::RopeKv),
            ("kv_norm_rope", Family::RopeKv),
            ("attention_split", Family::SplitDecode),
            ("attention_reduce", Family::SplitDecode),
            ("flash_attention", Family::FlashLegacy),
            ("flash_mma", Family::FlashMma),
            ("batch_decode_attention", Family::GroupedDecode),
            ("batch_decode_combine", Family::GroupedDecode),
        ];
        assert_eq!(want.len(), ATTENTION.len());
        for (name, family) in want {
            let rust = kir::attention::AttentionKernel::new(&t, family, want_shape.clone()).unwrap();
            let data = build_attention(name, &t, &s).unwrap();
            assert_eq!(data, rust, "{name}");
            assert_eq!(data.hash(), rust.hash(), "{name}");
        }
    }
}

#[test]
fn attention_descriptions_refuse_bad_shapes_and_families() {
    let (s, _) = attention_shape();
    let e = Description::attention("attention_split").unwrap().build_attention(&Target::sm_120(), &Shape::new()).unwrap_err();
    assert!(e.contains("which the model shape does not provide"), "{e}");
    let mut bad = Description::attention("attention_split").unwrap();
    bad.family = "mystery".into();
    let e = bad.build_attention(&Target::sm_120(), &s).unwrap_err();
    assert!(e.contains("unknown attention family"), "{e}");
    let mut no_shape = Description::attention("attention_split").unwrap();
    no_shape.shape = None;
    let e = no_shape.build_attention(&Target::sm_120(), &s).unwrap_err();
    assert!(e.contains("needs a [shape]"), "{e}");
    let mut wrong = Description::attention("attention_split").unwrap();
    wrong.shape.as_mut().unwrap().kv_planes = 3.into();
    let e = wrong.build_attention(&Target::sm_120(), &s).unwrap_err();
    assert!(e.contains("kv planes"), "{e}");
    let mut staged = Description::attention("attention_split").unwrap();
    staged.threads = kir::desc::Threads::Fixed(64);
    let e = staged.build_attention(&Target::sm_120(), &s).unwrap_err();
    assert!(e.contains("takes only [shape]"), "{e}");
}

#[test]
fn attention_descriptions_resolve_every_shape_symbol_from_the_model_shape() {
    let (s, want_shape) = attention_shape();
    let d = Description::attention("batch_decode_attention").unwrap();
    let k = d.build_attention(&Target::sm_120(), &s).unwrap();
    assert_eq!(k.shape, want_shape);
    assert!(k.shape.rope.mrope.as_ref().unwrap().interleaved);
    assert_eq!(k.shape.rope.freq_factors, vec![2.0, 3.0]);
    let missing = Shape::new();
    let e = d.build_attention(&Target::sm_120(), &missing).unwrap_err();
    assert!(e.contains("which the model shape does not provide"), "{e}");
}
