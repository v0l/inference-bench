use kir::gemv::{Gemv, GemvKernel};
use kir::ops::WgslShape;
use kir::prims::{Prim, PrimKernel, RowSource};
use kir::wgsl::OpKind;
use kir::Target;
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/prims").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn shape(rope_kind: u32, head_dim: usize) -> WgslShape {
    WgslShape {
        hidden: 1024,
        inter: 3072,
        vocab: 151936,
        heads: 16,
        kv_heads: 8,
        head_dim,
        qk_norm: true,
        eps: 1e-6,
        theta: 1e6,
        rope_kind,
        rope_factor: 4.0,
        rope_low: 1.0,
        rope_high: 4.0,
        rope_original: 8192.0,
        rope_mscale: 1.0,
        gated_gelu: false,
        max_ctx: 4096,
        q_dim: 16 * head_dim,
        qkv_row: 32 * head_dim,
        rotary_dim: head_dim,
    }
}

fn validate_wgsl(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
}

#[test]
fn every_wgsl_op_validates_and_matches_its_golden() {
    for (kind, tag) in [(OpKind::Embed, "embed"), (OpKind::Rope, "rope"), (OpKind::Attention, "attention"), (OpKind::Bias, "bias"), (OpKind::Argmax, "argmax"), (OpKind::Dequant, "dequant")] {
        let src = kir::wgsl::emit_op(kind, &shape(0, 128));
        validate_wgsl(tag, &src);
        golden(&format!("{tag}.wgsl"), &src);
    }
    for (tokens, tag) in [(1, "linear"), (8, "linear_t8")] {
        let src = kir::wgsl::emit_linear_op(&shape(0, 128), tokens, 64);
        validate_wgsl(tag, &src);
        golden(&format!("{tag}.wgsl"), &src);
    }
}

#[test]
fn wgsl_rope_and_attention_cover_every_rope_kind_and_wide_heads() {
    for kind in 0..4 {
        for hd in [64, 128, 256] {
            validate_wgsl("rope", &kir::wgsl::emit_op(OpKind::Rope, &shape(kind, hd)));
            validate_wgsl("attention", &kir::wgsl::emit_op(OpKind::Attention, &shape(kind, hd)));
        }
    }
}

#[test]
fn rope_frequency_is_one_builder_for_both_dialects() {
    let cuda = kir::rope::inv_freq_fn(kir::Dialect::Cuda, &[1, 2, 3, 4]);
    let wgsl = kir::rope::inv_freq_fn(kir::Dialect::Wgsl, &[1, 2, 3]);
    for src in [&cuda, &wgsl] {
        assert!(src.contains("ROPE_KIND == 1") && src.contains("ROPE_KIND == 2") && src.contains("ROPE_KIND == 3"));
        assert!(src.contains("6.283185307179586"));
    }
    assert!(cuda.contains("ROPE_FREQ_FACTORS[i]") && !wgsl.contains("ROPE_FREQ_FACTORS"));
    golden("rope_inv_freq.cu", &cuda);
}

#[test]
fn gumbel_is_one_hash_for_both_dialects() {
    let cuda = kir::prims::gumbel(kir::Dialect::Cuda);
    let wgsl = kir::prims::gumbel(kir::Dialect::Wgsl);
    for c in ["747796405u", "2891336453u", "2246822519u", "2246822507u", "3266489909u", "16777216.0"] {
        assert!(cuda.contains(c) && wgsl.contains(c), "{c}");
    }
}

#[test]
fn cuda_prims_golden_per_target() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        let ks = kir::prims::cuda_prims(&t, 1024, 151936, 1e-6, 1.0).unwrap();
        let src = kir::prims::emit_cuda(&ks).unwrap();
        for k in &ks {
            assert!(src.contains(&format!("void {}(", k.name)), "{}", k.name);
        }
        golden(&format!("prims_{tag}.cu"), &src);
    }
}

#[test]
fn prims_refuse_bad_shapes_and_wrong_dialects() {
    let t = Target::sm_120();
    assert!(PrimKernel::new(&t, "rms_split", Prim::RmsSplit { width: 1022, eps: 1e-6 }).is_err());
    assert!(PrimKernel::new(&t, "argmax_rows", Prim::ArgmaxRows { vocab: 0 }).is_err());
    assert!(PrimKernel::new(&t, "bad name", Prim::Bump).is_err());
    assert!(PrimKernel::new(&t, "bias", Prim::AddBias).unwrap_err().contains("lowering"));
    let w = Target::wgpu_baseline();
    assert!(PrimKernel::new(&w, "embed", Prim::Embed { width: 1024, scale: 1.0, source: RowSource::Bf16 }).is_err());
    assert!(PrimKernel::new(&w, "argmax_rows", Prim::ArgmaxRows { vocab: 8 }).is_err());
    assert!(PrimKernel::new(&w, "main", Prim::Sample { vocab: 8 }).is_ok());
}

#[test]
fn prims_hash_tracks_the_shape() {
    let t = Target::sm_120();
    let a = kir::prims::prims_hash(&kir::prims::cuda_prims(&t, 1024, 151936, 1e-6, 1.0).unwrap());
    assert_ne!(a, kir::prims::prims_hash(&kir::prims::cuda_prims(&t, 2048, 151936, 1e-6, 1.0).unwrap()));
    assert_ne!(a, kir::prims::prims_hash(&kir::prims::cuda_prims(&t, 1024, 151936, 1e-6, 2.0).unwrap()));
}

#[test]
fn gemv_wrappers_golden_and_refusals() {
    let t = Target::sm_120();
    let ks = vec![
        GemvKernel::new(&t, "qkv_proj", Gemv::Linear { decoder: "DecBf16".into(), n: 4096, k: 1024, bias: false, residual: false }).unwrap(),
        GemvKernel::new(&t, "qkv_gemm", Gemv::Tiled { decoder: "DecQ8".into(), n: 4096, k: 1024, bias: true, residual: false }).unwrap(),
        GemvKernel::new(&t, "lm_head_norm", Gemv::Norm { decoder: "DecFp8".into(), n: 151936, bias: false }).unwrap(),
        GemvKernel::new(&t, "gate_up_act", Gemv::GateUpAct { decoder: "DecBf16".into() }).unwrap(),
        GemvKernel::new(&t, "lin_ba_rows", Gemv::Bf16Rows { n: 96, k: 5120 }).unwrap(),
    ];
    golden("gemv_sm120.cu", &kir::gemv::emit_gemvs(&ks).unwrap());
    assert!(GemvKernel::new(&t, "x", Gemv::Bf16Rows { n: 96, k: 5121 }).is_err());
    assert!(GemvKernel::new(&Target::wgpu_baseline(), "x", Gemv::GateUpAct { decoder: "DecBf16".into() }).is_err());
    assert_ne!(ks[0].hash(), GemvKernel::new(&t, "qkv_proj", Gemv::Linear { decoder: "DecBf16".into(), n: 4096, k: 1024, bias: true, residual: false }).unwrap().hash());
}

#[test]
#[ignore = "writes the linear_coop WGSL to KIR_DUMP_COOP"]
fn dump_linear_coop_wgsl() {
    let Some(path) = std::env::var_os("KIR_DUMP_COOP") else { return };
    let variant = match std::env::var("KIR_DUMP_VARIANT").as_deref() {
        Ok("bf16") => kir::wgsl::SgVariant::Bf16,
        Ok("q4k") => kir::wgsl::SgVariant::Q4K,
        Ok("q6k") => kir::wgsl::SgVariant::Q6K,
        _ => kir::wgsl::SgVariant::Any,
    };
    std::fs::write(path, kir::wgsl::emit_linear_coop_op(&shape(0, 128), variant)).unwrap();
}
