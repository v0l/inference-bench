use kir::ggml::GgmlKernel;
use kir::vision::VisionKernel;
use kir::{Feature, Target};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ggml_vision").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

#[test]
fn ggml_decoder_golden_per_target() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        golden(&format!("ggml_{tag}.cu"), &GgmlKernel::new(&t).emit().unwrap());
    }
}

#[test]
fn ggml_decoders_cover_every_block_type() {
    let src = GgmlKernel::new(&Target::sm_120()).emit().unwrap();
    for name in ["gg_q4k_rows", "gg_q6k_rows", "gg_dequant8", "gg_dequant_rows", "gg_dequant_bf16", "struct DecGgml"] {
        assert!(src.contains(name), "{name}");
    }
    for (_, bytes) in kir::ggml::BLOCK_BYTES {
        assert!(src.contains(&format!("return {bytes};")), "{bytes}");
    }
    assert!(!src.contains("{{"));
}

#[test]
fn ggml_wgsl_decoder_shares_the_block_sizes() {
    let src = kir::ggml::wgsl_decoder();
    for (_, bytes) in kir::ggml::BLOCK_BYTES {
        assert!(src.contains(&format!("return {bytes}u;")), "{bytes}");
    }
    assert_eq!(kir::wgsl::ggml_bindings().len(), 4);
}

#[test]
fn ggml_wgsl_decoder_parses() {
    let head = "@group(0) @binding(0) var<storage, read> wdata: array<u32>;\n";
    naga::front::wgsl::parse_str(&format!("{head}{}", kir::ggml::wgsl_decoder())).unwrap();
    golden("ggml_decoder.wgsl", &kir::ggml::wgsl_decoder());
}

#[test]
fn vision_goldens() {
    for (tag, t) in [("sm120", Target::sm_120()), ("sm80", Target::sm_80())] {
        for (hd, precise) in [(64usize, false), (72, true), (128, false)] {
            let k = VisionKernel::new(&t, hd, precise).unwrap();
            golden(&format!("vision_{hd}_{precise}_{tag}.cu"), &format!("// {}\n{}", k.describe().replace('\n', "\n// "), kir::cuda::emit_vision(&k).unwrap()));
        }
    }
}

#[test]
fn vision_source_has_every_kernel_and_no_slots() {
    let src = kir::cuda::emit_vision(&VisionKernel::new(&Target::sm_120(), 72, true).unwrap()).unwrap();
    for name in kir::vision::KERNELS {
        assert!(src.contains(&format!("void {name}(")) || src.contains(&format!("{name}(")), "{name}");
    }
    assert!(!src.contains("{{"));
    assert!(src.contains("constexpr int AT_HDP = 80;"));
    assert!(src.contains("constexpr bool GEMM_SPLIT = true;"));
}

#[test]
fn vision_validate_refuses_bad_shapes_and_targets() {
    let t = Target::sm_120();
    assert!(VisionKernel::new(&t, 68, false).unwrap_err().0.contains("multiple of 8"));
    assert!(VisionKernel::new(&t, 136, false).is_err());
    assert!(VisionKernel::new(&Target::wgpu_baseline(), 64, false).unwrap_err().0.contains("lowering"));
    let no_mma = Target::sm_80().without_feature(Feature::Mma16816);
    assert!(VisionKernel::new(&no_mma, 64, false).unwrap_err().0.contains("Mma16816"));
    let mut small = t;
    small.smem_limit = 8 * 1024;
    assert!(VisionKernel::new(&small, 64, false).unwrap_err().0.contains("shared memory"));
}

#[test]
fn vision_hash_changes_with_precision_and_head_dim() {
    let t = Target::sm_120();
    let a = VisionKernel::new(&t, 64, false).unwrap();
    assert_ne!(a.hash(), VisionKernel::new(&t, 64, true).unwrap().hash());
    assert_ne!(a.hash(), VisionKernel::new(&t, 72, false).unwrap().hash());
}
