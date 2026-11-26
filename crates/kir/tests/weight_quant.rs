use kir::Target;
use kir::weight_quant::{ScaleSearch, WeightQuant};
use std::path::PathBuf;

fn golden(name: &str, actual: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/weight_quant").join(name);
    if std::env::var_os("KIR_BLESS").is_some() {
        std::fs::write(&path, actual).unwrap();
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing golden {name}, run with KIR_BLESS=1"));
    assert_eq!(actual, want, "golden {name} differs");
}

fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| c.is_whitespace()).filter(|t| !t.is_empty()).map(str::to_string).collect()
}

#[test]
fn nvfp4_weight_quant_spells_the_old_script_token_for_token() {
    let k = WeightQuant::nvfp4(&Target::sm_120()).unwrap();
    let src = kir::cuda::emit_weight_quant(&k).unwrap();
    let old = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/weight_quant/nvfp4_old.cu")).unwrap();
    assert_eq!(tokens(&src), tokens(&old));
    golden("nvfp4_sm120.cu", &src);
}

#[test]
fn bandwidth_probe_spells_the_old_script_token_for_token() {
    let b = kir::bandwidth::Bandwidth::probe(&Target::sm_120()).unwrap();
    let src = kir::cuda::emit_bandwidth(&b);
    let old = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/bandwidth/old_kernels.cu")).unwrap();
    assert_eq!(tokens(&src), tokens(&old));
    assert_eq!(b.grid(), 188 * 8);
    assert!(kir::bandwidth::Bandwidth::probe(&Target::wgpu_baseline()).is_err());
}

#[test]
fn scale_search_kernels_take_their_candidates_from_data() {
    let t = Target::sm_120();
    let k = ScaleSearch::new(&t, &[-1, -2, -3, -4, 1], &[1.0, 0.96875, 0.9375, 0.90625, 0.875]).unwrap();
    let src = kir::cuda::emit_scale_search(&k).unwrap();
    for name in kir::weight_quant::SEARCH_KERNELS {
        assert!(src.contains(&format!("void {name}(")), "{name} missing");
    }
    assert!(src.contains("const int steps[5] = {-1, -2, -3, -4, 1};"));
    assert!(src.contains(&format!("__uint_as_float({:#x}u)", 0.96875f32.to_bits())));
    golden("search_sm120.cu", &src);
    let fewer = ScaleSearch::new(&t, &[-1], &[1.0]).unwrap();
    assert_ne!(k.hash(), fewer.hash());
    assert!(kir::cuda::emit_scale_search(&fewer).unwrap().contains("const int steps[1] = {-1};"));
    assert_eq!((k.row_launch(130).grid, k.row_launch(130).threads), (3, 64));
    assert_eq!(k.group_launch(129).grid, 2);
    assert!(ScaleSearch::new(&t, &[], &[1.0]).is_err());
    assert!(ScaleSearch::new(&t, &[0], &[1.0]).is_err());
    assert!(ScaleSearch::new(&t, &[-1], &[]).is_err());
    assert!(ScaleSearch::new(&Target::wgpu_baseline(), &[-1], &[1.0]).is_err());
}

#[test]
fn nvfp4_weight_quant_launches_follow_the_target() {
    let k = WeightQuant::nvfp4(&Target::sm_120()).unwrap();
    assert_eq!((k.amax_launch().grid, k.amax_launch().threads), (188 * 8, 256));
    assert_eq!((k.quant_launch(129).grid, k.quant_launch(129).threads), (2, 128));
    assert_eq!(k.quant_launch(0).grid, 1);
    assert!(WeightQuant::nvfp4(&Target::wgpu_baseline()).is_err());
    assert_ne!(k.hash(), WeightQuant::nvfp4(&Target::sm_80()).unwrap().hash());
}
