use kir::Target;
use kir::weight_quant::WeightQuant;
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
fn nvfp4_weight_quant_launches_follow_the_target() {
    let k = WeightQuant::nvfp4(&Target::sm_120()).unwrap();
    assert_eq!((k.amax_launch().grid, k.amax_launch().threads), (188 * 8, 256));
    assert_eq!((k.quant_launch(129).grid, k.quant_launch(129).threads), (2, 128));
    assert_eq!(k.quant_launch(0).grid, 1);
    assert!(WeightQuant::nvfp4(&Target::wgpu_baseline()).is_err());
    assert_ne!(k.hash(), WeightQuant::nvfp4(&Target::sm_80()).unwrap().hash());
}
