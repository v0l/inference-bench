use specrt::spec::{ActQuant, Quant, Spec};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var_os("GEMMA3_27B_DIR").expect("GEMMA3_27B_DIR holds an FP8 or NVFP4 Gemma 3 27B checkpoint"))
}

#[test]
#[ignore = "needs GEMMA3_27B_DIR (RedHatAI/gemma-3-27b-it-FP8-dynamic)"]
fn the_gemma3_27b_fp8_config_maps_through_the_family_spec() {
    let json = std::fs::read_to_string(dir().join("config.json")).unwrap();
    let spec = Spec::from_hf_config_serving(&json).unwrap();
    assert_eq!(spec.layers.count, 62);
    assert_eq!((spec.attention.heads, spec.attention.kv_heads, spec.attention.head_dim), (32, 16, 128));
    assert_eq!(spec.attention.query_scalar, Some(168.0));
    assert_eq!(spec.layers.window, Some(1024));
    assert_eq!(spec.layers.pattern.len(), 62);
    assert_eq!(spec.quant, Quant::Fp8Channel);
    assert_eq!(spec.act_quant, ActQuant::Fp8);
    assert!(spec.dims.tie_embeddings && spec.norm.sandwich && spec.norm.unit_offset);
    let v = spec.vision.unwrap();
    assert_eq!((v.config.tokens_per_tile(), v.config.out_hidden_size), (256, 5376));
}
