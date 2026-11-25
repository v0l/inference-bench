use serde_json::Value;
use specrt::spec::{AttentionKind, Block, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-kimi-k3";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture() -> PathBuf {
    root().join("tests/fixtures").join(NAME)
}

fn config() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap()
}

fn without(keys: &[&str]) -> String {
    let mut v = config();
    let t = v["text_config"].as_object_mut().unwrap();
    for k in keys {
        t.remove(*k);
    }
    v.to_string()
}

fn spec_of(json: &str) -> Spec {
    let mut spec = Spec::from_hf_config_serving(json).unwrap_or_else(|e| panic!("{e:#}"));
    spec.name = NAME.into();
    spec.repo = fixture().display().to_string();
    spec
}

#[test]
fn situ_comes_from_the_family_as_a_gated_expression_with_the_config_betas() {
    let spec = spec_of(&without(&["attn_res_block_size", "routed_expert_hidden_size"]));
    let g = spec.mlp.gated.as_ref().expect("situ is a gated expression");
    assert_eq!((g.params["beta"], g.params["linear_beta"]), (4.0, 25.0));
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    assert!(module.source.contains("#define GATED_BODY(g, u) (") && module.source.contains("tanhf(g / 4e0f)") && module.source.contains("tanhf(u / 2.5e1f)"), "situ lowered into the gated body");
    let mut v = config();
    v["text_config"]["hidden_act"] = "silu".into();
    for k in ["attn_res_block_size", "routed_expert_hidden_size"] {
        v["text_config"].as_object_mut().unwrap().remove(k);
    }
    assert!(spec_of(&v.to_string()).mlp.gated.is_none(), "silu keeps the plain swiglu");
}

#[test]
fn mla_q_lora_and_the_output_gate_are_attention_nodes() {
    let spec = spec_of(&without(&["attn_res_block_size", "routed_expert_hidden_size"]));
    assert_eq!(spec.attention.kind, AttentionKind::Mla);
    assert_eq!(spec.qkv_rows(), 64 + 64 + 16 + 4 * 32, "q_a, kv_a and g_proj share the qkv GEMM");
    assert_eq!(spec.o_dim(), 4 * 32);
    let kernels: Vec<&str> = spec.attention.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert_eq!(kernels, ["mla_q_lora_norm", "mla_kv_lora_norm", "dsa_q_b", "dsa_latent_write", "dsa_absorb_q", "mla_lora_attention", "mla_expand_v_rows", "mla_out_gate"]);
    let mixer = spec.mixer.as_ref().unwrap();
    let projections: Vec<&str> = mixer.projection.iter().map(|p| p.tensor.as_str()).collect();
    assert_eq!(projections, ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_proj", "b_proj"]);
    assert_eq!(spec.mix_in_rows(), 4 * 128 + 64 + 2);
    let mixed: Vec<&str> = mixer.node.iter().filter_map(|n| n.kernel.as_deref()).collect();
    assert!(mixed.contains(&"kda_recurrence") && mixed.contains(&"kda_gated_norm_mix") && !mixed.contains(&"kda_out_gate_up"), "{mixed:?}");
    assert_eq!(mixer.params["kda_lower_bound"], -5.0);
    let module = specrt::codegen::Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap_or_else(|e| panic!("{e:#}"));
    for k in ["mla_out_gate", "mla_lora_attention", "kda_gated_norm_mix", "kda_recurrence", "dsa_q_b"] {
        assert!(module.kernels.contains(&k) && module.source.contains(&format!("{k}(")), "{k}");
    }
    assert!(module.source.contains("split_hi_lo((a * (1.0 / (1.0 + expf(-g)))), hi, lo)") || module.source.contains("split_hi_lo("), "the gate stores bf16 planes");
    assert!(spec.layers.pattern.iter().filter(|b| **b == Block::Full).count() == 2);
}
