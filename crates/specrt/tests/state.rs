use specrt::anatomy::ModelAnatomy;
use specrt::runtime::kv_block_bytes;
use specrt::spec::Spec;
use specrt::state::{self, Per, Rollback};
use std::path::Path;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn qwen3() -> Spec {
    Spec::load(&root().join("specs/qwen3-0.6b.toml")).unwrap()
}

fn hybrid() -> Spec {
    Spec::from_hf_config(&std::fs::read_to_string(root().join("tests/fixtures/qwen3.8-27b.config.json")).unwrap()).unwrap()
}

const LATENT: &str = r#"
[[state]]
name = "latent"
per = "token"
layers = "attention"
shape = ["1", "kv_lora_rank + rope_dim"]
dtype = "bf16"
rollback = "positional"

[attention.params]
kv_lora_rank = 512.0
rope_dim = 64.0

[[attention.node]]
op = "kernel"
kernel = "attn_sinks"
grid = ["heads", "rows"]
"#;

#[test]
fn stock_specs_declare_k_and_v_and_keep_the_old_block_size() {
    let s = qwen3();
    assert!(state::is_stock(&s.state));
    let a = &s.attention;
    assert_eq!(kv_block_bytes(&s, 1), 2 * s.kv_layers() * a.kv_heads * 16 * a.head_dim * 2);
    assert_eq!(kv_block_bytes(&s, 2), 2 * kv_block_bytes(&s, 1));
    assert!(!s.to_toml().unwrap().contains("[[state]]"), "stock states are not written out");
}

#[test]
fn the_hybrid_family_declares_checkpointed_delta_state_sized_from_the_config() {
    let s = hybrid();
    let la = s.linear_attention.clone().unwrap();
    let conv = state::named(&s, "conv").unwrap();
    let ssm = state::named(&s, "ssm").unwrap();
    assert_eq!((conv.per, conv.rollback, ssm.rollback), (Per::Sequence, Rollback::Checkpoint, Rollback::Checkpoint));
    let want = s.linear_layers() * (la.conv_dim() * (la.conv_kernel - 1) + la.value_heads * la.key_head_dim * la.key_head_dim) * 4;
    assert_eq!(state::sequence_bytes(&s), want as u64);
    assert_eq!(ModelAnatomy::new(&s, 0, None).state_bytes_per_seq, want as u64);
}

#[test]
fn a_declared_latent_cache_sizes_the_kv_block_and_the_fit() {
    let base = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    let s = Spec::parse(&format!("{base}\n{LATENT}")).unwrap();
    assert_eq!(s.state.len(), 1);
    let per_token = s.kv_layers() * 576 * 2;
    assert_eq!(state::kv_bytes_per_token_at(&s, 0), 576 * 2);
    assert_eq!(kv_block_bytes(&s, 1), per_token * 16);
    assert_eq!(ModelAnatomy::new(&s, 0, None).kv_bytes_per_token, per_token as u64);
    assert!(s.to_toml().unwrap().contains("[[state]]"), "a declared state is written out");
}

#[test]
fn bad_state_declarations_are_load_errors() {
    let base = std::fs::read_to_string(root().join("specs/qwen3-0.6b.toml")).unwrap();
    let parse = |extra: &str| Spec::parse(&format!("{base}\n{extra}")).map(|_| ()).map_err(|e| format!("{e:#}")).unwrap_err();
    let e = parse(&LATENT.replace("rollback = \"positional\"", "rollback = \"checkpoint\""));
    assert!(e.contains("positional"), "{e}");
    let e = parse(&LATENT.replace("rope_dim\"", "missing_dim\""));
    assert!(e.contains("missing_dim"), "{e}");
    let e = parse(&format!("{LATENT}\n[[state]]\nname = \"latent\"\nper = \"token\"\nlayers = \"attention\"\nshape = [\"1\"]\ndtype = \"bf16\"\nrollback = \"positional\"\n"));
    assert!(e.contains("repeated"), "{e}");
    let e = parse(&LATENT.replace("[[attention.node]]\nop = \"kernel\"\nkernel = \"attn_sinks\"\ngrid = [\"heads\", \"rows\"]\n", ""));
    assert!(e.contains("\"k\""), "stock attention needs the k and v states: {e}");
}
