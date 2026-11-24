use kir::desc::{Description, Shape};
use kir::{Body, Target};

const SCAN: &str = include_str!("../../specrt/specs/mamba2_scan.toml");
const CONV: &str = include_str!("../../specrt/specs/mamba2_conv.toml");
const NORM: &str = include_str!("../../specrt/specs/mamba2_gated_norm.toml");
const MIX: &str = include_str!("../../specrt/specs/hc_mix.toml");
const INJECT: &str = include_str!("../../specrt/specs/hc_inject_add.toml");
const EXPAND: &str = include_str!("../../specrt/specs/hc_expand.toml");

fn shape() -> Shape {
    Shape::new()
        .num("ssm_state", 16.0)
        .num("ssm_head_dim", 32.0)
        .num("ssm_heads", 16.0)
        .num("ssm_groups", 2.0)
        .num("ssm_inner", 512.0)
        .num("ssm_conv", 576.0)
        .num("ssm_in", 1104.0)
        .num("ssm_taps", 4.0)
        .num("linear_layers", 2.0)
        .num("norm_eps", 1e-5)
        .num("hidden", 128.0)
        .num("hc_streams", 4.0)
        .num("hc_width", 512.0)
}

fn build(text: &str) -> Result<kir::Kernel, String> {
    Description::parse(text)?.build(&Target::sm_120(), &shape())
}

#[test]
fn the_mamba2_descriptions_build_recurrence_and_sequence_bodies() {
    let scan = build(SCAN).unwrap();
    assert!(matches!(scan.body, Body::Recurrence(_)));
    assert_eq!(scan.threads, 32 * 4);
    let src = kir::cuda::emit_module(&[scan, build(CONV).unwrap(), build(NORM).unwrap()]).unwrap();
    assert!(src.contains("s[j] *= r_keep;"), "{src}");
    assert!(src.contains("(size_t)(head / 8) * 16"), "{src}");
    assert!(src.contains("float win[TAPS];"), "{src}");
    assert!(src.contains("split_hi_lo(o, hi, lo);"), "{src}");
}

#[test]
fn the_hash_follows_the_recurrence_stages() {
    let a = build(SCAN).unwrap();
    let b = build(&SCAN.replace("o + d * x", "o")).unwrap();
    assert_ne!(a.hash(), b.hash());
    assert_eq!(a.hash(), build(SCAN).unwrap().hash());
}

#[test]
fn bad_recurrence_descriptions_are_refused_with_the_kernel_name() {
    let e = build(&SCAN.replace("ksplit = 4", "ksplit = 64")).unwrap_err();
    assert!(e.contains("mamba2_scan") && e.contains("ksplit"), "{e}");
    let e = build(&SCAN.replace("key = \"b\"", "key = \"x\"")).unwrap_err();
    assert!(e.contains("mamba2_scan") && e.contains("\"x\""), "{e}");
    let e = build(&SCAN.replace("op = \"decay\"", "op = \"shuffle\"")).unwrap_err();
    assert!(e.contains("shuffle"), "{e}");
    let e = build(&SCAN.replace("state = \"ssm\"", "state = \"y\"")).unwrap_err();
    assert!(e.contains("store") || e.contains("state"), "{e}");
    let e = build(&SCAN.replace("name = \"slot_step\"", "name = \"slot_stepz\"")).unwrap_err();
    assert!(e.contains("slot_step"), "{e}");
    let e = build(&SCAN.replace("space = \"scalar\"\nrow = 0", "space = \"lane\"\nrow = 0")).unwrap_err();
    assert!(e.contains("lane"), "{e}");
    let e = Description::parse(SCAN).unwrap().build(&Target::wgpu_baseline(), &shape()).unwrap_err();
    assert!(e.contains("CUDA") || e.contains("WarpShuffle"), "{e}");
    let e = build(&CONV.replace("taps = \"ssm_taps\"", "taps = 9")).unwrap_err();
    assert!(e.contains("mamba2_conv") && e.contains("taps"), "{e}");
    let e = build(&NORM.replace("x * silu(g)", "x * silu(q)")).unwrap_err();
    assert!(e.contains("mamba2_gated_norm"), "{e}");
}

#[test]
fn group_sum_reduces_each_column_over_the_stream_groups() {
    let k = build(MIX).unwrap();
    assert!(matches!(k.body, Body::Seq(_)));
    let src = kir::cuda::emit_module(&[k]).unwrap().into_string();
    assert!(src.contains("for (int g = 0; g < 4; g++)"), "{src}");
    assert!(src.contains("a[(size_t)row * 512 + (size_t)g * 128 + h]"), "{src}");
    assert!(src.contains("y[(size_t)row * 128 + h] = s / (float)4;") || src.contains("y[(size_t)row * 128 + h] = (s / (float)4);"), "{src}");
}

#[test]
fn group_update_writes_every_stream_from_one_column_and_a_per_group_scalar() {
    let src = kir::cuda::emit_module(&[build(INJECT).unwrap(), build(EXPAND).unwrap()]).unwrap().into_string();
    assert!(src.contains("const float cv = c[(size_t)row * 4 + g];"), "{src}");
    assert!(src.contains("float* xs = x + (size_t)row * 512 + (size_t)g * 128 + h;"), "{src}");
    assert!(src.contains("hc_expand(float* x, const float* a)"), "{src}");
}

#[test]
fn group_families_check_their_expressions_and_params() {
    let e = build(&MIX.replace("sigmoid(a) * b", "sigmoid(a) * q")).unwrap_err();
    assert!(e.contains("hc_mix") && e.contains("\"q\""), "{e}");
    let e = build(&EXPAND.replace("expr = \"a\"", "expr = \"a * c\"")).unwrap_err();
    assert!(e.contains("hc_expand") && e.contains("\"c\""), "{e}");
    let e = build(&INJECT.replace("name = \"x\"\nrole = \"activation\"\naccess = \"write\"", "name = \"x\"\nrole = \"activation\"\naccess = \"read\"")).unwrap_err();
    assert!(e.contains("\"x\""), "{e}");
    let e = build(&MIX.replace("groups = \"hc_streams\"", "groups = 0")).unwrap_err();
    assert!(e.contains("group_sum"), "{e}");
}
