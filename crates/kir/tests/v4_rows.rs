use kir::desc::{Description, Shape};
use kir::{Kernel, Target};

fn shape() -> Shape {
    Shape::new().num("rot", 16.0).num("dim", 64.0).num("layers", 2.0)
}

fn pool(overlap_dim: u32, rot: u32) -> Description {
    Description::parse(&format!(
        r#"kind = "kir-kernel"
family = "block_pool"
name = "pool"
threads = 128

[args]
dim = {overlap_dim}
rot = {rot}
layers = 2
raw_width = 256
gate_offset = 128
eps = 1.0e-6

[[param]]
name = "raw"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "bias"
type = "f32"
role = "weight"
access = "read"

[[param]]
name = "norm"
type = "f32"
role = "weight"
access = "read"

[[param]]
name = "dst"
type = "bf16"
role = "activation"
access = "write"

[[param]]
name = "pos"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "layer"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "pos_step"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "row_step"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "block_table"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "slots"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "pages"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "ratio"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "overlap"
type = "i32"
role = "value"
access = "read"
"#
    ))
    .unwrap()
}

fn build(d: &Description) -> Result<Kernel, String> {
    d.build(&Target::sm_120(), &shape())
}

#[test]
fn the_pool_emits_a_softmax_gated_window_mean_then_norm_then_rope_at_the_window_start() {
    let src = kir::cuda::emit_module(&[build(&pool(64, 16)).unwrap()]).unwrap();
    assert!(src.contains("(p + 1) % ratio != 0") && src.contains("const int e = p / ratio;"));
    assert!(src.contains("expf(g - top)") && src.contains("rsqrtf(total / (float)DIM + 1e-6f)"));
    assert!(src.contains("(float)(e * ratio) * rope_inv_freq(i, layer)"), "{src}");
    assert!(src.contains("const bool previous = overlap && k < ratio;") && src.contains("if (previous && e == 0) continue;"));
}

#[test]
fn the_pool_refuses_odd_rope_and_wide_rows() {
    assert!(build(&pool(64, 15)).unwrap_err().contains("even rot"));
    assert!(build(&pool(2048, 16)).unwrap_err().contains("1024"));
    assert!(build(&pool(8, 16)).unwrap_err().contains("no wider than dim"));
}

#[test]
fn the_pool_has_no_wgsl_lowering() {
    assert!(pool(64, 16).build(&Target::wgpu_baseline(), &shape()).is_err());
}
