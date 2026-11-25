use kir::desc::{Description, Shape};
use kir::{Kernel, Target};

const POSITION: &str = r#"
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
"#;

fn shape() -> Shape {
    Shape::new().num("dim", 64.0).num("rot", 64.0).num("heads", 4.0).num("max_context", 4096.0)
}

fn pool_desc(dim: &str, rot: &str, unit: bool) -> Result<Description, String> {
    let text = format!(
        r#"kind = "kir-kernel"
family = "group_pool"
name = "pool"
threads = 64

[args]
dim = "{dim}"
rot = "{rot}"
layers = 1
ratio = 4
eps = 1.0e-6
unit = {unit}

[[param]]
name = "raw"
type = "bf16"
role = "activation"
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
{POSITION}"#
    );
    Description::parse(&text)
}

fn pool(dim: &str, rot: &str, unit: bool) -> Result<Kernel, String> {
    pool_desc(dim, rot, unit)?.build(&Target::sm_120(), &shape())
}

fn topk(topk: &str, max_groups: &str) -> Result<Kernel, String> {
    topk_with(topk, max_groups, "")
}

const SCORES: &str = r#"
[[param]]
name = "scores"
type = "f32"
role = "activation"
access = "write"
"#;

fn topk_with(topk: &str, max_groups: &str, extra: &str) -> Result<Kernel, String> {
    let text = format!(
        r#"kind = "kir-kernel"
family = "group_topk"
name = "select"
threads = 128

[args]
heads = "heads"
dim = "dim"
rot = "rot"
topk = {topk}
ratio = 4
max_groups = "{max_groups}"
q_row = "(heads + 1) * dim"
q_offset = 0
layers = 1
eps = 1.0e-6
unit = true

[[param]]
name = "q"
type = "f32"
role = "activation"
access = "read"

[[param]]
name = "norm"
type = "f32"
role = "weight"
access = "read"

[[param]]
name = "keys"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "idx"
type = "i32"
role = "activation"
access = "write"
{POSITION}
[[param]]
name = "scale"
type = "f32"
role = "value"
access = "read"
{extra}"#
    );
    Description::parse(&text)?.build(&Target::sm_120(), &shape())
}

#[test]
fn the_group_pool_means_complete_groups_then_norms_then_ropes_at_the_group_start() {
    let src = kir::cuda::emit_module(&[pool("dim", "rot", true).unwrap()]).unwrap();
    assert!(src.contains("if ((p + 1) % RATIO != 0) return;") && src.contains("const int start = p + 1 - RATIO;"), "{src}");
    assert!(src.contains("acc / (float)RATIO"), "{src}");
    assert!(src.contains("(1.0f + norm[d])"), "unit offset norm weight: {src}");
    assert!(src.contains("(float)start * rope_inv_freq(i, layer)"), "{src}");
    let plain = kir::cuda::emit_module(&[pool("dim", "rot", false).unwrap()]).unwrap();
    assert!(plain.contains("norm[d] * (pooled[d] * inv)") && !plain.contains("1.0f + norm[d]"));
}

#[test]
fn the_group_topk_scores_relu_dots_summed_over_heads_and_expands_groups_to_tokens_plus_the_tail() {
    let k = topk("4", "max_context / 4").unwrap();
    let src = kir::cuda::emit_module(&[k]).unwrap();
    assert!(src.contains("constexpr int OUT = TOPK * RATIO + RATIO - 1;"), "{src}");
    assert!(src.contains("const int n = min((p + 1) / RATIO, MAX_GROUPS);"), "{src}");
    assert!(src.contains("s += fmaxf(d, 0.0f);") && src.contains("score[g] = s * scale;"), "{src}");
    assert!(src.contains("(float)p * rope_inv_freq(j, layer)"), "the query is roped at its own position: {src}");
    assert!(src.contains("__shared__ float score[MAX_GROUPS];") && !src.contains("float* score"), "{src}");
    assert!(src.contains("for (int g = threadIdx.x; g < n; g += blockDim.x)"), "only scored groups are written: {src}");
    assert!(src.contains("const int tp = g * RATIO + RATIO - 1;"), "a pooled group lives at its last token: {src}");
    assert!(src.contains("idx[(size_t)row * OUT + r * RATIO + k] = at * RATIO + k;"), "{src}");
    assert!(src.contains("const int t = n * RATIO + (j - cnt * RATIO);") && src.contains("idx[(size_t)row * OUT + j] = t <= p ? t : -1;"), "{src}");
}

#[test]
fn the_group_topk_scores_in_a_per_row_global_scratch_past_the_shared_budget() {
    let shared = kir::cuda::emit_module(&[topk_with("4", "max_context / 4", SCORES).unwrap()]).unwrap();
    assert!(shared.contains("__shared__ float score[MAX_GROUPS];"), "a bound scratch is unused while the scores fit: {shared}");
    let k = topk_with("512", "262144 / 4", SCORES).unwrap();
    let src = kir::cuda::emit_module(&[k]).unwrap();
    assert!(src.contains("constexpr int MAX_GROUPS = 65536;"), "{src}");
    assert!(!src.contains("__shared__ float score[") && src.contains("float* score = scores + (size_t)row * MAX_GROUPS;"), "{src}");
    assert!(src.contains("score[g] = s * scale;") && src.contains("score[at] = -3.0e30f;"), "the same top-k runs over the scratch: {src}");
    assert!(topk_with("4", "262144 / 4", "").unwrap_err().contains("scores scratch"));
}

#[test]
fn the_group_rows_refuse_bad_shapes() {
    assert!(pool("dim", "15", true).unwrap_err().contains("even rot"));
    assert!(pool("2048", "rot", true).unwrap_err().contains("1024"));
    assert!(topk("4", "16384").unwrap_err().contains("12288"));
    assert!(topk("0", "64").unwrap_err().contains("positive"));
    assert!(topk("80", "64").unwrap_err().contains("more groups than it scores"));
}

#[test]
fn the_group_rows_have_no_wgsl_lowering() {
    assert!(pool_desc("dim", "rot", true).unwrap().build(&Target::wgpu_baseline(), &shape()).is_err());
}
