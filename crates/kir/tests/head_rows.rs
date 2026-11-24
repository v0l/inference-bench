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
"#;

fn shape() -> Shape {
    Shape::new().num("head_dim", 64.0).num("heads", 4.0).num("theta", 1.0e6)
}

fn rope(extra: &str) -> Result<Kernel, String> {
    let text = format!(
        r#"kind = "kir-kernel"
family = "span_rope"
name = "rope"
threads = 32

[args]
row = "heads * head_dim"
head = "head_dim"
offset = 0
rot = "head_dim"
{extra}

[[param]]
name = "x"
type = "f32"
role = "activation"
access = "write"
{POSITION}"#
    );
    Description::parse(&text)?.build(&Target::sm_120(), &shape())
}

fn rms(weighted: bool) -> Kernel {
    let w = if weighted { "\n[[param]]\nname = \"w\"\ntype = \"f32\"\nrole = \"weight\"\naccess = \"read\"\n" } else { "" };
    let text = format!(
        r#"kind = "kir-kernel"
family = "head_rms"
name = "rms"
threads = 64

[args]
row = "heads * head_dim"
head = "head_dim"
offset = 0
width = "head_dim"
eps = 1.0e-6

[[param]]
name = "x"
type = "f32"
role = "activation"
access = "write"
{w}"#
    );
    Description::parse(&text).unwrap().build(&Target::sm_120(), &shape()).unwrap()
}

fn emit(k: &Kernel) -> String {
    kir::cuda::emit_module(std::slice::from_ref(k)).unwrap().into_string()
}

#[test]
fn a_rope_without_theta_reads_the_model_rope_table() {
    let src = emit(&rope("").unwrap());
    assert!(src.contains("rope_inv_freq(i, layer)") && src.contains("i < 32;"), "{src}");
}

#[test]
fn an_explicit_theta_rotates_only_the_stated_angles_over_the_full_span() {
    let src = emit(&rope("theta = \"theta\"\nangles = 8").unwrap());
    assert!(src.contains("powf(1e6f, -(float)(2 * i) / (float)ROT_SPAN)"), "{src}");
    assert!(src.contains("i < 8;") && !src.contains("rope_inv_freq"), "{src}");
}

#[test]
fn rope_angles_are_checked() {
    assert!(rope("theta = 10000.0\nangles = 33").unwrap_err().contains("angles"));
    assert!(rope("angles = 4").unwrap_err().contains("explicit theta"));
}

#[test]
fn head_rms_scales_by_a_weight_only_when_it_has_one() {
    assert!(emit(&rms(true)).contains("v[i] *= inv * w[i];"));
    assert!(emit(&rms(false)).contains("v[i] *= inv;"));
}
