use kir::desc::{Description, Shape};
use kir::expr::Expr;
use kir::layout::Dialect;
use kir::{Body, Kernel, Target};


const SINKS: &str = r#"kind = "kir-kernel"
family = "scan_attention"
name = "attn_sinks"
threads = 128

[[param]]
name = "qkv"
type = "f32"
role = "activation"
access = "read"

[[param]]
name = "k_cache"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "v_cache"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "block_table"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "pos"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "slots"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "out_hi"
type = "bf16"
role = "activation"
access = "write"

[[param]]
name = "out_lo"
type = "bf16"
role = "activation"
access = "write"

[[param]]
name = "layer"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "pages"
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
name = "sinks"
type = "f32"
role = "weight"
access = "read"

[[param]]
name = "window"
type = "i32"
role = "value"
access = "read"

[[stage]]
op = "epilogue"
name = "score"
expr = "s * scale"

[[stage]]
op = "mask"
expr = "window <= 0 || k_pos > q_pos - window"

[[stage]]
op = "online_softmax"
value = "score"
sink = "sinks"
"#;
const CHUNKED: &str = r#"kind = "kir-kernel"
family = "scan_attention"
name = "attn_chunked"
threads = 128

[[param]]
name = "qkv"
type = "f32"
role = "activation"
access = "read"

[[param]]
name = "k_cache"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "v_cache"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "block_table"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "pos"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "slots"
type = "i32"
role = "activation"
access = "read"

[[param]]
name = "out_hi"
type = "bf16"
role = "activation"
access = "write"

[[param]]
name = "out_lo"
type = "bf16"
role = "activation"
access = "write"

[[param]]
name = "layer"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "pages"
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
name = "chunk"
type = "i32"
role = "value"
access = "read"

[[stage]]
op = "epilogue"
name = "score"
expr = "s * scale"

[[stage]]
op = "mask"
expr = "floor(k_pos / chunk) == floor(q_pos / chunk)"

[[stage]]
op = "online_softmax"
value = "score"
"#;

fn build(text: &str) -> Kernel {
    Description::parse(text).unwrap().build(&Target::sm_120(), &Shape::new()).unwrap()
}

fn emit(k: &Kernel) -> String {
    kir::cuda::emit_module(std::slice::from_ref(k)).unwrap()
}

#[test]
fn comparison_and_logic_operators_lower_with_c_precedence() {
    let e = Expr::parse("a == 1 && b != 2 || c >= 3").unwrap();
    let text = e.subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap().lower(Dialect::Cuda, &[]).unwrap();
    assert_eq!(text, "a == 1.0f && b != 2.0f || c >= 3.0f");
    let grouped = Expr::parse("a && (b || c)").unwrap().subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap().lower(Dialect::Cuda, &[]).unwrap();
    assert_eq!(grouped, "a && (b || c)");
    let wgsl = Expr::parse("floor(x / 4)").unwrap().subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap().lower(Dialect::Wgsl, &[]).unwrap();
    assert_eq!(wgsl, "floor(x / 4.0)");
}

#[test]
fn the_sinks_description_builds_a_scan_with_the_sink_in_the_denominator() {
    let k = build(SINKS);
    assert!(matches!(k.body, Body::Scan(_)));
    let src = emit(&k);
    assert!(src.contains("attn_sinks("), "{src}");
    assert!(src.contains("float sink_logit = sinks[head];"), "{src}");
    assert!(src.contains("float denom = expf(sink_logit - top);"), "{src}");
    assert!(src.contains("if (!(window <= 0.0f || (float)t > q_pos_i - window))") || src.contains("continue;"), "{src}");
}

#[test]
fn the_chunked_description_masks_by_chunk_and_scales_the_query() {
    let src = emit(&build(CHUNKED));
    assert!(src.contains("attn_chunked("), "{src}");
    assert!(src.contains("continue;"), "{src}");
    assert!(src.contains("floorf((float)t / (float)chunk)"), "{src}");
    assert!(src.contains("float denom = 0.0f;"), "{src}");
    assert!(!src.contains("sink_logit"), "{src}");
}

#[test]
fn the_hash_follows_the_stages() {
    let a = build(SINKS);
    let b = build(&SINKS.replace("s * scale", "s * scale * 0.5"));
    assert_ne!(a.hash(), b.hash());
    assert_eq!(a.hash(), build(SINKS).hash());
}

#[test]
fn bad_scan_descriptions_are_refused_with_the_kernel_name() {
    let t = Target::sm_120();
    let attempt = |text: String| Description::parse(&text).unwrap().build(&t, &Shape::new()).unwrap_err();
    let e = attempt(SINKS.replace("sink = \"sinks\"", "sink = \"window\""));
    assert!(e.contains("attn_sinks") && e.contains("sink"), "{e}");
    let e = attempt(SINKS.replace("value = \"score\"", "value = \"nothing\""));
    assert!(e.contains("attn_sinks") && e.contains("nothing"), "{e}");
    let e = attempt(SINKS.replace("name = \"pages\"", "name = \"pagez\""));
    assert!(e.contains("pages"), "{e}");
    let e = attempt(SINKS.replace("threads = 128", "threads = 16"));
    assert!(e.contains("multiple"), "{e}");
    let e = attempt(SINKS.replace("op = \"mask\"", "op = \"shuffle\""));
    assert!(e.contains("shuffle"), "{e}");
    let e = attempt(format!("{SINKS}\n[[stage]]\nop = \"epilogue\"\nname = \"late\"\nexpr = \"1.0\"\n"));
    assert!(e.contains("online_softmax"), "{e}");
    let e = Description::parse(&SINKS.replace("window <= 0", "nothing_here")).unwrap().build(&Target::wgpu_baseline(), &Shape::new()).unwrap_err();
    assert!(e.contains("WarpShuffle"), "{e}");
}
