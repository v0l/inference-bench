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
    kir::cuda::emit_module(std::slice::from_ref(k)).unwrap().into_string()
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

fn block_scan(index: &str, topk: &str) -> String {
    format!(
        r#"kind = "kir-kernel"
family = "scan_attention"
name = "blocks_scan"
threads = 128

[geometry]
layers = 1
kv_rows = 1
k_width = 32
v_stride = 32
v_offset = 0
v_width = 32
out_row = 64
out_head = 32
out_offset = 0

[[query]]
src = "qh"
row = 64
head = 32
offset = 0
width = 32

[blocks]
cache = "comp"
layers = 2
ratio = "ratio"
topk = {topk}
{index}

[[param]]
name = "qh"
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
name = "out"
type = "f32"
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
name = "comp"
type = "bf16"
role = "activation"
access = "read"

[[param]]
name = "ratio"
type = "i32"
role = "value"
access = "read"

[[param]]
name = "idx"
type = "i32"
role = "activation"
access = "read"

[[stage]]
op = "epilogue"
name = "score"
expr = "s * scale"

[[stage]]
op = "mask"
expr = "k_pos > q_pos - 8"

[[stage]]
op = "online_softmax"
value = "score"
"#
    )
}

#[test]
fn a_block_source_scans_the_selected_compressed_entries_after_the_masked_window() {
    let k = build(&block_scan("index = \"idx\"", "3"));
    let src = emit(&k);
    assert!(src.contains("for (int r = warp; r < 3; r += SCAN_WARPS)") && src.contains("idx[(size_t)row * 3 + r]"), "{src}");
    assert!(src.contains("const int visible = ratio > 0 ? (q_pos_i + 1) / ratio : 0;"));
    assert!(src.contains("const int t = e * ratio + ratio - 1;"));
    let block = src.split("const int visible").nth(1).unwrap();
    assert!(!block.contains("continue;\n            float dot") || !block.contains("q_pos - 8"), "the window mask must not gate compressed entries");
    assert!(!src.split("const int visible").next().unwrap().is_empty());
}

#[test]
fn a_block_source_without_an_index_scans_every_visible_entry() {
    let src = emit(&build(&block_scan("", "0")));
    assert!(src.contains("for (int r = warp; r < visible; r += SCAN_WARPS)") && src.contains("const int e = r;"), "{src}");
}

#[test]
fn bad_block_sources_are_refused_with_the_kernel_name() {
    let t = Target::sm_120();
    let shape = Shape::new();
    let err = |text: &str| Description::parse(text).unwrap().build(&t, &shape).unwrap_err();
    assert!(err(&block_scan("index = \"idx\"", "0")).contains("positive topk"));
    assert!(err(&block_scan("index = \"missing\"", "3")).contains("block index"));
    assert!(err(&block_scan("", "0").replace("ratio = \"ratio\"", "ratio = \"nope\"")).contains("block ratio"));
    assert!(err(&block_scan("", "0").replace("cache = \"comp\"", "cache = \"nope\"")).contains("block cache"));
}

#[test]
fn a_geometry_scan_can_write_the_split_attention_planes() {
    let planes = "[[param]]\nname = \"out_hi\"\ntype = \"bf16\"\nrole = \"activation\"\naccess = \"write\"\n\n[[param]]\nname = \"out_lo\"\ntype = \"bf16\"\nrole = \"activation\"\naccess = \"write\"\n";
    let text = block_scan("", "0").replace("[[param]]\nname = \"out\"\ntype = \"f32\"\nrole = \"activation\"\naccess = \"write\"\n", planes);
    let src = emit(&build(&text));
    assert!(src.contains("split_hi_lo(res, hi, lo);") && src.contains("out_hi[at] = (bf16_t)hi;") && !src.contains("out[at]"), "{src}");
    assert!(emit(&build(&block_scan("", "0"))).contains("out[at] = res;"));
}
