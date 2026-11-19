use specrt::codegen::{Module, described_kernels};
use specrt::kir_kernel::Registry;
use specrt::plan::{Op, Phase, Plan};
use specrt::spec::{Block, EMBEDDED, Spec};
use std::path::Path;

fn params(extra: &str) -> String {
    let base = [
        ("qkv", "f32", "activation", "read"),
        ("k_cache", "bf16", "activation", "read"),
        ("v_cache", "bf16", "activation", "read"),
        ("block_table", "i32", "activation", "read"),
        ("pos", "i32", "activation", "read"),
        ("slots", "i32", "activation", "read"),
        ("out_hi", "bf16", "activation", "write"),
        ("out_lo", "bf16", "activation", "write"),
        ("layer", "i32", "value", "read"),
        ("pages", "i32", "value", "read"),
        ("pos_step", "i32", "value", "read"),
        ("row_step", "i32", "value", "read"),
    ];
    let mut out = String::new();
    for (n, t, r, a) in base {
        out.push_str(&format!("[[param]]\nname = \"{n}\"\ntype = \"{t}\"\nrole = \"{r}\"\naccess = \"{a}\"\n\n"));
    }
    out + extra
}

fn probe() -> String {
    let extra = "[[param]]\nname = \"bias\"\ntype = \"f32\"\nrole = \"weight\"\naccess = \"read\"\n\n[[param]]\nname = \"cap\"\ntype = \"i32\"\nrole = \"value\"\naccess = \"read\"\n\n";
    format!(
        "kind = \"kir-kernel\"\nfamily = \"scan_attention\"\nname = \"probe_scan\"\nthreads = 128\n\n{}[[stage]]\nop = \"epilogue\"\nname = \"score\"\nexpr = \"s * scale\"\n\n[[stage]]\nop = \"mask\"\nexpr = \"cap <= 0 || k_pos > q_pos - cap\"\n\n[[stage]]\nop = \"online_softmax\"\nvalue = \"score\"\nsink = \"bias\"\n",
        params(extra)
    )
}

fn install() {
    let texts: Vec<String> = EMBEDDED.iter().map(|(_, t)| t.to_string()).chain([probe()]).collect();
    Registry::install(Registry::from_texts(texts.iter().map(String::as_str)).unwrap());
}

fn spec_with(nodes: &str) -> Spec {
    let base = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
    Spec::parse(&format!("{base}\n{nodes}")).unwrap_or_else(|e| panic!("{e:#}"))
}

const NODES: &str = r#"
[[attention.node]]
op = "qk_rope"
phase = "prefill"

[[attention.node]]
op = "kernel"
kernel = "probe_scan"
phase = "prefill"
grid = ["heads", "rows"]
[attention.node.bind]
bias = { weight = "sinks", len = "heads" }
cap = "window"

[[attention.node]]
op = "decode_rope"
phase = "rows"

[[attention.node]]
op = "kernel"
kernel = "probe_scan"
phase = "rows"
grid = ["heads", "rows"]
[attention.node.bind]
bias = { weight = "sinks", len = "heads" }
cap = "window"
"#;

#[test]
fn attention_nodes_replace_the_stock_attention_ops_in_every_phase() {
    let spec = spec_with(NODES);
    let prefill = Plan::build(&spec, Phase::Prefill, true);
    let ops = prefill.layer(Block::Dense);
    assert!(ops.contains(&Op::QkRope) && ops.contains(&Op::Kernel { node: 1 }), "{ops:?}");
    assert!(!ops.contains(&Op::Flash) && !ops.contains(&Op::DecodeAttention) && !ops.contains(&Op::DecodeRope), "{ops:?}");
    let decode = Plan::build(&spec, Phase::Decode, true);
    let ops = decode.layer(Block::Dense);
    assert!(ops.contains(&Op::DecodeRope) && ops.contains(&Op::Kernel { node: 3 }), "{ops:?}");
    assert!(!ops.contains(&Op::Flash) && !ops.contains(&Op::QkRope) && !ops.contains(&Op::DecodeAttention), "{ops:?}");
}

#[test]
fn a_spec_without_nodes_keeps_the_stock_plan() {
    let spec = spec_with("");
    let decode = Plan::build(&spec, Phase::Decode, true);
    assert!(decode.layer(Block::Dense).contains(&Op::DecodeAttention));
    let prefill = Plan::build(&spec, Phase::Prefill, true);
    assert!(prefill.layer(Block::Dense).contains(&Op::Flash));
}

#[test]
fn a_kernel_node_lists_its_weight_tensors_and_kernel() {
    let spec = spec_with(NODES);
    let target = kir::Target::sm_120();
    let op = Op::Kernel { node: 1 };
    assert_eq!(op.tensors(&spec), vec!["self_attn.sinks".to_string()]);
    assert_eq!(op.kernels(&spec, 0, 1, &target), vec!["probe_scan".to_string()]);
}

#[test]
fn the_module_carries_the_described_kernel_once() {
    install();
    let spec = spec_with(NODES);
    let module = Module::generate_kv(&spec, 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.kernels.iter().filter(|k| **k == "probe_scan").count() == 1);
    assert_eq!(module.source.matches("void __launch_bounds__(128) probe_scan(").count(), 1);
    let stock = Module::generate_kv(&spec_with(""), 256, 1, &kir::Target::sm_120()).unwrap();
    assert_ne!(module.kir_hash, stock.kir_hash);
}

#[test]
fn bad_nodes_are_refused_when_the_spec_loads_or_the_module_builds() {
    install();
    let base = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
    let parse = |nodes: &str| Spec::parse(&format!("{base}\n{nodes}")).map(|_| ()).unwrap_err().to_string();
    assert!(parse("[[attention.node]]\nop = \"flash\"\n").contains("not qk_rope"));
    assert!(parse("[[attention.node]]\nop = \"qk_rope\"\n").contains("no kernel node"));
    assert!(parse("[[attention.node]]\nop = \"kernel\"\ngrid = [\"heads\"]\n").contains("without `kernel`"));
    assert!(parse("[[attention.node]]\nop = \"kernel\"\nkernel = \"k\"\ngrid = [\"bogus\"]\n").contains("grid symbol"));
    let target = kir::Target::sm_120();
    let unknown = spec_with(&NODES.replace("probe_scan", "no_such_kernel"));
    let e = described_kernels(&unknown, 1, 256, &target).unwrap_err().to_string();
    assert!(e.contains("no_such_kernel"), "{e}");
    let unbound = spec_with(&NODES.replace("cap = \"window\"", "cap2 = \"window\""));
    let e = described_kernels(&unbound, 1, 256, &target).unwrap_err().to_string();
    assert!(e.contains("cap"), "{e}");
    let stray = spec_with(&NODES.replace("cap = \"window\"", "cap = \"nonsense\""));
    let e = described_kernels(&stray, 1, 256, &target).unwrap_err().to_string();
    assert!(e.contains("nonsense"), "{e}");
}

#[test]
fn nodes_round_trip_through_the_spec_text() {
    let spec = spec_with(NODES);
    let text = toml::to_string(&spec).unwrap();
    assert_eq!(Spec::parse(&text).unwrap(), spec);
}
