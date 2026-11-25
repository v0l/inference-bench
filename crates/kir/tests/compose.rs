use kir::compose::Stage;
use kir::desc::{COMPOSED, Description, Shape};
use kir::expr::Expr;
use kir::layout::Dialect;
use kir::{Activation, Body, Kernel, Pointwise, Target};

fn shape() -> Shape {
    Shape::new()
        .num("hidden", 1024.0)
        .num("inter", 3072.0)
        .num("vocab", 4096.0)
        .num("norm_eps", 1e-6)
        .num("residual_scale", 0.22)
        .num("logit_scale", 0.5)
        .num("logit_softcap", 30.0)
        .word("mlp_act", "silu")
}

fn composed(name: &str, t: &Target, s: &Shape) -> Kernel {
    Description::composed(name).unwrap().build(t, s).unwrap()
}

fn emit(k: &Kernel) -> String {
    match k.target.language {
        kir::Language::Cuda => kir::cuda::emit_module(std::slice::from_ref(k)).unwrap().into_string(),
        kir::Language::Wgsl => kir::wgsl::emit_kernel(k).unwrap(),
    }
}

fn cuda_targets() -> Vec<Target> {
    vec![Target::sm_120(), Target::sm_80()]
}

fn wgsl_targets() -> Vec<Target> {
    vec![Target::wgpu_baseline(), Target::wgpu_baseline().with_subgroups(32)]
}

#[test]
fn composed_rmsnorm_is_byte_identical_on_every_target() {
    let s = shape();
    for t in cuda_targets().into_iter().chain(wgsl_targets()) {
        let rust = Kernel::rmsnorm(&t, 1024, 1e-6, 1024).unwrap();
        let data = composed("rmsnorm", &t, &s);
        assert!(matches!(data.body, Body::Compose(_)));
        assert_eq!(emit(&data), emit(&rust), "{}", t.describe());
        assert_eq!((data.threads, data.smem_bytes, &data.params), (rust.threads, rust.smem_bytes, &rust.params));
    }
}

#[test]
fn composed_rmsnorm_offset_is_byte_identical() {
    let t = Target::sm_120();
    let mut d = Description::composed("rmsnorm").unwrap();
    d.consts[1].value = toml::Value::Float(1.0);
    let data = d.build(&t, &shape()).unwrap();
    assert_eq!(emit(&data), emit(&Kernel::rmsnorm_offset(&t, 1024, 1e-6, 1.0, 1024).unwrap()));
    let t = Target::wgpu_baseline().with_subgroups(32);
    let data = d.build(&t, &shape()).unwrap();
    assert_eq!(emit(&data), emit(&Kernel::rmsnorm_offset(&t, 1024, 1e-6, 1.0, 1024).unwrap()));
}

#[test]
fn composed_swiglu_is_byte_identical_for_both_activations() {
    for (word, act) in [("silu", Activation::Silu), ("gelu_tanh", Activation::GeluTanh), ("swiglu", Activation::Silu), ("geglu", Activation::GeluTanh)] {
        let s = shape().word("mlp_act", word);
        for t in cuda_targets().into_iter().chain(wgsl_targets()) {
            let rust = Kernel::gated_activation(&t, "swiglu", 3072, act, 256).unwrap();
            assert_eq!(emit(&composed("swiglu", &t, &s)), emit(&rust), "{word} {}", t.describe());
        }
    }
}

#[test]
fn composed_reduction_kernels_are_byte_identical_on_cuda() {
    let s = shape();
    for t in cuda_targets() {
        let cases = [
            ("norm_add", Kernel::norm_add(&t, 1024, 1e-6, 0.0, 1024).unwrap()),
            ("add_rmsnorm", Kernel::add_rmsnorm(&t, 1024, 1e-6, 0.0, 1024).unwrap()),
        ];
        for (name, rust) in cases {
            assert_eq!(emit(&composed(name, &t, &s)), emit(&rust), "{name} {}", t.describe());
        }
    }
}

#[test]
fn composed_pointwise_kernels_are_byte_identical_on_cuda() {
    let s = shape();
    for t in cuda_targets() {
        let cases = [
            ("residual_add", Kernel::add_scaled(&t, "residual_add", 1024, 0.22, 256).unwrap()),
            ("logit_softcap", Kernel::pointwise(&t, "logit_softcap", 4096, Pointwise::SoftCap(30.0), 256).unwrap()),
            ("logit_scale", Kernel::pointwise(&t, "logit_scale", 4096, Pointwise::Scale(0.5), 256).unwrap()),
            ("clear_branch", Kernel::pointwise(&t, "clear_branch", 1024, Pointwise::Fill(0.0), 256).unwrap()),
        ];
        for (name, rust) in cases {
            assert_eq!(emit(&composed(name, &t, &s)), emit(&rust), "{name} {}", t.describe());
        }
    }
}

#[test]
fn every_composed_description_builds_and_emits_on_cuda() {
    for (name, _) in COMPOSED {
        let k = composed(name, &Target::sm_120(), &shape());
        assert_eq!(k.hash(), composed(name, &Target::sm_120(), &shape()).hash());
        assert!(emit(&k).contains("__global__"), "{name}");
    }
}

#[test]
fn composed_hash_follows_the_resolved_shape() {
    let t = Target::sm_120();
    let a = composed("rmsnorm", &t, &shape());
    let b = composed("rmsnorm", &t, &shape().num("hidden", 2048.0));
    let c = composed("rmsnorm", &Target::sm_80(), &shape());
    assert_ne!(a.hash(), b.hash());
    assert_ne!(a.hash(), c.hash());
}

#[test]
fn a_new_kernel_needs_only_toml() {
    let text = r#"
kind = "kir-kernel"
family = "compose"
name = "clamped_swiglu"
threads = 256
[args]
width = 64
[rows]
iter = "grid"
address = "advance"
per_row = ["gate_up*2", "out"]
[[param]]
name = "gate_up"
role = "activation"
access = "read"
[[param]]
name = "out"
role = "activation"
access = "write"
[[const]]
name = "limit"
value = 7.0
[[const]]
name = "alpha"
value = 1.702
[[stage]]
op = "tile_load"
name = "g"
src = "gate_up"
layout = "(width,2):(2,1)"
at = ["i", "0"]
[[stage]]
op = "tile_load"
name = "u"
src = "gate_up"
layout = "(width,2):(2,1)"
at = ["i", "1"]
[[stage]]
op = "epilogue"
name = "y"
expr = "min(g, limit) * sigmoid(alpha * min(g, limit)) * (max(min(u, limit), -limit) + 1.0)"
[[stage]]
op = "tile_store"
dst = "out"
value = "y"
"#;
    let d = Description::parse(text).unwrap();
    let cuda = emit(&d.build(&Target::sm_120(), &Shape::new()).unwrap());
    assert!(cuda.contains("fminf(gate_up[i * 2], 7e0f)"), "{cuda}");
    assert!(cuda.contains("expf(-(1.702e0f * fminf"), "{cuda}");
    let wgsl = emit(&d.build(&Target::wgpu_baseline(), &Shape::new()).unwrap());
    assert!(wgsl.contains("min(gate_up[t * 2u * WIDTH + i * 2u], 7e0)"), "{wgsl}");
    naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&wgsl)));
}

#[test]
fn composed_wgsl_validates() {
    for t in wgsl_targets() {
        for name in ["rmsnorm", "swiglu"] {
            let src = emit(&composed(name, &t, &shape()));
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        }
    }
}

#[test]
fn expressions_lower_to_both_dialects() {
    let e = Expr::parse("cap * tanh(v / cap) + select(v < 0.0, 1.0, 2.0)").unwrap();
    let bound = e.subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap();
    assert_eq!(bound.lower(Dialect::Cuda, &[]).unwrap(), "cap * tanhf(v / cap) + (v < 0.0f ? 1.0f : 2.0f)");
    assert_eq!(bound.lower(Dialect::Wgsl, &[]).unwrap(), "cap * tanh(v / cap) + select(2.0, 1.0, v < 0.0)");
    let e = Expr::parse("-(a + b) * c - (d - e)").unwrap().subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap();
    assert_eq!(e.lower(Dialect::Cuda, &[]).unwrap(), "-(a + b) * c - (d - e)");
    let e = Expr::parse("sigmoid(x)").unwrap().subst(&|n| Some(Expr::Raw(n.to_string()))).unwrap();
    assert_eq!(e.lower(Dialect::Wgsl, &[]).unwrap(), "1.0 / (1.0 + exp(-x))");
}

#[test]
fn expression_errors_name_the_problem() {
    for (text, want) in [("1 +", "unexpected"), ("foo(1)", "unknown function"), ("tanh(1, 2)", "takes 1"), ("(1", "missing )"), ("1 $ 2", "unexpected"), ("1 2", "trailing")] {
        let e = Expr::parse(text).unwrap_err();
        assert!(e.contains(want), "{text}: {e}");
    }
}

fn first(name: &str) -> Description {
    Description::composed(name).unwrap()
}

#[test]
fn bad_compose_descriptions_are_refused_at_build() {
    let t = Target::sm_120();
    let s = shape();
    let mut d = first("rmsnorm");
    d.stage.pop();
    assert!(d.build(&t, &s).unwrap_err().contains("tile_store"));
    let mut d = first("rmsnorm");
    d.stage[0].src = Some("nope".into());
    assert!(d.build(&t, &s).unwrap_err().contains("unknown param"));
    let mut d = first("rmsnorm");
    d.stage[3].name = Some("v".into());
    assert!(d.build(&t, &s).unwrap_err().contains("defined twice"));
    let mut d = first("rmsnorm");
    d.stage[1].expr = Some("v * missing".into());
    assert!(d.build(&t, &s).unwrap_err().contains("missing"));
    let mut d = first("rmsnorm");
    d.stage[0].op = "mma".into();
    assert!(d.build(&t, &s).unwrap_err().contains("unknown stage op"));
    let mut d = first("rmsnorm");
    d.rows = None;
    assert!(d.build(&t, &s).unwrap_err().contains("rows"));
    let mut d = first("swiglu");
    d.stage[0].layout = Some("(width,2):(2)".into());
    assert!(d.build(&t, &s).is_err());
    let mut d = first("rmsnorm");
    d.consts[0].value = toml::Value::String("absent".into());
    assert!(d.build(&t, &s).unwrap_err().contains("absent"));
    let mut d = first("rmsnorm");
    d.args.insert("bogus".into(), toml::Value::Integer(1));
    assert!(d.build(&t, &s).is_err());
    let mut d = first("rmsnorm");
    d.family = "rmsnorm".into();
    assert!(d.build(&t, &s).unwrap_err().contains("only family compose"));
}

#[test]
fn a_block_sum_the_target_cannot_run_is_refused() {
    let tiny = Target::wgpu(1024, 1024);
    let e = Description::composed("rmsnorm").unwrap().build(&tiny, &shape()).unwrap_err();
    assert!(e.contains("shared memory"), "{e}");
}

#[test]
fn stages_are_data() {
    let k = composed("rmsnorm", &Target::sm_120(), &shape());
    let Body::Compose(c) = &k.body else { panic!() };
    assert_eq!(c.stages.iter().filter(|s| matches!(s, Stage::BlockSum { .. })).count(), 1);
    assert!(c.has_sum());
}

const GATHER: &str = r#"
kind = "kir-kernel"
family = "compose"
name = "embed_rows"
threads = 256
[args]
width = 64
[rows]
iter = "grid"
address = "advance"
per_row = ["out"]
[[param]]
name = "ids"
type = "i32"
role = "activation"
access = "read"
[[param]]
name = "table"
role = "weight"
access = "read"
[[param]]
name = "out"
role = "activation"
access = "write"
[[stage]]
op = "gather"
name = "row"
src = "table"
index = "ids"
[[stage]]
op = "epilogue"
name = "y"
expr = "row * 2.0"
[[stage]]
op = "tile_store"
dst = "out"
value = "y"
"#;

#[test]
fn a_gather_stage_reads_table_rows_by_an_index_buffer_from_toml_alone() {
    let d = Description::parse(GATHER).unwrap();
    let cuda = emit(&d.build(&Target::sm_120(), &Shape::new()).unwrap());
    assert!(cuda.contains("const int* ids"), "{cuda}");
    assert!(cuda.contains("float row = table[(size_t)ids[blockIdx.y] * WIDTH + i];"), "{cuda}");
}

#[test]
fn a_scatter_stage_writes_rows_by_an_index_buffer() {
    let text = GATHER
        .replace("name = \"embed_rows\"", "name = \"scatter_rows\"")
        .replace("per_row = [\"out\"]", "per_row = [\"rows\"]")
        .replace("name = \"table\"\nrole = \"weight\"", "name = \"rows\"\nrole = \"activation\"")
        .replace("op = \"gather\"\nname = \"row\"\nsrc = \"table\"\nindex = \"ids\"", "op = \"tile_load\"\nname = \"row\"\nsrc = \"rows\"")
        .replace("op = \"tile_store\"\ndst = \"out\"\nvalue = \"y\"", "op = \"scatter\"\ndst = \"out\"\nvalue = \"y\"\nindex = \"ids\"");
    let d = Description::parse(&text).unwrap();
    let cuda = emit(&d.build(&Target::sm_120(), &Shape::new()).unwrap());
    assert!(cuda.contains("out[(size_t)ids[blockIdx.y] * WIDTH + i] = "), "{cuda}");
}

#[test]
fn indexed_access_checks_the_index_buffer_and_has_no_wgsl_lowering() {
    let bad = GATHER.replace("name = \"ids\"\ntype = \"i32\"", "name = \"ids\"\ntype = \"f32\"");
    assert!(Description::parse(&bad).unwrap().build(&Target::sm_120(), &Shape::new()).unwrap_err().contains("i32"));
    let wgsl = Description::parse(GATHER).unwrap().build(&Target::wgpu_baseline(), &Shape::new());
    assert!(wgsl.is_err(), "{wgsl:?}");
}

const MODULATE: &str = include_str!("../../specrt/specs/dit_modulate.toml");

#[test]
fn two_block_sums_give_a_layernorm_in_three_passes() {
    let k = Description::parse(MODULATE).unwrap().build(&Target::sm_120(), &Shape::new().num("hidden", 64.0).num("norm_eps", 1e-6)).unwrap();
    let cuda = emit(&k);
    assert_eq!(cuda.matches("kir_block_sum(ss, scratch)").count(), 2, "{cuda}");
    assert!(cuda.contains("float total = kir_block_sum(ss, scratch);"), "{cuda}");
    assert!(cuda.contains("float mean = total / WIDTH;"), "{cuda}");
    assert!(cuda.contains("    ss = 0.0f;\n"), "{cuda}");
    let mean_at = cuda.find("float mean").unwrap();
    let second = cuda.find("ss += (x[i] - mean)").unwrap();
    assert!(mean_at < second, "{cuda}");
    assert!(cuda.contains("out_hi += (size_t)blockIdx.x * 64;"), "{cuda}");
    assert!(cuda.contains("kir_split_bf16((x[i] - mean) * inv * (1.0f + scale[i]) + shift[i], out_hi[i], out_lo[i]);"), "{cuda}");
    assert!(cuda.contains("void kir_split_bf16("), "{cuda}");
    assert!(cuda.contains("unsigned short* out_hi"), "{cuda}");
}

#[test]
fn several_block_sums_and_split_stores_have_no_wgsl_lowering() {
    let e = Description::parse(MODULATE).unwrap().build(&Target::wgpu_baseline(), &Shape::new().num("hidden", 64.0).num("norm_eps", 1e-6)).unwrap_err();
    assert!(e.contains("CUDA"), "{e}");
}

#[test]
fn index_stages_name_the_column_and_the_row() {
    let text = GATHER
        .replace("op = \"gather\"\nname = \"row\"\nsrc = \"table\"\nindex = \"ids\"", "op = \"index\"\nname = \"col\"\n[[stage]]\nop = \"index\"\nname = \"row\"")
        .replace("expr = \"row * 2.0\"", "expr = \"cos(row * col)\"");
    let cuda = emit(&Description::parse(&text).unwrap().build(&Target::sm_120(), &Shape::new()).unwrap());
    assert!(cuda.contains("out[i] = cosf((float)blockIdx.y * (float)i);"), "{cuda}");
    let bad = text.replace("name = \"col\"", "name = \"lane_id\"");
    assert!(Description::parse(&bad).unwrap().build(&Target::sm_120(), &Shape::new()).unwrap_err().contains("col or row"));
}
