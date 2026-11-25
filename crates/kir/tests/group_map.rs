use kir::Target;
use kir::desc::{Description, Shape};

fn build(text: &str, shape: &Shape) -> Result<String, String> {
    let k = Description::parse(text)?.build(&Target::sm_120(), shape)?;
    Ok(kir::cuda::emit_module(std::slice::from_ref(&k)).map_err(|e| e.to_string())?.to_string())
}

const HEAD: &str = r#"
kind = "kir-kernel"
family = "group_map"
name = "probe"
threads = 32

[args]
width = 8
"#;

fn param(name: &str, ty: &str, role: &str, access: &str) -> String {
    format!("\n[[param]]\nname = \"{name}\"\ntype = \"{ty}\"\nrole = \"{role}\"\naccess = \"{access}\"\n")
}

#[test]
fn a_stepped_load_reads_one_stream_column_at_a_time() {
    let text = format!("{HEAD}{}{}\n[[stage]]\nop = \"load\"\nname = \"v\"\nsrc = \"x\"\nrow = 32\nhead = 1\nstep = 4\n\n[[stage]]\nop = \"store\"\ndst = \"y\"\nvalue = \"v\"\nrow = 8\n", param("x", "f32", "activation", "read"), param("y", "f32", "activation", "write"));
    let src = build(&text, &Shape::new()).unwrap();
    assert!(src.contains("x[(size_t)row * 32 + (size_t)grp * 1 + (size_t)i * 4]"), "{src}");
}

#[test]
fn a_pick_reads_one_scalar_per_row_and_group() {
    let text = format!("{HEAD}{}{}\n[[stage]]\nop = \"pick\"\nname = \"c\"\nsrc = \"x\"\nrow = 16\nhead = 4\noffset = 1\n\n[[stage]]\nop = \"store\"\ndst = \"y\"\nvalue = \"c * 2.0\"\nrow = 8\n", param("x", "f32", "activation", "read"), param("y", "f32", "activation", "write"));
    let src = build(&text, &Shape::new()).unwrap();
    assert!(src.contains("const float c = x[(size_t)row * 16 + (size_t)grp * 4 + 1];"), "{src}");
    assert!(src.contains("y[(size_t)row * 8 + i] = c * 2.0f;"), "{src}");
}

#[test]
fn a_limited_gather_reads_row_zero_for_ids_past_the_table() {
    let text = format!("{HEAD}{}{}{}\n[[stage]]\nop = \"gather_row\"\nname = \"t\"\nsrc = \"table\"\nindex = \"idx\"\nrow = 8\nlimit = \"cap\"\n\n[[stage]]\nop = \"store\"\ndst = \"y\"\nvalue = \"t\"\nrow = 8\n", param("idx", "i32", "activation", "read"), param("table", "f32", "weight", "read"), param("y", "f32", "activation", "write"));
    let src = build(&text, &Shape::new().num("cap", 100.0)).unwrap();
    assert!(src.contains("(idx[row] < 100 ? idx[row] : 0)"), "{src}");
    let bad = text.replace("op = \"gather_row\"", "op = \"load\"").replace("index = \"idx\"\n", "");
    assert!(build(&bad, &Shape::new().num("cap", 100.0)).is_err(), "a limit on a plain load is refused");
}

#[test]
fn a_plane_store_splits_into_bf16_hi_and_lo() {
    let text = format!("{HEAD}{}{}{}\n[[stage]]\nop = \"load\"\nname = \"v\"\nsrc = \"x\"\nrow = 8\n\n[[stage]]\nop = \"store\"\ndst = \"out\"\nvalue = \"v + 1.0\"\nrow = 8\nplanes = true\n", param("x", "f32", "activation", "read"), param("out_hi", "bf16", "activation", "write"), param("out_lo", "bf16", "activation", "write"));
    let src = build(&text, &Shape::new()).unwrap();
    assert!(src.contains("split_hi_lo(x[(size_t)row * 8 + i] + 1.0f, hi, lo);"), "{src}");
    assert!(src.contains("out_hi[(size_t)row * 8 + i] = (bf16_t)hi;"), "{src}");
    let missing = text.replace(&param("out_lo", "bf16", "activation", "write"), "");
    assert!(build(&missing, &Shape::new()).is_err());
}

#[test]
fn probit_matches_the_normal_quantiles() {
    let p = |x: f64| kir::expr::Expr::parse(&format!("probit({x})")).unwrap().eval(&|_| None).unwrap();
    for (x, want) in [(0.5, 0.0), (0.95, 1.6448536269514722), (0.975, 1.959963984540054), (0.01, -2.3263478740408408), (0.999, 3.090232306167813)] {
        assert!((p(x) - want).abs() < 1e-8, "probit({x}) = {}, want {want}", p(x));
    }
    assert!(kir::expr::Expr::parse("probit(1.0)").unwrap().eval(&|_| None).is_none());
}
