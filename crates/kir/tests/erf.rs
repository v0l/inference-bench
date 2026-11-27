use kir::expr::{Expr, erf_poly};
use kir::layout::Dialect;

const TABLE: [(f64, f64); 9] = [
    (0.0, 0.0),
    (0.1, 0.1124629160182849),
    (0.5, 0.5204998778130465),
    (1.0, 0.8427007929497149),
    (1.5, 0.9661051464753108),
    (2.0, 0.9953222650189527),
    (2.5, 0.9995930479825550),
    (3.0, 0.9999779095030014),
    (4.0, 0.9999999845827421),
];

fn at(e: &Expr, x: f64) -> f64 {
    e.eval(&|n| (n == "x").then_some(x)).unwrap()
}

#[test]
fn erf_evaluates_to_the_table_on_the_host() {
    let e = Expr::parse("erf(x)").unwrap();
    for (x, want) in TABLE {
        assert!((at(&e, x) - want).abs() < 1e-13, "erf({x}) = {}", at(&e, x));
        assert!((at(&e, -x) + want).abs() < 1e-13, "erf(-{x})");
    }
}

#[test]
fn the_wgsl_polynomial_is_within_2e_7_of_the_table() {
    let p = erf_poly(Expr::Name("x".into()));
    for (x, want) in TABLE {
        assert!((at(&p, x) - want).abs() < 2e-7, "poly erf({x}) = {}", at(&p, x));
        assert!((at(&p, -x) + want).abs() < 2e-7, "poly erf(-{x})");
    }
}

#[test]
fn gelu_is_the_exact_erf_form() {
    let g = Expr::parse("gelu(x)").unwrap();
    for (x, want) in [(1.0, 0.8413447460685429), (-0.5, -0.15426876936299347), (2.0, 1.9544997361036416)] {
        assert!((at(&g, x) - want).abs() < 1e-12, "gelu({x}) = {}", at(&g, x));
    }
}

#[test]
fn elu_is_x_above_zero_and_exp_minus_one_below() {
    let e = Expr::parse("elu(x)").unwrap();
    for (x, want) in [(0.0, 0.0), (1.5, 1.5), (-1.0, (-1.0f64).exp() - 1.0), (-3.0, (-3.0f64).exp() - 1.0), (1e-3, 1e-3)] {
        assert!((at(&e, x) - want).abs() < 1e-15, "elu({x}) = {}", at(&e, x));
    }
    let lowered = kir::expr::apply_macro("elu", Expr::Raw("v".into())).unwrap();
    assert_eq!(lowered.lower(Dialect::Cuda, &[]).unwrap(), "(v > 0.0f ? v : expf(v) - 1.0f)");
    assert!(kir::expr::unary_macro("elu").is_some() && kir::expr::unary_macro("relu6").is_none());
}

#[test]
fn erf_lowers_to_erff_in_cuda_and_a_polynomial_in_wgsl() {
    let e = Expr::parse("erf(x * 0.5)").unwrap().subst(&|n| (n == "x").then(|| Expr::Raw("v".into()))).unwrap();
    assert_eq!(e.lower(Dialect::Cuda, &[]).unwrap(), "erff(v * 0.5f)");
    let w = e.lower(Dialect::Wgsl, &[]).unwrap();
    assert!(!w.contains("erf"), "{w}");
    let src = format!("fn f(v: f32) -> f32 {{ return {w}; }}");
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
}
