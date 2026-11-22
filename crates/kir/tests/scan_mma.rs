use kir::desc::{Description, Shape};
use kir::expr::Expr;
use kir::{Body, Kernel, Target};

const SINKS: &str = include_str!("../../specrt/specs/attn_sinks.toml");
const CHUNKED: &str = include_str!("../../specrt/specs/attn_chunked.toml");
const GOLDEN: &str = "tests/golden/scan_mma_sinks.cu";

fn shape() -> Shape {
    Shape::new().num("heads", 4.0).num("kv_heads", 2.0).num("head_dim", 64.0).num("chunk", 0.0).num("max_context", 256.0)
}

fn build(text: &str, t: &Target) -> Kernel {
    Description::parse(text).unwrap().build(t, &shape()).unwrap()
}

fn emit(k: &Kernel) -> String {
    kir::cuda::emit_module(std::slice::from_ref(k)).unwrap()
}

fn tile_part(src: &str) -> &str {
    &src[src.find("_mma(").map(|i| src[..i].rfind("extern").unwrap()).expect("a tile kernel")..]
}

#[test]
fn the_sinks_scan_emits_a_tile_prefill_kernel_that_matches_the_pinned_golden() {
    let src = emit(&build(SINKS, &Target::sm_120()));
    let tile = tile_part(&src);
    assert!(tile.contains("attn_sinks_mma("));
    assert!(tile.contains("mma_kv") && tile.contains("ldsm_x4") && tile.contains("fast_exp2"));
    assert!(!tile.contains("warp_sum(part)"));
    if std::env::var("KIR_UPDATE_GOLDEN").is_ok() {
        std::fs::write(GOLDEN, tile).unwrap();
    }
    assert_eq!(tile, std::fs::read_to_string(GOLDEN).unwrap());
}

#[test]
fn a_target_without_mma_keeps_only_the_scalar_kernel() {
    let t = Target::cuda(7, 5, 80, 64 * 1024);
    let src = emit(&build(SINKS, &t));
    assert!(!src.contains("_mma(") && src.contains("attn_sinks("));
}

#[test]
fn a_scan_without_sinks_starts_from_an_empty_softmax() {
    let src = emit(&build(CHUNKED, &Target::sm_120()));
    let tile = tile_part(&src);
    assert!(tile.contains("float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;"));
    assert!(!tile.contains("sink_l2"));
}

#[test]
fn the_tile_stages_bind_the_score_to_the_fragment_and_the_mask_to_live() {
    let tile_src = emit(&build(SINKS, &Target::sm_120()));
    let tile = tile_part(&tile_src);
    assert!(tile.contains("float score = "), "{tile}");
    assert!(tile.contains("s[nb][e]") && tile.contains("(float)key"));
    assert!(tile.contains("live = live && ("));
    assert!(matches!(build(SINKS, &Target::sm_120()).body, Body::Scan(_)));
}

fn eval(e: &Expr, vars: &[(&str, f64)]) -> f64 {
    e.eval(&|n| vars.iter().find(|(k, _)| *k == n).map(|(_, v)| *v)).unwrap()
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
}

#[test]
fn the_base_two_tiled_online_softmax_matches_the_plain_softmax_with_a_sink() {
    let score = Expr::parse("select(softcap > 0.0, softcap * tanh(s * scale / softcap), s * scale)").unwrap();
    let mask = Expr::parse("window <= 0 || k_pos > q_pos - window").unwrap();
    let (scale, softcap, window, sink) = (0.125f64, 8.0f64, 24.0f64, 0.7f64);
    let mut seed = 7u64;
    let (dim, len) = (8usize, 100usize);
    let q: Vec<f64> = (0..dim).map(|_| lcg(&mut seed) * 4.0).collect();
    let k: Vec<Vec<f64>> = (0..len).map(|_| (0..dim).map(|_| lcg(&mut seed) * 4.0).collect()).collect();
    let v: Vec<Vec<f64>> = (0..len).map(|_| (0..dim).map(|_| lcg(&mut seed)).collect()).collect();
    let q_pos = (len - 1) as f64;
    let log2e = std::f64::consts::LOG2_E;
    let logit = |t: usize| {
        let dot: f64 = q.iter().zip(&k[t]).map(|(a, b)| a * b).sum();
        let vars = [("s", dot), ("scale", scale), ("softcap", softcap), ("window", window), ("k_pos", t as f64), ("q_pos", q_pos)];
        (eval(&mask, &vars) != 0.0).then(|| eval(&score, &vars))
    };
    let live: Vec<(usize, f64)> = (0..len).filter_map(|t| logit(t).map(|s| (t, s))).collect();
    let top = live.iter().map(|x| x.1).fold(sink, f64::max);
    let denom = (sink - top).exp() + live.iter().map(|x| (x.1 - top).exp()).sum::<f64>();
    let want: Vec<f64> = (0..dim).map(|d| live.iter().map(|(t, s)| (s - top).exp() * v[*t][d]).sum::<f64>() / denom).collect();
    let (mut m, mut l, mut acc) = (sink * log2e, 1.0f64, vec![0.0f64; dim]);
    for tile in (0..len).step_by(32) {
        let end = (tile + 32).min(len);
        let s: Vec<Option<f64>> = (tile..end).map(|t| logit(t).map(|x| x * log2e)).collect();
        let mx = s.iter().flatten().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let n = m.max(mx);
        let a = (m - n).exp2();
        let p: Vec<f64> = s.iter().map(|x| x.map_or(0.0, |x| (x - n).exp2())).collect();
        l = l * a + p.iter().sum::<f64>();
        for d in 0..dim {
            acc[d] = acc[d] * a + p.iter().enumerate().map(|(i, p)| p * v[tile + i][d]).sum::<f64>();
        }
        m = n;
    }
    for d in 0..dim {
        assert!((acc[d] / l - want[d]).abs() < 1e-9, "dim {d}: {} vs {}", acc[d] / l, want[d]);
    }
}
