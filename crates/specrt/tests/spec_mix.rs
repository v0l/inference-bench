use specrt::plan::{Op, Phase, Plan};
use specrt::spec::{Block, Spec};
use specrt::weights::Checkpoint;
use std::path::Path;

const MIXES: [&str; 3] = ["tiny-sandwich-scaled", "tiny-sandwich-pre-qknorm", "tiny-scaled-qknorm"];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn spec(name: &str) -> Spec {
    let mut s = Spec::load(&root().join("tests/fixtures").join(format!("{name}.toml"))).unwrap();
    s.repo = root().join("tests/fixtures").join(name).display().to_string();
    s
}

#[test]
fn each_mix_is_a_spec_file_over_existing_parameters() {
    let a = spec("tiny-sandwich-scaled");
    assert!(a.norm.sandwich && !a.norm.pre && a.norm.residual_scale == Some(0.22) && a.attention.qk_norm_full);
    let b = spec("tiny-sandwich-pre-qknorm");
    assert!(b.norm.sandwich && b.norm.pre && b.norm.residual_scale.is_none() && b.attention.qk_norm_full);
    let c = spec("tiny-scaled-qknorm");
    assert!(!c.norm.sandwich && c.norm.residual_scale == Some(0.22) && c.attention.qk_norm_full && c.dims.embed_scale.is_some());
    for name in MIXES {
        let s = spec(name);
        for phase in [Phase::Prefill, Phase::Decode] {
            let plan = Plan::build(&s, phase, true);
            let ops = plan.layer(Block::Dense);
            assert!(ops.contains(&Op::QkNorm), "{name}");
            let branch = ops.iter().filter(|o| matches!(o, Op::NormAdd { .. } | Op::ScaledAdd { .. })).count();
            assert_eq!(branch, 2, "{name} {phase:?}");
        }
    }
}

#[test]
fn a_scaled_sandwich_folds_the_residual_scale_into_the_branch_norms() {
    let scaled = spec("tiny-sandwich-scaled");
    let mut plain = scaled.clone();
    plain.norm.residual_scale = None;
    let ck = Checkpoint::open(Path::new(&scaled.repo)).unwrap();
    let (a, b) = (ck.layer(&scaled, 1).unwrap(), ck.layer(&plain, 1).unwrap());
    let scale = |v: &[f32]| v.iter().map(|w| (*w as f64 * 0.22) as f32).collect::<Vec<f32>>();
    assert_eq!(a.ln2, scale(&b.ln2));
    assert_eq!(a.ln_post_mlp.unwrap(), scale(&b.ln_post_mlp.unwrap()));
    let pre = spec("tiny-sandwich-pre-qknorm");
    let ck = Checkpoint::open(Path::new(&pre.repo)).unwrap();
    let mut scaled_pre = pre.clone();
    scaled_pre.norm.residual_scale = Some(0.5);
    let (a, b) = (ck.layer(&scaled_pre, 0).unwrap(), ck.layer(&pre, 0).unwrap());
    assert_eq!(a.ln1, b.ln1);
    assert_eq!(a.ln_pre_mlp, b.ln_pre_mlp);
    assert_eq!(a.ln2, b.ln2.iter().map(|w| (*w as f64 * 0.5) as f32).collect::<Vec<f32>>());
}
