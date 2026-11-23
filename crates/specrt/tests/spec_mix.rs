use specrt::plan::{Op, Phase, Plan};
use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Block, Spec};
use specrt::weights::Checkpoint;
use std::path::{Path, PathBuf};

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

struct Reference {
    tokens: Vec<u32>,
    greedy: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn reference(name: &str) -> Reference {
    let dir = std::env::var_os("SPECRT_REF_DIR").map(PathBuf::from).unwrap_or_else(|| root().join("out"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.ref.json"))).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
    let bytes = std::fs::read(dir.join(format!("{name}.ref.bin"))).unwrap();
    Reference { tokens: ids("tokens"), greedy: ids("greedy"), vocab: meta["vocab"].as_u64().unwrap() as usize, logits: bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() }
}

struct Run {
    prompt: Vec<f32>,
    stepped: Vec<f32>,
    greedy: Vec<u32>,
}

fn run(s: &Spec, want: &Reference) -> Run {
    let mut rt = Runtime::load(s, 256).unwrap();
    let prompt = rt.prompt_logits(&want.tokens).unwrap();
    let stepped = rt.prefill_then_decode(&want.tokens[..64], &want.tokens[64..]).unwrap();
    let greedy = rt.generate(&want.tokens, want.greedy.len(), Sampling::GREEDY).unwrap().tokens;
    Run { prompt, stepped, greedy }
}

fn top(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
#[ignore = "needs a CUDA GPU and scripts/tiny_families.py output in SPECRT_REF_DIR"]
fn spec_only_mixes_match_the_torch_reference_with_and_without_the_store_epilogue() {
    for name in MIXES {
        let want = reference(name);
        let mut s = spec(name);
        s.runtime.split_activations = Some(true);
        let off = run(&s, &want);
        s.runtime.store_epilogue = true;
        let on = run(&s, &want);
        assert_eq!(bits(&on.prompt), bits(&off.prompt), "{name} prompt logits");
        assert_eq!(bits(&on.stepped), bits(&off.stepped), "{name} decode logits");
        assert_eq!(on.greedy, off.greedy, "{name} greedy");
        assert_eq!(off.prompt.len(), want.logits.len());
        let max_abs = off.prompt.iter().zip(&want.logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let v = want.vocab;
        let decode_abs = (0..want.tokens.len() - 64).map(|r| off.stepped[(r + 1) * v..(r + 2) * v].iter().zip(&want.logits[(64 + r) * v..(65 + r) * v]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
        eprintln!("{name} prompt max_abs {max_abs} decode max_abs {decode_abs}");
        assert!(max_abs < 2e-3 && decode_abs < 2e-3, "{name} max abs diff {max_abs} decode {decode_abs}");
        for r in 0..want.tokens.len() {
            assert_eq!(top(&off.prompt[r * v..(r + 1) * v]), top(&want.logits[r * v..(r + 1) * v]), "{name} row {r}");
        }
        assert_eq!(off.greedy, want.greedy, "{name} greedy");
    }
}
