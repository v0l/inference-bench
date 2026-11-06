use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use std::path::Path;

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + i * 7 % 5000).collect()
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn seeded_sampling_reproduces_and_greedy_is_the_default() {
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let p = prompt(32);
    let hot = |seed| Sampling { temperature: 1.5, seed };
    let a = rt.generate(&p, 24, hot(1)).unwrap().tokens;
    let b = rt.generate(&p, 24, hot(1)).unwrap().tokens;
    let c = rt.generate(&p, 24, hot(2)).unwrap().tokens;
    assert_eq!(a, b);
    assert_ne!(a, c);
    let greedy = rt.generate(&p, 24, Sampling::GREEDY).unwrap().tokens;
    assert_eq!(greedy, rt.generate(&p, 24, Sampling::GREEDY).unwrap().tokens);
    assert_ne!(greedy, a);
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn contexts_above_4096_generate_and_match_the_short_context_path() {
    let p = prompt(4500);
    let mut long = Runtime::load(&spec(), 4700).unwrap();
    let g = long.generate(&p, 16, Sampling::GREEDY).unwrap();
    assert_eq!(g.tokens.len(), 16);
    let mut short = Runtime::load(&spec(), 4096).unwrap();
    let head = &p[..3000];
    let a = long.generate(head, 16, Sampling::GREEDY).unwrap().tokens;
    let b = short.generate(head, 16, Sampling::GREEDY).unwrap().tokens;
    assert_eq!(a, b);
}
