use specrt::allreduce::Transport;
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::sched::{Config, Event, Executor, GpuExecutor, Request, Scheduler};
use specrt::spec::Spec;
use specrt::tp_exec::TpExecutor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const NEW: usize = 64;

fn fixture(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn device() -> usize {
    std::env::var("SPECRT_TP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn spec_for(dir: &str, serving: bool) -> Spec {
    let json = std::fs::read_to_string(fixture(dir).join("config.json")).unwrap();
    let mut s = if serving { Spec::from_hf_config_serving(&json).unwrap() } else { Spec::from_hf_config(&json).unwrap() };
    s.name = dir.into();
    s.repo = fixture(dir).display().to_string();
    s.vision = None;
    s
}

fn layout(spec: &Spec, slots: usize) -> Layout {
    let mut l = Layout::serving(spec, 512, slots, None);
    l.kv_blocks = slots * 512usize.div_ceil(specrt::codegen::PAGE) + 4;
    l.prefill_chunk = 128;
    l.kv_headroom = 0;
    l
}

fn prompts(vocab: u32) -> Vec<Vec<u32>> {
    (0..6u32).map(|i| (0..(7 + 23 * i) as usize).map(|j| (j as u32 * 37 + i * 101 + 5) % vocab).collect()).collect()
}

fn serve<E: Executor>(exec: E, ps: &[Vec<u32>], slots: usize) -> Vec<Vec<u32>> {
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: slots, max_batched_tokens: 256, ..Config::default() });
    let ids: Vec<u64> = ps
        .iter()
        .map(|p| sched.submit(Request { prompt: p.clone(), max_tokens: NEW, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap())
        .collect();
    let mut got: HashMap<u64, Vec<u32>> = HashMap::new();
    while !sched.idle() {
        let mut events = Vec::new();
        sched.step(&mut events).unwrap();
        for e in events {
            if let Event::Token { id, token } = e {
                got.entry(id).or_default().push(token);
            }
        }
    }
    ids.iter().map(|id| got.remove(id).unwrap_or_default()).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

struct Outcome {
    logit_err: f32,
    logit_scale: f32,
    top1: usize,
    rows: usize,
    tokens_equal: usize,
    streams: usize,
}

fn compare(dir: &str, serving: bool, tp: usize, force: Option<Transport>, graphs: bool) -> Outcome {
    let mut s = spec_for(dir, serving);
    s.runtime.graphs = graphs;
    let slots = 4;
    let lay = layout(&s, slots);
    let vocab = s.dims.vocab as u32;
    let ps = prompts(vocab);
    let probe: Vec<u32> = (0..48u32).map(|j| (j * 53 + 11) % vocab).collect();
    let (want_logits, want_tokens) = {
        let mut rt = Runtime::load_layout(&s, lay).unwrap();
        rt.acquire_state(0).unwrap();
        let logits = rt.prompt_logits(&probe).unwrap();
        rt.release_state(0);
        (logits, serve(GpuExecutor::new(rt).unwrap(), &ps, slots))
    };
    let exec = TpExecutor::load_with(&s, lay, &vec![device(); tp], force).unwrap();
    let got_logits = exec.prompt_logits(&probe).unwrap();
    let got_tokens = serve(exec, &ps, slots);
    let v = s.dims.vocab;
    let rows = probe.len();
    let top1 = (0..rows).filter(|&i| argmax(&got_logits[i * v..(i + 1) * v]) == argmax(&want_logits[i * v..(i + 1) * v])).count();
    Outcome {
        logit_err: max_abs(&got_logits, &want_logits),
        logit_scale: want_logits.iter().fold(0.0f32, |m, x| m.max(x.abs())),
        top1,
        rows,
        tokens_equal: got_tokens.iter().zip(&want_tokens).filter(|(a, b)| a == b).count(),
        streams: ps.len(),
    }
    .report(dir, tp, &got_tokens, &want_tokens)
}

impl Outcome {
    fn report(self, dir: &str, tp: usize, got: &[Vec<u32>], want: &[Vec<u32>]) -> Outcome {
        eprintln!(
            "{dir} tp{tp}: logits max |d| {:.3e} (of {:.2}), top-1 {}/{}, greedy streams identical {}/{}",
            self.logit_err, self.logit_scale, self.top1, self.rows, self.tokens_equal, self.streams
        );
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if g != w {
                let at = g.iter().zip(w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
                eprintln!("  stream {i}: first difference at token {at} of {}", w.len());
            }
        }
        self
    }
}

fn check(dir: &str, serving: bool, tps: &[usize]) {
    for &tp in tps {
        {
            let graphs = false;
            let o = compare(dir, serving, tp, None, graphs);
            assert_eq!(o.streams, 6);
            assert!(o.logit_err <= 2e-3 * o.logit_scale.max(1.0), "{dir} tp{tp} graphs {graphs}: logits drift by {}", o.logit_err);
            assert_eq!(o.top1, o.rows, "{dir} tp{tp} graphs {graphs}: top-1 changed");
            assert!(o.tokens_equal + 1 >= o.streams, "{dir} tp{tp} graphs {graphs}: only {}/{} greedy streams match", o.tokens_equal, o.streams);
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn dense_tp_matches_tp1_on_virtual_ranks() {
    check("tiny-mistral", false, &[2]);
    check("tiny-smollm3", false, &[2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fused_qkv_and_gate_up_split_by_head_and_by_column() {
    check("tiny-phi3", false, &[2, 4]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn hybrid_delta_net_tp_matches_tp1_on_virtual_ranks() {
    check("tiny-qwen3.5", false, &[2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn tiny27b_nvfp4_fp8_tp_matches_tp1_on_virtual_ranks() {
    check("tiny-qwen3.5-27b", false, &[2]);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_host_pinned_transport_gives_the_same_tokens() {
    let o = compare("tiny-qwen3.5-27b", false, 2, Some(Transport::HostPinned), false);
    assert_eq!(o.top1, o.rows);
    assert!(o.tokens_equal + 1 >= o.streams);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn mtp_speculation_under_tp_emits_the_tp1_greedy_tokens() {
    for dir in ["tiny-qwen3.5-mtp", "tiny-qwen3.5-27b"] {
        let mut s = spec_for(dir, false);
        let m = s.mtp.as_mut().unwrap();
        m.draft_tokens = 2;
        m.head_quant = Some(specrt::spec::Quant::Bf16);
        m.graph = false;
        s.runtime.graphs = false;
        let slots = 4;
        let lay = layout(&s, slots);
        let ps = prompts(s.dims.vocab as u32);
        let mut plain = s.clone();
        plain.mtp.as_mut().unwrap().speculate = false;
        let want = serve(GpuExecutor::new(Runtime::load_layout(&plain, lay).unwrap()).unwrap(), &ps, slots);
        let exec = TpExecutor::load(&s, lay, &[device(), device()]).unwrap();
        assert_eq!(exec.draft_tokens(), 2, "{dir}: the ranks speculate");
        let got = serve(exec, &ps, slots);
        let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        eprintln!("{dir} tp2 with mtp: greedy streams identical to tp1 without mtp {same}/{}", ps.len());
        assert!(same + 1 >= ps.len(), "{dir}: {same}/{} streams match", ps.len());
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn moe_tp_matches_tp1_on_virtual_ranks() {
    check("tiny-qwen3.5-moe", false, &[2]);
    check("tiny-qwen3.5-moe-fp8", false, &[2]);
}

#[test]
fn moe_specs_split_every_expert_and_refuse_widths_that_do_not_divide() {
    let s = spec_for("tiny-qwen3.5-moe", false);
    specrt::tp::supports(&s, 2).unwrap();
    let r = specrt::tp::rank_spec(&s, 2).unwrap();
    assert_eq!(r.moe.as_ref().unwrap().expert_intermediate, s.moe.as_ref().unwrap().expert_intermediate / 2);
    let mut odd = s.clone();
    odd.moe.as_mut().unwrap().expert_intermediate = 272;
    assert!(specrt::tp::supports(&odd, 2).unwrap_err().to_string().contains("expert"));
}

#[test]
fn head_counts_that_do_not_divide_are_refused() {
    let s = spec_for("tiny-qwen3.5", false);
    assert!(specrt::tp::supports(&s, 4).unwrap_err().to_string().contains("does not divide"));
    let p = spec_for("tiny-phi3", false);
    assert!(specrt::tp::supports(&p, 4).is_ok());
}
