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
    serve_with(exec, ps, slots, |_| Sampling::GREEDY)
}

fn serve_with<E: Executor>(exec: E, ps: &[Vec<u32>], slots: usize, sampling: impl Fn(usize) -> Sampling) -> Vec<Vec<u32>> {
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: slots, max_batched_tokens: 256, ..Config::default() });
    let ids: Vec<u64> = ps
        .iter()
        .enumerate()
        .map(|(i, p)| sched.submit(Request { prompt: p.clone(), max_tokens: NEW, sampling: sampling(i), stop: vec![], salt: String::new(), media: Vec::new(), grammar: None }).unwrap())
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
    compare_split(dir, serving, tp, force, graphs, specrt::spec::TpSplit::Auto)
}

fn compare_split(dir: &str, serving: bool, tp: usize, force: Option<Transport>, graphs: bool, split: specrt::spec::TpSplit) -> Outcome {
    compare_acts(dir, serving, tp, force, graphs, split, Some(true))
}

fn compare_acts(dir: &str, serving: bool, tp: usize, force: Option<Transport>, graphs: bool, split: specrt::spec::TpSplit, split_activations: Option<bool>) -> Outcome {
    let mut s = spec_for(dir, serving);
    s.runtime.split_activations = split_activations;
    if let Some(m) = s.moe.as_mut() {
        m.tp_split = split;
    }
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
fn single_plane_dense_tp_stays_close_to_single_plane_tp1() {
    for dir in ["tiny-mistral", "tiny-phi3", "tiny-smollm3"] {
        let o = compare_acts(dir, false, 2, None, false, specrt::spec::TpSplit::Auto, Some(false));
        assert!(o.logit_err <= 5e-3 * o.logit_scale.max(1.0), "{dir}: single-plane logits drift by {}", o.logit_err);
        assert!(o.tokens_equal * 2 >= o.streams, "{dir}: only {}/{} greedy streams match", o.tokens_equal, o.streams);
    }
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
    use specrt::spec::TpSplit;
    for dir in ["tiny-qwen3.5-moe", "tiny-qwen3.5-moe-fp8"] {
        for split in [TpSplit::Columns, TpSplit::Experts, TpSplit::Auto] {
            eprintln!("{dir} split {split:?}");
            let o = compare_split(dir, false, 2, None, false, split);
            assert!(o.logit_err <= 2e-3 * o.logit_scale.max(1.0), "{dir} {split:?}: logits drift by {}", o.logit_err);
            assert_eq!(o.top1, o.rows, "{dir} {split:?}: top-1 changed");
            assert!(o.tokens_equal + 1 >= o.streams, "{dir} {split:?}: only {}/{} greedy streams match", o.tokens_equal, o.streams);
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn moe_tp_stays_within_a_tight_logit_bound() {
    use specrt::spec::TpSplit;
    for dir in ["tiny-qwen3.5-moe", "tiny-qwen3.5-moe-fp8"] {
        for split in [TpSplit::Columns, TpSplit::Experts] {
            let o = compare_split(dir, false, 2, None, false, split);
            assert!(o.logit_err <= 1e-4 * o.logit_scale.max(1.0), "{dir} {split:?}: logits drift by {} of {}", o.logit_err, o.logit_scale);
            assert_eq!(o.top1, o.rows, "{dir} {split:?}: top-1 changed");
        }
    }
}

#[test]
fn moe_specs_split_every_expert_and_refuse_widths_that_do_not_divide() {
    let mut s = spec_for("tiny-qwen3.5-moe", false);
    s.moe.as_mut().unwrap().tp_split = specrt::spec::TpSplit::Columns;
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

fn tp3_fixture() -> PathBuf {
    use safetensors::tensor::{Dtype, TensorView};
    let dir = std::env::temp_dir().join(format!("specrt-tp3-vocab1000-{}", std::process::id()));
    if dir.join("model.safetensors").exists() {
        return dir;
    }
    std::fs::create_dir_all(&dir).unwrap();
    let (h, hd, heads, kv, inter, vocab, layers) = (192usize, 64usize, 6usize, 3usize, 288usize, 1000usize, 2usize);
    let config = serde_json::json!({
        "architectures": ["MistralForCausalLM"], "model_type": "mistral", "head_dim": hd, "hidden_act": "silu", "hidden_size": h,
        "intermediate_size": inter, "max_position_embeddings": 512, "num_attention_heads": heads, "num_hidden_layers": layers,
        "num_key_value_heads": kv, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "sliding_window": null, "tie_word_embeddings": false, "vocab_size": vocab, "bos_token_id": null, "eos_token_id": null, "pad_token_id": null
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let mut x = 0x2545f4914f6cdd1du64;
    let mut tensor = |rows: usize, cols: usize, scale: f32, base: f32| -> Vec<u8> {
        (0..rows * cols)
            .flat_map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                specrt::weights::f32_to_bf16(base + u * scale).to_le_bytes()
            })
            .collect()
    };
    let mut named: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut add = |name: String, rows: usize, cols: usize, scale: f32, base: f32| {
        let shape = if cols == 0 { vec![rows] } else { vec![rows, cols] };
        named.push((name, shape, tensor(rows, cols.max(1), scale, base)));
    };
    add("model.embed_tokens.weight".into(), vocab, h, 1.0, 0.0);
    add("lm_head.weight".into(), vocab, h, 1.0, 0.0);
    add("model.norm.weight".into(), h, 0, 0.4, 1.0);
    let lin = |k: usize| 2.0 / (k as f32).sqrt();
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        add(format!("{p}.input_layernorm.weight"), h, 0, 0.4, 1.0);
        add(format!("{p}.post_attention_layernorm.weight"), h, 0, 0.4, 1.0);
        add(format!("{p}.self_attn.q_proj.weight"), heads * hd, h, lin(h), 0.0);
        add(format!("{p}.self_attn.k_proj.weight"), kv * hd, h, lin(h), 0.0);
        add(format!("{p}.self_attn.v_proj.weight"), kv * hd, h, lin(h), 0.0);
        add(format!("{p}.self_attn.o_proj.weight"), h, heads * hd, lin(heads * hd), 0.0);
        add(format!("{p}.mlp.gate_proj.weight"), inter, h, lin(h), 0.0);
        add(format!("{p}.mlp.up_proj.weight"), inter, h, lin(h), 0.0);
        add(format!("{p}.mlp.down_proj.weight"), h, inter, lin(inter), 0.0);
    }
    let views: Vec<(String, TensorView)> = named.iter().map(|(n, s, d)| (n.clone(), TensorView::new(Dtype::BF16, s.clone(), d).unwrap())).collect();
    safetensors::tensor::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    dir
}

fn spec_at(dir: &Path, name: &str) -> Spec {
    let mut s = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    s.name = name.into();
    s.repo = dir.display().to_string();
    s.vision = None;
    s.runtime.graphs = false;
    s.runtime.split_activations = Some(true);
    s
}

fn sampled(i: usize) -> Sampling {
    let s = Sampling::new(0.8, 4242 + i as u32);
    match i % 3 {
        0 => s,
        1 => s.truncated(40, 1.0, 0.0),
        _ => s.truncated(0, 0.9, 0.02),
    }
}

fn sharded_head_streams(s: &Spec, tp: usize) {
    let slots = 4;
    let lay = layout(s, slots);
    let ps = prompts(s.dims.vocab as u32);
    let want_greedy = serve(GpuExecutor::new(Runtime::load_layout(s, lay).unwrap()).unwrap(), &ps, slots);
    let want_sampled = serve_with(GpuExecutor::new(Runtime::load_layout(s, lay).unwrap()).unwrap(), &ps, slots, sampled);
    let exec = TpExecutor::load(s, lay, &vec![device(); tp]).unwrap();
    let spans = exec.head_rows().unwrap();
    assert_eq!(spans.len(), tp);
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[tp - 1].end, s.dims.vocab);
    assert!(spans.windows(2).all(|w| w[0].end == w[1].start), "{}: {spans:?} do not tile the vocabulary", s.name);
    assert!(spans.iter().all(|r| r.len() < s.dims.vocab), "{} tp{tp}: a rank holds the whole head ({spans:?})", s.name);
    let got_greedy = serve(exec, &ps, slots);
    let exec = TpExecutor::load(s, lay, &vec![device(); tp]).unwrap();
    let got_sampled = serve_with(exec, &ps, slots, sampled);
    let same = |a: &[Vec<u32>], b: &[Vec<u32>]| a.iter().zip(b).filter(|(x, y)| x == y).count();
    let (g, k) = (same(&got_greedy, &want_greedy), same(&got_sampled, &want_sampled));
    eprintln!("{} tp{tp} vocab-sharded heads {spans:?}: greedy streams identical {g}/{}, sampled streams identical {k}/{}", s.name, ps.len(), ps.len());
    assert_eq!(g, ps.len(), "{} tp{tp}: greedy streams differ from tp1", s.name);
    assert_eq!(k, ps.len(), "{} tp{tp}: sampled streams differ from tp1", s.name);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn vocab_sharded_heads_give_the_tp1_greedy_and_sampled_streams() {
    sharded_head_streams(&spec_at(&fixture("tiny-mistral"), "tiny-mistral"), 2);
    sharded_head_streams(&spec_at(&fixture("tiny-qwen3.5-27b"), "tiny-qwen3.5-27b"), 2);
    let dir = tp3_fixture();
    sharded_head_streams(&spec_at(&dir, "tiny-tp3-vocab1000"), 3);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn vocab_sharded_draft_heads_speculate_the_tp1_sampled_streams() {
    let mut s = spec_at(&fixture("tiny-qwen3.5-27b"), "tiny-qwen3.5-27b");
    let m = s.mtp.as_mut().unwrap();
    m.draft_tokens = 2;
    m.graph = false;
    let slots = 4;
    let lay = layout(&s, slots);
    let ps = prompts(s.dims.vocab as u32);
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let want = serve_with(GpuExecutor::new(Runtime::load_layout(&plain, lay).unwrap()).unwrap(), &ps, slots, sampled);
    let one = serve_with(GpuExecutor::new(Runtime::load_layout(&s, lay).unwrap()).unwrap(), &ps, slots, sampled);
    let exec = TpExecutor::load(&s, lay, &[device(), device()]).unwrap();
    assert_eq!(exec.draft_tokens(), 2);
    assert!(exec.head_rows().unwrap().iter().all(|r| r.len() == s.dims.vocab / 2));
    let got = serve_with(exec, &ps, slots, sampled);
    let same = |a: &[Vec<u32>], b: &[Vec<u32>]| a.iter().zip(b).filter(|(x, y)| x == y).count();
    eprintln!("tiny-qwen3.5-27b tp2 nvfp4 draft head sharded, sampled with mtp: identical to tp1 with mtp {}/{}, to tp1 without mtp {}/{}", same(&got, &one), ps.len(), same(&got, &want), ps.len());
    assert_eq!(same(&got, &one), ps.len(), "tp2 speculation differs from tp1 speculation");
}
