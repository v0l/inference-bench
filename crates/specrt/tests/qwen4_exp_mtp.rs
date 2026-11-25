use specrt::plan::{Buf, GATED_DRAFT, Op, Phase, Plan, Weight};
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, SpecRow};
use specrt::spec::{Quant, Spec};
use std::path::{Path, PathBuf};

const NAME: &str = "tiny-qwen4-exp-mtp";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(NAME)
}

fn meta() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("reference").join(format!("{NAME}.json"))).unwrap()).unwrap()
}

fn floats(key: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture().join("reference").join(format!("{NAME}.{key}.bin"))).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn spec(k: usize) -> Spec {
    let mut s = Spec::from_hf_config_serving(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap_or_else(|e| panic!("{e:#}"));
    s.repo = fixture().display().to_string();
    s.runtime.split_activations = Some(true);
    let m = s.mtp.as_mut().expect("the fixture has an MTP layer");
    m.draft_tokens = k;
    m.graph = true;
    m.head_quant = Some(Quant::Bf16);
    s
}

#[test]
fn the_fixture_derives_a_streamed_mtp_layer_from_the_family_spec() {
    let s = spec(2);
    s.runnable().unwrap();
    assert!(s.refused.is_empty());
    assert!(s.streamed_mtp() && s.speculative().is_some());
    assert_eq!(s.draft_width(), 4 * 128);
    assert_eq!(s.kv_layers(), s.attention_layers() + 1);
    let lay = s.mtp_layout();
    assert_eq!(lay.fc_hidden.as_deref(), Some("mtp.fc_hidden.weight"));
    assert_eq!(lay.mixer.as_deref(), Some("mtp.hyper_connection_mixer"));
}

#[test]
fn a_streamed_layout_is_refused_without_its_pieces() {
    let mut s = spec(2);
    s.mtp.as_mut().unwrap().layout.as_mut().unwrap().fc_hidden = None;
    assert!(s.validate().unwrap_err().to_string().contains("fc_hidden and mixer"));
    let mut s = spec(2);
    s.mtp.as_mut().unwrap().layout.as_mut().unwrap().mixer = None;
    assert!(format!("{:#}", s.validate().unwrap_err()).contains("[mtp.layout] norm is empty"));
}

#[test]
fn the_draft_plan_widens_the_hidden_to_four_streams_and_collapses_before_the_head() {
    let s = spec(2);
    for phase in [Phase::Prefill, Phase::Decode, Phase::Verify] {
        let plan = Plan::build(&s, phase, true);
        let fc_hidden = plan.mtp.iter().position(|o| matches!(o, Op::Linear { weight: Weight::MtpFcHidden, input: Buf::DraftPlanes, output: Buf::Streams, .. }));
        let expand = plan.mtp.iter().position(|o| matches!(o, Op::Gated { site: specrt::plan::GATED_MODULE, .. }));
        assert!(expand.is_some() && fc_hidden > expand, "{phase:?}: the embedding is copied to every stream, then fc_hidden adds into them: {:?}", plan.mtp);
        assert!(plan.mtp.iter().any(|o| matches!(o, Op::Gated { site: 0, .. })) && plan.mtp.iter().any(|o| matches!(o, Op::Gated { site: 1, .. })));
        assert!(plan.mtp.iter().any(|o| matches!(o, Op::MoeExperts)));
        assert!(plan.mtp.iter().any(|o| matches!(o, Op::Kernel { .. })), "the draft attention runs the indexer nodes");
        assert!(!plan.mtp.iter().any(|o| matches!(o, Op::Norm { .. })));
        assert!(plan.mtp_head.iter().any(|o| matches!(o, Op::Gated { site: GATED_DRAFT, .. })));
        assert!(!plan.mtp_head.iter().any(|o| matches!(o, Op::MtpNorm)));
        specrt::signature::validate_plan(&plan, &s).unwrap_or_else(|e| panic!("{phase:?}: {e:#}"));
    }
}

#[test]
fn the_flat_draft_plan_of_qwen3_5_is_unchanged() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-moe-mtp");
    let s = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    assert!(!s.streamed_mtp() && s.draft_width() == s.dims.hidden);
    let plan = Plan::build(&s, Phase::Decode, true);
    assert!(matches!(plan.mtp[0], Op::MtpIn));
    assert!(!plan.mtp.iter().chain(&plan.mtp_head).any(|o| matches!(o, Op::Gated { .. })));
    assert!(plan.mtp_head.iter().any(|o| matches!(o, Op::MtpNorm)));
}

#[test]
fn the_loader_reads_both_projections_the_wide_norm_and_the_draft_mixer() {
    let s = spec(2);
    let ck = specrt::weights::Checkpoint::open(&fixture()).unwrap();
    let w = ck.mtp(&s).unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(w.norm_hidden.len(), 4 * 128);
    assert_eq!(w.norm_embed.len(), 128);
    assert!(w.fc_hidden.is_some());
    assert_eq!(w.layer.gated.len(), 2);
    let mixer: Vec<&str> = w.mixer.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(mixer, ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"]);
    assert!(w.layer.moe.is_some());
    let module = specrt::codegen::Module::generate_kv(&s, 256, 2, &kir::Target::sm_120()).unwrap();
    for k in ["mtp_streams_in", "mtp_copy_w", "mtp_state_save", "mtp_state_load", "mtp_replay_rows", "mtp_replay_seqs", "mtp_fc_mma", "mtp_fc_hidden_mma"] {
        assert!(module.kernels.iter().any(|n| n.starts_with(k)), "{k} missing");
    }
}

#[test]
fn the_reference_drafts_come_from_the_vllm_formulation() {
    let m = meta();
    assert_eq!(ids(&m["drafts"]).len(), 3);
    let vocab = m["vocab"].as_u64().unwrap() as usize;
    let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let logits = floats("logits");
    assert_eq!(logits.len(), m["rows"].as_u64().unwrap() as usize * vocab);
    assert_eq!(top(&logits[logits.len() - vocab..]), ids(&m["drafts"])[0]);
    assert_eq!(top(&floats("chain")), ids(&m["drafts"])[1]);
    assert_eq!(top(&floats("third")), ids(&m["drafts"])[2]);
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_draft_logits_and_tokens_match_the_reference_for_two_and_three_drafts() {
    let m = meta();
    let prompt = ids(&m["tokens"]);
    let want = ids(&m["drafts"]);
    for (k, key) in [(2, "chain"), (3, "third")] {
        let mut rt = Runtime::load(&spec(k), 256).unwrap_or_else(|e| panic!("{e:#}"));
        let (logits, drafts) = rt.mtp_prompt_logits(&prompt).unwrap();
        let reference = floats(key);
        let d = max_abs(&logits, &reference);
        eprintln!("k = {k}: draft logits max abs {d:e}, drafts {drafts:?}, reference {:?}", &want[..k]);
        assert!(d <= 1e-3, "k = {k}: {d}");
        assert_eq!(drafts, want[..k]);
    }
}

const NEW: usize = 40;

fn prompts() -> Vec<Vec<u32>> {
    (0..6).map(|i| (0..(9 + i * 7) as u32).map(|j| (j * 131 + i as u32 * 977 + 7) % 510 + 2).collect()).collect()
}

fn layout(slots: usize) -> Layout {
    let pages = 256usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 256, slots, kv_blocks: slots * pages, prefill_chunk: 64, kv_planes: 2, kv_headroom: 0, state_on_demand: false }
}

fn tables(rt: &mut Runtime, slots: usize) {
    let pages = 256usize.div_ceil(specrt::codegen::PAGE) as u32;
    for s in 0..slots {
        let blocks: Vec<u32> = (0..pages).map(|p| (s as u32 * pages + p * 5) % (slots as u32 * pages)).collect();
        rt.set_block_table(s, &blocks).unwrap();
    }
}

fn plain_greedy(s: &Spec, ps: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let mut rt = Runtime::load_layout(&plain, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    ps.iter()
        .map(|p| {
            let mut out = vec![rt.prefill_slot(&mut st, 0, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
            while out.len() < NEW {
                let row = DecodeRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32 };
                out.push(rt.decode(&mut st, &[row]).unwrap()[0]);
            }
            out
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Drafts {
    Mtp,
    Oracle,
    Wrong,
}

fn speculative(s: &Spec, ps: &[Vec<u32>], want: &[Vec<u32>], drafts: Drafts, batch: usize) -> (Vec<Vec<u32>>, specrt::runtime::SpecStats) {
    let k = s.mtp.as_ref().unwrap().draft_tokens;
    let mut rt = Runtime::load_layout(s, layout(batch)).unwrap_or_else(|e| panic!("{e:#}"));
    tables(&mut rt, batch);
    let mut st = rt.stepper().unwrap();
    let mut outs = Vec::new();
    for (group, ws) in ps.chunks(batch).zip(want.chunks(batch)) {
        let mut out: Vec<Vec<u32>> = group.iter().enumerate().map(|(slot, p)| vec![rt.prefill_slot(&mut st, slot, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()]).collect();
        loop {
            let live: Vec<usize> = (0..group.len()).filter(|&i| out[i].len() < NEW).collect();
            if live.is_empty() {
                break;
            }
            for &i in &live {
                let have = out[i].len();
                let mut d: Vec<u32> = (0..k).map(|j| ws[i].get(have + j).copied().unwrap_or(2)).collect();
                match drafts {
                    Drafts::Mtp => continue,
                    Drafts::Oracle => {}
                    Drafts::Wrong => {
                        let at = (have + i) % (k + 1);
                        if at < k {
                            d[at] = (d[at] + 1) % 512;
                        }
                    }
                }
                rt.set_drafts(i, &d).unwrap();
            }
            let rows: Vec<SpecRow> = live
                .iter()
                .map(|&i| SpecRow { slot: i, token: *out[i].last().unwrap(), pos: group[i].len() + out[i].len() - 1, sampling: Sampling::GREEDY, step: out[i].len() as u32, max_accept: NEW - out[i].len() - 1 })
                .collect();
            for (&i, e) in live.iter().zip(rt.speculate(&rows).unwrap()) {
                out[i].extend(e);
            }
        }
        for o in &mut out {
            o.truncate(NEW);
        }
        outs.extend(out);
    }
    (outs, rt.spec_stats())
}

fn check(name: &str, got: &[Vec<u32>], want: &[Vec<u32>]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let first = g.iter().zip(w).position(|(a, b)| a != b);
        assert_eq!(g, w, "{name}: prompt {i} differs first at token {first:?}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn speculative_greedy_with_the_streamed_draft_layer_is_identical_to_plain_greedy() {
    let ps = prompts();
    for k in [1, 2, 3] {
        let s = spec(k);
        let want = plain_greedy(&s, &ps);
        for batch in [1, 2] {
            let (got, stats) = speculative(&s, &ps, &want, Drafts::Mtp, batch);
            eprintln!("k = {k}, batch {batch}: MTP drafts accepted {} of {}", stats.accepted, stats.drafted);
            check(&format!("MTP drafts, k = {k}, batch {batch}"), &got, &want);
            let (got, stats) = speculative(&s, &ps, &want, Drafts::Oracle, batch);
            let mean = 1.0 + stats.accepted as f64 / stats.rows.max(1) as f64;
            assert!(mean > k as f64 + 0.5, "oracle drafts accept almost everything: {mean}");
            check(&format!("oracle drafts, k = {k}, batch {batch}"), &got, &want);
            let (got, _) = speculative(&s, &ps, &want, Drafts::Wrong, batch);
            check(&format!("rotating rejections, k = {k}, batch {batch}"), &got, &want);
        }
    }
}
