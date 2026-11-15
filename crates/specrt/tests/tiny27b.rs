use serde_json::Value;
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, SpecRow};
use specrt::sched::MediaInput;
use specrt::spec::{ActQuant, Quant, Spec};
use specrt::vision::{Patches, Processed};
use std::path::{Path, PathBuf};

const NEW: usize = 48;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-27b")
}

fn reference_dir() -> PathBuf {
    fixture().join("reference")
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(reference_dir().join(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn meta(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(reference_dir().join(name)).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

fn load_spec(serving: bool) -> Spec {
    let json = std::fs::read_to_string(fixture().join("config.json")).unwrap();
    let mut s = if serving { Spec::from_hf_config_serving(&json).unwrap() } else { Spec::from_hf_config(&json).unwrap() };
    s.name = "tiny-qwen3.5-27b".into();
    s.repo = fixture().display().to_string();
    s.runtime.vision.precision = specrt::vision_encoder::VisionPrecision::Exact;
    let m = s.mtp.as_mut().unwrap();
    m.head_quant = Some(Quant::Bf16);
    m.graph = true;
    s
}

fn weight_spec() -> Spec {
    load_spec(false)
}

fn act_spec() -> Spec {
    let s = load_spec(true);
    assert_eq!((s.act_quant, s.mlp.act_quant), (ActQuant::Fp8, ActQuant::Fp4));
    s
}

fn with_drafts(mut s: Spec, k: usize) -> Spec {
    s.mtp.as_mut().unwrap().draft_tokens = k;
    s
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let lse = max as f64 + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn mean_kl(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows)
        .map(|r| {
            let (a, b) = (log_softmax(&p[r * vocab..(r + 1) * vocab]), log_softmax(&q[r * vocab..(r + 1) * vocab]));
            a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn top1(p: &[f32], q: &[f32], vocab: usize) -> f64 {
    let rows = p.len() / vocab;
    (0..rows).filter(|&r| argmax(&p[r * vocab..(r + 1) * vocab]) == argmax(&q[r * vocab..(r + 1) * vocab])).count() as f64 / rows as f64
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn relative(a: &[f32], b: &[f32]) -> f32 {
    max_abs(a, b) / b.iter().fold(0f32, |m, v| m.max(v.abs()))
}

#[test]
fn fixture_spec_is_the_27b_shape_in_miniature() {
    let s = act_spec();
    assert_eq!(s.layers.count, 4);
    assert_eq!((s.linear_layers(), s.attention_layers()), (3, 1));
    assert_eq!(s.quant, Quant::Fp8Channel);
    assert_eq!((s.mlp.quant, s.mlp.quant_until), (Some(Quant::Nvfp4G16), Some(3)));
    assert!(s.attention.output_gate && s.attention.qk_norm);
    assert_eq!(s.rotary_dim(), 64);
    let m = s.rope.mrope.as_ref().unwrap();
    assert_eq!(m.section, [11, 11, 10]);
    assert!(m.interleaved);
    let mtp = s.mtp.as_ref().unwrap();
    assert_eq!((mtp.layers, mtp.speculate), (1, true));
    let v = s.vision.as_ref().unwrap();
    assert_eq!((v.config.depth, v.config.spatial_merge_size, v.config.out_hidden_size), (2, 2, s.dims.hidden));
    let real: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.config.json")).unwrap()).unwrap();
    let tiny: Value = serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    for g in ["group_0", "group_1"] {
        for part in ["weights", "input_activations"] {
            for key in ["num_bits", "strategy", "group_size", "type", "dynamic"] {
                assert_eq!(tiny["quantization_config"]["config_groups"][g][part][key], real["quantization_config"]["config_groups"][g][part][key], "{g}.{part}.{key}");
            }
        }
    }
    let real_types = real["text_config"]["layer_types"].as_array().unwrap();
    assert_eq!(&real_types[..4], tiny["text_config"]["layer_types"].as_array().unwrap().as_slice());
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn weight_only_logits_kl_and_greedy_match_transformers() {
    let s = weight_spec();
    let m = meta("tiny27b-text.json");
    let (prompt, greedy, seq) = (ids(&m["prompt"]), ids(&m["greedy"]), ids(&m["tokens"]));
    let want = floats("tiny27b-text.logits.bin");
    let mut rt = Runtime::load(&s, 256).unwrap();
    let ours = rt.prompt_logits(&seq).unwrap();
    let vocab = s.dims.vocab;
    let (kl, agree, err) = (mean_kl(&want, &ours, vocab), top1(&want, &ours, vocab), relative(&ours, &want));
    eprintln!("weight-only: KL {kl:e}, top-1 {agree}, max relative {err:e}");
    assert!(kl <= 1e-5, "KL {kl}");
    assert_eq!(agree, 1.0);
    assert!(err <= 5e-3, "{err}");
    let g = rt.generate(&prompt, greedy.len(), Sampling::GREEDY).unwrap();
    assert_eq!(g.tokens, greedy);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn act_quantised_logits_stay_inside_the_reference_activation_envelope() {
    let s = act_spec();
    let m = meta("tiny27b-text.json");
    let (prompt, seq) = (ids(&m["prompt"]), ids(&m["tokens"]));
    let weight_only = floats("tiny27b-text.logits.bin");
    let act_ref = floats("tiny27b-text-act.logits.bin");
    let vocab = s.dims.vocab;
    let ref_kl = mean_kl(&weight_only, &act_ref, vocab);
    let ref_top1 = top1(&weight_only, &act_ref, vocab);
    let mut rt = Runtime::load(&s, 256).unwrap();
    let ours = rt.prompt_logits(&seq).unwrap();
    let kl = mean_kl(&weight_only, &ours, vocab);
    let agree = top1(&weight_only, &ours, vocab);
    let to_act = mean_kl(&act_ref, &ours, vocab);
    eprintln!("act-quant: KL(weight-only||ours) {kl:e} against reference {ref_kl:e}, KL(act ref||ours) {to_act:e}, top-1 {agree} against {ref_top1}");
    assert!(kl <= 1.5 * ref_kl + 1e-6, "KL {kl} against the act reference's {ref_kl}");
    assert!(agree + 0.05 >= ref_top1, "top-1 {agree} against {ref_top1}");
    let g = rt.generate(&prompt, 8, Sampling::GREEDY).unwrap();
    assert_eq!(g.tokens.len(), 8);
}

fn layout(slots: usize) -> Layout {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE);
    Layout { max_ctx: 512, slots, kv_blocks: slots * pages, prefill_chunk: 128, kv_planes: 1, kv_headroom: 0, state_on_demand: false }
}

fn tables(rt: &mut Runtime, slots: usize) {
    let pages = 512usize.div_ceil(specrt::codegen::PAGE) as u32;
    for s in 0..slots {
        let blocks: Vec<u32> = (0..pages).map(|p| (s as u32 * pages + p * 7) % (slots as u32 * pages)).collect();
        rt.set_block_table(s, &blocks).unwrap();
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn mtp_drafts_and_chained_logits_match_transformers() {
    let s = with_drafts(weight_spec(), 2);
    let m = meta("tiny27b-mtp.json");
    let seq = ids(&meta("tiny27b-text.json")["tokens"]);
    let chain = floats("tiny27b-mtp.chain.bin");
    let vocab = s.dims.vocab;
    let mut rt = Runtime::load(&s, 512).unwrap();
    let (draft, chain_draft) = (ids(&m["draft"]), ids(&m["chain_draft"]));
    for (i, &n) in ids(&m["prefixes"]).iter().enumerate() {
        let (logits, drafts) = rt.mtp_prompt_logits(&seq[..n as usize]).unwrap();
        let want = &chain[i * vocab..(i + 1) * vocab];
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = max_abs(&logits, want);
        eprintln!("prefix {n}: chained draft logits max abs {err:e} (scale {scale}), drafts {drafts:?}");
        assert!(err <= 2e-3 * scale.max(1.0), "prefix {n}: {err}");
        assert_eq!(drafts, vec![draft[i], chain_draft[i]], "prefix {n}");
    }
}

fn plain_greedy(s: &Spec, prompts: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut plain = s.clone();
    plain.mtp.as_mut().unwrap().speculate = false;
    let mut rt = Runtime::load_layout(&plain, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    prompts
        .iter()
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

fn speculative_greedy(s: &Spec, prompts: &[Vec<u32>], oracle: Option<&[Vec<u32>]>) -> (Vec<Vec<u32>>, f64) {
    let k = s.mtp.as_ref().unwrap().draft_tokens;
    let mut rt = Runtime::load_layout(s, layout(1)).unwrap();
    tables(&mut rt, 1);
    let mut st = rt.stepper().unwrap();
    let mut outs = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let mut out = vec![rt.prefill_slot(&mut st, 0, p, 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap()];
        while out.len() < NEW {
            if let Some(want) = oracle {
                let d: Vec<u32> = (0..k).map(|j| want[i].get(out.len() + j).copied().unwrap_or(0)).collect();
                rt.set_drafts(0, &d).unwrap();
            }
            let row = SpecRow { slot: 0, token: *out.last().unwrap(), pos: p.len() + out.len() - 1, sampling: Sampling::GREEDY, step: out.len() as u32, max_accept: NEW - out.len() - 1 };
            out.extend(rt.speculate(&[row]).unwrap().remove(0));
        }
        out.truncate(NEW);
        outs.push(out);
    }
    let st = rt.spec_stats();
    (outs, 1.0 + st.accepted as f64 / st.rows.max(1) as f64)
}

fn speculation_prompts() -> Vec<Vec<u32>> {
    (0..4usize).map(|i| (0..(9 + i * 23) as u32).map(|j| (j * 131 + i as u32 * 977 + 7) % 1024).collect()).collect()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn speculative_greedy_equals_plain_greedy_and_oracle_drafts_accept() {
    for k in 1..=3 {
        let s = with_drafts(weight_spec(), k);
        let ps = speculation_prompts();
        let want = plain_greedy(&s, &ps);
        let (got, mean) = speculative_greedy(&s, &ps, None);
        eprintln!("k = {k}: mean acceptance length {mean:.3} with MTP drafts");
        assert_eq!(got, want, "MTP drafts, k = {k}");
        let (got, mean) = speculative_greedy(&s, &ps, Some(&want));
        eprintln!("k = {k}: mean acceptance length {mean:.3} with oracle drafts");
        assert!(mean > k as f64 + 0.8, "oracle drafts accept almost everything: {mean}");
        assert_eq!(got, want, "oracle drafts, k = {k}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn speculative_greedy_with_act_quant_equals_its_plain_greedy() {
    let s = with_drafts(act_spec(), 2);
    let ps = speculation_prompts();
    let want = plain_greedy(&s, &ps);
    let (got, _) = speculative_greedy(&s, &ps, None);
    assert_eq!(got, want);
}

fn vision_case(rt: &mut Runtime, case: &str) {
    let m = meta(&format!("tiny27b-vision-{case}.json"));
    let patch_dim = m["patch_dim"].as_u64().unwrap() as usize;
    let all = floats(&format!("tiny27b-vision-{case}.patches.bin"));
    let mut at = 0;
    let media: Vec<MediaInput> = m["grids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let grid = [g[0].as_u64().unwrap() as usize, g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize];
            let n = grid.iter().product::<usize>() * patch_dim;
            let data = all[at..at + n].to_vec();
            at += n;
            MediaInput { processed: Processed::Image(Patches { grid, data }), hash: [(case.len() * 8 + i) as u8 + 1; 32] }
        })
        .collect();
    let prompt = ids(&m["ids"]);
    let plan = rt.plan_media(&prompt, &media).unwrap();
    let want_embeds = floats(&format!("tiny27b-vision-{case}.embeds.bin"));
    let err = relative(&plan.embeds, &want_embeds);
    eprintln!("{case}: vision embeddings max relative {err:e}");
    assert!(err <= 2e-3, "{case}: {err}");
    rt.set_media(0, Some(&plan)).unwrap();
    let ours = rt.prompt_logits(&prompt).unwrap();
    let want = floats(&format!("tiny27b-vision-{case}.logits.bin"));
    let vocab = rt.spec().dims.vocab;
    let (kl, agree) = (mean_kl(&want, &ours, vocab), top1(&want, &ours, vocab));
    eprintln!("{case}: logits KL {kl:e}, top-1 {agree}");
    assert!(kl <= 1e-4, "{case}: KL {kl}");
    assert_eq!(agree, 1.0, "{case}");
    rt.set_media(0, None).unwrap();
    let text_only = rt.prompt_logits(&prompt).unwrap();
    assert!(mean_kl(&want, &text_only, vocab) > 10.0 * kl.max(1e-6), "{case}: the image changes the logits");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn vision_embeddings_and_mrope_logits_match_transformers() {
    let mut rt = Runtime::load(&weight_spec(), 256).unwrap();
    assert!(rt.has_vision());
    vision_case(&mut rt, "image");
    vision_case(&mut rt, "images2");
}
