use specrt::runtime::{Runtime, Sampling};
use specrt::spec::{Quant, Spec};
use specrt::weights::{Checkpoint, Matrix, bf16_to_f32, e4m3_to_f32};
use std::path::{Path, PathBuf};

const FIXTURES: [&str; 4] = ["tiny-qwen3.5", "tiny-qwen3.5-hd256", "tiny-qwen3.5-fp8", "tiny-qwen3.5-nvfp4"];
const HEADS: [Quant; 2] = [Quant::Fp8Channel, Quant::Nvfp4G16];

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str, head: Option<Quant>) -> Spec {
    let mut spec = match dir {
        "qwen3-0.6b" => Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap(),
        _ => {
            let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
            let mut spec = Spec::from_hf_config(&json).unwrap();
            spec.name = dir.into();
            spec.repo = fixture_dir(dir).display().to_string();
            spec
        }
    };
    spec.head_quant = head;
    spec.validate().unwrap();
    spec
}

fn checkpoint(spec: &Spec) -> Checkpoint {
    Checkpoint::for_repo(&spec.repo).unwrap()
}

fn prompts(vocab: u32) -> Vec<Vec<u32>> {
    (0..20u32)
        .map(|q| {
            let len = [1usize, 7, 33, 64, 129, 200][q as usize % 6] + q as usize;
            (0..len as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(q * 977) >> 9) % vocab.min(1024)).collect()
        })
        .collect()
}

fn dequant_head(spec: &Spec) -> Vec<f32> {
    let ck = checkpoint(spec);
    let out = ck.outside(spec).unwrap();
    let (vocab, hidden) = (spec.dims.vocab, spec.dims.hidden);
    match &out.lm_head {
        Matrix::Bf16(w) => w.iter().map(|&v| bf16_to_f32(v)).collect(),
        Matrix::Fp8 { values, scales } => (0..vocab * hidden)
            .map(|i| {
                let (r, c) = (i / hidden, i % hidden);
                e4m3_to_f32(values[i]) * scales[scales[r].to_bits() as usize + c / specrt::codegen::FP8_BLOCK]
            })
            .collect(),
        Matrix::Fp4(m) => (0..vocab).flat_map(|r| m.dequant_row(r)).collect(),
        Matrix::Q8 { .. } => unreachable!("no q8 head here"),
    }
}

fn cpu_logits(spec: &Spec, head: &[f32], hidden_rows: &[f32]) -> Vec<f32> {
    let norm = checkpoint(spec).outside(spec).unwrap().norm;
    let h = spec.dims.hidden;
    hidden_rows
        .chunks_exact(h)
        .flat_map(|x| {
            let ms = x.iter().map(|v| (v * v) as f64).sum::<f64>() / h as f64;
            let inv = 1.0 / (ms + spec.norm.eps).sqrt();
            let xn: Vec<f64> = x.iter().zip(&norm).map(|(&v, &w)| v as f64 * inv * w as f64).collect();
            head.chunks_exact(h).map(move |row| row.iter().zip(&xn).map(|(&w, &v)| w as f64 * v).sum::<f64>() as f32).collect::<Vec<_>>()
        })
        .collect()
}

fn relative(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len());
    let scale = theirs.iter().fold(1f32, |m, v| m.max(v.abs()));
    ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max) / scale
}

fn last_layer(layers: &[f32], spec: &Spec, rows: usize, row_major: bool) -> Vec<f32> {
    let (l, h) = (spec.layers.count, spec.dims.hidden);
    (0..rows)
        .flat_map(|r| {
            let at = if row_major { (r * l + l - 1) * h } else { ((l - 1) * rows + r) * h };
            layers[at..at + h].to_vec()
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn quantised_head_logits_match_the_cpu_dequantised_head_in_decode_and_prefill() {
    let tokens: Vec<u32> = (0..48u32).map(|i| (i * 131 + 7) % 1024).collect();
    for dir in FIXTURES {
        for head in HEADS {
            let s = spec_for(dir, Some(head));
            let w = dequant_head(&s);
            let mut rt = Runtime::load(&s, 256).unwrap();
            let d = rt.trace_decode(&tokens).unwrap();
            let decode = relative(&d.logits, &cpu_logits(&s, &w, &last_layer(&d.layers, &s, tokens.len(), true)));
            let p = rt.trace_prefill(&tokens).unwrap();
            let prefill = relative(&p.logits, &cpu_logits(&s, &w, &last_layer(&p.layers, &s, tokens.len(), false)));
            eprintln!("{dir} {head:?}: decode GEMV {decode:.2e}, prefill GEMM {prefill:.2e} (max error / max |logit|)");
            assert!(decode <= 1e-4, "{dir} {head:?} decode {decode}");
            assert!(prefill <= 1e-4, "{dir} {head:?} prefill {prefill}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn head_quant_leaves_the_embedding_and_every_layer_bit_identical() {
    let tokens: Vec<u32> = (0..40u32).map(|i| (i * 37 + 3) % 1024).collect();
    for dir in FIXTURES {
        let mut base = Runtime::load(&spec_for(dir, None), 256).unwrap();
        let want = base.trace_decode(&tokens).unwrap();
        let want_prefill = base.trace_prefill(&tokens).unwrap();
        drop(base);
        for head in HEADS {
            let mut rt = Runtime::load(&spec_for(dir, Some(head)), 256).unwrap();
            assert!(rt.trace_decode(&tokens).unwrap().layers == want.layers, "{dir} {head:?} decode layers");
            assert!(rt.trace_prefill(&tokens).unwrap().layers == want_prefill.layers, "{dir} {head:?} prefill layers");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn bf16_head_is_the_default_and_bit_identical_to_naming_it() {
    for dir in FIXTURES {
        let plain = spec_for(dir, None);
        assert_eq!(plain.lm_head_quant(), Quant::Bf16);
        let p = &prompts(1024)[3];
        let mut a = Runtime::load(&plain, 256).unwrap();
        let (logits, greedy) = (a.prompt_logits(p).unwrap(), a.generate(p, 24, Sampling::GREEDY).unwrap().tokens);
        drop(a);
        let mut b = Runtime::load(&spec_for(dir, Some(Quant::Bf16)), 256).unwrap();
        assert!(b.prompt_logits(p).unwrap() == logits, "{dir}");
        assert_eq!(b.generate(p, 24, Sampling::GREEDY).unwrap().tokens, greedy, "{dir}");
    }
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) as f64;
    let lse = max + row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn quantised_head_top1_and_kl_against_the_bf16_head_on_twenty_prompts() {
    let mut report = Vec::new();
    for dir in FIXTURES.iter().copied().chain(["qwen3-0.6b"]) {
        let vocab = spec_for(dir, None).dims.vocab;
        let prompts = prompts(vocab as u32);
        let mut base = Runtime::load(&spec_for(dir, None), 512).unwrap();
        let want: Vec<Vec<f32>> = prompts.iter().map(|p| base.prompt_logits(p).unwrap()).collect();
        drop(base);
        for head in HEADS {
            let mut rt = Runtime::load(&spec_for(dir, Some(head)), 512).unwrap();
            let (mut rows, mut top1, mut kl) = (0usize, 0usize, 0f64);
            for (p, w) in prompts.iter().zip(&want) {
                let got = rt.prompt_logits(p).unwrap();
                for (a, b) in w.chunks_exact(vocab).zip(got.chunks_exact(vocab)) {
                    let (lp, lq) = (log_softmax(a), log_softmax(b));
                    kl += lp.iter().zip(&lq).map(|(x, y)| x.exp() * (x - y)).sum::<f64>();
                    top1 += (argmax(a) == argmax(b)) as usize;
                    rows += 1;
                }
            }
            let (kl, top1) = (kl / rows as f64, top1 as f64 / rows as f64);
            eprintln!("{dir} {head:?}: {rows} positions, top-1 {:.2}%, mean KL(bf16 || quantised) {kl:.3e}", 100.0 * top1);
            report.push((dir, head, top1, kl));
        }
    }
    eprintln!("{report:?}");
    for (dir, head, top1, kl) in report {
        let (min_top1, max_kl) = if head == Quant::Fp8Channel { (0.9, 5e-3) } else { (0.75, 5e-2) };
        assert!(top1 >= min_top1 && kl <= max_kl, "{dir} {head:?}: top-1 {top1}, KL {kl}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn quantised_head_batch_decode_matches_single_sequence_greedy() {
    let prompts: Vec<Vec<u32>> = (0..4u32).map(|q| (0..40u32).map(|i| (i * 7 + q * 131) % 1024).collect()).collect();
    for dir in ["tiny-qwen3.5", "tiny-qwen3.5-nvfp4"] {
        for head in HEADS {
            let s = spec_for(dir, Some(head));
            let mut single = Runtime::load(&s, 256).unwrap();
            let expected: Vec<Vec<u32>> = prompts.iter().map(|p| single.generate(p, 24, Sampling::GREEDY).unwrap().tokens).collect();
            drop(single);
            let mut batch = Runtime::load_batched(&s, 256, 4).unwrap();
            assert_eq!(batch.generate_batch(&prompts, 24).unwrap().tokens, expected, "{dir} {head:?}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU and Qwen/Qwen3-0.6B"]
fn nvfp4_head_is_refused_without_the_tensor_core_path() {
    let mut s = spec_for("qwen3-0.6b", Some(Quant::Nvfp4G16));
    s.runtime.gemm_fp32 = true;
    let e = Runtime::load(&s, 256).err().expect("refused");
    assert!(format!("{e:#}").contains("head_quant"), "{e:#}");
    let mut fp8 = spec_for("qwen3-0.6b", Some(Quant::Fp8Channel));
    fp8.runtime.gemm_fp32 = true;
    let tokens: Vec<u32> = (0..20u32).collect();
    let mut rt = Runtime::load(&fp8, 256).unwrap();
    let w = dequant_head(&fp8);
    let d = rt.trace_decode(&tokens).unwrap();
    let p = rt.trace_prefill(&tokens).unwrap();
    assert!(relative(&p.logits, &cpu_logits(&fp8, &w, &last_layer(&p.layers, &fp8, tokens.len(), false))) <= 1e-4);
    assert!(relative(&d.logits, &cpu_logits(&fp8, &w, &last_layer(&d.layers, &fp8, tokens.len(), true))) <= 1e-4);
}

#[test]
#[ignore = "needs Qwen/Qwen3-0.6B"]
fn a_tied_head_quantises_only_the_output_copy() {
    for dir in ["qwen3-0.6b", "tiny-qwen3.5"] {
        let mut s = spec_for(dir, Some(Quant::Nvfp4G16));
        s.dims.tie_embeddings = true;
        let ck = checkpoint(&s);
        let out = ck.outside(&s).unwrap();
        let mut plain = s.clone();
        plain.head_quant = None;
        let bf16 = ck.outside(&plain).unwrap();
        assert!(matches!(out.lm_head, Matrix::Fp4(_)));
        assert!(matches!(bf16.lm_head, Matrix::Bf16(_)));
        assert!(out.embed == bf16.embed, "{dir}: the input embedding stays the bf16 table");
        let Matrix::Bf16(head) = &bf16.lm_head else { unreachable!() };
        assert!(head.as_ref() == out.embed.as_ref(), "{dir}: a tied head is the embedding");
        let mut fp8 = s.clone();
        fp8.head_quant = Some(Quant::Fp8Channel);
        let f = ck.outside(&fp8).unwrap();
        assert!(matches!(f.lm_head, Matrix::Fp8 { .. }) && f.embed == bf16.embed);
    }
}
