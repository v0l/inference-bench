use anyhow::{Context, Result, bail};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::spec::{KvDtype, Spec};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const DTYPES: [KvDtype; 2] = [KvDtype::F16, KvDtype::Fp8];
const DECODE_STEPS: usize = 8;

fn name(d: KvDtype) -> &'static str {
    if d.is_fp8() { "fp8" } else { "f16" }
}

fn tokenizer(spec: &Spec) -> Result<Tokenizer> {
    let path = specrt::weights::find_in_snapshot(&spec.repo, "tokenizer.json")?;
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec())
}

fn load(spec: &Spec, dtype: KvDtype, max_ctx: usize, batch: usize) -> Result<Runtime> {
    load_planes(spec, dtype, max_ctx, batch, 1)
}

fn load_planes(spec: &Spec, dtype: KvDtype, max_ctx: usize, batch: usize, planes: usize) -> Result<Runtime> {
    let mut s = spec.clone();
    s.runtime.kv_dtype = dtype;
    if batch > 1 {
        return Runtime::load_batched_planes(&s, max_ctx, batch, planes);
    }
    let layout = Layout { kv_planes: planes, ..Layout::contiguous(max_ctx, 1) };
    let mut rt = Runtime::load_layout(&s, layout)?;
    let pages = max_ctx.div_ceil(specrt::codegen::PAGE) as u32;
    rt.set_block_table(0, &(0..pages).collect::<Vec<_>>())?;
    Ok(rt)
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let z = x.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln();
    x.iter().map(|v| *v as f64 - m - z).collect()
}

fn kl(reference: &[f32], got: &[f32]) -> f64 {
    let (p, q) = (log_softmax(reference), log_softmax(got));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn text_prompts(spec: &Spec) -> Result<Vec<u32>> {
    let tok = tokenizer(spec)?;
    let text = include_str!("../../scripts/spec_prompts.txt");
    let mut all = Vec::new();
    for p in text.split("\n---\n") {
        all.extend(encode(&tok, p.trim_end())?);
    }
    Ok(all)
}

fn kl_mode(spec: &Spec, len: usize, n: usize, calibrate: bool) -> Result<()> {
    let base = text_prompts(spec)?;
    let prompts: Vec<Vec<u32>> = (0..n).map(|i| (0..len + DECODE_STEPS).map(|j| base[(j + i * 131) % base.len()]).collect()).collect();
    let mut logits: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
    for dtype in DTYPES {
        let planes = if calibrate && !dtype.is_fp8() { 2 } else { 1 };
        let dtype = if calibrate { KvDtype::F16 } else { dtype };
        let mut rt = load_planes(spec, dtype, len + DECODE_STEPS + 16, 1, planes)?;
        let vocab = spec.dims.vocab;
        let mut out = Vec::new();
        for p in &prompts {
            let flat = rt.prefill_then_decode(&p[..len], &p[len..])?;
            out.push(flat.chunks(vocab).map(|c| c.to_vec()).collect::<Vec<_>>());
        }
        logits.push(out);
    }
    let (mut sum, mut max, mut agree, mut total) = (0.0f64, 0.0f64, 0usize, 0usize);
    for (pa, pb) in logits[0].iter().zip(&logits[1]) {
        let per: Vec<String> = pa.iter().zip(pb).map(|(x, y)| format!("{:.4}", kl(x, y))).collect();
        println!("  per position kl: {}", per.join(" "));
        for (x, y) in pa.iter().zip(pb) {
            let d = kl(x, y);
            sum += d;
            max = max.max(d);
            total += 1;
            agree += (argmax(x) == argmax(y)) as usize;
        }
    }
    println!("kl len={len} prompts={n} positions={total} mean_kl={:.6} max_kl={:.6} top1_agree={:.4}", sum / total as f64, max, agree as f64 / total as f64);
    Ok(())
}

fn needle_prompt(tok: &Tokenizer, len: usize, depth: f64, code: &str) -> Result<Vec<u32>> {
    let question = encode(tok, "\n\nQuestion: What is the secret passcode?\nAnswer: The secret passcode is")?;
    let needle = encode(tok, &format!("\nThe secret passcode is {code}. Remember it.\n"))?;
    let mut out = Vec::new();
    let mut i = 0;
    let target_before = (len as f64 * depth) as usize;
    let filler = |i: usize| encode(tok, &format!("Note {i}: the garden path was quiet on day {}, and the river ran {} metres below the bridge.\n", i * 7 % 365, i * 13 % 90));
    while out.len() < target_before {
        out.extend(filler(i)?);
        i += 1;
    }
    out.extend(needle);
    while out.len() + question.len() < len {
        out.extend(filler(i)?);
        i += 1;
    }
    out.truncate(len - question.len());
    out.extend(question);
    Ok(out)
}

fn needle_mode(spec: &Spec, len: usize, depths: &[f64]) -> Result<()> {
    let tok = tokenizer(spec)?;
    for dtype in DTYPES {
        let mut rt = load(spec, dtype, len + 64, 1)?;
        for &depth in depths {
            let code = format!("{}", 1000 + (depth * 1000.0) as u32 * 7 % 9000);
            let prompt = needle_prompt(&tok, len, depth, &code)?;
            let g = rt.generate(&prompt, 48, Sampling::GREEDY)?;
            let text = tok.decode(&g.tokens, true).map_err(|e| anyhow::anyhow!("{e}"))?;
            let found = text.contains(&code);
            println!(
                "needle kv={} len={len} depth={depth} found={found} prefill={:.1} tok/s decode={:.1} tok/s answer={:?}",
                name(dtype),
                len as f64 / g.prefill_secs,
                (g.tokens.len().max(2) - 1) as f64 / g.decode_secs,
                text.chars().take(40).collect::<String>()
            );
        }
    }
    Ok(())
}

fn batch_mode(spec: &Spec, len: usize, batch: usize, new_tokens: usize) -> Result<()> {
    for dtype in DTYPES {
        let mut rt = load(spec, dtype, len + new_tokens, batch)?;
        let prompts: Vec<Vec<u32>> = (0..batch as u32).map(|s| (0..len as u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect()).collect();
        rt.generate_batch(&prompts, 4)?;
        let g = rt.generate_batch(&prompts, new_tokens)?;
        let rate = (batch * (new_tokens - 1)) as f64 / g.decode_secs;
        println!("batch kv={} len={len} batch={batch} decode={rate:.1} tok/s aggregate ({:.1} per sequence)", name(dtype), rate / batch as f64);
    }
    Ok(())
}

fn main() -> Result<()> {
    let usage = "usage: kvgate <spec.toml> kl <len> <prompts> | needle <len> <depth,..> | batch <len> <batch> <new>";
    let mut args = std::env::args().skip(1);
    let spec = Spec::load(&PathBuf::from(args.next().context(usage)?))?;
    let mode = args.next().context(usage)?;
    let mut num = |args: &mut std::iter::Skip<std::env::Args>| -> Result<usize> { Ok(args.next().context(usage)?.parse()?) };
    match mode.as_str() {
        "kl" => {
            let (len, n) = (num(&mut args)?, num(&mut args)?);
            let calibrate = args.next().is_some_and(|a| a == "f16-planes");
            kl_mode(&spec, len, n, calibrate)
        }
        "needle" => {
            let len = num(&mut args)?;
            let depths: Vec<f64> = args.next().context(usage)?.split(',').map(|d| d.parse()).collect::<Result<_, _>>()?;
            needle_mode(&spec, len, &depths)
        }
        "batch" => {
            let (len, b, new) = (num(&mut args)?, num(&mut args)?, num(&mut args)?);
            batch_mode(&spec, len, b, new)
        }
        _ => bail!(usage),
    }
}
