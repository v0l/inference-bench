use anyhow::{Context, Result, bail};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::sched::{Config, Event, Executor, GpuExecutor, Request, Scheduler};
use specrt::spec::Spec;
use specrt::tp_exec::TpExecutor;
use std::time::Instant;

struct Args {
    spec: String,
    tp: usize,
    batch: usize,
    decode: usize,
    prompt: usize,
    compare: Option<Vec<usize>>,
    max_ctx: usize,
}

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mut a = Args { spec: it.next().context("usage: tpbench SPEC.toml [--tp N] [--batch B] [--decode N] [--prompt N] [--compare 1,2]")?, tp: 1, batch: 1, decode: 128, prompt: 512, compare: None, max_ctx: 8192 };
    while let Some(k) = it.next() {
        let v = it.next().with_context(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--tp" => a.tp = v.parse()?,
            "--batch" => a.batch = v.parse()?,
            "--decode" => a.decode = v.parse()?,
            "--prompt" => a.prompt = v.parse()?,
            "--max-ctx" => a.max_ctx = v.parse()?,
            "--compare" => a.compare = Some(v.split(',').map(|t| t.parse()).collect::<Result<_, _>>()?),
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(a)
}

fn prompt(vocab: usize, len: usize, seed: usize) -> Vec<u32> {
    (0..len).map(|j| ((j * 7919 + seed * 104729 + 13) % (vocab - 1000) + 500) as u32).collect()
}

fn layout(spec: &Spec, a: &Args, tp: usize) -> Result<Layout> {
    let rank = specrt::tp::rank_spec(spec, tp)?;
    let mut l = Layout::serving(&rank, a.max_ctx, a.batch.max(1), None);
    l.kv_blocks = a.batch.max(1) * a.max_ctx.div_ceil(specrt::codegen::PAGE) + 8;
    l.prefill_chunk = l.prefill_chunk.min(4096);
    Ok(l)
}

fn run<E: Executor>(exec: E, spec: &Spec, a: &Args, tp: usize) -> Result<()> {
    let vocab = spec.dims.vocab;
    let mut sched = Scheduler::new(exec, Config { max_num_seqs: a.batch, max_batched_tokens: 8192, whole_prompt_tokens: 8192, ..Config::default() });
    let mut sink = Vec::new();
    sched.submit(Request { prompt: prompt(vocab, 64, 99), max_tokens: 8, sampling: Sampling::GREEDY, stop: vec![], salt: "warm".into(), media: Vec::new(), grammar: None })?;
    while !sched.idle() {
        sched.step(&mut sink)?;
    }
    let started = Instant::now();
    for b in 0..a.batch {
        sched.submit(Request { prompt: prompt(vocab, a.prompt, b), max_tokens: a.decode, sampling: Sampling::GREEDY, stop: vec![], salt: format!("p{b}"), media: Vec::new(), grammar: None })?;
    }
    let (mut prefill_secs, mut prefill_tokens, mut decode_secs, mut decode_tokens) = (0.0, 0usize, 0.0, 0usize);
    let mut first_token = None;
    while !sched.idle() {
        let mut events = Vec::new();
        let r = sched.step(&mut events)?;
        if r.prefill_tokens > 0 {
            prefill_secs += r.prefill_secs + r.decode_secs;
            prefill_tokens += r.prefill_tokens;
        } else {
            decode_secs += r.decode_secs;
            decode_tokens += r.decode_rows + r.accepted;
        }
        if first_token.is_none() && events.iter().any(|e| matches!(e, Event::Token { .. })) {
            first_token = Some(started.elapsed().as_secs_f64());
        }
    }
    println!(
        "{} tp={tp} batch={} prompt={} decode={} prefill={:.0} tok/s ({:.0} ms) decode={:.1} tok/s aggregate ({:.1} per stream, {:.2} ms/step) ttft={:.0} ms",
        spec.name,
        a.batch,
        a.prompt,
        a.decode,
        prefill_tokens as f64 / prefill_secs.max(1e-9),
        prefill_secs * 1e3,
        decode_tokens as f64 / decode_secs.max(1e-9),
        decode_tokens as f64 / decode_secs.max(1e-9) / a.batch as f64,
        decode_secs * 1e3 / (decode_tokens as f64 / a.batch as f64).max(1.0),
        first_token.unwrap_or(0.0) * 1e3,
    );
    Ok(())
}

fn logits_for(spec: &Spec, a: &Args, tp: usize, tokens: &[u32]) -> Result<Vec<f32>> {
    let l = layout(spec, a, tp)?;
    if tp == 1 {
        let mut rt = Runtime::load_layout(spec, l)?;
        rt.acquire_state(0)?;
        rt.prompt_logits(tokens)
    } else {
        TpExecutor::load(spec, l, &(0..tp).collect::<Vec<_>>())?.prompt_logits(tokens)
    }
}

fn softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let e: Vec<f64> = row.iter().map(|&x| (x as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn compare(spec: &Spec, a: &Args, tps: &[usize]) -> Result<()> {
    let tokens = prompt(spec.dims.vocab, a.prompt, 7);
    let base = logits_for(spec, a, tps[0], &tokens)?;
    let v = spec.dims.vocab;
    for &tp in &tps[1..] {
        let got = logits_for(spec, a, tp, &tokens)?;
        let rows = tokens.len();
        let (mut top1, mut kl, mut worst) = (0usize, 0.0f64, 0.0f64);
        for r in 0..rows {
            let (p, q) = (softmax(&base[r * v..(r + 1) * v]), softmax(&got[r * v..(r + 1) * v]));
            let am = |x: &[f64]| (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b });
            top1 += usize::from(am(&p) == am(&q));
            let k: f64 = p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-30)).ln()).sum();
            kl += k;
            worst = worst.max(k);
        }
        println!("{} tp{tp} vs tp{}: top-1 {top1}/{rows}, mean KL {:.3e}, max KL {:.3e}", spec.name, tps[0], kl / rows as f64, worst);
    }
    Ok(())
}

fn main() -> Result<()> {
    let a = parse()?;
    let spec = Spec::load(std::path::Path::new(&a.spec))?;
    if let Some(tps) = a.compare.clone() {
        return compare(&spec, &a, &tps);
    }
    let l = layout(&spec, &a, a.tp)?;
    if a.tp == 1 {
        run(GpuExecutor::new(Runtime::load_layout(&spec, l)?)?, &spec, &a, 1)
    } else {
        run(TpExecutor::load(&spec, l, &(0..a.tp).collect::<Vec<_>>())?, &spec, &a, a.tp)
    }
}
