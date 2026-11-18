use anyhow::{Context, Result, bail};
use specrt::runtime::{Layout, Runtime, Sampling};
use specrt::spec::{KvDtype, Spec};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const DTYPES: [KvDtype; 2] = [KvDtype::F16, KvDtype::Fp8];

fn decode_steps() -> usize {
    std::env::var("KVGATE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn arm_layers() -> Option<usize> {
    std::env::var("KVGATE_F16_LAYERS").ok().and_then(|v| v.parse().ok())
}

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
    if dtype.is_fp8() {
        s.runtime.kv_f16_layers = arm_layers().or(s.runtime.kv_f16_layers);
    }
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
    let steps = decode_steps();
    let corpus = std::env::var("KVGATE_CORPUS").is_ok();
    let base = if corpus { agentic_transcript(&tokenizer(spec)?, len + steps + 64 * n, std::env::var("KVGATE_CORPUS")?.as_str())? } else { text_prompts(spec)? };
    let only: Option<usize> = std::env::var("KVGATE_ONLY").ok().and_then(|v| v.parse().ok());
    let prompts: Vec<Vec<u32>> = (0..n).filter(|i| only.is_none_or(|o| o == *i)).map(|i| (0..len + steps).map(|j| base[(j + i * if corpus { 0 } else { 131 }) % base.len()]).collect()).collect();
    let mut logits: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
    for dtype in DTYPES {
        let planes = if calibrate && !dtype.is_fp8() { 2 } else { 1 };
        let dtype = if calibrate { KvDtype::F16 } else { dtype };
        let mut rt = load_planes(spec, dtype, len + steps + 16, 1, planes)?;
        let vocab = spec.dims.vocab;
        let mut out = Vec::new();
        for p in &prompts {
            let flat = rt.prefill_then_decode(&p[..len], &p[len..])?;
            out.push(flat.chunks(vocab).map(|c| c.to_vec()).collect::<Vec<_>>());
        }
        logits.push(out);
    }
    let (mut sum, mut max, mut agree, mut total, mut over) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize);
    for (pa, pb) in logits[0].iter().zip(&logits[1]) {
        let per: Vec<String> = pa.iter().zip(pb).map(|(x, y)| format!("{:.4}", kl(x, y))).collect();
        println!("  per position kl: {}", per.join(" "));
        let top: Vec<String> = pa.iter().map(|x| format!("{:.3}", log_softmax(x).into_iter().fold(f64::MIN, f64::max).exp())).collect();
        println!("  reference top1 prob: {}", top.join(" "));
        let gap: Vec<String> = pa.iter().zip(pb).map(|(x, y)| format!("{}", (argmax(x) == argmax(y)) as u8)).collect();
        println!("  top1 agree:          {}", gap.join(" "));
        for (x, y) in pa.iter().zip(pb) {
            let d = kl(x, y);
            sum += d;
            max = max.max(d);
            over += (d > 0.1) as usize;
            total += 1;
            agree += (argmax(x) == argmax(y)) as usize;
        }
    }
    println!("kl len={len} prompts={n} positions={total} mean_kl={:.6} max_kl={:.6} top1_agree={:.4} above_0.1={over}", sum / total as f64, max, agree as f64 / total as f64);
    Ok(())
}

fn collect_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs" || e == "md" || e == "toml" || e == "sh") {
            out.push(p);
        }
    }
}

const NARRATION: [&str; 6] = [
    "Writing the {} script now. I need to keep the viewstate and the event validation fields from the previous response, post them back with the form values, and check the redirect before moving to the next step.",
    "Next I will read {} to see how it handles the request, then adjust the parameters and run it again to compare the output with the last attempt.",
    "The last run of {} failed on the second step. Checking the headers and the cookie jar before retrying with the corrected payload.",
    "Writing the {} helper. It takes the page, extracts the hidden fields, builds the form body and returns the next url so the caller can follow it.",
    "Looking at {} again. The structure is clear now: parse, validate, submit, then verify the confirmation page contains the order number.",
    "Running the tests for {}. Two of them are slow because they start the full pipeline, so I will filter to the relevant module first.",
];

fn agentic_transcript(tok: &Tokenizer, len: usize, root: &str) -> Result<Vec<u32>> {
    let root = if root.is_empty() || root == "1" { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..") } else { PathBuf::from(root) };
    let mut files = Vec::new();
    collect_files(&root, &mut files);
    anyhow::ensure!(!files.is_empty(), "no corpus files under {}", root.display());
    let mut out = encode(tok, "<|im_start|>system\nYou are a coding agent with shell, read_file and edit_file tools. Work step by step.<|im_end|>\n<|im_start|>user\nAutomate the checkout flow of the order portal and keep notes as you go.<|im_end|>\n")?;
    let mut turn = 0usize;
    while out.len() < len {
        let path = &files[(turn * 7) % files.len()];
        let body = std::fs::read_to_string(path).unwrap_or_default();
        let body: String = body.chars().take(6000).collect();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let narration = NARRATION[turn % NARRATION.len()].replace("{}", &name);
        let call = format!(
            "<|im_start|>assistant\n<think>\n{narration}\n</think>\n\n{narration}\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n{}\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n",
            path.display()
        );
        out.extend(encode(tok, &call)?);
        turn += 1;
    }
    out.truncate(len);
    Ok(out)
}

fn loop_mode(spec: &Spec, len: usize, seeds: usize, new_tokens: usize, arms: &[&str]) -> Result<()> {
    let tok = tokenizer(spec)?;
    let root = std::env::var("KVGATE_CORPUS").unwrap_or_default();
    let tail = encode(&tok, "<|im_start|>assistant\n<think>\n")?;
    let mut prompt = agentic_transcript(&tok, len - tail.len(), &root)?;
    prompt.extend(tail);
    let stops = [248046u32, 248044];
    for arm in arms {
        let (dtype, layers) = match *arm {
            "f16" => (KvDtype::F16, None),
            "fp8" => (KvDtype::Fp8, None),
            other => (KvDtype::Fp8, Some(other.trim_start_matches("fp8:").parse::<usize>()?)),
        };
        let mut s = spec.clone();
        s.runtime.kv_dtype = dtype;
        s.runtime.kv_f16_layers = layers.or(s.runtime.kv_f16_layers);
        let mut rt = load(&s, dtype, len + new_tokens + 16, 1)?;
        for seed in 0..seeds as u32 {
            let sampling = Sampling::new(1.0, seed + 1).truncated(20, 0.95, 0.0);
            let g = rt.generate_until(&prompt, new_tokens, sampling, &stops, &mut |_| true)?;
            let report = specrt::loops::detect(&g.tokens);
            let text = tok.decode(&g.tokens, false).map_err(|e| anyhow::anyhow!("{e}"))?;
            println!(
                "loop arm={arm} len={len} seed={seed} tokens={} repeat_ratio={:.3} worst_repeats={} looping={} prefill={:.0}s",
                g.tokens.len(),
                report.repeat_ratio,
                report.worst_repeats,
                report.looping,
                g.prefill_secs
            );
            println!("  head: {:?}", text.chars().take(160).collect::<String>());
            println!("  tail: {:?}", text.chars().rev().take(160).collect::<Vec<_>>().into_iter().rev().collect::<String>());
        }
    }
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
    let only = std::env::var("KVGATE_DTYPE").ok();
    for dtype in DTYPES.into_iter().filter(|d| only.as_deref().is_none_or(|o| o == name(*d))) {
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
    let usage = "usage: kvgate <spec.toml> kl <len> <prompts> | needle <len> <depth,..> | batch <len> <batch> <new> | loop <len> <seeds> <new> [f16,fp8,fp8:16]";
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
        "loop" => {
            let (len, seeds, new) = (num(&mut args)?, num(&mut args)?, num(&mut args)?);
            let arms = args.next().unwrap_or_else(|| "f16,fp8".into());
            loop_mode(&spec, len, seeds, new, &arms.split(',').collect::<Vec<_>>())
        }
        "batch" => {
            let (len, b, new) = (num(&mut args)?, num(&mut args)?, num(&mut args)?);
            batch_mode(&spec, len, b, new)
        }
        _ => bail!(usage),
    }
}
