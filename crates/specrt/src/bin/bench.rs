use anyhow::{Context, Result};
use specrt::runtime::{Generation, Runtime, Sampling};
use specrt::spec::Spec;
use std::path::{Path, PathBuf};

struct Flags {
    runs: usize,
    serve: bool,
    cold: bool,
    profile: bool,
    fp8_act: bool,
    kv_gb: Option<f64>,
    mlp_act_fp4: bool,
    mlp_planes: u8,
    pdl: bool,
    ab: bool,
    rounds: usize,
    b_graphs: Option<bool>,
    drafts: Option<specrt::serve::DraftTokens>,
    vram_gb: Option<f64>,
    expert_slots: Option<usize>,
    host_gb: Option<f64>,
    nvme_path: Option<String>,
    layers: Option<usize>,
    hotlist: Option<String>,
    save_hotlist: Option<String>,
    a_set: Vec<String>,
    b_set: Vec<String>,
}

static FLAGS: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();

fn flags() -> &'static Flags {
    FLAGS.get().expect("flags are parsed first")
}

fn runs() -> usize {
    flags().runs
}

fn take_flags(args: &mut Vec<String>) -> Result<Flags> {
    let mut f = Flags { runs: 6, serve: false, cold: false, profile: false, fp8_act: false, kv_gb: None, mlp_act_fp4: false, mlp_planes: 1, pdl: false, ab: false, rounds: specrt::abstats::MIN_ROUNDS, b_graphs: None, drafts: None, vram_gb: None, expert_slots: None, host_gb: None, nvme_path: None, layers: None, hotlist: None, save_hotlist: None, a_set: Vec::new(), b_set: Vec::new() };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--serve" => f.serve = true,
            "--cold" => f.cold = true,
            "--profile" => f.profile = true,
            "--fp8-act" => f.fp8_act = true,
            "--pdl" => f.pdl = true,
            "--ab" => f.ab = true,
            "--mlp-act-fp4" => f.mlp_act_fp4 = true,
            "--mlp-act-planes" => f.mlp_planes = 2,
            "--runs" | "--kv-gb" | "--rounds" | "--b-graphs" | "--drafts" | "--vram-gb" | "--expert-slots" | "--host-gb" | "--nvme-path" | "--layers" | "--hotlist" | "--save-hotlist" | "--a-set" | "--b-set" => {
                let name = args.remove(i);
                anyhow::ensure!(i < args.len(), "{name} needs a value");
                if name == "--runs" {
                    f.runs = args[i].parse().context("--runs")?;
                } else if name == "--rounds" {
                    f.rounds = args[i].parse().context("--rounds")?;
                } else if name == "--vram-gb" {
                    f.vram_gb = Some(args[i].parse().context("--vram-gb")?);
                } else if name == "--host-gb" {
                    f.host_gb = Some(args[i].parse().context("--host-gb")?);
                } else if name == "--nvme-path" {
                    f.nvme_path = Some(args[i].clone());
                } else if name == "--layers" {
                    f.layers = Some(args[i].parse().context("--layers")?);
                } else if name == "--expert-slots" {
                    f.expert_slots = Some(args[i].parse().context("--expert-slots")?);
                } else if name == "--hotlist" {
                    f.hotlist = Some(args[i].clone());
                } else if name == "--save-hotlist" {
                    f.save_hotlist = Some(args[i].clone());
                } else if name == "--a-set" {
                    f.a_set.push(args[i].clone());
                } else if name == "--b-set" {
                    f.b_set.push(args[i].clone());
                } else if name == "--drafts" {
                    f.drafts = Some(match args[i].as_str() {
                        "auto" => specrt::serve::DraftTokens::Auto,
                        n => specrt::serve::DraftTokens::Fixed(n.parse().context("--drafts takes auto or a number")?),
                    });
                } else if name == "--b-graphs" {
                    f.b_graphs = Some(match args[i].as_str() {
                        "on" => true,
                        "off" => false,
                        other => anyhow::bail!("--b-graphs takes on or off, not {other}"),
                    });
                } else {
                    f.kv_gb = Some(args[i].parse().context("--kv-gb")?);
                }
            }
            _ => {
                i += 1;
                continue;
            }
        }
        args.remove(i);
    }
    Ok(f)
}

fn load_spec(path: &Path) -> Result<Spec> {
    let mut spec = load_spec_file(path)?;
    if flags().pdl {
        spec.runtime.pdl = true;
    }
    if let Some(gb) = flags().vram_gb {
        spec.runtime.offload.vram_budget = (gb * 1e9) as u64;
    }
    if let Some(gb) = flags().host_gb {
        spec.runtime.offload.host_budget = (gb * 1e9) as u64;
    }
    if let Some(p) = &flags().nvme_path {
        spec.runtime.offload.nvme_path = Some(p.clone());
    }
    if let Some(n) = flags().layers {
        spec.layers.count = n;
    }
    if let Some(n) = flags().expert_slots {
        spec.runtime.offload.expert_slots = n;
    }
    if let Some(p) = &flags().hotlist {
        spec.runtime.offload.hotlist = Some(p.clone());
    }
    Ok(spec)
}

fn load_spec_file(path: &Path) -> Result<Spec> {
    if path.extension().is_some_and(|e| e == "gguf") {
        return Spec::from_gguf_file(path);
    }
    if path.is_dir() {
        let config = std::fs::read_to_string(path.join("config.json")).with_context(|| path.display().to_string())?;
        let mut spec = Spec::from_hf_config_serving(&config)?;
        spec.name = path.file_name().and_then(|n| n.to_str()).unwrap_or("checkpoint").to_string();
        spec.repo = path.display().to_string();
        if flags().fp8_act {
            spec.mlp.act_quant = specrt::spec::ActQuant::Fp8;
        }
        if flags().mlp_act_fp4 {
            spec.mlp.act_quant = specrt::spec::ActQuant::Fp4;
            spec.mlp.up_act_planes = flags().mlp_planes;
            spec.mlp.down_act_planes = flags().mlp_planes;
            spec.validate()?;
        }
        return Ok(spec);
    }
    Spec::load(path)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let _ = FLAGS.set(take_flags(&mut args)?);
    let speculative = args.first().is_some_and(|a| a == "--speculative");
    if speculative {
        args.remove(0);
    }
    let sampled = args.first().is_some_and(|a| a == "--sampled");
    if sampled {
        args.remove(0);
    }
    let wgpu = args.first().is_some_and(|a| a == "--wgpu");
    if wgpu {
        args.remove(0);
    }
    if args.first().is_some_and(|a| a == "--text") {
        let model = args.get(1).context("--text <model> <prompt> [tokens]")?;
        let prompt = args.get(2).context("--text <model> <prompt> [tokens]")?;
        let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(32);
        let mut engine = specrt::engine::Engine::load(model, Some(512))?;
        println!("PROMPT_IDS {:?}", engine.tokens(prompt)?);
        let reply = engine.generate(prompt, n, Sampling::GREEDY)?;
        println!("TEXT_BEGIN{}TEXT_END", reply.text);
        return Ok(());
    }
    if flags().ab && (!flags().a_set.is_empty() || !flags().b_set.is_empty()) {
        return bench_ab_option(args);
    }
    if flags().ab {
        return bench_ab(args);
    }
    let mut args = args.into_iter();
    let usage = "usage: bench [--wgpu] [--runs N] [--serve] [--kv-gb GB] [--profile] [--speculative] [--sampled] <spec.toml | checkpoint dir | model.gguf> [prompt_len[,prompt_len..]] [new_tokens] [batch]";
    let spec_path = PathBuf::from(args.next().context(usage)?);
    let prompts: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "128".into())
        .split(',')
        .map(|s| s.parse().context(usage))
        .collect::<Result<_>>()?;
    let new_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
    let batch: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0);
    let spec = load_spec(&spec_path)?;
    let longest = *prompts.iter().max().context(usage)?;
    if wgpu {
        return bench_wgpu(&spec, &prompts, new_tokens);
    }
    if speculative {
        return bench_spec(&spec, new_tokens, batch.max(1), longest, sampled);
    }
    if flags().cold {
        return bench_cold(&spec, longest, new_tokens);
    }
    if flags().serve {
        return bench_serve(&spec, longest, new_tokens, batch.max(1), sampled);
    }
    if batch > 0 {
        return bench_batch(&spec, longest, new_tokens, batch);
    }
    let mut rt = Runtime::load(&spec, longest + new_tokens)?;
    if let Some(plan) = rt.offload_plan() {
        println!("offload: {} of {} layers streamed, staging {:.2} GB, per token {:.2} GB over the link", rt.streamed_layers(), plan.layers.len(), rt.staging_bytes() as f64 / 1e9, plan.streamed_bytes_per_token() as f64 / 1e9);
    }
    println!(
        "{} quant={:?} arch={} jit_cache_hit={} jit_compile={:.2}s jit_load_module={:.2}s load={:.2}s weights={:.2} GB in {:.2}s device={:.2} GB peak_host_rss={:.2} GB",
        spec.name,
        spec.quant,
        rt.arch,
        rt.cache_hit,
        rt.jit_compile_secs,
        rt.jit_module_secs,
        rt.load_secs,
        rt.weight_bytes as f64 / 1e9,
        rt.weight_secs,
        rt.device_bytes as f64 / 1e9,
        rt.peak_host_rss_bytes as f64 / 1e9,
    );
    for &prompt_len in &prompts {
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
        rt.generate(&prompt, 8.min(new_tokens), Sampling::GREEDY)?;
        let runs: Vec<Generation> =
            (0..runs()).map(|_| rt.generate(&prompt, new_tokens, Sampling::GREEDY)).collect::<Result<_>>()?;
        let prefill = median(runs.iter().map(|g| prompt_len as f64 / g.prefill_secs).collect());
        let decode = median(runs.iter().map(|g| (new_tokens - 1) as f64 / g.decode_secs).collect());
        let peak = runs.iter().map(|g| g.peak_device_bytes).max().unwrap_or(0);
        let all = |f: &dyn Fn(&Generation) -> f64| runs.iter().map(|g| format!("{:.1}", f(g))).collect::<Vec<_>>().join(",");
        println!(
            "  prompt={prompt_len} prefill={prefill:.1} tok/s decode={new_tokens} tokens {decode:.1} tok/s peak_device={:.2} GB (median of {}) prefill_runs=[{}] decode_runs=[{}]",
            peak as f64 / 1e9,
            runs.len(),
            all(&|g| prompt_len as f64 / g.prefill_secs),
            all(&|g| (new_tokens - 1) as f64 / g.decode_secs),
        );
    }
    if let Some(s) = rt.offload_stats() {
        println!("  offload stats: {s:?} hit_ratio={:.4}", s.hit_ratio());
        println!("  nvme: reads={} bytes={} failures={} wait={:.3}s", s.nvme_reads, s.nvme_read_bytes, s.nvme_read_failures, s.nvme_wait_nanos as f64 / 1e9);
    }
    if let Some(p) = &flags().save_hotlist {
        rt.save_expert_hotlist(std::path::Path::new(p))?;
    }
    {
        let prompt: Vec<u32> = (0..longest as u32).map(|i| 1000 + i * 7 % 5000).collect();
        let g = rt.generate(&prompt, new_tokens, Sampling::GREEDY)?;
        println!("  first64 {:?}", &g.tokens[..64.min(g.tokens.len())]);
    }
    Ok(())
}

fn spec_prompts(spec: &Spec) -> Result<Vec<Vec<u32>>> {
    let path = specrt::weights::find_in_snapshot(&spec.repo, "tokenizer.json")?;
    let tok = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let text = include_str!("../../scripts/spec_prompts.txt");
    text.split("\n---\n")
        .map(|p| Ok(tok.encode(p.trim_end(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()))
        .collect()
}

fn bench_spec(spec: &Spec, new_tokens: usize, batch: usize, max_ctx: usize, sampled: bool) -> Result<()> {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let prompts = spec_prompts(spec)?;
    for speculate in [true, false] {
        let mut s = spec.clone();
        let mut adaptive = false;
        if let Some(d) = flags().drafts {
            let o = specrt::serve::SpeculativeOptions { draft_tokens: Some(d), ..Default::default() };
            o.apply(&mut s)?;
            adaptive = d == specrt::serve::DraftTokens::Auto;
        }
        if let Some(m) = s.mtp.as_mut() {
            m.speculate = speculate;
        }
        let layout = specrt::runtime::Layout::serving(&s, max_ctx, batch, Some(1 << 30));
        let rt = Runtime::load_layout(&s, layout)?;
        let mut sched = Scheduler::new(GpuExecutor::new(rt)?, Config { max_num_seqs: batch, adaptive_drafts: adaptive, ..Config::default() });
        let mut warm = Vec::new();
        sched.submit(Request { prompt: prompts[0].clone(), max_tokens: 16, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None })?;
        while !sched.idle() {
            sched.step(&mut warm)?;
        }
        let before = sched.exec.rt.spec_stats();
        let (mut decode_secs, mut tokens, mut drafted, mut accepted) = (0.0, 0usize, 0usize, 0usize);
        let mut outs: std::collections::HashMap<u64, Vec<u32>> = Default::default();
        let mut ids = Vec::new();
        for (n, p) in prompts.iter().enumerate() {
            let sampling = if sampled { Sampling::new(1.0, 100 + n as u32).truncated(20, 0.95, 0.0) } else { Sampling::GREEDY };
            ids.push(sched.submit(Request { prompt: p.clone(), max_tokens: new_tokens, sampling, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None })?);
        }
        while !sched.idle() {
            let mut events = Vec::new();
            let r = sched.step(&mut events)?;
            if r.prefill_tokens == 0 {
                decode_secs += r.decode_secs;
                tokens += r.decode_rows + r.accepted;
            }
            drafted += r.drafted;
            accepted += r.accepted;
            for e in events {
                if let Event::Token { id, token } = e {
                    outs.entry(id).or_default().push(token);
                }
            }
        }
        let after = sched.exec.rt.spec_stats();
        if let Some(c) = sched.draft_chooser() {
            println!("  adaptive steps per k (0..=3) {:?}", c.chosen());
        }
        let rows = (after.rows - before.rows).max(1) as f64;
        println!(
            "{} speculate={speculate} k={} batch={batch} prompts={} new={new_tokens} decode={:.1} tok/s aggregate ({:.1} per sequence) mean_acceptance_length={:.3} draft_acceptance={:.3} drafted={drafted} accepted={accepted}",
            spec.name,
            s.mtp.as_ref().map_or(0, |m| m.draft_tokens),
            prompts.len(),
            tokens as f64 / decode_secs,
            tokens as f64 / decode_secs / batch as f64,
            1.0 + (after.accepted - before.accepted) as f64 / rows,
            accepted as f64 / drafted.max(1) as f64,
        );
        let k = s.mtp.as_ref().map_or(0, |m| m.draft_tokens);
        let per_position: Vec<f64> = (0..k.min(4)).map(|j| (after.position_accepted[j] - before.position_accepted[j]) as f64 / rows).collect();
        println!(
            "  per_position_accepted={:?} per_position_proposed={rows} per_position_rate={per_position:.3?}",
            after.position_accepted[..k.min(4)].iter().zip(&before.position_accepted).map(|(a, b)| a - b).collect::<Vec<_>>(),
        );
        println!(
            "{}",
            serde_json::json!({
                "speculate": speculate, "k": k, "prompts": prompts.len(), "new_tokens": new_tokens,
                "steps": after.steps - before.steps, "rows": rows as u64,
                "mean_acceptance_length": 1.0 + (after.accepted - before.accepted) as f64 / rows,
                "per_position_accepted": (0..k.min(4)).map(|j| after.position_accepted[j] - before.position_accepted[j]).collect::<Vec<_>>(),
                "per_position_proposed": rows as u64,
                "per_position_rate": per_position,
            })
        );
        let first: Vec<u32> = outs[&ids[0]].iter().take(24).copied().collect();
        println!("  first prompt tokens {first:?}");
    }
    Ok(())
}

fn bench_cold(spec: &Spec, prompt_len: usize, new_tokens: usize) -> Result<()> {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let mut spec = spec.clone();
    if let Some(m) = spec.mtp.as_mut() {
        m.speculate = false;
    }
    let spec = &spec;
    let max_ctx = prompt_len + new_tokens + 16;
    let kv_bytes = flags().kv_gb.map(|gb| (gb * (1u64 << 30) as f64) as usize);
    let layout = specrt::runtime::Layout::serving(spec, max_ctx, 32, kv_bytes);
    let rt = Runtime::load_layout(spec, layout)?;
    println!(
        "{} cold_start load={:.2}s jit_cache_hit={} jit_compile={:.2}s jit_load_module={:.2}s weights={:.2}s",
        spec.name, rt.load_secs, rt.cache_hit, rt.jit_compile_secs, rt.jit_module_secs, rt.weight_secs
    );
    let mut sched = Scheduler::new(GpuExecutor::new(rt)?, Config { max_num_seqs: 32, ..Config::default() });
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
    let mut ttfts = Vec::new();
    for _ in 0..11 {
        let id = sched.submit(Request { prompt: prompt.clone(), max_tokens: new_tokens, sampling: Sampling::GREEDY, stop: vec![], salt: "cold".into(), media: Vec::new(), grammar: None })?;
        let started = std::time::Instant::now();
        let mut ttft = None;
        while !sched.idle() {
            let mut events = Vec::new();
            sched.step(&mut events)?;
            if ttft.is_none() && events.iter().any(|e| matches!(e, Event::Token { id: i, .. } if *i == id)) {
                ttft = Some(started.elapsed().as_secs_f64());
            }
        }
        ttfts.push(ttft.context("no token")?);
    }
    let rest = median(ttfts[1..].to_vec());
    println!("ttft_first={:.3}s ttft_next10_median={:.3}s ratio={:.1}x prompt={prompt_len}", ttfts[0], rest, ttfts[0] / rest);
    let mut n = 1;
    while n <= 32 {
        for i in 0..n {
            sched.submit(Request { prompt: prompt.clone(), max_tokens: 24, sampling: Sampling::GREEDY, stop: vec![], salt: format!("g{n}-{i}"), media: Vec::new(), grammar: None })?;
        }
        let mut events = Vec::new();
        while !sched.idle() {
            sched.step(&mut events)?;
        }
        n *= 2;
    }
    let secs: Vec<String> = sched.exec.capture_secs().iter().map(|(n, s)| format!("{n}:{s:.3}")).collect();
    println!("graph_capture_secs rows:secs [{}]", secs.join(" "));
    let spec_secs: Vec<String> = sched.exec.rt.speculation_capture_secs().iter().map(|(n, s)| format!("{n}:{s:.3}")).collect();
    println!("speculation_graph_capture_secs rows:secs [{}]", spec_secs.join(" "));
    Ok(())
}

fn bench_serve(spec: &Spec, prompt_len: usize, new_tokens: usize, batch: usize, sampled: bool) -> Result<()> {
    use specrt::sched::{Config, Event, GpuExecutor, Request, Scheduler};
    let max_ctx = prompt_len + new_tokens + 16;
    let kv_bytes = flags().kv_gb.map(|gb| (gb * (1u64 << 30) as f64) as usize);
    let layout = specrt::runtime::Layout::serving(spec, max_ctx, batch, kv_bytes);
    let rt = Runtime::load_layout(spec, layout)?;
    let mut sched = Scheduler::new(
        GpuExecutor::new(rt)?,
        Config { max_num_seqs: batch, max_batched_tokens: 16384, step_cost: 0.0, itl_target: std::time::Duration::from_secs(10), ..Config::default() },
    );
    for round in 0..runs() {
        for s in 0..batch as u32 {
            let prompt = (0..prompt_len as u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect();
            let sampling = if sampled { Sampling::new(1.0, 100 + s).truncated(20, 0.95, 0.0) } else { Sampling::GREEDY };
            sched.submit(Request { prompt, max_tokens: new_tokens, sampling, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None })?;
        }
        let (mut decode_secs, mut decode_tokens, mut prefill_secs, mut steps) = (0.0, 0, 0.0, 0);
        let (mut pure_secs, mut pure_tokens, mut pure_steps, mut preempted) = (0.0, 0, 0, 0);
        let mut profiling = false;
        let started = std::time::Instant::now();
        while !sched.idle() {
            let mut events = Vec::new();
            let r = sched.step(&mut events)?;
            decode_secs += r.decode_secs;
            decode_tokens += r.decode_rows + r.accepted;
            prefill_secs += r.prefill_secs;
            steps += 1;
            preempted += r.preemptions;
            if r.prefill_tokens == 0 && r.decode_rows == batch && !profiling && flags().profile {
                profiling = true;
                unsafe { cudarc::driver::sys::cuProfilerStart() };
            }
            if r.prefill_tokens == 0 && r.decode_rows == batch {
                pure_secs += r.decode_secs;
                pure_tokens += r.decode_rows + r.accepted;
                pure_steps += 1;
            }
            let _ = events.iter().filter(|e| matches!(e, Event::Token { .. })).count();
        }
        if flags().profile {
            unsafe { cudarc::driver::sys::cuProfilerStop() };
        }
        println!(
            "{} serve round={round} batch={batch} prompt={prompt_len} new={new_tokens} steps={steps} decode={:.1} tok/s aggregate (decode step {:.2} ms, {decode_secs:.3} s decoding) full-batch decode={:.1} tok/s over {pure_steps} steps of {:.2} ms preemptions={preempted} kv_blocks={} prefill {:.2} s wall {:.2} s",
            spec.name,
            decode_tokens as f64 / decode_secs,
            decode_secs * 1e3 / (steps as f64),
            pure_tokens as f64 / pure_secs,
            pure_secs * 1e3 / pure_steps.max(1) as f64,
            sched.exec.rt.kv_blocks(),
            prefill_secs,
            started.elapsed().as_secs_f64(),
        );
    }
    Ok(())
}

fn bench_batch(spec: &Spec, prompt_len: usize, new_tokens: usize, batch: usize) -> Result<()> {
    let mut rt = Runtime::load_batched(spec, prompt_len + new_tokens, batch)?;
    let prompts: Vec<Vec<u32>> = (0..batch as u32)
        .map(|s| (0..prompt_len as u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect())
        .collect();
    rt.generate_batch(&prompts, 8.min(new_tokens))?;
    let gens: Vec<_> = (0..runs()).map(|_| rt.generate_batch(&prompts, new_tokens)).collect::<Result<_>>()?;
    let rate = median(gens.iter().map(|g| (batch * (new_tokens - 1)) as f64 / g.decode_secs).collect());
    let prefill = median(gens.iter().map(|g| (batch * prompt_len) as f64 / g.prefill_secs).collect());
    println!(
        "{} prefill={prefill:.1} tok/s batch={} prompt={} new={} decode={:.1} tok/s aggregate, {:.1} per sequence (median of {})",
        spec.name,
        batch,
        prompt_len,
        new_tokens,
        rate,
        rate / batch as f64,
        runs()
    );
    Ok(())
}

#[cfg(feature = "wgpu")]
fn bench_wgpu(spec: &Spec, prompts: &[usize], new_tokens: usize) -> Result<()> {
    let started = std::time::Instant::now();
    let be = specrt::backends::wgpu::Wgpu::new(None)?;
    let mut dense = specrt::backends::dense::Dense::load(be, spec, prompts.iter().max().copied().unwrap_or(128) + new_tokens)?;
    println!("{} adapter={} load={:.2}s weights={:.2} GB", spec.name, dense.description(), started.elapsed().as_secs_f64(), dense.weight_bytes() as f64 / 1e9);
    for &prompt_len in prompts {
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
        dense.generate(&prompt, 8.min(new_tokens), Sampling::GREEDY)?;
        let runs: Vec<Generation> = (0..runs()).map(|_| dense.generate(&prompt, new_tokens, Sampling::GREEDY)).collect::<Result<_>>()?;
        let prefill = median(runs.iter().map(|g| prompt_len as f64 / g.prefill_secs).collect());
        let decode = median(runs.iter().map(|g| (new_tokens - 1) as f64 / g.decode_secs).collect());
        println!("  prompt={prompt_len} prefill={prefill:.1} tok/s decode={decode:.1} tok/s (median of {})", runs.len());
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn bench_wgpu(_: &Spec, _: &[usize], _: usize) -> Result<()> {
    anyhow::bail!("build with --features wgpu")
}

fn with_options(spec: &Spec, sets: &[String]) -> Result<(Spec, specrt::serve::ServeOptions)> {
    let mut spec = spec.clone();
    let mut opts = specrt::serve::ServeOptions::default();
    let mut speculative = serde_json::Map::new();
    let mut runtime = serde_json::to_value(&spec.runtime)?;
    let known: Vec<String> = specrt::options::schema().into_iter().map(|f| f.key).collect();
    let runtime_before = runtime.clone();
    let mut spec_value = serde_json::to_value(&spec)?;
    for set in sets {
        let (key, raw) = set.split_once('=').with_context(|| format!("--a-set/--b-set takes key=value, not {set}"))?;
        let value: serde_json::Value = serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        if let Some(path) = key.strip_prefix("spec.") {
            let mut node = &mut spec_value;
            let parts: Vec<&str> = path.split('.').collect();
            for (i, part) in parts.iter().enumerate() {
                let map = node.as_object_mut().with_context(|| format!("spec.{path} is not a spec table path"))?;
                if i + 1 == parts.len() {
                    map.insert(part.to_string(), value.clone());
                    break;
                }
                node = map.entry(part.to_string()).or_insert_with(|| serde_json::json!({}));
            }
            continue;
        }
        let full = if known.iter().any(|k| k == key) {
            key.to_string()
        } else {
            let hits: Vec<&String> = known.iter().filter(|k| k.rsplit('.').next() == Some(key)).collect();
            anyhow::ensure!(hits.len() == 1, "{key} is not one option key ({hits:?})");
            hits[0].clone()
        };
        if let Some(rest) = full.strip_prefix("speculative.") {
            speculative.insert(rest.to_string(), value);
        } else if let Some(path) = full.strip_prefix("runtime.") {
            let mut node = &mut runtime;
            let parts: Vec<&str> = path.split('.').collect();
            for (i, part) in parts.iter().enumerate() {
                let map = node.as_object_mut().context("runtime path")?;
                if i + 1 == parts.len() {
                    map.insert(part.to_string(), value.clone());
                    break;
                }
                node = map.entry(part.to_string()).or_insert_with(|| serde_json::json!({}));
            }
        } else {
            anyhow::bail!("bench --ab takes spec.*, speculative.* and runtime.* options, not {full}");
        }
    }
    spec = serde_json::from_value(spec_value).context("spec.* overrides")?;
    if runtime != runtime_before {
        spec.runtime = serde_json::from_value(runtime).context("runtime options")?;
    }
    spec.runnable()?;
    opts.speculative = serde_json::from_value(serde_json::Value::Object(speculative)).context("speculative options")?;
    Ok((spec, opts))
}

struct Arm {
    sched: specrt::sched::Scheduler<specrt::sched::GpuExecutor>,
    drafts: specrt::serve::DraftTokens,
}

fn load_arm(base: &Spec, sets: &[String], batch: usize, max_ctx: usize) -> Result<Arm> {
    let (spec, mut opts) = with_options(base, sets)?;
    opts.max_num_seqs = batch;
    opts.max_model_len = Some(max_ctx);
    opts.kv_cache_bytes = Some(1 << 30);
    let drafts = opts.speculative.draft_tokens.unwrap_or(specrt::serve::DraftTokens::Fixed(spec.mtp.as_ref().map_or(0, |m| m.draft_tokens)));
    Ok(Arm { sched: specrt::serve::scheduler(&spec, opts, 0)?, drafts })
}

fn option_round(arm: &mut Arm, prompts: &[Vec<u32>], new_tokens: usize) -> Result<f64> {
    use specrt::sched::Request;
    for p in prompts {
        arm.sched.submit(Request { prompt: p.clone(), max_tokens: new_tokens, sampling: Sampling::GREEDY, stop: vec![], salt: String::new(), media: Vec::new(), grammar: None })?;
    }
    let (mut secs, mut tokens) = (0.0, 0usize);
    while !arm.sched.idle() {
        let mut events = Vec::new();
        let r = arm.sched.step(&mut events)?;
        if r.prefill_tokens == 0 {
            secs += r.decode_secs;
            tokens += r.decode_rows + r.accepted;
        }
    }
    Ok(tokens as f64 / secs)
}

fn bench_ab_option(args: Vec<String>) -> Result<()> {
    use specrt::abstats::{ratios, summarize, verdict_with_min_rounds};
    let usage = "usage: bench --ab [--rounds N] [--a-set key=value].. [--b-set key=value].. <spec> [batch] [new_tokens]";
    let mut args = args.into_iter();
    let spec_path = PathBuf::from(args.next().context(usage)?);
    let batch: usize = args.next().map(|s| s.parse()).transpose().context(usage)?.unwrap_or(4);
    let new_tokens: usize = args.next().map(|s| s.parse()).transpose().context(usage)?.unwrap_or(128);
    let rounds = flags().rounds;
    let spec = load_spec(&spec_path)?;
    let prompts = spec_prompts(&spec)?;
    let max_ctx = prompts.iter().map(Vec::len).max().unwrap_or(0) + new_tokens + 16;
    let mut a = load_arm(&spec, &flags().a_set, batch, max_ctx)?;
    let mut b = load_arm(&spec, &flags().b_set, batch, max_ctx)?;
    println!("A {:?} drafts={:?} | B {:?} drafts={:?} | batch={batch} prompts={} new={new_tokens} rounds={rounds}", flags().a_set, a.drafts, flags().b_set, b.drafts, prompts.len());
    for arm in [&mut a, &mut b] {
        option_round(arm, &prompts, new_tokens)?;
    }
    let (mut control, mut passes, mut a_all, mut b_all) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for pass in 1..=2 {
        let (mut c1, mut c2, mut ra, mut rb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rounds {
            c1.push(option_round(&mut a, &prompts, new_tokens)?);
            c2.push(option_round(&mut a, &prompts, new_tokens)?);
        }
        for _ in 0..rounds {
            ra.push(option_round(&mut a, &prompts, new_tokens)?);
            rb.push(option_round(&mut b, &prompts, new_tokens)?);
        }
        control.push(summarize(&ratios(&c1, &c2)));
        passes.push(summarize(&ratios(&ra, &rb)));
        println!("pass {pass}: A/A {}", describe("ratio", &control[pass - 1]));
        println!("pass {pass}: A/B {}", describe("ratio", &passes[pass - 1]));
        a_all.extend(ra);
        b_all.extend(rb);
    }
    println!("A: {:.1} tok/s, B: {:.1} tok/s (median aggregate decode)", summarize(&a_all).median, summarize(&b_all).median);
    for (name, arm) in [("A", &a), ("B", &b)] {
        if let Some(c) = arm.sched.draft_chooser() {
            println!("{name}: adaptive steps per k (0..=3) {:?}", c.chosen());
        }
    }
    let v = verdict_with_min_rounds(&control, &passes, rounds);
    println!("verdict (decode, B over A): {}", v.label());
    anyhow::ensure!(!matches!(v, specrt::abstats::Verdict::Refused(_)), "refused: rerun on a quiet card");
    Ok(())
}

fn looks_like_spec(arg: &str) -> bool {
    let p = Path::new(arg);
    p.is_dir() || p.extension().is_some_and(|e| e == "toml" || e == "gguf")
}

struct Round {
    prefill: f64,
    decode: f64,
}

fn timed(rt: &mut Runtime, prompt: &[u32], new_tokens: usize) -> Result<Round> {
    let g = rt.generate(prompt, new_tokens, Sampling::GREEDY)?;
    Ok(Round { prefill: prompt.len() as f64 / g.prefill_secs, decode: (new_tokens - 1) as f64 / g.decode_secs })
}

fn control_pass(a: &mut Runtime, prompt: &[u32], new_tokens: usize, rounds: usize) -> Result<(Vec<Round>, Vec<Round>)> {
    let (mut first, mut second) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        first.push(timed(a, prompt, new_tokens)?);
        second.push(timed(a, prompt, new_tokens)?);
    }
    Ok((first, second))
}

fn ab_pass(a: &mut Runtime, b: &mut Runtime, prompt: &[u32], new_tokens: usize, rounds: usize) -> Result<(Vec<Round>, Vec<Round>)> {
    let (mut ra, mut rb) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        ra.push(timed(a, prompt, new_tokens)?);
        rb.push(timed(b, prompt, new_tokens)?);
    }
    Ok((ra, rb))
}

fn describe(name: &str, s: &specrt::abstats::Summary) -> String {
    format!("{name} median {:.4} IQR {:.4} ({:.1}% of median) n={}", s.median, s.iqr(), s.spread() * 100.0, s.n)
}

fn bench_ab(args: Vec<String>) -> Result<()> {
    use specrt::abstats::{ratios, summarize, verdict};
    let usage = "usage: bench --ab [--rounds N] [--b-graphs on|off] <A spec> [B spec] [prompt_len] [new_tokens]";
    let mut args = args.into_iter().peekable();
    let a_path = PathBuf::from(args.next().context(usage)?);
    let b_path = match args.peek() {
        Some(next) if looks_like_spec(next) => PathBuf::from(args.next().unwrap()),
        _ => a_path.clone(),
    };
    let prompt_len: usize = args.next().map(|s| s.parse()).transpose().context(usage)?.unwrap_or(128);
    let new_tokens: usize = args.next().map(|s| s.parse()).transpose().context(usage)?.unwrap_or(128);
    anyhow::ensure!(new_tokens >= 2, "new_tokens must be at least 2");
    let rounds = flags().rounds;
    let spec_a = load_spec(&a_path)?;
    let mut spec_b = load_spec(&b_path)?;
    if let Some(graphs) = flags().b_graphs {
        spec_b.runtime.graphs = graphs;
    }
    let max_ctx = prompt_len + new_tokens;
    let mut a = Runtime::load(&spec_a, max_ctx)?;
    let mut b = Runtime::load(&spec_b, max_ctx)?;
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
    for rt in [&mut a, &mut b] {
        for _ in 0..3 {
            rt.generate(&prompt, 8.min(new_tokens), Sampling::GREEDY)?;
        }
    }
    let wall = specrt::bandwidth::probe(a.stream().context())?;
    println!(
        "A {} graphs={} | B {} graphs={} | prompt={prompt_len} new={new_tokens} rounds={rounds}",
        a_path.display(),
        spec_a.runtime.graphs,
        b_path.display(),
        spec_b.runtime.graphs
    );
    println!("bandwidth wall (same process): read {:.0} GB/s, copy {:.0} GB/s", wall.read_gbs, wall.copy_gbs);

    let d = |r: &[Round]| r.iter().map(|x| x.decode).collect::<Vec<_>>();
    let p = |r: &[Round]| r.iter().map(|x| x.prefill).collect::<Vec<_>>();
    let (mut control, mut control_prefill, mut passes, mut passes_prefill) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a_decode, mut b_decode) = (Vec::new(), Vec::new());
    for pass in 1..=2 {
        let (c1, c2) = control_pass(&mut a, &prompt, new_tokens, rounds)?;
        control.push(summarize(&ratios(&d(&c1), &d(&c2))));
        control_prefill.push(summarize(&ratios(&p(&c1), &p(&c2))));
        let (ra, rb) = ab_pass(&mut a, &mut b, &prompt, new_tokens, rounds)?;
        passes.push(summarize(&ratios(&d(&ra), &d(&rb))));
        passes_prefill.push(summarize(&ratios(&p(&ra), &p(&rb))));
        a_decode.extend(d(&ra));
        b_decode.extend(d(&rb));
        println!("pass {pass}: A/A decode  {}", describe("ratio", &control[pass - 1]));
        println!("pass {pass}: A/B decode  {}", describe("ratio", &passes[pass - 1]));
        println!("pass {pass}: A/A prefill {}", describe("ratio", &control_prefill[pass - 1]));
        println!("pass {pass}: A/B prefill {}", describe("ratio", &passes_prefill[pass - 1]));
    }
    for (name, weight_bytes, values) in [("A", a.weight_bytes as f64, &a_decode), ("B", b.weight_bytes as f64, &b_decode)] {
        let tps = summarize(values).median;
        println!(
            "{name}: decode {tps:.1} tok/s, {:.2} GB weights per token, {:.1} GB/s = {:.0}% of the read wall",
            weight_bytes / 1e9,
            tps * weight_bytes / 1e9,
            tps * weight_bytes / 1e9 / wall.read_gbs * 100.0
        );
    }
    let v = verdict(&control, &passes);
    println!("verdict (decode, B over A): {}", v.label());
    anyhow::ensure!(!matches!(v, specrt::abstats::Verdict::Refused(_)), "refused: rerun on a quiet card");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moe_gguf() -> Spec {
        Spec::from_gguf_file(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3moe-gguf/tiny-qwen3moe-Q4_K_M.gguf")).unwrap()
    }

    #[test]
    fn an_arm_sets_a_spec_key_beside_runtime_options() {
        let base = moe_gguf();
        let (s, _) = with_options(&base, &["spec.moe.quantized_experts=true".into(), "pointer_table=true".into()]).unwrap();
        assert!(s.moe.as_ref().unwrap().quantized_experts);
        assert!(s.runtime.offload.pointer_table);
        assert!(!base.moe.as_ref().unwrap().quantized_experts);
        let (s, _) = with_options(&base, &["spec.runtime.offload.expert_slots=3".into()]).unwrap();
        assert_eq!(s.runtime.offload.expert_slots, 3);
    }

    #[test]
    fn a_spec_key_that_does_not_exist_or_breaks_the_spec_is_refused() {
        let base = moe_gguf();
        assert!(with_options(&base, &["spec.nope.x=1".into()]).is_err());
        assert!(with_options(&base, &["spec.moe.no_such_field=1".into()]).is_err());
        let mut bf16 = base.clone();
        bf16.quant = specrt::spec::Quant::Bf16;
        assert!(with_options(&bf16, &["spec.moe.quantized_experts=true".into()]).is_err());
    }
}
