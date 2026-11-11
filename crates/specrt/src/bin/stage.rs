use anyhow::{Context, Result};
use specrt::stage::{Config, Generated, generate, serve};
use specrt::wire::Chain;
use std::path::PathBuf;
use std::time::Duration;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn usage() -> &'static str {
    "usage: stage serve <config.toml> | stage ping <config.toml> | stage run <config.toml> [prompt_len[,prompt_len..]] [new_tokens] [runs] [chunk]"
}

fn hops(g: &Generated, stages: usize) -> Vec<f64> {
    (0..stages)
        .map(|i| {
            median(
                g.decode_steps
                    .iter()
                    .map(|t| ms(t.stage[i]) - ms(t.compute[i]))
                    .collect(),
            )
        })
        .collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let mut args = std::env::args().skip(1);
    let command = args.next().context(usage())?;
    let config = Config::load(&PathBuf::from(args.next().context(usage())?))?;
    match command.as_str() {
        "serve" => {
            let stage = config.stage.context("the config has no [stage] table")?;
            let served = serve(&stage, config.placement.as_ref())?;
            tracing::info!(addr = %served.server.addr(), layers = ?stage.layers, mdns = served.announcer.is_some(), "stage ready");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        "ping" => {
            let p = config.pipeline.context("the config has no [pipeline] table")?;
            let mut chain = Chain::connect(&p.stages, &p.key()?, p.timeout())?;
            for _ in 0..50 {
                chain.ping_all()?;
            }
            let rounds: Vec<Vec<Duration>> = (0..500).map(|_| chain.ping_all()).collect::<Result<_>>()?;
            for (i, addr) in p.stages.iter().enumerate() {
                let m = median(rounds.iter().map(|r| ms(r[i])).collect());
                println!("stage {i} {addr}: round trip median {m:.3} ms over {} pings", rounds.len());
            }
            Ok(())
        }
        "run" => {
            let p = config.pipeline.context("the config has no [pipeline] table")?;
            let prompts: Vec<usize> = args
                .next()
                .unwrap_or_else(|| "128".into())
                .split(',')
                .map(|s| s.parse().context(usage()))
                .collect::<Result<_>>()?;
            let new_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
            let runs: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(3);
            let chunk: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(1024);
            let mut chain = Chain::connect(&p.stages, &p.key()?, p.timeout())?;
            println!("chain of {} stages covering {} layers", chain.len(), chain.layers());
            for &prompt_len in &prompts {
                let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7 % 5000).collect();
                generate(&mut chain, 0, &prompt, 4.min(new_tokens), chunk)?;
                let results: Vec<Generated> = (0..runs).map(|_| generate(&mut chain, 0, &prompt, new_tokens, chunk)).collect::<Result<_>>()?;
                let prefill = median(results.iter().map(|g| prompt_len as f64 / g.prefill.as_secs_f64()).collect());
                let decode = median(results.iter().map(|g| (new_tokens - 1) as f64 / g.decode.as_secs_f64()).collect());
                let step = median(results.iter().map(|g| ms(g.decode) / (new_tokens - 1) as f64).collect());
                let hop = hops(&results[0], chain.len());
                let compute: Vec<f64> = (0..chain.len()).map(|i| median(results[0].decode_steps.iter().map(|t| ms(t.compute[i])).collect())).collect();
                println!(
                    "  prompt={prompt_len} prefill={prefill:.1} tok/s decode={decode:.2} tok/s step={step:.2} ms stage_compute_ms={compute:.2?} stage_hop_ms={hop:.3?} tokens={:?}",
                    &results[0].tokens[..results[0].tokens.len().min(8)]
                );
            }
            Ok(())
        }
        _ => anyhow::bail!(usage()),
    }
}
