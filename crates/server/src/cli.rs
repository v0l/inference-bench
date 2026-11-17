use crate::config::ServerConfig;
use crate::serve::Server;
use anyhow::{Result, bail};
use clap::Args;
use inference_bench_core::logging::{LogArgs, init};
use std::path::PathBuf;

#[derive(Args, Debug, Clone, Default)]
pub struct ServeArgs {
    #[arg(long, short = 'c', env = "INFERENCE_BENCH_SERVER_CONFIG", help = "TOML config file")]
    pub config: Option<PathBuf>,
    #[arg(long, help = "Address to bind, overrides the config")]
    pub bind: Option<String>,
    #[arg(long, help = "Model id or spec path to serve, repeatable")]
    pub model: Vec<String>,
    #[arg(long, help = "Directory of specrt spec files to serve")]
    pub models_dir: Option<PathBuf>,
    #[arg(long, hide = true)]
    pub backend: Option<String>,
    #[arg(long, hide = true)]
    pub upstream: Option<String>,
    #[arg(long, help = "Context length the native engine allocates")]
    pub max_ctx: Option<usize>,
    #[arg(long, help = "Concurrent generations")]
    pub max_concurrent: Option<usize>,
    #[arg(long, help = "Serve with continuous batching: most sequences decoded together")]
    pub max_num_seqs: Option<usize>,
    #[arg(long, help = "Longest prompt plus reply with continuous batching (default the model's own maximum)")]
    pub max_model_len: Option<usize>,
    #[arg(long, help = "KV cache size in GB with continuous batching (default: free memory after the weights)")]
    pub kv_cache_gb: Option<f64>,
    #[arg(long, help = "Token budget per scheduler step (default 8192)")]
    pub max_batched_tokens: Option<usize>,
    #[arg(long, help = "Inter-token latency target in ms that bounds prefill beside decoders (default 100)")]
    pub itl_target_ms: Option<u64>,
    #[arg(long, help = "Prefill a whole prompt in one step when it is at most this many tokens (default 1024, 0 is off)")]
    pub whole_prompt_tokens: Option<usize>,
    #[arg(long, help = "Prompt tokens prefilled every step beside decoders (default 0, off)")]
    pub min_prefill_tokens: Option<usize>,
    #[arg(long, help = "Pinned host memory in GB where preempted requests park their KV cache instead of recomputing (default 0, off)")]
    pub swap_space_gb: Option<f64>,
    #[arg(long, help = "Tokens of generation to reserve KV room for at admission (default 0, off)")]
    pub admit_lookahead: Option<usize>,
    #[arg(long, help = "Seconds before an idle model is unloaded")]
    pub idle_unload_secs: Option<u64>,
    #[arg(long, help = "Write an access log (JSON lines) here")]
    pub access_log: Option<PathBuf>,
    #[arg(long, help = "Delete every saved session snapshot and exit")]
    pub purge_sessions: bool,
}

impl ServeArgs {
    pub fn into_config(&self, log_prompts: bool) -> Result<ServerConfig> {
        let mut config = match &self.config {
            Some(path) => ServerConfig::load(path)?,
            None => ServerConfig::default(),
        };
        if let Some(b) = &self.bind {
            config.bind = b.clone();
        }
        if let Some(d) = &self.models_dir {
            config.models_dir = Some(d.clone());
            config.discover_models()?;
        }
        if self.backend.as_deref().is_some_and(|b| b != "specrt" && b != "native") || self.upstream.is_some() {
            bail!("the server only runs models on the native engine; --backend and --upstream were removed");
        }
        if let Some(n) = self.max_ctx {
            config.max_ctx = Some(n);
        }
        for m in &self.model {
            if config.model(m).is_none() {
                let id = m.rsplit('/').next().unwrap_or(m).trim_end_matches(".toml").to_string();
                config.models.push(crate::config::ModelConfig {
                    id,
                    model: Some(m.clone()),
                    ..Default::default()
                });
            }
        }
        if let Some(n) = self.max_concurrent {
            config.max_concurrent = n;
        }
        config.max_num_seqs = self.max_num_seqs.or(config.max_num_seqs);
        config.max_model_len = self.max_model_len.or(config.max_model_len);
        config.kv_cache_gb = self.kv_cache_gb.or(config.kv_cache_gb);
        config.max_batched_tokens = self.max_batched_tokens.or(config.max_batched_tokens);
        config.itl_target_ms = self.itl_target_ms.or(config.itl_target_ms);
        config.whole_prompt_tokens = self.whole_prompt_tokens.or(config.whole_prompt_tokens);
        config.min_prefill_tokens = self.min_prefill_tokens.or(config.min_prefill_tokens);
        config.swap_space_gb = self.swap_space_gb.or(config.swap_space_gb);
        config.admit_lookahead = self.admit_lookahead.or(config.admit_lookahead);
        if let Some(n) = self.idle_unload_secs {
            config.idle_unload_secs = Some(n);
        }
        if let Some(p) = &self.access_log {
            config.access_log = Some(p.clone());
        }
        config.log_prompts |= log_prompts;
        config.validate()?;
        Ok(config)
    }

    pub fn purge(&self) -> Result<usize> {
        let config = self.into_config(false)?;
        let root = if config.sessions.dir.is_empty() { inference_bench_core::specrt_sessions::default_dir() } else { PathBuf::from(&config.sessions.dir) };
        inference_bench_core::specrt_sessions::purge_root(&root)
    }

    pub fn run(&self, log: &LogArgs) -> Result<()> {
        if self.purge_sessions {
            println!("deleted {} session snapshots", self.purge()?);
            return Ok(());
        }
        let _guard = init(&log.config())?;
        self.run_initialised(log)
    }

    pub fn run_initialised(&self, log: &LogArgs) -> Result<()> {
        let config = self.into_config(log.log_prompts)?;
        let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        runtime.block_on(async move {
            let handle = Server::new(config).start().await?;
            tokio::select! {
                r = shutdown_signal() => r?,
                _ = wait_forever() => {}
            }
            handle.stop().await
        })
    }
}

pub fn run_config(
    config: ServerConfig,
    preload: Option<crate::config::ModelConfig>,
    on_ready: impl FnOnce(&crate::serve::ServerHandle) -> Result<()>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        let handle = Server::new(config).start().await?;
        if let Some(model) = &preload {
            let loaded = tokio::select! {
                r = handle.state.manager.preload(model) => r.map_err(|e| anyhow::anyhow!("{}", e.message)),
                r = shutdown_signal() => r.and_then(|_| Err(anyhow::anyhow!("stopped while loading"))),
            };
            if let Err(e) = loaded {
                let _ = handle.stop().await;
                return Err(e);
            }
        }
        if let Err(e) = on_ready(&handle) {
            let _ = handle.stop().await;
            return Err(e);
        }
        shutdown_signal().await?;
        handle.stop().await
    })
}

async fn wait_forever() {
    std::future::pending::<()>().await
}

pub async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate())?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutdown signal received");
    Ok(())
}
