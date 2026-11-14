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
    #[arg(long, help = "Seconds before an idle model is unloaded")]
    pub idle_unload_secs: Option<u64>,
    #[arg(long, help = "Write an access log (JSON lines) here")]
    pub access_log: Option<PathBuf>,
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

    pub fn run(&self, log: &LogArgs) -> Result<()> {
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

async fn wait_forever() {
    std::future::pending::<()>().await
}

async fn shutdown_signal() -> Result<()> {
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
