use crate::config::ServerConfig;
use crate::serve::Server;
use anyhow::{Context, Result};
use clap::Args;
use inference_bench_core::backend::BackendConfig;
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
    #[arg(long, value_parser = ["openai", "transformers", "specrt"], help = "Default backend kind")]
    pub backend: Option<String>,
    #[arg(long, help = "Upstream base URL for the openai backend")]
    pub upstream: Option<String>,
    #[arg(long, help = "Concurrent generations")]
    pub max_concurrent: Option<usize>,
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
        match self.backend.as_deref() {
            Some("openai") => {
                let url = self.upstream.clone().context("--upstream is required with --backend openai")?;
                config.default_backend = BackendConfig::Openai { url, api_key: None, label: None };
            }
            Some("transformers") => {
                config.default_backend = BackendConfig::Transformers { python: None, device: None }
            }
            Some(_) => config.default_backend = BackendConfig::Specrt { worker: None, max_ctx: None },
            None => {}
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
