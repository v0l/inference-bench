use crate::app::{AppState, router};
use crate::config::ServerConfig;
use crate::manager::{EngineFactory, NativeFactory, GpuProbe, nvidia_smi_probe};
use anyhow::{Context, Result};
use inference_bench_core::specs::{Outcome, SpecStore, SpecsConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

pub struct Server {
    config: Arc<ServerConfig>,
    factory: Arc<dyn EngineFactory>,
    probe: GpuProbe,
}

pub struct ServerHandle {
    pub addr: SocketAddr,
    pub state: Arc<AppState>,
    shutdown: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<()>>,
}

impl ServerHandle {
    pub fn url(&self) -> String {
        format!("{}://{}", if self.state.config.tls.is_some() { "https" } else { "http" }, self.addr)
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown.send(true);
    }

    pub async fn wait(self) -> Result<()> {
        self.task.await.context("server task")?
    }

    pub async fn stop(self) -> Result<()> {
        self.shutdown();
        self.wait().await
    }
}

impl Server {
    pub fn new(config: ServerConfig) -> Self {
        Self { config: Arc::new(config), factory: Arc::new(NativeFactory), probe: nvidia_smi_probe() }
    }

    pub fn with_factory(mut self, factory: impl EngineFactory) -> Self {
        self.factory = Arc::new(factory);
        self
    }

    pub fn with_gpu_probe(mut self, probe: GpuProbe) -> Self {
        self.probe = probe;
        self
    }

    pub async fn start(self) -> Result<ServerHandle> {
        let addr: SocketAddr = self
            .config
            .bind
            .parse()
            .with_context(|| format!("bind address {}", self.config.bind))?;
        let listener = std::net::TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let state = AppState::new(self.config.clone(), self.factory, self.probe)?;
        let app = router(state.clone());
        let (tx, rx) = watch::channel(false);
        spawn_reaper(state.clone(), rx.clone());
        spawn_preload(state.clone());
        spawn_spec_updates(state.clone(), rx.clone());
        let grace = Duration::from_secs(self.config.shutdown_grace_secs);
        tracing::info!(
            %addr,
            tls = self.config.tls.is_some(),
            auth = state.auth_required(),
            max_concurrent = self.config.max_concurrent,
            queue_depth = self.config.queue_depth,
            models = self.config.models.len(),
            "server listening"
        );
        let task = match &self.config.tls {
            #[cfg(feature = "tls")]
            Some(tls) => {
                let _ = rustls::crypto::ring::default_provider().install_default();
                let rustls_config =
                    axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls.cert, &tls.key)
                        .await
                        .context("loading TLS certificate and key")?;
                let handle = axum_server::Handle::new();
                let h = handle.clone();
                let mut rx = rx.clone();
                tokio::spawn(async move {
                    let _ = rx.wait_for(|v| *v).await;
                    h.graceful_shutdown(Some(grace));
                });
                let state = state.clone();
                tokio::spawn(async move {
                    axum_server::from_tcp_rustls(listener, rustls_config)
                        .handle(handle)
                        .serve(app.into_make_service())
                        .await
                        .context("tls server")?;
                    finish(&state);
                    Ok(())
                })
            }
            #[cfg(not(feature = "tls"))]
            Some(_) => anyhow::bail!("this build has no TLS support"),
            None => {
                let listener = tokio::net::TcpListener::from_std(listener)?;
                let mut rx = rx.clone();
                let rx_grace = rx.clone();
                let state = state.clone();
                tokio::spawn(async move {
                    let serving = axum::serve(listener, app).with_graceful_shutdown(async move {
                        let _ = rx.wait_for(|v| *v).await;
                        tracing::info!(grace_secs = grace.as_secs(), "shutting down, draining in-flight requests");
                    });
                    let mut serving = Box::pin(serving.into_future());
                    let mut rx2 = rx_grace;
                    let result = tokio::select! {
                        r = &mut serving => r,
                        _ = async {
                            let _ = rx2.wait_for(|v| *v).await;
                            tokio::time::sleep(grace).await;
                        } => {
                            tracing::warn!("grace period over, closing remaining connections");
                            Ok(())
                        }
                    };
                    finish(&state);
                    result.context("server")
                })
            }
        };
        Ok(ServerHandle { addr, state, shutdown: tx, task })
    }
}

fn finish(state: &Arc<AppState>) {
    state.manager.unload_all();
    tracing::info!("server stopped");
}

fn spawn_reaper(state: Arc<AppState>, mut rx: watch::Receiver<bool>) {
    let Some(secs) = state.config.idle_unload_secs else { return };
    let idle = Duration::from_secs(secs);
    let tick = Duration::from_millis((secs * 1000 / 4).clamp(50, 5000));
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(tick) => {
                    let n = state.manager.reap_idle(idle);
                    if n > 0 {
                        tracing::info!(unloaded = n, idle_secs = secs, "idle timer unloaded models");
                    }
                }
                _ = rx.wait_for(|v| *v) => return,
            }
        }
    });
}

fn check_specs(state: &Arc<AppState>, cfg: &SpecsConfig, data: &std::path::Path, force: bool) {
    let store = SpecStore::new(data);
    if !force && !store.due(cfg, chrono::Utc::now()) {
        return;
    }
    let result = match store.update(cfg) {
        Ok(Outcome::Updated(_)) => "updated",
        Ok(_) => "unchanged",
        Err(_) => "failed",
    };
    state.metrics.spec_check(result, store.state().last_success.map(|t| t.timestamp()));
}

fn spawn_spec_updates(state: Arc<AppState>, mut rx: watch::Receiver<bool>) {
    let section = state.config.specs.clone();
    let cfg = section.updater();
    let data = section.data();
    state.metrics.spec_last_success(SpecStore::new(&data).state().last_success.map(|t| t.timestamp()));
    if !cfg.auto_update {
        tracing::info!("spec updates are off");
        return;
    }
    let tick = cfg.interval().map(|i| i.min(Duration::from_secs(300)));
    tokio::spawn(async move {
        let mut force = true;
        loop {
            let (s, c, d) = (state.clone(), cfg.clone(), data.clone());
            let _ = tokio::task::spawn_blocking(move || check_specs(&s, &c, &d, force)).await;
            force = false;
            let Some(tick) = tick else { return };
            tokio::select! {
                _ = tokio::time::sleep(tick) => {}
                _ = rx.wait_for(|v| *v) => return,
            }
        }
    });
}

fn spawn_preload(state: Arc<AppState>) {
    let ids: Vec<_> = state.config.models.iter().filter(|m| m.preload).cloned().collect();
    if ids.is_empty() {
        return;
    }
    tokio::spawn(async move {
        for m in ids {
            if let Err(e) = state.manager.preload(&m).await {
                tracing::error!(model = %m.id, error = %e.message, "preload failed");
            }
        }
    });
}
