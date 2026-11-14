use crate::config::{ModelConfig, ServerConfig};
use crate::error::ApiError;
use crate::metrics::Metrics;
use inference_bench_core::target::Target;
use inference_bench_core::result::BackendError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::Instrument;

use inference_bench_core::specrt_serve::{RuntimeOptions, ServeOptions};

pub trait EngineFactory: Send + Sync + 'static {
    fn build(&self, max_ctx: Option<usize>) -> Result<Box<dyn Target>, BackendError>;

    fn build_with(&self, max_ctx: Option<usize>, runtime: Option<RuntimeOptions>) -> Result<Box<dyn Target>, BackendError> {
        let _ = runtime;
        self.build(max_ctx)
    }

    fn build_serving(&self, max_ctx: Option<usize>, opts: ServeOptions) -> Result<Box<dyn Target>, BackendError> {
        let _ = opts;
        self.build(max_ctx)
    }
}

pub struct NativeFactory;

impl EngineFactory for NativeFactory {
    fn build(&self, max_ctx: Option<usize>) -> Result<Box<dyn Target>, BackendError> {
        Ok(Box::new(inference_bench_core::native::Native::new(max_ctx)))
    }

    fn build_with(&self, max_ctx: Option<usize>, runtime: Option<RuntimeOptions>) -> Result<Box<dyn Target>, BackendError> {
        Ok(Box::new(inference_bench_core::native::Native::with_runtime(max_ctx, runtime)))
    }

    fn build_serving(&self, _max_ctx: Option<usize>, opts: ServeOptions) -> Result<Box<dyn Target>, BackendError> {
        Ok(Box::new(inference_bench_core::serving::ServingTarget::new(opts)))
    }
}

impl<F> EngineFactory for F
where
    F: Fn(Option<usize>) -> Result<Box<dyn Target>, BackendError> + Send + Sync + 'static,
{
    fn build(&self, max_ctx: Option<usize>) -> Result<Box<dyn Target>, BackendError> {
        self(max_ctx)
    }
}

pub type GpuProbe = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

#[derive(Debug, Clone, PartialEq)]
pub struct SourceModel {
    pub id: String,
    pub aliases: Vec<String>,
}

pub trait ModelSource: Send + Sync + 'static {
    fn loaded(&self) -> Vec<SourceModel>;
}

pub fn nvidia_smi_probe() -> GpuProbe {
    Arc::new(crate::gpu::used_bytes)
}

pub struct LoadedModel {
    pub id: String,
    pub backend_label: String,
    pub vram: u64,
    pub loaded_at: Instant,
    last_used: Mutex<Instant>,
    active: AtomicUsize,
    pool: Mutex<Vec<Box<dyn Target>>>,
    sem: Arc<Semaphore>,
    stats: Mutex<Option<Box<dyn Target>>>,
}

impl LoadedModel {
    pub fn idle_for(&self) -> Duration {
        self.last_used.lock().unwrap().elapsed()
    }

    pub fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }
}

pub struct Lease {
    model: Arc<LoadedModel>,
    permit: Option<OwnedSemaphorePermit>,
    backend: Option<Box<dyn Target>>,
}

impl Lease {
    pub fn backend(&mut self) -> &mut dyn Target {
        self.backend.as_deref_mut().expect("lease holds a backend")
    }

    pub fn model(&self) -> &Arc<LoadedModel> {
        &self.model
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(b) = self.backend.take() {
            self.model.pool.lock().unwrap().push(b);
        }
        self.permit.take();
        *self.model.last_used.lock().unwrap() = Instant::now();
        self.model.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub backend: String,
    pub loaded: bool,
    pub vram_bytes: u64,
    pub active_requests: usize,
    pub idle_secs: Option<f64>,
}

pub struct Manager {
    config: Arc<ServerConfig>,
    factory: Arc<dyn EngineFactory>,
    probe: GpuProbe,
    metrics: Arc<Metrics>,
    loaded: Mutex<HashMap<String, Arc<LoadedModel>>>,
    load_lock: tokio::sync::Mutex<()>,
    source: Option<Arc<dyn ModelSource>>,
}

impl Manager {
    pub fn new(
        config: Arc<ServerConfig>,
        factory: Arc<dyn EngineFactory>,
        probe: GpuProbe,
        metrics: Arc<Metrics>,
        source: Option<Arc<dyn ModelSource>>,
    ) -> Self {
        Self { config, factory, probe, metrics, loaded: Mutex::default(), load_lock: Default::default(), source }
    }

    pub fn gpu_used(&self) -> Option<u64> {
        (self.probe)()
    }

    pub fn resolve(&self, id: &str) -> Result<ModelConfig, ApiError> {
        if let Some(source) = &self.source {
            return match source.loaded().into_iter().find(|m| m.id == id || m.aliases.iter().any(|a| a == id)) {
                Some(m) => Ok(ModelConfig { id: m.id, ..ModelConfig::default() }),
                None => Err(ApiError::model_not_found(id)),
            };
        }
        if let Some(m) = self.config.model(id) {
            return Ok(m.clone());
        }
        match self.config.models.is_empty() {
            true => Ok(ModelConfig { id: id.to_string(), ..ModelConfig::default() }),
            false => Err(ApiError::model_not_found(id)),
        }
    }

    pub fn requested_model(&self, named: Option<&str>) -> Result<String, ApiError> {
        match named.filter(|m| !m.is_empty()) {
            Some(m) => Ok(m.to_string()),
            None => self.default_model().ok_or_else(|| match &self.source {
                Some(_) => ApiError::new(axum::http::StatusCode::NOT_FOUND, "not_found_error", "no model is loaded")
                    .with_code("model_not_found", "model"),
                None => ApiError::bad_request("model is required"),
            }),
        }
    }

    pub fn default_model(&self) -> Option<String> {
        if let Some(source) = &self.source {
            return source.loaded().into_iter().next().map(|m| m.id);
        }
        self.config
            .default_model
            .clone()
            .or_else(|| (self.config.models.len() == 1).then(|| self.config.models[0].id.clone()))
    }

    fn get(&self, id: &str) -> Option<Arc<LoadedModel>> {
        let loaded = self.loaded.lock().unwrap();
        let m = loaded.get(id)?;
        m.active.fetch_add(1, Ordering::SeqCst);
        Some(m.clone())
    }

    pub fn loaded_snapshot(&self) -> Vec<(String, u64)> {
        if let Some(source) = &self.source {
            let loaded = self.loaded.lock().unwrap();
            let mut v: Vec<_> = source
                .loaded()
                .into_iter()
                .map(|m| {
                    let vram = loaded.get(&m.id).map(|l| l.vram).unwrap_or(0);
                    (m.id, vram)
                })
                .collect();
            v.sort();
            return v;
        }
        let mut v: Vec<_> = self.loaded.lock().unwrap().values().map(|m| (m.id.clone(), m.vram)).collect();
        v.sort();
        v
    }

    pub fn engine_metrics(&self) -> String {
        let mut models: Vec<Arc<LoadedModel>> = self.loaded.lock().unwrap().values().cloned().collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.iter().filter_map(|m| m.stats.lock().unwrap().as_ref().and_then(|t| t.metrics())).collect()
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        let loaded = self.loaded.lock().unwrap();
        if let Some(source) = &self.source {
            return source
                .loaded()
                .into_iter()
                .map(|m| {
                    let l = loaded.get(&m.id);
                    ModelStatus {
                        id: m.id,
                        backend: l.map(|l| l.backend_label.clone()).unwrap_or_else(|| "engine".to_string()),
                        loaded: true,
                        vram_bytes: l.map(|l| l.vram).unwrap_or(0),
                        active_requests: l.map(|l| l.active()).unwrap_or(0),
                        idle_secs: l.map(|l| l.idle_for().as_secs_f64()),
                    }
                })
                .collect();
        }
        let mut out: Vec<ModelStatus> = self
            .config
            .models
            .iter()
            .map(|m| {
                let l = loaded.get(&m.id);
                ModelStatus {
                    id: m.id.clone(),
                    backend: inference_bench_core::target::NATIVE.to_string(),
                    loaded: l.is_some(),
                    vram_bytes: l.map(|l| l.vram).unwrap_or(0),
                    active_requests: l.map(|l| l.active()).unwrap_or(0),
                    idle_secs: l.map(|l| l.idle_for().as_secs_f64()),
                }
            })
            .collect();
        for (id, l) in loaded.iter() {
            if !out.iter().any(|s| &s.id == id) {
                out.push(ModelStatus {
                    id: id.clone(),
                    backend: l.backend_label.clone(),
                    loaded: true,
                    vram_bytes: l.vram,
                    active_requests: l.active(),
                    idle_secs: Some(l.idle_for().as_secs_f64()),
                });
            }
        }
        out
    }

    pub async fn acquire(&self, id: &str) -> Result<Lease, ApiError> {
        let cfg = self.resolve(id)?;
        let model = match self.get(&cfg.id) {
            Some(m) => m,
            None => self.load(&cfg).await?,
        };
        let mut lease = Lease { model: model.clone(), permit: None, backend: None };
        let permit = model
            .sem
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ApiError::unavailable("model was unloaded"))?;
        lease.permit = Some(permit);
        lease.backend = model.pool.lock().unwrap().pop();
        if lease.backend.is_none() {
            return Err(ApiError::unavailable("model has no free instance"));
        }
        Ok(lease)
    }

    pub async fn preload(&self, cfg: &ModelConfig) -> Result<(), ApiError> {
        let model = self.load(cfg).await?;
        model.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    pub async fn load(&self, cfg: &ModelConfig) -> Result<Arc<LoadedModel>, ApiError> {
        let _guard = self.load_lock.lock().await;
        if let Some(m) = self.get(&cfg.id) {
            return Ok(m);
        }
        self.make_room(cfg)?;
        let max_ctx = self.config.max_ctx_for(cfg);
        let serving = self.config.serve_options(cfg);
        let runtime = self.config.runtime.clone();
        let name = self.config.specs.resolve(cfg.upstream_name());
        let spec_hash = crate::config::spec_hash_of(&name);
        let vram_before = (self.probe)();
        let started = Instant::now();
        tracing::info!(model = %cfg.id, upstream = %name, ?max_ctx, ?serving, %spec_hash, vram_before_bytes = ?vram_before, "loading model");
        let factory = self.factory.clone();
        let serving_in = serving.clone();
        let span = tracing::Span::current();
        let built = tokio::task::spawn_blocking(move || {
            span.in_scope(|| {
                let mut b = match serving_in.clone() {
                    Some(opts) => factory.build_serving(max_ctx, opts)?,
                    None => factory.build_with(max_ctx, runtime)?,
                };
                b.load(&name)?;
                let mut backends = Vec::new();
                if let Some(opts) = &serving_in {
                    for _ in 1..opts.max_num_seqs {
                        match b.share() {
                            Some(more) => backends.push(more),
                            None => break,
                        }
                    }
                }
                backends.push(b);
                Ok::<_, BackendError>(backends)
            })
        })
        .instrument(tracing::Span::current())
        .await
        .map_err(|e| ApiError::internal(format!("load task failed: {e}")))?;
        let backends = match built {
            Ok(b) => b,
            Err(e) => {
                self.metrics.model_load_failed();
                tracing::error!(model = %cfg.id, error = %e, load_secs = started.elapsed().as_secs_f64(), "model load failed");
                return Err(e.into());
            }
        };
        let vram_after = (self.probe)();
        let measured = match (vram_before, vram_after) {
            (Some(b), Some(a)) if a > b => a - b,
            _ => 0,
        };
        let vram = if measured > 0 { measured } else { cfg.vram_bytes.unwrap_or(0) };
        let backend_label = backends.first().map(|b| b.name().to_string()).unwrap_or_default();
        let stats = backends.first().and_then(|b| b.share()).filter(|_| serving.is_some());
        tracing::info!(
            model = %cfg.id,
            backend = %backend_label,
            %spec_hash,
            load_secs = started.elapsed().as_secs_f64(),
            vram_before_bytes = ?vram_before,
            vram_after_bytes = ?vram_after,
            vram_bytes = vram,
            "model loaded"
        );
        self.metrics.model_loaded(&cfg.id);
        let model = Arc::new(LoadedModel {
            id: cfg.id.clone(),
            backend_label,
            vram,
            loaded_at: Instant::now(),
            last_used: Mutex::new(Instant::now()),
            active: AtomicUsize::new(1),
            sem: Arc::new(Semaphore::new(backends.len())),
            pool: Mutex::new(backends),
            stats: Mutex::new(stats),
        });
        self.loaded.lock().unwrap().insert(cfg.id.clone(), model.clone());
        Ok(model)
    }

    fn make_room(&self, incoming: &ModelConfig) -> Result<(), ApiError> {
        loop {
            let (count, used, victim) = {
                let loaded = self.loaded.lock().unwrap();
                let victim = loaded
                    .values()
                    .filter(|m| m.active() == 0)
                    .max_by_key(|m| m.idle_for())
                    .map(|m| m.id.clone());
                (loaded.len(), loaded.values().map(|m| m.vram).sum::<u64>(), victim)
            };
            let over_count = count >= self.config.max_loaded_models;
            let over_budget = self
                .config
                .vram_budget_bytes
                .is_some_and(|b| used + incoming.vram_bytes.unwrap_or(0) > b && count > 0);
            if !over_count && !over_budget {
                return Ok(());
            }
            match victim {
                Some(id) => {
                    tracing::info!(model = %id, incoming = %incoming.id, reason = if over_count { "model limit" } else { "vram budget" }, "evicting idle model");
                    self.unload(&id);
                }
                None => {
                    return Err(ApiError::unavailable(format!(
                        "no capacity to load {}: loaded models are busy",
                        incoming.id
                    )));
                }
            }
        }
    }

    pub fn unload(&self, id: &str) -> bool {
        let removed = {
            let mut loaded = self.loaded.lock().unwrap();
            match loaded.get(id) {
                Some(m) if m.active() == 0 => loaded.remove(id),
                _ => None,
            }
        };
        match removed {
            Some(m) => {
                tracing::info!(model = %id, vram_bytes = m.vram, loaded_secs = m.loaded_at.elapsed().as_secs_f64(), "model unloaded");
                self.metrics.model_unloaded();
                drop(m);
                true
            }
            None => false,
        }
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.lock().unwrap().contains_key(id)
    }

    pub fn is_busy(&self, id: &str) -> bool {
        self.loaded.lock().unwrap().get(id).is_some_and(|m| m.active() > 0)
    }

    pub fn reap_idle(&self, idle: Duration) -> usize {
        let stale: Vec<String> = self
            .loaded
            .lock()
            .unwrap()
            .values()
            .filter(|m| m.active() == 0 && m.idle_for() >= idle)
            .map(|m| m.id.clone())
            .collect();
        stale.iter().filter(|id| self.unload(id)).count()
    }

    pub fn unload_all(&self) {
        let ids: Vec<String> = self.loaded.lock().unwrap().keys().cloned().collect();
        for id in ids {
            self.unload(&id);
        }
    }
}
