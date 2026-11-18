use crate::placement::{DeviceConfig, Discovery, PlacementConfig};
use crate::runtime::{DecodeRow, Layout, Runtime, Sampling, StageRange, Stepper};
use crate::spec::Spec;
use crate::wire::{Chain, ChainTimes, Input, Message, StageHandler, StageServer};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StageConfig {
    pub spec: PathBuf,
    #[serde(default)]
    pub layers: Option<[usize; 2]>,
    #[serde(default)]
    pub device: Option<DeviceConfig>,
    pub bind: String,
    pub key: Option<String>,
    pub key_file: Option<PathBuf>,
    #[serde(default = "default_ctx")]
    pub max_ctx: usize,
    #[serde(default = "default_slots")]
    pub slots: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    #[serde(default)]
    pub stages: Vec<String>,
    pub key: Option<String>,
    pub key_file: Option<PathBuf>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub stage: Option<StageConfig>,
    pub pipeline: Option<PipelineConfig>,
    pub placement: Option<PlacementConfig>,
}

fn default_ctx() -> usize {
    4096
}

fn default_slots() -> usize {
    1
}

fn default_timeout_ms() -> u64 {
    600_000
}

pub(crate) fn read_key(inline: &Option<String>, file: &Option<PathBuf>) -> Result<Vec<u8>> {
    let key = match (inline, file) {
        (Some(k), None) => k.clone().into_bytes(),
        (None, Some(f)) => std::fs::read_to_string(f).with_context(|| f.display().to_string())?.trim().as_bytes().to_vec(),
        (Some(_), Some(_)) => anyhow::bail!("set key or key_file, not both"),
        (None, None) => anyhow::bail!("a pre-shared key is required: set key or key_file"),
    };
    ensure!(!key.is_empty(), "the pre-shared key is empty");
    Ok(key)
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        toml::from_str(&text).with_context(|| path.display().to_string())
    }
}

impl StageConfig {
    pub fn key(&self) -> Result<Vec<u8>> {
        read_key(&self.key, &self.key_file)
    }
}

impl PipelineConfig {
    pub fn key(&self) -> Result<Vec<u8>> {
        read_key(&self.key, &self.key_file)
    }

    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bytes_f32(b: &[u8]) -> Result<Vec<f32>> {
    ensure!(b.len().is_multiple_of(4), "hidden payload of {} bytes is not f32", b.len());
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

enum Engine {
    Cuda { rt: Runtime, st: Stepper },
    #[cfg(feature = "wgpu")]
    Wgpu(crate::backends::dense::Dense<crate::backends::wgpu::Wgpu>),
}

struct Loaded {
    engine: Engine,
    start: u32,
    end: u32,
}

unsafe impl Send for Loaded {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Cuda,
    Wgpu,
}

#[derive(Clone)]
pub struct MemoryGuard {
    pub host: String,
    pub reserve_bytes: u64,
    pub free: Arc<dyn Fn(usize) -> Result<u64> + Send + Sync>,
}

impl MemoryGuard {
    pub fn check(&self, spec: &Spec, layers: [usize; 2], max_ctx: usize, slots: usize, device: usize) -> Result<()> {
        let need = crate::placement::CostModel::new(spec, max_ctx, slots).stage_bytes(layers[0], layers[1]);
        let free = (self.free)(device)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        anyhow::ensure!(
            need + self.reserve_bytes <= free,
            "{} refuses layers {}-{} of {}: they need about {:.1} GiB, GPU {device} has {:.1} GiB free and {:.1} GiB is kept in reserve",
            self.host,
            layers[0],
            layers[1],
            spec.name,
            gib(need),
            gib(free),
            gib(self.reserve_bytes)
        );
        Ok(())
    }
}

pub struct RuntimeStage {
    guard: Mutex<Option<MemoryGuard>>,
    spec: Spec,
    max_ctx: usize,
    slots: usize,
    device: usize,
    api: Api,
    inner: Mutex<Option<Loaded>>,
}

#[cfg(feature = "wgpu")]
fn load_wgpu(spec: &Spec, layers: [usize; 2], max_ctx: usize, slots: usize) -> Result<Loaded> {
    let be = crate::backends::wgpu::Wgpu::new(spec.runtime.cache_dir.clone().or_else(|| Some(crate::jit::default_cache_dir())))?;
    let dense = crate::backends::dense::Dense::load_stage(be, spec, max_ctx, slots, layers[0], layers[1])?;
    Ok(Loaded { engine: Engine::Wgpu(dense), start: layers[0] as u32, end: layers[1] as u32 })
}

#[cfg(not(feature = "wgpu"))]
fn load_wgpu(_: &Spec, _: [usize; 2], _: usize, _: usize) -> Result<Loaded> {
    anyhow::bail!("this build has no wgpu support (build with --features wgpu)")
}

fn load_range(spec: &Spec, layers: [usize; 2], max_ctx: usize, slots: usize, device: usize, api: Api) -> Result<Loaded> {
    if api != Api::Cuda {
        return load_wgpu(spec, layers, max_ctx, slots);
    }
    let pages = max_ctx.div_ceil(crate::codegen::PAGE);
    let layout = Layout { kv_planes: 1, ..Layout::contiguous(max_ctx, slots) };
    let range = StageRange { device, start: layers[0], end: layers[1] };
    let mut rt = Runtime::load_stage(spec, layout, range)?;
    for slot in 0..slots {
        let blocks: Vec<u32> = (0..pages as u32).map(|p| (slot * pages) as u32 + p).collect();
        rt.set_block_table(slot, &blocks)?;
    }
    let st = rt.stepper()?;
    Ok(Loaded { engine: Engine::Cuda { rt, st }, start: layers[0] as u32, end: layers[1] as u32 })
}

impl RuntimeStage {
    pub fn load(spec: &Spec, layers: [usize; 2], max_ctx: usize, slots: usize) -> Result<RuntimeStage> {
        let api = if spec.runtime.backend == crate::spec::BackendChoice::Wgpu { Api::Wgpu } else { Api::Cuda };
        let loaded = load_range(spec, layers, max_ctx, slots, 0, api)?;
        Ok(RuntimeStage { spec: spec.clone(), max_ctx, slots, device: 0, api, inner: Mutex::new(Some(loaded)), guard: Mutex::new(None) })
    }

    pub fn unloaded(spec: &Spec, max_ctx: usize, slots: usize) -> RuntimeStage {
        Self::unloaded_on(spec, max_ctx, slots, 0)
    }

    pub fn unloaded_on(spec: &Spec, max_ctx: usize, slots: usize, device: usize) -> RuntimeStage {
        let api = if spec.runtime.backend == crate::spec::BackendChoice::Wgpu { Api::Wgpu } else { Api::Cuda };
        Self::unloaded_with(spec, max_ctx, slots, device, api)
    }

    pub fn unloaded_with(spec: &Spec, max_ctx: usize, slots: usize, device: usize, api: Api) -> RuntimeStage {
        RuntimeStage { spec: spec.clone(), max_ctx, slots, device, api, inner: Mutex::new(None), guard: Mutex::new(None) }
    }

    pub fn guard_memory(&self, guard: Option<MemoryGuard>) {
        *self.guard.lock().unwrap() = guard;
    }

    pub fn assign(&self, layers: [usize; 2]) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();
        if guard.as_ref().is_some_and(|l| [l.start as usize, l.end as usize] == layers) {
            return Ok(());
        }
        *guard = None;
        if let Some(g) = self.guard.lock().unwrap().clone() {
            g.check(&self.spec, layers, self.max_ctx, self.slots, self.device)?;
        }
        *guard = Some(load_range(&self.spec, layers, self.max_ctx, self.slots, self.device, self.api)?);
        Ok(())
    }

    pub fn layers(&self) -> Option<[usize; 2]> {
        self.inner.lock().unwrap().as_ref().map(|l| [l.start as usize, l.end as usize])
    }

    fn forward(&self, slot: u32, position: u32, input: Input, decode: bool) -> Result<Message> {
        ensure!((slot as usize) < self.slots, "slot {slot} out of {}", self.slots);
        let started = Instant::now();
        let mut guard = self.inner.lock().unwrap();
        let Loaded { engine, .. } = guard.as_mut().context("this stage has no layers loaded")?;
        let (rt, st) = match engine {
            Engine::Cuda { rt, st } => (rt, st),
            #[cfg(feature = "wgpu")]
            Engine::Wgpu(d) => return forward_wgpu(d, slot, position, input, decode, started),
        };
        let role = rt.role();
        let (tokens, rows) = match input {
            Input::Tokens(t) => {
                ensure!(role.first, "this stage has no embedding, send hidden rows");
                let rows = t.len();
                (t, rows)
            }
            Input::Hidden { rows, data } => {
                ensure!(!role.first, "this stage owns the embedding, send token ids");
                rt.import_hidden(&bytes_f32(&data)?)?;
                (vec![0; rows as usize], rows as usize)
            }
        };
        let token = if decode {
            ensure!(rows == 1, "a decode step carries one row");
            let row = DecodeRow { slot: slot as usize, token: tokens[0], pos: position as usize, sampling: Sampling::GREEDY, step: 0 };
            rt.decode(st, &[row])?[0]
        } else {
            let sample = role.last.then_some((Sampling::GREEDY, 0));
            rt.prefill_slot(st, slot as usize, &tokens, position as usize, sample)?.unwrap_or(0)
        };
        if role.last {
            let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            return Ok(Message::Token { token, compute_us });
        }
        let h = rt.export_hidden(rows)?;
        let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
        Ok(Message::Output { rows: rows as u32, data: f32_bytes(&h), compute_us })
    }

    fn decode_batch(&self, slots: Vec<u32>, positions: Vec<u32>, input: Input) -> Result<Message> {
        let started = Instant::now();
        ensure!(!slots.is_empty() && slots.len() == positions.len(), "a decode batch needs one position per slot");
        ensure!(slots.iter().all(|s| (*s as usize) < self.slots), "decode slot out of {}", self.slots);
        let mut guard = self.inner.lock().unwrap();
        let Loaded { engine, .. } = guard.as_mut().context("this stage has no layers loaded")?;
        let (rt, st) = match engine {
            Engine::Cuda { rt, st } => (rt, st),
            #[cfg(feature = "wgpu")]
            Engine::Wgpu(d) => return decode_batch_wgpu(d, &slots, &positions, input, started),
        };
        let role = rt.role();
        let rows = slots.len();
        let tokens = match input {
            Input::Tokens(t) => {
                ensure!(role.first, "this stage has no embedding, send hidden rows");
                ensure!(t.len() == rows, "{} tokens for {rows} rows", t.len());
                t
            }
            Input::Hidden { rows: n, data } => {
                ensure!(!role.first, "this stage owns the embedding, send token ids");
                ensure!(n as usize == rows, "{n} hidden rows for {rows} slots");
                rt.import_hidden(&bytes_f32(&data)?)?;
                vec![0; rows]
            }
        };
        let batch: Vec<DecodeRow> = (0..rows)
            .map(|i| DecodeRow { slot: slots[i] as usize, token: tokens[i], pos: positions[i] as usize, sampling: Sampling::GREEDY, step: 0 })
            .collect();
        let out = rt.decode(st, &batch)?;
        if role.last {
            let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            return Ok(Message::Tokens { tokens: out, compute_us });
        }
        let h = rt.export_hidden(rows)?;
        let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
        Ok(Message::Output { rows: rows as u32, data: f32_bytes(&h), compute_us })
    }
}

impl StageHandler for RuntimeStage {
    fn ready(&self) -> Message {
        let (layer_start, layer_end) = self.layers().map_or((0, 0), |[a, b]| (a as u32, b as u32));
        Message::Ready { layer_start, layer_end, hidden: self.spec.dims.hidden as u32 }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { slot, position, input, decode } => match self.forward(slot, position, input, decode) {
                Ok(reply) => reply,
                Err(e) => Message::Error { message: format!("{e:#}") },
            },
            Message::DecodeBatch { slots, positions, input } => match self.decode_batch(slots, positions, input) {
                Ok(reply) => reply,
                Err(e) => Message::Error { message: format!("{e:#}") },
            },
            Message::Assign { layer_start, layer_end } => match self.assign([layer_start as usize, layer_end as usize]) {
                Ok(()) => self.ready(),
                Err(e) => Message::Error { message: format!("{e:#}") },
            },
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            other => Message::Error { message: format!("unexpected {other:?}") },
        }
    }
}

#[cfg(feature = "wgpu")]
fn micros(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

#[cfg(feature = "wgpu")]
fn wgpu_input<'a>(input: &'a Input, hidden: &'a mut Vec<f32>) -> Result<crate::backends::dense::StageInput<'a>> {
    use crate::backends::dense::StageInput;
    Ok(match input {
        Input::Tokens(t) => StageInput::Tokens(t),
        Input::Hidden { data, .. } => {
            *hidden = bytes_f32(data)?;
            StageInput::Hidden(hidden)
        }
    })
}

#[cfg(feature = "wgpu")]
fn forward_wgpu(d: &mut crate::backends::dense::Dense<crate::backends::wgpu::Wgpu>, slot: u32, position: u32, input: Input, decode: bool, started: Instant) -> Result<Message> {
    use crate::backends::dense::StageOutput;
    let mut hidden = Vec::new();
    let rows = match &input {
        Input::Tokens(t) => t.len(),
        Input::Hidden { rows, .. } => *rows as usize,
    };
    ensure!(!decode || rows == 1, "a decode step carries one row");
    let out = d.stage_forward(slot as usize, position as usize, wgpu_input(&input, &mut hidden)?, decode, true)?;
    Ok(match out {
        StageOutput::Token(token) => Message::Token { token, compute_us: micros(started) },
        StageOutput::Hidden(h) => Message::Output { rows: rows as u32, data: f32_bytes(&h), compute_us: micros(started) },
    })
}

#[cfg(feature = "wgpu")]
fn decode_batch_wgpu(d: &mut crate::backends::dense::Dense<crate::backends::wgpu::Wgpu>, slots: &[u32], positions: &[u32], input: Input, started: Instant) -> Result<Message> {
    use crate::backends::dense::{StageInput, StageOutput};
    let hidden_width = d.hidden();
    let (tokens, hidden) = match &input {
        Input::Tokens(t) => {
            ensure!(t.len() == slots.len(), "{} tokens for {} rows", t.len(), slots.len());
            (t.clone(), Vec::new())
        }
        Input::Hidden { rows, data } => {
            ensure!(*rows as usize == slots.len(), "{rows} hidden rows for {} slots", slots.len());
            (Vec::new(), bytes_f32(data)?)
        }
    };
    let (mut ids, mut carried) = (Vec::new(), Vec::new());
    for (i, (&slot, &pos)) in slots.iter().zip(positions).enumerate() {
        let one = if tokens.is_empty() { StageInput::Hidden(&hidden[i * hidden_width..(i + 1) * hidden_width]) } else { StageInput::Tokens(&tokens[i..=i]) };
        match d.stage_forward(slot as usize, pos as usize, one, true, true)? {
            StageOutput::Token(t) => ids.push(t),
            StageOutput::Hidden(h) => carried.extend(h),
        }
    }
    if d.stage_role().1 {
        return Ok(Message::Tokens { tokens: ids, compute_us: micros(started) });
    }
    Ok(Message::Output { rows: slots.len() as u32, data: f32_bytes(&carried), compute_us: micros(started) })
}

pub struct ServedStage {
    pub server: StageServer,
    pub stage: Arc<RuntimeStage>,
    pub announcer: Option<crate::discovery::Announcer>,
}

pub fn serve(config: &StageConfig, placement: Option<&PlacementConfig>) -> Result<ServedStage> {
    let spec = Spec::load(&config.spec)?;
    let stage = Arc::new(match config.layers {
        Some(layers) => RuntimeStage::load(&spec, layers, config.max_ctx, config.slots)?,
        None => RuntimeStage::unloaded(&spec, config.max_ctx, config.slots),
    });
    let server = StageServer::start(&config.bind, &config.key()?, stage.clone())?;
    let announcer = match (placement.map(|p| p.discovery), &config.device) {
        (Some(Discovery::Mdns), Some(device)) => {
            let port = placement.map_or(5353, |p| p.mdns_port);
            let interface = placement.and_then(|p| p.mdns_interface.as_deref());
            Some(crate::discovery::Announcer::start_on(device, server.addr(), port, interface)?)
        }
        (Some(Discovery::Mdns), None) => anyhow::bail!("[placement] discovery = \"mdns\" needs a [stage.device] table to announce"),
        _ => None,
    };
    Ok(ServedStage { server, stage, announcer })
}

pub struct Generated {
    pub tokens: Vec<u32>,
    pub prefill: Duration,
    pub decode: Duration,
    pub decode_steps: Vec<ChainTimes>,
    pub prefill_steps: Vec<ChainTimes>,
}

pub fn generate(chain: &mut Chain, slot: u32, prompt: &[u32], new_tokens: usize, chunk: usize) -> Result<Generated> {
    ensure!(!prompt.is_empty() && new_tokens >= 1 && chunk >= 1, "need a prompt, at least one new token and a chunk size");
    let prefill_started = Instant::now();
    let mut prefill_steps = Vec::new();
    let mut next = 0;
    let mut at = 0usize;
    for piece in prompt.chunks(chunk) {
        let (token, times) = chain.forward(slot, at as u32, piece.to_vec(), false)?;
        prefill_steps.push(times);
        next = token;
        at += piece.len();
    }
    let prefill = prefill_started.elapsed();
    let mut tokens = vec![next];
    let decode_started = Instant::now();
    let mut decode_steps = Vec::new();
    while tokens.len() < new_tokens {
        let (token, times) = chain.forward(slot, at as u32, vec![*tokens.last().unwrap()], true)?;
        decode_steps.push(times);
        tokens.push(token);
        at += 1;
    }
    let decode = decode_started.elapsed();
    chain.free(slot)?;
    Ok(Generated { tokens, prefill, decode, decode_steps, prefill_steps })
}
