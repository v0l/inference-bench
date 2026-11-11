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
    pub layers: [usize; 2],
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

fn read_key(inline: &Option<String>, file: &Option<PathBuf>) -> Result<Vec<u8>> {
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
    ensure!(b.len() % 4 == 0, "hidden payload of {} bytes is not f32", b.len());
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

struct Inner {
    rt: Runtime,
    st: Stepper,
}

unsafe impl Send for Inner {}

pub struct RuntimeStage {
    inner: Mutex<Inner>,
    start: u32,
    end: u32,
    hidden: u32,
    slots: u32,
}

impl RuntimeStage {
    pub fn load(spec: &Spec, layers: [usize; 2], max_ctx: usize, slots: usize) -> Result<RuntimeStage> {
        let pages = max_ctx.div_ceil(crate::codegen::PAGE);
        let layout = Layout { kv_planes: 1, ..Layout::contiguous(max_ctx, slots) };
        let range = StageRange { device: 0, start: layers[0], end: layers[1] };
        let mut rt = Runtime::load_stage(spec, layout, range)?;
        for slot in 0..slots {
            let blocks: Vec<u32> = (0..pages as u32).map(|p| (slot * pages) as u32 + p).collect();
            rt.set_block_table(slot, &blocks)?;
        }
        let st = rt.stepper()?;
        Ok(RuntimeStage {
            inner: Mutex::new(Inner { rt, st }),
            start: layers[0] as u32,
            end: layers[1] as u32,
            hidden: spec.dims.hidden as u32,
            slots: slots as u32,
        })
    }

    fn forward(&self, slot: u32, position: u32, input: Input, decode: bool) -> Result<Message> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let started = Instant::now();
        let mut guard = self.inner.lock().unwrap();
        let Inner { rt, st } = &mut *guard;
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
        let compute_us = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
        if role.last {
            return Ok(Message::Token { token, compute_us });
        }
        let h = rt.export_hidden(rows)?;
        Ok(Message::Output { rows: rows as u32, data: f32_bytes(&h), compute_us })
    }
}

impl StageHandler for RuntimeStage {
    fn ready(&self) -> Message {
        Message::Ready { layer_start: self.start, layer_end: self.end, hidden: self.hidden }
    }

    fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward { slot, position, input, decode } => match self.forward(slot, position, input, decode) {
                Ok(reply) => reply,
                Err(e) => Message::Error { message: format!("{e:#}") },
            },
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            other => Message::Error { message: format!("unexpected {other:?}") },
        }
    }
}

pub fn serve(config: &StageConfig) -> Result<StageServer> {
    let spec = Spec::load(&config.spec)?;
    let stage = Arc::new(RuntimeStage::load(&spec, config.layers, config.max_ctx, config.slots)?);
    StageServer::start(&config.bind, &config.key()?, stage)
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
