use super::prompt::{self, Prompt, PromptParts};
use super::sampling::{Interrupted, Mode, TextState};
use crate::codec::HostCodec;
use crate::codec::cuda::{CudaCodec, FrameBufs, Sessions as CodecSessions};
use crate::flow_head::FlowHead;
use crate::flow_head_cuda::{CudaHead, CudaSession};
use crate::runtime::step::{DecodeRow, Stepper, row_capacity};
use crate::runtime::{Layout, Runtime, Sampling};
use crate::spec::{Duplex, Spec};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const BATCHES: [usize; 6] = [1, 2, 4, 8, 16, 32];

pub fn padded(n: usize) -> usize {
    BATCHES.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub slots: usize,
    pub max_ctx: usize,
    pub graphs: bool,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub max_frames: Option<usize>,
    pub window_frames: Option<usize>,
    pub kv_bytes: Option<usize>,
}

impl Default for EngineOptions {
    fn default() -> EngineOptions {
        EngineOptions { slots: 1, max_ctx: 4096, graphs: true, temperature: None, top_k: None, max_frames: None, window_frames: None, kv_bytes: None }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StartOptions {
    pub card: String,
    pub reference: Option<Vec<f32>>,
    pub speak_first: bool,
    pub seed: u64,
    pub guidance: Option<f32>,
    pub greedy: bool,
    pub mode: Mode,
    pub noise: Option<RecordedNoise>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RecordedNoise {
    pub head: Vec<f32>,
    pub user: Vec<f32>,
    pub user_scale: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub audio: Vec<f32>,
    pub token: u32,
    pub text: String,
    pub index: u64,
    pub latent: Vec<f32>,
    pub done: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Live {
    pub slot: usize,
    pub frame: u64,
    pub prompt_len: usize,
    pub kept: usize,
    pub token: u32,
    pub latent: Vec<f32>,
    pub text: TextState,
    pub seed: u64,
    pub greedy: bool,
    pub guidance: f32,
    pub noise: Option<RecordedNoise>,
}

pub type SessionId = u64;

struct Batch {
    head: CudaSession,
    enc: CodecSessions,
    dec: CodecSessions,
    enc_io: FrameBufs,
    dec_io: FrameBufs,
    user_noise: CudaSlice<f32>,
    user_scale: CudaSlice<f32>,
    model_latent: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    edit_tok: CudaSlice<i32>,
    edit_add: CudaSlice<f32>,
    edit_count: CudaSlice<i32>,
    allow: CudaSlice<i32>,
    allow_count: CudaSlice<i32>,
    exact: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    zero: CudaSlice<i32>,
    seg_off: [Vec<usize>; 2],
    seg_len: [CudaSlice<i32>; 2],
    graph: Option<CudaGraph>,
    guidance: Vec<f32>,
}

pub struct Engine {
    pub spec: Spec,
    pub duplex: Duplex,
    pub rt: Runtime,
    pub st: Stepper,
    pub head: CudaHead,
    pub codec: CudaCodec,
    host_codec: HostCodec,
    tokenizer: Option<tokenizers::Tokenizer>,
    stream: Arc<CudaStream>,
    batches: BTreeMap<usize, Batch>,
    pools: Pools,
    sessions: BTreeMap<SessionId, Live>,
    free: Vec<usize>,
    next_id: SessionId,
    silence: Vec<f32>,
    opts: EngineOptions,
    edit_stride: usize,
    pub capture_secs: Vec<(usize, f64)>,
}

struct Pools {
    enc: CudaSlice<f32>,
    dec: CudaSlice<f32>,
    enc_floats: usize,
    dec_floats: usize,
}

fn cfg(x: u32, y: u32, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (x.max(1), y.max(1), 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

fn spread(n: usize) -> u32 {
    n.div_ceil(256).clamp(1, 1024) as u32
}

impl Engine {
    pub fn load(dir: &Path, opts: EngineOptions) -> Result<Engine> {
        let text = std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("{}/config.json", dir.display()))?;
        let mut spec = Spec::from_hf_config_duplex(&text)?;
        spec.repo = dir.display().to_string();
        spec.name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| spec.name.clone());
        Self::load_spec(spec, opts)
    }

    pub fn load_spec(mut spec: Spec, opts: EngineOptions) -> Result<Engine> {
        let mut duplex = spec.duplex.clone().context("the spec declares no [duplex] section")?;
        if let Some(t) = opts.temperature {
            duplex.sampling.temperature = t;
        }
        if let Some(k) = opts.top_k {
            duplex.sampling.top_k = k;
        }
        if let Some(m) = opts.max_frames {
            duplex.max_frames = m;
        }
        if let Some(w) = opts.window_frames {
            duplex.window_frames = w;
        }
        spec.runtime.split_activations.get_or_insert(true);
        let dir = PathBuf::from(&spec.repo);
        let pages = opts.max_ctx.div_ceil(crate::codegen::PAGE);
        let kv_blocks = match opts.kv_bytes {
            Some(b) => (b / crate::runtime::kv_block_bytes(&spec, 2)).max(pages),
            None => opts.slots * pages,
        };
        let layout = Layout { max_ctx: opts.max_ctx, slots: opts.slots, kv_blocks, prefill_chunk: duplex.prefill_chunk.min(opts.max_ctx), kv_planes: 2, kv_headroom: 0, state_on_demand: false };
        let rt = Runtime::load_layout(&spec, layout)?;
        let st = rt.stepper()?;
        let stream = rt.stream().clone();
        let cache = spec.runtime.cache_dir.clone().unwrap_or_else(crate::jit::default_cache_dir);
        let jit = crate::jit::Jit::new(stream.context().clone(), cache)?;
        let head = CudaHead::load(&stream, &jit, FlowHead::from_dir(&dir, None)?, &dir, spec.runtime.fuse)?;
        let host_codec = HostCodec::open(&dir)?;
        let mut shapes: Vec<(usize, usize)> = BATCHES.iter().copied().filter(|&b| b <= padded(opts.slots)).map(|b| (1, b)).collect();
        shapes.push((duplex.silence_frames, 1));
        let codec = CudaCodec::load_on(&stream, &jit, &host_codec, &shapes)?;
        let tokenizer = ["tokenizer.json"].iter().map(|n| dir.join(n)).find(|p| p.exists()).map(tokenizers::Tokenizer::from_file).transpose().map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let enc_floats = codec.sessions(&codec.encoder, 1)?.floats_per_session();
        let dec_floats = codec.sessions(&codec.decoder, 1)?.floats_per_session();
        let pools = Pools { enc: stream.alloc_zeros(opts.slots * enc_floats)?, dec: stream.alloc_zeros(opts.slots * dec_floats)?, enc_floats, dec_floats };
        let edit_stride = 8 + duplex.sampling.forced_bans.len() + duplex.sampling.dry.as_ref().map_or(0, |d| d.history_size());
        let mut e = Engine {
            free: (0..opts.slots).rev().collect(),
            spec,
            duplex,
            rt,
            st,
            head,
            codec,
            host_codec,
            tokenizer,
            stream,
            batches: BTreeMap::new(),
            pools,
            sessions: BTreeMap::new(),
            next_id: 1,
            silence: Vec::new(),
            opts,
            edit_stride,
            capture_secs: Vec::new(),
        };
        e.silence = e.encode_whole(&vec![0.0; e.duplex.silence_frames * e.duplex.frame_samples])?.split_off((e.duplex.silence_frames - 1) * e.duplex.latent_dim);
        Ok(e)
    }

    pub fn silence_latent(&self) -> &[f32] {
        &self.silence
    }

    pub fn host_codec(&self) -> &HostCodec {
        &self.host_codec
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let t = self.tokenizer.as_ref().context("the model folder has no tokenizer.json")?;
        Ok(t.encode(text, false).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?.get_ids().to_vec())
    }

    pub fn detokenize(&self, ids: &[u32]) -> String {
        match &self.tokenizer {
            Some(t) => t.decode(ids, false).unwrap_or_default(),
            None => ids.iter().map(|i| format!("<{i}>")).collect(),
        }
    }

    fn encode_whole(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let f = self.duplex.frame_samples;
        let padded_len = audio.len().div_ceil(f) * f;
        let mut a = audio.to_vec();
        a.resize(padded_len, 0.0);
        let mut s = self.codec.sessions(&self.codec.encoder, 1)?;
        let mut out = Vec::new();
        for chunk in a.chunks(self.duplex.silence_frames * f) {
            out.extend(self.codec.run(&self.codec.encoder, chunk, &mut s)?);
        }
        Ok(out)
    }

    fn noise_latents(&self, mean: &[f32], seed: u64, frame: u64) -> Vec<f32> {
        let d = self.duplex.latent_dim;
        let k = self.duplex.noise_scale();
        let rows = mean.len() / d;
        let scale = crate::sampler::noise(seed ^ 0x5eed_0000 ^ frame, rows);
        let n = crate::sampler::noise(seed ^ 0xa11c_e000 ^ frame.wrapping_mul(31), mean.len());
        mean.iter().enumerate().map(|(i, m)| m + scale[i / d] * k * n[i]).collect()
    }

    pub fn sessions(&self) -> impl Iterator<Item = (&SessionId, &Live)> {
        self.sessions.iter()
    }

    pub fn live(&self, id: SessionId) -> Result<&Live> {
        self.sessions.get(&id).with_context(|| format!("no duplex session {id}"))
    }

    fn live_mut(&mut self, id: SessionId) -> Result<&mut Live> {
        self.sessions.get_mut(&id).with_context(|| format!("no duplex session {id}"))
    }

    pub fn prompt_for(&self, o: &StartOptions) -> Result<Prompt> {
        let card_text = prompt::card_text(&self.duplex, &o.card);
        let card = if card_text.is_empty() { Vec::new() } else { self.tokenize(&card_text)? };
        let reference = o.reference.as_ref().map(|r| prompt::trim_reference(&self.duplex, r)).filter(|r| !r.is_empty());
        let reference_latents = match &reference {
            Some(r) => self.noise_latents(&self.encode_whole(r)?, o.seed, u64::MAX),
            None => Vec::new(),
        };
        prompt::build(&self.duplex, &PromptParts { card: &card, reference: reference.as_deref(), reference_latents: &reference_latents, silence: &self.silence, speak_first: o.speak_first })
    }

    pub fn start(&mut self, o: StartOptions) -> Result<(SessionId, Frame)> {
        let p = self.prompt_for(&o)?;
        self.start_prompt(&p, o)
    }

    pub fn start_prompt(&mut self, p: &Prompt, o: StartOptions) -> Result<(SessionId, Frame)> {
        ensure!(!p.is_empty(), "an empty prompt");
        ensure!(p.len() < self.opts.max_ctx, "a prompt of {} positions does not fit a context of {}", p.len(), self.opts.max_ctx);
        let slot = self.free.pop().context("every duplex slot is in use")?;
        let started = self.prefill(slot, p);
        if let Err(e) = started {
            self.free.push(slot);
            return Err(e);
        }
        let guidance = o.guidance.unwrap_or(self.duplex.guidance as f32);
        let id = self.next_id;
        self.next_id += 1;
        let live = Live {
            slot,
            frame: 0,
            prompt_len: p.len(),
            kept: p.len(),
            token: 0,
            latent: vec![0.0; self.duplex.latent_dim],
            text: TextState { mode: o.mode, ..TextState::new(if o.speak_first { self.duplex.sampling.forced_text } else { 0 }) },
            seed: o.seed,
            greedy: o.greedy,
            guidance,
            noise: o.noise,
        };
        self.zero_codec(slot)?;
        self.sessions.insert(id, live);
        let frame = self.first_frame(id)?;
        Ok((id, frame))
    }

    fn prefill(&mut self, slot: usize, p: &Prompt) -> Result<()> {
        self.rt.set_position_offset(slot, 0)?;
        let pages = self.rt.kv_pages();
        let blocks: Vec<u32> = (0..pages as u32).map(|i| (slot * pages) as u32 + i).collect();
        self.rt.set_block_table(slot, &blocks)?;
        let chunk = self.duplex.prefill_chunk.min(self.rt.input_capacity()).min(self.rt.prefill_chunk_limit());
        let d = self.duplex.latent_dim;
        for start in (0..p.len()).step_by(chunk) {
            let end = (start + chunk).min(p.len());
            self.rt.set_inputs(end - start, &[
                ("model_latent", &p.model_latents[start * d..end * d]),
                ("user_latent", &p.user_latents[start * d..end * d]),
                ("model_mask", &p.model_mask[start..end]),
                ("user_mask", &p.user_mask[start..end]),
            ])?;
            self.rt.prefill_slot(&mut self.st, slot, &p.tokens[start..end], start, None)?;
        }
        Ok(())
    }

    fn zero_codec(&mut self, slot: usize) -> Result<()> {
        let (e, d) = (self.pools.enc_floats, self.pools.dec_floats);
        self.stream.memset_zeros(&mut self.pools.enc.slice_mut(slot * e..(slot + 1) * e))?;
        self.stream.memset_zeros(&mut self.pools.dec.slice_mut(slot * d..(slot + 1) * d))?;
        Ok(())
    }

    fn batch(&mut self, n: usize) -> Result<&mut Batch> {
        if !self.batches.contains_key(&n) {
            let d = self.duplex.latent_dim;
            let rows = row_capacity(self.opts.slots);
            ensure!(n < rows + 1, "a batch of {n} rows for {} slots", self.opts.slots);
            let exact: Vec<i32> = self.duplex.sampling.exact.iter().map(|&t| t as i32).collect();
            let enc = self.codec.sessions(&self.codec.encoder, n)?.fixed();
            let dec = self.codec.sessions(&self.codec.decoder, n)?.fixed();
            let mut seg_off: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut seg_len_host: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
            for (i, (s, total)) in [(&enc, self.pools.enc_floats), (&dec, self.pools.dec_floats)].into_iter().enumerate() {
                let mut off = 0;
                for (_, _, len) in s.segments(&self.stream) {
                    seg_off[i].push(off);
                    seg_len_host[i].push(len as i32);
                    off += len;
                }
                ensure!(off == total, "codec state of {off} floats, pool rows of {total}");
            }
            let b = Batch {
                head: self.head.session(&vec![self.duplex.guidance as f32; n])?,
                enc,
                dec,
                enc_io: self.codec.frame_buffers(&self.codec.encoder, self.duplex.frame_samples, n)?,
                dec_io: self.codec.frame_buffers(&self.codec.decoder, 1, n)?,
                user_noise: self.stream.alloc_zeros(n * d)?,
                user_scale: self.stream.alloc_zeros(n)?,
                model_latent: self.stream.alloc_zeros(n * d)?,
                ones: self.stream.clone_htod(&vec![1.0f32; n])?,
                edit_tok: self.stream.alloc_zeros(n * self.edit_stride)?,
                edit_add: self.stream.alloc_zeros(n * self.edit_stride)?,
                edit_count: self.stream.alloc_zeros(n)?,
                allow: self.stream.alloc_zeros(n * 4)?,
                allow_count: self.stream.alloc_zeros(n)?,
                exact: self.stream.clone_htod(&if exact.is_empty() { vec![0] } else { exact })?,
                slots: self.stream.alloc_zeros(n)?,
                zero: self.stream.alloc_zeros(1)?,
                seg_off,
                seg_len: [self.stream.clone_htod(&seg_len_host[0])?, self.stream.clone_htod(&seg_len_host[1])?],
                graph: None,
                guidance: vec![self.duplex.guidance as f32; n],
            };
            self.batches.insert(n, b);
        }
        Ok(self.batches.get_mut(&n).unwrap())
    }

    fn move_codec(&mut self, n: usize, slots: &[usize], to_batch: bool) -> Result<()> {
        let stream = self.stream.clone();
        let rows = slots.len();
        let kernel = self.rt.module_kernel("duplex_state_move")?.clone();
        let pools = [(self.pools.enc.device_ptr(&stream).0, self.pools.enc_floats), (self.pools.dec.device_ptr(&stream).0, self.pools.dec_floats)];
        let slot_ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        let b = self.batches.get_mut(&n).context("no batch")?;
        stream.memcpy_htod(&slot_ids, &mut b.slots.slice_mut(..rows))?;
        for (i, (pool, total)) in pools.into_iter().enumerate() {
            let sess = if i == 0 { &b.enc } else { &b.dec };
            for (j, &(a, _, _)) in sess.segments(&stream).iter().enumerate() {
                let off = b.seg_off[i][j];
                let base = pool + (off as u64) * 4;
                let (one_off, one_len) = (b.zero.device_ptr(&stream).0, b.seg_len[i].device_ptr(&stream).0 + (j as u64) * 4);
                let mut lb = stream.launch_builder(&kernel);
                let (t, r, tb) = (total as i32, n as i32, to_batch as i32);
                lb.arg(&base).arg(&a).arg(&b.slots).arg(&one_off).arg(&one_len).arg(&t).arg(&r).arg(&tb);
                unsafe { lb.launch(cfg(1, rows as u32, 256)) }?;
            }
        }
        Ok(())
    }

    fn first_frame(&mut self, id: SessionId) -> Result<Frame> {
        let n = 1;
        self.batch(n)?;
        self.rt.prompt_logits_to_row(&mut self.st, 0)?;
        let h = self.spec.dims.hidden;
        let stream = self.stream.clone();
        let join = self.rt.module_kernel("duplex_join")?.clone();
        let (hi, lo) = self.rt.normed_planes();
        let (hp, lp) = (hi.device_ptr(&stream).0, lo.device_ptr(&stream).0);
        let b = self.batches.get_mut(&n).unwrap();
        let hidden_ptr = self.head.input_mut(&mut b.head, "hidden")?.device_ptr(&stream).0;
        let total = h as i32;
        let mut lb = stream.launch_builder(&join);
        lb.arg(&hp).arg(&lp).arg(&hidden_ptr).arg(&total);
        unsafe { lb.launch(cfg(spread(h), 1, 256)) }?;
        self.text_and_head(&[id], n)?;
        let out = self.finish(&[id], n)?;
        Ok(out.into_iter().next().unwrap())
    }

    fn plans(&mut self, ids: &[SessionId], n: usize) -> Result<()> {
        let s = self.duplex.sampling.clone();
        let (pad, eop) = (self.duplex.pad, self.duplex.eop);
        let stride = self.edit_stride;
        let mut tok = vec![0i32; n * stride];
        let mut add = vec![0f32; n * stride];
        let mut count = vec![0i32; n];
        let mut allow = vec![0i32; n * 4];
        let mut allow_n = vec![0i32; n];
        for (r, id) in ids.iter().enumerate() {
            let live = self.sessions.get_mut(id).unwrap();
            let plan = live.text.plan(&s, pad, eop);
            ensure!(plan.edits.len() <= stride && plan.allow.len() <= 4, "a sampler plan of {} edits", plan.edits.len());
            for (j, (t, a)) in plan.edits.iter().enumerate() {
                tok[r * stride + j] = *t as i32;
                add[r * stride + j] = *a;
            }
            count[r] = plan.edits.len() as i32;
            for (j, t) in plan.allow.iter().enumerate() {
                allow[r * 4 + j] = *t as i32;
            }
            allow_n[r] = plan.allow.len() as i32;
        }
        let b = self.batches.get_mut(&n).unwrap();
        self.stream.memcpy_htod(&tok, &mut b.edit_tok)?;
        self.stream.memcpy_htod(&add, &mut b.edit_add)?;
        self.stream.memcpy_htod(&count, &mut b.edit_count)?;
        self.stream.memcpy_htod(&allow, &mut b.allow)?;
        self.stream.memcpy_htod(&allow_n, &mut b.allow_count)?;
        Ok(())
    }

    fn sampling_of(&self, live: &Live) -> Sampling {
        if live.greedy {
            return Sampling::GREEDY;
        }
        let s = &self.duplex.sampling;
        Sampling { temperature: s.temperature, seed: (live.seed as u32) ^ (live.slot as u32).wrapping_mul(0x9e37_79b9), top_k: s.top_k, top_p: s.top_p, min_p: 0.0, logprobs: None }
    }

    fn noise_inputs(&mut self, ids: &[SessionId], n: usize, with_user: bool) -> Result<()> {
        let d = self.duplex.latent_dim;
        let mut head = vec![0f32; n * d];
        let mut user = vec![0f32; n * d];
        let mut scale = vec![0f32; n];
        for (r, id) in ids.iter().enumerate() {
            let live = &self.sessions[id];
            let f = live.frame as usize;
            match &live.noise {
                Some(rec) => {
                    head[r * d..(r + 1) * d].copy_from_slice(rec.head.get(f * d..(f + 1) * d).context("the recorded head noise ran out")?);
                    if with_user && f > 0 {
                        user[r * d..(r + 1) * d].copy_from_slice(rec.user.get((f - 1) * d..f * d).context("the recorded user noise ran out")?);
                        scale[r] = *rec.user_scale.get(f - 1).context("the recorded user scale ran out")?;
                    }
                }
                None => {
                    head[r * d..(r + 1) * d].copy_from_slice(&crate::flow_head::frame_noise(live.seed, live.frame, d));
                    if with_user {
                        let un = crate::flow_head::frame_noise(live.seed ^ 0x75_5e72, live.frame, d + 1);
                        scale[r] = un[d];
                        user[r * d..(r + 1) * d].copy_from_slice(&un[..d]);
                    }
                }
            }
        }
        let b = self.batches.get_mut(&n).unwrap();
        let noise = self.head.input_mut(&mut b.head, "noise")?;
        self.stream.memcpy_htod(&head, noise)?;
        self.stream.memcpy_htod(&user, &mut b.user_noise)?;
        self.stream.memcpy_htod(&scale, &mut b.user_scale)?;
        Ok(())
    }

    fn set_guidance(&mut self, ids: &[SessionId], n: usize) -> Result<()> {
        let mut g: Vec<f32> = ids.iter().map(|id| self.sessions[id].guidance).collect();
        g.resize(n, self.duplex.guidance as f32);
        let b = self.batches.get_mut(&n).unwrap();
        if b.guidance != g {
            self.head.set_guidance(&mut b.head, &g)?;
            b.guidance = g;
        }
        Ok(())
    }

    fn text_and_head(&mut self, ids: &[SessionId], n: usize) -> Result<()> {
        self.set_guidance(ids, n)?;
        self.plans(ids, n)?;
        self.noise_inputs(ids, n, false)?;
        let rows: Vec<DecodeRow> = ids.iter().map(|id| {
            let l = &self.sessions[id];
            DecodeRow { slot: l.slot, token: 0, pos: 0, sampling: self.sampling_of(l), step: l.frame as u32 }
        }).collect();
        self.stage_params(&rows, n)?;
        self.text_kernels(n)?;
        let b = self.batches.get_mut(&n).unwrap();
        self.head.run_frame(&mut b.head)?;
        Ok(())
    }

    fn stage_params(&mut self, rows: &[DecodeRow], n: usize) -> Result<()> {
        let p: Vec<i32> = rows.iter().map(|r| crate::runtime::step::sample_params(r.sampling, r.step)).chain(std::iter::repeat([0i32; 6])).take(n).flatten().collect();
        self.stream.memcpy_htod(&p, &mut self.st.params_mut().slice_mut(..6 * n))?;
        Ok(())
    }

    fn text_kernels(&self, n: usize) -> Result<()> {
        let stream = &self.stream;
        let b = &self.batches[&n];
        let vocab = self.spec.dims.vocab as i32;
        let logits = self.st.row_logits().device_ptr(stream).0;
        let (hi, lo) = self.rt.normed_planes();
        let (hp, lp) = (hi.device_ptr(stream).0, lo.device_ptr(stream).0);
        let table = self.rt.embed_table().device_ptr(stream).0;
        let exact_n = self.duplex.sampling.exact.len() as u32;
        if exact_n > 0 {
            let mut lb = stream.launch_builder(self.rt.module_kernel("duplex_exact_logits")?);
            let ex = b.exact.device_ptr(stream).0;
            lb.arg(&hp).arg(&lp).arg(&table).arg(&ex).arg(&logits);
            unsafe { lb.launch(cfg(n as u32, exact_n, 256)) }?;
        }
        let stride = self.edit_stride as i32;
        let mut lb = stream.launch_builder(self.rt.module_kernel("duplex_edit_logits")?);
        lb.arg(&logits).arg(&b.edit_tok).arg(&b.edit_add).arg(&b.edit_count).arg(&stride);
        unsafe { lb.launch(cfg(n as u32, 1, 64)) }?;
        let four = 4i32;
        let mut lb = stream.launch_builder(self.rt.module_kernel("duplex_allow")?);
        lb.arg(&logits).arg(&b.allow).arg(&b.allow_count).arg(&four);
        unsafe { lb.launch(cfg(n as u32, (vocab as u32).div_ceil(256).min(64), 256)) }?;
        self.rt.sample_staged(&self.st, n)?;
        Ok(())
    }

    fn finish(&mut self, ids: &[SessionId], n: usize) -> Result<Vec<Frame>> {
        let d = self.duplex.latent_dim;
        let f = self.duplex.frame_samples;
        let slots: Vec<usize> = ids.iter().map(|id| self.sessions[id].slot).collect();
        let stream = self.stream.clone();
        {
            let b = self.batches.get_mut(&n).unwrap();
            let latent = self.head.output(&b.head).clone();
            stream.memcpy_dtod(&latent, &mut b.model_latent)?;
            let input_ptr = b.dec_io.input_ptr();
            let src = b.model_latent.device_ptr(&stream).0;
            unsafe { cudarc::driver::result::memcpy_dtod_async(input_ptr, src, n * d * 4, stream.cu_stream()) }?;
        }
        self.move_codec(n, &slots, true)?;
        {
            let b = self.batches.get_mut(&n).unwrap();
            self.codec.launch_frame(&self.codec.decoder, &b.dec_io, &mut b.dec)?;
        }
        self.move_codec(n, &slots, false)?;
        let b = &self.batches[&n];
        let audio = stream.clone_dtoh(&b.dec_io.output().slice(..n * f))?;
        let latents = stream.clone_dtoh(&b.model_latent.slice(..n * d))?;
        let tokens = stream.clone_dtoh(&self.st.row_tokens().slice(..n))?;
        self.emit(ids, &tokens, &latents, &audio)
    }

    fn emit(&mut self, ids: &[SessionId], tokens: &[i32], latents: &[f32], audio: &[f32]) -> Result<Vec<Frame>> {
        let d = self.duplex.latent_dim;
        let f = self.duplex.frame_samples;
        let s = self.duplex.sampling.clone();
        let silent = s.silent.clone();
        let max_frames = self.duplex.max_frames as u64;
        let mut out = Vec::with_capacity(ids.len());
        for (r, id) in ids.iter().enumerate() {
            let token = tokens[r] as u32;
            let mut live = self.sessions.remove(id).unwrap();
            live.text.accept(token, &s);
            let decode = |ids: &[u32]| self.detokenize(ids);
            let text = live.text.text(token, &silent, &decode);
            live.token = token;
            live.latent = latents[r * d..(r + 1) * d].to_vec();
            let index = live.frame;
            live.frame += 1;
            let room = live.kept + 1 < self.opts.max_ctx || self.duplex.window_frames > 0;
            let done = live.frame >= max_frames || !room;
            out.push(Frame { audio: audio[r * f..(r + 1) * f].iter().map(|v| v.clamp(-1.0, 1.0)).collect(), token, text, index, latent: live.latent.clone(), done });
            self.sessions.insert(*id, live);
        }
        Ok(out)
    }

    pub fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> Result<Vec<Frame>> {
        ensure!(!frames.is_empty(), "a step with no sessions");
        let f = self.duplex.frame_samples;
        let d = self.duplex.latent_dim;
        let ids: Vec<SessionId> = frames.iter().map(|(id, _)| *id).collect();
        for (id, pcm) in frames {
            ensure!(pcm.len() == f, "a user frame is {f} samples, got {}", pcm.len());
            let l = self.live(*id)?;
            ensure!(l.frame < self.duplex.max_frames as u64, "session {id} reached max_frames");
        }
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        ensure!(sorted.len() == ids.len(), "a session appears twice in one step");
        for id in &ids {
            self.window(*id)?;
        }
        let n = padded(ids.len());
        self.batch(n)?;
        let mut rows: Vec<DecodeRow> = Vec::new();
        for id in &ids {
            let l = &self.sessions[id];
            rows.push(DecodeRow { slot: l.slot, token: l.token, pos: l.kept, sampling: self.sampling_of(l), step: l.frame as u32 });
        }
        let slots: Vec<usize> = ids.iter().map(|id| self.sessions[id].slot).collect();
        self.rt.stage_rows(&mut self.st, &rows, n)?;
        self.set_guidance(&ids, n)?;
        self.plans(&ids, n)?;
        self.noise_inputs(&ids, n, true)?;
        let mut pcm = vec![0f32; n * f];
        let mut model = vec![0f32; n * d];
        for (r, (id, frame)) in frames.iter().enumerate() {
            pcm[r * f..(r + 1) * f].copy_from_slice(frame);
            model[r * d..(r + 1) * d].copy_from_slice(&self.sessions[id].latent);
        }
        {
            let b = self.batches.get_mut(&n).unwrap();
            self.stream.memcpy_htod(&pcm, b.enc_io.input())?;
            self.stream.memcpy_htod(&model, &mut b.model_latent)?;
        }
        self.move_codec(n, &slots, true)?;
        let graphs = self.opts.graphs;
        if graphs {
            if self.batches[&n].graph.is_none() {
                let started = std::time::Instant::now();
                self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let recorded = self.frame_body(n);
                let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
                recorded?;
                let graph = graph.context("an empty duplex frame graph")?;
                graph.upload()?;
                self.stream.synchronize()?;
                self.capture_secs.push((n, started.elapsed().as_secs_f64()));
                self.batches.get_mut(&n).unwrap().graph = Some(graph);
            }
            self.batches[&n].graph.as_ref().unwrap().launch()?;
        } else {
            self.frame_body(n)?;
        }
        self.move_codec(n, &slots, false)?;
        let b = &self.batches[&n];
        let audio = self.stream.clone_dtoh(&b.dec_io.output().slice(..n * f))?;
        let latents = self.stream.clone_dtoh(&b.model_latent.slice(..n * d))?;
        let tokens = self.stream.clone_dtoh(&self.st.row_tokens().slice(..n))?;
        for id in &ids {
            self.sessions.get_mut(id).unwrap().kept += 1;
        }
        self.emit(&ids, &tokens, &latents, &audio)
    }

    fn frame_body(&mut self, n: usize) -> Result<()> {
        let stream = self.stream.clone();
        let d = self.duplex.latent_dim;
        let h = self.spec.dims.hidden;
        let k = self.duplex.noise_scale();
        {
            let b = self.batches.get_mut(&n).unwrap();
            self.codec.launch_frame(&self.codec.encoder, &b.enc_io, &mut b.enc)?;
        }
        let b = &self.batches[&n];
        let mean = b.enc_io.output().device_ptr(&stream).0;
        let user_ptr = self.rt.input_buffer("user_latent")?.device_ptr(&stream).0;
        {
            let mut lb = stream.launch_builder(self.rt.module_kernel("duplex_user_latent")?);
            let (noise, scale) = (b.user_noise.device_ptr(&stream).0, b.user_scale.device_ptr(&stream).0);
            let rows = n as i32;
            lb.arg(&mean).arg(&noise).arg(&scale).arg(&user_ptr).arg(&rows).arg(&k);
            unsafe { lb.launch(cfg(spread(n * d), 1, 256)) }?;
        }
        let (model_src, ones) = (b.model_latent.device_ptr(&stream).0, b.ones.device_ptr(&stream).0);
        for (name, src, len) in [("model_latent", model_src, n * d), ("model_mask", ones, n), ("user_mask", ones, n)] {
            let dst = self.rt.input_buffer(name)?.device_ptr(&stream).0;
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, len * 4, stream.cu_stream()) }?;
        }
        self.rt.forward_rows(&self.st, n)?;
        {
            let join = self.rt.module_kernel("duplex_join")?.clone();
            let (hi, lo) = self.rt.normed_planes();
            let b = self.batches.get_mut(&n).unwrap();
            let hidden = self.head.input_mut(&mut b.head, "hidden")?.device_ptr(&stream).0;
            let (hp, lp) = (hi.device_ptr(&stream).0, lo.device_ptr(&stream).0);
            let total = (n * h) as i32;
            let mut lb = stream.launch_builder(&join);
            lb.arg(&hp).arg(&lp).arg(&hidden).arg(&total);
            unsafe { lb.launch(cfg(spread(n * h), 1, 256)) }?;
        }
        self.text_kernels(n)?;
        {
            let b = self.batches.get_mut(&n).unwrap();
            self.head.run_frame(&mut b.head)?;
            let latent = self.head.output(&b.head).device_ptr(&stream).0;
            let (ml, di) = (b.model_latent.device_ptr(&stream).0, b.dec_io.input_ptr());
            unsafe { cudarc::driver::result::memcpy_dtod_async(ml, latent, n * d * 4, stream.cu_stream()) }?;
            unsafe { cudarc::driver::result::memcpy_dtod_async(di, latent, n * d * 4, stream.cu_stream()) }?;
            self.codec.launch_frame(&self.codec.decoder, &b.dec_io, &mut b.dec)?;
        }
        Ok(())
    }

    fn window(&mut self, id: SessionId) -> Result<()> {
        let w = self.duplex.window_frames;
        let live = self.live(id)?.clone();
        if w == 0 || live.kept < live.prompt_len + w {
            return Ok(());
        }
        let page = crate::codegen::PAGE;
        let sink_pages = live.prompt_len.div_ceil(page);
        let drop_from = sink_pages * page;
        ensure!(live.kept >= drop_from + page, "the window of {w} frames is shorter than one KV page past the prompt");
        let mut blocks = self.rt.block_table_of(live.slot);
        let used = live.kept.div_ceil(page);
        let freed = blocks.remove(sink_pages);
        blocks.insert(used - 1, freed);
        self.rt.set_block_table(live.slot, &blocks)?;
        let l = self.live_mut(id)?;
        l.kept -= page;
        let delta = (l.prompt_len as i64 + l.frame as i64 - 1) - l.kept as i64;
        let slot = l.slot;
        self.rt.set_position_offset(slot, delta as i32)?;
        Ok(())
    }

    pub fn set_feedback(&mut self, id: SessionId, token: u32, latent: &[f32]) -> Result<()> {
        ensure!(latent.len() == self.duplex.latent_dim, "a latent of {} values", latent.len());
        let l = self.live_mut(id)?;
        l.token = token;
        l.latent = latent.to_vec();
        Ok(())
    }

    pub fn say(&mut self, id: SessionId, text: &str) -> Result<usize> {
        let ids = self.tokenize(text)?;
        let n = ids.len();
        self.live_mut(id)?.text.say(ids);
        Ok(n)
    }

    pub fn say_tokens(&mut self, id: SessionId, tokens: Vec<u32>) -> Result<()> {
        self.live_mut(id)?.text.say(tokens);
        Ok(())
    }

    pub fn set_mode(&mut self, id: SessionId, mode: Mode) -> Result<()> {
        self.live_mut(id)?.text.mode = mode;
        Ok(())
    }

    pub fn interrupt(&mut self, id: SessionId) -> Result<(Interrupted, String)> {
        let r = self.live_mut(id)?.text.interrupt();
        let said = self.detokenize(&r.said);
        Ok((r, said))
    }

    pub fn end(&mut self, id: SessionId) -> Result<()> {
        let l = self.sessions.remove(&id).with_context(|| format!("no duplex session {id}"))?;
        self.free.push(l.slot);
        Ok(())
    }

    pub fn ids(&self) -> Vec<SessionId> {
        self.sessions.keys().copied().collect()
    }

    pub fn slot_codec(&self, slot: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (e, d) = (self.pools.enc_floats, self.pools.dec_floats);
        Ok((self.stream.clone_dtoh(&self.pools.enc.slice(slot * e..(slot + 1) * e))?, self.stream.clone_dtoh(&self.pools.dec.slice(slot * d..(slot + 1) * d))?))
    }

    pub fn set_slot_codec(&mut self, slot: usize, enc: &[f32], dec: &[f32]) -> Result<()> {
        let (e, d) = (self.pools.enc_floats, self.pools.dec_floats);
        ensure!(enc.len() == e && dec.len() == d, "codec state of {} and {} floats, expected {e} and {d}", enc.len(), dec.len());
        self.stream.memcpy_htod(enc, &mut self.pools.enc.slice_mut(slot * e..(slot + 1) * e))?;
        self.stream.memcpy_htod(dec, &mut self.pools.dec.slice_mut(slot * d..(slot + 1) * d))?;
        Ok(())
    }

    pub(super) fn take_slot(&mut self) -> Result<usize> {
        self.free.pop().context("every duplex slot is in use")
    }

    pub(super) fn give_slot(&mut self, slot: usize) {
        self.free.push(slot);
    }

    pub(super) fn insert(&mut self, live: Live) -> SessionId {
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(id, live);
        id
    }

    pub fn opts(&self) -> &EngineOptions {
        &self.opts
    }

    pub fn bail_unknown(&self, id: SessionId) -> Result<()> {
        if !self.sessions.contains_key(&id) {
            bail!("no duplex session {id}");
        }
        Ok(())
    }
}
