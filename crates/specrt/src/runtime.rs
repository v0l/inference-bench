use crate::codegen::{Module, PAGE};
use kir::linear::LinearTuning;
use crate::jit::{Jit, default_cache_dir};
use crate::spec::{LinearAttention, Spec};
use crate::statepool::StatePool;
use crate::weights::{Checkpoint, Feed, Matrix, Mixer, peak_rss_bytes};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, CudaView, LaunchConfig,
};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::time::Instant;

mod exec;
mod fused;
mod moe;
mod mtp;
pub(crate) mod snapshot;
mod stage;
mod step;
mod stream;
pub use fused::PrefillPart;
use exec::Plans;
pub use mtp::{SpecRow, SpecStats};
pub use stage::{StageRange, StageRole};
pub use step::{DecodeRow, GRAPH_SIZES, Mask, Stepper, pad_rows};

struct Plan {
    groups: Vec<Vec<String>>,
    work: Vec<Work>,
}

fn weight_plan(ck: &Checkpoint, spec: &Spec, range: StageRange, role: StageRole) -> Plan {
    let root = ck.root();
    let embed = format!("{root}.embed_tokens.weight");
    let tied = spec.dims.tie_embeddings;
    let mut head = Vec::new();
    if role.last {
        head.push(format!("{root}.norm.weight"));
        head.push(if tied { embed.clone() } else { "lm_head.weight".to_string() });
    }
    let mut plan = Plan { groups: Vec::new(), work: Vec::new() };
    let mut push = |names: Vec<String>, work: Work| {
        plan.groups.push(names);
        plan.work.push(work);
    };
    if role.last {
        push(ck.tensor_names(&head), Work::Head);
    }
    let layers: Vec<usize> = (0..spec.layers.count).collect();
    let half = layers.len() / 2;
    for &i in &layers[..half] {
        push(ck.names_under(&format!("{root}.layers.{}.", range.start + i)), Work::Layer(i));
    }
    if role.first && !(tied && role.last) {
        push(ck.tensor_names(&[embed]), Work::Embed);
    }
    if spec.speculative().is_some() {
        let lay = spec.mtp_layout();
        let mut names = ck.names_under(&format!("{}.", lay.layer));
        names.extend(ck.tensor_names(&[lay.fc, lay.norm_embed, lay.norm_hidden, lay.norm]));
        names.extend(lay.head.iter().filter(|h| ck.has(h)).cloned());
        push(names, Work::Mtp);
    }
    for &i in &layers[half..] {
        push(ck.names_under(&format!("{root}.layers.{}.", range.start + i)), Work::Layer(i));
    }
    plan
}

#[derive(Clone, Copy)]
enum Work {
    Head,
    Embed,
    Mtp,
    Layer(usize),
}

enum Uploaded {
    Head { norm: CudaSlice<f32>, lm_head: DeviceMatrix, embed: Option<CudaSlice<u16>>, draft: Option<DeviceMatrix> },
    Embed(CudaSlice<u16>),
    Mtp(Option<mtp::DeviceMtp>),
    Layer(usize, DeviceLayer),
}

struct Weights {
    layers: Vec<DeviceLayer>,
    embed: CudaSlice<u16>,
    norm: CudaSlice<f32>,
    lm_head: DeviceMatrix,
    mtp: Option<mtp::DeviceMtp>,
    direct: bool,
}

fn expert_cache_mode(spec: &Spec) -> bool {
    spec.runtime.offload.enabled() && spec.moe.is_some()
}

fn upload_layer(stream: &Arc<CudaStream>, ck: &Checkpoint, spec: &Spec, store: &moe::experts::ExpertStore, i: usize, global: usize) -> Result<DeviceLayer> {
    let l = ck.layer_at(spec, i, global).with_context(|| format!("layer {global}"))?;
    let cached = match &spec.moe {
        Some(m) if expert_cache_mode(spec) && l.moe.is_some() => Some((store, m.local_experts() + m.shared_slots())),
        _ => None,
    };
    let up_m = |m: &Matrix| DeviceMatrix::upload(stream, m);
    let up_opt = |v: &Option<Vec<f32>>| -> Result<Option<CudaSlice<f32>>> {
        Ok(match v {
            Some(v) => Some(stream.clone_htod(v)?),
            None => None,
        })
    };
    let before = |linear: bool| (0..i).filter(|&j| spec.block(j).is_linear() == linear).count() as i32;
    let mixer = match &l.mixer {
        Mixer::None => DeviceMixer::None,
        Mixer::Attention(a) => DeviceMixer::Attention(DeviceAttn {
            qkv: up_m(&a.qkv)?,
            qkv_bias: up_opt(&a.qkv_bias)?,
            o_bias: up_opt(&a.o_bias)?,
            q_norm: up_opt(&a.q_norm)?,
            k_norm: up_opt(&a.k_norm)?,
            o: up_m(&a.o)?,
            kv: spec.attention_before(i) as i32,
            window: if spec.block(i).windowed() { spec.layers.window.unwrap_or(0) as i32 } else { 0 },
            nope: spec.nope(i),
            extra: a.extra.iter().map(|(n, v)| Ok((n.clone(), stream.clone_htod(v)?))).collect::<Result<_>>()?,
        }),
        Mixer::Linear(g) => DeviceMixer::Linear(DeviceGdn {
            in_proj: up_m(&g.in_proj)?,
            ba: up_m(&g.ba)?,
            conv: stream.clone_htod(&g.conv)?,
            neg_a: stream.clone_htod(&g.neg_a)?,
            dt_bias: stream.clone_htod(&g.dt_bias)?,
            norm: stream.clone_htod(&g.norm)?,
            out: up_m(&g.out)?,
            slot: before(true),
        }),
        Mixer::Described(d) => DeviceMixer::Described(DeviceDescribed {
            in_proj: up_m(&d.in_proj)?,
            out: up_m(&d.out)?,
            extra: d.extra.iter().map(|(n, v)| Ok((n.clone(), stream.clone_htod(v)?))).collect::<Result<_>>()?,
            slot: before(true),
        }),
    };
    let layer = DeviceLayer { ln1: stream.clone_htod(&l.ln1)?, ln2: stream.clone_htod(&l.ln2)?, ln_pre_mlp: up_opt(&l.ln_pre_mlp)?, ln_post_mlp: up_opt(&l.ln_post_mlp)?, mixer, gate_up: up_m(&l.gate_up)?, down: up_m(&l.down)?, moe: l.moe.as_ref().map(|m| moe::DeviceMoe::upload(stream, m, i, cached.as_ref().map(|(p, n)| (*p, *n)))).transpose()? };
    copied(stream)?;
    Ok(layer)
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        if let Some(c) = self.expert_cache.take() {
            c.detach(&self.stream);
        }
        if let Some(s) = self.streamer.take() {
            s.detach(&mut self.layers, &self.stream);
        }
    }
}

fn offload_layers(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, spec: &Spec, layers: &mut [DeviceLayer], weight_bytes: u64) -> Result<(Option<stream::Streamer>, Option<crate::offload::OffloadPlan>)> {
    use crate::offload::{Tier, plan_layers};
    let off = &spec.runtime.offload;
    if !off.enabled() {
        return Ok((None, None));
    }
    let sizes: Vec<u64> = layers.iter().map(DeviceLayer::bytes).collect();
    let always = weight_bytes - sizes.iter().sum::<u64>();
    let mut settings = off.clone();
    if settings.host_budget == 0 {
        settings.host_budget = u64::MAX;
    }
    let plan = plan_layers(&sizes, always, &settings)?;
    let dir = match (&off.nvme_path, plan.nvme_bytes) {
        (_, 0) => None,
        (Some(p), _) => Some(std::path::PathBuf::from(p)),
        (None, n) => anyhow::bail!("[runtime.offload] {n} bytes of layers fit neither vram_budget nor host_budget and nvme_path is not set"),
    };
    let pool = crate::hostpool::HostPool::new(ctx.clone());
    let mut spilled = Vec::new();
    let mut writer = None;
    let mut disk_path = None;
    if let Some(dir) = &dir {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("layers-{}.pages", std::process::id()));
        disk_path = Some(path.clone());
        writer = Some(crate::diskio::Appender::create(path)?);
    }
    for l in plan.layers.iter().filter(|l| l.tier != Tier::Vram) {
        match (&mut writer, l.tier) {
            (Some(w), Tier::Nvme) => spilled.push(stream::spill_to_disk(stream, &pool, w, l.layer, &mut layers[l.layer])?),
            _ => spilled.push(stream::spill(stream, &pool, l.layer, &mut layers[l.layer])?),
        }
    }
    if spilled.is_empty() {
        return Ok((None, Some(plan)));
    }
    let disk = match (writer, disk_path) {
        (Some(w), Some(path)) => {
            w.finish()?;
            let file: Arc<dyn crate::diskio::ReadAt> = Arc::new(crate::diskio::DirectFile::open(&path)?);
            Some(stream::DiskTier::new(file, path))
        }
        _ => None,
    };
    let streamer = stream::Streamer::new(stream, layers, spilled, off.prefetch_depth, disk, &pool)?;
    ctx.synchronize()?;
    unsafe {
        let pool = cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device())?;
        cudarc::driver::result::mem_pool::trim_to(pool, 0)?;
    }
    Ok((Some(streamer), Some(plan)))
}

fn copied(stream: &Arc<CudaStream>) -> Result<()> {
    let done = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
    done.synchronize()?;
    Ok(())
}

fn upload_head(stream: &Arc<CudaStream>, ck: &Checkpoint, spec: &Spec, role: StageRole) -> Result<Uploaded> {
    let (norm, head) = ck.head(spec)?;
    let embed = if spec.dims.tie_embeddings && role.first { Some(stream.clone_htod(ck.embedding(spec)?.as_ref())?) } else { None };
    let up = Uploaded::Head {
        norm: stream.clone_htod(&norm)?,
        lm_head: DeviceMatrix::upload(stream, &head)?,
        embed,
        draft: match spec.draft_head_quant() {
            Some(q) => {
                let on_device = match q {
                    crate::spec::Quant::Nvfp4G16 => ck.draft_head_bf16(spec)?,
                    _ => None,
                };
                match on_device {
                    Some((w, cols)) => {
                        let cache = spec.runtime.cache_dir.clone().unwrap_or_else(default_cache_dir);
                        let d = crate::nvfp4::quantize_rows_on_device(stream, &cache, &w, cols, spec.head_rows())?;
                        Some(DeviceMatrix::Fp4 { packed: d.packed, scales: d.scales, act: false })
                    }
                    None => Some(DeviceMatrix::upload(stream, &ck.draft_head(spec, q)?)?),
                }
            }
            None => None,
        },
    };
    copied(stream)?;
    Ok(up)
}

fn upload_group(stream: &Arc<CudaStream>, ck: &Checkpoint, spec: &Spec, store: &moe::experts::ExpertStore, range: StageRange, role: StageRole, work: Work) -> Result<Uploaded> {
    match work {
        Work::Head => upload_head(stream, ck, spec, role),
        Work::Embed => {
            let e = stream.clone_htod(ck.embedding(spec)?.as_ref())?;
            copied(stream)?;
            Ok(Uploaded::Embed(e))
        }
        Work::Mtp => {
            let m = mtp::DeviceMtp::load(stream, ck, spec, false)?;
            copied(stream)?;
            Ok(Uploaded::Mtp(m))
        }
        Work::Layer(i) => Ok(Uploaded::Layer(i, upload_layer(stream, ck, spec, store, i, range.start + i)?)),
    }
}

fn load_weights(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, ck: &Checkpoint, spec: &Spec, range: StageRange, role: StageRole) -> Result<Weights> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    let opts = &spec.runtime.loader;
    let Plan { groups, work } = weight_plan(ck, spec, range, role);
    let count = groups.len();
    let names = groups.clone();
    let budget = crate::loader::Budget::new(opts.host_budget_bytes());
    let store = &moe::experts::ExpertStore::new(ctx.clone(), spec.runtime.offload.host_budget, spec.runtime.offload.nvme_path.as_deref());
    let feed = Mutex::new(Feed::new(ck, groups, opts, ctx, budget.clone())?);
    let direct = feed.lock().unwrap().direct();
    let reader = feed.lock().unwrap().reader();
    let next = Mutex::new(0usize);
    let failed = AtomicBool::new(false);
    let results: Mutex<Vec<Uploaded>> = Mutex::new(Vec::with_capacity(count));
    let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
    let sink = crate::progress::current();
    let workers = opts.workers.min(count).max(1);
    let streams = (0..workers).map(|_| ctx.new_stream()).collect::<Result<Vec<_>, _>>()?;
    std::thread::scope(|s| {
        for ws in &streams {
            let (feed, next, failed, results, errors, sink, names, work, budget) = (&feed, &next, &failed, &results, &errors, sink.clone(), &names, &work, &budget);
            s.spawn(move || {
                crate::progress::with_sink(sink, || {
                    let _ = ctx.bind_to_thread();
                    while !failed.load(Ordering::Relaxed) {
                        let claimed = {
                            let mut n = next.lock().unwrap();
                            if *n >= count {
                                break;
                            }
                            let g = *n;
                            *n += 1;
                            g
                        };
                        let reader = feed.lock().unwrap().reader();
                        let (claimed, lease) = match reader {
                            Some(r) => (r.take(claimed).map(|b| (claimed, Some(b))), None),
                            None => (Ok((claimed, None)), Some(budget.acquire(feed.lock().unwrap().size(claimed)))),
                        };
                        let outcome = claimed.and_then(|(g, batch)| {
                            let view = ck.with_staged(batch);
                            let up = upload_group(ws, &view, spec, store, range, role, work[g])?;
                            if !view.has_staged() {
                                view.release_names(&names[g]);
                            }
                            drop(view);
                            Ok((g, up))
                        });
                        drop(lease);
                        match outcome {
                            Ok((g, up)) => {
                                results.lock().unwrap().push(up);
                                feed.lock().unwrap().finished(g);
                            }
                            Err(e) => {
                                failed.store(true, Ordering::Relaxed);
                                errors.lock().unwrap().push(e);
                            }
                        }
                    }
                })
            });
        }
    });
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    drop(feed);
    drop(reader);
    ck.release();
    for ws in &streams {
        ws.synchronize()?;
    }
    stream.synchronize()?;
    let mut layers: Vec<Option<DeviceLayer>> = (0..spec.layers.count).map(|_| None).collect();
    let (mut head, mut embed, mut mtp, mut draft) = (None, None, None, None);
    for up in results.into_inner().unwrap() {
        match up {
            Uploaded::Layer(i, l) => layers[i] = Some(l),
            Uploaded::Head { norm, lm_head, embed: tied, draft: d } => {
                head = Some((norm, lm_head));
                embed = embed.or(tied);
                draft = d;
            }
            Uploaded::Embed(e) => embed = Some(e),
            Uploaded::Mtp(m) => mtp = m,
        }
    }
    if let Some(m) = mtp.as_mut() {
        m.set_head(draft);
    }
    let embed = match embed {
        Some(e) => e,
        None if role.first => anyhow::bail!("the embedding was not loaded"),
        None => stream.alloc_zeros(1)?,
    };
    let (norm, lm_head) = match head {
        Some(h) => h,
        None if role.last => anyhow::bail!("the head was not loaded"),
        None => (stream.alloc_zeros(1)?, DeviceMatrix::Bf16(stream.alloc_zeros(1)?)),
    };
    let layers = layers.into_iter().enumerate().map(|(i, l)| l.with_context(|| format!("layer {} was not loaded", range.start + i))).collect::<Result<Vec<_>>>()?;
    Ok(Weights { layers, embed, norm, lm_head, mtp, direct })
}

enum DeviceMatrix {
    Bf16(CudaSlice<u16>),
    Q8 { values: CudaSlice<i8>, scales: CudaSlice<f32> },
    Fp8 { values: CudaSlice<u8>, scales: CudaSlice<f32> },
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8>, act: bool },
    Ggml { data: CudaSlice<u8>, table: CudaSlice<u32> },
}

impl DeviceMatrix {
    fn upload(stream: &Arc<CudaStream>, m: &Matrix) -> Result<Self> {
        Ok(match m {
            Matrix::Bf16(w) => DeviceMatrix::Bf16(stream.clone_htod(w.as_ref())?),
            Matrix::Q8 { values, scales } => {
                DeviceMatrix::Q8 { values: stream.clone_htod(values)?, scales: stream.clone_htod(scales)? }
            }
            Matrix::Fp8 { values, scales } => {
                DeviceMatrix::Fp8 { values: stream.clone_htod(values.as_ref())?, scales: stream.clone_htod(scales)? }
            }
            Matrix::Ggml(m) => DeviceMatrix::Ggml { data: stream.clone_htod(m.data().as_ref())?, table: stream.clone_htod(&m.rows)? },
            Matrix::Fp4(m) => DeviceMatrix::Fp4 {
                packed: stream.clone_htod(m.packed.as_ref())?,
                scales: stream.clone_htod(&m.device_scales())?,
                act: m.input_global > 0.0,
            },
        })
    }

    fn bytes(&self) -> u64 {
        (match self {
            DeviceMatrix::Bf16(w) => w.num_bytes(),
            DeviceMatrix::Q8 { values, scales } => values.num_bytes() + scales.num_bytes(),
            DeviceMatrix::Fp8 { values, scales } => values.num_bytes() + scales.num_bytes(),
            DeviceMatrix::Fp4 { packed, scales, .. } => packed.num_bytes() + scales.num_bytes(),
            DeviceMatrix::Ggml { data, table } => data.num_bytes() + table.num_bytes(),
        }) as u64
    }

    fn kind(&self) -> crate::codegen::WeightKind {
        use crate::codegen::WeightKind;
        match self {
            DeviceMatrix::Bf16(_) => WeightKind::Bf16,
            DeviceMatrix::Q8 { .. } => WeightKind::Q8,
            DeviceMatrix::Fp8 { .. } => WeightKind::Fp8,
            DeviceMatrix::Fp4 { .. } => WeightKind::Fp4,
            DeviceMatrix::Ggml { .. } => WeightKind::Ggml,
        }
    }

    fn args<'a>(&'a self, b: &mut crate::launch::Launch<'a>, dummy: &'a CudaSlice<f32>) {
        match self {
            DeviceMatrix::Bf16(w) => b.arg(w).arg(dummy),
            DeviceMatrix::Q8 { values, scales } => b.arg(values).arg(scales),
            DeviceMatrix::Fp8 { values, scales } => b.arg(values).arg(scales),
            DeviceMatrix::Fp4 { packed, scales, .. } => b.arg(packed).arg(scales),
            DeviceMatrix::Ggml { data, table } => b.arg(data).arg(table),
        };
    }
}

struct DescribedLaunch {
    name: &'static str,
    threads: u32,
    params: Vec<kir::Param>,
    chunk: Option<kir::recur::ChunkForm>,
}

struct DeviceAttn {
    qkv: DeviceMatrix,
    qkv_bias: Option<CudaSlice<f32>>,
    o_bias: Option<CudaSlice<f32>>,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    o: DeviceMatrix,
    kv: i32,
    window: i32,
    nope: bool,
    extra: std::collections::BTreeMap<String, CudaSlice<f32>>,
}

struct DeviceGdn {
    in_proj: DeviceMatrix,
    ba: DeviceMatrix,
    conv: CudaSlice<f32>,
    neg_a: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    out: DeviceMatrix,
    slot: i32,
}

enum DeviceMixer {
    Attention(DeviceAttn),
    Linear(DeviceGdn),
    Described(DeviceDescribed),
    None,
}

struct DeviceDescribed {
    in_proj: DeviceMatrix,
    out: DeviceMatrix,
    extra: std::collections::BTreeMap<String, CudaSlice<f32>>,
    slot: i32,
}

struct SeqState {
    name: String,
    pool: StatePool,
    slice: ManuallyDrop<CudaSlice<f32>>,
}

struct MixBuffers {
    states: Vec<SeqState>,
    mix: CudaSlice<f32>,
    planes: Planes,
    scratch: std::collections::BTreeMap<String, CudaSlice<f32>>,
    chunk_state: CudaSlice<f32>,
    chunk_decay: CudaSlice<f32>,
}

impl MixBuffers {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, spec: &Spec, slots: usize, chunk: usize, (tiles, tile_floats): (usize, usize)) -> Result<Self> {
        let mut states = Vec::new();
        for s in crate::state::sequence(spec).filter(|s| s.layers == crate::state::Layers::Linear) {
            ensure!(s.dtype == crate::state::Dtype::F32, "[[state]] {} on a mixer is not f32", s.name);
            let pool = StatePool::new(ctx, s.sequence_bytes(spec), slots)?;
            let slice = ManuallyDrop::new(pool.slice(stream));
            states.push(SeqState { name: s.name.clone(), pool, slice });
        }
        let m = spec.mixer.as_ref().context("no [mixer]")?;
        let scratch = m.scratch.iter().map(|(n, w)| Ok((n.clone(), stream.alloc_zeros(chunk * crate::state::dim(spec, w)?)?))).collect::<Result<_>>()?;
        Ok(MixBuffers {
            states,
            mix: stream.alloc_zeros(chunk * spec.mix_in_rows())?,
            planes: Planes::new(stream, chunk * spec.mix_out_cols())?,
            scratch,
            chunk_state: stream.alloc_zeros(tile_floats.max(1))?,
            chunk_decay: stream.alloc_zeros(tiles.max(1))?,
        })
    }
}

impl Drop for MixBuffers {
    fn drop(&mut self) {
        for s in &mut self.states {
            unsafe { ManuallyDrop::take(&mut s.slice) }.leak();
        }
    }
}

struct DeviceLayer {
    ln1: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    ln_pre_mlp: Option<CudaSlice<f32>>,
    ln_post_mlp: Option<CudaSlice<f32>>,
    mixer: DeviceMixer,
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
    moe: Option<moe::DeviceMoe>,
}

fn opt_bytes(v: &Option<CudaSlice<f32>>) -> u64 {
    v.as_ref().map_or(0, |s| s.num_bytes() as u64)
}

impl DeviceLayer {
    fn bytes(&self) -> u64 {
        let mixer = match &self.mixer {
            DeviceMixer::Attention(a) => {
                a.qkv.bytes()
                    + a.o.bytes()
                    + opt_bytes(&a.qkv_bias)
                    + opt_bytes(&a.o_bias)
                    + opt_bytes(&a.q_norm)
                    + opt_bytes(&a.k_norm)
            }
            DeviceMixer::Linear(g) => {
                g.in_proj.bytes()
                    + g.ba.bytes()
                    + g.out.bytes()
                    + (g.conv.num_bytes() + g.neg_a.num_bytes() + g.dt_bias.num_bytes() + g.norm.num_bytes()) as u64
            }
            DeviceMixer::Described(d) => d.in_proj.bytes() + d.out.bytes() + d.extra.values().map(|v| v.num_bytes() as u64).sum::<u64>(),
            DeviceMixer::None => 0,
        };
        mixer
            + self.gate_up.bytes()
            + self.down.bytes()
            + self.moe.as_ref().map_or(0, moe::DeviceMoe::bytes)
            + (self.ln1.num_bytes() + self.ln2.num_bytes()) as u64
    }
}

struct GdnBuffers {
    la: LinearAttention,
    conv_pool: StatePool,
    ssm_pool: StatePool,
    conv_state: ManuallyDrop<CudaSlice<f32>>,
    ssm_state: ManuallyDrop<CudaSlice<f32>>,
    lin: CudaSlice<f32>,
    ba: CudaSlice<f32>,
    mix: CudaSlice<f32>,
    out: CudaSlice<f32>,
    pre_lin: CudaSlice<f32>,
    pre_ba: CudaSlice<f32>,
    pre_mix: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    g: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    w: CudaSlice<f32>,
    u: CudaSlice<f32>,
    aqk: CudaSlice<f32>,
    gc: CudaSlice<f32>,
    qe: CudaSlice<f32>,
    kt: CudaSlice<f32>,
    o: CudaSlice<f32>,
    planes: Planes,
}

impl GdnBuffers {
    fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        la: &LinearAttention,
        layers: usize,
        slots: usize,
        chunk: usize,
        tuning: LinearTuning,
    ) -> Result<Self> {
        let d = la.key_head_dim;
        let la_chunk = tuning.chunk;
        let conv_pool = StatePool::new(ctx, layers * la.conv_dim() * (la.conv_kernel - 1) * 4, slots)?;
        let ssm_pool = StatePool::new(ctx, layers * la.value_heads * d * d * 4, slots)?;
        let conv_state = ManuallyDrop::new(conv_pool.slice(stream));
        let ssm_state = ManuallyDrop::new(ssm_pool.slice(stream));
        let chunks = chunk.div_ceil(la_chunk);
        let padded = chunks * la_chunk;
        let zeros = |n: usize| stream.alloc_zeros::<f32>(n);
        Ok(GdnBuffers {
            la: la.clone(),
            conv_pool,
            ssm_pool,
            conv_state,
            ssm_state,
            lin: zeros(la.in_proj_rows())?,
            ba: zeros(la.ba_rows())?,
            mix: zeros(la.conv_dim())?,
            out: zeros(la.value_dim())?,
            pre_lin: zeros(chunk * la.in_proj_rows())?,
            pre_ba: zeros(chunk * la.ba_rows())?,
            pre_mix: zeros(chunk * la.conv_dim())?,
            qn: zeros(chunk * la.key_heads * d)?,
            kn: zeros(chunk * la.key_heads * d)?,
            g: zeros(chunk * la.value_heads)?,
            beta: zeros(chunk * la.value_heads)?,
            w: zeros(padded * la.value_heads * d)?,
            u: zeros(padded * la.value_heads * d)?,
            aqk: zeros(chunks * la.value_heads * la_chunk * la_chunk)?,
            gc: zeros(chunks * la.value_heads * la_chunk)?,
            qe: zeros(padded * la.value_heads * d)?,
            kt: zeros(padded * la.value_heads * d)?,
            o: zeros(chunk * la.value_dim())?,
            planes: Planes::new(stream, chunk * la.value_dim())?,
        })
    }
}

impl Drop for GdnBuffers {
    fn drop(&mut self) {
        let conv = unsafe { ManuallyDrop::take(&mut self.conv_state) };
        let ssm = unsafe { ManuallyDrop::take(&mut self.ssm_state) };
        conv.leak();
        ssm.leak();
    }
}

pub struct Trace {
    pub layers: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct LinearStates {
    pub conv: Vec<f32>,
    pub state: Vec<f32>,
}

struct Buffers {
    x: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    attn_part: CudaSlice<f32>,
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    caches: Vec<CudaSlice<u16>>,
    block_table: CudaSlice<i32>,
    zero_slots: CudaSlice<i32>,
    token: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    history: CudaSlice<i32>,
    count: CudaSlice<i32>,
    dummy: CudaSlice<f32>,
}

impl Buffers {
    fn k_cache(&self) -> &CudaSlice<u16> {
        &self.caches[0]
    }

    fn v_cache(&self) -> &CudaSlice<u16> {
        self.caches.get(1).unwrap_or(&self.caches[0])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub seed: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub logprobs: Option<u32>,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, seed: 0, top_k: 0, top_p: 1.0, min_p: 0.0, logprobs: None };

    pub fn new(temperature: f32, seed: u32) -> Sampling {
        Sampling { temperature, seed, ..Sampling::GREEDY }
    }

    pub fn truncated(self, top_k: u32, top_p: f32, min_p: f32) -> Sampling {
        Sampling { top_k, top_p, min_p, ..self }
    }

    pub fn with_logprobs(self, top: Option<u32>) -> Sampling {
        Sampling { logprobs: top, ..self }
    }

    pub fn inverse_temperature(&self) -> f32 {
        if self.temperature > 0.0 { 1.0 / self.temperature } else { 0.0 }
    }
}

struct Planes {
    hi: CudaSlice<u16>,
    lo: CudaSlice<u16>,
}

impl Planes {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        Ok(Planes { hi: stream.alloc_zeros(len)?, lo: stream.alloc_zeros(len)? })
    }

    fn rows(&self, from: usize, to: usize, width: usize) -> (CudaView<'_, u16>, CudaView<'_, u16>) {
        (self.hi.slice(from * width..to * width), self.lo.slice(from * width..to * width))
    }

    fn all(&self) -> (CudaView<'_, u16>, CudaView<'_, u16>) {
        (self.hi.as_view(), self.lo.as_view())
    }
}

enum GemmOut<'a> {
    F32(&'a CudaSlice<f32>),
    Planes(&'a Planes),
}

struct Fp8Act {
    q: CudaSlice<u8>,
    scale: CudaSlice<f32>,
    per_token: bool,
}

struct Fp4Act {
    q: CudaSlice<u8>,
    scales: CudaSlice<u8>,
}

struct PrefillBuffers {
    tokens: CudaSlice<i32>,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    branch: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    xn_planes: Planes,
    attn_planes: Planes,
    act_planes: Planes,
    dummy_u16: CudaSlice<u16>,
    dummy_u8: CudaSlice<u8>,
}

pub struct BatchGeneration {
    pub tokens: Vec<Vec<u32>>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

struct BatchBuffers {
    tokens: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    history: CudaSlice<i32>,
    part_acc: CudaSlice<f32>,
    part_ml: CudaSlice<f32>,
    pos: CudaSlice<i32>,
    slots: CudaSlice<i32>,
}

pub struct Generation {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub peak_device_bytes: u64,
}

pub struct Runtime {
    spec: Spec,
    max_ctx: usize,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    smem: std::collections::BTreeMap<&'static str, u32>,
    linear_tuning: LinearTuning,
    embed: CudaSlice<u16>,
    norm: CudaSlice<f32>,
    lm_head: DeviceMatrix,
    layers: Vec<DeviceLayer>,
    described: Vec<Option<DescribedLaunch>>,
    scratch: std::collections::BTreeMap<String, CudaSlice<f32>>,
    int_scratch: std::collections::BTreeMap<String, CudaSlice<i32>>,
    buf: Buffers,
    pre: PrefillBuffers,
    gdn: Option<GdnBuffers>,
    mixer: Option<MixBuffers>,
    mixer_launch: Vec<DescribedLaunch>,
    moe: Option<moe::MoeBuffers>,
    fp8_act: Option<Fp8Act>,
    fp4_act: Option<Fp4Act>,
    f4b_part: CudaSlice<f32>,
    f8b_part: CudaSlice<f32>,
    ggb_scratch: CudaSlice<u16>,
    chunk: usize,
    prefill_limit: usize,
    use_mma: bool,
    plans: Plans,
    role: StageRole,
    pdl: bool,
    sms: u32,
    target: kir::Target,
    seq: i32,
    batch: usize,
    slots: usize,
    pages: usize,
    kv_blocks: usize,
    kv_planes: usize,
    tables: Vec<i32>,
    pub cache_hit: bool,
    pub jit_key: String,
    pub jit_compile_secs: f64,
    pub jit_module_secs: f64,
    pub arch: String,
    pub device_bytes: u64,
    pub weight_bytes: u64,
    pub weight_secs: f64,
    pub load_secs: f64,
    pub peak_host_rss_bytes: u64,
    ctx: Arc<CudaContext>,
    baseline_free: usize,
    mtp: Option<mtp::MtpState>,
    snap: snapshot::SnapshotPool,
    host: crate::hostpool::HostPool,
    streamer: Option<stream::Streamer>,
    expert_cache: Option<moe::experts::ExpertCache>,
    offload_plan: Option<crate::offload::OffloadPlan>,
    mm_meta: CudaSlice<i32>,
    mm_table: CudaSlice<i32>,
    mm_bidi: CudaSlice<i32>,
    media: Vec<Option<MediaDevice>>,
    vision: Option<Arc<crate::media_encode::MediaEncoder>>,
    partial: CudaSlice<f32>,
    head_part: Option<CudaSlice<f32>>,
    comm: Option<crate::allreduce::Comm>,
}

const HEAD_PART_ROWS: usize = 64;

struct MediaDevice {
    embeds: CudaSlice<f32>,
    rows: Vec<i32>,
}

fn grid(n: usize, per_block: usize) -> u32 {
    n.div_ceil(per_block) as u32
}

fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

const LINEAR_ROWS_PER_BLOCK: usize = 8;
const FP4_GEMV_THREADS: u32 = crate::codegen::FP4_GEMV_THREADS;

fn gemv_cfg(name: &str, rows: usize, w: &DeviceMatrix, sms: u32) -> LaunchConfig {
    if name == "lm_head_norm_f4" {
        let per_block = FP4_GEMV_THREADS as usize / 32 * if rows.is_multiple_of(2) { 2 } else { 1 };
        cfg(grid(rows, per_block), FP4_GEMV_THREADS)
    } else if name.ends_with("_f4") {
        cfg(grid(rows, FP4_GEMV_THREADS as usize / 32), FP4_GEMV_THREADS)
    } else {
        let format = w.kind().format();
        let pair = format.gemv_rows();
        let per_warp = if name.starts_with("gate_up_act") || !rows.is_multiple_of(pair) { 1 } else { pair };
        let blocks = grid(rows, LINEAR_ROWS_PER_BLOCK * per_warp);
        let cap = if format == crate::codegen::WeightFormat::Ggml { sms * GEMV_BLOCKS_PER_SM } else { u32::MAX };
        cfg(blocks.min(cap), 256)
    }
}

const GEMV_BLOCKS_PER_SM: u32 = 8;
const PREFILL_CHUNK: usize = 2048;
const STATE_FREE_MARGIN: usize = 256 << 20;
const ATTN_SPLITS: usize = 64;
const ATTN_BLOCK: u32 = 128;
const GEMM_TILE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub max_ctx: usize,
    pub slots: usize,
    pub kv_blocks: usize,
    pub prefill_chunk: usize,
    pub kv_planes: usize,
    pub kv_headroom: usize,
    pub state_on_demand: bool,
}

impl Layout {
    pub fn contiguous(max_ctx: usize, slots: usize) -> Layout {
        Layout {
            max_ctx,
            slots,
            kv_blocks: slots * max_ctx.div_ceil(PAGE),
            prefill_chunk: PREFILL_CHUNK,
            kv_planes: 2,
            kv_headroom: 0,
            state_on_demand: false,
        }
    }

    pub fn serving(spec: &Spec, max_ctx: usize, slots: usize, kv_bytes: Option<usize>) -> Layout {
        Layout {
            max_ctx,
            slots,
            kv_blocks: kv_bytes.map_or(0, |b| (b / kv_block_bytes(spec, 1)).max(1)),
            prefill_chunk: spec.runtime.prefill_chunk.unwrap_or(PREFILL_CHUNK),
            kv_planes: 1,
            kv_headroom: (1 << 30) + slots * crate::serve::slot_state_bytes(spec) as usize,
            state_on_demand: true,
        }
    }

    pub fn pages(&self) -> usize {
        self.max_ctx.div_ceil(PAGE)
    }
}

pub fn kv_block_bytes(spec: &Spec, planes: usize) -> usize {
    crate::state::kv_block_bytes(spec, planes)
}

pub struct Warmup {
    pub key: String,
    pub cache_hit: bool,
    pub binary: bool,
    pub compile_secs: f64,
    pub module_secs: f64,
    pub cache_dir: std::path::PathBuf,
}

pub fn warmup_spec(spec: &Spec) -> Result<Spec> {
    let mut resolved = spec.clone();
    crate::engine::complete_spec(&mut resolved)?;
    crate::engine::apply_rope_extension(&resolved)
}

pub fn warmup(spec: &Spec, max_ctx: usize, slots: usize) -> Result<Warmup> {
    ensure!(slots >= 1, "need at least one slot");
    let resolved = warmup_spec(spec)?;
    let layout = Layout::serving(&resolved, max_ctx, slots, None);
    let layout = if resolved.runtime.kv_dtype.is_fp8() { Layout { kv_planes: 1, ..layout } } else { layout };
    let cache_dir = resolved.runtime.cache_dir.clone().unwrap_or_else(default_cache_dir);
    crate::hardware::ensure_nvrtc(&cache_dir)?;
    let ctx = CudaContext::new(0).context("cuda init")?;
    let target = crate::codegen::cuda_target(&ctx)?;
    let module = Module::generate_kv(&resolved, layout.max_ctx, layout.kv_planes, &target)?;
    let jit = Jit::new(ctx, cache_dir.clone())?.with_limits(resolved.runtime.cache_max_bytes.unwrap_or(crate::jit::DEFAULT_CACHE_MAX_BYTES), resolved.runtime.compile_jobs.unwrap_or(crate::jit::default_compile_jobs()));
    let loaded = jit.load(&module)?;
    Ok(Warmup { key: loaded.key, cache_hit: loaded.cache_hit, binary: loaded.binary, compile_secs: loaded.compile_secs, module_secs: loaded.module_secs, cache_dir })
}

impl Runtime {
    pub fn load(spec: &Spec, max_ctx: usize) -> Result<Self> {
        Self::load_batched(spec, max_ctx, 1)
    }

    pub fn load_batched(spec: &Spec, max_ctx: usize, batch: usize) -> Result<Self> {
        Self::load_batched_planes(spec, max_ctx, batch, 2)
    }

    pub fn load_batched_planes(spec: &Spec, max_ctx: usize, batch: usize, kv_planes: usize) -> Result<Self> {
        ensure!(batch >= 1, "batch must be at least 1");
        let mut rt = Self::load_layout(spec, Layout { kv_planes, ..Layout::contiguous(max_ctx, batch) })?;
        rt.batch = batch;
        let pages = rt.pages;
        for slot in 0..batch {
            let blocks: Vec<u32> = (0..pages as u32).rev().map(|p| (slot * pages) as u32 + p).collect();
            rt.set_block_table(slot, &blocks)?;
        }
        Ok(rt)
    }

    pub fn load_layout(spec: &Spec, layout: Layout) -> Result<Self> {
        Self::load_stage(spec, layout, StageRange::whole(spec))
    }

    pub fn load_stage(spec: &Spec, layout: Layout, range: StageRange) -> Result<Self> {
        Self::load_inner(spec, layout, range, None)
    }

    pub fn chunk_rows(layout: &Layout) -> usize {
        layout.max_ctx.min(layout.prefill_chunk).max(layout.slots) + crate::runtime::step::row_capacity(layout.slots)
    }

    pub fn load_rank(spec: &Spec, layout: Layout, device: usize, rank: usize, fabric: &crate::allreduce::Fabric) -> Result<Self> {
        let range = StageRange { device, ..StageRange::whole(spec) };
        Self::load_inner(spec, layout, range, Some((fabric.ranks(), rank, fabric)))
    }

    pub fn tp_rank(&self) -> Option<(usize, usize)> {
        self.comm.as_ref().map(|c| (c.ranks(), c.rank()))
    }

    pub fn head_rows(&self) -> std::ops::Range<usize> {
        self.spec.head_rows()
    }

    fn load_inner(spec: &Spec, layout: Layout, range: StageRange, tp: Option<(usize, usize, &crate::allreduce::Fabric)>) -> Result<Self> {
        let resolved = crate::engine::apply_rope_extension(spec)?;
        range.check(&resolved)?;
        let role = StageRole::of(&resolved, range);
        let full = if role.whole() { resolved.clone() } else { resolved.stage(range.start, range.end) };
        let staged = match tp {
            Some((n, rank, _)) => {
                ensure!(role.whole(), "a tensor parallel rank runs every layer");
                crate::tp::rank_spec_of(&full, n, rank)?
            }
            None => full.clone(),
        };
        let mut staged = staged;
        if staged.runtime.offload.enabled() {
            staged.runtime.graphs = false;
        }
        let spec = &staged;
        ensure!(layout.slots >= 1, "need at least one slot");
        let layout = if spec.runtime.kv_dtype.is_fp8() { Layout { kv_planes: 1, ..layout } } else { layout };
        let max_ctx = layout.max_ctx;
        let batch = 1;
        let load_started = Instant::now();
        crate::hardware::ensure_nvrtc(&spec.runtime.cache_dir.clone().unwrap_or_else(default_cache_dir))?;
        let ctx = CudaContext::new(range.device).context("cuda init")?;
        let free_before = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let cache_dir = spec.runtime.cache_dir.clone().unwrap_or_else(default_cache_dir);
        let jit = Jit::new(ctx.clone(), cache_dir.clone())?.with_limits(spec.runtime.cache_max_bytes.unwrap_or(crate::jit::DEFAULT_CACHE_MAX_BYTES), spec.runtime.compile_jobs.unwrap_or(crate::jit::default_compile_jobs()));
        let comm = match tp {
            Some((n, rank, fabric)) => {
                let gather = match &spec.dims.head_rows {
                    Some(_) => Some((spec.dims.vocab as u32, crate::tp::vocab_each(spec.dims.vocab, n)? as u32)),
                    None => None,
                };
                Some(fabric.comm(rank, &stream, cache_dir, Some((spec.dims.hidden as u32, spec.norm.eps as f32)), gather)?)
            }
            None => None,
        };
        let target = crate::codegen::cuda_target(&ctx)?;
        let module = Module::generate_kv(spec, max_ctx, layout.kv_planes, &target)?;
        crate::progress::report(crate::progress::Stage::Kernels, 0, 1);
        let jit_started = Instant::now();
        let loaded = jit.load(&module)?;
        let jit_secs = jit_started.elapsed().as_secs_f64();
        let mut kernels = HashMap::new();
        for &name in &module.kernels {
            let f = loaded.function(name)?;
            if let Some(&bytes) = module.smem.get(name) {
                use cudarc::driver::sys::CUfunction_attribute as A;
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)?;
                f.set_attribute(
                    A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .with_context(|| format!("{name}: {bytes} bytes of shared memory"))?;
            }
            kernels.insert(name, f);
        }
        let smem = module.smem.clone();
        let described: Vec<Option<DescribedLaunch>> = crate::codegen::described_kernels(spec, layout.kv_planes, max_ctx, &target)?
            .into_iter()
            .map(|k| k.map(|k| DescribedLaunch { name: k.name, threads: k.threads, chunk: None, params: k.params }))
            .collect();
        let linear_tuning = LinearTuning::for_target(&target);
        crate::progress::report(crate::progress::Stage::Kernels, 1, 1);
        let checkpoint = match tp {
            Some((n, rank, _)) => Checkpoint::for_repo(&spec.repo)?.renamed(&spec.tensors).for_rank(crate::tp::RankCut::new(&full, crate::graph::Graph::decoder(), n, rank)?)?,
            None => Checkpoint::for_repo(&spec.repo)?.renamed(&spec.tensors),
        };
        let weights_started = Instant::now();
        let Weights { mut layers, embed, norm, lm_head, mtp: mtp_weights, direct: weights_direct } =
            load_weights(&ctx, &stream, &checkpoint, spec, range, role)?;
        let weight_secs = weights_started.elapsed().as_secs_f64();
        let weight_bytes = layers.iter().map(DeviceLayer::bytes).sum::<u64>()
            + (embed.num_bytes() + norm.num_bytes()) as u64
            + lm_head.bytes()
            + mtp_weights.as_ref().map_or(0, mtp::DeviceMtp::bytes);
        let (streamer, offload_plan) = offload_layers(&ctx, &stream, spec, &mut layers, weight_bytes)?;
        let d = &spec.dims;
        let pages = layout.pages();
        let block_elems = kv_block_bytes(spec, layout.kv_planes) / 4;
        let prefill_limit = max_ctx.min(layout.prefill_chunk).max(layout.slots);
        let chunk = prefill_limit + crate::runtime::step::row_capacity(layout.slots);
        let use_mma = ctx.compute_capability()?.0 >= 8 && !spec.runtime.gemm_fp32;
        ensure!(described.iter().all(Option::is_none) || use_mma, "described attention needs the tensor core path (sm_80 or newer)");
        ensure!(
            use_mma || (spec.lm_head_quant() != crate::spec::Quant::Nvfp4G16 && spec.draft_head_quant() != Some(crate::spec::Quant::Nvfp4G16)),
            "head_quant = \"nvfp4_g16\" needs the tensor-core prefill path (an Ampere or newer GPU and [runtime] gemm_fp32 = false)"
        );
        ensure!(use_mma || tp.is_none(), "tensor parallel needs the tensor core path (an Ampere or newer GPU and [runtime] gemm_fp32 = false)");
        let mma_only = |n: usize| if use_mma { 1 } else { n };
        let buf = Buffers {
            x: stream.alloc_zeros(d.hidden)?,
            qkv: stream.alloc_zeros(spec.qkv_rows())?,
            attn: stream.alloc_zeros(spec.q_dim())?,
            attn_part: stream.alloc_zeros(spec.attention.heads * ATTN_SPLITS * (spec.attention.head_dim + 2))?,
            gate_up: stream.alloc_zeros(2 * d.intermediate)?,
            act: stream.alloc_zeros(d.intermediate)?,
            logits: stream.alloc_zeros(d.vocab)?,
            caches: vec![stream.alloc_zeros(1)?],
            block_table: stream.alloc_zeros((layout.slots + 1) * pages)?,
            zero_slots: stream.alloc_zeros(chunk)?,
            token: stream.alloc_zeros(1)?,
            pos: stream.alloc_zeros(1)?,
            history: stream.alloc_zeros(max_ctx)?,
            count: stream.alloc_zeros(1)?,
            dummy: stream.alloc_zeros(1)?,
        };
        let pre = PrefillBuffers {
            tokens: stream.alloc_zeros(chunk)?,
            x: stream.alloc_zeros(chunk * d.hidden)?,
            xn: stream.alloc_zeros(mma_only(chunk * d.hidden))?,
            branch: stream.alloc_zeros(if spec.uses_branch() { chunk * d.hidden } else { 1 })?,
            qkv: stream.alloc_zeros(chunk * spec.qkv_rows())?,
            attn: stream.alloc_zeros(mma_only(chunk * spec.q_dim().max(spec.o_dim())))?,
            gate_up: stream.alloc_zeros(mma_only(chunk * 2 * d.intermediate))?,
            act: stream.alloc_zeros(mma_only(chunk * d.intermediate))?,
            xn_planes: Planes::new(&stream, chunk * d.hidden)?,
            attn_planes: Planes::new(&stream, chunk * spec.q_dim().max(spec.o_dim()))?,
            act_planes: Planes::new(&stream, chunk * d.intermediate)?,
            dummy_u16: stream.alloc_zeros(8)?,
            dummy_u8: stream.alloc_zeros(8)?,
        };
        let gdn = match &spec.linear_attention {
            Some(la) => Some(GdnBuffers::new(&ctx, &stream, la, spec.linear_layers(), layout.slots + 1, chunk, linear_tuning)?),
            None => None,
        };
        ensure!(use_mma || gdn.is_none(), "linear attention needs the tensor core path (sm_80 or newer)");
        let mixer_launch: Vec<DescribedLaunch> = crate::codegen::mixer_kernels(spec, layout.kv_planes, max_ctx, &target)?
            .into_iter()
            .map(|k| DescribedLaunch { name: k.name, threads: k.threads, chunk: k.chunk_form().filter(|_| spec.runtime.chunked_scan), params: k.params })
            .collect();
        let chunk_tiles = match &spec.mixer {
            Some(m) => mixer_launch
                .iter()
                .zip(&m.node)
                .filter_map(|(l, n)| Some((l.chunk.as_ref()?, n)))
                .map(|(f, n)| Ok((crate::state::dim(spec, &n.grid[0])? * chunk.div_ceil(f.min_rows as usize), f.state_floats as usize)))
                .collect::<Result<Vec<(usize, usize)>>>()?
                .into_iter()
                .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.0 * b.1))),
            None => (0, 0),
        };
        let mixer = match &spec.mixer {
            Some(_) => Some(MixBuffers::new(&ctx, &stream, spec, layout.slots + 1, chunk, chunk_tiles)?),
            None => None,
        };
        ensure!(use_mma || mixer.is_none(), "a described mixer needs the tensor core path (sm_80 or newer)");
        ensure!(use_mma || (!spec.layers.split && spec.mlp.mapped.is_none()), "mlp-only layers and non-gated mlps need the tensor core path (sm_80 or newer)");
        let widest = [d.hidden, d.intermediate, spec.q_dim()]
            .into_iter()
            .chain(spec.linear_attention.as_ref().map(|l| l.value_dim()))
            .max()
            .unwrap_or(d.hidden);
        let fp8_act = if spec.quant.is_fp8() && crate::codegen::fp8_act_enabled(spec) {
            Some(Fp8Act {
                q: stream.alloc_zeros(chunk * widest)?,
                scale: stream.alloc_zeros(chunk * widest / crate::codegen::FP8_BLOCK)?,
                per_token: spec.quant == crate::spec::Quant::Fp8Channel,
            })
        } else {
            None
        };
        let fp4_act = if spec.mlp.quant == Some(crate::spec::Quant::Nvfp4G16) && crate::codegen::fp4_act_enabled(spec) {
            let widest = d.hidden.max(d.intermediate);
            Some(Fp4Act {
                q: stream.alloc_zeros(chunk * widest / 2)?,
                scales: stream.alloc_zeros(chunk * widest / crate::nvfp4::GROUP)?,
            })
        } else {
            None
        };
        let target = crate::codegen::cuda_target(&ctx)?;
        let sms = target.sm_count;
        let f4b_part = if spec.mlp.quant == Some(crate::spec::Quant::Nvfp4G16) || spec.serves_fp4_mlp() || !spec.requanted_linears().is_empty() || spec.draft_head_quant() == Some(crate::spec::Quant::Nvfp4G16) {
            stream.alloc_zeros(crate::gemm::batch_part_len(spec, &target, true))?
        } else {
            stream.alloc_zeros(1)?
        };
        let f8b_part = if spec.quant == crate::spec::Quant::Gguf {
            stream.alloc_zeros(crate::gemm::ggml_part_len(spec, &target))?
        } else if spec.quant.is_fp8() {
            stream.alloc_zeros(crate::gemm::batch_part_len(spec, &target, false))?
        } else {
            stream.alloc_zeros(1)?
        };
        let mm_meta = stream.alloc_zeros(2 * (layout.slots + 1))?;
        let mm_table = stream.alloc_zeros(if spec.rope.mrope.is_some() { (layout.slots + 1) * max_ctx * 3 } else { 1 })?;
        let mm_bidi = stream.alloc_zeros(if spec.vision.as_ref().is_some_and(|v| v.config.bidirectional) { (layout.slots + 1) * max_ctx } else { 1 })?;
        let mut rt = Runtime {
            smem,
            linear_tuning,
            pre,
            gdn,
            mixer,
            mixer_launch,
            moe: match &spec.moe {
                Some(_) => Some(moe::MoeBuffers::new(&stream, chunk, spec, crate::codegen::moe_mma_enabled(spec, &target), crate::codegen::moe_mma_fp8_enabled(spec, &target), crate::codegen::moe_mma_bf16_enabled(spec, &target))?),
                None => None,
            },
            fp8_act,
            fp4_act,
            f4b_part,
            f8b_part,
            ggb_scratch: stream.alloc_zeros(if spec.quant == crate::spec::Quant::Gguf { crate::codegen::ggb_scratch_len(spec) } else { 8 })?,
            chunk,
            prefill_limit,
            seq: 0,
            batch,
            slots: layout.slots,
            pages,
            kv_blocks: layout.kv_blocks,
            kv_planes: layout.kv_planes,
            tables: vec![0; (layout.slots + 1) * pages],
            pdl: spec.runtime.pdl && target.pdl,
            sms,
            target,
            use_mma,
            plans: match tp {
                Some(_) => Plans::build(spec, use_mma).for_stage(role).for_tp(&crate::graph::Graph::decoder().shards()?, spec.runtime.fuse),
                None => Plans::build(spec, use_mma).for_stage(role),
            },
            partial: stream.alloc_zeros(if tp.is_some() { chunk * d.hidden } else { 1 })?,
            head_part: match (&spec.dims.head_rows, tp) {
                (Some(_), Some((n, _, _))) => Some(stream.alloc_zeros(HEAD_PART_ROWS.max(crate::runtime::step::row_capacity(layout.slots) + 1).min(chunk) * crate::tp::vocab_each(d.vocab, n)?)?),
                _ => None,
            },
            comm,
            role,
            spec: spec.clone(),
            max_ctx,
            embed,
            norm,
            lm_head,
            layers,
            described,
            scratch: spec
                .attention
                .scratch
                .iter()
                .map(|(n, w)| Ok((n.clone(), stream.alloc_zeros(chunk * crate::state::dim(spec, w)?)?)))
                .collect::<Result<_>>()?,
            int_scratch: spec
                .attention
                .int_scratch
                .iter()
                .map(|(n, w)| Ok((n.clone(), stream.alloc_zeros(chunk * crate::state::dim(spec, w)?)?)))
                .collect::<Result<_>>()?,
            buf,
            kernels,
            stream,
            cache_hit: loaded.cache_hit,
            jit_key: loaded.key,
            jit_compile_secs: loaded.compile_secs,
            jit_module_secs: loaded.module_secs,
            arch: jit.arch().to_string(),
            device_bytes: 0,
            weight_bytes,
            weight_secs,
            load_secs: 0.0,
            peak_host_rss_bytes: 0,
            ctx: ctx.clone(),
            baseline_free: free_before,
            mtp: None,
            snap: Default::default(),
            host: crate::hostpool::HostPool::new(ctx.clone()),
            streamer,
            expert_cache: None,
            offload_plan,
            mm_meta,
            mm_table,
            mm_bidi,
            media: (0..=layout.slots).map(|_| None).collect(),
            vision: None,
        };
        if let Some(w) = mtp_weights {
            rt.mtp = Some(mtp::MtpState::new(&rt, w)?);
        }
        rt.build_expert_cache(weight_bytes)?;
        let hosts_vision = role.first && tp.is_none_or(|(_, rank, _)| rank == 0);
        if let Some(v) = spec.vision.as_ref().filter(|_| hosts_vision)
            && crate::vision_encoder::has_tower(&checkpoint, &v.config) {
                let opts = spec.runtime.vision;
                let stream = rt.ctx.new_stream()?;
                let enc = crate::vision_encoder::VisionEncoder::load_with(&rt.ctx, &stream, &jit, &checkpoint, &v.config, opts.precision)?;
                rt.vision = Some(Arc::new(crate::media_encode::MediaEncoder::with_tiles(Box::new(enc), v.tokens.clone(), v.config.spatial_merge_size, if v.config.pooled() { v.config.tokens_per_tile() } else { 0 }, opts)));
                checkpoint.release();
            }
        rt.stream.synchronize()?;
        crate::progress::report(crate::progress::Stage::KvCache, 0, 1);
        let kv_blocks = match layout.kv_blocks {
            0 => {
                let free = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
                let spare = free.saturating_sub(layout.kv_headroom);
                ensure!(spare >= 2 * block_elems * 4, "no device memory left for the KV cache ({free} bytes free)");
                spare / (block_elems.max(1) * 4)
            }
            n => n,
        };
        rt.buf.caches = crate::state::positional(spec)
            .map(|s| rt.stream.alloc_zeros(((kv_blocks + 1) * s.block_bytes(spec, layout.kv_planes) / 2).max(1)).with_context(|| format!("KV cache {}", s.name)))
            .collect::<Result<_>>()?;
        if rt.buf.caches.is_empty() {
            rt.buf.caches.push(rt.stream.alloc_zeros(1)?);
        }
        rt.kv_blocks = kv_blocks;
        let pad_row = layout.slots * pages;
        rt.tables[pad_row] = kv_blocks as i32;
        rt.stream.memcpy_htod(&[kv_blocks as i32], &mut rt.buf.block_table.slice_mut(pad_row..pad_row + 1))?;
        for slot in 0..layout.slots.min(kv_blocks / pages.max(1)) {
            let blocks: Vec<u32> = (0..pages as u32).map(|p| (slot * pages) as u32 + p).collect();
            rt.set_block_table(slot, &blocks)?;
        }
        rt.acquire_state(layout.slots)?;
        if !layout.state_on_demand {
            for slot in 0..layout.slots {
                rt.acquire_state(slot)?;
            }
        }
        rt.stream.synchronize()?;
        crate::progress::report(crate::progress::Stage::KvCache, 1, 1);
        let free_after = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        rt.device_bytes = free_before.saturating_sub(free_after) as u64;
        rt.load_secs = load_started.elapsed().as_secs_f64();
        rt.peak_host_rss_bytes = peak_rss_bytes().unwrap_or(0);
        tracing::info!(
            spec = %rt.spec.name,
            jit_key = %rt.jit_key,
            jit_cache_hit = rt.cache_hit,
            jit_secs = jit_secs,
            jit_compile_secs = rt.jit_compile_secs,
            jit_module_secs = rt.jit_module_secs,
            arch = %rt.arch,
            shards = checkpoint.files.len(),
            weights_direct,
            weight_bytes = rt.weight_bytes,
            weight_secs = rt.weight_secs,
            peak_host_rss_bytes = rt.peak_host_rss_bytes,
            vram_before_bytes = free_before,
            vram_after_bytes = free_after,
            device_bytes = rt.device_bytes,
            load_secs = rt.load_secs,
            "runtime loaded"
        );
        Ok(rt)
    }

    fn launcher<'a>(&'a self, f: &'a CudaFunction) -> crate::launch::Launch<'a> {
        crate::launch::Launch::new(&self.stream, f, self.pdl, self.sms * 8)
    }

    fn kernel(&self, name: &str) -> &CudaFunction {
        &self.kernels[name]
    }

    fn slot_table(&self, slot: usize) -> CudaView<'_, i32> {
        self.buf.block_table.slice(slot * self.pages..(slot + 1) * self.pages)
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    pub fn offload_stats(&self) -> Option<crate::offload::OffloadStats> {
        let s = self.streamer.as_ref().map(|s| s.stats());
        let e = self.expert_cache.as_ref().map(|c| c.stats());
        match (s, e) {
            (Some(a), Some(b)) => Some(crate::offload::OffloadStats {
                vram_hits: a.vram_hits + b.vram_hits,
                host_hits: a.host_hits + b.host_hits,
                nvme_reads: a.nvme_reads + b.nvme_reads,
                nvme_read_bytes: a.nvme_read_bytes + b.nvme_read_bytes,
                host_to_vram_bytes: a.host_to_vram_bytes + b.host_to_vram_bytes,
                evictions: a.evictions + b.evictions,
            }),
            (a, b) => a.or(b),
        }
    }

    pub fn expert_slots(&self) -> usize {
        self.expert_cache.as_ref().map_or(0, |c| c.slots)
    }

    pub fn expert_hotlist(&self) -> Option<crate::offload::ExpertHotlist> {
        self.expert_cache.as_ref().map(|c| c.hotlist())
    }

    pub fn save_expert_hotlist(&self, path: &std::path::Path) -> Result<()> {
        let h = self.expert_hotlist().context("no expert cache is active")?;
        h.save(path)
    }

    fn build_expert_cache(&mut self, weight_bytes: u64) -> Result<()> {
        let Some(m) = self.spec.moe.as_ref().filter(|_| expert_cache_mode(&self.spec)) else { return Ok(()) };
        let Some(like) = self.layers.iter().find_map(|l| l.moe.as_ref().and_then(|x| x.host.as_ref())) else { return Ok(()) };
        let off = &self.spec.runtime.offload;
        let (local, shared, top_k) = (m.local_experts(), m.shared_slots(), m.top_k());
        let entry = (like.entry_bytes() / (local + shared).max(1)).max(1) as u64;
        let slots = if off.expert_slots > 0 {
            off.expert_slots
        } else {
            let resident = weight_bytes.saturating_sub(self.layers.iter().filter_map(|l| l.moe.as_ref()).filter_map(|x| x.host.as_ref()).map(|h| h.host_bytes()).sum::<u64>());
            let room = off.vram_budget.saturating_sub(resident + self.staging_bytes());
            (room / entry).min(usize::MAX as u64 / 2) as usize
        };
        let slots = slots.clamp(top_k.min(local), local);
        let prior = match &off.hotlist {
            Some(p) if std::path::Path::new(p).exists() => Some(crate::offload::ExpertHotlist::load(std::path::Path::new(p))?),
            _ => None,
        };
        let cache = moe::experts::ExpertCache::new(&self.stream, like, self.layers.len(), slots, local, shared, off.policy, prior)?;
        let hosts: Vec<(usize, &moe::experts::HostExperts)> = self.layers.iter().enumerate().filter_map(|(i, l)| l.moe.as_ref().and_then(|x| x.host.as_ref()).map(|h| (i, h))).collect();
        cache.warm(&self.stream, &hosts)?;
        self.expert_cache = Some(cache);
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn offload_plan(&self) -> Option<&crate::offload::OffloadPlan> {
        self.offload_plan.as_ref()
    }

    pub fn streamed_layers(&self) -> usize {
        self.streamer.as_ref().map_or(0, |s| s.streamed_layers())
    }

    pub fn staging_bytes(&self) -> u64 {
        self.streamer.as_ref().map_or(0, |s| s.slot_bytes())
    }

    fn enter_layer(&self, layer: usize) -> Result<()> {
        match &self.streamer {
            Some(s) => s.enter(layer, &self.stream),
            None => Ok(()),
        }
    }

    fn leave_layer(&self, layer: usize) -> Result<()> {
        match &self.streamer {
            Some(s) => s.leave(layer, &self.stream),
            None => Ok(()),
        }
    }

    pub fn kv_blocks(&self) -> usize {
        self.kv_blocks
    }

    pub fn host_pool(&self) -> &crate::hostpool::HostPool {
        &self.host
    }

    pub fn kv_planes(&self) -> usize {
        self.kv_planes
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn set_block_table(&mut self, slot: usize, blocks: &[u32]) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        ensure!(blocks.len() <= self.pages, "{} blocks for a table of {}", blocks.len(), self.pages);
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "block outside the pool");
        let row = &mut self.tables[slot * self.pages..(slot + 1) * self.pages];
        let old = row[..blocks.len()].to_vec();
        for (dst, &b) in row.iter_mut().zip(blocks) {
            *dst = b as i32;
        }
        let changed = old.iter().zip(blocks).position(|(&o, &b)| o != b as i32);
        let Some(from) = changed else { return Ok(()) };
        let at = slot * self.pages + from;
        let host = self.tables[at..slot * self.pages + blocks.len()].to_vec();
        self.stream.memcpy_htod(&host, &mut self.buf.block_table.slice_mut(at..at + host.len()))?;
        Ok(())
    }

    fn linear(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let mut b = self.launcher(self.kernel(name));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(x).arg(out);
        unsafe { b.launch(gemv_cfg(name, rows, w, self.sms)) }?;
        Ok(())
    }

    fn linear_norm(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        norm: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let mut b = self.launcher(self.kernel(name));
        w.args(&mut b, &self.buf.dummy);
        if !name.starts_with("gate_up_act") {
            b.arg(bias.unwrap_or(&self.buf.dummy));
        }
        b.arg(x).arg(norm).arg(out);
        unsafe { b.launch(gemv_cfg(name, rows, w, self.sms)) }?;
        Ok(())
    }

    fn step(&self) -> Result<()> {
        self.step_traced(None)
    }

    fn mlp_decode(&self, l: &DeviceLayer) -> Result<()> {
        if l.moe.is_some() {
            return self.moe_block(l, &self.buf.x, &self.buf.token, 1);
        }
        let s = &self.spec;
        let buf = &self.buf;
        let fp4 = matches!(l.gate_up, DeviceMatrix::Fp4 { .. });
        let (act, down) = if fp4 { ("gate_up_act_f4", "down_proj_f4") } else { ("gate_up_act", "down_proj") };
        if s.norm.pre {
            let pre = l.ln_pre_mlp.as_ref().unwrap_or(&l.ln2);
            self.linear_norm(act, &l.gate_up, None, &buf.x, pre, &buf.act, s.dims.intermediate)?;
        } else {
            self.linear("gate_up_proj", &l.gate_up, None, &buf.x, &buf.gate_up, 2 * s.dims.intermediate)?;
            let mut b = self.launcher(self.kernel("swiglu"));
            b.arg(&buf.gate_up).arg(&buf.act);
            unsafe { b.launch(cfg(grid(s.dims.intermediate, 256), 256)) }?;
        }
        if s.uses_branch() {
            self.clear_branch()?;
            self.linear(down, &l.down, None, &buf.act, &self.pre.branch, s.dims.hidden)?;
            return self.merge_branch(l.ln_post_mlp.as_ref());
        }
        self.linear(down, &l.down, None, &buf.act, &buf.x, s.dims.hidden)
    }

    fn merge_branch(&self, post: Option<&CudaSlice<f32>>) -> Result<()> {
        match post {
            Some(weight) => self.add_normed_branch(weight),
            None => {
                let mut b = self.launcher(self.kernel("residual_add"));
                b.arg(&self.buf.x).arg(&self.pre.branch).arg(&self.buf.x);
                unsafe { b.launch(cfg(grid(self.spec.dims.hidden, 256), 256)) }?;
                Ok(())
            }
        }
    }

    fn clear_branch(&self) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let mut b = self.launcher(self.kernel("clear_branch"));
        b.arg(&self.pre.branch).arg(&self.pre.branch);
        unsafe { b.launch(cfg(grid(hidden, 256), 256)) }?;
        Ok(())
    }

    fn add_normed_branch(&self, weight: &CudaSlice<f32>) -> Result<()> {
        let mut b = self.launcher(self.kernel("norm_add"));
        b.arg(&self.pre.branch).arg(weight).arg(&self.buf.x).arg(&self.buf.x);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn attention_out_decode(&self, l: &DeviceAttn, layer: &DeviceLayer) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        if self.spec.uses_branch() {
            self.clear_branch()?;
            self.linear("o_proj", &l.o, l.o_bias.as_ref(), &self.buf.attn, &self.pre.branch, hidden)?;
            return self.merge_branch(self.spec.norm.sandwich.then_some(&layer.ln2));
        }
        self.linear("o_proj", &l.o, l.o_bias.as_ref(), &self.buf.attn, &self.buf.x, hidden)
    }

    fn qk_norm_decode(&self, l: &DeviceAttn) -> Result<()> {
        if !self.spec.attention.qk_norm_full {
            return Ok(());
        }
        for (name, w) in [("q_norm_rows", &l.q_norm), ("k_norm_rows", &l.k_norm)] {
            let mut b = self.launcher(self.kernel(name));
            b.arg(&self.buf.qkv).arg(w.as_ref().expect("full-width qk norm has weights")).arg(&self.buf.qkv);
            unsafe { b.launch(cfg(1, 1024)) }?;
        }
        Ok(())
    }

    fn softcap_decode(&self) -> Result<()> {
        let vocab = self.spec.dims.vocab;
        for (present, name) in [(self.spec.dims.logit_scale.is_some(), "logit_scale"), (self.spec.dims.logit_softcap.is_some(), "logit_softcap")] {
            if present {
                let mut b = self.launcher(self.kernel(name));
                b.arg(&self.buf.logits).arg(&self.buf.logits);
                unsafe { b.launch(cfg(grid(vocab, 256), 256)) }?;
            }
        }
        Ok(())
    }

    fn trace_layer(&self, trace: Option<&CudaSlice<f32>>, layer: usize, tokens: usize) -> Result<()> {
        let Some(trace) = trace else { return Ok(()) };
        let n = tokens * self.spec.dims.hidden;
        let src = if tokens == 1 { self.buf.x.slice(..n) } else { self.pre.x.slice(..n) };
        let dst = trace.slice(layer * n..(layer + 1) * n);
        self.copy_rows(&src, &dst, n)
    }

    fn copy_rows(&self, src: &CudaView<f32>, dst: &CudaView<f32>, n: usize) -> Result<()> {
        let mut b = self.launcher(self.kernel("copy_rows"));
        let count = n as i32;
        b.arg(src).arg(dst).arg(&count);
        unsafe { b.launch(cfg(grid(n, 256).min(1024), 256)) }?;
        Ok(())
    }

    fn gdn_decode(&self, g: &DeviceGdn, ln1: &CudaSlice<f32>, x: &CudaSlice<f32>) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        self.linear_norm("lin_in_norm", &g.in_proj, None, x, ln1, &gb.lin, la.in_proj_rows())?;
        self.linear_norm("lin_ba_norm", &g.ba, None, x, ln1, &gb.ba, la.ba_rows())?;
        let zero = 0i32;
        let mut b = self.launcher(self.kernel("gdn_conv_step"));
        b.arg(&gb.lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.mix).arg(&g.slot).arg(&zero).arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(grid(la.conv_dim(), 256), 256)) }?;
        let mut b = self.launcher(self.kernel("gdn_step"));
        b.arg(&gb.mix)
            .arg(&gb.lin)
            .arg(&gb.ba)
            .arg(&g.neg_a)
            .arg(&g.dt_bias)
            .arg(&g.norm)
            .arg(&*gb.ssm_state)
            .arg(&gb.out)
            .arg(&g.slot)
            .arg(&zero)
            .arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(la.value_heads as u32, (la.value_head_dim * self.linear_tuning.ksplit) as u32)) }?;
        self.linear("lin_out_proj", &g.out, None, &gb.out, x, self.spec.dims.hidden)
    }

    fn step_traced(&self, trace: Option<&CudaSlice<f32>>) -> Result<()> {
        ensure!(self.head_part.is_none(), "a vocabulary sharded rank decodes through the plan, not the single stream step");
        ensure!(self.spec.requanted_linears().is_empty(), "requantised attention and delta-net projections run on the plan decode path (the scheduler), not the single stream step");
        ensure!(self.spec.attention.node.is_empty() && self.spec.mixer.is_none() && !self.spec.layers.split && self.spec.mlp.mapped.is_none(), "described attention, mixers, mlp-only layers and non-gated mlps run on the batched decode path (generate_batch and the scheduler), not the single stream step");
        let s = &self.spec;
        let a = &s.attention;
        let buf = &self.buf;
        let mut b = self.launcher(self.kernel("embed"));
        b.arg(&self.embed).arg(&buf.token).arg(&buf.x);
        unsafe { b.launch(cfg(grid(s.dims.hidden, 256), 256)) }?;
        let qkv_rows = s.qkv_rows();
        for (i, layer) in self.layers.iter().enumerate() {
            self.enter_layer(i)?;
            let l = match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    self.gdn_decode(g, &layer.ln1, &buf.x)?;
                    self.mlp_decode(layer)?;
                    self.trace_layer(trace, i, 1)?;
                    self.leave_layer(i)?;
                    continue;
                }
                DeviceMixer::Attention(l) => l,
                DeviceMixer::Described(_) | DeviceMixer::None => unreachable!("a described mixer is refused above"),
            };
            let kv_layer = l.kv;
            let window = l.window;
            let table = self.slot_table(self.seq as usize);
            let (pages, mm_stride) = (self.pages as i32, self.max_ctx as i32);
            if s.norm.pre {
                self.linear_norm("qkv_proj_norm", &l.qkv, l.qkv_bias.as_ref(), &buf.x, &layer.ln1, &buf.qkv, qkv_rows)?;
            } else {
                self.linear("qkv_proj", &l.qkv, l.qkv_bias.as_ref(), &buf.x, &buf.qkv, qkv_rows)?;
            }
            self.qk_norm_decode(l)?;
            let mut b = self.launcher(self.kernel("qk_norm_rope"));
            b.arg(&buf.qkv)
                .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
                .arg(buf.k_cache())
                .arg(buf.v_cache())
                .arg(&table)
                .arg(&buf.pos)
                .arg(&buf.zero_slots)
                .arg(&kv_layer)
                .arg(&0i32)
                .arg(&pages)
                .arg(&mm_stride)
                .arg(&self.mm_meta)
                .arg(&self.mm_table)
                .arg(&self.seq);
            unsafe { b.launch(cfg((a.heads + 2 * a.kv_heads) as u32, a.head_dim.max(32) as u32)) }?;
            let mut b = self.launcher(self.kernel("attention_split"));
            b.arg(&buf.qkv)
                .arg(buf.k_cache())
                .arg(buf.v_cache())
                .arg(&table)
                .arg(&buf.pos)
                .arg(&kv_layer)
                .arg(&buf.attn_part)
                .arg(&window);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (a.kv_heads as u32, ATTN_SPLITS as u32, 1),
                    block_dim: (ATTN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let mut b = self.launcher(self.kernel("attention_reduce"));
            b.arg(&buf.attn_part).arg(&buf.pos).arg(&buf.qkv).arg(&buf.attn).arg(&window);
            unsafe { b.launch(cfg(a.heads as u32, a.head_dim.max(32) as u32)) }?;
            self.attention_out_decode(l, layer)?;
            self.mlp_decode(layer)?;
            self.trace_layer(trace, i, 1)?;
            self.leave_layer(i)?;
        }
        let head_kernel = if matches!(self.lm_head, DeviceMatrix::Fp4 { .. }) { "lm_head_norm_f4" } else { "lm_head_norm" };
        self.linear_norm(
            head_kernel,
            &self.lm_head,
            None,
            &buf.x,
            &self.norm,
            &buf.logits,
            s.dims.vocab,
        )?;
        self.softcap_decode()
    }

    fn gemm(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaView<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.launcher(self.kernel(name));
        w.args(&mut b, &self.buf.dummy);
        let n = tokens as i32;
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(x).arg(out).arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, GEMM_TILE), grid(tokens, GEMM_TILE), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc(
        &self,
        name: &str,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let choice = crate::codegen::select_act(&self.spec, &self.target, name, tokens, w.kind(), !matches!(w, DeviceMatrix::Fp4 { act: false, .. }))?;
        use crate::codegen::GemmPath;
        match (choice.path, &self.fp8_act, &self.fp4_act) {
            (GemmPath::F8a, Some(fa), _) => return self.tc_fp8(fa, &choice, w, bias, x, out, rows, tokens),
            (GemmPath::F4a, _, Some(fa)) => return self.tc_fp4(fa, &choice, w, x, out, rows, tokens),
            (GemmPath::F8b, _, _) => return self.tc_f8b(&choice, w, bias, x, out, rows, tokens),
            (GemmPath::F4b, _, _) => return self.tc_f4b(&choice, w, x, out, rows, tokens),
            (GemmPath::Rows, _, _) => return self.tc_rows(&choice, w, x, out, rows, tokens),
            (GemmPath::Ggr, _, _) => return self.tc_ggr(&choice, w, bias, x, out, rows, tokens),
            (GemmPath::Ggv, _, _) => return self.tc_ggv(&choice, w, bias, x, out, rows),
            (GemmPath::Ggb, _, _) => return self.tc_ggb(&choice, w, bias, x, out, rows, tokens),
            _ => {}
        }
        let (kernel, tile) = (choice.name.as_str(), choice.tile);
        let mut b = self.launcher(self.kernel(kernel));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let n = tokens as i32;
        b.arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, tile.bn), grid(tokens, tile.bm), 1),
            block_dim: (tile.threads(), 1, 1),
            shared_mem_bytes: self.smem[kernel],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn tc_rows(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let GemmOut::F32(o) = out else { unreachable!("tc_rows writes f32") };
        let n = tokens as i32;
        let mut b = self.launcher(self.kernel(&choice.name));
        w.args(&mut b, &self.buf.dummy);
        b.arg(&self.buf.dummy).arg(&x.0).arg(&x.1).arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16).arg(&n);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (rows as u32, tokens.div_ceil(crate::codegen::ROWS_MAX_TOKENS) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_ggb(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let DeviceMatrix::Ggml { data, table } = w else { unreachable!("tc_ggb takes GGUF weights") };
        let cols = self.ggb_cols(choice.name.as_str());
        let chunks = rows * cols / 8;
        let (r, c) = (rows as i32, cols as i32);
        let mut b = self.launcher(self.kernel("gg_dequant_bf16"));
        b.arg(data).arg(table).arg(&self.ggb_scratch).arg(&r).arg(&c);
        unsafe { b.launch(cfg(grid(chunks, 256), 256)) }?;
        let tile = choice.tile;
        let kernel = choice.name.as_str();
        let mut b = self.launcher(self.kernel(kernel));
        b.arg(&self.ggb_scratch).arg(&self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let n = tokens as i32;
        b.arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(tokens, tile.bm), grid(rows, tile.bn), 1),
            block_dim: (tile.threads(), 1, 1),
            shared_mem_bytes: self.smem[kernel],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_ggv(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
    ) -> Result<()> {
        let DeviceMatrix::Ggml { data, table } = w else { unreachable!("tc_ggv takes GGUF weights") };
        let mut b = self.launcher(self.kernel(&choice.name));
        b.arg(data).arg(table);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f8b_part).arg(&0i32).arg(&0i32);
        let per_block = crate::codegen::GGML_GEMV_THREADS as usize / 32 * crate::gemm::ggml_gemv_rows(rows);
        let blocks = grid(rows, per_block).min(self.sms * GEMV_BLOCKS_PER_SM);
        unsafe { b.launch(cfg(blocks, crate::codegen::GGML_GEMV_THREADS)) }?;
        Ok(())
    }

    fn tc_ggr(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::{BATCH_REDUCE_BLOCK as GGR_REDUCE_BLOCK, BatchShape};
        let DeviceMatrix::Ggml { data, table } = w else { unreachable!("tc_ggr takes GGUF weights") };
        let linear = choice.name.split("_ggr").next().expect("a ggr kernel");
        let cols = crate::codegen::linear_dims(&self.spec, linear).1;
        let shape = BatchShape::for_ggml(tokens);
        let splits = shape.ggml_splits(&self.target, rows, cols);
        let (n, s) = (tokens as i32, splits as i32);
        let mut b = self.launcher(self.kernel(&choice.name));
        b.arg(data).arg(table);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f8b_part).arg(&n).arg(&s);
        let config = LaunchConfig {
            grid_dim: (grid(rows, shape.rows()), splits as u32, 1),
            block_dim: (shape.threads(), 1, 1),
            shared_mem_bytes: shape.smem(),
        };
        unsafe { b.launch(config) }?;
        if splits == 1 {
            return Ok(());
        }
        let reduce = choice.reduce.as_deref().expect("ggr has a reduce kernel");
        let mut b = self.launcher(self.kernel(reduce));
        b.arg(&self.f8b_part).arg(bias.unwrap_or(&self.buf.dummy));
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let outs = match out {
            GemmOut::Planes(_) => tokens * self.spec.gate_up_out(rows),
            GemmOut::F32(_) => tokens * rows,
        };
        b.arg(&n).arg(&s);
        unsafe { b.launch(cfg(grid(outs, GGR_REDUCE_BLOCK), GGR_REDUCE_BLOCK as u32)) }?;
        Ok(())
    }

    fn ggb_cols(&self, kernel: &str) -> usize {
        let linear = crate::codegen::GGB_LINEARS.iter().find(|l| kernel.starts_with(&format!("{l}_ggb"))).expect("a ggb kernel");
        crate::codegen::linear_dims(&self.spec, linear).1
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_f8b(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::{BATCH_REDUCE_BLOCK as F8B_REDUCE_BLOCK, BatchShape};
        let DeviceMatrix::Fp8 { values, .. } = w else { unreachable!("tc_f8b takes FP8 weights") };
        let width = values.len() / rows;
        let shape = BatchShape::for_tokens(tokens);
        let splits = shape.splits(&self.target, rows, width);
        let (n, s) = (tokens as i32, splits as i32);
        let mut b = self.launcher(self.kernel(&choice.name));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f8b_part).arg(&n).arg(&s);
        let config = LaunchConfig {
            grid_dim: (grid(rows, shape.rows()), splits as u32, 1),
            block_dim: (shape.threads(), 1, 1),
            shared_mem_bytes: shape.smem(),
        };
        unsafe { b.launch(config) }?;
        if splits == 1 {
            return Ok(());
        }
        let reduce = choice.reduce.as_deref().expect("f8b has a reduce kernel");
        let mut b = self.launcher(self.kernel(reduce));
        b.arg(&self.f8b_part).arg(bias.unwrap_or(&self.buf.dummy));
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let outs = match out {
            GemmOut::Planes(_) => tokens * self.spec.gate_up_out(rows),
            GemmOut::F32(_) => tokens * rows,
        };
        b.arg(&n).arg(&s);
        unsafe { b.launch(cfg(grid(outs, F8B_REDUCE_BLOCK), F8B_REDUCE_BLOCK as u32)) }?;
        Ok(())
    }

    fn tc_f4b(
        &self,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        use crate::codegen::{BATCH_REDUCE_BLOCK as F4B_REDUCE_BLOCK, BatchShape};
        let DeviceMatrix::Fp4 { packed, .. } = w else { unreachable!("tc_f4b takes NVFP4 weights") };
        let width = packed.len() * 2 / rows;
        let shape = BatchShape::for_tokens(tokens);
        let splits = shape.splits(&self.target, rows, width);
        let (n, s) = (tokens as i32, splits as i32);
        let mut b = self.launcher(self.kernel(&choice.name));
        w.args(&mut b, &self.buf.dummy);
        b.arg(&x.0).arg(&x.1);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        b.arg(&self.f4b_part).arg(&n).arg(&s);
        let config = LaunchConfig {
            grid_dim: (grid(rows, shape.rows()), splits as u32, 1),
            block_dim: (shape.threads(), 1, 1),
            shared_mem_bytes: shape.smem(),
        };
        unsafe { b.launch(config) }?;
        if splits == 1 {
            return Ok(());
        }
        let reduce = choice.reduce.as_deref().expect("f4b has a reduce kernel");
        let mut b = self.launcher(self.kernel(reduce));
        b.arg(&self.f4b_part);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let outs = match out {
            GemmOut::Planes(_) => tokens * self.spec.gate_up_out(rows),
            GemmOut::F32(_) => tokens * rows,
        };
        b.arg(&n).arg(&s);
        unsafe { b.launch(cfg(grid(outs, F4B_REDUCE_BLOCK), F4B_REDUCE_BLOCK as u32)) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_fp4(
        &self,
        fa: &Fp4Act,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let DeviceMatrix::Fp4 { packed, scales, .. } = w else { unreachable!("tc_fp4 takes NVFP4 weights") };
        let width = packed.len() * 2 / rows;
        let width_i = width as i32;
        let groups = width / crate::nvfp4::GROUP;
        let mut b = self.launcher(self.kernel(choice.quant.expect("f4a quantises its input")));
        b.arg(&x.0).arg(&x.1).arg(scales).arg(&fa.q).arg(&fa.scales).arg(&width_i);
        let config = LaunchConfig {
            grid_dim: (grid(groups, 128), tokens as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let (kernel, tile) = (choice.name.as_str(), choice.tile);
        let mut b = self.launcher(self.kernel(kernel));
        b.arg(packed).arg(scales).arg(&fa.q).arg(&fa.scales);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let n = tokens as i32;
        b.arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, tile.bn), grid(tokens, tile.bm), 1),
            block_dim: (tile.threads(), 1, 1),
            shared_mem_bytes: self.smem[kernel],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tc_fp8(
        &self,
        fa: &Fp8Act,
        choice: &crate::codegen::KernelChoice,
        w: &DeviceMatrix,
        bias: Option<&CudaSlice<f32>>,
        x: (CudaView<u16>, CudaView<u16>),
        out: GemmOut,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let width = match w {
            DeviceMatrix::Fp8 { values, .. } => values.len() / rows,
            _ => unreachable!("tc_fp8 takes FP8 weights"),
        };
        let width_i = width as i32;
        let mut b = self.launcher(self.kernel(choice.quant.expect("f8 quantises its input")));
        b.arg(&x.0).arg(&x.1).arg(&fa.q).arg(&fa.scale).arg(&width_i);
        let config = if fa.per_token {
            LaunchConfig { grid_dim: (tokens as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
        } else {
            LaunchConfig {
                grid_dim: ((width / crate::codegen::FP8_BLOCK) as u32, tokens as u32, 1),
                block_dim: ((crate::codegen::FP8_BLOCK / 2) as u32, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        unsafe { b.launch(config) }?;
        let (kernel, tile) = (choice.name.as_str(), choice.tile);
        let mut b = self.launcher(self.kernel(kernel));
        w.args(&mut b, &self.buf.dummy);
        b.arg(bias.unwrap_or(&self.buf.dummy)).arg(&fa.q).arg(&fa.scale);
        match out {
            GemmOut::F32(o) => b.arg(o).arg(&self.pre.dummy_u16).arg(&self.pre.dummy_u16),
            GemmOut::Planes(p) => b.arg(&self.buf.dummy).arg(&p.hi).arg(&p.lo),
        };
        let n = tokens as i32;
        b.arg(&n);
        let config = LaunchConfig {
            grid_dim: (grid(rows, tile.bn), grid(tokens, tile.bm), 1),
            block_dim: (tile.threads(), 1, 1),
            shared_mem_bytes: self.smem[kernel],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn rms_split(&self, x: &CudaView<f32>, w: &CudaSlice<f32>, out: &Planes, tokens: usize) -> Result<()> {
        self.rms_split_into(x, w, &out.hi.as_view(), &out.lo.as_view(), tokens)
    }

    fn rms_split_into(
        &self,
        x: &CudaView<f32>,
        w: &CudaSlice<f32>,
        hi: &CudaView<u16>,
        lo: &CudaView<u16>,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.launcher(self.kernel("rms_split"));
        b.arg(x).arg(w).arg(hi).arg(lo);
        unsafe { b.launch(cfg(tokens as u32, 256)) }?;
        Ok(())
    }

    fn rmsnorm_rows(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<()> {
        let mut b = self.launcher(self.kernel("rmsnorm"));
        b.arg(x).arg(w).arg(out);
        unsafe { b.launch(cfg(tokens as u32, 1024)) }?;
        Ok(())
    }

    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start: usize,
        all_logits: Option<&CudaSlice<f32>>,
    ) -> Result<()> {
        self.prefill_chunk_traced(tokens, start, all_logits, None)
    }

    fn gdn_prefill_rows(&self, g: &DeviceGdn, n: usize, at: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let slot = g.slot + self.seq * self.spec.linear_layers() as i32;
        let d = la.key_head_dim;
        let lin = gb.pre_lin.slice(at * la.in_proj_rows()..);
        let ba = gb.pre_ba.slice(at * la.ba_rows()..);
        let mix = gb.pre_mix.slice(at * la.conv_dim()..);
        let o = gb.o.slice(at * la.value_dim()..);
        let hi = gb.planes.hi.slice(at * la.value_dim()..);
        let lo = gb.planes.lo.slice(at * la.value_dim()..);
        let launch = |name: &str, grid_dim: (u32, u32, u32), block: u32, smem: u32| LaunchConfig {
            grid_dim,
            block_dim: (block, 1, 1),
            shared_mem_bytes: if name.is_empty() { 0 } else { smem },
        };
        let mut b = self.launcher(self.kernel("gdn_conv_prefill"));
        b.arg(&lin).arg(&g.conv).arg(&*gb.conv_state).arg(&mix).arg(&slot);
        unsafe { b.launch(launch("", (grid(la.conv_dim(), 256), n as u32, 1), 256, 0)) }?;
        let ntok = n as i32;
        let mut b = self.launcher(self.kernel("gdn_conv_keep"));
        b.arg(&lin).arg(&*gb.conv_state).arg(&slot).arg(&ntok);
        unsafe { b.launch(cfg(grid(la.conv_dim(), 256), 256)) }?;
        let mut b = self.launcher(self.kernel("gdn_qk_prep"));
        b.arg(&mix).arg(&gb.qn).arg(&gb.kn);
        unsafe { b.launch(launch("", (la.key_heads as u32, n as u32, 1), d as u32, 0)) }?;
        let mut b = self.launcher(self.kernel("gdn_gates"));
        b.arg(&ba).arg(&g.neg_a).arg(&g.dt_bias).arg(&gb.g).arg(&gb.beta);
        unsafe { b.launch(cfg(n as u32, 64)) }?;
        let chunks = n.div_ceil(self.linear_tuning.chunk) as u32;
        let mut b = self.launcher(self.kernel("gdn_chunk_prep"));
        b.arg(&gb.qn)
            .arg(&gb.kn)
            .arg(&mix)
            .arg(&gb.g)
            .arg(&gb.beta)
            .arg(&gb.w)
            .arg(&gb.u)
            .arg(&gb.aqk)
            .arg(&gb.gc)
            .arg(&gb.qe)
            .arg(&gb.kt)
            .arg(&ntok);
        let smem = self.smem["gdn_chunk_prep"];
        unsafe { b.launch(launch("prep", (la.value_heads as u32, chunks, 1), 256, smem)) }?;
        let mut b = self.launcher(self.kernel("gdn_chunk_scan"));
        b.arg(&gb.w)
            .arg(&gb.qe)
            .arg(&gb.kt)
            .arg(&gb.u)
            .arg(&gb.aqk)
            .arg(&gb.gc)
            .arg(&*gb.ssm_state)
            .arg(&o)
            .arg(&slot)
            .arg(&ntok);
        unsafe { b.launch(launch("", (la.value_heads as u32, (d / self.linear_tuning.cols) as u32, 1), self.linear_tuning.chunk_threads(), 0)) }?;
        let mut b = self.launcher(self.kernel("gdn_gated_norm"));
        b.arg(&o).arg(&lin).arg(&g.norm).arg(&hi).arg(&lo);
        unsafe { b.launch(launch("", (la.value_heads as u32, n as u32, 1), (d / 2) as u32, 0)) }?;
        Ok(())
    }

    fn attn_prefill_rows(&self, l: &DeviceAttn, n: usize, at: usize) -> Result<()> {
        self.attn_rope_rows(l, n, at)?;
        self.attn_flash_rows(l, n, at)
    }

    fn attn_rope_rows(&self, l: &DeviceAttn, n: usize, at: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let mm_stride = self.max_ctx as i32;
        let qkv = pre.qkv.slice(at * self.spec.qkv_rows()..);
        let mut b = self.launcher(self.kernel("kv_norm_rope"));
        b.arg(&qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(buf.k_cache())
            .arg(buf.v_cache())
            .arg(&table)
            .arg(&buf.pos)
            .arg(&l.kv)
            .arg(&self.mm_meta)
            .arg(&self.mm_table)
            .arg(&mm_stride)
            .arg(&self.seq);
        let config = LaunchConfig {
            grid_dim: ((2 * a.kv_heads) as u32, n as u32, 1),
            block_dim: (a.head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn attn_flash_rows(&self, l: &DeviceAttn, n: usize, at: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let mm_stride = self.max_ctx as i32;
        let qkv = pre.qkv.slice(at * self.spec.qkv_rows()..);
        let hi = pre.attn_planes.hi.slice(at * self.spec.q_dim()..);
        let lo = pre.attn_planes.lo.slice(at * self.spec.q_dim()..);
        let mut b = self.launcher(self.kernel("flash_mma"));
        let ntok = n as i32;
        b.arg(&qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(buf.k_cache())
            .arg(buf.v_cache())
            .arg(&table)
            .arg(&buf.pos)
            .arg(&l.kv)
            .arg(&hi)
            .arg(&lo)
            .arg(&ntok)
            .arg(&self.mm_meta)
            .arg(&self.mm_table)
            .arg(&mm_stride)
            .arg(&self.seq)
            .arg(&l.window)
            .arg(&self.mm_bidi);
        let rows = crate::codegen::flash_rows(a.head_dim, self.kv_planes);
        let config = LaunchConfig {
            grid_dim: (grid(n, rows), a.heads as u32, 1),
            block_dim: (rows as u32 * 2, 1, 1),
            shared_mem_bytes: self.smem["flash_mma"],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn mlp_prefill(&self, l: &DeviceLayer, n: usize) -> Result<()> {
        if l.moe.is_some() {
            return self.moe_block(l, &self.pre.x, &self.pre.tokens, n);
        }
        let pre = &self.pre;
        let (hidden, inter) = (self.spec.dims.hidden, self.spec.dims.intermediate);
        self.rms_split(&pre.x.as_view(), &l.ln2, &pre.xn_planes, n)?;
        self.tc("gate_up", &l.gate_up, None, pre.xn_planes.all(), GemmOut::Planes(&pre.act_planes), 2 * inter, n)?;
        self.tc("down", &l.down, None, pre.act_planes.all(), GemmOut::F32(&pre.x), hidden, n)
    }

    fn prefill_chunk_traced(
        &mut self,
        tokens: &[u32],
        start: usize,
        all_logits: Option<&CudaSlice<f32>>,
        trace: Option<&CudaSlice<f32>>,
    ) -> Result<()> {
        let n = tokens.len();
        self.check_tokens(tokens)?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.stream.memcpy_htod(&ids, &mut self.pre.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&[start as i32], &mut self.buf.pos)?;
        let plan = &self.plans.prefill;
        let out = all_logits.unwrap_or(&self.buf.logits);
        let rows = if all_logits.is_some() { 0..n } else { n - 1..n };
        let run = exec::Run { n, tokens: &self.pre.tokens, logits: out, rows, bb: None, start: Some(start), qlen: 1, mtp: None };
        crate::executor::run_plan(self, plan, &run, &mut |i| self.trace_layer(trace, i, n))
    }

    fn prefill(&mut self, tokens: &[u32], all_logits: bool) -> Result<Vec<f32>> {
        let vocab = self.spec.dims.vocab;
        let scratch = if all_logits {
            Some(self.stream.alloc_zeros::<f32>(self.prefill_limit * vocab)?)
        } else {
            None
        };
        let mut out = Vec::new();
        for (c, piece) in tokens.chunks(self.prefill_limit).enumerate() {
            self.prefill_chunk(piece, c * self.prefill_limit, scratch.as_ref())?;
            if let Some(scratch) = &scratch {
                out.extend(self.stream.clone_dtoh(&scratch.slice(..piece.len() * vocab))?);
            }
        }
        Ok(out)
    }

    fn advance(&self, sampling: Sampling) -> Result<()> {
        let buf = &self.buf;
        let mut b = self.launcher(self.kernel("argmax_advance"));
        b.arg(&buf.logits).arg(&buf.token).arg(&buf.pos).arg(&buf.history).arg(&buf.count);
        let inv_temp = sampling.inverse_temperature();
        let top_k = sampling.top_k as i32;
        b.arg(&inv_temp).arg(&sampling.seed).arg(&top_k).arg(&sampling.top_p).arg(&sampling.min_p);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<()> {
        let vocab = self.spec.dims.vocab;
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            anyhow::bail!("token id {t} outside the vocabulary of {vocab}");
        }
        Ok(())
    }

    fn set(&mut self, token: u32, pos: usize) -> Result<()> {
        self.check_tokens(&[token])?;
        self.stream.memcpy_htod(&[token as i32], &mut self.buf.token)?;
        self.stream.memcpy_htod(&[pos as i32], &mut self.buf.pos)?;
        Ok(())
    }

    pub fn prompt_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(tokens.len() <= self.max_ctx, "prompt longer than max_ctx");
        self.reset_linear_state()?;
        self.prefill(tokens, true)
    }

    fn capture(&self, sampling: Sampling) -> Result<CudaGraph> {
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.step().and_then(|_| self.advance(sampling));
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        graph.context("empty graph")
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
    ) -> Result<Generation> {
        self.generate_with(prompt, new_tokens, sampling, &[], false, &mut |_| true)
    }

    pub fn generate_until(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Generation> {
        self.generate_with(prompt, new_tokens, sampling, stop, true, on_tokens)
    }

    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
        sampling: Sampling,
        stop: &[u32],
        chunked: bool,
        on_tokens: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Generation> {
        ensure!(self.comm.is_none(), "a tensor parallel rank generates through the scheduler, not the single stream loop");
        ensure!(!prompt.is_empty() && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(
            prompt.len() + new_tokens <= self.max_ctx,
            "prompt plus generation exceeds max_ctx"
        );
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        self.reset_linear_state()?;
        self.prefill(prompt, false)?;
        self.stream.memcpy_htod(&[prompt.len() as i32 - 1], &mut self.buf.pos)?;
        self.advance(sampling)?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        let graph = if self.spec.runtime.graphs {
            let graph = self.capture(sampling)?;
            graph.upload()?;
            Some(graph)
        } else {
            None
        };
        self.stream.synchronize()?;
        let started = Instant::now();
        let mut made = 1;
        let mut tokens: Vec<u32>;
        let mut shown = 0;
        loop {
            let seen = self.stream.clone_dtoh(&self.buf.history.slice(..made))?;
            tokens = seen.iter().map(|&t| t as u32).collect();
            let (kept, stopped) = cut_at_stop(&tokens, stop);
            let mut wanted = true;
            if kept > shown {
                wanted = on_tokens(&tokens[shown..kept]);
                shown = kept;
            }
            if stopped || !wanted || made >= new_tokens {
                tokens.truncate(kept);
                break;
            }
            let step = if chunked { DECODE_CHUNK.min(new_tokens - made) } else { new_tokens - made };
            for _ in 0..step {
                match &graph {
                    Some(graph) => graph.launch()?,
                    None => {
                        self.step()?;
                        self.advance(sampling)?;
                    }
                }
            }
            made += step;
        }
        self.stream.synchronize()?;
        let decode_secs = started.elapsed().as_secs_f64();
        let free_now = self.ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let peak_device_bytes =
            (self.baseline_free.saturating_sub(free_now) as u64).max(self.device_bytes);
        Ok(Generation { tokens, prefill_secs, decode_secs, peak_device_bytes })
    }

    fn first_token(&self, bb: &BatchBuffers, seq: usize) -> Result<()> {
        let stride = self.max_ctx as i32;
        let mut b = self.launcher(self.kernel("argmax_rows"));
        let tokens = bb.tokens.slice(seq..seq + 1);
        let history = bb.history.slice(seq * self.max_ctx..);
        b.arg(&self.buf.logits).arg(&tokens).arg(&history).arg(&self.buf.count).arg(&stride);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn gdn_decode_rows(&self, g: &DeviceGdn, slots: &CudaSlice<i32>, n: usize) -> Result<()> {
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let seq_slots = self.spec.linear_layers() as i32;
        let mut b = self.launcher(self.kernel("gdn_conv_step"));
        b.arg(&gb.pre_lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.pre_mix).arg(&g.slot).arg(&seq_slots).arg(slots);
        let config = LaunchConfig {
            grid_dim: (grid(la.conv_dim(), 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.launcher(self.kernel("gdn_step"));
        b.arg(&gb.pre_mix)
            .arg(&gb.pre_lin)
            .arg(&gb.pre_ba)
            .arg(&g.neg_a)
            .arg(&g.dt_bias)
            .arg(&g.norm)
            .arg(&*gb.ssm_state)
            .arg(&gb.o)
            .arg(&g.slot)
            .arg(&seq_slots)
            .arg(slots);
        let config = LaunchConfig {
            grid_dim: (la.value_heads as u32, n as u32, 1),
            block_dim: ((la.value_head_dim * self.linear_tuning.ksplit) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.launcher(self.kernel("split_rows"));
        let count = (n * la.value_dim()) as i32;
        b.arg(&gb.o).arg(&gb.planes.hi).arg(&gb.planes.lo).arg(&count);
        unsafe { b.launch(cfg(grid(n * la.value_dim(), 512), 256)) }?;
        Ok(())
    }

    fn reset_linear_state(&mut self) -> Result<()> {
        self.reset_slot(self.seq as usize)
    }

    pub fn state_slot_bytes(&self) -> usize {
        if let Some(mb) = &self.mixer {
            return mb.states.iter().map(|s| s.pool.stride_bytes()).sum();
        }
        self.gdn.as_ref().map_or(0, |gb| gb.conv_pool.stride_bytes() + gb.ssm_pool.stride_bytes())
    }

    pub fn state_mapped_bytes(&self) -> usize {
        if let Some(mb) = &self.mixer {
            return mb.states.iter().map(|s| s.pool.mapped_bytes()).sum();
        }
        self.gdn.as_ref().map_or(0, |gb| gb.conv_pool.mapped_bytes() + gb.ssm_pool.mapped_bytes())
    }

    pub fn state_live_slots(&self) -> usize {
        if let Some(mb) = &self.mixer {
            return mb.states.first().map_or(0, |s| s.pool.live_slots());
        }
        self.gdn.as_ref().map_or(0, |gb| gb.ssm_pool.live_slots())
    }

    fn mixer_state_live(&self, slot: usize) -> bool {
        self.mixer.as_ref().is_none_or(|mb| mb.states.iter().all(|s| s.pool.is_live(slot)))
    }

    fn acquire_mixer_state(&mut self, slot: usize, free: usize) -> Result<()> {
        let need = self.state_slot_bytes();
        let Some(mb) = self.mixer.as_mut() else { return Ok(()) };
        if mb.states.iter().all(|s| s.pool.is_live(slot)) {
            return Ok(());
        }
        ensure!(free >= need + STATE_FREE_MARGIN, "{free} bytes of device memory free, a recurrent state slot needs {need} plus a margin");
        for i in 0..mb.states.len() {
            if let Err(e) = mb.states[i].pool.acquire(slot) {
                for s in &mut mb.states[..i] {
                    s.pool.release(slot);
                }
                return Err(e);
            }
        }
        for s in &mut mb.states {
            let n = s.pool.stride_bytes() / 4;
            self.stream.memset_zeros(&mut s.slice.slice_mut(slot * n..(slot + 1) * n))?;
        }
        Ok(())
    }

    fn zero_mixer_state(&mut self, slot: usize) -> Result<()> {
        ensure!(self.mixer_state_live(slot), "recurrent state of slot {slot} is not allocated");
        let Some(mb) = self.mixer.as_mut() else { return Ok(()) };
        for s in &mut mb.states {
            let n = s.pool.stride_bytes() / 4;
            self.stream.memset_zeros(&mut s.slice.slice_mut(slot * n..(slot + 1) * n))?;
        }
        Ok(())
    }

    pub fn acquire_state(&mut self, slot: usize) -> Result<()> {
        ensure!(slot <= self.slots, "slot {slot} out of {}", self.slots);
        let need = self.state_slot_bytes();
        let free = self.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
        if self.mixer.is_some() {
            return self.acquire_mixer_state(slot, free);
        }
        let Some(gb) = self.gdn.as_mut() else { return Ok(()) };
        if gb.ssm_pool.is_live(slot) {
            return Ok(());
        }
        ensure!(
            free >= need + STATE_FREE_MARGIN,
            "{free} bytes of device memory free, a recurrent state slot needs {need} plus a margin"
        );
        gb.conv_pool.acquire(slot)?;
        if let Err(e) = gb.ssm_pool.acquire(slot) {
            gb.conv_pool.release(slot);
            return Err(e);
        }
        let conv = gb.conv_pool.stride_bytes() / 4;
        let ssm = gb.ssm_pool.stride_bytes() / 4;
        self.stream.memset_zeros(&mut gb.conv_state.slice_mut(slot * conv..(slot + 1) * conv))?;
        self.stream.memset_zeros(&mut gb.ssm_state.slice_mut(slot * ssm..(slot + 1) * ssm))?;
        Ok(())
    }

    pub fn release_state(&mut self, slot: usize) {
        if slot >= self.slots {
            return;
        }
        let _ = self.stream.synchronize();
        if let Some(gb) = self.gdn.as_mut() {
            gb.ssm_pool.release(slot);
            gb.conv_pool.release(slot);
        }
        if let Some(mb) = self.mixer.as_mut() {
            for s in &mut mb.states {
                s.pool.release(slot);
            }
        }
    }

    pub fn reset_slot(&mut self, seq: usize) -> Result<()> {
        ensure!(seq < self.slots, "slot {seq} out of {}", self.slots);
        self.mtp_forget(seq);
        if self.mixer.is_some() {
            return self.zero_mixer_state(seq);
        }
        if let Some(gb) = self.gdn.as_ref() {
            ensure!(gb.ssm_pool.is_live(seq), "recurrent state of slot {seq} is not allocated");
        }
        let slots = self.spec.linear_layers();
        let Some(gb) = self.gdn.as_mut() else { return Ok(()) };
        let conv = slots * gb.la.conv_dim() * (gb.la.conv_kernel - 1);
        let ssm = slots * gb.la.value_heads * gb.la.key_head_dim * gb.la.key_head_dim;
        self.stream.memset_zeros(&mut gb.conv_state.slice_mut(seq * conv..(seq + 1) * conv))?;
        self.stream.memset_zeros(&mut gb.ssm_state.slice_mut(seq * ssm..(seq + 1) * ssm))?;
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn plan_media(&mut self, prompt: &[u32], media: &[crate::sched::MediaInput]) -> Result<Arc<crate::vision::MediaPlan>> {
        let enc = self.vision.as_ref().context("this model was loaded without a vision encoder")?;
        Ok(Arc::new(enc.plan(prompt, media)?))
    }

    pub fn media_encoder(&self) -> Option<Arc<crate::media_encode::MediaEncoder>> {
        self.vision.clone()
    }

    pub fn set_media(&mut self, slot: usize, plan: Option<&crate::vision::MediaPlan>) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let Some(plan) = plan else {
            self.media[slot] = None;
            self.stream.memcpy_htod(&[0i32, 0], &mut self.mm_meta.slice_mut(2 * slot..2 * slot + 2))?;
            return Ok(());
        };
        ensure!(self.spec.rope.mrope.is_some() || self.spec.vision.as_ref().is_some_and(|v| v.config.pooled()), "the model has no mrope section in its spec");
        let n = plan.positions.len();
        ensure!(n <= self.max_ctx, "media prompt longer than max_ctx");
        let hidden = self.spec.dims.hidden;
        ensure!(plan.embeds.len() % hidden == 0, "media embeddings are not rows of {hidden}");
        let table: Vec<i32> = plan.positions.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        let base = slot * self.max_ctx * 3;
        if self.spec.rope.mrope.is_some() {
            self.stream.memcpy_htod(&table, &mut self.mm_table.slice_mut(base..base + table.len()))?;
        }
        self.stream.memcpy_htod(&[n as i32, plan.delta], &mut self.mm_meta.slice_mut(2 * slot..2 * slot + 2))?;
        if !plan.spans.is_empty() && self.mm_bidi.len() >= (slot + 1) * self.max_ctx {
            self.stream.memcpy_htod(&plan.spans, &mut self.mm_bidi.slice_mut(slot * self.max_ctx..slot * self.max_ctx + n))?;
        }
        let rows: Vec<i32> = plan.positions.iter().map(|p| p[3]).collect();
        let embeds = if self.role.first { self.stream.clone_htod(plan.embeds.as_slice())? } else { self.stream.alloc_zeros(1)? };
        self.media[slot] = Some(MediaDevice { embeds, rows });
        Ok(())
    }

    pub(super) fn splice_media(&self, slot: usize, start: usize, n: usize, row_offset: usize) -> Result<()> {
        if !self.role.first {
            return Ok(());
        }
        let Some(m) = self.media.get(slot).and_then(|m| m.as_ref()) else { return Ok(()) };
        let rows: Vec<i32> = (0..n).map(|i| m.rows.get(start + i).copied().unwrap_or(-1)).collect();
        if rows.iter().all(|&r| r < 0) {
            return Ok(());
        }
        let hidden = self.spec.dims.hidden;
        let rows_dev = self.stream.clone_htod(&rows)?;
        let x = self.pre.x.slice(row_offset * hidden..);
        let mut b = self.launcher(self.kernel("splice_rows"));
        b.arg(&x).arg(&rows_dev).arg(&m.embeds);
        unsafe { b.launch(LaunchConfig { grid_dim: (grid(hidden, 256), n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    pub fn last_logits(&self) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(&self.buf.logits)?)
    }

    pub fn linear_states(&self) -> Result<LinearStates> {
        let gb = self.gdn.as_ref().context("no linear layers")?;
        let layers = self.spec.linear_layers();
        let conv = self.slots * layers * gb.la.conv_dim() * (gb.la.conv_kernel - 1);
        let ssm = self.slots * layers * gb.la.value_heads * gb.la.key_head_dim * gb.la.key_head_dim;
        Ok(LinearStates {
            conv: self.stream.clone_dtoh(&gb.conv_state.slice(..conv))?,
            state: self.stream.clone_dtoh(&gb.ssm_state.slice(..ssm))?,
        })
    }

    pub fn trace_prefill(&mut self, tokens: &[u32]) -> Result<Trace> {
        ensure!(tokens.len() <= self.prefill_limit, "trace prefill fits one chunk");
        let s = &self.spec;
        let n = tokens.len();
        let layers = self.stream.alloc_zeros::<f32>(s.layers.count * n * s.dims.hidden)?;
        let logits = self.stream.alloc_zeros::<f32>(n * s.dims.vocab)?;
        self.reset_linear_state()?;
        self.prefill_chunk_traced(tokens, 0, Some(&logits), Some(&layers))?;
        Ok(Trace { layers: self.stream.clone_dtoh(&layers)?, logits: self.stream.clone_dtoh(&logits)? })
    }

    pub fn trace_decode(&mut self, tokens: &[u32]) -> Result<Trace> {
        let s = self.spec.clone();
        let per_step = s.layers.count * s.dims.hidden;
        let layers = self.stream.alloc_zeros::<f32>(per_step)?;
        self.reset_linear_state()?;
        let mut trace = Trace { layers: Vec::new(), logits: Vec::new() };
        for (pos, &t) in tokens.iter().enumerate() {
            self.set(t, pos)?;
            self.step_traced(Some(&layers))?;
            trace.layers.extend(self.stream.clone_dtoh(&layers)?);
            trace.logits.extend(self.stream.clone_dtoh(&self.buf.logits)?);
        }
        Ok(trace)
    }

    pub fn prefill_then_decode(&mut self, prompt: &[u32], decode: &[u32]) -> Result<Vec<f32>> {
        ensure!(prompt.len() + decode.len() <= self.max_ctx, "longer than max_ctx");
        self.reset_linear_state()?;
        self.prefill(prompt, false)?;
        let mut logits = self.stream.clone_dtoh(&self.buf.logits)?;
        for (i, &t) in decode.iter().enumerate() {
            self.set(t, prompt.len() + i)?;
            self.step()?;
            logits.extend(self.stream.clone_dtoh(&self.buf.logits)?);
        }
        Ok(logits)
    }

    fn batch_step(&self, bb: &BatchBuffers, n: usize) -> Result<()> {
        self.decode_rows(bb, n)?;
        let stride = self.max_ctx as i32;
        let mut b = self.launcher(self.kernel("argmax_rows"));
        b.arg(&bb.logits).arg(&bb.tokens).arg(&bb.history).arg(&self.buf.count).arg(&stride);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        let mut b = self.launcher(self.kernel("bump"));
        b.arg(&bb.pos).arg(&self.buf.count);
        unsafe { b.launch(cfg(1, n as u32)) }?;
        Ok(())
    }

    fn decode_rope_rows(&self, l: &DeviceAttn, bb: &BatchBuffers, n: usize) -> Result<()> {
        let a = &self.spec.attention;
        let (pages, mm_stride) = (self.pages as i32, self.max_ctx as i32);
        let pre = &self.pre;
        let buf = &self.buf;
        let mut b = self.launcher(self.kernel("qk_norm_rope"));
        b.arg(&pre.qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(buf.k_cache())
            .arg(buf.v_cache())
            .arg(&buf.block_table)
            .arg(&bb.pos)
            .arg(&bb.slots)
            .arg(&l.kv)
            .arg(&1i32)
            .arg(&pages)
            .arg(&mm_stride)
            .arg(&self.mm_meta)
            .arg(&self.mm_table)
            .arg(&0i32);
        let config = LaunchConfig {
            grid_dim: ((a.heads + 2 * a.kv_heads) as u32, n as u32, 1),
            block_dim: (a.head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn attn_decode_rows(&self, l: &DeviceAttn, bb: &BatchBuffers, n: usize, qlen: usize) -> Result<()> {
        let a = &self.spec.attention;
        debug_assert!(n.is_multiple_of(qlen) && qlen <= crate::codegen::DECODE_MAX_QLEN);
        let qlen_i = qlen as i32;
        let (pages, _) = (self.pages as i32, self.max_ctx as i32);
        let splits_i = crate::codegen::decode_splits(self.max_ctx) as i32;
        let pre = &self.pre;
        let buf = &self.buf;
        self.decode_rope_rows(l, bb, n)?;
        let mut b = self.launcher(self.kernel("batch_decode_attention"));
        b.arg(&pre.qkv)
            .arg(buf.k_cache())
            .arg(buf.v_cache())
            .arg(&buf.block_table)
            .arg(&bb.pos)
            .arg(&bb.slots)
            .arg(&l.kv)
            .arg(&qlen_i)
            .arg(&pages)
            .arg(&splits_i)
            .arg(&bb.part_acc)
            .arg(&bb.part_ml)
            .arg(&l.window);
        let config = LaunchConfig {
            grid_dim: (a.kv_heads as u32, (n / qlen) as u32, crate::codegen::decode_splits(self.max_ctx) as u32),
            block_dim: (32 * crate::codegen::decode_warps(a.heads / a.kv_heads, qlen) as u32, 1, 1),
            shared_mem_bytes: self.smem["batch_decode_attention"],
        };
        unsafe { b.launch(config) }?;
        let mut b = self.launcher(self.kernel("batch_decode_combine"));
        b.arg(&bb.part_acc).arg(&bb.part_ml).arg(&pre.qkv).arg(&bb.pos).arg(&qlen_i).arg(&splits_i).arg(&pre.attn_planes.hi).arg(&pre.attn_planes.lo);
        let config = LaunchConfig {
            grid_dim: (a.heads as u32, n as u32, 1),
            block_dim: (a.head_dim as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn decode_rows(&self, bb: &BatchBuffers, n: usize) -> Result<()> {
        let plan = &self.plans.decode;
        let run = exec::Run { n, tokens: &bb.tokens, logits: &bb.logits, rows: 0..n, bb: Some(bb), start: None, qlen: 1, mtp: None };
        crate::executor::run_plan(self, plan, &run, &mut |_| Ok(()))
    }

    fn batch_buffers(&self, rows: usize) -> Result<BatchBuffers> {
        let parts = rows * self.spec.attention.heads * crate::codegen::decode_splits(self.max_ctx);
        Ok(BatchBuffers {
            tokens: self.stream.alloc_zeros(rows)?,
            logits: self.stream.alloc_zeros(rows * self.spec.dims.vocab)?,
            history: self.stream.alloc_zeros(rows * self.max_ctx)?,
            part_acc: self.stream.alloc_zeros(parts * self.spec.attention.head_dim)?,
            part_ml: self.stream.alloc_zeros(parts * 2)?,
            pos: self.stream.alloc_zeros(rows)?,
            slots: self.stream.alloc_zeros(rows)?,
        })
    }

    pub fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        new_tokens: usize,
    ) -> Result<BatchGeneration> {
        ensure!(self.use_mma, "batch decode needs the tensor core path");
        ensure!(prompts.len() == self.batch, "expected {} prompts", self.batch);
        let len = prompts[0].len();
        ensure!(len > 0 && new_tokens > 0, "empty prompt or nothing to generate");
        ensure!(prompts.iter().all(|p| p.len() == len), "batch prompts must have the same length");
        ensure!(len + new_tokens <= self.max_ctx, "prompt plus generation exceeds max_ctx");
        let mut bb = self.batch_buffers(self.batch)?;
        let started = Instant::now();
        self.stream.memcpy_htod(&[0i32], &mut self.buf.count)?;
        for (seq, p) in prompts.iter().enumerate() {
            self.seq = seq as i32;
            let prefilled = self.reset_linear_state().and_then(|_| self.prefill(p, false));
            self.seq = 0;
            prefilled?;
            self.first_token(&bb, seq)?;
        }
        let slots: Vec<i32> = (0..self.batch as i32).collect();
        self.stream.memcpy_htod(&slots, &mut bb.slots)?;
        self.stream.memcpy_htod(&vec![len as i32; self.batch], &mut bb.pos)?;
        self.stream.memcpy_htod(&[1i32], &mut self.buf.count)?;
        self.stream.synchronize()?;
        let prefill_secs = started.elapsed().as_secs_f64();
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.batch_step(&bb, self.batch);
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        let graph = graph.context("empty graph")?;
        graph.upload()?;
        self.stream.synchronize()?;
        let started = Instant::now();
        for _ in 1..new_tokens {
            graph.launch()?;
        }
        self.stream.synchronize()?;
        let decode_secs = started.elapsed().as_secs_f64();
        let history = self.stream.clone_dtoh(&bb.history)?;
        let tokens = history
            .chunks(self.max_ctx)
            .map(|h| h[..new_tokens].iter().map(|&t| t as u32).collect())
            .collect();
        Ok(BatchGeneration { tokens, prefill_secs, decode_secs })
    }
}

const DECODE_CHUNK: usize = 16;

fn argmax_host(row: &[f32]) -> u32 {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }) as u32
}

pub fn cut_at_stop(tokens: &[u32], stop: &[u32]) -> (usize, bool) {
    match tokens.iter().position(|t| stop.contains(t)) {
        Some(i) => (i, true),
        None => (tokens.len(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::cut_at_stop;

    #[test]
    fn generation_is_cut_before_the_first_stop_token() {
        assert_eq!(cut_at_stop(&[5, 6, 151645, 9, 9], &[151645, 151643]), (2, true));
        assert_eq!(cut_at_stop(&[5, 6], &[151645]), (2, false));
        assert_eq!(cut_at_stop(&[151643], &[151645, 151643]), (0, true));
        assert_eq!(cut_at_stop(&[1, 2, 3], &[]), (3, false));
    }
}
