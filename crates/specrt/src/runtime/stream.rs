use super::{DeviceAttn, DeviceDescribed, DeviceGdn, DeviceLayer, DeviceMatrix, DeviceMixer};
use crate::hostpool::{HostPool, PinnedBuf};
use crate::diskio::{Appender, IoPool, ReadAt, Ticket, round_up};
use crate::offload::OffloadStats;
use anyhow::Result;
use cudarc::driver::sys::{CUdeviceptr, CUevent_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::sync::{Arc, Mutex};

const ALIGN: usize = 256;

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

pub(super) trait Walk {
    fn slice<T: DeviceRepr + ValidAsZeroBits>(&mut self, s: &mut CudaSlice<T>) -> Result<()>;
}

fn opt<W: Walk>(w: &mut W, s: &mut Option<CudaSlice<f32>>) -> Result<()> {
    if let Some(s) = s {
        w.slice(s)?;
    }
    Ok(())
}

fn extra<W: Walk>(w: &mut W, m: &mut std::collections::BTreeMap<String, CudaSlice<f32>>) -> Result<()> {
    for s in m.values_mut() {
        w.slice(s)?;
    }
    Ok(())
}

impl DeviceMatrix {
    fn walk<W: Walk>(&mut self, w: &mut W) -> Result<()> {
        match self {
            DeviceMatrix::Bf16(a) => w.slice(a),
            DeviceMatrix::Q8 { values, scales } => w.slice(values).and_then(|_| w.slice(scales)),
            DeviceMatrix::Fp8 { values, scales } => w.slice(values).and_then(|_| w.slice(scales)),
            DeviceMatrix::Fp4 { packed, scales } => w.slice(packed).and_then(|_| w.slice(scales)),
            DeviceMatrix::Ggml { data, table } => w.slice(data).and_then(|_| w.slice(table)),
        }
    }
}

impl DeviceAttn {
    fn walk<W: Walk>(&mut self, w: &mut W) -> Result<()> {
        self.qkv.walk(w)?;
        opt(w, &mut self.qkv_bias)?;
        opt(w, &mut self.o_bias)?;
        opt(w, &mut self.q_norm)?;
        opt(w, &mut self.k_norm)?;
        self.o.walk(w)?;
        extra(w, &mut self.extra)
    }
}

impl DeviceGdn {
    fn walk<W: Walk>(&mut self, w: &mut W) -> Result<()> {
        self.in_proj.walk(w)?;
        self.ba.walk(w)?;
        w.slice(&mut self.conv)?;
        w.slice(&mut self.neg_a)?;
        w.slice(&mut self.dt_bias)?;
        w.slice(&mut self.norm)?;
        self.out.walk(w)
    }
}

impl DeviceDescribed {
    fn walk<W: Walk>(&mut self, w: &mut W) -> Result<()> {
        self.in_proj.walk(w)?;
        self.out.walk(w)?;
        extra(w, &mut self.extra)
    }
}

impl DeviceLayer {
    pub(super) fn walk<W: Walk>(&mut self, w: &mut W) -> Result<()> {
        w.slice(&mut self.ln1)?;
        w.slice(&mut self.ln2)?;
        opt(w, &mut self.ln_pre_mlp)?;
        opt(w, &mut self.ln_post_mlp)?;
        match &mut self.mixer {
            DeviceMixer::Attention(a) => a.walk(w)?,
            DeviceMixer::Linear(g) => g.walk(w)?,
            DeviceMixer::Described(d) => d.walk(w)?,
            DeviceMixer::None => {}
        }
        self.gate_up.walk(w)?;
        self.down.walk(w)?;
        if self.moe.as_ref().is_some_and(|m| !m.is_cached()) {
            anyhow::bail!("offload streams dense layers; a routed layer needs the expert cache");
        }
        Ok(())
    }
}

struct Measure(usize);

impl Walk for Measure {
    fn slice<T: DeviceRepr + ValidAsZeroBits>(&mut self, s: &mut CudaSlice<T>) -> Result<()> {
        self.0 += aligned(s.num_bytes());
        Ok(())
    }
}

struct Spill<'a> {
    stream: &'a Arc<CudaStream>,
    blob: &'a mut [u8],
    at: usize,
    lens: Vec<usize>,
}

impl Walk for Spill<'_> {
    fn slice<T: DeviceRepr + ValidAsZeroBits>(&mut self, s: &mut CudaSlice<T>) -> Result<()> {
        let bytes = s.num_bytes();
        let (ptr, _sync) = s.device_ptr(self.stream);
        unsafe { cudarc::driver::result::memcpy_dtoh_async(&mut self.blob[self.at..self.at + bytes], ptr, self.stream.cu_stream()) }?;
        drop(_sync);
        self.stream.synchronize()?;
        self.at += aligned(bytes);
        self.lens.push(bytes);
        *s = self.stream.alloc_zeros::<T>(1)?;
        Ok(())
    }
}

struct Alias<'a> {
    stream: &'a Arc<CudaStream>,
    base: CUdeviceptr,
    at: usize,
    lens: std::slice::Iter<'a, usize>,
}

impl Walk for Alias<'_> {
    fn slice<T: DeviceRepr + ValidAsZeroBits>(&mut self, s: &mut CudaSlice<T>) -> Result<()> {
        let bytes = *self.lens.next().expect("spilled layout matches the layer");
        let aliased = unsafe { self.stream.upgrade_device_ptr::<T>(self.base + self.at as u64, bytes / std::mem::size_of::<T>()) };
        self.at += aligned(bytes);
        drop(std::mem::replace(s, aliased));
        Ok(())
    }
}

struct Detach<'a>(&'a Arc<CudaStream>);

impl Walk for Detach<'_> {
    fn slice<T: DeviceRepr + ValidAsZeroBits>(&mut self, s: &mut CudaSlice<T>) -> Result<()> {
        let held = std::mem::replace(s, self.0.alloc_zeros::<T>(1)?);
        held.leak();
        Ok(())
    }
}

pub(super) enum Blob {
    Host(PinnedBuf),
    Disk { offset: u64, len: usize },
}

impl Blob {
    fn len(&self) -> usize {
        match self {
            Blob::Host(b) => b.len(),
            Blob::Disk { len, .. } => *len,
        }
    }
}

pub(super) struct Spilled {
    pub layer: usize,
    pub blob: Blob,
    pub lens: Vec<usize>,
}

pub(super) struct DiskTier {
    file: Arc<dyn ReadAt>,
    pool: IoPool,
    path: std::path::PathBuf,
}

impl DiskTier {
    pub(super) fn new(file: Arc<dyn ReadAt>, path: std::path::PathBuf) -> Self {
        DiskTier { file, pool: IoPool::new(4), path }
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(super) fn spill(stream: &Arc<CudaStream>, pool: &HostPool, layer: usize, l: &mut DeviceLayer) -> Result<Spilled> {
    let mut m = Measure(0);
    l.walk(&mut m)?;
    let mut blob = pool.alloc(m.0)?;
    let mut sp = Spill { stream, blob: blob.as_mut_slice()?, at: 0, lens: Vec::new() };
    l.walk(&mut sp)?;
    let lens = sp.lens;
    Ok(Spilled { layer, blob: Blob::Host(blob), lens })
}

pub(super) fn spill_to_disk(stream: &Arc<CudaStream>, pool: &HostPool, writer: &mut Appender, layer: usize, l: &mut DeviceLayer) -> Result<Spilled> {
    let s = spill(stream, pool, layer, l)?;
    let Blob::Host(buf) = &s.blob else { unreachable!() };
    let len = buf.len();
    let offset = writer.append(buf.as_slice()?)?;
    Ok(Spilled { layer, blob: Blob::Disk { offset, len }, lens: s.lens })
}

struct Slot {
    memory: CudaSlice<u8>,
    copied: CudaEvent,
    released: CudaEvent,
    holds: Option<usize>,
    staging: Option<PinnedBuf>,
    pending: Option<(Ticket, usize)>,
}

struct State {
    slots: Vec<Slot>,
    stats: OffloadStats,
}

pub(super) struct Streamer {
    copy: Arc<CudaStream>,
    order: Vec<usize>,
    position: Vec<Option<usize>>,
    blobs: Vec<Option<(Blob, Vec<usize>)>>,
    disk: Option<DiskTier>,
    depth: usize,
    state: Mutex<State>,
}

impl Streamer {
    pub(super) fn new(compute: &Arc<CudaStream>, layers: &mut [DeviceLayer], spilled: Vec<Spilled>, depth: usize, disk: Option<DiskTier>, pool: &HostPool) -> Result<Self> {
        let ctx = compute.context().clone();
        let copy = ctx.new_stream()?;
        let mut order: Vec<usize> = spilled.iter().map(|s| s.layer).collect();
        order.sort_unstable();
        let mut position = vec![None; layers.len()];
        for (k, &l) in order.iter().enumerate() {
            position[l] = Some(k);
        }
        let mut blobs: Vec<Option<(Blob, Vec<usize>)>> = (0..layers.len()).map(|_| None).collect();
        let mut slot_bytes = 0;
        for s in spilled {
            slot_bytes = slot_bytes.max(s.blob.len());
            blobs[s.layer] = Some((s.blob, s.lens));
        }
        let count = (depth + 1).min(order.len()).max(1);
        let mut slots = Vec::new();
        let any_disk = blobs.iter().flatten().any(|(b, _)| matches!(b, Blob::Disk { .. }));
        for _ in 0..count {
            slots.push(Slot {
                staging: if any_disk { Some(pool.alloc(round_up(slot_bytes))?) } else { None },
                pending: None,
                memory: compute.alloc_zeros::<u8>(slot_bytes.max(1))?,
                copied: ctx.new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                released: ctx.new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                holds: None,
            });
        }
        for &l in &order {
            let k = position[l].expect("streamed layer has a position");
            let (_, lens) = blobs[l].as_ref().expect("blob stored");
            let (ptr, _sync) = slots[k % count].memory.device_ptr(compute);
            let mut alias = Alias { stream: compute, base: ptr, at: 0, lens: lens.iter() };
            layers[l].walk(&mut alias)?;
        }
        Ok(Streamer { copy, order, position, blobs, disk, depth, state: Mutex::new(State { slots, stats: OffloadStats::default() }) })
    }

    pub(super) fn slot_bytes(&self) -> u64 {
        let g = self.state.lock().unwrap();
        g.slots.iter().map(|s| s.memory.num_bytes() as u64).sum()
    }

    pub(super) fn streamed_layers(&self) -> usize {
        self.order.len()
    }

    pub(super) fn stats(&self) -> OffloadStats {
        self.state.lock().unwrap().stats
    }

    fn issue(&self, g: &mut State, layer: usize, slot: usize) -> Result<()> {
        let (blob, _) = self.blobs[layer].as_ref().expect("streamed layer has a blob");
        let s = &mut g.slots[slot];
        match blob {
            Blob::Host(buf) => {
                if let Some((stale, _)) = s.pending.take() {
                    stale.wait()?;
                }
                self.copy.wait(&s.released)?;
                let (ptr, _sync) = s.memory.device_ptr(&self.copy);
                let src = buf.as_slice()?;
                unsafe { cudarc::driver::result::memcpy_htod_async(ptr, src, self.copy.cu_stream()) }?;
                drop(_sync);
                s.copied.record(&self.copy)?;
                g.stats.host_to_vram_bytes += src.len() as u64;
            }
            Blob::Disk { offset, len } => {
                let disk = self.disk.as_ref().expect("disk layers have a disk tier");
                if let Some((stale, _)) = s.pending.take() {
                    stale.wait()?;
                }
                s.copied.synchronize()?;
                let staging = s.staging.as_mut().expect("disk layers have staging");
                let dst = staging.as_mut_slice()?.as_mut_ptr();
                s.pending = Some((unsafe { disk.pool.read(&disk.file, *offset, dst, round_up(*len)) }, *len));
                g.stats.nvme_reads += 1;
                g.stats.nvme_read_bytes += *len as u64;
            }
        }
        s.holds = Some(layer);
        Ok(())
    }

    fn finish(&self, g: &mut State, slot: usize, block: bool) -> Result<()> {
        let s = &mut g.slots[slot];
        let Some((ticket, len)) = s.pending.take() else { return Ok(()) };
        if !block && !ticket.is_done() {
            s.pending = Some((ticket, len));
            return Ok(());
        }
        ticket.wait()?;
        self.copy.wait(&s.released)?;
        let (ptr, _sync) = s.memory.device_ptr(&self.copy);
        let src = &s.staging.as_ref().expect("staging").as_slice()?[..len];
        unsafe { cudarc::driver::result::memcpy_htod_async(ptr, src, self.copy.cu_stream()) }?;
        drop(_sync);
        s.copied.record(&self.copy)?;
        g.stats.host_to_vram_bytes += len as u64;
        Ok(())
    }

    pub(super) fn enter(&self, layer: usize, compute: &Arc<CudaStream>) -> Result<()> {
        let mut g = self.state.lock().unwrap();
        let Some(k) = self.position[layer] else {
            g.stats.vram_hits += 1;
            return Ok(());
        };
        let count = g.slots.len();
        let slot = k % count;
        if g.slots[slot].holds == Some(layer) {
            g.stats.vram_hits += 1;
        } else {
            g.stats.host_hits += 1;
            self.issue(&mut g, layer, slot)?;
        }
        self.finish(&mut g, slot, true)?;
        compute.wait(&g.slots[slot].copied)?;
        let len = self.order.len();
        for d in 1..=self.depth.min(len.saturating_sub(1)) {
            let tp = (k + d) % len;
            let next = self.order[tp];
            let next_slot = tp % count;
            let holder = g.slots[next_slot].holds;
            if holder == Some(next) || holder == Some(layer) {
                continue;
            }
            self.issue(&mut g, next, next_slot)?;
        }
        for k in 0..g.slots.len() {
            self.finish(&mut g, k, false)?;
        }
        Ok(())
    }

    pub(super) fn leave(&self, layer: usize, compute: &Arc<CudaStream>) -> Result<()> {
        let Some(k) = self.position[layer] else { return Ok(()) };
        let g = self.state.lock().unwrap();
        g.slots[k % g.slots.len()].released.record(compute)?;
        Ok(())
    }

    pub(super) fn detach(&self, layers: &mut [DeviceLayer], compute: &Arc<CudaStream>) {
        let _ = compute.synchronize();
        let _ = self.copy.synchronize();
        for &l in &self.order {
            let _ = layers[l].walk(&mut Detach(compute));
        }
    }
}
