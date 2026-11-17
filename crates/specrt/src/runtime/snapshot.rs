use super::Runtime;
use crate::prefix::{SnapshotData, StateSnapshot};
use crate::hostpool::PinnedBuf;
use crate::sched::HostSeq;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, result};
use std::sync::{Arc, Mutex};

struct Parked {
    kv: PinnedBuf,
    state: Option<StateSnapshot>,
}

type HostPool = Arc<Mutex<Vec<PinnedHostSlice<f32>>>>;

#[derive(Default)]
pub(super) struct SnapshotPool {
    free: HostPool,
    staging: Option<CudaSlice<f32>>,
    side: Option<Arc<CudaStream>>,
    in_flight: Option<Arc<PinnedSnapshot>>,
}

struct PinnedSnapshot {
    host: Option<PinnedHostSlice<f32>>,
    event: CudaEvent,
    len: usize,
    pool: HostPool,
}

impl PinnedSnapshot {
    fn wait(&self) -> Result<()> {
        Ok(self.event.synchronize()?)
    }
}

impl SnapshotData for PinnedSnapshot {
    fn floats(&self) -> &[f32] {
        self.event.synchronize().expect("snapshot copy");
        self.host.as_ref().expect("host buffer").as_slice().expect("pinned slice")
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for PinnedSnapshot {
    fn drop(&mut self) {
        let _ = self.event.synchronize();
        if let Some(host) = self.host.take() {
            self.pool.lock().expect("pool").push(host);
        }
    }
}

impl Runtime {
    fn state_lens(&self) -> Result<(usize, usize)> {
        let gb = self.gdn.as_ref().context("no linear layers")?;
        let layers = self.spec.linear_layers();
        Ok((layers * gb.la.conv_dim() * (gb.la.conv_kernel - 1), layers * gb.la.value_heads * gb.la.key_head_dim * gb.la.key_head_dim))
    }

    fn pending_len(&self) -> usize {
        if self.mtp.is_some() { self.spec.dims.hidden } else { 0 }
    }

    fn ensure_snapshot_slot(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let gb = self.gdn.as_ref().context("no linear layers")?;
        ensure!(gb.ssm_pool.is_live(slot), "recurrent state of slot {slot} is not allocated");
        Ok(())
    }

    pub fn snapshot_state(&mut self, slot: usize) -> Result<StateSnapshot> {
        if self.spec.runtime.prefix_cache.async_snapshots {
            match self.snapshot_state_async(slot) {
                Ok(s) => return Ok(s),
                Err(e) => tracing::warn!(error = %format!("{e:#}"), "asynchronous state snapshot failed, copying synchronously"),
            }
        }
        self.snapshot_state_blocking(slot)
    }

    pub fn snapshot_state_blocking(&self, slot: usize) -> Result<StateSnapshot> {
        self.ensure_snapshot_slot(slot)?;
        let (conv, ssm) = self.state_lens()?;
        let gb = self.gdn.as_ref().context("no linear layers")?;
        self.stream.synchronize()?;
        let pending = if self.mtp.is_some() { self.mtp_pending_host(slot)? } else { Vec::new() };
        Ok(StateSnapshot::new(
            self.stream.clone_dtoh(&gb.conv_state.slice(slot * conv..(slot + 1) * conv))?,
            self.stream.clone_dtoh(&gb.ssm_state.slice(slot * ssm..(slot + 1) * ssm))?,
            pending,
        ))
    }

    pub fn snapshot_state_async(&mut self, slot: usize) -> Result<StateSnapshot> {
        self.ensure_snapshot_slot(slot)?;
        let (conv, ssm) = self.state_lens()?;
        let pending = self.pending_len();
        let total = conv + ssm + pending;
        if self.snap.staging.is_none() {
            self.snap.staging = Some(self.stream.alloc_zeros::<f32>(total).context("snapshot staging buffer")?);
        }
        if self.snap.side.is_none() {
            self.snap.side = Some(self.stream.fork()?);
        }
        if let Some(previous) = self.snap.in_flight.take() {
            previous.wait()?;
        }
        let recycled = self.snap.free.lock().expect("pool").pop();
        let mut host = match recycled {
            Some(h) => h,
            None => unsafe { self.ctx.alloc_pinned_with_flags::<f32>(total, 0) }.context("pinned snapshot buffer")?,
        };
        let mut staging = self.snap.staging.take().expect("staging");
        let gb = self.gdn.as_ref().context("no linear layers")?;
        self.stream.memcpy_dtod(&gb.conv_state.slice(slot * conv..(slot + 1) * conv), &mut staging.slice_mut(..conv))?;
        self.stream.memcpy_dtod(&gb.ssm_state.slice(slot * ssm..(slot + 1) * ssm), &mut staging.slice_mut(conv..conv + ssm))?;
        if pending > 0 {
            let mut dst = staging.slice_mut(conv + ssm..total);
            self.mtp_pending_to(slot, &mut dst)?;
        }
        let ready = self.stream.record_event(None)?;
        let side = self.snap.side.clone().expect("side stream");
        side.wait(&ready)?;
        let (src, _guard) = staging.device_ptr(&side);
        let dst = host.as_mut_slice()?;
        unsafe { result::memcpy_dtoh_async(&mut dst[..total], src, side.cu_stream()) }?;
        drop(_guard);
        let event = side.record_event(None)?;
        self.snap.staging = Some(staging);
        let data = Arc::new(PinnedSnapshot { host: Some(host), event, len: total, pool: self.snap.free.clone() });
        self.snap.in_flight = Some(data.clone());
        Ok(StateSnapshot::shared(data, conv, ssm))
    }

    pub fn restore_state(&mut self, slot: usize, snap: &StateSnapshot) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        let (conv, ssm) = self.state_lens()?;
        let pending = self.pending_len();
        ensure!(
            snap.conv().len() == conv && snap.ssm().len() == ssm && snap.pending().len() == pending,
            "snapshot does not match this model's recurrent state"
        );
        self.acquire_state(slot)?;
        let gb = self.gdn.as_mut().context("no linear layers")?;
        self.stream.memcpy_htod(snap.conv(), &mut gb.conv_state.slice_mut(slot * conv..(slot + 1) * conv))?;
        self.stream.memcpy_htod(snap.ssm(), &mut gb.ssm_state.slice_mut(slot * ssm..(slot + 1) * ssm))?;
        if pending > 0 {
            self.mtp_pending_from(slot, snap.pending())?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn copy_kv_block(&mut self, src: u32, dst: u32) -> Result<()> {
        ensure!((src as usize) <= self.kv_blocks && (dst as usize) <= self.kv_blocks, "KV block out of range");
        let elems = super::kv_block_bytes(&self.spec, self.kv_planes) / 4;
        let bytes = elems * std::mem::size_of::<u16>();
        for cache in [&self.buf.k_cache, &self.buf.v_cache] {
            let (base, _guard) = cache.device_ptr(&self.stream);
            unsafe {
                result::memcpy_dtod_async(base + (dst as usize * bytes) as u64, base + (src as usize * bytes) as u64, bytes, self.stream.cu_stream())
            }?;
        }
        Ok(())
    }

    fn kv_cache_block_bytes(&self) -> usize {
        super::kv_block_bytes(&self.spec, self.kv_planes) / 4 * std::mem::size_of::<u16>()
    }

    pub fn swap_bytes(&self, blocks: usize) -> usize {
        let state = if self.gdn.is_some() { self.state_lens().map_or(0, |(c, s)| (c + s + self.pending_len()) * 4) } else { 0 };
        blocks * 2 * self.kv_cache_block_bytes() + state
    }

    pub fn park(&mut self, slot: usize, blocks: &[u32]) -> Result<HostSeq> {
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let state = if self.gdn.is_some() { Some(self.snapshot_state_blocking(slot)?) } else { None };
        let per = self.kv_cache_block_bytes();
        let total = blocks.len() * 2 * per;
        let mut host = self.host.alloc(total)?;
        if self.snap.side.is_none() {
            self.snap.side = Some(self.stream.fork()?);
        }
        let side = self.snap.side.clone().expect("side stream");
        let ready = self.stream.record_event(None)?;
        side.wait(&ready)?;
        {
            let dst = host.as_mut_slice()?;
            for (c, cache) in [&self.buf.k_cache, &self.buf.v_cache].into_iter().enumerate() {
                let (base, _guard) = cache.device_ptr(&side);
                for (i, &b) in blocks.iter().enumerate() {
                    let at = (c * blocks.len() + i) * per;
                    unsafe { result::memcpy_dtoh_async(&mut dst[at..at + per], base + (b as usize * per) as u64, side.cu_stream()) }?;
                }
            }
        }
        side.synchronize()?;
        let bytes = total + state.as_ref().map_or(0, |s| (s.conv().len() + s.ssm().len() + s.pending().len()) * 4);
        Ok(HostSeq { bytes, blocks: blocks.len(), payload: Box::new(Parked { kv: host, state }) })
    }

    pub fn resume(&mut self, slot: usize, blocks: &[u32], seq: &HostSeq) -> Result<()> {
        let parked = seq.payload.downcast_ref::<Parked>().context("not a parked sequence of this runtime")?;
        ensure!(blocks.len() == seq.blocks, "{} blocks for a parked sequence of {}", blocks.len(), seq.blocks);
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let per = self.kv_cache_block_bytes();
        let src = parked.kv.as_slice()?;
        self.stream.synchronize()?;
        for (c, cache) in [&self.buf.k_cache, &self.buf.v_cache].into_iter().enumerate() {
            let (base, _guard) = cache.device_ptr(&self.stream);
            for (i, &b) in blocks.iter().enumerate() {
                let at = (c * blocks.len() + i) * per;
                unsafe { result::memcpy_htod_async(base + (b as usize * per) as u64, &src[at..at + per], self.stream.cu_stream()) }?;
            }
        }
        self.stream.synchronize()?;
        if let Some(state) = &parked.state {
            self.restore_state(slot, state)?;
        }
        Ok(())
    }

    pub fn session_compat(&self) -> crate::sessions::Compat {
        crate::sessions::compat_for(&self.spec, self.kv_planes)
    }

    pub fn save_kv_blocks(&mut self, blocks: &[u32]) -> Result<(Vec<u8>, Vec<u8>)> {
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let per = self.kv_cache_block_bytes();
        let total = blocks.len() * per;
        let mut host = self.host.alloc(total * 2)?;
        if self.snap.side.is_none() {
            self.snap.side = Some(self.stream.fork()?);
        }
        let side = self.snap.side.clone().expect("side stream");
        let ready = self.stream.record_event(None)?;
        side.wait(&ready)?;
        let mut out = Vec::with_capacity(2);
        for (c, cache) in [&self.buf.k_cache, &self.buf.v_cache].into_iter().enumerate() {
            let (base, _guard) = cache.device_ptr(&side);
            let dst = host.as_mut_slice()?;
            for (i, &b) in blocks.iter().enumerate() {
                let at = c * total + i * per;
                unsafe { result::memcpy_dtoh_async(&mut dst[at..at + per], base + (b as usize * per) as u64, side.cu_stream()) }?;
            }
        }
        side.synchronize()?;
        let all = host.as_slice()?;
        out.push(all[..total].to_vec());
        out.push(all[total..].to_vec());
        let v = out.pop().expect("v");
        let k = out.pop().expect("k");
        Ok((k, v))
    }

    pub fn load_kv_blocks(&mut self, blocks: &[u32], k: &[u8], v: &[u8]) -> Result<()> {
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let per = self.kv_cache_block_bytes();
        ensure!(k.len() == blocks.len() * per && v.len() == k.len(), "session KV is {} bytes, expected {}", k.len(), blocks.len() * per);
        for (cache, data) in [(&self.buf.k_cache, k), (&self.buf.v_cache, v)] {
            let (base, _guard) = cache.device_ptr(&self.stream);
            for (i, &b) in blocks.iter().enumerate() {
                unsafe { result::memcpy_htod_async(base + (b as usize * per) as u64, &data[i * per..(i + 1) * per], self.stream.cu_stream()) }?;
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }
}
