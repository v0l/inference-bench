use super::Runtime;
use crate::prefix::{SnapshotData, StateSnapshot};
use crate::hostpool::PinnedBuf;
use crate::sched::HostSeq;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, result};
use std::sync::{Arc, Mutex};

struct Parked {
    kv: Kv,
    state: Option<StateSnapshot>,
}

enum Kv {
    Pinned(PinnedBuf),
    Disk(crate::diskio::SwapFile),
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

    fn mtp_len(&self) -> usize {
        if self.mtp.is_some() { self.mtp_width() } else { 0 }
    }

    fn pending_len(&self) -> usize {
        self.mtp_len() + self.prelude_floats()
    }

    fn pending_host(&self, slot: usize) -> Result<Vec<f32>> {
        let mut out = if self.mtp.is_some() { self.mtp_pending_host(slot)? } else { Vec::new() };
        out.extend(self.prelude_host(slot)?);
        Ok(out)
    }

    fn pending_from(&mut self, slot: usize, host: &[f32]) -> Result<()> {
        let m = self.mtp_len();
        if m > 0 {
            self.mtp_pending_from(slot, &host[..m])?;
        }
        self.prelude_from(slot, &host[m..])
    }

    fn ensure_snapshot_slot(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        if self.mixer.is_some() {
            ensure!(self.mixer_state_live(slot), "recurrent state of slot {slot} is not allocated");
            return Ok(());
        }
        let gb = self.gdn.as_ref().context("no linear layers")?;
        ensure!(gb.ssm_pool.is_live(slot), "recurrent state of slot {slot} is not allocated");
        Ok(())
    }

    pub(super) fn has_seq_state(&self) -> bool {
        self.gdn.is_some() || self.mixer.is_some()
    }

    fn mixer_lens(&self) -> Vec<usize> {
        self.mixer.iter().flat_map(|mb| mb.states.iter().map(|s| s.pool.stride_bytes() / 4)).collect()
    }

    fn mixer_snapshot(&self, slot: usize) -> Result<StateSnapshot> {
        self.ensure_snapshot_slot(slot)?;
        let mb = self.mixer.as_ref().context("no mixer")?;
        self.stream.synchronize()?;
        let mut parts = Vec::new();
        for s in &mb.states {
            let n = s.pool.stride_bytes() / 4;
            parts.push(self.stream.clone_dtoh(&s.slice.slice(slot * n..(slot + 1) * n))?);
        }
        let first = if parts.is_empty() { Vec::new() } else { parts.remove(0) };
        Ok(StateSnapshot::new(first, parts.concat(), Vec::new()))
    }

    fn mixer_restore(&mut self, slot: usize, snap: &StateSnapshot) -> Result<()> {
        let lens = self.mixer_lens();
        let first = lens.first().copied().unwrap_or(0);
        let rest: usize = lens.iter().skip(1).sum();
        ensure!(snap.conv().len() == first && snap.ssm().len() == rest && snap.pending().is_empty(), "snapshot does not match this model's recurrent state");
        self.acquire_state(slot)?;
        let mb = self.mixer.as_mut().context("no mixer")?;
        let mut at = 0;
        for (i, s) in mb.states.iter_mut().enumerate() {
            let n = s.pool.stride_bytes() / 4;
            let src = if i == 0 { snap.conv() } else { &snap.ssm()[at..at + n] };
            if i > 0 {
                at += n;
            }
            self.stream.memcpy_htod(src, &mut s.slice.slice_mut(slot * n..(slot + 1) * n))?;
        }
        self.stream.synchronize()?;
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
        if self.mixer.is_some() {
            return self.mixer_snapshot(slot);
        }
        self.ensure_snapshot_slot(slot)?;
        let (conv, ssm) = self.state_lens()?;
        let gb = self.gdn.as_ref().context("no linear layers")?;
        self.stream.synchronize()?;
        let pending = self.pending_host(slot)?;
        Ok(StateSnapshot::new(
            self.stream.clone_dtoh(&gb.conv_state.slice(slot * conv..(slot + 1) * conv))?,
            self.stream.clone_dtoh(&gb.ssm_state.slice(slot * ssm..(slot + 1) * ssm))?,
            pending,
        ))
    }

    pub fn snapshot_state_async(&mut self, slot: usize) -> Result<StateSnapshot> {
        if self.mixer.is_some() {
            return self.mixer_snapshot(slot);
        }
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
        let mtp = self.mtp_len();
        if mtp > 0 {
            let mut dst = staging.slice_mut(conv + ssm..conv + ssm + mtp);
            self.mtp_pending_to(slot, &mut dst)?;
        }
        if pending > mtp {
            let mut dst = staging.slice_mut(conv + ssm + mtp..total);
            self.prelude_to(slot, &mut dst)?;
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
        if self.mixer.is_some() {
            return self.mixer_restore(slot, snap);
        }
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
            self.pending_from(slot, snap.pending())?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    fn cache_block_bytes(&self) -> Vec<usize> {
        crate::state::positional(&self.spec).map(|s| s.block_bytes(&self.spec, self.kv_planes)).collect()
    }

    fn kv_bytes_per_block(&self) -> usize {
        self.cache_block_bytes().iter().sum()
    }

    pub fn copy_kv_block(&mut self, src: u32, dst: u32) -> Result<()> {
        ensure!((src as usize) <= self.kv_blocks && (dst as usize) <= self.kv_blocks, "KV block out of range");
        for (cache, bytes) in self.buf.caches.iter().zip(self.cache_block_bytes()) {
            let (base, _guard) = cache.device_ptr(&self.stream);
            unsafe {
                result::memcpy_dtod_async(base + (dst as usize * bytes) as u64, base + (src as usize * bytes) as u64, bytes, self.stream.cu_stream())
            }?;
        }
        Ok(())
    }

    pub fn swap_bytes(&self, blocks: usize) -> usize {
        let state = if self.mixer.is_some() {
            self.mixer_lens().iter().sum::<usize>() * 4
        } else if self.gdn.is_some() {
            self.state_lens().map_or(0, |(c, s)| (c + s + self.pending_len()) * 4)
        } else {
            0
        };
        blocks * self.kv_bytes_per_block() + state
    }

    fn blocks_to_host(&mut self, blocks: &[u32], dst: &mut [u8]) -> Result<()> {
        if self.snap.side.is_none() {
            self.snap.side = Some(self.stream.fork()?);
        }
        let side = self.snap.side.clone().expect("side stream");
        let ready = self.stream.record_event(None)?;
        side.wait(&ready)?;
        let mut at = 0;
        for (cache, per) in self.buf.caches.iter().zip(self.cache_block_bytes()) {
            let (base, _guard) = cache.device_ptr(&side);
            for &b in blocks {
                unsafe { result::memcpy_dtoh_async(&mut dst[at..at + per], base + (b as usize * per) as u64, side.cu_stream()) }?;
                at += per;
            }
        }
        side.synchronize()?;
        Ok(())
    }

    fn blocks_from_host(&mut self, blocks: &[u32], src: &[u8]) -> Result<()> {
        let mut at = 0;
        for (cache, per) in self.buf.caches.iter().zip(self.cache_block_bytes()) {
            let (base, _guard) = cache.device_ptr(&self.stream);
            for &b in blocks {
                unsafe { result::memcpy_htod_async(base + (b as usize * per) as u64, &src[at..at + per], self.stream.cu_stream()) }?;
                at += per;
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn park(&mut self, slot: usize, blocks: &[u32]) -> Result<HostSeq> {
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let state = if self.has_seq_state() { Some(self.snapshot_state_blocking(slot)?) } else { None };
        let total = blocks.len() * self.kv_bytes_per_block();
        if let Some(plan) = &self.offload_plan {
            plan.check_swap_fits(self.spec.runtime.offload.host_budget, self.host.used() as u64, total as u64)?;
        }
        let mut host = self.host.alloc(total)?;
        self.blocks_to_host(blocks, host.as_mut_slice()?)?;
        let kv = match self.spec.runtime.offload.nvme_path.as_deref() {
            Some(dir) => Kv::Disk(crate::diskio::SwapFile::write(std::path::Path::new(dir), host.as_slice()?)?),
            None => Kv::Pinned(host),
        };
        let bytes = total + state.as_ref().map_or(0, |s| (s.conv().len() + s.ssm().len() + s.pending().len()) * 4);
        Ok(HostSeq { bytes, blocks: blocks.len(), payload: Box::new(Parked { kv, state }) })
    }

    pub fn resume(&mut self, slot: usize, blocks: &[u32], seq: &HostSeq) -> Result<()> {
        let parked = seq.payload.downcast_ref::<Parked>().context("not a parked sequence of this runtime")?;
        ensure!(blocks.len() == seq.blocks, "{} blocks for a parked sequence of {}", blocks.len(), seq.blocks);
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        self.stream.synchronize()?;
        match &parked.kv {
            Kv::Pinned(host) => self.blocks_from_host(blocks, host.as_slice()?)?,
            Kv::Disk(file) => {
                let mut staging = self.host.alloc(crate::diskio::round_up(file.len()) + crate::diskio::BLOCK)?;
                let raw = staging.as_mut_slice()?;
                let skew = raw.as_ptr() as usize % crate::diskio::BLOCK;
                let start = if skew == 0 { 0 } else { crate::diskio::BLOCK - skew };
                file.read_into(&mut raw[start..])?;
                self.blocks_from_host(blocks, &staging.as_slice()?[start..start + file.len()])?;
            }
        }
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
        let total = blocks.len() * self.kv_bytes_per_block();
        let first = blocks.len() * self.cache_block_bytes().first().copied().unwrap_or(0);
        let mut host = self.host.alloc(total)?;
        self.blocks_to_host(blocks, host.as_mut_slice()?)?;
        let all = host.as_slice()?;
        Ok((all[..first].to_vec(), all[first..total].to_vec()))
    }

    pub fn load_kv_blocks(&mut self, blocks: &[u32], k: &[u8], v: &[u8]) -> Result<()> {
        ensure!(blocks.iter().all(|&b| (b as usize) < self.kv_blocks), "KV block out of range");
        let total = blocks.len() * self.kv_bytes_per_block();
        ensure!(k.len() + v.len() == total, "session KV is {} bytes, expected {total}", k.len() + v.len());
        let mut host = self.host.alloc(total)?;
        {
            let dst = host.as_mut_slice()?;
            dst[..k.len()].copy_from_slice(k);
            dst[k.len()..total].copy_from_slice(v);
        }
        self.blocks_from_host(blocks, host.as_slice()?)
    }
}
