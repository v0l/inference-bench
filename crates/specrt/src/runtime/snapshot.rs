use super::Runtime;
use crate::prefix::{SnapshotData, StateSnapshot};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, result};
use std::sync::{Arc, Mutex};

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
}
