use anyhow::{Result, ensure};
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostSlice, ValidAsZeroBits};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkMode {
    Peer,
    Staged,
}

pub struct DeviceLink {
    src: Arc<CudaContext>,
    dst: Arc<CudaContext>,
    mode: LinkMode,
}

pub fn peer_access(src: &Arc<CudaContext>, dst: &Arc<CudaContext>) -> bool {
    if src.ordinal() == dst.ordinal() {
        return true;
    }
    let mut can = 0;
    let r = unsafe { sys::cuDeviceCanAccessPeer(&mut can, dst.cu_device(), src.cu_device()) };
    r == sys::CUresult::CUDA_SUCCESS && can != 0
}

impl DeviceLink {
    pub fn new(src: &Arc<CudaContext>, dst: &Arc<CudaContext>) -> Result<Self> {
        let mode = if peer_access(src, dst) { LinkMode::Peer } else { LinkMode::Staged };
        Self::with_mode(src, dst, mode)
    }

    pub fn with_mode(src: &Arc<CudaContext>, dst: &Arc<CudaContext>, mode: LinkMode) -> Result<Self> {
        if mode == LinkMode::Peer && src.ordinal() != dst.ordinal() {
            ensure!(peer_access(src, dst), "no peer access from device {} to {}", src.ordinal(), dst.ordinal());
            for (from, to) in [(dst, src), (src, dst)] {
                from.bind_to_thread()?;
                let r = unsafe { sys::cuCtxEnablePeerAccess(to.cu_ctx(), 0) };
                ensure!(
                    r == sys::CUresult::CUDA_SUCCESS || r == sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
                    "enable peer access {} to {}: {r:?}",
                    from.ordinal(),
                    to.ordinal()
                );
            }
        }
        Ok(DeviceLink { src: src.clone(), dst: dst.clone(), mode })
    }

    pub fn mode(&self) -> LinkMode {
        self.mode
    }

    pub fn copy<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        src_stream: &Arc<CudaStream>,
        dst: &mut CudaSlice<T>,
        dst_stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<()> {
        ensure!(len <= src.len() && len <= dst.len(), "copy of {len} elements exceeds a buffer");
        ensure!(src_stream.context().ordinal() == self.src.ordinal() && dst_stream.context().ordinal() == self.dst.ordinal(), "stream on the wrong device");
        let bytes = len * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        let ready = src_stream.record_event(None)?;
        dst_stream.wait(&ready)?;
        match self.mode {
            LinkMode::Peer => {
                let (s, _rs) = src.device_ptr(src_stream);
                let (d, _rd) = dst.device_ptr_mut(dst_stream);
                unsafe { cudarc::driver::result::memcpy_peer_async(self.dst.cu_ctx(), d, self.src.cu_ctx(), s, bytes, dst_stream.cu_stream())? };
            }
            LinkMode::Staged => {
                let mut host: PinnedHostSlice<T> = unsafe { self.src.alloc_pinned(len)? };
                {
                    let view = src.slice(0..len);
                    src_stream.memcpy_dtoh(&view, &mut host)?;
                }
                let staged = src_stream.record_event(None)?;
                dst_stream.wait(&staged)?;
                let mut view = dst.slice_mut(0..len);
                dst_stream.memcpy_htod(&host, &mut view)?;
                dst_stream.synchronize()?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairProbe {
    pub a: usize,
    pub b: usize,
    pub p2p: bool,
    pub latency_ms: f64,
    pub bandwidth_gbs: f64,
}

const PROBE_LATENCY_BYTES: usize = 4096;
const PROBE_BANDWIDTH_BYTES: usize = 64 << 20;
const PROBE_ROUNDS: usize = 20;

fn median_secs(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn timed_copies(link: &DeviceLink, sa: &Arc<CudaStream>, sb: &Arc<CudaStream>, bytes: usize) -> Result<f64> {
    let src = sa.alloc_zeros::<u8>(bytes)?;
    let mut dst = sb.alloc_zeros::<u8>(bytes)?;
    link.copy(&src, sa, &mut dst, sb, bytes)?;
    sb.synchronize()?;
    let mut times = Vec::with_capacity(PROBE_ROUNDS);
    for _ in 0..PROBE_ROUNDS {
        let started = std::time::Instant::now();
        link.copy(&src, sa, &mut dst, sb, bytes)?;
        sb.synchronize()?;
        times.push(started.elapsed().as_secs_f64());
    }
    Ok(median_secs(times))
}

impl PairProbe {
    pub fn from_timings(a: usize, b: usize, p2p: bool, small_secs: f64, large_secs: f64) -> PairProbe {
        PairProbe { a, b, p2p, latency_ms: small_secs * 1e3, bandwidth_gbs: PROBE_BANDWIDTH_BYTES as f64 / large_secs / 1e9 }
    }
}

pub fn probe_mode(a_to_b: bool, b_to_a: bool) -> LinkMode {
    if a_to_b && b_to_a { LinkMode::Peer } else { LinkMode::Staged }
}

pub fn median_of(v: Vec<f64>) -> f64 {
    median_secs(v)
}

pub fn probe_pair(a: usize, b: usize) -> Result<PairProbe> {
    ensure!(a != b, "a link probe needs two different devices");
    let (ca, cb) = (CudaContext::new(a)?, CudaContext::new(b)?);
    let mode = probe_mode(peer_access(&ca, &cb), peer_access(&cb, &ca));
    let link = DeviceLink::with_mode(&ca, &cb, mode)?;
    let (sa, sb) = (ca.new_stream()?, cb.new_stream()?);
    let small = timed_copies(&link, &sa, &sb, PROBE_LATENCY_BYTES)?;
    let large = timed_copies(&link, &sa, &sb, PROBE_BANDWIDTH_BYTES)?;
    Ok(PairProbe::from_timings(a, b, mode == LinkMode::Peer, small, large))
}

pub fn probe_local() -> Result<Vec<PairProbe>> {
    let count = CudaContext::device_count()? as usize;
    let mut out = Vec::new();
    for a in 0..count {
        for b in a + 1..count {
            out.push(probe_pair(a, b)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timings_become_latency_and_bandwidth() {
        let p = PairProbe::from_timings(0, 1, true, 12e-6, 64.0 * 1048576.0 / 25e9);
        assert!((p.latency_ms - 0.012).abs() < 1e-9);
        assert!((p.bandwidth_gbs - 25.0).abs() < 1e-6);
        assert!(p.p2p);
    }

    #[test]
    fn peer_access_in_only_one_direction_falls_back_to_host_staging() {
        assert_eq!(probe_mode(true, true), LinkMode::Peer);
        assert_eq!(probe_mode(true, false), LinkMode::Staged);
        assert_eq!(probe_mode(false, false), LinkMode::Staged);
    }

    #[test]
    fn the_median_ignores_one_slow_outlier() {
        assert_eq!(median_of(vec![3.0, 1.0, 100.0, 2.0, 2.5]), 2.5);
    }
}
