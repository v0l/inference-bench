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
