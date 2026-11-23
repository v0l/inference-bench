use crate::rdma::{ConnInfo, Endpoint, Event, LinkOptions, Verbs};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, sys};
use std::collections::VecDeque;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

const GRANULE: usize = 2 << 20;

fn check(r: sys::CUresult, what: &str) -> Result<()> {
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "{what}: {r:?}");
    Ok(())
}

pub struct DirectRegion {
    ctx: Arc<CudaContext>,
    ptr: u64,
    len: usize,
    fd: i32,
}

unsafe impl Send for DirectRegion {}
unsafe impl Sync for DirectRegion {}

impl DirectRegion {
    pub fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<DirectRegion> {
        ensure!(bytes > 0, "an empty direct region");
        let len = bytes.div_ceil(GRANULE) * GRANULE;
        ctx.bind_to_thread()?;
        unsafe {
            let mut supported = 0;
            check(sys::cuDeviceGetAttribute(&mut supported, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, ctx.cu_device()), "dma-buf attribute")?;
            ensure!(supported != 0, "this device cannot export dma-buf");
            let mut ptr = 0;
            check(sys::cuMemAlloc_v2(&mut ptr, len), "cuMemAlloc")?;
            let mut fd: i32 = -1;
            let r = sys::cuMemGetHandleForAddressRange(&mut fd as *mut i32 as *mut std::ffi::c_void, ptr, len, sys::CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
            if r != sys::CUresult::CUDA_SUCCESS {
                sys::cuMemFree_v2(ptr);
                bail!("dma-buf export: {r:?}");
            }
            Ok(DirectRegion { ctx: ctx.clone(), ptr, len, fd })
        }
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn write_host(&self, data: &[u8]) -> Result<()> {
        ensure!(data.len() <= self.len, "{} bytes exceed the region", data.len());
        self.ctx.bind_to_thread()?;
        unsafe {
            check(sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const _, data.len()), "htod")?;
            check(sys::cuCtxSynchronize(), "sync")
        }
    }

    pub fn read_host(&self, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes <= self.len, "{bytes} bytes exceed the region");
        self.ctx.bind_to_thread()?;
        let mut out = vec![0u8; bytes];
        unsafe { check(sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, self.ptr, bytes), "dtoh")? };
        Ok(out)
    }

    pub fn copy_from(&self, other: &DirectRegion, bytes: usize) -> Result<()> {
        ensure!(bytes <= self.len && bytes <= other.len, "{bytes} bytes exceed a region");
        ensure!(self.ctx.ordinal() == other.ctx.ordinal(), "regions live on different devices");
        self.ctx.bind_to_thread()?;
        unsafe { check(sys::cuMemcpyDtoD_v2(self.ptr, other.ptr, bytes), "dtod") }
    }
}

impl Drop for DirectRegion {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            if self.fd >= 0 {
                libc::close(self.fd);
            }
            sys::cuMemFree_v2(self.ptr);
        }
    }
}

pub struct DirectLink {
    ep: Endpoint,
    pub outbox: Arc<DirectRegion>,
    pub landing: Arc<DirectRegion>,
    remote: ConnInfo,
    remote_len: usize,
    lkey: u32,
    pending: VecDeque<u32>,
    sent: usize,
}

impl DirectLink {
    pub fn offer(outbox: &Arc<DirectRegion>, landing: &Arc<DirectRegion>, link: &LinkOptions, local_ip: IpAddr) -> Result<(PendingLink, ConnInfo)> {
        let verbs = Verbs::load(link.library())?;
        ensure!(verbs.supports_dmabuf(), "this libibverbs has no ibv_reg_dmabuf_mr");
        let ep = Endpoint::open(verbs, local_ip)?;
        let out_mr = ep.register_dmabuf(outbox.fd, 0, outbox.len, outbox.ptr)?;
        let land_mr = ep.register_dmabuf(landing.fd, 0, landing.len, landing.ptr)?;
        let (lkey, _) = ep.keys(out_mr);
        let (_, rkey) = ep.keys(land_mr);
        let mut info = ep.info();
        info.addr = landing.ptr;
        info.rkey = rkey;
        Ok((PendingLink { ep, outbox: outbox.clone(), landing: landing.clone(), lkey }, info))
    }

    pub fn capacity(&self) -> usize {
        self.outbox.len.min(self.remote_len)
    }

    pub fn push(&mut self, bytes: usize, timeout: Duration) -> Result<()> {
        ensure!(bytes > 0 && bytes <= self.capacity(), "{bytes} bytes do not fit the direct buffers");
        self.ep.write_with_imm(7, self.outbox.ptr, bytes, self.lkey, self.remote.addr, self.remote.rkey, bytes as u32)?;
        let started = Instant::now();
        let target = self.sent + 1;
        loop {
            self.pump()?;
            if self.sent >= target {
                return Ok(());
            }
            ensure!(started.elapsed() < timeout, "the direct write was not acknowledged");
            std::hint::spin_loop();
        }
    }

    fn pump(&mut self) -> Result<()> {
        for event in self.ep.events()? {
            match event {
                Event::Sent => self.sent += 1,
                Event::Imm(len) => self.pending.push_back(len),
            }
        }
        Ok(())
    }

    pub fn wait(&mut self, bytes: usize, timeout: Duration) -> Result<()> {
        let started = Instant::now();
        loop {
            if let Some(len) = self.pending.pop_front() {
                ensure!(len as usize == bytes, "the peer wrote {len} bytes into the landing buffer, expected {bytes}");
                self.landing.context().bind_to_thread()?;
                unsafe {
                    let _ = sys::cuFlushGPUDirectRDMAWrites(sys::CUflushGPUDirectRDMAWritesTarget::CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX, sys::CUflushGPUDirectRDMAWritesScope::CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
                }
                return Ok(());
            }
            self.pump()?;
            if !self.pending.is_empty() {
                continue;
            }
            ensure!(started.elapsed() < timeout, "no direct write arrived for {bytes} bytes");
            if started.elapsed() > Duration::from_micros(200) {
                std::thread::sleep(Duration::from_micros(20));
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

pub struct PendingLink {
    ep: Endpoint,
    outbox: Arc<DirectRegion>,
    landing: Arc<DirectRegion>,
    lkey: u32,
}

impl PendingLink {
    pub fn connect(mut self, remote: ConnInfo, remote_len: usize) -> Result<DirectLink> {
        self.ep.connect(&remote).context("connect the direct queue pair")?;
        Ok(DirectLink { ep: self.ep, outbox: self.outbox, landing: self.landing, remote, remote_len, lkey: self.lkey, pending: VecDeque::new(), sent: 0 })
    }
}
