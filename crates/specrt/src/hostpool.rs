use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, PinnedHostSlice};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
pub struct HostPool {
    ctx: Arc<CudaContext>,
    used: Arc<AtomicUsize>,
    limit: Arc<AtomicUsize>,
}

pub struct PinnedBuf {
    host: PinnedHostSlice<u8>,
    len: usize,
    used: Arc<AtomicUsize>,
}

impl HostPool {
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        HostPool { ctx, used: Arc::new(AtomicUsize::new(0)), limit: Arc::new(AtomicUsize::new(usize::MAX)) }
    }

    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::SeqCst);
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }

    pub fn alloc(&self, len: usize) -> Result<PinnedBuf> {
        let limit = self.limit.load(Ordering::SeqCst);
        let before = self.used.fetch_add(len, Ordering::SeqCst);
        if before.saturating_add(len) > limit {
            self.used.fetch_sub(len, Ordering::SeqCst);
            ensure!(false, "pinned host pool is full: {before} of {limit} bytes used, {len} more asked");
        }
        match unsafe { self.ctx.alloc_pinned_with_flags::<u8>(len.max(1), 0) }.context("pinned host buffer") {
            Ok(host) => Ok(PinnedBuf { host, len, used: self.used.clone() }),
            Err(e) => {
                self.used.fetch_sub(len, Ordering::SeqCst);
                Err(e)
            }
        }
    }
}

impl PinnedBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> Result<&[u8]> {
        Ok(&self.host.as_slice()?[..self.len])
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [u8]> {
        let len = self.len;
        Ok(&mut self.host.as_mut_slice()?[..len])
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        self.used.fetch_sub(self.len, Ordering::SeqCst);
    }
}
