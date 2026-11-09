use anyhow::{Result, ensure};
use cudarc::driver::sys::{self, CUdeviceptr};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

pub struct StatePool {
    ctx: Arc<CudaContext>,
    base: CUdeviceptr,
    stride: usize,
    slots: usize,
    granularity: usize,
    pages: usize,
    refs: Vec<u32>,
    handles: HashMap<usize, sys::CUmemGenericAllocationHandle>,
    live: Vec<bool>,
    outstanding: usize,
}

fn check(r: sys::CUresult, what: &str) -> Result<()> {
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "{what}: {r:?}");
    Ok(())
}

impl StatePool {
    pub fn new(ctx: &Arc<CudaContext>, stride_bytes: usize, slots: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let prop = Self::prop(ctx);
        let mut granularity = 0usize;
        check(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "allocation granularity",
        )?;
        let pages = (stride_bytes * slots).div_ceil(granularity);
        let mut base: CUdeviceptr = 0;
        check(unsafe { sys::cuMemAddressReserve(&mut base, pages * granularity, 0, 0, 0) }, "reserve address range")?;
        Ok(StatePool {
            ctx: ctx.clone(),
            base,
            stride: stride_bytes,
            slots,
            granularity,
            pages,
            refs: vec![0; pages],
            handles: HashMap::new(),
            live: vec![false; slots],
            outstanding: 0,
        })
    }

    fn prop(ctx: &Arc<CudaContext>) -> sys::CUmemAllocationProp {
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = ctx.ordinal() as i32;
        prop
    }

    pub fn slice(&self, stream: &Arc<CudaStream>) -> CudaSlice<f32> {
        unsafe { stream.upgrade_device_ptr(self.base, self.stride * self.slots / 4) }
    }

    pub fn stride_bytes(&self) -> usize {
        self.stride
    }

    pub fn is_live(&self, slot: usize) -> bool {
        self.live.get(slot).copied().unwrap_or(false)
    }

    pub fn mapped_bytes(&self) -> usize {
        self.handles.len() * self.granularity
    }

    pub fn outstanding_allocations(&self) -> usize {
        self.outstanding
    }

    pub fn live_slots(&self) -> usize {
        self.live.iter().filter(|&&l| l).count()
    }

    fn page_range(&self, slot: usize) -> std::ops::Range<usize> {
        let start = slot * self.stride / self.granularity;
        let end = ((slot + 1) * self.stride).div_ceil(self.granularity);
        start..end.min(self.pages)
    }

    fn map_page(&mut self, page: usize) -> Result<()> {
        let prop = Self::prop(&self.ctx);
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        check(unsafe { sys::cuMemCreate(&mut handle, self.granularity, &prop, 0) }, "allocate state page")?;
        self.outstanding += 1;
        let at = self.base + (page * self.granularity) as u64;
        let mapped = check(unsafe { sys::cuMemMap(at, self.granularity, 0, handle, 0) }, "map state page");
        if let Err(e) = mapped {
            unsafe { sys::cuMemRelease(handle) };
            self.outstanding -= 1;
            return Err(e);
        }
        let mut access: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location = prop.location;
        access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        let granted = check(unsafe { sys::cuMemSetAccess(at, self.granularity, &access, 1) }, "grant access");
        if let Err(e) = granted {
            unsafe {
                sys::cuMemUnmap(at, self.granularity);
                sys::cuMemRelease(handle);
            }
            self.outstanding -= 1;
            return Err(e);
        }
        self.handles.insert(page, handle);
        Ok(())
    }

    fn unmap_page(&mut self, page: usize) {
        if let Some(handle) = self.handles.remove(&page) {
            unsafe {
                sys::cuMemUnmap(self.base + (page * self.granularity) as u64, self.granularity);
                sys::cuMemRelease(handle);
            }
            self.outstanding -= 1;
        }
    }

    pub fn acquire(&mut self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots, "state slot {slot} out of {}", self.slots);
        ensure!(!self.live[slot], "state slot {slot} is already allocated");
        self.ctx.bind_to_thread()?;
        let range = self.page_range(slot);
        let mut taken = Vec::new();
        for page in range.clone() {
            if self.refs[page] == 0 {
                if let Err(e) = self.map_page(page) {
                    for p in taken {
                        self.unmap_page(p);
                    }
                    return Err(e);
                }
                taken.push(page);
            }
        }
        for page in range {
            self.refs[page] += 1;
        }
        self.live[slot] = true;
        Ok(())
    }

    pub fn release(&mut self, slot: usize) {
        if slot >= self.slots || !self.live[slot] {
            return;
        }
        let _ = self.ctx.bind_to_thread();
        self.live[slot] = false;
        for page in self.page_range(slot) {
            self.refs[page] -= 1;
            if self.refs[page] == 0 {
                self.unmap_page(page);
            }
        }
    }
}

impl Drop for StatePool {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        let pages: Vec<usize> = self.handles.keys().copied().collect();
        for page in pages {
            self.unmap_page(page);
        }
        unsafe { sys::cuMemAddressFree(self.base, self.pages * self.granularity) };
    }
}
