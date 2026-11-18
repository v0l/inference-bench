use crate::codegen::{Module, cuda_target};
use crate::jit::Jit;
use anyhow::{Result, bail, ensure};
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use kir::allreduce::{AllReduceKernel, Dtype, MAX_BLOCKS, MAX_RANKS, Mode, allreduce_hash, flag_words};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const ONE_SHOT_MAX_BYTES: usize = 1 << 20;
const DEFAULT_BLOCKS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Peer,
    SameDevice,
    HostPinned,
}

pub fn transport_for(devices: &[usize], peer_ok: impl Fn(usize, usize) -> bool) -> Transport {
    let mut distinct: Vec<usize> = devices.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() == 1 {
        return Transport::SameDevice;
    }
    let all = devices.iter().all(|&a| devices.iter().all(|&b| a == b || peer_ok(a, b)));
    if all { Transport::Peer } else { Transport::HostPinned }
}

pub fn pick_mode(bytes: usize) -> Mode {
    if bytes <= ONE_SHOT_MAX_BYTES { Mode::OneShot } else { Mode::TwoShot }
}

enum Region {
    Device(sys::CUdeviceptr),
    Host(*mut std::ffi::c_void),
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

struct Owned {
    ctx: Arc<CudaContext>,
    data: Region,
    flags: Region,
}

impl Drop for Owned {
    fn drop(&mut self) {
        for r in [&self.data, &self.flags] {
            let _ = self.ctx.bind_to_thread();
            match r {
                Region::Host(p) => unsafe { cudarc::driver::result::free_host(*p).ok() },
                Region::Device(p) => unsafe { sys::cuMemFree_v2(*p).result().ok() },
            };
        }
    }
}

pub struct Fabric {
    owned: Vec<Arc<Owned>>,
    table: Vec<u64>,
    ctxs: Vec<Arc<CudaContext>>,
    transport: Transport,
    capacity_bytes: usize,
    blocks: u32,
}

pub fn blocks_for(transport: Transport, ranks: usize) -> u32 {
    match transport {
        Transport::SameDevice => (16 / ranks.max(1) as u32).max(2),
        _ => DEFAULT_BLOCKS,
    }
    .min(MAX_BLOCKS)
}

fn enable_peers(ctxs: &[Arc<CudaContext>]) -> Result<()> {
    for from in ctxs {
        for to in ctxs {
            if from.ordinal() == to.ordinal() {
                continue;
            }
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
    Ok(())
}

fn host_region(ctx: &Arc<CudaContext>, bytes: usize) -> Result<(Region, u64)> {
    ctx.bind_to_thread()?;
    let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP)? };
    unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
    let mut dev: sys::CUdeviceptr = 0;
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, ptr, 0).result()? };
    Ok((Region::Host(ptr), dev))
}

fn device_region(ctx: &Arc<CudaContext>, bytes: usize) -> Result<(Region, u64)> {
    ctx.bind_to_thread()?;
    let mut ptr: sys::CUdeviceptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut ptr, bytes).result()?;
        sys::cuMemsetD8_v2(ptr, 0, bytes).result()?;
    }
    Ok((Region::Device(ptr), ptr))
}

fn pci_path(ctx: &Arc<CudaContext>) -> Result<String> {
    use sys::CUdevice_attribute as A;
    let domain = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?;
    let bus = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?;
    let dev = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?;
    Ok(format!("{domain:04x}:{bus:02x}:{dev:02x}.0"))
}

pub fn iommu_translates(group_type: &str) -> bool {
    matches!(group_type.trim(), "DMA" | "DMA-FQ")
}

pub fn iommu_mode(ctx: &Arc<CudaContext>) -> Option<String> {
    let path = format!("/sys/bus/pci/devices/{}/iommu_group/type", pci_path(ctx).ok()?);
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

pub fn contexts(devices: &[usize]) -> Result<Vec<Arc<CudaContext>>> {
    let mut ctxs: Vec<Arc<CudaContext>> = Vec::new();
    for &d in devices {
        ctxs.push(match ctxs.iter().find(|c| c.ordinal() == d) {
            Some(c) => c.clone(),
            None => CudaContext::new(d)?,
        });
    }
    Ok(ctxs)
}

impl Fabric {
    pub fn new(ctxs: &[Arc<CudaContext>], capacity_bytes: usize, force: Option<Transport>) -> Result<Fabric> {
        let n = ctxs.len();
        ensure!(n >= 2 && n <= MAX_RANKS as usize, "an allreduce group needs 2..={MAX_RANKS} ranks, got {n}");
        ensure!(capacity_bytes > 0, "empty allreduce buffer");
        let ordinals: Vec<usize> = ctxs.iter().map(|c| c.ordinal()).collect();
        let peer_ok = |a: usize, b: usize| {
            let (ca, cb) = (ctxs.iter().find(|c| c.ordinal() == a), ctxs.iter().find(|c| c.ordinal() == b));
            matches!((ca, cb), (Some(ca), Some(cb)) if crate::link::peer_access(ca, cb) && crate::link::peer_access(cb, ca))
        };
        let mut transport = force.unwrap_or_else(|| transport_for(&ordinals, peer_ok));
        if transport == Transport::Peer && force.is_none() {
            let translated: Vec<(usize, String)> = ctxs.iter().filter_map(|c| iommu_mode(c).filter(|m| iommu_translates(m)).map(|m| (c.ordinal(), m))).collect();
            if !translated.is_empty() {
                tracing::warn!(devices = ?translated, "the IOMMU translates GPU DMA, so kernel stores to a peer GPU do not land (boot with iommu=pt to fix); the allreduce goes through pinned host memory");
                transport = Transport::HostPinned;
            }
        }
        if transport == Transport::Peer {
            let mut distinct: Vec<Arc<CudaContext>> = Vec::new();
            for c in ctxs {
                if !distinct.iter().any(|d| d.ordinal() == c.ordinal()) {
                    distinct.push(c.clone());
                }
            }
            enable_peers(&distinct)?;
        }
        let flag_bytes = flag_words() * 4;
        let (mut owned, mut data, mut flags) = (Vec::new(), Vec::new(), Vec::new());
        for ctx in ctxs {
            let (d, dp, f, fp) = if transport == Transport::HostPinned {
                let (d, dp) = host_region(ctx, capacity_bytes)?;
                let (f, fp) = host_region(ctx, flag_bytes)?;
                (d, dp, f, fp)
            } else {
                let (d, dp) = device_region(ctx, capacity_bytes)?;
                let (f, fp) = device_region(ctx, flag_bytes)?;
                (d, dp, f, fp)
            };
            data.push(dp);
            flags.push(fp);
            owned.push(Arc::new(Owned { ctx: ctx.clone(), data: d, flags: f }));
        }
        data.extend(flags);
        Ok(Fabric { owned, table: data, ctxs: ctxs.to_vec(), transport, capacity_bytes, blocks: blocks_for(transport, n) })
    }

    pub fn transport(&self) -> Transport {
        self.transport
    }

    pub fn ranks(&self) -> usize {
        self.ctxs.len()
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn comm(&self, rank: usize, stream: &Arc<CudaStream>, cache_dir: std::path::PathBuf, norm: Option<(u32, f32)>) -> Result<Comm> {
        ensure!(rank < self.ranks(), "rank {rank} outside {}", self.ranks());
        ensure!(stream.context().ordinal() == self.ctxs[rank].ordinal(), "rank {rank} stream is on the wrong device");
        let target = cuda_target(&self.ctxs[rank])?;
        let n = self.ranks() as u32;
        let blocks = self.blocks;
        let mut kernels = Vec::new();
        for dtype in [Dtype::F32, Dtype::Bf16] {
            for mode in [Mode::OneShot, Mode::TwoShot] {
                kernels.push(AllReduceKernel::new(&target, n, dtype, mode, blocks).map_err(anyhow::Error::msg)?);
            }
        }
        if let Some((width, eps)) = norm {
            kernels.push(AllReduceKernel::new(&target, n, Dtype::F32, Mode::AddRmsNorm { width, eps }, blocks).map_err(anyhow::Error::msg)?);
        }
        let source = kir::cuda::emit_allreduce(&kernels).map_err(anyhow::Error::msg)?;
        let names: Vec<&'static str> = kernels.iter().map(|k| &*Box::leak(k.name().into_boxed_str())).collect();
        let module = Module { source, kernels: names.clone(), smem: BTreeMap::new(), arch_features: false, kir_hash: allreduce_hash(&kernels) };
        let loaded = Jit::new(self.ctxs[rank].clone(), cache_dir)?.load(&module)?;
        let mut functions = BTreeMap::new();
        for (k, name) in kernels.iter().zip(names) {
            functions.insert(name, (loaded.function(name)?, k.threads));
        }
        let table = stream.clone_htod(&self.table)?;
        stream.synchronize()?;
        Ok(Comm { rank, ranks: self.ranks(), table, functions, blocks, capacity_bytes: self.capacity_bytes, _owned: self.owned.clone(), norm })
    }
}

pub struct Comm {
    rank: usize,
    ranks: usize,
    table: CudaSlice<u64>,
    functions: BTreeMap<&'static str, (CudaFunction, u32)>,
    blocks: u32,
    capacity_bytes: usize,
    _owned: Vec<Arc<Owned>>,
    norm: Option<(u32, f32)>,
}

fn kernel_name(mode: Mode, dtype: Dtype) -> String {
    match mode {
        Mode::AddRmsNorm { .. } => "ar_add_rmsnorm_f32".into(),
        m => format!("ar_{}_{}", m.tag(), dtype.tag()),
    }
}

impl Comm {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    pub fn has_norm(&self) -> bool {
        self.norm.is_some()
    }

    fn function(&self, name: &str) -> Result<&(CudaFunction, u32)> {
        self.functions.get(name).ok_or_else(|| anyhow::anyhow!("allreduce kernel {name} is not loaded"))
    }

    fn launch_config(&self, threads: u32) -> LaunchConfig {
        LaunchConfig { grid_dim: (self.blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
    }

    pub fn add_into(&self, stream: &Arc<CudaStream>, partial: &CudaView<f32>, x: &CudaView<f32>, n: usize) -> Result<()> {
        self.check(n * 4)?;
        let (f, threads) = self.function(&kernel_name(pick_mode(n * 4), Dtype::F32))?;
        let mut b = stream.launch_builder(f);
        let (rank, count, add, clear) = (self.rank as i32, n as i32, 1i32, 1i32);
        b.arg(partial).arg(x).arg(&self.table).arg(&rank).arg(&count).arg(&add).arg(&clear);
        unsafe { b.launch(self.launch_config(*threads)) }?;
        Ok(())
    }

    pub fn allreduce_raw(&self, stream: &Arc<CudaStream>, dtype: Dtype, input: &CudaSlice<u8>, out: &mut CudaSlice<u8>, n: usize, add: bool) -> Result<()> {
        self.check(n * dtype.bytes())?;
        let (f, threads) = self.function(&kernel_name(pick_mode(n * dtype.bytes()), dtype))?;
        let mut b = stream.launch_builder(f);
        let (rank, count, add, clear) = (self.rank as i32, n as i32, add as i32, 0i32);
        b.arg(input).arg(out).arg(&self.table).arg(&rank).arg(&count).arg(&add).arg(&clear);
        unsafe { b.launch(self.launch_config(*threads)) }?;
        Ok(())
    }

    pub fn allreduce_add_norm(&self, stream: &Arc<CudaStream>, input: &CudaView<f32>, x: &CudaView<f32>, weight: &CudaSlice<f32>, hi: &CudaView<u16>, lo: &CudaView<u16>, rows: usize) -> Result<()> {
        let (width, _) = self.norm.ok_or_else(|| anyhow::anyhow!("this rank has no fused norm kernel"))?;
        self.check(rows * width as usize * 4)?;
        let (f, threads) = self.function("ar_add_rmsnorm_f32")?;
        let mut b = stream.launch_builder(f);
        let (rank, count) = (self.rank as i32, rows as i32);
        b.arg(input).arg(x).arg(weight).arg(hi).arg(lo).arg(&self.table).arg(&rank).arg(&count);
        unsafe { b.launch(self.launch_config(*threads)) }?;
        Ok(())
    }

    fn check(&self, bytes: usize) -> Result<()> {
        if bytes > self.capacity_bytes {
            bail!("{bytes} bytes exceed the {} byte allreduce buffer", self.capacity_bytes);
        }
        Ok(())
    }
}

pub fn bail_unsupported(tp: usize) -> Result<()> {
    if tp > MAX_RANKS as usize {
        bail!("tp {tp} is above the {MAX_RANKS} rank limit of the allreduce");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_device_is_virtual_ranks() {
        assert_eq!(transport_for(&[1, 1, 1, 1], |_, _| false), Transport::SameDevice);
    }

    #[test]
    fn peer_access_everywhere_is_p2p_and_any_gap_goes_through_host_memory() {
        assert_eq!(transport_for(&[0, 1], |_, _| true), Transport::Peer);
        assert_eq!(transport_for(&[0, 1], |a, b| !(a == 1 && b == 0)), Transport::HostPinned);
        assert_eq!(transport_for(&[0, 1, 2], |a, b| a.abs_diff(b) == 1), Transport::HostPinned);
    }

    #[test]
    fn small_messages_go_one_shot_and_large_two_shot() {
        assert_eq!(pick_mode(5120 * 2), Mode::OneShot);
        assert_eq!(pick_mode(64 * 5120 * 2), Mode::OneShot);
        assert_eq!(pick_mode(16384 * 5120 * 2), Mode::TwoShot);
    }

    #[test]
    fn virtual_ranks_leave_most_of_the_gpu_to_the_other_ranks() {
        assert_eq!(blocks_for(Transport::SameDevice, 2), 8);
        assert_eq!(blocks_for(Transport::SameDevice, 4), 4);
        assert_eq!(blocks_for(Transport::Peer, 2), DEFAULT_BLOCKS);
    }

    #[test]
    fn translated_iommu_groups_rule_out_kernel_peer_access() {
        assert!(iommu_translates("DMA-FQ\n"));
        assert!(iommu_translates("DMA"));
        assert!(!iommu_translates("identity"));
        assert!(!iommu_translates("blocked"));
    }

    #[test]
    fn the_rank_limit_is_checked() {
        assert!(bail_unsupported(8).is_ok());
        assert!(bail_unsupported(9).is_err());
    }
}
