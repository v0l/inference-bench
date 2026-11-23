use crate::codegen::{Module, cuda_target};
use crate::jit::Jit;
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use kir::allreduce::{AllReduceKernel, Dtype, MAX_BLOCKS, MAX_RANKS, Mode, allreduce_hash, flag_words};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const ONE_SHOT_MAX_BYTES: usize = 1 << 20;
pub const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);
const PROBE_MAGIC: u32 = 0x6587_1a2b;
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
                Region::Host(p) if p.is_null() => None,
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

fn distinct_contexts(ctxs: &[Arc<CudaContext>]) -> Vec<Arc<CudaContext>> {
    let mut distinct: Vec<Arc<CudaContext>> = Vec::new();
    for c in ctxs {
        if !distinct.iter().any(|d| d.ordinal() == c.ordinal()) {
            distinct.push(c.clone());
        }
    }
    distinct
}

struct ProbeKernels {
    _module: Arc<cudarc::driver::CudaModule>,
    wait: CudaFunction,
    store: CudaFunction,
}

fn probe_kernels(ctx: &Arc<CudaContext>) -> Result<ProbeKernels> {
    let module = crate::jit::load_uncached(ctx, &kir::cuda::emit_peer_probe()).map_err(|e| anyhow::anyhow!("peer probe: {e:#}"))?;
    Ok(ProbeKernels { wait: module.load_function(kir::cuda::PROBE_WAIT)?, store: module.load_function(kir::cuda::PROBE_STORE)?, _module: module })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    Landed,
    TimedOut,
}

pub fn probe_pair(from: &Arc<CudaContext>, to: &Arc<CudaContext>, timeout: std::time::Duration, write: bool) -> Result<ProbeOutcome> {
    let (pf, pt) = (probe_kernels(from)?, probe_kernels(to)?);
    let (sf, st) = (from.new_stream()?, to.new_stream()?);
    let (region, words) = device_region(to, 8)?;
    let (flag, result) = (words, words + 4);
    let launched = (|| -> Result<ProbeOutcome> {
        let (magic, ns) = (PROBE_MAGIC, timeout.as_nanos() as u64);
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut b = st.launch_builder(&pt.wait);
        b.arg(&flag).arg(&magic).arg(&ns).arg(&result);
        unsafe { b.launch(one) }?;
        if write {
            let mut b = sf.launch_builder(&pf.store);
            b.arg(&flag).arg(&magic);
            unsafe { b.launch(one) }?;
        }
        st.synchronize()?;
        sf.synchronize()?;
        let mut got = 0u32;
        to.bind_to_thread()?;
        unsafe { sys::cuMemcpyDtoH_v2(&mut got as *mut u32 as *mut std::ffi::c_void, result, 4).result()? };
        Ok(if got == 1 { ProbeOutcome::Landed } else { ProbeOutcome::TimedOut })
    })();
    drop(Owned { ctx: to.clone(), data: region, flags: Region::Host(std::ptr::null_mut()) });
    launched
}

pub fn peer_writes_land(ctxs: &[Arc<CudaContext>], timeout: std::time::Duration, write: bool) -> Result<Vec<(usize, usize, ProbeOutcome)>> {
    let distinct = distinct_contexts(ctxs);
    let mut out = Vec::new();
    for a in &distinct {
        for b in &distinct {
            if a.ordinal() != b.ordinal() || distinct.len() == 1 {
                let outcome = probe_pair(a, b, timeout, write).with_context(|| format!("peer write probe gpu{} to gpu{} faulted; force the host-pinned transport on this machine", a.ordinal(), b.ordinal()))?;
                out.push((a.ordinal(), b.ordinal(), outcome));
            }
        }
    }
    Ok(out)
}

pub fn settle(candidate: Transport, probes: &[(usize, usize, ProbeOutcome)]) -> Transport {
    if candidate == Transport::HostPinned || probes.iter().all(|p| p.2 == ProbeOutcome::Landed) { candidate } else { Transport::HostPinned }
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
        Self::with_probe(ctxs, capacity_bytes, force, |c| peer_writes_land(c, PROBE_TIMEOUT, true))
    }

    pub fn with_probe(ctxs: &[Arc<CudaContext>], capacity_bytes: usize, force: Option<Transport>, probe: impl FnOnce(&[Arc<CudaContext>]) -> Result<Vec<(usize, usize, ProbeOutcome)>>) -> Result<Fabric> {
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
            enable_peers(&distinct_contexts(ctxs))?;
        }
        if force.is_none() && transport != Transport::HostPinned {
            let probes = probe(ctxs)?;
            let settled = settle(transport, &probes);
            if settled != transport {
                let missed: Vec<(usize, usize)> = probes.iter().filter(|p| p.2 != ProbeOutcome::Landed).map(|p| (p.0, p.1)).collect();
                tracing::warn!(pairs = ?missed, timeout_ms = PROBE_TIMEOUT.as_millis() as u64, "a kernel store to a peer GPU did not arrive, so the allreduce goes through pinned host memory");
            }
            transport = settled;
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

    pub fn comm(&self, rank: usize, stream: &Arc<CudaStream>, cache_dir: std::path::PathBuf, norm: Option<(u32, f32)>, gather: Option<(u32, u32)>) -> Result<Comm> {
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
        if let Some((width, each)) = gather {
            kernels.push(AllReduceKernel::new(&target, n, Dtype::F32, Mode::Gather { width, each }, blocks).map_err(anyhow::Error::msg)?);
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
        Ok(Comm { rank, ranks: self.ranks(), table, functions, blocks, capacity_bytes: self.capacity_bytes, _owned: self.owned.clone(), norm, gather })
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
    gather: Option<(u32, u32)>,
}

pub fn kernel_name(mode: Mode, dtype: Dtype) -> String {
    match mode {
        Mode::AddRmsNorm { .. } => "ar_add_rmsnorm_f32".into(),
        Mode::Gather { .. } => "ar_gather_f32".into(),
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

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn gather_each(&self) -> Option<usize> {
        self.gather.map(|(_, each)| each as usize)
    }

    pub fn gather_rows(&self, stream: &Arc<CudaStream>, part: &CudaView<f32>, out: &CudaView<f32>, rows: usize) -> Result<()> {
        let (width, each) = self.gather.ok_or_else(|| anyhow::anyhow!("this rank has no gather kernel"))?;
        self.check(rows * each as usize * 4)?;
        let own = kir::allreduce::gather_span(width, each, self.rank as u32).len();
        ensure!(part.len() >= rows * own && out.len() >= rows * width as usize, "gather of {rows} rows does not fit its buffers");
        let (f, threads) = self.function("ar_gather_f32")?;
        let mut b = stream.launch_builder(f);
        let (rank, count) = (self.rank as i32, rows as i32);
        b.arg(part).arg(out).arg(&self.table).arg(&rank).arg(&count);
        unsafe { b.launch(self.launch_config(*threads)) }?;
        Ok(())
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
    fn one_missing_peer_write_sends_the_group_through_host_memory() {
        let ok = (0, 1, ProbeOutcome::Landed);
        let lost = (1, 0, ProbeOutcome::TimedOut);
        assert_eq!(settle(Transport::Peer, &[ok, ok]), Transport::Peer);
        assert_eq!(settle(Transport::Peer, &[ok, lost]), Transport::HostPinned);
        assert_eq!(settle(Transport::SameDevice, &[lost]), Transport::HostPinned);
        assert_eq!(settle(Transport::HostPinned, &[ok]), Transport::HostPinned);
    }

    #[test]
    fn the_rank_limit_is_checked() {
        assert!(bail_unsupported(8).is_ok());
        assert!(bail_unsupported(9).is_err());
    }
}
