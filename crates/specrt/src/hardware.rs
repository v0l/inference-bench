use crate::placement::{DeviceConfig, LinkConfig};
use anyhow::{Result, anyhow};
use std::net::IpAddr;

const GIB: f64 = (1u64 << 30) as f64;
const FLOPS_PER_SM_CLOCK: f64 = 1024.0;
pub const PEER_LINK: LinkConfig = LinkConfig { latency_ms: 0.01, bandwidth_gbs: 25.0 };
pub const STAGED_LINK: LinkConfig = LinkConfig { latency_ms: 0.05, bandwidth_gbs: 12.0 };

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuBackend {
    #[default]
    Cuda,
    Metal,
    Vulkan,
    Dx12,
}

impl GpuBackend {
    pub fn key(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "cuda",
            GpuBackend::Metal => "metal",
            GpuBackend::Vulkan => "vulkan",
            GpuBackend::Dx12 => "dx12",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Metal => "Metal",
            GpuBackend::Vulkan => "Vulkan",
            GpuBackend::Dx12 => "DX12",
        }
    }

    pub fn from_key(key: &str) -> GpuBackend {
        match key {
            "metal" => GpuBackend::Metal,
            "vulkan" => GpuBackend::Vulkan,
            "dx12" => GpuBackend::Dx12,
            _ => GpuBackend::Cuda,
        }
    }

    pub fn is_wgpu(self) -> bool {
        self != GpuBackend::Cuda
    }
}

pub fn gpu_label(name: &str, backend: GpuBackend) -> String {
    match backend {
        GpuBackend::Cuda => short_gpu(name).to_string(),
        other => format!("{name} ({})", other.label()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    pub backend: GpuBackend,
    pub ordinal: usize,
    pub name: String,
    pub compute: (u32, u32),
    pub sms: u32,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub bandwidth_gbs: f64,
    pub tflops: f64,
}

impl GpuInfo {
    pub fn arch(&self) -> String {
        match self.backend {
            GpuBackend::Cuda => format!("sm_{}{}", self.compute.0, self.compute.1),
            other => other.key().to_string(),
        }
    }

    pub fn label(&self) -> String {
        gpu_label(&self.name, self.backend)
    }

    pub fn total_gib(&self) -> f64 {
        self.total_bytes as f64 / GIB
    }

    pub fn free_gib(&self) -> f64 {
        self.free_bytes as f64 / GIB
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    PeerToPeer,
    HostStaged,
}

impl LinkKind {
    pub fn label(self) -> &'static str {
        match self {
            LinkKind::PeerToPeer => "peer to peer",
            LinkKind::HostStaged => "staged through host memory",
        }
    }

    pub fn config(self) -> LinkConfig {
        match self {
            LinkKind::PeerToPeer => PEER_LINK,
            LinkKind::HostStaged => STAGED_LINK,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuLink {
    pub a: usize,
    pub b: usize,
    pub kind: LinkKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetInterface {
    pub name: String,
    pub ip: IpAddr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub host: String,
    pub gpus: Vec<GpuInfo>,
    pub links: Vec<GpuLink>,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub interfaces: Vec<NetInterface>,
}

pub fn device_name(host: &str, ordinal: usize) -> String {
    format!("{host}/gpu{ordinal}")
}

pub fn display_name(device: &str) -> String {
    match device.rsplit_once("/gpu") {
        Some((host, n)) if n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty() => format!("{host} / GPU {n}"),
        _ => device.to_string(),
    }
}

pub fn host_of(device: &str) -> &str {
    device.rsplit_once("/gpu").map_or(device, |(h, _)| h)
}

impl Machine {
    pub fn link_between(&self, a: usize, b: usize) -> LinkKind {
        self.links.iter().find(|l| (l.a, l.b) == (a, b) || (l.a, l.b) == (b, a)).map_or(LinkKind::HostStaged, |l| l.kind)
    }

    pub fn peer_link(&self, ordinal: usize) -> Option<LinkConfig> {
        let kinds: Vec<LinkKind> = self.gpus.iter().filter(|g| g.ordinal != ordinal).map(|g| self.link_between(ordinal, g.ordinal)).collect();
        let worst = if kinds.contains(&LinkKind::HostStaged) { LinkKind::HostStaged } else { *kinds.first()? };
        Some(worst.config())
    }

    pub fn devices(&self, excluded: &[usize]) -> Vec<DeviceConfig> {
        let mut out: Vec<DeviceConfig> = Vec::new();
        let mut previous: Option<usize> = None;
        for g in self.gpus.iter().filter(|g| !excluded.contains(&g.ordinal)) {
            let link = previous.map_or(LinkConfig::default(), |p| self.link_between(p, g.ordinal).config());
            out.push(DeviceConfig {
                name: device_name(&self.host, g.ordinal),
                memory_gib: g.free_gib().min(g.total_gib()),
                bandwidth_gbs: g.bandwidth_gbs,
                tflops: g.tflops,
                ordinal: Some(g.ordinal),
                address: None,
                reserve_gib: None,
                link,
                offload: None,
                host: Some(self.host.clone()),
                peer: self.peer_link(g.ordinal),
                wgpu: g.backend.is_wgpu(),
            });
            previous = Some(g.ordinal);
        }
        out
    }

    pub fn summary(&self) -> String {
        summarize_labels(&self.host, &self.gpus.iter().map(|g| g.label()).collect::<Vec<_>>())
    }
}

pub fn summarize_labels(host: &str, labels: &[String]) -> String {
    match labels {
        [] => format!("{host}: no GPU"),
        [one] => format!("{host}: {one}"),
        many if many.iter().all(|n| *n == many[0]) => format!("{host}: {} × {}", many.len(), many[0]),
        many => format!("{host}: {}", many.join(" + ")),
    }
}

pub fn summarize(host: &str, names: &[&str]) -> String {
    match names {
        [] => format!("{host}: no GPU"),
        [one] => format!("{host}: {}", short_gpu(one)),
        many if many.iter().all(|n| *n == many[0]) => format!("{host}: {} × {}", many.len(), short_gpu(many[0])),
        many => format!("{host}: {}", many.iter().map(|n| short_gpu(n)).collect::<Vec<_>>().join(" + ")),
    }
}

pub fn short_gpu(name: &str) -> &str {
    name.strip_prefix("NVIDIA ").unwrap_or(name)
}

pub fn parse_meminfo(text: &str) -> (u64, u64) {
    let kib = |key: &str| {
        text.lines().find_map(|l| l.strip_prefix(key)?.trim().trim_end_matches("kB").trim().parse::<u64>().ok()).map_or(0, |v| v << 10)
    };
    (kib("MemTotal:"), kib("MemAvailable:"))
}

pub fn interfaces() -> Vec<NetInterface> {
    if_addrs::get_if_addrs()
        .map(|all| all.into_iter().filter(|i| i.is_oper_up() && !i.is_loopback() && !i.is_link_local() && i.ip().is_ipv4()).map(|i| NetInterface { name: i.name.clone(), ip: i.ip() }).collect())
        .unwrap_or_default()
}

#[cfg(not(unix))]
fn os_hostname() -> Option<String> {
    std::env::var("COMPUTERNAME").ok().filter(|n| !n.is_empty())
}

#[cfg(unix)]
fn os_hostname() -> Option<String> {
    let mut buf = [0u8; 256];
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..end]).trim().to_string();
    (!name.is_empty()).then_some(name)
}

pub fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(os_hostname)
        .or_else(|| std::env::var("HOSTNAME").ok().filter(|s| !s.is_empty()))
        .or_else(|| std::env::var("COMPUTERNAME").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "localhost".into())
}

pub fn estimated_tflops(sms: u32, clock_khz: i32) -> f64 {
    sms as f64 * clock_khz as f64 * 1e3 * FLOPS_PER_SM_CLOCK / 1e12
}

pub fn estimated_bandwidth_gbs(memory_clock_khz: i32, bus_bits: i32) -> f64 {
    2.0 * memory_clock_khz as f64 * 1e3 * (bus_bits as f64 / 8.0) / 1e9
}

pub fn cuda_present() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() }
}

pub fn probe_idle_cuda() -> Result<Machine> {
    if !cuda_present() {
        return Err(anyhow!("no CUDA driver library on this machine"));
    }
    use cudarc::driver::sys::CUdevice_attribute as A;
    use cudarc::driver::result;
    result::init().map_err(|e| anyhow!("no CUDA driver: {e:?}"))?;
    let count = result::device::get_count().map_err(|e| anyhow!("no CUDA driver: {e:?}"))? as usize;
    let mut gpus = Vec::new();
    for ordinal in 0..count {
        let dev = result::device::get(ordinal as i32).map_err(|e| anyhow!("CUDA device {ordinal}: {e:?}"))?;
        let attr = |a| unsafe { result::device::get_attribute(dev, a) }.unwrap_or(0);
        let total = unsafe { result::device::total_mem(dev) }.map_err(|e| anyhow!("{e:?}"))? as u64;
        let sms = attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as u32;
        gpus.push(GpuInfo {
            backend: GpuBackend::Cuda,
            ordinal,
            name: result::device::get_name(dev).map_err(|e| anyhow!("{e:?}"))?,
            compute: (attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) as u32, attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) as u32),
            sms,
            total_bytes: total,
            free_bytes: total,
            bandwidth_gbs: estimated_bandwidth_gbs(attr(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE), attr(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)),
            tflops: estimated_tflops(sms, attr(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)),
        });
    }
    Ok(machine_with(gpus, Vec::new()))
}

pub fn probe_cuda() -> Result<Machine> {
    if !cuda_present() {
        return Err(anyhow!("no CUDA driver library on this machine"));
    }
    use cudarc::driver::sys::CUdevice_attribute as A;
    use cudarc::driver::{CudaContext, result, sys};
    let count = CudaContext::device_count().map_err(|e| anyhow!("no CUDA driver: {e:?}"))? as usize;
    let mut gpus = Vec::new();
    for ordinal in 0..count {
        let ctx = CudaContext::new(ordinal).map_err(|e| anyhow!("CUDA device {ordinal}: {e:?}"))?;
        let attr = |a| ctx.attribute(a).unwrap_or(0);
        let (major, minor) = ctx.compute_capability().map_err(|e| anyhow!("{e:?}"))?;
        let (free, total) = ctx.mem_get_info().map_err(|e| anyhow!("{e:?}"))?;
        let sms = attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as u32;
        gpus.push(GpuInfo {
            backend: GpuBackend::Cuda,
            ordinal,
            name: ctx.name().map_err(|e| anyhow!("{e:?}"))?,
            compute: (major as u32, minor as u32),
            sms,
            total_bytes: total as u64,
            free_bytes: free as u64,
            bandwidth_gbs: estimated_bandwidth_gbs(attr(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE), attr(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)),
            tflops: estimated_tflops(sms, attr(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)),
        });
    }
    let mut links = Vec::new();
    for a in 0..count {
        for b in a + 1..count {
            let (da, db) = (result::device::get(a as i32).map_err(|e| anyhow!("{e:?}"))?, result::device::get(b as i32).map_err(|e| anyhow!("{e:?}"))?);
            let mut ok = 0;
            let rc = unsafe { sys::cuDeviceCanAccessPeer(&mut ok, da, db) };
            let kind = if rc == sys::CUresult::CUDA_SUCCESS && ok != 0 { LinkKind::PeerToPeer } else { LinkKind::HostStaged };
            links.push(GpuLink { a, b, kind });
        }
    }
    Ok(machine_with(gpus, links))
}

pub fn probe_with(cuda: impl FnOnce() -> Result<Machine>, wgpu: impl FnOnce() -> Result<Machine>) -> Result<Machine> {
    match cuda() {
        Ok(m) if !m.gpus.is_empty() => Ok(m),
        Ok(_) => wgpu().map_err(|e| anyhow!("no CUDA device and {e:#}")),
        Err(cuda) => wgpu().map_err(|e| anyhow!("{cuda:#}; wgpu fallback: {e:#}")),
    }
}

pub fn probe() -> Result<Machine> {
    probe_with(probe_cuda, probe_wgpu)
}

pub fn probe_idle() -> Result<Machine> {
    probe_with(probe_idle_cuda, probe_wgpu)
}

const APPLE_CHIPS: &[(&str, f64, f64)] = &[
    ("M1 Ultra", 800.0, 21.0),
    ("M1 Max", 400.0, 10.4),
    ("M1 Pro", 200.0, 5.2),
    ("M2 Ultra", 800.0, 27.2),
    ("M2 Max", 400.0, 13.6),
    ("M2 Pro", 200.0, 6.8),
    ("M3 Ultra", 800.0, 28.2),
    ("M3 Max", 400.0, 16.4),
    ("M3 Pro", 150.0, 7.4),
    ("M4 Max", 546.0, 18.4),
    ("M4 Pro", 273.0, 9.2),
    ("M1", 68.0, 2.6),
    ("M2", 100.0, 3.6),
    ("M3", 100.0, 4.1),
    ("M4", 120.0, 4.3),
];

pub fn apple_chip_specs(name: &str) -> (f64, f64) {
    APPLE_CHIPS.iter().find(|(chip, _, _)| name.contains(chip)).map_or((100.0, 3.0), |&(_, gbs, tflops)| (gbs, tflops))
}

pub fn parse_vm_stat(text: &str) -> u64 {
    let page = text
        .lines()
        .next()
        .and_then(|l| l.split("page size of").nth(1))
        .and_then(|r| r.trim().split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(16384);
    let pages = |key: &str| text.lines().find_map(|l| l.strip_prefix(key)?.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
    (pages("Pages free:") + pages("Pages inactive:") + pages("Pages speculative:") + pages("Pages purgeable:")) * page
}

#[cfg(target_os = "macos")]
pub fn unified_memory() -> Option<(u64, u64)> {
    let mut total: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = std::ffi::CString::new("hw.memsize").ok()?;
    let rc = unsafe { libc::sysctlbyname(name.as_ptr(), &mut total as *mut u64 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) };
    if rc != 0 || total == 0 {
        return None;
    }
    let free = std::process::Command::new("vm_stat").output().ok().map(|o| parse_vm_stat(&String::from_utf8_lossy(&o.stdout))).filter(|f| *f > 0).unwrap_or(total / 2);
    Some((total, free.min(total / 4 * 3)))
}

#[cfg(not(target_os = "macos"))]
pub fn unified_memory() -> Option<(u64, u64)> {
    None
}

#[cfg(feature = "wgpu")]
pub fn probe_wgpu() -> Result<Machine> {
    let adapters: Vec<_> = crate::backends::wgpu::adapters().into_iter().filter(|a| a.device_type != wgpu::DeviceType::Cpu).collect();
    machine_from_adapters(adapters.iter().map(|a| (a.name.clone(), wgpu_backend(a.backend))).collect(), unified_memory())
}

#[cfg(feature = "wgpu")]
fn wgpu_backend(b: wgpu::Backend) -> GpuBackend {
    match b {
        wgpu::Backend::Metal => GpuBackend::Metal,
        wgpu::Backend::Dx12 => GpuBackend::Dx12,
        _ => GpuBackend::Vulkan,
    }
}

#[cfg(not(feature = "wgpu"))]
pub fn probe_wgpu() -> Result<Machine> {
    Err(anyhow!("this build has no wgpu support (build with --features wgpu)"))
}

pub fn machine_from_adapters(adapters: Vec<(String, GpuBackend)>, memory: Option<(u64, u64)>) -> Result<Machine> {
    if adapters.is_empty() {
        return Err(anyhow!("no GPU adapter is available to wgpu (Vulkan, Metal or DX12)"));
    }
    let (total, free) = memory.unwrap_or((0, 0));
    let gpus = adapters
        .into_iter()
        .enumerate()
        .map(|(ordinal, (name, backend))| {
            let (bandwidth_gbs, tflops) = if backend == GpuBackend::Metal { apple_chip_specs(&name) } else { (0.0, 0.0) };
            GpuInfo { backend, ordinal, name, compute: (0, 0), sms: 0, total_bytes: total, free_bytes: free, bandwidth_gbs, tflops }
        })
        .collect();
    Ok(machine_with(gpus, Vec::new()))
}

pub fn machine_with(gpus: Vec<GpuInfo>, links: Vec<GpuLink>) -> Machine {
    let (mut ram_total_bytes, mut ram_available_bytes) = std::fs::read_to_string("/proc/meminfo").map(|t| parse_meminfo(&t)).unwrap_or((0, 0));
    if ram_total_bytes == 0
        && let Some((total, free)) = unified_memory()
    {
        (ram_total_bytes, ram_available_bytes) = (total, free);
    }
    Machine { host: host_name(), gpus, links, ram_total_bytes, ram_available_bytes, interfaces: interfaces() }
}

pub fn fake_gpus(host: &str, n: usize, p2p: bool) -> Machine {
    let gpus: Vec<GpuInfo> = (0..n)
        .map(|i| GpuInfo { backend: GpuBackend::Cuda, ordinal: i, name: "NVIDIA RTX PRO 6000 Blackwell Max-Q".into(), compute: (12, 0), sms: 188, total_bytes: 96 << 30, free_bytes: (90 - i as u64) << 30, bandwidth_gbs: 1792.0, tflops: 500.0 })
        .collect();
    let mut links = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            links.push(GpuLink { a, b, kind: if p2p { LinkKind::PeerToPeer } else { LinkKind::HostStaged } });
        }
    }
    Machine { host: host.into(), gpus, links, ram_total_bytes: 64 << 30, ram_available_bytes: 40 << 30, interfaces: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_is_read_in_bytes() {
        let (total, avail) = parse_meminfo("MemTotal:       65536000 kB\nMemFree: 1 kB\nMemAvailable:   32768000 kB\n");
        assert_eq!((total, avail), (65536000 << 10, 32768000 << 10));
    }

    #[test]
    fn bandwidth_and_flops_come_from_the_driver_attributes() {
        assert!((estimated_bandwidth_gbs(14_001_000, 512) - 1792.1).abs() < 1.0);
        assert!((estimated_tflops(188, 2_400_000) - 462.0).abs() < 5.0);
    }

    #[test]
    fn a_four_gpu_machine_yields_one_device_per_gpu_with_the_link_between_neighbours() {
        let mut m = fake_gpus("box", 4, true);
        m.links.retain(|l| !(l.a == 1 && l.b == 2));
        let devices = m.devices(&[]);
        assert_eq!(devices.len(), 4);
        assert_eq!(devices[2].name, "box/gpu2");
        assert_eq!(devices[2].ordinal, Some(2));
        assert_eq!(devices[1].link, PEER_LINK);
        assert_eq!(devices[2].link, STAGED_LINK);
        assert_eq!(devices[0].memory_gib, 90.0);
    }

    #[test]
    fn excluded_gpus_are_left_out_and_the_next_link_is_between_the_survivors() {
        let m = fake_gpus("box", 3, true);
        let devices = m.devices(&[1]);
        assert_eq!(devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["box/gpu0", "box/gpu2"]);
        assert_eq!(devices[1].link, PEER_LINK);
    }

    #[test]
    fn names_display_as_host_and_gpu() {
        assert_eq!(display_name("spark/gpu1"), "spark / GPU 1");
        assert_eq!(display_name("legacy-node"), "legacy-node");
        assert_eq!(host_of("spark/gpu1"), "spark");
        assert_eq!(fake_gpus("desk", 2, true).summary(), "desk: 2 × RTX PRO 6000 Blackwell Max-Q");
        assert_eq!(summarize("x", &[]), "x: no GPU");
    }

    fn metal_machine() -> Machine {
        machine_from_adapters(vec![("Apple M1 Pro".into(), GpuBackend::Metal)], Some((16 << 30, 9 << 30))).unwrap()
    }

    #[test]
    fn without_cuda_the_probe_falls_back_to_the_wgpu_adapter() {
        let m = probe_with(|| Err(anyhow!("no CUDA driver")), || Ok(metal_machine())).unwrap();
        assert_eq!(m.gpus[0].backend, GpuBackend::Metal);
        assert_eq!(m.summary().split_once(": ").unwrap().1, "Apple M1 Pro (Metal)");
        assert_eq!(m.gpus[0].arch(), "metal");
        let cuda = probe_with(|| Ok(fake_gpus("box", 1, true)), || panic!("wgpu must not run when CUDA works")).unwrap();
        assert_eq!(cuda.gpus[0].backend, GpuBackend::Cuda);
        let none = probe_with(|| Err(anyhow!("no CUDA driver")), || Err(anyhow!("no GPU adapter"))).unwrap_err().to_string();
        assert!(none.contains("no CUDA driver") && none.contains("no GPU adapter"), "{none}");
    }

    #[test]
    fn an_apple_chip_gets_its_bandwidth_and_unified_memory() {
        let g = &metal_machine().gpus[0];
        assert_eq!((g.bandwidth_gbs, g.tflops), (200.0, 5.2));
        assert_eq!((g.total_gib(), g.free_gib()), (16.0, 9.0));
        assert_eq!(apple_chip_specs("Apple M2 Max"), (400.0, 13.6));
        assert_eq!(apple_chip_specs("Apple M4"), (120.0, 4.3));
        assert!(machine_from_adapters(vec![], None).is_err());
    }

    #[test]
    fn vm_stat_free_memory_counts_reclaimable_pages() {
        let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               1000.\nPages active:                            5000.\nPages inactive:                           500.\nPages speculative:                        100.\nPages purgeable:                          50.\n";
        assert_eq!(parse_vm_stat(text), 1650 * 16384);
    }

    #[test]
    fn a_metal_device_is_flagged_for_the_planner() {
        let devices = metal_machine().devices(&[]);
        assert!(devices[0].wgpu);
        assert!(!fake_gpus("box", 1, true).devices(&[])[0].wgpu);
    }
}
