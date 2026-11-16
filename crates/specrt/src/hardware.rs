use crate::placement::{DeviceConfig, LinkConfig};
use anyhow::{Result, anyhow};
use std::net::IpAddr;

const GIB: f64 = (1u64 << 30) as f64;
const FLOPS_PER_SM_CLOCK: f64 = 1024.0;
pub const PEER_LINK: LinkConfig = LinkConfig { latency_ms: 0.01, bandwidth_gbs: 25.0 };
pub const STAGED_LINK: LinkConfig = LinkConfig { latency_ms: 0.05, bandwidth_gbs: 12.0 };

#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
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
        format!("sm_{}{}", self.compute.0, self.compute.1)
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
            });
            previous = Some(g.ordinal);
        }
        out
    }

    pub fn summary(&self) -> String {
        summarize(&self.host, &self.gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>())
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

pub fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

pub fn probe() -> Result<Machine> {
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

pub fn machine_with(gpus: Vec<GpuInfo>, links: Vec<GpuLink>) -> Machine {
    let (ram_total_bytes, ram_available_bytes) = std::fs::read_to_string("/proc/meminfo").map(|t| parse_meminfo(&t)).unwrap_or((0, 0));
    Machine { host: host_name(), gpus, links, ram_total_bytes, ram_available_bytes, interfaces: interfaces() }
}

pub fn fake_gpus(host: &str, n: usize, p2p: bool) -> Machine {
    let gpus: Vec<GpuInfo> = (0..n)
        .map(|i| GpuInfo { ordinal: i, name: "NVIDIA RTX PRO 6000 Blackwell Max-Q".into(), compute: (12, 0), sms: 188, total_bytes: 96 << 30, free_bytes: (90 - i as u64) << 30, bandwidth_gbs: 1792.0, tflops: 500.0 })
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
}
