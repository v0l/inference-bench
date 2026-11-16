use crate::placement::{DeviceConfig, LinkConfig, Placement, PlacementConfig};
use crate::spec::Spec;
use anyhow::{Result, anyhow, bail};
use std::collections::BTreeMap;

pub const SERVICE_NAME: &str = "_specrt-stage._tcp.local.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Static,
    Mdns,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub device: DeviceConfig,
    pub address: Option<String>,
    pub source: Source,
    pub last_seen_ms: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Join { device: DeviceConfig, address: String, now_ms: u64 },
    Leave { name: String },
}

pub fn encode_announcement(device: &DeviceConfig, address: &str) -> Vec<String> {
    let mut txt = vec![
        format!("name={}", device.name),
        format!("addr={address}"),
        format!("mem={}", device.memory_gib),
        format!("bw={}", device.bandwidth_gbs),
        format!("tflops={}", device.tflops),
        format!("lat={}", device.link.latency_ms),
        format!("link={}", device.link.bandwidth_gbs),
    ];
    if let Some(r) = device.reserve_gib {
        txt.push(format!("reserve={r}"));
    }
    txt
}

pub fn decode_announcement(txt: &[String]) -> Result<(DeviceConfig, String)> {
    let mut fields = BTreeMap::new();
    for entry in txt {
        let (k, v) = entry.split_once('=').ok_or_else(|| anyhow!("announcement entry {entry:?} has no '='"))?;
        fields.insert(k.to_string(), v.to_string());
    }
    let text = |k: &str| fields.get(k).cloned().ok_or_else(|| anyhow!("announcement lacks {k}"));
    let number = |k: &str| -> Result<f64> {
        let v: f64 = text(k)?.parse().map_err(|_| anyhow!("announcement field {k} is not a number"))?;
        if !v.is_finite() || v < 0.0 {
            bail!("announcement field {k} is out of range");
        }
        Ok(v)
    };
    let defaults = LinkConfig::default();
    let link = LinkConfig {
        latency_ms: if fields.contains_key("lat") { number("lat")? } else { defaults.latency_ms },
        bandwidth_gbs: if fields.contains_key("link") { number("link")? } else { defaults.bandwidth_gbs },
    };
    let name = text("name")?;
    if name.is_empty() {
        bail!("announcement has an empty name");
    }
    let device = DeviceConfig {
        name,
        memory_gib: number("mem")?,
        bandwidth_gbs: number("bw")?,
        tflops: number("tflops")?,
        ordinal: None,
        address: None,
        reserve_gib: if fields.contains_key("reserve") { Some(number("reserve")?) } else { None },
        link,
        offload: None,
        host: None,
        peer: None,
    };
    Ok((device, text("addr")?))
}

pub const APP_SERVICE_NAME: &str = "_inference-bench._tcp.local.";
pub const ANNOUNCEMENT_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct PeerGpu {
    pub device: DeviceConfig,
    pub gpu_name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Peer {
    pub version: u32,
    pub host: String,
    pub app_version: Option<String>,
    pub busy: bool,
    pub serving: Option<String>,
    pub control: Option<String>,
    pub gpus: Vec<PeerGpu>,
}

impl Peer {
    pub fn summary(&self) -> String {
        crate::hardware::summarize(&self.host, &self.gpus.iter().map(|g| g.gpu_name.as_str()).collect::<Vec<_>>())
    }

    pub fn memory_gib(&self) -> f64 {
        self.gpus.iter().map(|g| g.device.memory_gib).sum()
    }

    pub fn devices(&self) -> Vec<DeviceConfig> {
        self.gpus.iter().map(|g| g.device.clone()).collect()
    }
}

fn clean(text: &str) -> String {
    text.replace(['|', '\n'], " ")
}

pub fn encode_peer(peer: &Peer) -> Vec<String> {
    let mut txt = vec![format!("v={ANNOUNCEMENT_VERSION}"), format!("host={}", clean(&peer.host)), format!("busy={}", peer.busy as u8), format!("gpus={}", peer.gpus.len())];
    if let Some(a) = &peer.app_version {
        txt.push(format!("app={}", clean(a)));
    }
    if let Some(s) = &peer.serving {
        txt.push(format!("serving={}", clean(s)));
    }
    if let Some(c) = &peer.control {
        txt.push(format!("ctl={c}"));
    }
    for (i, g) in peer.gpus.iter().enumerate() {
        let d = &g.device;
        txt.push(format!(
            "g{i}={}|{}|{}|{}|{}|{}|{}|{}|{}",
            d.ordinal.unwrap_or(i),
            clean(&g.gpu_name),
            d.memory_gib,
            d.bandwidth_gbs,
            d.tflops,
            d.address.clone().unwrap_or_default(),
            d.link.latency_ms,
            d.link.bandwidth_gbs,
            d.reserve_gib.map_or(String::new(), |r| r.to_string())
        ));
        if let Some(p) = d.peer {
            let last = txt.last_mut().expect("gpu entry");
            last.push_str(&format!("|{}|{}", p.latency_ms, p.bandwidth_gbs));
        }
    }
    txt
}

fn field_map(txt: &[String]) -> Result<BTreeMap<String, String>> {
    let mut fields = BTreeMap::new();
    for entry in txt {
        let (k, v) = entry.split_once('=').ok_or_else(|| anyhow!("announcement entry {entry:?} has no '='"))?;
        fields.insert(k.to_string(), v.to_string());
    }
    Ok(fields)
}

fn parse_gpu(host: &str, i: usize, raw: &str) -> Result<PeerGpu> {
    let f: Vec<&str> = raw.split('|').collect();
    if f.len() < 8 {
        bail!("announcement gpu {i} has {} fields, expected 8 or more", f.len());
    }
    let number = |k: &str, v: &str| -> Result<f64> {
        let n: f64 = v.parse().map_err(|_| anyhow!("announcement gpu {i} field {k} is not a number"))?;
        if !n.is_finite() || n < 0.0 {
            bail!("announcement gpu {i} field {k} is out of range");
        }
        Ok(n)
    };
    let ordinal: usize = f[0].parse().map_err(|_| anyhow!("announcement gpu {i} has a bad ordinal"))?;
    let address = (!f[5].is_empty()).then(|| f[5].to_string());
    let reserve = match f.get(8) {
        Some(r) if !r.is_empty() => Some(number("reserve", r)?),
        _ => None,
    };
    let peer = match (f.get(9), f.get(10)) {
        (Some(l), Some(b)) => Some(LinkConfig { latency_ms: number("peer_lat", l)?, bandwidth_gbs: number("peer_bw", b)? }),
        _ => None,
    };
    Ok(PeerGpu {
        gpu_name: f[1].to_string(),
        device: DeviceConfig {
            name: crate::hardware::device_name(host, ordinal),
            memory_gib: number("mem", f[2])?,
            bandwidth_gbs: number("bw", f[3])?,
            tflops: number("tflops", f[4])?,
            ordinal: Some(ordinal),
            address,
            reserve_gib: reserve,
            link: LinkConfig { latency_ms: number("lat", f[6])?, bandwidth_gbs: number("link", f[7])? },
            offload: None,
            host: Some(host.to_string()),
            peer,
        },
    })
}

pub fn decode_peer(txt: &[String]) -> Result<Peer> {
    let fields = field_map(txt)?;
    let Some(version) = fields.get("v") else {
        let (device, address) = decode_announcement(txt)?;
        let host = device.name.clone();
        let mut device = device;
        device.address = Some(address);
        return Ok(Peer { version: 1, host, app_version: None, busy: false, serving: None, control: None, gpus: vec![PeerGpu { gpu_name: device.name.clone(), device }] });
    };
    let version: u32 = version.parse().map_err(|_| anyhow!("announcement version {version:?} is not a number"))?;
    let host = fields.get("host").cloned().filter(|h| !h.is_empty()).ok_or_else(|| anyhow!("announcement lacks host"))?;
    let count: usize = fields.get("gpus").ok_or_else(|| anyhow!("announcement lacks gpus"))?.parse().map_err(|_| anyhow!("announcement gpus is not a number"))?;
    if count > 64 {
        bail!("announcement lists {count} gpus");
    }
    let gpus = (0..count)
        .map(|i| parse_gpu(&host, i, fields.get(&format!("g{i}")).ok_or_else(|| anyhow!("announcement lacks g{i}"))?))
        .collect::<Result<Vec<_>>>()?;
    Ok(Peer {
        version,
        host,
        app_version: fields.get("app").cloned(),
        busy: fields.get("busy").is_some_and(|b| b == "1"),
        serving: fields.get("serving").cloned().filter(|s| !s.is_empty()),
        control: fields.get("ctl").cloned().filter(|s| !s.is_empty()),
        gpus,
    })
}

#[derive(Clone, Debug)]
pub struct Membership {
    members: Vec<Member>,
    ttl_ms: u64,
}

impl Membership {
    pub fn from_static(devices: &[DeviceConfig], ttl_ms: u64) -> Membership {
        let members = devices
            .iter()
            .map(|d| Member { device: d.clone(), address: d.address.clone(), source: Source::Static, last_seen_ms: 0 })
            .collect();
        Membership { members, ttl_ms }
    }

    pub fn members(&self) -> &[Member] {
        &self.members
    }

    pub fn names(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.device.name.as_str()).collect()
    }

    pub fn devices(&self) -> Vec<DeviceConfig> {
        self.members.iter().map(|m| m.device.clone()).collect()
    }

    pub fn apply(&mut self, event: Event) -> bool {
        match event {
            Event::Join { device, address, now_ms } => {
                if let Some(m) = self.members.iter_mut().find(|m| m.device.name == device.name) {
                    m.last_seen_ms = now_ms;
                    if m.source == Source::Static {
                        if m.address.as_deref() != Some(address.as_str()) {
                            m.address = Some(address);
                            return true;
                        }
                        return false;
                    }
                    let changed = m.device != device || m.address.as_deref() != Some(address.as_str());
                    m.device = device;
                    m.address = Some(address);
                    return changed;
                }
                self.members.push(Member { device, address: Some(address), source: Source::Mdns, last_seen_ms: now_ms });
                true
            }
            Event::Leave { name } => {
                let before = self.members.len();
                self.members.retain(|m| !(m.device.name == name && m.source == Source::Mdns));
                self.members.len() != before
            }
        }
    }

    pub fn expire(&mut self, now_ms: u64) -> Vec<String> {
        let ttl = self.ttl_ms;
        let mut gone = Vec::new();
        self.members.retain(|m| {
            let stale = m.source == Source::Mdns && now_ms.saturating_sub(m.last_seen_ms) > ttl;
            if stale {
                gone.push(m.device.name.clone());
            }
            !stale
        });
        gone
    }

    pub fn replan(&self, spec: &Spec, base: &PlacementConfig) -> Result<Placement> {
        let mut config = base.clone();
        let mut devices = self.devices();
        config.stages.retain(|s| devices.iter().any(|d| d.name == s.device));
        let mut ordered = Vec::new();
        for s in &config.stages {
            if let Some(i) = devices.iter().position(|d| d.name == s.device) {
                ordered.push(devices.remove(i));
            }
        }
        ordered.extend(devices);
        config.devices = ordered;
        config.place(spec)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replan {
    pub moved_stages: usize,
    pub unchanged: bool,
}

pub fn diff(old: &Placement, new: &Placement) -> Replan {
    let same = |a: &Placement, b: &Placement| {
        a.stages.len() == b.stages.len()
            && a.stages.iter().zip(&b.stages).all(|(x, y)| x.device_name == y.device_name && x.start == y.start && x.end == y.end)
    };
    let moved = new
        .stages
        .iter()
        .filter(|n| !old.stages.iter().any(|o| o.device_name == n.device_name && o.start == n.start && o.end == n.end))
        .count();
    Replan { moved_stages: moved, unchanged: same(old, new) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn dev(name: &str, gib: f64) -> DeviceConfig {
        DeviceConfig {
            name: name.into(),
            memory_gib: gib,
            bandwidth_gbs: 1800.0,
            tflops: 1000.0,
            ordinal: None,
            address: None,
            reserve_gib: Some(0.0),
            link: LinkConfig::default(),
            offload: None,
            host: None,
            peer: None,
        }
    }

    fn spec() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap()
    }

    fn base() -> PlacementConfig {
        PlacementConfig { context: 8192, slots: 8, ..PlacementConfig::default() }
    }

    #[test]
    fn pinned_stages_keep_their_order_whatever_order_nodes_joined_in() {
        let spec = spec();
        let mut m = Membership::from_static(&[], 5000);
        m.apply(Event::Join { device: dev("tail", 96.0), address: "h:2".into(), now_ms: 0 });
        m.apply(Event::Join { device: dev("head", 96.0), address: "h:1".into(), now_ms: 0 });
        let mut cfg = base();
        cfg.stages = vec![
            crate::placement::StageConfig { device: "head".into(), layers: [0, 40] },
            crate::placement::StageConfig { device: "tail".into(), layers: [40, 64] },
        ];
        let p = m.replan(&spec, &cfg).unwrap();
        assert_eq!(p.stages.iter().map(|s| s.device_name.as_str()).collect::<Vec<_>>(), ["head", "tail"]);
    }

    #[test]
    fn announcement_round_trips() {
        let mut d = dev("spark", 110.0);
        d.link = LinkConfig { latency_ms: 0.3, bandwidth_gbs: 1.1 };
        let txt = encode_announcement(&d, "10.0.0.5:7600");
        assert_eq!(decode_announcement(&txt).unwrap(), (d, "10.0.0.5:7600".to_string()));
    }

    fn peer(host: &str, n: usize) -> Peer {
        let gpus = (0..n)
            .map(|i| {
                let mut d = dev(&crate::hardware::device_name(host, i), 96.0 - i as f64);
                d.ordinal = Some(i);
                d.address = Some(format!("10.0.0.5:{}", 7600 + i));
                d.reserve_gib = None;
                d.host = Some(host.into());
                d.peer = Some(LinkConfig { latency_ms: 0.01, bandwidth_gbs: 25.0 });
                PeerGpu { device: d, gpu_name: "NVIDIA RTX PRO 6000".into() }
            })
            .collect();
        Peer { version: ANNOUNCEMENT_VERSION, host: host.into(), app_version: Some("0.4.0".into()), busy: true, serving: Some("Qwen3-0.6B".into()), control: Some("10.0.0.5:7599".into()), gpus }
    }

    #[test]
    fn a_two_gpu_peer_announcement_round_trips_per_gpu() {
        let p = peer("desk", 2);
        let txt = encode_peer(&p);
        assert_eq!(decode_peer(&txt).unwrap(), p);
        assert_eq!(p.summary(), "desk: 2 × RTX PRO 6000");
        assert_eq!(p.memory_gib(), 96.0 + 95.0);
        assert!(txt.iter().all(|e| e.len() <= 255));
    }

    #[test]
    fn a_v1_stage_announcement_decodes_as_a_one_gpu_peer() {
        let d = dev("spark", 110.0);
        let txt = encode_announcement(&d, "10.0.0.9:7600");
        let p = decode_peer(&txt).unwrap();
        assert_eq!((p.version, p.host.as_str(), p.busy, p.control.clone()), (1, "spark", false, None));
        assert_eq!(p.gpus.len(), 1);
        assert_eq!(p.gpus[0].device.address.as_deref(), Some("10.0.0.9:7600"));
        assert_eq!(p.gpus[0].device.memory_gib, 110.0);
    }

    #[test]
    fn a_newer_announcement_with_unknown_fields_still_decodes() {
        let mut txt = encode_peer(&peer("desk", 1));
        txt.push("future=1".into());
        txt.retain(|e| !e.starts_with("v="));
        txt.push("v=9".into());
        let p = decode_peer(&txt).unwrap();
        assert_eq!((p.version, p.gpus.len()), (9, 1));
    }

    #[test]
    fn a_v2_announcement_missing_a_gpu_is_rejected() {
        let mut txt = encode_peer(&peer("desk", 2));
        txt.retain(|e| !e.starts_with("g1="));
        assert!(decode_peer(&txt).is_err());
        let mut bad = encode_peer(&peer("desk", 1));
        bad.retain(|e| !e.starts_with("g0="));
        bad.push("g0=0|x|nan|1|1||0|1".into());
        assert!(decode_peer(&bad).is_err());
    }

    #[test]
    fn malformed_announcements_are_rejected() {
        assert!(decode_announcement(&["name=a".into()]).is_err());
        assert!(decode_announcement(&["junk".into()]).is_err());
        let mut txt = encode_announcement(&dev("a", 10.0), "x:1");
        txt.push("mem=-3".into());
        assert!(decode_announcement(&txt).is_err());
    }

    #[test]
    fn a_joining_node_is_added_once() {
        let mut m = Membership::from_static(&[dev("a", 20.0)], 5000);
        let join = Event::Join { device: dev("b", 20.0), address: "h:1".into(), now_ms: 10 };
        assert!(m.apply(join.clone()));
        assert!(!m.apply(join));
        assert_eq!(m.names(), ["a", "b"]);
    }

    #[test]
    fn static_members_never_leave_or_expire() {
        let mut m = Membership::from_static(&[dev("a", 20.0)], 100);
        assert!(!m.apply(Event::Leave { name: "a".into() }));
        assert!(m.expire(1_000_000).is_empty());
        assert_eq!(m.names(), ["a"]);
    }

    #[test]
    fn silent_discovered_members_expire() {
        let mut m = Membership::from_static(&[dev("a", 20.0)], 100);
        m.apply(Event::Join { device: dev("b", 20.0), address: "h:1".into(), now_ms: 0 });
        m.apply(Event::Join { device: dev("c", 20.0), address: "h:2".into(), now_ms: 90 });
        assert_eq!(m.expire(150), ["b"]);
        assert_eq!(m.names(), ["a", "c"]);
    }

    #[test]
    fn replanning_on_join_spreads_the_model_and_on_leave_fails_when_it_no_longer_fits() {
        let spec = spec();
        let mut m = Membership::from_static(&[dev("a", 20.0)], 5000);
        assert!(m.replan(&spec, &base()).is_err());
        m.apply(Event::Join { device: dev("b", 20.0), address: "h:1".into(), now_ms: 0 });
        m.apply(Event::Join { device: dev("c", 20.0), address: "h:2".into(), now_ms: 0 });
        let planned = m.replan(&spec, &base()).unwrap();
        assert!(planned.stages.len() >= 2);
        m.apply(Event::Leave { name: "c".into() });
        m.apply(Event::Leave { name: "b".into() });
        assert!(m.replan(&spec, &base()).is_err());
    }

    #[test]
    fn a_leaving_node_drops_its_pinned_stage_and_the_diff_reports_moves() {
        let spec = spec();
        let mut m = Membership::from_static(&[dev("a", 96.0)], 5000);
        let alone = m.replan(&spec, &base()).unwrap();
        m.apply(Event::Join { device: dev("b", 96.0), address: "h:1".into(), now_ms: 0 });
        let mut cfg = base();
        cfg.objective = crate::placement::Objective::Throughput;
        let both = m.replan(&spec, &cfg).unwrap();
        let change = diff(&alone, &both);
        assert!(!change.unchanged);
        assert_eq!(diff(&both, &both), Replan { moved_stages: 0, unchanged: true });
    }
}
