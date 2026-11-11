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
    };
    Ok((device, text("addr")?))
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
