use crate::fit::{GB, GPU_UTILIZATION};
use crate::gpu::GpuStats;
use crate::settings::ClusterSettings;
use specrt::cluster::Peer;
use specrt::hardware::{Machine, gpu_label};
use specrt::placement::{DeviceConfig, LinkConfig};

#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub name: String,
    pub uuid: String,
    pub host: String,
    pub label: String,
    pub ordinal: Option<usize>,
    pub local: bool,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub display: bool,
    pub selected: bool,
    device: DeviceConfig,
}

impl Card {
    pub fn short(&self) -> String {
        match (self.local, self.ordinal) {
            (true, Some(o)) => format!("GPU {o}"),
            _ => specrt::hardware::display_name(&self.name),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inventory {
    pub host: String,
    pub cards: Vec<Card>,
}

fn stats_for<'a>(stats: &'a [GpuStats], uuid: &str, ordinal: usize) -> Option<&'a GpuStats> {
    if !uuid.is_empty()
        && let Some(s) = stats.iter().find(|s| s.uuid == uuid)
    {
        return Some(s);
    }
    stats.get(ordinal).filter(|s| uuid.is_empty() || s.uuid.is_empty())
}

fn usable(free: u64) -> f64 {
    free as f64 * GPU_UTILIZATION / GB
}

pub fn inventory(machine: Option<&Machine>, stats: &[GpuStats], peers: &[Peer], settings: &ClusterSettings) -> Inventory {
    let host = machine.map_or_else(specrt::hardware::host_name, |m| m.host.clone());
    let mut cards = Vec::new();
    match machine {
        Some(m) => {
            for (g, mut d) in m.gpus.iter().zip(m.devices(&[])) {
                let s = stats_for(stats, &g.uuid, g.ordinal);
                let free = s.map_or(g.free_bytes, GpuStats::available_bytes);
                d.memory_gib = usable(free);
                d.reserve_gib = Some(0.0);
                cards.push(Card {
                    selected: !settings.disabled_gpus.contains(&d.name),
                    name: d.name.clone(),
                    uuid: if g.uuid.is_empty() { s.map_or_else(String::new, |s| s.uuid.clone()) } else { g.uuid.clone() },
                    host: host.clone(),
                    label: g.label(),
                    ordinal: Some(g.ordinal),
                    local: true,
                    total_bytes: s.map_or(g.total_bytes, |s| s.total_bytes),
                    free_bytes: free,
                    display: s.is_some_and(|s| s.display),
                    device: d,
                });
            }
        }
        None => {
            for (i, s) in stats.iter().enumerate() {
                let mut d = crate::discover::pool_devices_on(&host, &[(i, s.available_bytes())]).remove(0);
                d.memory_gib = usable(s.available_bytes());
                d.host = Some(host.clone());
                d.peer = Some(specrt::hardware::STAGED_LINK);
                cards.push(Card {
                    selected: !settings.disabled_gpus.contains(&d.name),
                    name: d.name.clone(),
                    uuid: s.uuid.clone(),
                    host: host.clone(),
                    label: specrt::hardware::short_gpu(&s.name).to_string(),
                    ordinal: Some(i),
                    local: true,
                    total_bytes: s.total_bytes,
                    free_bytes: s.available_bytes(),
                    display: s.display,
                    device: d,
                });
            }
        }
    }
    for peer in peers.iter().filter(|p| p.host != host && settings.paired.contains(&p.host)) {
        for g in &peer.gpus {
            let bytes = (g.device.memory_gib * GB) as u64;
            cards.push(Card {
                selected: settings.peers_in_use.contains(&peer.host) && !settings.disabled_gpus.contains(&g.device.name),
                name: g.device.name.clone(),
                uuid: String::new(),
                host: peer.host.clone(),
                label: gpu_label(&g.gpu_name, g.backend),
                ordinal: g.device.ordinal,
                local: false,
                total_bytes: bytes,
                free_bytes: bytes,
                display: false,
                device: g.device.clone(),
            });
        }
    }
    Inventory { host, cards }
}

impl Inventory {
    pub fn local(&self) -> impl Iterator<Item = &Card> {
        self.cards.iter().filter(|c| c.local)
    }

    pub fn selected(&self) -> impl Iterator<Item = &Card> {
        self.cards.iter().filter(|c| c.selected)
    }

    pub fn free_local(&self) -> Vec<u64> {
        self.local().filter(|c| c.selected).map(|c| c.free_bytes).collect()
    }

    pub fn card(&self, name: &str) -> Option<&Card> {
        self.cards.iter().find(|c| c.name == name)
    }

    pub fn devices(&self, machine: Option<&Machine>, network: LinkConfig) -> Vec<DeviceConfig> {
        let mut out = Vec::new();
        let mut previous: Option<&Card> = None;
        for c in self.selected() {
            let mut d = c.device.clone();
            d.link = match previous {
                None => LinkConfig::default(),
                Some(p) if p.host != c.host => network,
                Some(p) => match (machine.filter(|_| c.local), p.ordinal, c.ordinal) {
                    (Some(m), Some(a), Some(b)) => m.link_between(a, b).config(),
                    _ if c.local => LinkConfig::default(),
                    _ => c.device.link,
                },
            };
            out.push(d);
            previous = Some(c);
        }
        out
    }

    pub fn where_landed(&self, landed: &[(String, u64)], planned: &[String]) -> Option<Landing> {
        if landed.is_empty() {
            return None;
        }
        let on: Vec<&Card> = self.local().filter(|c| landed.iter().any(|(u, _)| *u == c.uuid)).collect();
        let planned_here: Vec<&str> = planned.iter().map(String::as_str).filter(|p| self.local().any(|c| c.name == *p)).collect();
        let mut names: Vec<&str> = on.iter().map(|c| c.name.as_str()).collect();
        let mut want = planned_here.clone();
        names.sort_unstable();
        want.sort_unstable();
        let bytes: u64 = landed.iter().map(|l| l.1).sum();
        let where_ = on.iter().map(|c| c.short()).collect::<Vec<_>>().join(" + ");
        Some(Landing { text: format!("running on {where_} ({})", crate::library::human_size(bytes)), matches_plan: names == want })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Landing {
    pub text: String,
    pub matches_plan: bool,
}

pub fn planned_names(solved: &specrt::solve::Solved) -> Vec<String> {
    solved.placement.stages.iter().flat_map(|s| s.device_name.split('+').map(str::to_string).collect::<Vec<_>>()).collect()
}

pub fn toggle(settings: &mut ClusterSettings, inv: &Inventory, name: &str) {
    let Some(card) = inv.card(name) else { return };
    if card.local {
        if !settings.disabled_gpus.remove(name) {
            settings.disabled_gpus.insert(name.to_string());
        }
        return;
    }
    if card.selected {
        settings.disabled_gpus.insert(name.to_string());
        if !inv.cards.iter().any(|c| c.host == card.host && c.name != name && c.selected) {
            settings.peers_in_use.remove(&card.host);
        }
    } else {
        settings.disabled_gpus.remove(name);
        settings.peers_in_use.insert(card.host.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use specrt::cluster::{ANNOUNCEMENT_VERSION, PeerGpu};
    use specrt::hardware::{device_name, fake_gpus};

    fn stats(uuid: &str, used: u64, display: bool) -> GpuStats {
        GpuStats { name: "NVIDIA RTX PRO 6000".into(), used_bytes: used << 30, total_bytes: 96 << 30, util_percent: None, uuid: uuid.into(), own_bytes: 0, display }
    }

    fn peer(host: &str) -> Peer {
        let mut d = fake_gpus(host, 1, true).devices(&[]).remove(0);
        d.name = device_name(host, 0);
        d.memory_gib = 100.0;
        Peer { version: ANNOUNCEMENT_VERSION, host: host.into(), app_version: None, busy: false, serving: None, control: None, gpus: vec![PeerGpu { device: d, gpu_name: "NVIDIA GB10".into(), backend: specrt::hardware::GpuBackend::Cuda }] }
    }

    #[test]
    fn local_cards_take_free_memory_and_the_display_flag_by_uuid_not_by_position() {
        let m = fake_gpus("desk", 2, true);
        let s = vec![stats("GPU-fake-1", 30, false), stats("GPU-fake-0", 10, true)];
        let inv = inventory(Some(&m), &s, &[], &ClusterSettings::default());
        assert_eq!(inv.cards.len(), 2);
        assert_eq!((inv.cards[0].free_bytes, inv.cards[0].display), (86 << 30, true));
        assert_eq!((inv.cards[1].free_bytes, inv.cards[1].display), (66 << 30, false));
        assert_eq!(inv.cards[1].short(), "GPU 1");
    }

    #[test]
    fn a_switched_off_gpu_stays_listed_but_is_not_a_placement_device() {
        let m = fake_gpus("desk", 2, true);
        let mut settings = ClusterSettings::default();
        settings.disabled_gpus.insert("desk/gpu0".into());
        let inv = inventory(Some(&m), &[], &[], &settings);
        assert_eq!(inv.cards.len(), 2);
        assert!(!inv.cards[0].selected && inv.cards[1].selected);
        let devices = inv.devices(Some(&m), settings.network_link());
        assert_eq!(devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["desk/gpu1"]);
        assert_eq!(devices[0].ordinal, Some(1));
        assert_eq!(devices[0].link, LinkConfig::default());
    }

    #[test]
    fn only_paired_peers_are_listed_and_only_peers_in_use_are_devices() {
        let m = fake_gpus("desk", 1, true);
        let peers = vec![peer("spark"), peer("stranger")];
        let mut settings = ClusterSettings::default();
        settings.paired.insert("spark".into());
        let inv = inventory(Some(&m), &[], &peers, &settings);
        assert_eq!(inv.cards.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["desk/gpu0", "spark/gpu0"]);
        assert!(!inv.cards[1].selected);
        assert_eq!(inv.devices(Some(&m), settings.network_link()).len(), 1);
        toggle(&mut settings, &inv, "spark/gpu0");
        assert!(settings.peers_in_use.contains("spark"));
        let inv = inventory(Some(&m), &[], &peers, &settings);
        let devices = inv.devices(Some(&m), settings.network_link());
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[1].link, settings.network_link());
        toggle(&mut settings, &inv, "spark/gpu0");
        assert!(!settings.peers_in_use.contains("spark"));
    }

    #[test]
    fn toggling_a_local_gpu_writes_the_same_setting_the_hardware_page_reads() {
        let m = fake_gpus("desk", 2, true);
        let mut settings = ClusterSettings::default();
        let inv = inventory(Some(&m), &[], &[], &settings);
        toggle(&mut settings, &inv, "desk/gpu0");
        assert!(settings.disabled_gpus.contains("desk/gpu0"));
        toggle(&mut settings, &inv, "desk/gpu0");
        assert!(settings.disabled_gpus.is_empty());
    }

    #[test]
    fn landing_is_named_by_uuid_and_checked_against_the_plan() {
        let m = fake_gpus("desk", 2, true);
        let inv = inventory(Some(&m), &[], &[], &ClusterSettings::default());
        let landed = vec![("GPU-fake-1".to_string(), 40 << 30)];
        let l = inv.where_landed(&landed, &["desk/gpu1".into()]).unwrap();
        assert!(l.text.starts_with("running on GPU 1 ("), "{}", l.text);
        assert!(l.matches_plan);
        assert!(!inv.where_landed(&landed, &["desk/gpu0".into()]).unwrap().matches_plan);
        let both = vec![("GPU-fake-0".to_string(), 1 << 30), ("GPU-fake-1".to_string(), 1 << 30)];
        let l = inv.where_landed(&both, &["desk/gpu0".into(), "desk/gpu1".into()]).unwrap();
        assert!(l.text.starts_with("running on GPU 0 + GPU 1"), "{}", l.text);
        assert!(l.matches_plan);
        assert!(inv.where_landed(&[], &[]).is_none());
    }

    #[test]
    fn with_no_probe_the_cards_come_from_the_gpu_readings() {
        let s = vec![stats("GPU-a", 10, true), stats("GPU-b", 0, false)];
        let inv = inventory(None, &s, &[], &ClusterSettings::default());
        assert_eq!(inv.free_local(), vec![86 << 30, 96 << 30]);
        assert!(inv.cards[0].display);
    }

    #[test]
    fn cards_from_gpu_readings_belong_to_this_host_so_they_can_pair_for_tp() {
        let s = vec![stats("GPU-a", 50, false), stats("GPU-b", 20, false)];
        let inv = inventory(None, &s, &[], &ClusterSettings::default());
        let devices = inv.devices(None, LinkConfig::default());
        assert!(devices.iter().all(|d| d.host.as_deref() == Some(inv.host.as_str())), "{devices:?}");
    }
}
