use crate::settings::ClusterSettings;
use anyhow::{Context, Result, bail};
use specrt::cluster::Peer;
use specrt::hardware::{self, Machine, PEER_LINK, STAGED_LINK, host_of};
use specrt::host::{HostedStage, release_stages, request_stages, spawn_stage};
use specrt::placement::{DeviceConfig, Objective, Placement, PlacementConfig, RemoteConfig, StageConfig};
use specrt::spec::Spec;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type ClusterSlot = Arc<Mutex<Option<PlacementConfig>>>;

pub fn key_path(dir: &Path) -> PathBuf {
    dir.join("cluster.key")
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    pub peers: BTreeSet<String>,
    pub excluded: BTreeSet<String>,
}

impl Selection {
    pub fn toggle_peer(&mut self, host: &str) {
        if !self.peers.remove(host) {
            self.peers.insert(host.to_string());
        }
    }

    pub fn toggle_gpu(&mut self, device: &str) {
        if !self.excluded.remove(device) {
            self.excluded.insert(device.to_string());
        }
    }

    pub fn includes(&self, device: &str) -> bool {
        !self.excluded.contains(device)
    }
}

pub fn candidate_devices(local: &Machine, peers: &[Peer], selection: &Selection, network: specrt::placement::LinkConfig) -> Vec<DeviceConfig> {
    let skip: Vec<usize> = local.gpus.iter().map(|g| g.ordinal).filter(|o| !selection.includes(&hardware::device_name(&local.host, *o))).collect();
    let mut devices = local.devices(&skip);
    for peer in peers.iter().filter(|p| selection.peers.contains(&p.host)) {
        let mut first = true;
        for g in peer.gpus.iter().filter(|g| selection.includes(&g.device.name)) {
            let mut d = g.device.clone();
            if first {
                d.link = network;
                first = false;
            }
            devices.push(d);
        }
    }
    devices
}

#[derive(Clone, Debug)]
pub struct Preview {
    pub devices: Vec<DeviceConfig>,
    pub latency: Result<Placement, String>,
    pub throughput: Result<Placement, String>,
}

pub fn preview(spec: &Spec, devices: Vec<DeviceConfig>, ctx: usize, slots: usize) -> Preview {
    let run = |objective: Objective| {
        let config = PlacementConfig { objective, context: ctx, slots, devices: devices.clone(), ..PlacementConfig::default() };
        config.place(spec).map_err(|e| format!("{e:#}"))
    };
    Preview { latency: run(Objective::Latency), throughput: run(Objective::Throughput), devices }
}

pub fn runnable_plan(spec: &Spec, devices: &[DeviceConfig], ctx: usize, slots: usize, objective: Objective) -> Result<Placement, String> {
    let flat: Vec<DeviceConfig> = devices.iter().cloned().map(|mut d| {
        d.host = None;
        d
    }).collect();
    let config = PlacementConfig { objective, context: ctx, slots, devices: flat, ..PlacementConfig::default() };
    let placement = config.place(spec).map_err(|e| format!("{e:#}"))?;
    runnable(&placement)?;
    Ok(placement)
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageRow {
    pub label: String,
    pub layers: String,
    pub gib: f64,
    pub of_gib: f64,
    pub step_ms: f64,
    pub hop_ms: f64,
    pub hop: String,
    pub needs_tensor_parallel: bool,
}

pub fn hop_label(previous: Option<&DeviceConfig>, current: &DeviceConfig, local_host: &str) -> String {
    let Some(prev) = previous else { return "first stage".into() };
    let (a, b) = (host_of(&prev.name), host_of(&current.name));
    if a != b {
        return "network".into();
    }
    if a == local_host || current.link == PEER_LINK || current.link == STAGED_LINK {
        return if current.link == STAGED_LINK { "host staged".into() } else { "peer copy".into() };
    }
    "same host".into()
}

pub fn stage_rows(placement: &Placement, devices: &[DeviceConfig], local_host: &str) -> Vec<StageRow> {
    const GIB: f64 = (1u64 << 30) as f64;
    let mut previous: Option<&DeviceConfig> = None;
    placement
        .stages
        .iter()
        .map(|s| {
            let device = devices.iter().find(|d| d.name == s.device_name);
            let hop = match device {
                Some(d) => hop_label(previous, d, local_host),
                None => String::new(),
            };
            previous = device;
            StageRow {
                label: if s.tp > 1 { s.label() } else { hardware::display_name(&s.device_name) },
                layers: format!("{}..{}", s.start, s.end),
                gib: s.used_bytes() as f64 / GIB,
                of_gib: s.memory_bytes as f64 / GIB,
                step_ms: s.compute_secs * 1e3,
                hop_ms: s.hop_secs * 1e3,
                hop,
                needs_tensor_parallel: s.tp > 1,
            }
        })
        .collect()
}

pub const TP_NOTE: &str = "needs tensor parallel (#658)";

pub fn runnable(placement: &Placement) -> Result<(), String> {
    placement.ensure_runnable().map_err(|e| format!("{e:#}"))
}

pub fn bottleneck(placement: &Placement) -> Option<String> {
    let worst = placement.stages.iter().max_by(|a, b| a.total_secs().total_cmp(&b.total_secs()))?;
    Some(hardware::display_name(&worst.device_name))
}

pub struct Prepared {
    pub placement: PlacementConfig,
    pub local: Vec<HostedStage>,
    pub peers: Vec<String>,
    pub key: String,
}

impl Prepared {
    pub fn release(&mut self) {
        self.local.clear();
        for control in std::mem::take(&mut self.peers) {
            let _ = release_stages(&control, &self.key);
        }
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        self.release();
    }
}

pub struct Run<'a> {
    pub model_id: &'a str,
    pub ctx: usize,
    pub slots: usize,
    pub placement: &'a Placement,
    pub devices: &'a [DeviceConfig],
    pub local: &'a Machine,
    pub peers: &'a [Peer],
    pub settings: &'a ClusterSettings,
    pub key_file: &'a Path,
}

pub fn is_remote(run: &Run) -> bool {
    run.placement.stages.iter().any(|s| host_of(&s.device_name) != run.local.host)
}

pub fn prepare(run: &Run) -> Result<Prepared> {
    let key = specrt::pairing::ensure_key(run.key_file)?;
    let mut devices: Vec<DeviceConfig> = run.devices.iter().cloned().map(|mut d| {
        d.host = None;
        d
    }).collect();
    let mut stages = Vec::new();
    let mut local = Vec::new();
    let mut used_peers = Vec::new();
    if !is_remote(run) {
        if let [only] = run.placement.stages.as_slice()
            && only.ordinal != 0
        {
            bail!("the model fits on {} alone, and a single GPU run always uses GPU 0: include GPU 0 or add a second GPU", hardware::display_name(&only.device_name));
        }
        stages = run.placement.stages.iter().map(|s| StageConfig { device: s.device_name.clone(), layers: [s.start, s.end] }).collect();
        let placement = PlacementConfig { objective: run.placement.objective, context: run.ctx, slots: run.slots, devices, stages, ..PlacementConfig::default() };
        return Ok(Prepared { placement, local, peers: used_peers, key });
    }
    let result = (|| -> Result<()> {
        for s in &run.placement.stages {
            stages.push(StageConfig { device: s.device_name.clone(), layers: [s.start, s.end] });
        }
        let hosts: BTreeSet<&str> = run.placement.stages.iter().map(|s| host_of(&s.device_name)).collect();
        for host in hosts {
            let gpus: Vec<(String, usize)> = run
                .placement
                .stages
                .iter()
                .filter(|s| host_of(&s.device_name) == host)
                .map(|s| (s.device_name.clone(), s.ordinal))
                .collect();
            let addrs: Vec<String> = if host == run.local.host {
                gpus.iter()
                    .map(|(_, gpu)| {
                        let hosted = spawn_stage(run.model_id, run.ctx, run.slots, *gpu, key.as_bytes(), "127.0.0.1:0")?;
                        let addr = hosted.server.addr().to_string();
                        local.push(hosted);
                        Ok(addr)
                    })
                    .collect::<Result<_>>()?
            } else {
                let peer = run.peers.iter().find(|p| p.host == host).with_context(|| format!("{host} is no longer on the network"))?;
                let control = peer.control.clone().with_context(|| format!("{host} announced no control address"))?;
                let ordinals: Vec<usize> = gpus.iter().map(|(_, g)| *g).collect();
                let addrs = request_stages(&control, &key, run.model_id, run.ctx, run.slots, &ordinals).with_context(|| format!("{host} refused the stage request"))?;
                if addrs.len() != ordinals.len() {
                    bail!("{host} opened {} stage servers for {} GPUs", addrs.len(), ordinals.len());
                }
                used_peers.push(control);
                addrs
            };
            for ((name, _), addr) in gpus.iter().zip(addrs) {
                if let Some(d) = devices.iter_mut().find(|d| &d.name == name) {
                    d.address = Some(addr);
                }
            }
        }
        Ok(())
    })();
    let mut prepared = Prepared { placement: PlacementConfig::default(), local, peers: used_peers, key };
    if let Err(e) = result {
        prepared.release();
        return Err(e);
    }
    devices.retain(|d| stages.iter().any(|s| s.device == d.name));
    prepared.placement = PlacementConfig {
        objective: run.placement.objective,
        context: run.ctx,
        slots: run.slots,
        devices,
        stages,
        mdns_port: run.settings.mdns_port,
        remote: Some(RemoteConfig { key: None, key_file: Some(run.key_file.to_path_buf()), timeout_ms: 600_000, member_ttl_ms: 30_000, connect_wait_ms: 15_000 }),
        ..PlacementConfig::default()
    };
    Ok(prepared)
}

pub fn serving_factory(slot: ClusterSlot) -> crate::engine::ServingFactory {
    Arc::new(move |mut opts| {
        if let Some(p) = slot.lock().unwrap().clone() {
            opts.placement = Some(p);
        }
        Some(Box::new(crate::serving::ServingTarget::new(opts)) as Box<dyn crate::target::Target>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use specrt::cluster::{ANNOUNCEMENT_VERSION, PeerGpu};
    use specrt::hardware::fake_gpus;

    fn spec() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/qwen3.8-27b.toml")).unwrap()
    }

    fn fixture() -> String {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures/qwen3.8-27b.toml").display().to_string()
    }

    fn remote_peer(host: &str, gib: f64) -> Peer {
        let machine = fake_gpus(host, 1, true);
        let mut d = machine.devices(&[]).remove(0);
        d.memory_gib = gib;
        d.address = Some(":7600".into());
        Peer { version: ANNOUNCEMENT_VERSION, host: host.into(), app_version: Some("x".into()), busy: false, serving: None, control: Some("10.0.0.7:7599".into()), gpus: vec![PeerGpu { device: d, gpu_name: "NVIDIA GB10".into() }] }
    }

    fn small_local() -> Machine {
        let mut m = fake_gpus("desk", 2, true);
        for g in &mut m.gpus {
            g.free_bytes = 12 << 30;
        }
        m
    }

    #[test]
    fn two_local_gpus_and_one_remote_are_all_placement_devices_in_both_objectives() {
        let local = small_local();
        let peers = vec![remote_peer("spark", 14.0)];
        let mut sel = Selection::default();
        sel.toggle_peer("spark");
        let net = ClusterSettings::default().network_link();
        let devices = candidate_devices(&local, &peers, &sel, net);
        assert_eq!(devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["desk/gpu0", "desk/gpu1", "spark/gpu0"]);
        assert_eq!(devices[1].link, PEER_LINK);
        assert_eq!(devices[2].link, net);
        let p = preview(&spec(), devices.clone(), 2048, 1);
        for placement in [p.latency.as_ref().unwrap(), p.throughput.as_ref().unwrap()] {
            assert_eq!(placement.stages.iter().map(|s| s.layers()).sum::<usize>(), placement.layers);
            assert!(placement.step_secs > 0.0 && placement.bottleneck_secs > 0.0);
        }
        let grouped = stage_rows(p.latency.as_ref().unwrap(), &devices, "desk");
        assert!(grouped.iter().any(|r| r.needs_tensor_parallel && r.label.contains("GPU 0+1 (tp 2)")));
        assert!(runnable(p.latency.as_ref().unwrap()).is_err());
        let placement = runnable_plan(&spec(), &devices, 2048, 1, Objective::Latency).unwrap();
        assert!(placement.stages.len() >= 2);
        let rows = stage_rows(&placement, &devices, "desk");
        assert!(rows.iter().all(|r| !r.needs_tensor_parallel));
        assert_eq!(rows[0].label, "desk / GPU 0");
        assert_eq!(rows[0].hop, "first stage");
        assert_eq!(rows[1].hop, "peer copy");
        assert!(bottleneck(&placement).is_some());
        if let Some(r) = rows.iter().find(|r| r.label.starts_with("spark")) {
            assert_eq!(r.hop, "network");
        }
    }

    #[test]
    fn excluding_a_gpu_removes_it_from_the_candidates_and_an_unselected_peer_never_appears() {
        let local = small_local();
        let peers = vec![remote_peer("spark", 14.0)];
        let mut sel = Selection::default();
        sel.toggle_gpu("desk/gpu1");
        let devices = candidate_devices(&local, &peers, &sel, ClusterSettings::default().network_link());
        assert_eq!(devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["desk/gpu0"]);
        sel.toggle_gpu("desk/gpu1");
        assert!(sel.includes("desk/gpu1"));
    }

    #[test]
    fn a_model_that_fits_nowhere_reports_the_planner_error_for_both_objectives() {
        let mut local = fake_gpus("desk", 1, true);
        local.gpus[0].free_bytes = 2 << 30;
        let devices = candidate_devices(&local, &[], &Selection::default(), ClusterSettings::default().network_link());
        let p = preview(&spec(), devices, 2048, 1);
        assert!(p.latency.is_err() && p.throughput.is_err());
    }

    #[test]
    fn a_local_only_split_prepares_an_in_process_placement_without_peers() {
        let mut local = small_local();
        for g in &mut local.gpus {
            g.free_bytes = 20 << 30;
        }
        let devices = candidate_devices(&local, &[], &Selection::default(), ClusterSettings::default().network_link());
        let placement = runnable_plan(&spec(), &devices, 2048, 1, Objective::Latency).unwrap();
        let dir = std::env::temp_dir().join(format!("cl-{}", std::process::id()));
        let run = Run { model_id: "m", ctx: 2048, slots: 1, placement: &placement, devices: &devices, local: &local, peers: &[], settings: &ClusterSettings::default(), key_file: &dir.join("cluster.key") };
        assert!(!is_remote(&run));
        let prepared = prepare(&run).unwrap();
        assert!(prepared.placement.remote.is_none());
        assert_eq!(prepared.placement.stages.len(), placement.stages.len());
        assert!(prepared.peers.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_peer_that_vanished_fails_the_prepare_with_its_name() {
        let local = small_local();
        let peers = vec![remote_peer("spark", 14.0)];
        let mut sel = Selection::default();
        sel.toggle_peer("spark");
        let devices = candidate_devices(&local, &peers, &sel, ClusterSettings::default().network_link());
        let placement = runnable_plan(&spec(), &devices, 2048, 1, Objective::Latency).unwrap();
        assert!(placement.stages.iter().any(|s| s.device_name.starts_with("spark")));
        let dir = std::env::temp_dir().join(format!("cl2-{}", std::process::id()));
        let run = Run { model_id: &fixture(), ctx: 2048, slots: 1, placement: &placement, devices: &devices, local: &local, peers: &[], settings: &ClusterSettings::default(), key_file: &dir.join("cluster.key") };
        let err = prepare(&run).err().unwrap();
        assert!(format!("{err:#}").contains("spark"), "{err:#}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
