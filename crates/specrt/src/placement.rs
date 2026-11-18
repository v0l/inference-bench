use crate::anatomy::{LayerKind, ModelAnatomy, quant_bytes};
use crate::spec::Spec;
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

const GIB: f64 = (1u64 << 30) as f64;
const DEFAULT_RESERVE_GIB: f64 = 2.0;
const FLOPS_PER_PARAM: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkConfig {
    #[serde(default = "default_latency_ms")]
    pub latency_ms: f64,
    #[serde(default = "default_link_gbs")]
    pub bandwidth_gbs: f64,
}

fn default_latency_ms() -> f64 {
    0.01
}

fn default_link_gbs() -> f64 {
    25.0
}

impl Default for LinkConfig {
    fn default() -> Self {
        LinkConfig { latency_ms: default_latency_ms(), bandwidth_gbs: default_link_gbs() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffloadConfig {
    pub host_gib: f64,
    pub bandwidth_gbs: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConfig {
    pub name: String,
    pub memory_gib: f64,
    pub bandwidth_gbs: f64,
    pub tflops: f64,
    #[serde(default)]
    pub ordinal: Option<usize>,
    #[serde(default)]
    pub address: Option<String>,
    #[serde(default)]
    pub reserve_gib: Option<f64>,
    #[serde(default)]
    pub link: LinkConfig,
    #[serde(default)]
    pub offload: Option<OffloadConfig>,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub peer: Option<LinkConfig>,
    #[serde(default)]
    pub wgpu: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Objective {
    #[default]
    Latency,
    Throughput,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Discovery {
    #[default]
    Static,
    Mdns,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageConfig {
    pub device: String,
    pub layers: [usize; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteConfig {
    pub key: Option<String>,
    pub key_file: Option<std::path::PathBuf>,
    #[serde(default = "default_remote_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_member_ttl_ms")]
    pub member_ttl_ms: u64,
    #[serde(default = "default_connect_wait_ms")]
    pub connect_wait_ms: u64,
}

fn default_connect_wait_ms() -> u64 {
    15_000
}

fn default_remote_timeout_ms() -> u64 {
    600_000
}

fn default_member_ttl_ms() -> u64 {
    30_000
}

impl RemoteConfig {
    pub fn key(&self) -> Result<Vec<u8>> {
        crate::stage::read_key(&self.key, &self.key_file)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlacementConfig {
    pub objective: Objective,
    pub context: usize,
    pub slots: usize,
    pub devices: Vec<DeviceConfig>,
    pub stages: Vec<StageConfig>,
    pub micro_batch: Option<usize>,
    pub discovery: Discovery,
    pub mdns_port: u16,
    pub mdns_interface: Option<String>,
    pub remote: Option<RemoteConfig>,
    pub max_tp: Option<usize>,
    pub weight_bytes: Option<u64>,
    pub kv_pool_bytes: Option<u64>,
}

pub const REMOTE_MICRO_BATCH: usize = 256;

impl PlacementConfig {
    pub fn micro_batch_for_remote(&self) -> usize {
        self.micro_batch.unwrap_or(REMOTE_MICRO_BATCH)
    }

    pub fn micro_batch_in_process(&self) -> usize {
        self.micro_batch.unwrap_or(0)
    }
}

impl Default for PlacementConfig {
    fn default() -> Self {
        PlacementConfig { objective: Objective::Latency, context: 8192, slots: 1, devices: Vec::new(), stages: Vec::new(), micro_batch: None, discovery: Discovery::Static, mdns_port: 5353, mdns_interface: None, remote: None, max_tp: None, weight_bytes: None, kv_pool_bytes: None }
    }
}

impl PlacementConfig {
    pub fn apply_peer_probes(&mut self, probes: &[crate::link::PairProbe]) {
        for i in 0..self.devices.len() {
            let Some(host) = self.devices[i].host.clone() else { continue };
            if self.devices[i].peer.is_some() {
                continue;
            }
            let ordinal = self.devices[i].ordinal.unwrap_or(i);
            let measured: Vec<&crate::link::PairProbe> = probes
                .iter()
                .filter(|p| (p.a == ordinal || p.b == ordinal) && self.devices.iter().enumerate().any(|(j, d)| j != i && d.host.as_deref() == Some(&host) && d.ordinal.unwrap_or(j) == if p.a == ordinal { p.b } else { p.a }))
                .collect();
            if measured.is_empty() {
                continue;
            }
            self.devices[i].peer = Some(LinkConfig {
                latency_ms: measured.iter().map(|p| p.latency_ms).fold(0.0, f64::max),
                bandwidth_gbs: measured.iter().map(|p| p.bandwidth_gbs).fold(f64::INFINITY, f64::min),
            });
        }
    }

    pub fn apply_network_rtts(&mut self, rtts_ms: &[(String, f64)]) {
        for d in &mut self.devices {
            if let Some((_, rtt)) = rtts_ms.iter().find(|(a, _)| Some(a) == d.address.as_ref()) {
                d.link.latency_ms = rtt / 2.0;
            }
        }
    }

    pub fn place(&self, spec: &Spec) -> Result<Placement> {
        ensure!(self.slots >= 1 && self.context >= 1, "placement needs at least one slot and one token of context");
        let mut model = CostModel::new(spec, self.context, self.slots);
        if let Some(bytes) = self.weight_bytes {
            model.rescale_weights(bytes);
        }
        if let Some(bytes) = self.kv_pool_bytes {
            model.scale_kv_pool(bytes);
        }
        if let Some(tp) = self.max_tp {
            model.max_tp = tp.max(1);
        }
        let devices: Vec<Device> = self.devices.iter().map(Device::from_config).collect();
        if self.stages.is_empty() {
            return plan(&model, &devices, self.objective);
        }
        let mut ranges = Vec::new();
        for s in &self.stages {
            let Some(device) = self.devices.iter().position(|d| d.name == s.device) else {
                bail!("stage names device {:?} which is not in [[placement.devices]]", s.device)
            };
            ranges.push((device, s.layers[0], s.layers[1]));
        }
        evaluate(&model, &devices, &ranges, self.objective)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub name: String,
    pub ordinal: Option<usize>,
    pub memory_bytes: u64,
    pub reserve_bytes: u64,
    pub bandwidth: f64,
    pub flops: f64,
    pub link_latency: f64,
    pub link_bandwidth: f64,
    pub offload_bytes: u64,
    pub offload_bandwidth: f64,
    pub host: Option<String>,
    pub peer_latency: f64,
    pub peer_bandwidth: f64,
    pub tp: usize,
    pub ordinals: Vec<usize>,
    pub allreduce_latency: f64,
    pub allreduce_bandwidth: f64,
    pub wgpu: bool,
}

impl Device {
    pub fn group(members: &[(usize, &Device)]) -> Device {
        let first = members[0].1;
        if members.len() == 1 {
            let mut d = first.clone();
            d.ordinals = vec![first.ordinal.unwrap_or(members[0].0)];
            return d;
        }
        let inner = &members[1..];
        Device {
            name: members.iter().map(|m| m.1.name.as_str()).collect::<Vec<_>>().join("+"),
            ordinal: Some(first.ordinal.unwrap_or(members[0].0)),
            memory_bytes: members.iter().map(|m| m.1.memory_bytes).min().unwrap_or(0),
            reserve_bytes: members.iter().map(|m| m.1.reserve_bytes).max().unwrap_or(0),
            bandwidth: members.iter().map(|m| m.1.bandwidth).fold(f64::INFINITY, f64::min),
            flops: members.iter().map(|m| m.1.flops).fold(f64::INFINITY, f64::min),
            link_latency: first.link_latency,
            link_bandwidth: first.link_bandwidth,
            offload_bytes: members.iter().map(|m| m.1.offload_bytes).min().unwrap_or(0),
            offload_bandwidth: members.iter().map(|m| m.1.offload_bandwidth).fold(f64::INFINITY, f64::min),
            host: first.host.clone(),
            peer_latency: first.peer_latency,
            peer_bandwidth: first.peer_bandwidth,
            tp: members.len(),
            ordinals: members.iter().map(|m| m.1.ordinal.unwrap_or(m.0)).collect(),
            allreduce_latency: inner.iter().map(|m| m.1.peer_latency).fold(0.0, f64::max),
            allreduce_bandwidth: inner.iter().map(|m| m.1.peer_bandwidth).fold(f64::INFINITY, f64::min),
            wgpu: first.wgpu,
        }
    }

    pub fn from_config(c: &DeviceConfig) -> Device {
        let offload = c.offload.unwrap_or(OffloadConfig { host_gib: 0.0, bandwidth_gbs: 1.0 });
        Device {
            name: c.name.clone(),
            ordinal: c.ordinal,
            memory_bytes: (c.memory_gib * GIB) as u64,
            reserve_bytes: (c.reserve_gib.unwrap_or(DEFAULT_RESERVE_GIB) * GIB) as u64,
            bandwidth: c.bandwidth_gbs * 1e9,
            flops: c.tflops * 1e12,
            link_latency: c.link.latency_ms * 1e-3,
            link_bandwidth: c.link.bandwidth_gbs * 1e9,
            offload_bytes: (offload.host_gib * GIB) as u64,
            offload_bandwidth: offload.bandwidth_gbs * 1e9,
            host: c.host.clone(),
            peer_latency: c.peer.map_or(c.link.latency_ms, |p| p.latency_ms) * 1e-3,
            peer_bandwidth: c.peer.map_or(c.link.bandwidth_gbs, |p| p.bandwidth_gbs) * 1e9,
            tp: 1,
            ordinals: Vec::new(),
            allreduce_latency: 0.0,
            allreduce_bandwidth: f64::INFINITY,
            wgpu: c.wgpu,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerCost {
    kind: LayerKind,
    weight_bytes: u64,
    read_bytes: f64,
    flops: f64,
    kv_bytes: u64,
    state_bytes: u64,
}

#[derive(Clone)]
pub struct CostModel {
    pub name: String,
    pub context: usize,
    pub slots: usize,
    layers: Vec<LayerCost>,
    pub wgpu_blocker: Option<String>,
    embedding_bytes: u64,
    head_bytes: u64,
    head_flops: f64,
    norm_bytes: u64,
    pub hidden: usize,
    pub hop_bytes_per_element: usize,
    head_counts: Vec<(&'static str, usize)>,
    pub max_tp: usize,
}

const ALLREDUCES_PER_LAYER: f64 = 2.0;
const ALLREDUCE_BYTES_PER_ELEMENT: usize = 2;

impl CostModel {
    pub fn new(spec: &Spec, context: usize, slots: usize) -> CostModel {
        let anatomy = ModelAnatomy::new(spec, 0, Some(context));
        let f16_kv_per_token = (2 * spec.attention.kv_heads * spec.attention.head_dim * 2) as u64;
        let fp8_kv = spec.runtime.kv_dtype.is_fp8();
        let f16_kv_layers = spec.kv_f16_layers();
        let kv_per_token = |attention_index: usize| -> u64 {
            if !fp8_kv {
                return f16_kv_per_token;
            }
            let rows = (2 * spec.attention.kv_heads) as u64;
            let head = spec.attention.head_dim as u64;
            if attention_index < f16_kv_layers { rows * head * 2 } else { rows * (head + 4) }
        };
        let mut attention_index = 0usize;
        let state_per_seq = match &spec.linear_attention {
            Some(la) => {
                let ssm = la.value_heads * la.key_head_dim * la.key_head_dim;
                let conv = la.conv_dim() * (la.conv_kernel - 1);
                ((ssm + conv) * 4) as u64
            }
            None => 0,
        };
        let rows = slots as f64;
        let layers = anatomy
            .layers
            .iter()
            .map(|l| {
                let (kv_bytes, state_bytes) = match l.kind {
                    LayerKind::DeltaNet => (0, state_per_seq),
                    _ => {
                        let bytes = kv_per_token(attention_index) * context as u64;
                        attention_index += 1;
                        (bytes, 0)
                    }
                };
                let traffic = kv_bytes as f64 * rows + 2.0 * state_bytes as f64 * rows;
                LayerCost {
                    kind: l.kind,
                    weight_bytes: l.bytes,
                    read_bytes: l.active_bytes as f64 + traffic,
                    flops: FLOPS_PER_PARAM * l.active_params as f64 * rows,
                    kv_bytes,
                    state_bytes,
                }
            })
            .collect();
        let matrix = spec.dims.vocab as u64 * spec.dims.hidden as u64;
        CostModel {
            name: spec.name.clone(),
            context,
            slots,
            layers,
            wgpu_blocker: crate::wgsl::unsupported(&Spec { mtp: None, ..spec.clone() }),
            embedding_bytes: 2 * matrix,
            head_bytes: quant_bytes(matrix, anatomy.head_quant),
            head_flops: FLOPS_PER_PARAM * matrix as f64 * rows,
            norm_bytes: 2 * spec.dims.hidden as u64,
            hidden: spec.dims.hidden,
            hop_bytes_per_element: 4,
            max_tp: usize::MAX,
            head_counts: {
                let mut counts = vec![("attention heads", spec.attention.heads), ("kv heads", spec.attention.kv_heads)];
                if let Some(la) = &spec.linear_attention {
                    counts.push(("delta-net key heads", la.key_heads));
                    counts.push(("delta-net value heads", la.value_heads));
                }
                counts
            },
        }
    }

    pub fn rescale_weights(&mut self, weight_bytes: u64) {
        let own: u64 = self.layers.iter().map(|l| l.weight_bytes).sum::<u64>() + self.embedding_bytes + self.head_bytes + self.norm_bytes;
        let factor = if own == 0 { 1.0 } else { weight_bytes as f64 / own as f64 };
        let scale = |b: u64| (b as f64 * factor) as u64;
        for l in &mut self.layers {
            l.weight_bytes = scale(l.weight_bytes);
        }
        self.embedding_bytes = scale(self.embedding_bytes);
        self.head_bytes = scale(self.head_bytes);
    }

    pub fn scale_kv_pool(&mut self, pool_bytes: u64) {
        let own: u64 = self.layers.iter().map(|l| l.kv_bytes).sum::<u64>() * self.slots as u64;
        if own == 0 {
            return;
        }
        let factor = pool_bytes as f64 / own as f64;
        for l in &mut self.layers {
            l.kv_bytes = (l.kv_bytes as f64 * factor) as u64;
        }
    }

    pub fn set_kv(&mut self, kv_bytes_per_token: u64, context: usize) {
        let kv_layers = self.layers.iter().filter(|l| l.kind != LayerKind::DeltaNet).count().max(1) as u64;
        for l in &mut self.layers {
            if l.kind != LayerKind::DeltaNet {
                l.kv_bytes = kv_bytes_per_token / kv_layers * context as u64;
            }
        }
        self.context = context;
    }

    pub fn stage_bytes(&self, start: usize, end: usize) -> u64 {
        let slice = &self.layers[start.min(self.layers.len())..end.min(self.layers.len())];
        let mut bytes: u64 = slice.iter().map(|l| l.weight_bytes).sum();
        if start == 0 {
            bytes += self.embedding_bytes;
        }
        if end >= self.layers.len() {
            bytes += self.head_bytes + self.norm_bytes;
        }
        bytes + slice.iter().map(|l| (l.kv_bytes + l.state_bytes) * self.slots as u64).sum::<u64>()
    }

    pub fn allows_tp(&self, tp: usize) -> Result<(), String> {
        if tp == 0 {
            return Err("tp must be at least 1".into());
        }
        if tp > self.max_tp {
            return Err(format!("tp {tp} is above the allowed maximum of {}", self.max_tp));
        }
        for (what, count) in &self.head_counts {
            if count % tp != 0 {
                return Err(format!("tp {tp} does not divide the {count} {what} of {}", self.name));
            }
        }
        Ok(())
    }

    pub fn allreduce_bytes(&self) -> f64 {
        (self.slots * self.hidden * ALLREDUCE_BYTES_PER_ELEMENT) as f64
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        self.layers[layer].kind
    }

    pub fn layer_step_secs(&self, layer: usize, bandwidth: f64, flops: f64) -> f64 {
        let l = &self.layers[layer];
        (l.read_bytes / bandwidth).max(l.flops / flops)
    }

    pub fn hop_bytes(&self) -> f64 {
        (self.slots * self.hidden * self.hop_bytes_per_element) as f64
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Stage {
    pub device: usize,
    pub host: Option<String>,
    pub tp: usize,
    pub ordinals: Vec<usize>,
    pub allreduce_secs: f64,
    pub memory_bound_secs: f64,
    pub flop_bound_secs: f64,
    pub ordinal: usize,
    pub device_name: String,
    pub start: usize,
    pub end: usize,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub state_bytes: u64,
    pub reserve_bytes: u64,
    pub memory_bytes: u64,
    pub offloaded_from: Option<usize>,
    pub offloaded_bytes: u64,
    pub compute_secs: f64,
    pub hop_secs: f64,
}

impl Stage {
    pub fn layers(&self) -> usize {
        self.end - self.start
    }

    pub fn used_bytes(&self) -> u64 {
        self.weight_bytes - self.offloaded_bytes + self.kv_bytes + self.state_bytes + self.reserve_bytes
    }

    pub fn total_secs(&self) -> f64 {
        self.compute_secs + self.hop_secs
    }

    pub fn label(&self) -> String {
        match &self.host {
            Some(host) => {
                let gpus = self.ordinals.iter().map(|o| o.to_string()).collect::<Vec<_>>().join("+");
                format!("{host} / GPU {gpus} (tp {})", self.tp)
            }
            None => self.device_name.clone(),
        }
    }

    pub fn allreduce_share(&self) -> f64 {
        if self.compute_secs > 0.0 { self.allreduce_secs / self.compute_secs } else { 0.0 }
    }

    pub fn bottleneck(&self) -> &'static str {
        let parts = [
            (self.memory_bound_secs, "memory bandwidth"),
            (self.flop_bound_secs, "compute"),
            (self.allreduce_secs, "allreduce"),
            (self.hop_secs, "hop"),
        ];
        parts.iter().fold((-1.0, ""), |best, &(t, n)| if t > best.0 { (t, n) } else { best }).1
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Placement {
    pub model: String,
    pub objective: Objective,
    pub context: usize,
    pub slots: usize,
    pub layers: usize,
    pub stages: Vec<Stage>,
    pub step_secs: f64,
    pub bottleneck_secs: f64,
}

impl Placement {
    pub fn single_stream_tokens_per_second(&self) -> f64 {
        self.slots as f64 / self.step_secs
    }

    pub fn pipelined_tokens_per_second(&self) -> f64 {
        self.slots as f64 / self.bottleneck_secs
    }

    pub fn ensure_runnable(&self) -> Result<()> {
        if let Some((i, s)) = self.stages.iter().enumerate().find(|(_, s)| s.tp > 1) {
            bail!("stage {i} ({}) is a tensor parallel group with tp {}, which the engine cannot run yet; leave `host` unset on those devices", s.label(), s.tp);
        }
        Ok(())
    }

    pub fn ranges(&self) -> Vec<(usize, usize, usize)> {
        self.stages.iter().map(|s| (s.ordinal, s.start, s.end)).collect()
    }

    pub fn stage_of(&self, layer: usize) -> Option<usize> {
        self.stages.iter().position(|s| (s.start..s.end).contains(&layer))
    }

    pub fn render(&self) -> String {
        let gib = |b: u64| b as f64 / GIB;
        let mut out = format!(
            "{}: {} layers in {} stage{}, context {} per slot, {} slot{}, objective {:?}\n",
            self.model,
            self.layers,
            self.stages.len(),
            if self.stages.len() == 1 { "" } else { "s" },
            self.context,
            self.slots,
            if self.slots == 1 { "" } else { "s" },
            self.objective
        );
        for (i, s) in self.stages.iter().enumerate() {
            out.push_str(&format!(
                "  stage {i}  {:<28} layers {:>3}..{:<3} ({:>2})  weights {:>7.2} GiB  kv {:>6.2}  state {:>6.2}  of {:>6.1} GiB  step {:>7.3} ms  allreduce {:>6.3} ms ({:>4.1}%)  hop {:>6.3} ms  bound by {}",
                s.label(),
                s.start,
                s.end,
                s.layers(),
                gib(s.weight_bytes),
                gib(s.kv_bytes),
                gib(s.state_bytes),
                gib(s.memory_bytes),
                s.compute_secs * 1e3,
                s.allreduce_secs * 1e3,
                s.allreduce_share() * 100.0,
                s.hop_secs * 1e3,
                s.bottleneck()
            ));
            if let Some(from) = s.offloaded_from {
                out.push_str(&format!("  offload layers {from}..{} ({:.2} GiB to host)", s.end, gib(s.offloaded_bytes)));
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "  decode step {:.3} ms (one stream {:.1} tok/s), slowest stage {:.3} ms (pipeline full {:.1} tok/s)\n",
            self.step_secs * 1e3,
            self.single_stream_tokens_per_second(),
            self.bottleneck_secs * 1e3,
            self.pipelined_tokens_per_second()
        ));
        out
    }
}

struct StageFit {
    stage: Stage,
}

fn hop_between(devices: &[Device], from: usize, to: usize, bytes: f64) -> f64 {
    let hops = &devices[from + 1..=to];
    let latency: f64 = hops.iter().map(|d| d.link_latency).sum();
    let bandwidth = hops.iter().map(|d| d.link_bandwidth).fold(f64::INFINITY, f64::min);
    latency + bytes / bandwidth
}

fn fit_stage(model: &CostModel, device_index: usize, device: &Device, start: usize, end: usize) -> Result<StageFit, String> {
    let n = model.layers.len();
    let tp = device.tp.max(1);
    let share = |bytes: u64| bytes.div_ceil(tp as u64);
    if device.wgpu
        && let Some(why) = &model.wgpu_blocker
    {
        return Err(format!("{} runs on the wgpu backend and can't take layers of {}: {why} (#491)", device.name, model.name));
    }
    let slice = &model.layers[start..end];
    let mut weights: u64 = slice.iter().map(|l| l.weight_bytes).sum();
    if start == 0 {
        weights += model.embedding_bytes;
    }
    if end == n {
        weights += model.head_bytes + model.norm_bytes;
    }
    let weights = share(weights);
    let kv: u64 = share(slice.iter().map(|l| l.kv_bytes).sum::<u64>() * model.slots as u64);
    let state: u64 = share(slice.iter().map(|l| l.state_bytes).sum::<u64>() * model.slots as u64);
    let available = device.memory_bytes.saturating_sub(device.reserve_bytes);
    let needed = weights + kv + state;
    let (mut offloaded_from, mut offloaded_bytes) = (None, 0u64);
    if needed > available {
        let excess = needed - available;
        let mut moved = 0u64;
        let mut from = end;
        while moved < excess && from > start {
            from -= 1;
            moved += share(model.layers[from].weight_bytes);
        }
        if moved < excess || moved > device.offload_bytes {
            return Err(format!(
                "stage {start}..{end} on {} needs {:.2} GiB ({:.2} GiB weights, {:.2} GiB kv, {:.2} GiB state){} and {} has {:.2} GiB after its reserve{}",
                device.name,
                needed as f64 / GIB,
                weights as f64 / GIB,
                kv as f64 / GIB,
                state as f64 / GIB,
                if tp > 1 { " per rank" } else { "" },
                device.name,
                available as f64 / GIB,
                if device.offload_bytes == 0 { String::new() } else { format!(" and {:.2} GiB of host offload", device.offload_bytes as f64 / GIB) }
            ));
        }
        offloaded_from = Some(from);
        offloaded_bytes = moved;
    }
    let ranks = tp as f64;
    let (mut memory_bound, mut flop_bound) = (0.0, 0.0);
    let mut add = |reads: f64, flops: f64, bandwidth: f64| {
        let (r, f) = (reads / ranks / bandwidth, flops / ranks / device.flops);
        if r >= f { memory_bound += r } else { flop_bound += f }
    };
    for (i, l) in slice.iter().enumerate() {
        let layer = start + i;
        let hosted = offloaded_from.is_some_and(|f| layer >= f);
        add(l.read_bytes, l.flops, if hosted { device.offload_bandwidth } else { device.bandwidth });
    }
    if end == n {
        add(model.head_bytes as f64, model.head_flops, device.bandwidth);
    }
    let allreduce = if tp > 1 {
        slice.len() as f64 * ALLREDUCES_PER_LAYER * (device.allreduce_latency + model.allreduce_bytes() / device.allreduce_bandwidth)
    } else {
        0.0
    };
    let ordinal = device.ordinal.unwrap_or(device_index);
    Ok(StageFit {
        stage: Stage {
            device: device_index,
            host: device.host.clone(),
            tp,
            ordinals: if device.ordinals.is_empty() { vec![ordinal] } else { device.ordinals.clone() },
            allreduce_secs: allreduce,
            memory_bound_secs: memory_bound,
            flop_bound_secs: flop_bound,
            ordinal,
            device_name: device.name.clone(),
            start,
            end,
            weight_bytes: weights,
            kv_bytes: kv,
            state_bytes: state,
            reserve_bytes: device.reserve_bytes,
            memory_bytes: device.memory_bytes,
            offloaded_from,
            offloaded_bytes,
            compute_secs: memory_bound + flop_bound + allreduce,
            hop_secs: 0.0,
        },
    })
}

fn assemble(model: &CostModel, devices: &[Device], objective: Objective, mut stages: Vec<Stage>, spans: &[(usize, usize)]) -> Placement {
    let mut previous: Option<usize> = None;
    for (s, &(first, last)) in stages.iter_mut().zip(spans) {
        s.hop_secs = previous.map_or(0.0, |p| hop_between(devices, p, first, model.hop_bytes()));
        previous = Some(last);
    }
    Placement {
        model: model.name.clone(),
        objective,
        context: model.context,
        slots: model.slots,
        layers: model.layer_count(),
        step_secs: stages.iter().map(Stage::total_secs).sum(),
        bottleneck_secs: stages.iter().map(Stage::total_secs).fold(0.0, f64::max),
        stages,
    }
}

pub fn evaluate(model: &CostModel, devices: &[Device], ranges: &[(usize, usize, usize)], objective: Objective) -> Result<Placement> {
    let groups: Vec<(usize, usize, usize, usize)> = ranges.iter().map(|&(d, s, e)| (d, d, s, e)).collect();
    evaluate_groups(model, devices, &groups, objective)
}

pub fn evaluate_groups(model: &CostModel, devices: &[Device], groups: &[(usize, usize, usize, usize)], objective: Objective) -> Result<Placement> {
    ensure!(!groups.is_empty(), "no stages");
    let mut next = 0;
    let mut last_device: Option<usize> = None;
    let mut stages = Vec::new();
    let mut spans = Vec::new();
    for &(first, last, start, end) in groups {
        ensure!(first <= last && last < devices.len(), "stage uses devices {first}..={last} of {}", devices.len());
        ensure!(last_device.is_none_or(|d| first > d), "stages must follow the device order, each device at most once");
        ensure!(start == next && end > start, "stages must be non-empty and contiguous: expected a stage starting at layer {next}, got {start}..{end}");
        let device = group_device(model, devices, first, last).map_err(anyhow::Error::msg)?;
        match fit_stage(model, first, &device, start, end) {
            Ok(fit) => stages.push(fit.stage),
            Err(why) => bail!(why),
        }
        spans.push((first, last));
        next = end;
        last_device = Some(last);
    }
    ensure!(next == model.layer_count(), "stages end at layer {next} but the model has {}", model.layer_count());
    Ok(assemble(model, devices, objective, stages, &spans))
}

fn group_device(model: &CostModel, devices: &[Device], first: usize, last: usize) -> Result<Device, String> {
    let tp = last - first + 1;
    if tp > 1 {
        let host = devices[first].host.as_ref().ok_or_else(|| format!("{} has no host, so it cannot join a tensor parallel group", devices[first].name))?;
        if devices[first..=last].iter().any(|d| d.host.as_ref() != Some(host)) {
            return Err(format!("a tensor parallel group must stay on one host, {host} is not the host of every member"));
        }
        model.allows_tp(tp)?;
    }
    let members: Vec<(usize, &Device)> = (first..=last).map(|i| (i, &devices[i])).collect();
    Ok(Device::group(&members))
}

fn tp_groups(model: &CostModel, devices: &[Device]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = (0..devices.len()).map(|i| (i, i)).collect();
    let mut run_start = 0;
    while run_start < devices.len() {
        let Some(host) = &devices[run_start].host else {
            run_start += 1;
            continue;
        };
        let mut run_end = run_start;
        while run_end + 1 < devices.len() && devices[run_end + 1].host.as_ref() == Some(host) {
            run_end += 1;
        }
        let count = run_end - run_start + 1;
        for tp in 2..=count {
            if count % tp != 0 || model.allows_tp(tp).is_err() {
                continue;
            }
            let mut f = run_start;
            while f + tp <= run_end + 1 {
                groups.push((f, f + tp - 1));
                f += tp;
            }
        }
        run_start = run_end + 1;
    }
    groups.sort_by_key(|&(f, l)| (l, f));
    groups
}

#[derive(Clone, Copy, Debug)]
struct Score {
    primary: f64,
    secondary: f64,
}

impl Score {
    fn better_than(&self, other: &Score) -> bool {
        const EPS: f64 = 1e-12;
        if (self.primary - other.primary).abs() > EPS {
            self.primary < other.primary
        } else {
            self.secondary < other.secondary - EPS
        }
    }
}

#[derive(Clone, Copy)]
struct Link {
    score: Score,
    previous: usize,
    start: usize,
    group: usize,
}

pub fn plan(model: &CostModel, devices: &[Device], objective: Objective) -> Result<Placement> {
    ensure!(!devices.is_empty(), "no devices to place the model on");
    let n = model.layer_count();
    let d = devices.len();
    let groups = tp_groups(model, devices);
    let mut table: Vec<Vec<Option<Link>>> = vec![vec![None; n + 1]; d];
    let mut reasons: Vec<String> = Vec::new();
    let hop_bytes = model.hop_bytes();
    for (gi, &(first, last)) in groups.iter().enumerate() {
        let device = group_device(model, devices, first, last).expect("candidate groups are valid");
        for end in 1..=n {
            let mut best: Option<Link> = table[last][end];
            for start in 0..end {
                let fit = match fit_stage(model, first, &device, start, end) {
                    Ok(f) => f,
                    Err(why) => {
                        if reasons.len() < 64 && start == 0 {
                            reasons.push(why);
                        }
                        continue;
                    }
                };
                let compute = fit.stage.compute_secs;
                let candidate = if start == 0 {
                    Some(Link { score: Score { primary: compute, secondary: compute }, previous: usize::MAX, start: 0, group: gi })
                } else {
                    let mut chosen: Option<Link> = None;
                    for prev in 0..first {
                        let Some(p) = table[prev][start] else { continue };
                        let cost = compute + hop_between(devices, prev, first, hop_bytes);
                        let total = match objective {
                            Objective::Latency => Score { primary: p.score.primary + cost, secondary: p.score.secondary.max(cost) },
                            Objective::Throughput => Score { primary: p.score.primary.max(cost), secondary: p.score.secondary + cost },
                        };
                        if chosen.as_ref().is_none_or(|c| total.better_than(&c.score)) {
                            chosen = Some(Link { score: total, previous: prev, start, group: gi });
                        }
                    }
                    chosen
                };
                if let Some(c) = candidate
                    && best.as_ref().is_none_or(|b| c.score.better_than(&b.score)) {
                        best = Some(c);
                    }
            }
            table[last][end] = best;
        }
    }
    let mut finish: Option<(Score, usize)> = None;
    for dev in 0..d {
        if let Some(l) = table[dev][n]
            && finish.as_ref().is_none_or(|f| l.score.better_than(&f.0)) {
                finish = Some((l.score, dev));
            }
    }
    let Some((_, mut dev)) = finish else {
        let total: u64 = model.layers.iter().map(|l| l.weight_bytes).sum::<u64>() + model.embedding_bytes + model.head_bytes + model.norm_bytes;
        let kv: u64 = model.layers.iter().map(|l| l.kv_bytes).sum::<u64>() * model.slots as u64;
        let state: u64 = model.layers.iter().map(|l| l.state_bytes).sum::<u64>() * model.slots as u64;
        let capacity: u64 = devices.iter().map(|x| x.memory_bytes.saturating_sub(x.reserve_bytes) + x.offload_bytes).sum();
        bail!(
            "{} does not fit: it needs {:.2} GiB ({:.2} GiB weights, {:.2} GiB kv for {} tokens x {} slots, {:.2} GiB recurrent state) and the {} device{} offer {:.2} GiB after their reserves{}",
            model.name,
            (total + kv + state) as f64 / GIB,
            total as f64 / GIB,
            kv as f64 / GIB,
            model.context,
            model.slots,
            state as f64 / GIB,
            devices.len(),
            if devices.len() == 1 { "" } else { "s" },
            capacity as f64 / GIB,
            reasons.first().map(|r| format!(" ({r})")).unwrap_or_default()
        );
    };
    let mut chosen = Vec::new();
    let mut end = n;
    loop {
        let l = table[dev][end].expect("the chosen chain is filled");
        let (first, last) = groups[l.group];
        chosen.push((first, last, l.start, end));
        if l.previous == usize::MAX {
            break;
        }
        dev = l.previous;
        end = l.start;
    }
    chosen.reverse();
    evaluate_groups(model, devices, &chosen, objective)
}

#[cfg(test)]
mod tests {
    #[test]
    fn max_tp_one_keeps_a_same_host_pair_in_separate_stages() {
        let spec = fixture("qwen3.8-27b-exact.toml");
        let mut config = PlacementConfig { context: 2048, slots: 1, ..PlacementConfig::default() };
        for i in 0..2 {
            config.devices.push(DeviceConfig { name: format!("box/gpu{i}"), memory_gib: 20.0, bandwidth_gbs: 1792.0, tflops: 500.0, ordinal: Some(i), address: None, reserve_gib: Some(0.0), link: LinkConfig::default(), offload: None, host: Some("box".into()), peer: Some(LinkConfig::default()), wgpu: false });
        }
        let open = config.place(&spec).unwrap();
        config.max_tp = Some(1);
        let capped = config.place(&spec).unwrap();
        assert!(capped.stages.iter().all(|s| s.tp == 1));
        assert!(capped.ensure_runnable().is_ok());
        assert!(open.stages.iter().any(|s| s.tp == 2));
    }

    #[test]
    fn micro_batch_defaults_to_overlap_for_remote_stages_and_off_in_process() {
        let mut p = PlacementConfig::default();
        assert_eq!((p.micro_batch_for_remote(), p.micro_batch_in_process()), (REMOTE_MICRO_BATCH, 0));
        p.micro_batch = Some(0);
        assert_eq!((p.micro_batch_for_remote(), p.micro_batch_in_process()), (0, 0));
        p.micro_batch = Some(64);
        assert_eq!((p.micro_batch_for_remote(), p.micro_batch_in_process()), (64, 64));
        let parsed: PlacementConfig = toml::from_str("micro_batch = 128").unwrap();
        assert_eq!(parsed.micro_batch, Some(128));
        let unset: PlacementConfig = toml::from_str("slots = 2").unwrap();
        assert_eq!(unset.micro_batch, None);
    }

    use super::*;
    use std::path::Path;

    fn fixture(name: &str) -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
    }

    fn device(name: &str, gib: f64, gbs: f64) -> Device {
        Device::from_config(&DeviceConfig {
            name: name.into(),
            memory_gib: gib,
            bandwidth_gbs: gbs,
            tflops: 1000.0,
            ordinal: None,
            address: None,
            reserve_gib: Some(0.0),
            link: LinkConfig { latency_ms: 0.01, bandwidth_gbs: 25.0 },
            offload: None,
            host: None,
            peer: None,
            wgpu: false,
        })
    }

    fn qwen27() -> Spec {
        fixture("qwen3.8-27b-exact.toml")
    }

    fn qwen3() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
    }

    #[test]
    fn a_wgpu_device_is_never_given_layers_the_backend_cannot_run() {
        let mut metal = device("mac", 12.0, 200.0);
        metal.wgpu = true;
        let cuda = device("desk", 96.0, 1800.0);
        let hybrid = CostModel::new(&qwen27(), 4096, 1);
        assert!(hybrid.wgpu_blocker.as_deref().is_some_and(|w| w.contains("delta-net")));
        let p = plan(&hybrid, &[metal.clone(), cuda.clone()], Objective::Latency).unwrap();
        assert!(p.stages.iter().all(|s| s.device_name == "desk"), "{}", p.render());
        let only = plan(&hybrid, &[metal.clone()], Objective::Latency).unwrap_err().to_string();
        assert!(only.contains("mac") && only.contains("wgpu") && only.contains("delta-net"), "{only}");
        let dense = CostModel::new(&qwen3(), 2048, 1);
        assert!(dense.wgpu_blocker.is_none());
        let p = plan(&dense, &[metal], Objective::Latency).unwrap();
        assert_eq!(p.stages[0].device_name, "mac");
    }

    fn covers(p: &Placement, layers: usize) {
        let mut next = 0;
        for s in &p.stages {
            assert_eq!(s.start, next);
            assert!(s.end > s.start);
            next = s.end;
        }
        assert_eq!(next, layers);
    }

    #[test]
    fn a_model_that_fits_one_device_stays_on_it_when_minimising_latency() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let p = plan(&m, &[device("a", 96.0, 1800.0), device("b", 96.0, 1800.0)], Objective::Latency).unwrap();
        assert_eq!(p.stages.len(), 1);
        covers(&p, 64);
    }

    #[test]
    fn stages_respect_the_memory_of_each_device() {
        let m = CostModel::new(&qwen27(), 8192, 8);
        let devices = [device("a", 20.0, 1800.0), device("b", 20.0, 1800.0), device("c", 20.0, 1800.0)];
        let p = plan(&m, &devices, Objective::Latency).unwrap();
        covers(&p, 64);
        assert!(p.stages.len() >= 2);
        for s in &p.stages {
            assert!(s.used_bytes() <= s.memory_bytes, "{} uses {} of {}", s.device_name, s.used_bytes(), s.memory_bytes);
        }
        let first = &p.stages[0];
        assert!(first.weight_bytes >= m.embedding_bytes);
        assert!(p.stages.last().unwrap().weight_bytes >= m.head_bytes);
    }

    #[test]
    fn a_faster_device_takes_more_layers() {
        let m = CostModel::new(&qwen27(), 4096, 8);
        let devices = [device("slow", 60.0, 300.0), device("fast", 60.0, 1800.0)];
        let p = plan(&m, &devices, Objective::Throughput).unwrap();
        covers(&p, 64);
        assert_eq!(p.stages.len(), 2);
        assert!(p.stages[1].layers() > p.stages[0].layers(), "{}", p.render());
        let swapped = [device("fast", 60.0, 1800.0), device("slow", 60.0, 300.0)];
        let q = plan(&m, &swapped, Objective::Throughput).unwrap();
        assert!(q.stages[0].layers() > q.stages[1].layers(), "{}", q.render());
    }

    #[test]
    fn attention_layers_cost_more_than_delta_net_layers_at_long_context() {
        let m = CostModel::new(&qwen27(), 131072, 8);
        let attention = (0..64).find(|&i| m.layer_kind(i) == LayerKind::Attention).unwrap();
        let delta = (0..64).find(|&i| m.layer_kind(i) == LayerKind::DeltaNet).unwrap();
        assert!(m.layer_step_secs(attention, 1.8e12, 1e15) > m.layer_step_secs(delta, 1.8e12, 1e15));
    }

    #[test]
    fn the_planner_beats_an_equal_layer_split_on_a_hybrid_model() {
        let m = CostModel::new(&qwen27(), 131072, 8);
        let devices = [device("a", 70.0, 1800.0), device("b", 70.0, 1800.0), device("c", 70.0, 1800.0)];
        let planned = plan(&m, &devices, Objective::Throughput).unwrap();
        let equal = evaluate(&m, &devices, &[(0, 0, 21), (1, 21, 42), (2, 42, 64)], Objective::Throughput).unwrap();
        assert!(planned.bottleneck_secs <= equal.bottleneck_secs);
        let counts: Vec<usize> = planned.stages.iter().map(Stage::layers).collect();
        let attention_in = |s: &Stage| (s.start..s.end).filter(|&i| m.layer_kind(i) == LayerKind::Attention).count();
        let loads: Vec<usize> = planned.stages.iter().map(attention_in).collect();
        assert!(counts.iter().sum::<usize>() == 64 && loads.iter().sum::<usize>() == 16);
    }

    #[test]
    fn an_infeasible_plan_names_what_is_missing() {
        let m = CostModel::new(&qwen27(), 8192, 8);
        let err = plan(&m, &[device("a", 8.0, 1800.0), device("b", 8.0, 1800.0)], Objective::Latency).unwrap_err().to_string();
        assert!(err.contains("does not fit"), "{err}");
        assert!(err.contains("GiB"), "{err}");
        assert!(err.contains("2 devices"), "{err}");
    }

    #[test]
    fn a_slow_link_pulls_the_split_towards_fewer_stages() {
        let m = CostModel::new(&qwen3(), 2048, 1);
        let mut far = device("far", 8.0, 1800.0);
        far.link_latency = 0.05;
        let p = plan(&m, &[device("near", 8.0, 1800.0), far], Objective::Latency).unwrap();
        assert_eq!(p.stages.len(), 1);
    }

    #[test]
    fn overflowing_layers_are_marked_for_host_offload() {
        let m = CostModel::new(&qwen27(), 4096, 1);
        let mut d = device("small", 20.0, 1800.0);
        d.offload_bytes = (64.0 * GIB) as u64;
        d.offload_bandwidth = 50e9;
        let p = plan(&m, &[d], Objective::Latency).unwrap();
        let s = &p.stages[0];
        assert!(s.offloaded_from.is_some() && s.offloaded_bytes > 0);
        assert!(s.used_bytes() <= s.memory_bytes);
        assert!(s.compute_secs > 0.01);
    }

    #[test]
    fn a_pinned_stage_list_is_checked_by_the_same_model() {
        let spec = qwen3();
        let toml_text = r#"
            context = 1024
            [[devices]]
            name = "a"
            memory_gib = 8
            bandwidth_gbs = 1000
            tflops = 100
            [[devices]]
            name = "b"
            memory_gib = 8
            bandwidth_gbs = 1000
            tflops = 100
            [[stages]]
            device = "a"
            layers = [0, 10]
            [[stages]]
            device = "b"
            layers = [10, 28]
        "#;
        let cfg: PlacementConfig = toml::from_str(toml_text).unwrap();
        let p = cfg.place(&spec).unwrap();
        assert_eq!(p.ranges(), vec![(0, 0, 10), (1, 10, 28)]);
        assert_eq!(p.stage_of(9), Some(0));
        assert_eq!(p.stage_of(10), Some(1));
        let mut gap = cfg.clone();
        gap.stages[1].layers = [11, 28];
        assert!(gap.place(&spec).unwrap_err().to_string().contains("contiguous"));
        let mut unknown = cfg.clone();
        unknown.stages[0].device = "zzz".into();
        assert!(unknown.place(&spec).unwrap_err().to_string().contains("zzz"));
    }

    #[test]
    fn the_hand_off_is_the_hidden_state_of_every_slot() {
        let m = CostModel::new(&qwen27(), 4096, 8);
        assert_eq!(m.hop_bytes(), (8 * 5120 * 4) as f64);
    }

    #[test]
    fn two_planner_devices_can_share_a_cuda_ordinal() {
        let spec = Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap();
        let toml_text = r#"
            context = 512
            [[devices]]
            name = "a"
            ordinal = 0
            memory_gib = 8
            bandwidth_gbs = 1000
            tflops = 100
            [[devices]]
            name = "b"
            ordinal = 0
            memory_gib = 8
            bandwidth_gbs = 1000
            tflops = 100
            [[stages]]
            device = "a"
            layers = [0, 10]
            [[stages]]
            device = "b"
            layers = [10, 28]
        "#;
        let p = toml::from_str::<PlacementConfig>(toml_text).unwrap().place(&spec).unwrap();
        assert_eq!(p.ranges(), vec![(0, 0, 10), (0, 10, 28)]);
        assert_eq!(p.stages[1].device, 1);
    }

    fn gpu(name: &str, host: &str, gib: f64) -> Device {
        let mut d = device(name, gib, 1800.0);
        d.host = Some(host.into());
        d.peer_latency = 0.000_01;
        d.peer_bandwidth = 50e9;
        d
    }

    fn pcie_only(devices: &mut [Device]) {
        for d in devices {
            d.peer_latency = 0.004;
            d.peer_bandwidth = 6e9;
        }
    }

    #[test]
    fn a_two_gpu_host_picks_tp2_when_it_fits_one_gpu_but_tp_is_faster() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let p = plan(&m, &devices, Objective::Latency).unwrap();
        assert_eq!(p.stages.len(), 1, "{}", p.render());
        assert_eq!(p.stages[0].tp, 2, "{}", p.render());
        assert_eq!(p.stages[0].ordinals, vec![0, 1]);
        let single = evaluate(&m, &devices, &[(0, 0, 64)], Objective::Latency).unwrap();
        assert!(p.step_secs < single.step_secs);
        assert!(p.stages[0].allreduce_secs > 0.0);
    }

    #[test]
    fn tp_halves_the_memory_each_rank_holds() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let tp2 = evaluate_groups(&m, &devices, &[(0, 1, 0, 64)], Objective::Latency).unwrap();
        let tp1 = evaluate(&m, &devices, &[(0, 0, 64)], Objective::Latency).unwrap();
        let (a, b) = (tp2.stages[0].weight_bytes, tp1.stages[0].weight_bytes);
        assert!(a.abs_diff(b / 2) <= 1);
    }

    #[test]
    fn a_two_gpu_host_beside_a_spark_keeps_the_model_local_until_memory_forces_a_pipeline() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let mut spark = device("spark", 100.0, 270.0);
        spark.host = Some("spark".into());
        spark.link_latency = 0.0005;
        spark.link_bandwidth = 1.2e9;
        let roomy = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0), spark.clone()];
        let p = plan(&m, &roomy, Objective::Latency).unwrap();
        assert_eq!(p.stages.len(), 1, "{}", p.render());
        assert_eq!(p.stages[0].tp, 2);
        let tight = [gpu("g0", "box", 5.0), gpu("g1", "box", 5.0), spark];
        let q = plan(&m, &tight, Objective::Latency).unwrap();
        assert_eq!(q.stages.len(), 2, "{}", q.render());
        assert_eq!(q.stages[0].tp, 2, "{}", q.render());
        assert_eq!(q.stages[1].device_name, "spark");
        assert!(q.stages.iter().all(|s| s.used_bytes() <= s.memory_bytes));
    }

    #[test]
    fn heads_that_tp_does_not_divide_are_rejected() {
        let mut spec = qwen27();
        spec.attention.kv_heads = 3;
        let m = CostModel::new(&spec, 4096, 1);
        assert!(m.allows_tp(2).unwrap_err().contains("kv heads"));
        let devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let err = evaluate_groups(&m, &devices, &[(0, 1, 0, 64)], Objective::Latency).unwrap_err().to_string();
        assert!(err.contains("does not divide"), "{err}");
        let p = plan(&m, &devices, Objective::Latency).unwrap();
        assert!(p.stages.iter().all(|s| s.tp == 1));
        let mut delta = qwen27();
        delta.linear_attention.as_mut().unwrap().value_heads = 3;
        assert!(CostModel::new(&delta, 4096, 1).allows_tp(2).unwrap_err().contains("delta-net"));
    }

    #[test]
    fn a_plan_with_a_tensor_parallel_stage_is_refused_by_name() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let p = plan(&m, &devices, Objective::Latency).unwrap();
        let err = p.ensure_runnable().unwrap_err().to_string();
        assert!(err.contains("stage 0") && err.contains("tp 2") && err.contains("box / GPU 0+1"), "{err}");
        let single = evaluate(&m, &devices, &[(0, 0, 64)], Objective::Latency).unwrap();
        assert!(single.ensure_runnable().is_ok());
    }

    #[test]
    fn a_group_must_stay_on_one_host() {
        let m = CostModel::new(&qwen27(), 4096, 1);
        let devices = [gpu("g0", "a", 96.0), gpu("g1", "b", 96.0)];
        let err = evaluate_groups(&m, &devices, &[(0, 1, 0, 64)], Objective::Latency).unwrap_err().to_string();
        assert!(err.contains("one host"), "{err}");
        let p = plan(&m, &devices, Objective::Throughput).unwrap();
        assert!(p.stages.iter().all(|s| s.tp == 1));
    }

    #[test]
    fn a_probe_without_p2p_raises_the_allreduce_cost_and_can_flip_the_choice() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let mut devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let fast = plan(&m, &devices, Objective::Latency).unwrap();
        assert_eq!(fast.stages[0].tp, 2);
        pcie_only(&mut devices);
        let staged = plan(&m, &devices, Objective::Latency).unwrap();
        assert_eq!(staged.stages[0].tp, 1, "{}", staged.render());
        let forced = evaluate_groups(&m, &devices, &[(0, 1, 0, 64)], Objective::Latency).unwrap();
        assert!(forced.stages[0].allreduce_secs > fast.stages[0].allreduce_secs * 10.0);
    }

    #[test]
    fn probes_override_the_declared_peer_link_unless_one_is_configured() {
        let mut cfg: PlacementConfig = toml::from_str(
            r#"
            [[devices]]
            name = "g0"
            host = "box"
            ordinal = 0
            memory_gib = 96
            bandwidth_gbs = 1800
            tflops = 1000
            [[devices]]
            name = "g1"
            host = "box"
            ordinal = 1
            memory_gib = 96
            bandwidth_gbs = 1800
            tflops = 1000
            [devices.peer]
            latency_ms = 0.5
            bandwidth_gbs = 2
        "#,
        )
        .unwrap();
        cfg.apply_peer_probes(&[crate::link::PairProbe { a: 0, b: 1, p2p: false, latency_ms: 0.05, bandwidth_gbs: 12.0 }]);
        assert_eq!(cfg.devices[0].peer, Some(LinkConfig { latency_ms: 0.05, bandwidth_gbs: 12.0 }));
        assert_eq!(cfg.devices[1].peer, Some(LinkConfig { latency_ms: 0.5, bandwidth_gbs: 2.0 }));
    }

    #[test]
    fn the_plan_output_names_the_group_and_what_bounds_each_stage() {
        let m = CostModel::new(&qwen27(), 8192, 4);
        let devices = [gpu("g0", "box", 96.0), gpu("g1", "box", 96.0)];
        let p = plan(&m, &devices, Objective::Latency).unwrap();
        let text = p.render();
        let golden = include_str!("../tests/golden/plan-tp2.txt");
        assert_eq!(text, golden, "{text}");
    }
}
