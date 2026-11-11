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
#[serde(default, deny_unknown_fields)]
pub struct PlacementConfig {
    pub objective: Objective,
    pub context: usize,
    pub slots: usize,
    pub devices: Vec<DeviceConfig>,
    pub stages: Vec<StageConfig>,
    pub micro_batch: usize,
    pub discovery: Discovery,
    pub mdns_port: u16,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        PlacementConfig { objective: Objective::Latency, context: 8192, slots: 1, devices: Vec::new(), stages: Vec::new(), micro_batch: 0, discovery: Discovery::Static, mdns_port: 5353 }
    }
}

impl PlacementConfig {
    pub fn place(&self, spec: &Spec) -> Result<Placement> {
        ensure!(self.slots >= 1 && self.context >= 1, "placement needs at least one slot and one token of context");
        let model = CostModel::new(spec, self.context, self.slots);
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
}

impl Device {
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

pub struct CostModel {
    pub name: String,
    pub context: usize,
    pub slots: usize,
    layers: Vec<LayerCost>,
    embedding_bytes: u64,
    head_bytes: u64,
    head_flops: f64,
    norm_bytes: u64,
    pub hidden: usize,
    pub hop_bytes_per_element: usize,
}

impl CostModel {
    pub fn new(spec: &Spec, context: usize, slots: usize) -> CostModel {
        let anatomy = ModelAnatomy::new(spec, 0, Some(context));
        let kv_per_token = (2 * spec.attention.kv_heads * spec.attention.head_dim * 2) as u64;
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
                    _ => (kv_per_token * context as u64, 0),
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
            embedding_bytes: 2 * matrix,
            head_bytes: quant_bytes(matrix, anatomy.head_quant),
            head_flops: FLOPS_PER_PARAM * matrix as f64 * rows,
            norm_bytes: 2 * spec.dims.hidden as u64,
            hidden: spec.dims.hidden,
            hop_bytes_per_element: 4,
        }
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

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub device: usize,
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
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn ranges(&self) -> Vec<(usize, usize, usize)> {
        self.stages.iter().map(|s| (s.ordinal, s.start, s.end)).collect()
    }

    pub fn stage_of(&self, layer: usize) -> Option<usize> {
        self.stages.iter().position(|s| (s.start..s.end).contains(&layer))
    }

    pub fn render(&self) -> String {
        let gib = |b: u64| b as f64 / GIB;
        let mut out = format!(
            "{}: {} layers in {} stage{}, context {}, {} slot{}, objective {:?}\n",
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
                "  stage {i}  {:<16} layers {:>3}..{:<3} ({:>2})  weights {:>7.2} GiB  kv {:>6.2}  state {:>6.2}  of {:>6.1} GiB  step {:>7.3} ms  hop {:>6.3} ms",
                s.device_name,
                s.start,
                s.end,
                s.layers(),
                gib(s.weight_bytes),
                gib(s.kv_bytes),
                gib(s.state_bytes),
                gib(s.memory_bytes),
                s.compute_secs * 1e3,
                s.hop_secs * 1e3
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
    let slice = &model.layers[start..end];
    let mut weights: u64 = slice.iter().map(|l| l.weight_bytes).sum();
    if start == 0 {
        weights += model.embedding_bytes;
    }
    if end == n {
        weights += model.head_bytes + model.norm_bytes;
    }
    let kv: u64 = slice.iter().map(|l| l.kv_bytes).sum::<u64>() * model.slots as u64;
    let state: u64 = slice.iter().map(|l| l.state_bytes).sum::<u64>() * model.slots as u64;
    let available = device.memory_bytes.saturating_sub(device.reserve_bytes);
    let needed = weights + kv + state;
    let (mut offloaded_from, mut offloaded_bytes) = (None, 0u64);
    if needed > available {
        let excess = needed - available;
        let mut moved = 0u64;
        let mut from = end;
        while moved < excess && from > start {
            from -= 1;
            moved += model.layers[from].weight_bytes;
        }
        if moved < excess || moved > device.offload_bytes {
            return Err(format!(
                "stage {start}..{end} on {} needs {:.2} GiB ({:.2} GiB weights, {:.2} GiB kv, {:.2} GiB state) and {} has {:.2} GiB after its reserve{}",
                device.name,
                needed as f64 / GIB,
                weights as f64 / GIB,
                kv as f64 / GIB,
                state as f64 / GIB,
                device.name,
                available as f64 / GIB,
                if device.offload_bytes == 0 { String::new() } else { format!(" and {:.2} GiB of host offload", device.offload_bytes as f64 / GIB) }
            ));
        }
        offloaded_from = Some(from);
        offloaded_bytes = moved;
    }
    let mut compute = 0.0;
    for (i, l) in slice.iter().enumerate() {
        let layer = start + i;
        let hosted = offloaded_from.is_some_and(|f| layer >= f);
        let bandwidth = if hosted { device.offload_bandwidth } else { device.bandwidth };
        compute += (l.read_bytes / bandwidth).max(l.flops / device.flops);
    }
    if end == n {
        compute += (model.head_bytes as f64 / device.bandwidth).max(model.head_flops / device.flops);
    }
    Ok(StageFit {
        stage: Stage {
            device: device_index,
            ordinal: device.ordinal.unwrap_or(device_index),
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
            compute_secs: compute,
            hop_secs: 0.0,
        },
    })
}

fn assemble(model: &CostModel, devices: &[Device], objective: Objective, mut stages: Vec<Stage>) -> Placement {
    let mut previous: Option<usize> = None;
    for s in &mut stages {
        s.hop_secs = previous.map_or(0.0, |p| hop_between(devices, p, s.device, model.hop_bytes()));
        previous = Some(s.device);
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
    ensure!(!ranges.is_empty(), "no stages");
    let mut next = 0;
    let mut last_device: Option<usize> = None;
    let mut stages = Vec::new();
    for &(device, start, end) in ranges {
        ensure!(device < devices.len(), "stage uses device {device} of {}", devices.len());
        ensure!(last_device.is_none_or(|d| device > d), "stages must follow the device order, each device at most once");
        ensure!(start == next && end > start, "stages must be non-empty and contiguous: expected a stage starting at layer {next}, got {start}..{end}");
        match fit_stage(model, device, &devices[device], start, end) {
            Ok(fit) => stages.push(fit.stage),
            Err(why) => bail!(why),
        }
        next = end;
        last_device = Some(device);
    }
    ensure!(next == model.layer_count(), "stages end at layer {next} but the model has {}", model.layer_count());
    Ok(assemble(model, devices, objective, stages))
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

pub fn plan(model: &CostModel, devices: &[Device], objective: Objective) -> Result<Placement> {
    ensure!(!devices.is_empty(), "no devices to place the model on");
    let n = model.layer_count();
    let d = devices.len();
    let mut table: Vec<Vec<Option<(Score, usize, usize)>>> = vec![vec![None; n + 1]; d];
    let mut reasons: Vec<String> = Vec::new();
    let hop_bytes = model.hop_bytes();
    for dev in 0..d {
        for end in 1..=n {
            let mut best: Option<(Score, usize, usize)> = None;
            for start in 0..end {
                let fit = match fit_stage(model, dev, &devices[dev], start, end) {
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
                    Some((Score { primary: compute, secondary: compute }, usize::MAX, 0))
                } else {
                    let mut chosen: Option<(Score, usize, usize)> = None;
                    for prev in 0..dev {
                        let Some((p, _, _)) = table[prev][start] else { continue };
                        let cost = compute + hop_between(devices, prev, dev, hop_bytes);
                        let total = match objective {
                            Objective::Latency => Score { primary: p.primary + cost, secondary: p.secondary.max(cost) },
                            Objective::Throughput => Score { primary: p.primary.max(cost), secondary: p.secondary + cost },
                        };
                        if chosen.as_ref().is_none_or(|c| total.better_than(&c.0)) {
                            chosen = Some((total, prev, start));
                        }
                    }
                    chosen
                };
                if let Some(c) = candidate {
                    if best.as_ref().is_none_or(|b| c.0.better_than(&b.0)) {
                        best = Some(c);
                    }
                }
            }
            table[dev][end] = best;
        }
    }
    let mut finish: Option<(Score, usize)> = None;
    for dev in 0..d {
        if let Some((s, _, _)) = table[dev][n] {
            if finish.as_ref().is_none_or(|f| s.better_than(&f.0)) {
                finish = Some((s, dev));
            }
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
    let mut ranges = Vec::new();
    let mut end = n;
    loop {
        let (_, prev, start) = table[dev][end].expect("the chosen chain is filled");
        ranges.push((dev, start, end));
        if prev == usize::MAX {
            break;
        }
        dev = prev;
        end = start;
    }
    ranges.reverse();
    evaluate(model, devices, &ranges, objective)
}

#[cfg(test)]
mod tests {
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
        })
    }

    fn qwen27() -> Spec {
        fixture("qwen3.8-27b.toml")
    }

    fn qwen3() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
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
}
