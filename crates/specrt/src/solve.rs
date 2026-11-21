use crate::anatomy::ModelAnatomy;
use crate::options::Profile;
use crate::placement::{CostModel, Device, DeviceConfig, Objective, Placement, evaluate, evaluate_groups, plan};
use crate::spec::{KvDtype, Spec};
use anyhow::{Result, bail, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const TYPICAL_REQUEST: usize = 4096;
const SLOT_CHOICES: [usize; 6] = [32, 16, 8, 4, 2, 1];
const MATCH_TOLERANCE: f64 = 0.05;
const GIB: f64 = (1u64 << 30) as f64;
const MAX_PERFORMANCE_CONTEXT: usize = 32768;
const LOW_MEMORY_CONTEXT: usize = 8192;

pub const PIN_KV_DTYPE: &str = "runtime.kv_dtype";
pub const PIN_KV_F16_LAYERS: &str = "runtime.kv_f16_layers";
pub const PIN_SLOTS: &str = "serve.max_num_seqs";
pub const PIN_MTP: &str = "speculative.enabled";
pub const PIN_POOL: &str = "serve.kv_cache_bytes";
pub const PIN_TP: &str = "serve.tp";

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Target {
    pub context: Option<usize>,
    pub memory: Option<u64>,
    pub slots: Option<usize>,
}

pub struct PlacementRequest<'a> {
    pub mode: Profile,
    pub target: Target,
    pub pins: BTreeMap<String, Value>,
    pub devices: &'a [DeviceConfig],
    pub spec: &'a Spec,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Solved {
    pub mode: Profile,
    pub placement: Placement,
    pub tp: usize,
    pub kv_dtype: KvDtype,
    pub kv_f16_layers: Option<usize>,
    pub kv_pool_bytes: u64,
    pub slots: usize,
    pub context: usize,
    pub mtp: bool,
    pub options: BTreeMap<String, Value>,
    pub strategy: String,
    pub reasons: Vec<String>,
    pub shortfall: Option<String>,
}

impl Solved {
    pub fn spans(&self) -> bool {
        self.placement.stages.len() > 1
    }

    pub fn single_device(&self) -> usize {
        self.placement.stages.first().map_or(0, |s| s.ordinals.first().copied().unwrap_or(s.ordinal))
    }

    pub fn ordinals(&self) -> Vec<usize> {
        self.placement.stages.iter().flat_map(|s| s.ordinals.clone()).collect()
    }

    pub fn log_line(&self) -> String {
        format!(
            "{}; kv {:?} pool {:.2} GiB, {} slots, context {}, mtp {}",
            self.strategy,
            self.kv_dtype,
            self.kv_pool_bytes as f64 / GIB,
            self.slots,
            self.context,
            if self.mtp { "on" } else { "off" }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Single(usize),
    Tp(usize, usize),
    Pipeline,
}

#[derive(Clone, Copy, Debug)]
struct Opt {
    dtype: KvDtype,
    mtp: bool,
    slots: usize,
}

struct Search<'a> {
    req: &'a PlacementRequest<'a>,
    devices: Vec<Device>,
    native: usize,
    kinds: Vec<Kind>,
}

fn matched(a: &Device, b: &Device) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= MATCH_TOLERANCE * x.max(y);
    close(a.bandwidth, b.bandwidth) && close(a.flops, b.flops)
}

fn tp_kinds(spec: &Spec, devices: &[Device]) -> Vec<Kind> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < devices.len() {
        let Some(host) = devices[start].host.as_ref() else {
            start += 1;
            continue;
        };
        let mut end = start;
        while end + 1 < devices.len() && devices[end + 1].host.as_ref() == Some(host) && matched(&devices[start], &devices[end + 1]) {
            end += 1;
        }
        for tp in (2..=end - start + 1).rev() {
            if crate::tp::supports(spec, tp).is_ok() {
                out.push(Kind::Tp(start, tp));
                break;
            }
        }
        start = end + 1;
    }
    out
}

impl<'a> Search<'a> {
    fn new(req: &'a PlacementRequest<'a>) -> Result<Self> {
        ensure!(!req.devices.is_empty(), "no devices to place the model on");
        let devices: Vec<Device> = req.devices.iter().map(Device::from_config).collect();
        let native = crate::engine::model_max_context(req.spec).unwrap_or(crate::anatomy::DEFAULT_MAX_CONTEXT);
        let mut order: Vec<usize> = (0..devices.len()).collect();
        let mut kinds = Vec::new();
        match req.mode {
            Profile::MaxPerformance => {
                kinds.extend(order.iter().map(|&i| Kind::Single(i)));
                kinds.extend(tp_kinds(req.spec, &devices));
                if devices.len() > 1 {
                    kinds.push(Kind::Pipeline);
                }
            }
            Profile::LowMemory => {
                order.sort_by(|&a, &b| devices[a].memory_bytes.cmp(&devices[b].memory_bytes).then(devices[b].bandwidth.total_cmp(&devices[a].bandwidth)).then(a.cmp(&b)));
                kinds.extend(order.iter().map(|&i| Kind::Single(i)));
                if devices.len() > 1 {
                    kinds.push(Kind::Pipeline);
                }
                kinds.extend(tp_kinds(req.spec, &devices));
            }
        }
        let mut search = Search { req, devices, native, kinds };
        match search.pinned_tp() {
            Some(1) => search.kinds.retain(|k| !matches!(k, Kind::Tp(..))),
            Some(n) if n > 1 => {
                search.kinds.retain(|k| matches!(k, Kind::Tp(_, t) if *t == n));
                if search.kinds.is_empty() {
                    bail!("tp {n} is not possible on these devices for {}", req.spec.name);
                }
            }
            _ => {}
        }
        Ok(search)
    }

    fn pin<T>(&self, key: &str, f: impl Fn(&Value) -> Option<T>) -> Option<T> {
        self.req.pins.get(key).and_then(f)
    }

    fn pinned_dtype(&self) -> Option<KvDtype> {
        self.pin(PIN_KV_DTYPE, |v| match v.as_str() {
            Some("fp8") => Some(KvDtype::Fp8),
            Some("f16") => Some(KvDtype::F16),
            _ => None,
        })
    }

    fn pinned_slots(&self) -> Option<usize> {
        self.req.target.slots.or_else(|| self.pin(PIN_SLOTS, Value::as_u64).map(|n| n as usize))
    }

    fn pinned_mtp(&self) -> Option<bool> {
        self.pin(PIN_MTP, Value::as_bool)
    }

    fn pinned_tp(&self) -> Option<usize> {
        self.pin(PIN_TP, Value::as_u64).map(|n| n as usize)
    }

    fn pinned_pool(&self) -> Option<u64> {
        self.pin(PIN_POOL, Value::as_u64).filter(|b| *b > 0)
    }

    fn pinned_f16_layers(&self) -> Option<usize> {
        self.pin(PIN_KV_F16_LAYERS, Value::as_u64).map(|n| n as usize)
    }

    fn requested_context(&self) -> usize {
        let default = match self.req.mode {
            Profile::MaxPerformance => MAX_PERFORMANCE_CONTEXT,
            Profile::LowMemory => LOW_MEMORY_CONTEXT,
        };
        self.req.target.context.unwrap_or(default).clamp(1, self.native)
    }

    fn options(&self, fallback: bool) -> Vec<Opt> {
        let has_mtp = self.req.spec.speculative().is_some() || self.req.spec.mtp.is_some();
        let dtypes: Vec<KvDtype> = match (self.pinned_dtype(), self.req.mode, fallback) {
            (Some(d), _, _) => vec![d],
            (None, _, true) => vec![KvDtype::Fp8],
            (None, Profile::MaxPerformance, false) => vec![KvDtype::F16, KvDtype::Fp8],
            (None, Profile::LowMemory, false) => vec![KvDtype::Fp8],
        };
        let mtps: Vec<bool> = match (self.pinned_mtp(), self.req.mode, has_mtp, fallback) {
            (_, _, false, _) => vec![false],
            (Some(m), _, _, _) => vec![m],
            (None, Profile::MaxPerformance, true, false) => vec![true, false],
            (None, _, true, _) => vec![false],
        };
        let slots: Vec<usize> = match (self.pinned_slots(), self.req.mode, fallback) {
            (Some(s), _, _) => vec![s.max(1)],
            (None, _, true) => vec![1],
            (None, Profile::MaxPerformance, false) => SLOT_CHOICES.to_vec(),
            (None, Profile::LowMemory, false) => vec![2, 1],
        };
        let mut out = Vec::new();
        for &dtype in &dtypes {
            for &mtp in &mtps {
                for &s in &slots {
                    out.push(Opt { dtype, mtp, slots: s });
                }
            }
        }
        out
    }

    fn kv_spec(&self, dtype: KvDtype, context: usize) -> Spec {
        let mut s = self.req.spec.clone();
        s.runtime.kv_dtype = dtype;
        s.runtime.kv_f16_layers = self.pinned_f16_layers();
        s.for_kv_context(context)
    }

    fn kv_bytes_per_token(&self, dtype: KvDtype, context: usize) -> u64 {
        (crate::runtime::kv_block_bytes(&self.kv_spec(dtype, context), 1) / crate::codegen::PAGE) as u64
    }

    fn weight_bytes(&self, mtp: bool, context: usize) -> u64 {
        let layers = if mtp { self.req.spec.mtp.as_ref().map_or(0, |m| m.layers) } else { 0 };
        ModelAnatomy::new(self.req.spec, layers, Some(context)).total_bytes
    }

    fn model(&self, o: &Opt, tokens: usize, context: usize) -> CostModel {
        let spec = self.kv_spec(o.dtype, context);
        let mut m = CostModel::new(&spec, tokens.div_ceil(o.slots).max(1), o.slots);
        m.rescale_weights(self.weight_bytes(o.mtp, context));
        m
    }

    fn candidate(&self, kind: Kind, model: &CostModel) -> Result<Placement> {
        let n = model.layer_count();
        match kind {
            Kind::Single(i) => evaluate(model, &self.devices, &[(i, 0, n)], Objective::Throughput),
            Kind::Tp(first, tp) => evaluate_groups(model, &self.devices, &[(first, first + tp - 1, 0, n)], Objective::Throughput),
            Kind::Pipeline => {
                let mut m = model.clone();
                m.max_tp = 1;
                plan(&m, &self.devices, Objective::Throughput)
            }
        }
    }

    fn within(&self, p: &Placement, memory: Option<u64>) -> Result<()> {
        if let Some(limit) = memory {
            let used: u64 = p.stages.iter().map(|s| (s.weight_bytes + s.kv_bytes + s.state_bytes) * s.tp as u64).sum();
            ensure!(used <= limit, "needs {:.2} GiB against a budget of {:.2} GiB", used as f64 / GIB, limit as f64 / GIB);
        }
        Ok(())
    }

    fn best(&self, o: &Opt, tokens: usize, context: usize, memory: Option<u64>) -> Result<(Kind, Placement), String> {
        let model = self.model(o, tokens, context);
        let mut best: Option<(Kind, Placement, f64)> = None;
        let mut last = String::from("no placement fits");
        for &kind in &self.kinds {
            match self.candidate(kind, &model).and_then(|p| self.within(&p, memory).map(|_| p)) {
                Ok(p) => {
                    if self.req.mode == Profile::LowMemory {
                        return Ok((kind, p));
                    }
                    let score = p.single_stream_tokens_per_second();
                    if best.as_ref().is_none_or(|b| score > b.2 * 1.0001) {
                        best = Some((kind, p, score));
                    }
                }
                Err(e) => last = format!("{e:#}"),
            }
        }
        best.map(|(k, p, _)| (k, p)).ok_or(last)
    }

    fn same_kind(&self, kind: Kind, o: &Opt, tokens: usize, context: usize, memory: Option<u64>) -> Option<Placement> {
        let model = self.model(o, tokens, context);
        self.candidate(kind, &model).and_then(|p| self.within(&p, memory).map(|_| p)).ok()
    }

    fn attempt(&self, context: usize, memory: Option<u64>, fallback: bool) -> Result<(Solved, Opt), String> {
        let mut last = String::from("no placement fits");
        for o in self.options(fallback) {
            let bpt = self.kv_bytes_per_token(o.dtype, context).max(1);
            let pinned = self.pinned_pool().map(|b| (b / bpt) as usize);
            let need = pinned.unwrap_or(context + (o.slots - 1) * TYPICAL_REQUEST);
            if pinned.is_some_and(|p| p < context) {
                last = format!("the pinned KV pool holds {} tokens, below the context {context}", pinned.unwrap_or(0));
                continue;
            }
            let (kind, mut placement) = match self.best(&o, need, context, memory) {
                Ok(r) => r,
                Err(e) => {
                    last = e;
                    continue;
                }
            };
            let mut tokens = need;
            if self.req.mode == Profile::MaxPerformance && pinned.is_none() {
                let cap = (context * o.slots).max(need);
                let (mut lo, mut hi) = (need, cap);
                while lo < hi {
                    let mid = lo + (hi - lo).div_ceil(2);
                    if self.same_kind(kind, &o, mid, context, memory).is_some() {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                if lo > need
                    && let Some(p) = self.same_kind(kind, &o, lo, context, memory)
                {
                    tokens = lo;
                    placement = p;
                }
            }
            return Ok((self.assemble(&o, kind, placement, tokens, context, bpt, fallback), o));
        }
        Err(last)
    }

    fn assemble(&self, o: &Opt, kind: Kind, placement: Placement, tokens: usize, context: usize, bpt: u64, fallback: bool) -> Solved {
        let spec = self.kv_spec(o.dtype, context);
        let tp = placement.stages.iter().map(|s| s.tp).max().unwrap_or(1);
        let mut reasons = Vec::new();
        let max = self.req.mode == Profile::MaxPerformance;
        if o.dtype.is_fp8() && self.pinned_dtype().is_none() {
            reasons.push(if max || fallback { format!("fp8 KV: f16 KV does not fit context {context} here") } else { "low memory uses fp8 KV".to_string() });
        }
        if !max && self.req.spec.mtp.is_some() && self.pinned_mtp().is_none() {
            reasons.push("MTP off to save memory".to_string());
        }
        if max && self.req.spec.mtp.is_some() && !o.mtp && self.pinned_mtp().is_none() {
            reasons.push("MTP off: the draft head does not fit".to_string());
        }
        if self.pinned_slots().is_none() {
            reasons.push(format!("{} slot(s): the most the {} token pool serves with requests of {TYPICAL_REQUEST} tokens", o.slots, tokens));
        }
        if context < self.req.target.context.unwrap_or(context) {
            reasons.push(format!("context limited to the model's {context}"));
        }
        let strategy = self.strategy_line(kind, &placement, o, tokens, context);
        let mut options = self.req.mode.values();
        options.insert(PIN_KV_DTYPE.into(), json!(if o.dtype.is_fp8() { "fp8" } else { "f16" }));
        if o.dtype.is_fp8() {
            options.insert(PIN_KV_F16_LAYERS.into(), json!(spec.kv_f16_layers()));
        }
        options.insert(PIN_SLOTS.into(), json!(o.slots));
        if self.req.spec.mtp.is_some() {
            options.insert(PIN_MTP.into(), json!(o.mtp));
        }
        options.insert("serve.tp".into(), json!(tp));
        for (k, v) in &self.req.pins {
            if k != PIN_POOL && k != PIN_TP {
                options.insert(k.clone(), v.clone());
            }
        }
        Solved {
            mode: self.req.mode,
            tp,
            kv_dtype: o.dtype,
            kv_f16_layers: o.dtype.is_fp8().then(|| spec.kv_f16_layers()),
            kv_pool_bytes: self.pinned_pool().unwrap_or(tokens as u64 * bpt),
            slots: o.slots,
            context,
            mtp: o.mtp,
            options,
            strategy,
            reasons,
            shortfall: None,
            placement,
        }
    }

    fn strategy_line(&self, kind: Kind, p: &Placement, o: &Opt, tokens: usize, context: usize) -> String {
        match kind {
            Kind::Single(i) => {
                let d = &self.devices[i];
                let why = if self.req.mode == Profile::LowMemory { "smallest that fits" } else { "fastest placement for this load" };
                format!("fits on {} ({:.0} GB, {why})", crate::hardware::display_name(&d.name), d.memory_bytes as f64 / 1e9)
            }
            Kind::Tp(first, tp) => {
                let names: Vec<String> = self.devices[first..first + tp].iter().map(|d| crate::hardware::display_name(&d.name)).collect();
                let one = self.same_kind(Kind::Single(first), o, tokens, context, None).map(|s| s.single_stream_tokens_per_second());
                let tail = match one {
                    Some(t) if t > 0.0 => format!("{:.1}x the throughput of one GPU", p.single_stream_tokens_per_second() / t),
                    _ => "the model does not fit on one GPU".to_string(),
                };
                format!("TP {tp} across {}: {tail}", names.join(" + "))
            }
            Kind::Pipeline => {
                let names: Vec<String> = p.stages.iter().map(|s| format!("{} layers {}..{}", crate::hardware::display_name(&s.device_name), s.start, s.end)).collect();
                format!("pipeline over {} GPUs split by memory and speed ({})", p.stages.len(), names.join(", "))
            }
        }
    }

    fn largest_context(&self, memory: Option<u64>) -> Option<(usize, Solved)> {
        let (mut lo, mut hi) = (1usize, self.native);
        let mut best: Option<(usize, Solved)> = None;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            match self.attempt(mid, memory, true) {
                Ok((s, _)) => {
                    best = Some((mid, s));
                    lo = mid + 1;
                }
                Err(_) => {
                    if mid == 1 {
                        break;
                    }
                    hi = mid - 1;
                }
            }
        }
        best
    }
}

pub fn for_options(spec: &Spec, opts: &crate::serve::ServeOptions) -> Result<Option<Solved>> {
    if let Some(s) = &opts.solved {
        return Ok(Some((**s).clone()));
    }
    let Some(config) = opts.placement.as_ref().filter(|p| !p.devices.is_empty()) else {
        if opts.tp <= 1 {
            return Ok(None);
        }
        let machine = crate::hardware::probe()?;
        return for_devices(spec, opts, opts.tp, &machine.devices(&[])).map(Some);
    };
    if config.stages.is_empty() {
        let tp = if config.max_tp == Some(1) { 1 } else { opts.tp };
        return for_devices(spec, opts, tp, &config.devices).map(Some);
    }
    let mut resolved = spec.clone();
    if let Some(r) = &opts.runtime {
        resolved.runtime = r.clone();
    }
    opts.speculative.apply(&mut resolved)?;
    let mut sized = config.clone();
    sized.kv_pool_bytes = sized.kv_pool_bytes.or(opts.kv_cache_bytes);
    let placement = sized.place(&resolved)?;
    placement.ensure_runnable()?;
    let context = opts.model_len(spec)?;
    Ok(Some(Solved {
        mode: Profile::MaxPerformance,
        tp: 1,
        kv_dtype: resolved.runtime.kv_dtype,
        kv_f16_layers: resolved.runtime.kv_f16_layers,
        kv_pool_bytes: opts.kv_cache_bytes.unwrap_or(0),
        slots: opts.max_num_seqs,
        context,
        mtp: resolved.speculative().is_some(),
        options: BTreeMap::new(),
        strategy: format!("pinned stages: {} stage(s)", placement.stages.len()),
        reasons: vec!["stages pinned by [[placement.stages]]".into()],
        shortfall: None,
        placement,
    }))
}

fn for_devices(spec: &Spec, opts: &crate::serve::ServeOptions, tp: usize, devices: &[DeviceConfig]) -> Result<Solved> {
    let mut pins = BTreeMap::new();
    pins.insert(PIN_SLOTS.to_string(), json!(opts.max_num_seqs));
    let capped = opts.placement.as_ref().is_some_and(|p| p.max_tp == Some(1));
    if tp > 1 || capped {
        pins.insert(PIN_TP.to_string(), json!(tp.max(1)));
    }
    if let Some(b) = opts.kv_cache_bytes {
        pins.insert(PIN_POOL.to_string(), json!(b));
    }
    if let Some(r) = &opts.runtime {
        pins.insert(PIN_KV_DTYPE.to_string(), json!(if r.kv_dtype.is_fp8() { "fp8" } else { "f16" }));
    }
    if let Some(on) = opts.speculative.enabled {
        pins.insert(PIN_MTP.to_string(), json!(on));
    }
    let target = Target { context: opts.max_model_len, memory: None, slots: Some(opts.max_num_seqs) };
    solve(&PlacementRequest { mode: Profile::MaxPerformance, target, pins, devices, spec })
}

pub fn solve(req: &PlacementRequest) -> Result<Solved> {
    let search = Search::new(req)?;
    let context = search.requested_context();
    let memory = req.target.memory;
    let first = search.attempt(context, memory, false);
    let error = match first {
        Ok((s, _)) => return Ok(s),
        Err(e) => e,
    };
    if memory.is_some()
        && let Ok((mut s, _)) = search.attempt(context, None, false)
    {
        let used: u64 = s.placement.stages.iter().map(|x| (x.weight_bytes + x.kv_bytes + x.state_bytes) * x.tp as u64).sum();
        s.shortfall = Some(format!("memory budget {:.1} GiB is below the {:.1} GiB this needs for context {context}: {error}", memory.unwrap_or(0) as f64 / GIB, used as f64 / GIB));
        return Ok(s);
    }
    if let Some((best, mut s)) = search.largest_context(None) {
        s.shortfall = Some(format!(
            "context {context} does not fit: the most is {best} with {} KV and {} slot(s) ({error})",
            if s.kv_dtype.is_fp8() { "fp8" } else { "f16" },
            s.slots
        ));
        return Ok(s);
    }
    bail!("{} does not fit on these devices: {error}", req.spec.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::LinkConfig;
    use std::path::Path;

    fn spec() -> Spec {
        Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap()
    }

    fn gpu(i: usize, gib: f64, gbs: f64) -> DeviceConfig {
        DeviceConfig {
            name: format!("box/gpu{i}"),
            memory_gib: gib,
            bandwidth_gbs: gbs,
            tflops: 500.0,
            ordinal: Some(i),
            address: None,
            reserve_gib: Some(1.0),
            link: LinkConfig::default(),
            offload: None,
            host: Some("box".into()),
            peer: Some(LinkConfig { latency_ms: 0.01, bandwidth_gbs: 50.0 }),
            wgpu: false,
        }
    }

    fn pair() -> Vec<DeviceConfig> {
        vec![gpu(0, 86.0, 1792.0), gpu(1, 86.0, 1792.0)]
    }

    fn run(mode: Profile, target: Target, pins: &[(&str, Value)], devices: &[DeviceConfig], spec: &Spec) -> Solved {
        let pins = pins.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        solve(&PlacementRequest { mode, target, pins, devices, spec }).unwrap()
    }

    #[test]
    fn server_options_with_devices_and_no_tp_are_solved_for_max_performance() {
        let opts = crate::serve::ServeOptions {
            max_num_seqs: 8,
            placement: Some(crate::placement::PlacementConfig { devices: pair(), ..Default::default() }),
            ..Default::default()
        };
        let s = for_options(&spec(), &opts).unwrap().unwrap();
        assert_eq!((s.tp, s.slots), (2, 8), "{}", s.strategy);
        let one = crate::serve::ServeOptions {
            placement: Some(crate::placement::PlacementConfig { devices: pair(), max_tp: Some(1), ..Default::default() }),
            ..opts.clone()
        };
        assert_eq!(for_options(&spec(), &one).unwrap().unwrap().tp, 1);
        let plain = crate::serve::ServeOptions::default();
        assert!(for_options(&spec(), &plain).unwrap().is_none());
    }

    #[test]
    fn max_performance_on_two_matched_gpus_is_tp2_with_f16_kv_and_many_slots() {
        let s = run(Profile::MaxPerformance, Target::default(), &[], &pair(), &spec());
        assert_eq!(s.tp, 2, "{}", s.strategy);
        assert_eq!(s.placement.stages.len(), 1);
        assert_eq!(s.kv_dtype, KvDtype::F16);
        assert!(s.slots >= 8, "{} slots", s.slots);
        assert!(s.mtp);
        let names: Vec<String> = pair().iter().map(|d| crate::hardware::display_name(&d.name)).collect();
        assert!(s.strategy.starts_with(&format!("TP 2 across {} + {}:", names[0], names[1])), "{}", s.strategy);
        assert!(!s.strategy.contains('×'), "{}", s.strategy);
        assert!(s.shortfall.is_none());
    }

    #[test]
    fn low_memory_on_the_same_box_is_one_gpu_fp8_kv_and_two_slots() {
        let s = run(Profile::LowMemory, Target::default(), &[], &pair(), &spec());
        assert_eq!(s.tp, 1);
        assert_eq!(s.placement.stages.len(), 1);
        assert_eq!(s.kv_dtype, KvDtype::Fp8);
        assert_eq!(s.slots, 2);
        assert!(!s.mtp);
        assert!(s.strategy.contains("smallest that fits"), "{}", s.strategy);
    }

    #[test]
    fn a_card_that_drives_the_display_still_matches_its_twin_for_tp() {
        let devices = vec![gpu(0, 88.6, 1792.0), gpu(1, 94.4, 1792.0)];
        let s = run(Profile::MaxPerformance, Target { context: Some(32768), ..Default::default() }, &[], &devices, &spec());
        assert_eq!(s.tp, 2, "{}", s.strategy);
        let mixed = vec![gpu(0, 24.0, 936.0), gpu(1, 24.0, 1792.0)];
        let s = run(Profile::MaxPerformance, Target { context: Some(8192), ..Default::default() }, &[], &mixed, &spec());
        assert_eq!(s.tp, 1, "{}", s.strategy);
    }

    #[test]
    fn with_gpu0_disabled_a_single_gpu_load_runs_on_device_1() {
        let only_second = vec![gpu(1, 86.0, 1792.0)];
        for mode in [Profile::MaxPerformance, Profile::LowMemory] {
            let s = run(mode, Target::default(), &[], &only_second, &spec());
            assert_eq!(s.placement.stages.len(), 1);
            assert_eq!(s.single_device(), 1, "{mode:?}: {}", s.strategy);
        }
        let too_small_first = run(Profile::LowMemory, Target::default(), &[], &[gpu(0, 20.0, 900.0), gpu(1, 86.0, 1792.0)], &spec());
        assert_eq!(too_small_first.single_device(), 1, "{}", too_small_first.strategy);
    }

    #[test]
    fn the_solved_pool_passes_the_engines_own_pool_check_in_both_modes() {
        for mode in [Profile::MaxPerformance, Profile::LowMemory] {
            for context in [8192, 32768, 262144] {
                let spec = spec();
                let s = run(mode, Target { context: Some(context), ..Default::default() }, &[], &pair(), &spec);
                let opts = crate::serve::ServeOptions::from_solved(&s, Default::default()).unwrap();
                let len = opts.model_len(&spec).unwrap();
                let (spec, opts) = opts.for_kv_context(&spec, len);
                opts.check_pool(&spec).unwrap_or_else(|e| panic!("{mode:?} context {context}: {e} ({})", s.log_line()));
            }
        }
    }

    #[test]
    fn max_performance_at_262144_context_keeps_tp2_and_names_what_it_chose() {
        let s = run(Profile::MaxPerformance, Target { context: Some(262144), ..Default::default() }, &[], &pair(), &spec());
        assert_eq!(s.tp, 2);
        assert_eq!(s.context, 262144);
        assert!(s.kv_pool_bytes / (s.kv_pool_bytes / s.context as u64).max(1) >= 262144);
        assert!(!s.reasons.is_empty());
        assert!(s.shortfall.is_none());
    }

    #[test]
    fn a_model_too_big_for_one_gpu_is_split_with_tp_when_heads_divide() {
        let s = run(Profile::MaxPerformance, Target::default(), &[], &[gpu(0, 12.0, 1792.0), gpu(1, 12.0, 1792.0)], &spec());
        assert_eq!(s.tp, 2, "{}", s.strategy);
        assert!(s.strategy.contains("does not fit on one GPU"), "{}", s.strategy);
    }

    #[test]
    fn a_spec_tp_cannot_split_gets_a_pipeline() {
        let mut s = spec();
        s.attention.kv_heads = 3;
        let solved = run(Profile::MaxPerformance, Target::default(), &[], &[gpu(0, 12.0, 1792.0), gpu(1, 12.0, 1792.0)], &s);
        assert_eq!(solved.tp, 1);
        assert_eq!(solved.placement.stages.len(), 2, "{}", solved.strategy);
        assert!(solved.strategy.starts_with("pipeline over 2 GPUs"), "{}", solved.strategy);
    }

    #[test]
    fn two_equal_cards_with_unequal_free_memory_still_pair_for_tp() {
        let busy = [gpu(0, 42.8, 1792.0), gpu(1, 76.3, 1792.0)];
        let pinned = PlacementRequest { mode: Profile::MaxPerformance, target: Target::default(), pins: [(PIN_TP.to_string(), json!(2))].into(), devices: &busy, spec: &spec() };
        let s = solve(&pinned).unwrap();
        assert_eq!((s.tp, s.ordinals()), (2, vec![0, 1]), "{}", s.strategy);
        assert!(s.strategy.starts_with("TP 2 across"), "{}", s.strategy);
        let small = Spec::from_hf_config(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3-0.6b.config.json")).unwrap()).unwrap();
        let pinned = PlacementRequest { spec: &small, ..pinned };
        let s = solve(&pinned).unwrap();
        assert_eq!(s.tp, 2, "{}", s.strategy);
    }

    #[test]
    fn mismatched_gpus_never_form_a_tp_group_and_low_memory_takes_the_smaller_card() {
        let devices = [gpu(0, 86.0, 1792.0), gpu(1, 28.0, 900.0)];
        let max = run(Profile::MaxPerformance, Target::default(), &[], &devices, &spec());
        assert_eq!(max.tp, 1);
        let low = run(Profile::LowMemory, Target::default(), &[], &devices, &spec());
        assert_eq!(low.placement.stages.len(), 1);
        assert_eq!(low.placement.stages[0].device_name, "box/gpu1");
    }

    #[test]
    fn one_gpu_gives_that_gpu() {
        for mode in Profile::ALL {
            let s = run(mode, Target::default(), &[], &[gpu(0, 86.0, 1792.0)], &spec());
            assert_eq!(s.placement.stages.len(), 1);
            assert_eq!(s.tp, 1);
        }
    }

    #[test]
    fn a_memory_budget_picks_the_context_that_fits_or_names_the_shortfall() {
        let big = run(Profile::LowMemory, Target { memory: Some(40 << 30), context: Some(8192), ..Default::default() }, &[], &pair(), &spec());
        assert!(big.shortfall.is_none());
        assert_eq!(big.kv_dtype, KvDtype::Fp8);
        let small = run(Profile::LowMemory, Target { memory: Some(4 << 30), context: Some(8192), ..Default::default() }, &[], &pair(), &spec());
        let why = small.shortfall.expect("the budget cannot hold the weights");
        assert!(why.contains("memory budget"), "{why}");
    }

    #[test]
    fn a_context_nothing_holds_returns_the_closest_plan_with_the_shortfall() {
        let s = run(Profile::LowMemory, Target { context: Some(262144), ..Default::default() }, &[], &[gpu(0, 30.0, 1792.0)], &spec());
        let why = s.shortfall.as_deref().expect("shortfall");
        assert!(why.contains("context 262144 does not fit: the most is"), "{why}");
        assert!(s.context < 262144);
    }

    #[test]
    fn pinned_overrides_are_kept_and_the_rest_is_tuned_around_them() {
        let s = run(Profile::MaxPerformance, Target::default(), &[(PIN_SLOTS, json!(21)), (PIN_KV_DTYPE, json!("fp8"))], &pair(), &spec());
        assert_eq!(s.slots, 21);
        assert_eq!(s.kv_dtype, KvDtype::Fp8);
        assert_eq!(s.options[PIN_SLOTS], json!(21));
        let p = run(Profile::MaxPerformance, Target::default(), &[(PIN_MTP, json!(false))], &pair(), &spec());
        assert!(!p.mtp);
    }

    #[test]
    fn solved_serialises_for_the_load_log() {
        let s = run(Profile::LowMemory, Target::default(), &[], &pair(), &spec());
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"strategy\""));
        assert!(!s.log_line().is_empty());
    }
}
