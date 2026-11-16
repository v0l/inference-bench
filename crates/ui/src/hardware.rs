use crate::App;
use crate::job::Pending;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::cluster::{Preview, Prepared, Run, Selection, StageRow, TP_NOTE, bottleneck, candidate_devices, preview, runnable, runnable_plan, stage_rows};
use inference_bench_core::engine::Command;
use inference_bench_core::library::human_size;
use specrt::cluster::Peer;
use specrt::discovery::{PeerBrowser, PeerEvent};
use specrt::hardware::{Machine, display_name, short_gpu};
use specrt::host::{Host, HostConfig};
use specrt::placement::{DeviceConfig, Objective};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub struct Shown {
    key: String,
    pub preview: Result<Preview, String>,
    pub devices: Vec<DeviceConfig>,
}

pub struct HardwareState {
    pub auto: bool,
    pub machine: Option<Machine>,
    probe: Pending<Result<Machine, String>>,
    probe_at: Option<Instant>,
    pub probe_error: Option<String>,
    pub host: Option<Host>,
    pub host_error: Option<String>,
    host_sig: Option<(bool, String)>,
    browser: Option<PeerBrowser>,
    pub peers: BTreeMap<String, Peer>,
    pub selection: Selection,
    pub model: String,
    pub ctx: usize,
    pub slots: usize,
    pub objective: Objective,
    pub pair_input: BTreeMap<String, String>,
    pub pair_notice: Option<(bool, String)>,
    pub code: Option<String>,
    pub run: Option<Prepared>,
    pub run_label: Option<String>,
    pub slot: inference_bench_core::cluster::ClusterSlot,
    announced: Option<String>,
    pub shown: Option<Shown>,
}

impl Default for HardwareState {
    fn default() -> Self {
        Self {
            auto: false,
            machine: None,
            probe: Pending::default(),
            probe_at: None,
            probe_error: None,
            host: None,
            host_error: None,
            host_sig: None,
            browser: None,
            peers: BTreeMap::new(),
            selection: Selection::default(),
            model: String::new(),
            ctx: 8192,
            slots: 1,
            objective: Objective::Latency,
            pair_input: BTreeMap::new(),
            pair_notice: None,
            code: None,
            run: None,
            run_label: None,
            slot: Default::default(),
            announced: None,
            shown: None,
        }
    }
}

impl HardwareState {
    pub fn with_slot(slot: inference_bench_core::cluster::ClusterSlot) -> Self {
        Self { slot, ..Self::default() }
    }

    pub fn release(&mut self) {
        *self.slot.lock().unwrap() = None;
        if let Some(mut r) = self.run.take() {
            r.release();
        }
        self.run_label = None;
    }

    pub fn peer_list(&self, own: &str) -> Vec<Peer> {
        self.peers.values().filter(|p| p.host != own).cloned().collect()
    }
}

pub fn gpu_lines(m: &Machine) -> Vec<String> {
    m.gpus
        .iter()
        .map(|g| {
            format!(
                "GPU {}  {}  {}  {} total  {} free  {:.0} GB/s  {:.0} TFLOP/s",
                g.ordinal,
                short_gpu(&g.name),
                g.arch(),
                human_size(g.total_bytes),
                human_size(g.free_bytes),
                g.bandwidth_gbs,
                g.tflops
            )
        })
        .collect()
}

pub fn link_lines(m: &Machine) -> Vec<String> {
    m.links.iter().map(|l| format!("GPU {} and GPU {}: {}", l.a, l.b, l.kind.label())).collect()
}

pub fn peer_line(p: &Peer) -> String {
    let state = if p.busy { "busy" } else { "idle" };
    let serving = p.serving.as_deref().map_or(String::new(), |s| format!(", serving {s}"));
    let address = p.control.as_deref().unwrap_or("no address");
    format!("{}  {:.0} GB  {state}{serving}  {address}", p.summary(), p.memory_gib() * 1.073_741_824)
}

pub fn pump(app: &mut App) {
    let wake = app.wake.clone();
    let s = &mut app.hardware;
    if let Some(result) = s.probe.poll() {
        match result {
            Ok(m) => {
                s.machine = Some(m);
                s.probe_error = None;
            }
            Err(e) => s.probe_error = Some(e),
        }
    }
    let wanted = s.auto && (app.tab == crate::Tab::Hardware || app.settings.cluster.share);
    if wanted && !s.probe.busy() && s.probe_at.is_none_or(|t| t.elapsed() > Duration::from_secs(if s.machine.is_some() { 5 } else { 30 })) {
        s.probe_at = Some(Instant::now());
        s.probe.start(crate::job::spawn(&wake, || specrt::hardware::probe().map_err(|e| format!("{e:#}"))));
    }
    if wanted && s.browser.is_none() {
        match PeerBrowser::start(app.settings.cluster.mdns_port, app.settings.cluster.interface()) {
            Ok(b) => s.browser = Some(b),
            Err(e) => s.host_error = Some(format!("discovery: {e:#}")),
        }
    }
    if let Some(b) = &s.browser {
        while let Ok(e) = b.events().try_recv() {
            match e {
                PeerEvent::Up(p) => {
                    s.peers.insert(p.host.clone(), p);
                }
                PeerEvent::Down { host } => {
                    s.peers.remove(&host);
                }
            }
        }
    }
    let sig = (app.settings.cluster.share, app.settings.cluster.interface.clone());
    if s.auto && s.host_sig.as_ref() != Some(&sig) {
        s.host = None;
        s.host_sig = Some(sig.clone());
        s.browser = None;
        if sig.0
            && let Some(machine) = s.machine.clone()
        {
            let config = HostConfig {
                key_file: inference_bench_core::cluster::key_path(&app.dir),
                mdns_port: app.settings.cluster.mdns_port,
                interface: app.settings.cluster.interface().map(str::to_string),
                bind: "0.0.0.0:0".into(),
                app_version: env!("CARGO_PKG_VERSION").into(),
            };
            let shared = machine.gpus.iter().map(|g| g.ordinal).collect();
            match Host::start(config, machine, shared) {
                Ok(h) => {
                    s.host = Some(h);
                    s.host_error = None;
                }
                Err(e) => s.host_error = Some(format!("{e:#}")),
            }
        } else if sig.0 {
            s.host_sig = None;
        }
    }
    if let Some(h) = &s.host {
        let serving = app.engine.loaded().map(|l| l.display);
        if s.announced != serving {
            s.announced = serving.clone();
            if let Err(e) = h.set_serving(serving) {
                s.host_error = Some(format!("{e:#}"));
            }
        }
        if let Some(m) = &s.machine {
            let _ = h.refresh(m.clone());
        }
    }
}

fn model_ids(app: &App) -> Vec<(String, String)> {
    crate::modelbar::usable(app, "").into_iter().map(|i| (app.library[i].name.clone(), app.library[i].model_id(inference_bench_core::target::NATIVE))).collect()
}

fn refresh_preview(app: &mut App) {
    let Some(machine) = app.hardware.machine.clone() else { return };
    let own = machine.host.clone();
    let peers = app.hardware.peer_list(&own);
    let net = app.settings.cluster.network_link();
    let sel = app.hardware.selection.clone();
    let devices = candidate_devices(&machine, &peers, &sel, net);
    let ids = model_ids(app);
    let Some((_, id)) = ids.iter().find(|(n, _)| *n == app.hardware.model).cloned() else {
        app.hardware.shown = None;
        return;
    };
    let (ctx, slots) = (app.hardware.ctx, app.hardware.slots);
    let key = format!("{id}|{ctx}|{slots}|{devices:?}");
    if app.hardware.shown.as_ref().is_some_and(|s| s.key == key) {
        return;
    }
    let preview = specrt::engine::resolve_spec(&id).map_err(|e| format!("{e:#}")).map(|spec| preview(&spec, devices.clone(), ctx, slots));
    app.hardware.shown = Some(Shown { key, preview, devices });
}

fn stage_table(ui: &mut egui::Ui, rows: &[StageRow]) {
    for (i, r) in rows.iter().enumerate() {
        let mut text = format!("stage {i}  {}  layers {}  {:.1} of {:.1} GiB  step {:.2} ms  hop {:.2} ms ({})", r.label, r.layers, r.gib, r.of_gib, r.step_ms, r.hop_ms, r.hop);
        if r.needs_tensor_parallel {
            text.push_str(&format!("  {TP_NOTE}"));
        }
        note(ui, text, if r.needs_tensor_parallel { WARN } else { VALUE });
    }
}

pub fn placement_lines(placement: &specrt::placement::Placement, devices: &[DeviceConfig], local: &str) -> (Vec<StageRow>, String) {
    let rows = stage_rows(placement, devices, local);
    let summary = format!(
        "step {:.2} ms ({:.1} tok/s one stream), slowest stage {:.2} ms ({:.1} tok/s pipelined), bottleneck {}",
        placement.step_secs * 1e3,
        placement.single_stream_tokens_per_second(),
        placement.bottleneck_secs * 1e3,
        placement.pipelined_tokens_per_second(),
        bottleneck(placement).unwrap_or_default()
    );
    (rows, summary)
}

fn objective_view(ui: &mut egui::Ui, label: &str, result: &Result<specrt::placement::Placement, String>, devices: &[DeviceConfig], local: &str) {
    Line::new().legend(label).show(ui);
    match result {
        Err(e) => status(ui, false, e),
        Ok(p) => {
            let (rows, summary) = placement_lines(p, devices, local);
            stage_table(ui, &rows);
            note(ui, summary, LEGEND);
            if let Err(e) = runnable(p) {
                status(ui, false, &e);
            }
        }
    }
}

fn this_machine(app: &mut App, ui: &mut egui::Ui) {
    section(ui, "this machine", "read from the CUDA driver", |ui| {
        let Some(m) = &app.hardware.machine else {
            match &app.hardware.probe_error {
                Some(e) => status(ui, false, e),
                None => note(ui, "reading the GPUs", LEGEND),
            }
            return;
        };
        Line::new().legend("host").value(m.host.clone()).gap(14.0).legend("ram").measured(format!("{} / {} available", human_size(m.ram_available_bytes), human_size(m.ram_total_bytes))).show(ui);
        if m.gpus.is_empty() {
            note(ui, "no CUDA GPU found", WARN);
        }
        for line in gpu_lines(m) {
            note(ui, line, VALUE);
        }
        for line in link_lines(m) {
            note(ui, line, LEGEND);
        }
        if m.interfaces.is_empty() {
            note(ui, "no network interface is up for mDNS", WARN);
        }
        for i in &m.interfaces {
            note(ui, format!("network {} {}", i.name, i.ip), LEGEND);
        }
    });
}

fn sharing(app: &mut App, ui: &mut egui::Ui) {
    section(ui, "cluster", "other inference-bench machines on this network", |ui| {
        let mut share = app.settings.cluster.share;
        if switch(ui, "share", &mut share, "Share this machine's GPU in a cluster", "Off by default. On announces this machine over mDNS and lets a paired coordinator load layers on its GPUs.") {
            app.settings.cluster.share = share;
            app.save_settings();
        }
        let mut names = vec![(String::new(), "all interfaces".to_string())];
        if let Some(m) = &app.hardware.machine {
            names.extend(m.interfaces.iter().map(|i| (i.name.clone(), format!("{} {}", i.name, i.ip))));
        }
        let mut picked = app.settings.cluster.interface.clone();
        row(ui, "interface", |ui| {
            if choice(ui, "cluster-interface", &mut picked, names) {
                app.settings.cluster.interface = picked.clone();
                app.save_settings();
            }
        });
        if let Some(e) = &app.hardware.host_error {
            status(ui, false, e);
        }
        if let Some(h) = &app.hardware.host {
            let paired = h.paired();
            ui.horizontal(|ui| {
                if ui.add(Key::new("show pairing code").outline(READOUT)).clicked() {
                    match h.show_code() {
                        Ok(c) => app.hardware.code = Some(c),
                        Err(e) => app.hardware.pair_notice = Some((false, format!("{e:#}"))),
                    }
                }
                lamp(ui, if paired { "paired" } else { "not paired" }, paired, false);
            });
            if let Some((code, left)) = h.code() {
                Line::new().legend("code").set(code).gap(12.0).note(format!("type it on the machine that will run the cluster, {left} s left")).show(ui);
            }
        }
        if let Some((ok, text)) = &app.hardware.pair_notice {
            status(ui, *ok, text);
        }
        let own = app.hardware.machine.as_ref().map(|m| m.host.clone()).unwrap_or_default();
        let peers = app.hardware.peer_list(&own);
        if peers.is_empty() {
            note(ui, "no other inference-bench machine found yet", LEGEND);
        }
        for p in &peers {
            note(ui, peer_line(p), VALUE);
        }
    });
}

fn pairing(app: &mut App, ui: &mut egui::Ui) {
    let own = app.hardware.machine.as_ref().map(|m| m.host.clone()).unwrap_or_default();
    let peers = app.hardware.peer_list(&own);
    if peers.is_empty() {
        return;
    }
    section(ui, "pair", "a peer shows a short code, enter it here once", |ui| {
        for p in peers {
            let paired = app.settings.cluster.paired.contains(&p.host);
            ui.horizontal(|ui| {
                Line::new().legend(&p.host).show(ui);
                if paired {
                    lamp(ui, "paired", true, false);
                    return;
                }
                let mut input = app.hardware.pair_input.get(&p.host).cloned().unwrap_or_default();
                let mut pressed = false;
                let ready = !input.trim().is_empty() && p.control.is_some();
                field_then(ui, &mut input, "code shown on that machine", 90.0, |ui| {
                    pressed = ui.add(Key::new("pair").outline(READOUT).enabled(ready).id_salt(("pair", &p.host))).clicked();
                });
                app.hardware.pair_input.insert(p.host.clone(), input.clone());
                if pressed {
                    let control = p.control.clone().unwrap_or_default();
                    let key_file = inference_bench_core::cluster::key_path(&app.dir);
                    let outcome = specrt::pairing::ensure_key(&key_file).and_then(|k| specrt::host::pair_with(&control, &input, &k));
                    match outcome {
                        Ok(()) => {
                            app.settings.cluster.paired.insert(p.host.clone());
                            app.save_settings();
                            app.hardware.pair_notice = Some((true, format!("paired with {}", p.host)));
                        }
                        Err(e) => app.hardware.pair_notice = Some((false, format!("{e:#}"))),
                    }
                }
            });
        }
    });
}

fn run_section(app: &mut App, ui: &mut egui::Ui) {
    let Some(machine) = app.hardware.machine.clone() else { return };
    section(ui, "run a cluster", "this app is the coordinator, peers run the other layers", |ui| {
        let ids = model_ids(app);
        if app.hardware.model.is_empty()
            && let Some((n, _)) = ids.first()
        {
            app.hardware.model = n.clone();
        }
        let mut picked = app.hardware.model.clone();
        row(ui, "model", |ui| {
            choice(ui, "cluster-model", &mut picked, ids.iter().map(|(n, _)| (n.clone(), n.clone())));
        });
        app.hardware.model = picked;
        row(ui, "context", |ui| {
            ui.add(egui::DragValue::new(&mut app.hardware.ctx).range(256..=1 << 20).speed(64.0));
        });
        row(ui, "parallel", |ui| {
            ui.add(egui::DragValue::new(&mut app.hardware.slots).range(1..=64));
        });
        Line::new().legend("gpus").show(ui);
        for g in &machine.gpus {
            let name = specrt::hardware::device_name(&machine.host, g.ordinal);
            let mut on = app.hardware.selection.includes(&name);
            if ui.checkbox(&mut on, display_name(&name)).changed() {
                app.hardware.selection.toggle_gpu(&name);
            }
        }
        let peers = app.hardware.peer_list(&machine.host);
        for p in &peers {
            let paired = app.settings.cluster.paired.contains(&p.host);
            let mut on = app.hardware.selection.peers.contains(&p.host);
            let label = if paired { format!("use {}", p.summary()) } else { format!("use {} (not paired)", p.summary()) };
            if ui.add_enabled(paired, egui::Checkbox::new(&mut on, label)).changed() {
                app.hardware.selection.toggle_peer(&p.host);
            }
            if on {
                for g in &p.gpus {
                    let mut gon = app.hardware.selection.includes(&g.device.name);
                    if ui.checkbox(&mut gon, format!("    {}", display_name(&g.device.name))).changed() {
                        app.hardware.selection.toggle_gpu(&g.device.name);
                    }
                }
            }
        }
        refresh_preview(app);
        let mut runnable_choice = None;
        match &app.hardware.shown {
            None => note(ui, "pick a model on disk to see the placement", LEGEND),
            Some(shown) => match &shown.preview {
                Err(e) => status(ui, false, e),
                Ok(p) => {
                    objective_view(ui, "latency", &p.latency, &shown.devices, &machine.host);
                    objective_view(ui, "throughput", &p.throughput, &shown.devices, &machine.host);
                    runnable_choice = Some((p.latency.as_ref().ok().is_some_and(|x| runnable(x).is_ok()), p.throughput.as_ref().ok().is_some_and(|x| runnable(x).is_ok())));
                }
            },
        }
        let mut objective = app.hardware.objective;
        segments(ui, &mut objective, &[(Objective::Latency, "latency", None), (Objective::Throughput, "throughput", None)]);
        app.hardware.objective = objective;
        let can = runnable_choice.is_some_and(|(l, t)| if objective == Objective::Latency { l } else { t }) && app.loading.is_none();
        ui.horizontal(|ui| {
            if ui.add(Key::new("load across the cluster").solid(READOUT).enabled(can).min_width(200.0)).clicked() {
                load(app);
            }
            if app.hardware.run.is_some() && ui.add(Key::new("stop cluster").outline(FAULT)).clicked() {
                app.unload();
            }
        });
        if let Some(l) = &app.hardware.run_label {
            note(ui, format!("running {l}"), OK);
        }
    });
}

pub fn load(app: &mut App) {
    let Some(machine) = app.hardware.machine.clone() else { return };
    let Some(shown) = &app.hardware.shown else { return };
    let devices = shown.devices.clone();
    let Some(model) = app.library.iter().find(|m| m.name == app.hardware.model).cloned() else { return };
    let id = model.model_id(inference_bench_core::target::NATIVE);
    let (ctx, slots) = (app.hardware.ctx, app.hardware.slots);
    let spec = match specrt::engine::resolve_spec(&id) {
        Ok(s) => s,
        Err(e) => {
            app.notice = Some((false, format!("{e:#}")));
            return;
        }
    };
    let placement = match runnable_plan(&spec, &devices, ctx, slots, app.hardware.objective) {
        Ok(p) => p,
        Err(e) => {
            app.notice = Some((false, e));
            return;
        }
    };
    app.hardware.release();
    let peers = app.hardware.peer_list(&machine.host);
    let key_file = inference_bench_core::cluster::key_path(&app.dir);
    let prepared = inference_bench_core::cluster::prepare(&Run { model_id: &id, ctx, slots, placement: &placement, devices: &devices, local: &machine, peers: &peers, settings: &app.settings.cluster, key_file: &key_file });
    match prepared {
        Ok(p) => {
            *app.hardware.slot.lock().unwrap() = Some(p.placement.clone());
            app.hardware.run_label = Some(format!("{} on {} stages", model.name, p.placement.stages.len()));
            app.hardware.run = Some(p);
            let engine = app.settings.engine_for(&model.name);
            app.engine.send(Command::Load { model_id: id, display: model.name.clone(), max_ctx: Some(ctx), max_num_seqs: Some(slots), kv_cache_bytes: None, served_names: app.settings.served_names_for(&model.name), engine });
        }
        Err(e) => app.notice = Some((false, format!("{e:#}"))),
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    pump(app);
    page(ui, "hardware", |ui| {
        this_machine(app, ui);
        sharing(app, ui);
        pairing(app, ui);
        run_section(app, ui);
    });
}

pub fn pipeline_rows(stats: &specrt::serve::Stats) -> Vec<String> {
    let Some(c) = &stats.chain else { return Vec::new() };
    let mut rows: Vec<String> = c
        .stages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = if m.name.is_empty() { format!("stage {i}") } else { format!("stage {i} {}", display_name(&m.name)) };
            format!("{name}  layers {}..{}  busy {:.1} s  {} calls  {} tokens", m.start, m.end, m.busy_secs, m.calls, m.tokens)
        })
        .collect();
    let hop_ms = if c.hops > 0 { c.hop_secs / c.hops as f64 * 1e3 } else { 0.0 };
    rows.push(format!("hops {}  {:.2} ms each  {} moved", c.hops, hop_ms, human_size(c.hop_bytes)));
    if let Some(lost) = &c.lost {
        rows.push(format!("lost: {lost}"));
    }
    rows
}

pub fn pipeline_section(app: &App, ui: &mut egui::Ui) {
    let Some(stats) = app.engine.handle().serving_stats() else { return };
    let rows = pipeline_rows(&stats);
    if rows.is_empty() {
        return;
    }
    section(ui, "pipeline", "per stage work and the hops between stages", |ui| {
        for r in rows {
            note(ui, r.clone(), if r.starts_with("lost") { FAULT } else { VALUE });
        }
    });
}
