use crate::App;
use crate::job::Pending;
use crate::shell::page;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::cluster::{Planned, Prepared, Run, Selection};
use inference_bench_core::engine::Command;
use inference_bench_core::library::human_size;
use specrt::cluster::Peer;
use specrt::discovery::{PeerBrowser, PeerEvent};
use specrt::hardware::{GpuInfo, Machine, display_name};
use specrt::host::{Host, HostConfig};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Default)]
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
    pub pair_input: BTreeMap<String, String>,
    pub pair_notice: Option<(bool, String)>,
    pub code: Option<String>,
    pub run: Option<Prepared>,
    pub run_label: Option<String>,
    pub slot: inference_bench_core::cluster::ClusterSlot,
    announced: Option<String>,
}


impl HardwareState {
    pub fn selection(&self, settings: &inference_bench_core::settings::ClusterSettings) -> Selection {
        Selection { peers: settings.peers_in_use.clone(), excluded: settings.disabled_gpus.clone() }
    }

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

fn tag(ui: &mut egui::Ui, text: &str, tint: egui::Color32, lit: bool) {
    let r = chip(ui, text, tint, lit);
    r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, text));
}

fn spoken(r: egui::Response, text: String) {
    r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, true, text.clone()));
}

fn peer_row(ui: &mut egui::Ui, p: &Peer, use_state: Option<bool>) -> bool {
    let mut toggled = false;
    ui.horizontal(|ui| {
        let r = Line::new().value(p.summary()).gap(10.0).measured(format!("{:.0} GB", p.memory_gib() * 1.073_741_824)).show(ui);
        spoken(r, format!("{}  {:.0} GB", p.summary(), p.memory_gib() * 1.073_741_824));
        tag(ui, if p.busy { "busy" } else { "idle" }, if p.busy { WARN } else { OK }, true);
        if let Some(s) = &p.serving {
            let text = format!("serving {s}");
            let r = Line::new().note(text.clone()).show(ui);
            spoken(r, text);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(on) = use_state {
                toggled = ui.add(Key::new(if on { "in use" } else { "not in use" }).outline(if on { OK } else { LEGEND }).id_salt(("peer-use", &p.host))).clicked();
            }
            let address = p.control.as_deref().unwrap_or("no address");
            let r = Line::new().note(address).show(ui);
            spoken(r, address.to_string());
        });
    });
    toggled
}

fn gpu_card(ui: &mut egui::Ui, g: &GpuInfo, on: bool, display: bool) -> bool {
    let mut toggled = false;
    let used = g.total_bytes.saturating_sub(g.free_bytes);
    card(
        ui,
        if on { Some(TRACE) } else { None },
        |ui| {
            let r = Line::new().legend(&format!("gpu {}", g.ordinal)).gap(10.0).value(g.label()).show(ui);
            spoken(r, format!("GPU {}  {}", g.ordinal, g.label()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                toggled = ui.add(Key::new(if on { "in use" } else { "switched off" }).outline(if on { OK } else { LEGEND }).id_salt(("gpu-switch", g.ordinal))).clicked();
                tag(ui, &g.arch(), LEGEND, false);
                if display {
                    tag(ui, "drives a display", WARN, false);
                }
            });
        },
        |ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 8.0), egui::Sense::hover());
            bar(&ui.painter_at(rect), rect, used as f32 / g.total_bytes.max(1) as f32, TRACE);
            let free = format!("{} free of {}", human_size(g.free_bytes), human_size(g.total_bytes));
            let r = Line::new().legend("memory").measured(format!("{} used", human_size(used))).gap(10.0).note(free.clone()).show(ui);
            spoken(r, free);
            let speed = format!("{:.0} GB/s  {:.0} TFLOP/s", g.bandwidth_gbs, g.tflops);
            let r = Line::new().legend("bandwidth").measured(format!("{:.0} GB/s", g.bandwidth_gbs)).gap(18.0).legend("compute").measured(format!("{:.0} TFLOP/s", g.tflops)).show(ui);
            spoken(r, speed);
        },
    );
    toggled
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
    let Some(stats) = app.loaded().and_then(|l| app.engine.handle().serving_stats_of(&l.display)) else { return };
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

pub fn planning_machine(app: &App) -> Option<specrt::hardware::Machine> {
    app.hardware.machine.clone().or_else(|| specrt::hardware::probe().ok())
}

pub fn load_devices(app: &App) -> Vec<specrt::placement::DeviceConfig> {
    app.inventory().devices(app.hardware.machine.as_ref(), app.settings.cluster.network_link())
}

pub fn load_spanning(app: &mut App, model: &inference_bench_core::library::LocalModel, planned: Planned, ctx: usize, seqs: usize, kv_cache_bytes: Option<u64>, served_names: Vec<String>, engine: specrt::options::EngineChoice) -> Result<(), String> {
    let machine = planning_machine(app).ok_or_else(|| "no GPU reading yet".to_string())?;
    let id = model.model_id(inference_bench_core::target::NATIVE);
    app.hardware.release();
    let own = machine.host.clone();
    let peers: Vec<Peer> = app.hardware.peer_list(&own);
    let key_file = inference_bench_core::cluster::key_path(&app.dir);
    let prepared = inference_bench_core::cluster::prepare(&Run { model_id: &id, ctx: planned.ctx, slots: planned.slots, sizing: planned.sizing, placement: &planned.placement, devices: &planned.devices, local: &machine, peers: &peers, settings: &app.settings.cluster, key_file: &key_file }).map_err(|e| format!("{e:#}"))?;
    *app.hardware.slot.lock().unwrap() = Some(prepared.placement.clone());
    app.hardware.run_label = Some(format!("{} on {} stages", model.name, prepared.placement.stages.len()));
    app.hardware.run = Some(prepared);
    app.engine.send(Command::Load { alongside: false, model_id: id, display: model.name.clone(), max_ctx: Some(ctx), max_num_seqs: Some(seqs), kv_cache_bytes, served_names, engine });
    Ok(())
}

fn this_machine(app: &mut App, ui: &mut egui::Ui) {
    section(ui, "this machine", "every GPU is used for loading unless you switch it off", |ui| {
        let Some(m) = app.hardware.machine.clone() else {
            match &app.hardware.probe_error {
                Some(e) => status(ui, false, e),
                None => crate::working::working(ui, "reading the GPUs"),
            }
            return;
        };
        let gpus = m.gpus.len().to_string();
        readouts(ui, &[("host", m.host.clone(), VALUE), ("gpus", gpus, VALUE), ("ram free", human_size(m.ram_available_bytes), TRACE), ("ram total", human_size(m.ram_total_bytes), VALUE)]);
        if m.gpus.is_empty() {
            note(ui, "no CUDA GPU found", WARN);
        }
        for g in &m.gpus {
            let name = specrt::hardware::device_name(&m.host, g.ordinal);
            let on = !app.settings.cluster.disabled_gpus.contains(&name);
            let display = app.inventory().card(&name).is_some_and(|c| c.display);
            if gpu_card(ui, g, on, display) {
                if !app.settings.cluster.disabled_gpus.remove(&name) {
                    app.settings.cluster.disabled_gpus.insert(name);
                }
                app.save_settings();
            }
        }
        if !m.links.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for l in &m.links {
                    let ok = l.kind == specrt::hardware::LinkKind::PeerToPeer;
                    tag(ui, &format!("GPU {} and GPU {}: {}", l.a, l.b, l.kind.label()), if ok { OK } else { WARN }, true);
                }
            });
        }
        if m.interfaces.is_empty() {
            note(ui, "no network interface is up for mDNS", WARN);
        }
        ui.horizontal_wrapped(|ui| {
            for i in &m.interfaces {
                tag(ui, &format!("{} {}", i.name, i.ip), LEGEND, false);
            }
        });
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
            let paired = app.settings.cluster.paired.contains(&p.host);
            let state = paired.then(|| app.settings.cluster.peers_in_use.contains(&p.host));
            if peer_row(ui, p, state) {
                if !app.settings.cluster.peers_in_use.remove(&p.host) {
                    app.settings.cluster.peers_in_use.insert(p.host.clone());
                }
                app.save_settings();
            }
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

fn running_section(app: &App, ui: &mut egui::Ui) {
    let Some(label) = &app.hardware.run_label else { return };
    let Some(machine) = &app.hardware.machine else { return };
    section(ui, "running now", "the model is spread over these GPUs", |ui| {
        note(ui, format!("running {label}"), OK);
        if let Some(p) = app.hardware.slot.lock().unwrap().clone() {
            for d in &p.stages {
                let name = d.device.clone();
                Line::new().legend("stage").value(format!("{}  layers {}..{}", display_name(&name), d.layers[0], d.layers[1])).show(ui);
            }
        }
        let _ = machine;
    });
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    pump(app);
    page(ui, "hardware", |ui| {
        this_machine(app, ui);
        sharing(app, ui);
        pairing(app, ui);
        running_section(app, ui);
    });
}
