use crate::cluster::{Event, SERVICE_NAME, decode_announcement, encode_announcement};
use crate::placement::DeviceConfig;
use anyhow::{Context, Result};
use mdns_sd::{IfKind, ServiceDaemon, ServiceEvent, ServiceInfo};
use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

fn daemon_on(port: u16, interface: Option<&str>) -> Result<ServiceDaemon> {
    let daemon = ServiceDaemon::new_with_port(port).context("mdns daemon")?;
    if let Some(name) = interface {
        daemon.disable_interface(IfKind::All).context("mdns interfaces")?;
        let kind = match name.parse::<std::net::IpAddr>() {
            Ok(ip) => IfKind::Addr(ip),
            Err(_) => IfKind::Name(name.to_string()),
        };
        daemon.enable_interface(kind).with_context(|| format!("mdns interface {name}"))?;
    }
    Ok(daemon)
}

pub struct Announcer {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Announcer {
    pub fn start(device: &DeviceConfig, addr: SocketAddr, port: u16) -> Result<Announcer> {
        Self::start_on(device, addr, port, None)
    }

    pub fn start_on(device: &DeviceConfig, addr: SocketAddr, port: u16, interface: Option<&str>) -> Result<Announcer> {
        let daemon = daemon_on(port, interface)?;
        let txt: Vec<(String, String)> = encode_announcement(device, &addr.to_string())
            .into_iter()
            .filter_map(|e| e.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect();
        let host = format!("{}.local.", device.name);
        let ip = if addr.ip().is_unspecified() { String::new() } else { addr.ip().to_string() };
        let mut info = ServiceInfo::new(SERVICE_NAME, &device.name, &host, ip.as_str(), addr.port(), &txt[..]).context("mdns service info")?;
        if addr.ip().is_unspecified() {
            info = info.enable_addr_auto();
        }
        let fullname = info.get_fullname().to_string();
        daemon.register(info).context("mdns register")?;
        Ok(Announcer { daemon, fullname })
    }

    pub fn withdraw(&self) {
        if let Ok(done) = self.daemon.unregister(&self.fullname) {
            let _ = done.recv_timeout(std::time::Duration::from_secs(2));
        }
    }
}

impl Drop for Announcer {
    fn drop(&mut self) {
        self.withdraw();
        if let Ok(done) = self.daemon.shutdown() {
            let _ = done.recv_timeout(std::time::Duration::from_secs(2));
        }
    }
}

pub struct Browser {
    daemon: ServiceDaemon,
    events: Receiver<Event>,
}

fn instance_name(fullname: &str) -> String {
    fullname.strip_suffix(SERVICE_NAME).map(|n| n.trim_end_matches('.')).unwrap_or(fullname).to_string()
}

impl Browser {
    pub fn start(port: u16) -> Result<Browser> {
        Self::start_on(port, None)
    }

    pub fn start_on(port: u16, interface: Option<&str>) -> Result<Browser> {
        let daemon = daemon_on(port, interface)?;
        let found = daemon.browse(SERVICE_NAME).context("mdns browse")?;
        let (tx, events) = channel();
        let epoch = Instant::now();
        std::thread::Builder::new()
            .name("specrt-mdns".into())
            .spawn(move || {
                while let Ok(event) = found.recv() {
                    let out = match event {
                        ServiceEvent::ServiceResolved(r) => {
                            let txt: Vec<String> = r.get_properties().iter().map(|p| format!("{}={}", p.key(), p.val_str())).collect();
                            match decode_announcement(&txt) {
                                Ok((device, address)) => Some(Event::Join { device, address, now_ms: epoch.elapsed().as_millis() as u64 }),
                                Err(e) => {
                                    tracing::warn!(service = %r.get_fullname(), error = %format!("{e:#}"), "ignoring a malformed stage announcement");
                                    None
                                }
                            }
                        }
                        ServiceEvent::ServiceRemoved(_, fullname) => Some(Event::Leave { name: instance_name(&fullname) }),
                        ServiceEvent::SearchStopped(_) => break,
                        _ => None,
                    };
                    if let Some(e) = out
                        && tx.send(e).is_err()
                    {
                        break;
                    }
                }
            })
            .context("mdns browse thread")?;
        Ok(Browser { daemon, events })
    }

    pub fn events(&self) -> &Receiver<Event> {
        &self.events
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.daemon.stop_browse(SERVICE_NAME);
        if let Ok(done) = self.daemon.shutdown() {
            let _ = done.recv_timeout(std::time::Duration::from_secs(2));
        }
    }
}
