use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use serde::Deserialize;
use specrt::hardware::Machine;
use specrt::host::{Host, HostConfig};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Args, Clone)]
pub struct HostArgs {
    #[arg(long, global = true, help = "Host config TOML (bind, mdns_port, interface, key_file, gpus, max_memory_gb)")]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub cmd: Option<HostCmd>,
}

#[derive(Subcommand, Clone)]
pub enum HostCmd {
    #[command(about = "Print a fresh pairing code from the running host")]
    Pair,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct HostFile {
    pub bind: String,
    pub mdns_port: u16,
    pub interface: Option<String>,
    pub key_file: Option<PathBuf>,
    pub control_socket: Option<PathBuf>,
    pub gpus: Vec<usize>,
    pub max_memory_gb: Option<f64>,
}

impl Default for HostFile {
    fn default() -> Self {
        Self { bind: "0.0.0.0:0".into(), mdns_port: 5353, interface: None, key_file: None, control_socket: None, gpus: Vec::new(), max_memory_gb: None }
    }
}

impl HostFile {
    pub fn parse(text: &str) -> Result<HostFile> {
        let file: HostFile = toml::from_str(text).context("host config")?;
        if let Some(gb) = file.max_memory_gb
            && !(gb > 0.0)
        {
            bail!("max_memory_gb must be above 0");
        }
        Ok(file)
    }

    pub fn load(path: Option<&Path>) -> Result<HostFile> {
        match path {
            Some(p) => Self::parse(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?),
            None => Ok(HostFile::default()),
        }
    }

    pub fn key_file(&self, data: &Path) -> PathBuf {
        self.key_file.clone().unwrap_or_else(|| inference_bench_core::cluster::key_path(data))
    }

    pub fn socket(&self, data: &Path) -> PathBuf {
        self.control_socket.clone().unwrap_or_else(|| self.key_file(data).with_file_name("host.sock"))
    }

    pub fn shared(&self, machine: &Machine) -> Result<Vec<usize>> {
        let all: Vec<usize> = machine.gpus.iter().map(|g| g.ordinal).collect();
        if self.gpus.is_empty() {
            return Ok(all);
        }
        if let Some(bad) = self.gpus.iter().find(|g| !all.contains(g)) {
            bail!("gpus lists GPU {bad}, this machine has {all:?}");
        }
        Ok(self.gpus.clone())
    }

    pub fn capped(&self, mut machine: Machine) -> Machine {
        if let Some(gb) = self.max_memory_gb {
            let cap = (gb * (1u64 << 30) as f64) as u64;
            for g in &mut machine.gpus {
                g.total_bytes = g.total_bytes.min(cap);
                g.free_bytes = g.free_bytes.min(cap);
            }
        }
        machine
    }
}

fn serve_socket(host: Arc<Host>, listener: UnixListener, stop: Arc<AtomicBool>) -> Result<std::thread::JoinHandle<()>> {
    listener.set_nonblocking(true)?;
    Ok(std::thread::Builder::new().name("host-pair-socket".into()).spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let line = match host.show_code() {
                        Ok(code) => format!("{code} {}\n", host.code().map_or(0, |(_, s)| s)),
                        Err(e) => format!("error {e:#}\n"),
                    };
                    let _ = stream.write_all(line.as_bytes());
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
    })?)
}

pub struct Running {
    host: Arc<Host>,
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Running {
    pub fn control_addr(&self) -> std::net::SocketAddr {
        self.host.control_addr()
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}

pub fn start(file: &HostFile, data: &Path, machine: Machine) -> Result<Running> {
    let shared = file.shared(&machine)?;
    let machine = file.capped(machine);
    let config = HostConfig { key_file: file.key_file(data), mdns_port: file.mdns_port, interface: file.interface.clone(), bind: file.bind.clone(), app_version: env!("CARGO_PKG_VERSION").into() };
    let socket = file.socket(data);
    if let Some(dir) = socket.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let _ = std::fs::remove_file(&socket);
    let host = Arc::new(Host::start(config, machine, shared)?);
    let listener = UnixListener::bind(&socket).with_context(|| format!("bind {}", socket.display()))?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    let stop = Arc::new(AtomicBool::new(false));
    let thread = serve_socket(host.clone(), listener, stop.clone())?;
    Ok(Running { host, socket, stop, thread: Some(thread) })
}

pub fn request_code(socket: &Path) -> Result<(String, u64)> {
    let stream = UnixStream::connect(socket).with_context(|| format!("no running host at {} (is inference-bench-host.service up?)", socket.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    let line = line.trim();
    if let Some(e) = line.strip_prefix("error ") {
        bail!("{e}");
    }
    let (code, secs) = line.rsplit_once(' ').context("bad reply from the host")?;
    Ok((code.to_string(), secs.parse()?))
}

static TERM: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    TERM.store(true, Ordering::SeqCst);
}

pub fn run(args: &HostArgs, data: &Path) -> Result<()> {
    let file = HostFile::load(args.config.as_deref())?;
    if let Some(HostCmd::Pair) = args.cmd {
        let (code, secs) = request_code(&file.socket(data))?;
        println!("{code}");
        println!("expires in {secs} s");
        return Ok(());
    }
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    let machine = specrt::hardware::probe()?;
    let running = start(&file, data, machine)?;
    tracing::info!(control = %running.control_addr(), socket = %running.socket().display(), mdns_port = file.mdns_port, "host is sharing this machine, pair with `inference-bench host pair`");
    while !TERM.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
    }
    tracing::info!("stopping host");
    drop(running);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use specrt::discovery::{PeerBrowser, PeerEvent};
    use specrt::hardware::fake_gpus;
    use specrt::host::pair_with;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ib-host-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_config_parses_and_rejects_unknown_keys() {
        let f = HostFile::parse("bind = \"10.100.50.2:7710\"\nmdns_port = 5353\ninterface = \"enp1s0f1np1\"\nkey_file = \"/var/lib/x/cluster.key\"\ngpus = [0]\nmax_memory_gb = 20\n").unwrap();
        assert_eq!(f.bind, "10.100.50.2:7710");
        assert_eq!(f.interface.as_deref(), Some("enp1s0f1np1"));
        assert_eq!(f.gpus, vec![0]);
        assert_eq!(f.max_memory_gb, Some(20.0));
        assert_eq!(f.socket(Path::new("/d")), PathBuf::from("/var/lib/x/host.sock"));
        assert_eq!(HostFile::parse("").unwrap(), HostFile::default());
        assert!(HostFile::parse("blind = 1").is_err());
        assert!(HostFile::parse("max_memory_gb = 0").is_err());
    }

    #[test]
    fn gpus_and_the_memory_cap_shape_what_is_shared() {
        let m = fake_gpus("h", 2, true);
        let f = HostFile::parse("gpus = [1]\nmax_memory_gb = 20").unwrap();
        assert_eq!(f.shared(&m).unwrap(), vec![1]);
        assert!(HostFile::parse("gpus = [3]").unwrap().shared(&m).is_err());
        let capped = f.capped(m);
        assert!(capped.gpus.iter().all(|g| g.total_bytes == 20 << 30 && g.free_bytes <= 20 << 30));
    }

    #[test]
    fn host_pair_gets_a_code_from_a_running_host_that_pairs_a_client() {
        let d = dir("pair");
        let file = HostFile { bind: "127.0.0.1:0".into(), mdns_port: 15601, ..HostFile::default() };
        let mut machine = fake_gpus("pair-host", 1, true);
        machine.host = "pair-host".into();
        let running = start(&file, &d, machine).unwrap();
        let (code, secs) = request_code(running.socket()).unwrap();
        assert_eq!(code.replace('-', "").len(), 8);
        assert!(secs > 0 && secs <= 300);
        let key = specrt::pairing::generate_key().unwrap();
        pair_with(&running.control_addr().to_string(), &code, &key).unwrap();
        assert_eq!(std::fs::read_to_string(d.join("cluster.key")).unwrap().trim(), key);
        let socket = running.socket().to_path_buf();
        drop(running);
        assert!(!socket.exists());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn stopping_the_host_withdraws_the_mdns_announcement() {
        let d = dir("mdns");
        let port = 15610;
        let file = HostFile { bind: "127.0.0.1:0".into(), mdns_port: port, ..HostFile::default() };
        let Ok(browser) = PeerBrowser::start(port, None) else { return };
        let mut machine = fake_gpus("mdns-host", 1, true);
        machine.host = "mdns-host".into();
        let running = start(&file, &d, machine).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(8);
        let mut up = false;
        while std::time::Instant::now() < deadline && !up {
            if let Ok(PeerEvent::Up(p)) = browser.events().recv_timeout(Duration::from_millis(200)) {
                up = p.host == "mdns-host";
            }
        }
        if !up {
            eprintln!("multicast is not available here, skipping");
            return;
        }
        drop(running);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        let mut down = false;
        while std::time::Instant::now() < deadline && !down {
            if let Ok(PeerEvent::Down { host }) = browser.events().recv_timeout(Duration::from_millis(200)) {
                down = host == "mdns-host";
            }
        }
        assert!(down, "the announcement was not withdrawn");
        let _ = std::fs::remove_dir_all(d);
    }
}
