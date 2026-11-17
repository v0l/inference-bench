use specrt::discovery::{PeerBrowser, PeerEvent};
use std::time::{Duration, Instant};

#[test]
#[ignore = "browses the real LAN over mDNS for 12 seconds and prints every inference-bench host"]
fn live_browse_prints_every_peer() {
    let browser = PeerBrowser::start(5353, None).unwrap();
    let end = Instant::now() + Duration::from_secs(12);
    while Instant::now() < end {
        while let Ok(e) = browser.events().try_recv() {
            match e {
                PeerEvent::Up(p) => println!("UP {} app={:?} control={:?} gpus={:?}", p.host, p.app_version, p.control, p.gpus.iter().map(|g| (g.gpu_name.clone(), g.backend, g.device.memory_gib)).collect::<Vec<_>>()),
                PeerEvent::Down { host } => println!("DOWN {host}"),
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
