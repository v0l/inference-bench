use specrt::hardware::fake_gpus;
use specrt::host::{Host, HostConfig, pair_with, release_stages, request_stages};
use specrt::pairing::generate_key;
use specrt::wire::StageClient;
use std::path::PathBuf;
use std::time::Duration;

fn spec_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml").display().to_string()
}

fn host(name: &str, port: u16) -> (Host, PathBuf) {
    let dir = std::env::temp_dir().join(format!("host-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let key_file = dir.join("cluster.key");
    let mut machine = fake_gpus(name, 2, true);
    machine.host = name.into();
    let config = HostConfig { key_file: key_file.clone(), mdns_port: port, interface: None, bind: "127.0.0.1:0".into(), app_version: "test".into() };
    (Host::start(config, machine, vec![0, 1]).unwrap(), dir)
}

#[test]
fn pairing_then_a_stage_request_opens_authenticated_stage_servers_for_each_shared_gpu() {
    let (peer, dir) = host("pair-peer", 15501);
    let control = peer.control_addr().to_string();
    let key = generate_key().unwrap();

    assert!(request_stages(&control, &key, &spec_path(), 512, 1, &[0]).unwrap_err().to_string().contains("not paired"));

    assert!(pair_with(&control, "AAAA-AAAA", &key).unwrap_err().to_string().contains("no pairing code"));
    let code = peer.show_code().unwrap();
    assert!(pair_with(&control, "ZZZZ-ZZZZ", &key).unwrap_err().to_string().contains("wrong pairing code"));
    pair_with(&control, &code, &key).unwrap();
    assert!(peer.paired());
    assert_eq!(std::fs::read_to_string(dir.join("cluster.key")).unwrap().trim(), key);
    assert!(peer.code().is_none());

    let wrong = generate_key().unwrap();
    assert!(request_stages(&control, &wrong, &spec_path(), 512, 1, &[0]).unwrap_err().to_string().contains("authentication failed"));
    assert!(request_stages(&control, &key, &spec_path(), 512, 1, &[5]).unwrap_err().to_string().contains("not shared"));

    let addrs = request_stages(&control, &key, &spec_path(), 512, 1, &[0, 1]).unwrap();
    assert_eq!(addrs.len(), 2);
    assert_eq!(peer.staged(), 2);
    let raw = specrt::pairing::unhex(&key).unwrap();
    for a in &addrs {
        assert!(StageClient::connect(a, key.as_bytes(), Duration::from_secs(5)).is_err());
        StageClient::connect(a, &raw, Duration::from_secs(5)).unwrap();
    }
    release_stages(&control, &key).unwrap();
    assert_eq!(peer.staged(), 0);
    let _ = std::fs::remove_dir_all(dir);
}
