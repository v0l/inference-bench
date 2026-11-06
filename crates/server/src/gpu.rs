use std::process::Command;

pub fn used_bytes() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mib: u64 = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}
