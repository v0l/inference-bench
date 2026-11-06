use std::process::Command;

pub fn smi_used_bytes() -> Option<u64> {
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

#[cfg(feature = "cuda-probe")]
pub fn driver_used_bytes() -> Option<u64> {
    use std::sync::OnceLock;
    static CTX: OnceLock<Option<std::sync::Arc<cudarc::driver::CudaContext>>> = OnceLock::new();
    let ctx = CTX.get_or_init(|| cudarc::driver::CudaContext::new(0).ok()).as_ref()?;
    ctx.bind_to_thread().ok()?;
    let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
    Some(total.saturating_sub(free) as u64)
}

#[cfg(not(feature = "cuda-probe"))]
pub fn driver_used_bytes() -> Option<u64> {
    None
}

pub fn used_bytes() -> Option<u64> {
    smi_used_bytes().or_else(driver_used_bytes)
}
