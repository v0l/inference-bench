use cudarc::driver::sys;
use specrt::rdma::{DEFAULT_LIBRARY, Endpoint, Verbs};

fn check(r: sys::CUresult, what: &str) -> Result<(), String> {
    if r == sys::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(format!("{what}: {r:?}")) }
}

fn gpu_allowed() -> bool {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    let first = visible.split(',').next().unwrap_or("").trim().to_string();
    let on_gpu0 = first.is_empty() || first == "0";
    if on_gpu0 && std::env::var("SPECRT_DMABUF_ALLOW_GPU0").as_deref() != Ok("1") {
        eprintln!("SKIPPED: CUDA_VISIBLE_DEVICES={visible:?} would use GPU0 (the user's app). Run under `~/.agents/bin/gpu-run 1 --`.");
        return false;
    }
    eprintln!("using physical GPU {first} (CUDA_VISIBLE_DEVICES={visible:?})");
    true
}

fn export_and_register() -> Result<String, String> {
    let verbs = Verbs::load(DEFAULT_LIBRARY).map_err(|e| format!("{e:#}"))?;
    if !verbs.supports_dmabuf() {
        return Err("libibverbs has no ibv_reg_dmabuf_mr".into());
    }
    let ep = Endpoint::open(verbs, "10.100.50.1".parse().unwrap()).map_err(|e| format!("{e:#}"))?;
    let ordinal: i32 = 0;
    unsafe {
        check(sys::cuInit(0), "cuInit")?;
        let mut dev = 0;
        check(sys::cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
        let mut supported = 0;
        check(sys::cuDeviceGetAttribute(&mut supported, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev), "dma-buf attribute")?;
        if supported == 0 {
            return Err("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED is 0".into());
        }
        let mut ctx = std::ptr::null_mut();
        check(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        check(sys::cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
        let len = 8usize << 20;
        let mut ptr = 0;
        check(sys::cuMemAlloc_v2(&mut ptr, len), "cuMemAlloc")?;
        let mut fd: i32 = -1;
        check(
            sys::cuMemGetHandleForAddressRange(&mut fd as *mut i32 as *mut std::ffi::c_void, ptr, len, sys::CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
            "cuMemGetHandleForAddressRange",
        )?;
        let mr = ep.register_dmabuf(fd, 0, len, ptr).map_err(|e| format!("{e:#} (errno {})", std::io::Error::last_os_error()))?;
        let _ = mr;
        Ok(format!("registered {len} bytes of device ordinal {ordinal} memory (fd {fd}) with ibv_reg_dmabuf_mr"))
    }
}

fn bench() -> Result<(), String> {
    let verbs = Verbs::load(DEFAULT_LIBRARY).map_err(|e| format!("{e:#}"))?;
    let ip = "10.100.50.1".parse().unwrap();
    let mut a = Endpoint::open(verbs.clone(), ip).map_err(|e| format!("{e:#}"))?;
    let mut b = Endpoint::open(verbs, ip).map_err(|e| format!("{e:#}"))?;
    let (ia, ib) = (a.info(), b.info());
    a.connect(&ib).map_err(|e| format!("{e:#}"))?;
    b.connect(&ia).map_err(|e| format!("{e:#}"))?;
    let len = 64usize << 20;
    unsafe {
        check(sys::cuInit(0), "cuInit")?;
        let mut dev = 0;
        check(sys::cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
        let mut ctx = std::ptr::null_mut();
        check(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev), "retain")?;
        check(sys::cuCtxSetCurrent(ctx), "set current")?;
        let (mut src, mut dst) = (0, 0);
        check(sys::cuMemAlloc_v2(&mut src, len), "alloc src")?;
        check(sys::cuMemAlloc_v2(&mut dst, len), "alloc dst")?;
        let export = |ptr| -> Result<i32, String> {
            let mut fd: i32 = -1;
            check(sys::cuMemGetHandleForAddressRange(&mut fd as *mut i32 as *mut std::ffi::c_void, ptr, len, sys::CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0), "export")?;
            Ok(fd)
        };
        let mr_src = a.register_dmabuf(export(src)?, 0, len, src).map_err(|e| format!("{e:#}"))?;
        let mr_dst = b.register_dmabuf(export(dst)?, 0, len, dst).map_err(|e| format!("{e:#}"))?;
        let (lkey, _) = a.keys(mr_src);
        let (_, rkey) = b.keys(mr_dst);
        let stage = 1usize << 20;
        let mut host_a = vec![0u8; stage];
        let mut host_b = vec![0u8; stage];
        let (hlkey, _) = a.register_host(host_a.as_mut_ptr(), stage).map_err(|e| format!("{e:#}"))?;
        let (_, hrkey) = b.register_host(host_b.as_mut_ptr(), stage).map_err(|e| format!("{e:#}"))?;
        let hop = |a: &mut Endpoint, b: &mut Endpoint, local: u64, lk: u32, remote: u64, rk: u32, n: usize| -> Result<(), String> {
            a.write_imm(local, n, lk, remote, rk).map_err(|e| format!("{e:#}"))?;
            let (mut sent, mut got) = (false, false);
            while !(sent && got) {
                sent |= a.completions().map_err(|e| format!("{e:#}"))?.0 > 0;
                got |= b.completions().map_err(|e| format!("{e:#}"))?.1 > 0;
            }
            Ok(())
        };
        for (label, n, iters) in [("20KB", 20 * 1024usize, 2000u32), ("4MB", 4 << 20, 100), ("64MB", 64 << 20, 10)] {
            hop(&mut a, &mut b, src, lkey, dst, rkey, n)?;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                hop(&mut a, &mut b, src, lkey, dst, rkey, n)?;
            }
            let direct = t.elapsed() / iters;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                let mut off = 0;
                while off < n {
                    let piece = stage.min(n - off);
                    check(sys::cuMemcpyDtoH_v2(host_a.as_mut_ptr() as *mut _, src + off as u64, piece), "dtoh")?;
                    hop(&mut a, &mut b, host_a.as_ptr() as u64, hlkey, host_b.as_ptr() as u64, hrkey, piece)?;
                    check(sys::cuMemcpyHtoD_v2(dst + off as u64, host_b.as_ptr() as *const _, piece), "htod")?;
                    off += piece;
                }
            }
            let staged = t.elapsed() / iters;
            let gbs = |d: std::time::Duration| n as f64 / d.as_secs_f64() / 1e9;
            eprintln!("DMABUF {label}: gpu-direct {direct:?} ({:.2} GB/s), pinned host staging {staged:?} ({:.2} GB/s)", gbs(direct), gbs(staged));
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a CUDA GPU, run under gpu-run 1"]
fn gpu_to_gpu_hop_vs_host_staging() {
    if !gpu_allowed() {
        return;
    }
    if let Err(e) = bench() {
        eprintln!("DMABUF BENCH FAILED: {e}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU, run under gpu-run 1"]
fn gpu_memory_registers_through_dmabuf() {
    if !gpu_allowed() {
        return;
    }
    match export_and_register() {
        Ok(m) => eprintln!("DMABUF OK: {m}"),
        Err(e) => eprintln!("DMABUF REFUSED: {e}"),
    }
}
