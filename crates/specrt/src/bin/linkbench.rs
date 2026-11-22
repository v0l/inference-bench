use anyhow::Result;
use cudarc::driver::CudaContext;
use std::time::Instant;

const BYTES: usize = 1 << 30;
const RUNS: usize = 5;

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let mut host = unsafe { ctx.alloc_pinned_with_flags::<u8>(BYTES, 0) }?;
    host.as_mut_slice()?.fill(1);
    let dev = stream.alloc_zeros::<u8>(BYTES)?;
    let mut back = unsafe { ctx.alloc_pinned_with_flags::<u8>(BYTES, 0) }?;
    let (ptr, _sync) = {
        use cudarc::driver::DevicePtr;
        dev.device_ptr(&stream)
    };
    let mut up = Vec::new();
    let mut down = Vec::new();
    for run in 0..=RUNS {
        let started = Instant::now();
        unsafe { cudarc::driver::result::memcpy_htod_async(ptr, host.as_slice()?, stream.cu_stream()) }?;
        stream.synchronize()?;
        let h2d = BYTES as f64 / started.elapsed().as_secs_f64() / 1e9;
        let started = Instant::now();
        unsafe { cudarc::driver::result::memcpy_dtoh_async(back.as_mut_slice()?, ptr, stream.cu_stream()) }?;
        stream.synchronize()?;
        let d2h = BYTES as f64 / started.elapsed().as_secs_f64() / 1e9;
        if run > 0 {
            up.push(h2d);
            down.push(d2h);
        }
    }
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(", ");
    println!("pinned host to device, 1 GiB x {RUNS}: [{}] GB/s", fmt(&up));
    println!("pinned device to host, 1 GiB x {RUNS}: [{}] GB/s", fmt(&down));
    Ok(())
}
