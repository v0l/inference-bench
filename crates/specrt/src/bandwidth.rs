use anyhow::{Result, anyhow};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

const PROBE_SOURCE: &str = include_str!("../scripts/bandwidth.cu");
const BUFFER_BYTES: usize = 1 << 30;
const ROUNDS: usize = 10;

#[derive(Clone, Copy, Debug)]
pub struct Wall {
    pub read_gbs: f64,
    pub copy_gbs: f64,
}

pub fn probe(ctx: &std::sync::Arc<CudaContext>) -> Result<Wall> {
    let (major, minor) = ctx.compute_capability()?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let source = format!("#define SPECRT_NVRTC 1\n{PROBE_SOURCE}");
    let ptx = compile_ptx_with_opts(&source, CompileOptions { arch: Some(arch), ..Default::default() }).map_err(|e| anyhow!("nvrtc bandwidth probe: {e:?}"))?;
    let module = ctx.load_module(Ptx::from_src(ptx.to_src()))?;
    let read = module.load_function("read_sum")?;
    let copy = module.load_function("copy")?;
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let src = stream.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let dst = stream.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let sink = stream.alloc_zeros::<u32>(1)?;
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    let cfg = LaunchConfig { grid_dim: (sms * 8, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
    let n = (BUFFER_BYTES / 16) as u64;
    let (mut best_read, mut best_copy) = (f32::MAX, f32::MAX);
    for round in 0..ROUNDS + 2 {
        let (s, e) = (ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?, ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?);
        s.record(&stream)?;
        unsafe { stream.launch_builder(&read).arg(&src).arg(&n).arg(&sink).launch(cfg) }?;
        e.record(&stream)?;
        e.synchronize()?;
        let read_ms = s.elapsed_ms(&e)?;
        s.record(&stream)?;
        unsafe { stream.launch_builder(&copy).arg(&src).arg(&dst).arg(&n).launch(cfg) }?;
        e.record(&stream)?;
        e.synchronize()?;
        let copy_ms = s.elapsed_ms(&e)?;
        if round >= 2 {
            best_read = best_read.min(read_ms);
            best_copy = best_copy.min(copy_ms);
        }
    }
    let bytes = BUFFER_BYTES as f64;
    Ok(Wall { read_gbs: bytes / best_read as f64 / 1e6, copy_gbs: 2.0 * bytes / best_copy as f64 / 1e6 })
}
