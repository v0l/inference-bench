use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

const BUFFER_BYTES: usize = 1 << 30;
const ROUNDS: usize = 10;

#[derive(Clone, Copy, Debug)]
pub struct Wall {
    pub read_gbs: f64,
    pub copy_gbs: f64,
}

pub fn probe(ctx: &std::sync::Arc<CudaContext>) -> Result<Wall> {
    let target = crate::codegen::cuda_target(ctx)?;
    let b = kir::bandwidth::Bandwidth::probe(&target).map_err(|e| anyhow!(e))?;
    let module = crate::jit::load_uncached(ctx, &kir::cuda::emit_bandwidth(&b)).map_err(|e| anyhow!("bandwidth probe: {e:#}"))?;
    let read = module.load_function(kir::bandwidth::READ)?;
    let copy = module.load_function(kir::bandwidth::COPY)?;
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let src = stream.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let dst = stream.alloc_zeros::<u8>(BUFFER_BYTES)?;
    let sink = stream.alloc_zeros::<u32>(1)?;
    let cfg = LaunchConfig { grid_dim: (b.grid(), 1, 1), block_dim: (b.threads, 1, 1), shared_mem_bytes: 0 };
    let n = (BUFFER_BYTES / kir::bandwidth::VECTOR_BYTES) as u64;
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
