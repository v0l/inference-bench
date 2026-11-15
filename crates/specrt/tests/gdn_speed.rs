use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

fn data(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len).map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0xffff) as f32 / 65535.0 * scale - scale / 2.0).collect()
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn gdn_chunk_kernels_speed_on_a_27b_shaped_layer() {
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    let la = spec.linear_attention.clone().unwrap();
    let (kh, vh, d) = (la.key_heads, la.value_heads, la.key_head_dim);
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-gdn-speed-{}", std::process::id()))).unwrap();
    let stream = ctx.default_stream();
    let module = Module::generate_kv(&spec, 4096, 1, &specrt::codegen::cuda_target(&ctx).unwrap()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let prep = loaded.function("gdn_chunk_prep").unwrap();
    let scan = loaded.function("gdn_chunk_scan").unwrap();
    let c = specrt::codegen::LA_CHUNK;
    let cols = specrt::codegen::LA_COLS;
    let smem = module.smem["gdn_chunk_prep"];
    use cudarc::driver::sys::CUfunction_attribute as A;
    prep.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
    let conv_dim = la.conv_dim();
    let lens: Vec<usize> = std::env::var("SPECRT_GDN_TOKENS").unwrap_or("1024,4096".into()).split(',').map(|v| v.parse().unwrap()).collect();
    for tokens in lens {
        let chunks = tokens.div_ceil(c);
        let unit = |v: Vec<f32>| stream.clone_htod(&v).unwrap();
        let qn = unit(data(tokens * kh * d, 1, 0.2));
        let kn = unit(data(tokens * kh * d, 2, 0.2));
        let mix = unit(data(tokens * conv_dim, 3, 1.0));
        let g = unit(data(tokens * vh, 4, -0.1).iter().map(|v| v - 0.05).collect());
        let beta = unit(data(tokens * vh, 5, 1.0).iter().map(|v| v + 0.5).collect());
        let w = stream.alloc_zeros::<f32>(tokens * vh * d).unwrap();
        let u = stream.alloc_zeros::<f32>(tokens * vh * d).unwrap();
        let o = stream.alloc_zeros::<f32>(tokens * vh * d).unwrap();
        let aqk = stream.alloc_zeros::<f32>(chunks * vh * c * c).unwrap();
        let qe = stream.alloc_zeros::<f32>(tokens * vh * d).unwrap();
        let kt = stream.alloc_zeros::<f32>(tokens * vh * d).unwrap();
        let gc = stream.alloc_zeros::<f32>(chunks * vh * c).unwrap();
        let state = stream.alloc_zeros::<f32>(vh * d * d).unwrap();
        let ntok = tokens as i32;
        let slot = 0i32;
        let time = |f: &dyn Fn()| {
            f();
            f();
            let mut times = Vec::new();
            for _ in 0..10 {
                let e0 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e0.record(&stream).unwrap();
                f();
                let e1 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e1.record(&stream).unwrap();
                stream.synchronize().unwrap();
                times.push(e0.elapsed_ms(&e1).unwrap());
            }
            times.sort_by(f32::total_cmp);
            (times[0], times[5])
        };
        let run_prep = || {
            let mut b = stream.launch_builder(&prep);
            b.arg(&qn).arg(&kn).arg(&mix).arg(&g).arg(&beta).arg(&w).arg(&u).arg(&aqk).arg(&gc).arg(&qe).arg(&kt).arg(&ntok);
            let cfg = LaunchConfig { grid_dim: (vh as u32, chunks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
            unsafe { b.launch(cfg) }.unwrap();
        };
        let run_scan = || {
            let mut b = stream.launch_builder(&scan);
            b.arg(&w).arg(&qe).arg(&kt).arg(&u).arg(&aqk).arg(&gc).arg(&state).arg(&o).arg(&slot).arg(&ntok);
            let cfg = LaunchConfig { grid_dim: (vh as u32, (d / cols) as u32, 1), block_dim: ((c / 16 * 32) as u32, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(cfg) }.unwrap();
        };
        let p = time(&run_prep);
        let s = time(&run_scan);
        eprintln!("gdn tokens={tokens}: prep min {:.3} ms median {:.3} ms; scan min {:.3} ms median {:.3} ms", p.0, p.1, s.0, s.1);
    }
}
