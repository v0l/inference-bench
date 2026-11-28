use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

fn data(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len).map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0xffff) as f32 / 65535.0 * scale - scale / 2.0).collect()
}

fn time_ms(ctx: &std::sync::Arc<CudaContext>, stream: &std::sync::Arc<cudarc::driver::CudaStream>, f: &dyn Fn()) -> f32 {
    f();
    f();
    let mut times = Vec::new();
    for _ in 0..10 {
        let e0 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
        e0.record(stream).unwrap();
        f();
        let e1 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
        e1.record(stream).unwrap();
        stream.synchronize().unwrap();
        times.push(e0.elapsed_ms(&e1).unwrap());
    }
    times.sort_by(f32::total_cmp);
    times[5]
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn gdn_recurrent_state_kernels_speed_on_27b_shaped_layers() {
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    let la = spec.linear_attention.clone().unwrap();
    let (vh, d) = (la.value_heads, la.key_head_dim);
    let layers = spec.linear_layers();
    let ctx = CudaContext::new(0).unwrap();
    let wall = specrt::bandwidth::probe(&ctx).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-gdn-state-{}", std::process::id()))).unwrap();
    let stream = ctx.default_stream();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, 4096, 1, &target).unwrap();
    let loaded = jit.load(&module).unwrap();
    let verify = loaded.function("gdn_verify").unwrap();
    let commit = loaded.function("gdn_commit").unwrap();
    let step = loaded.function("gdn_step").unwrap();
    let threads = (d * kir::linear::LinearTuning::for_target(&target).ksplit) as u32;
    let k1 = spec.speculative().map_or(3, |m| m.draft_tokens + 1);
    let state_seq = layers * vh * d * d;
    let unit = |v: Vec<f32>| stream.clone_htod(&v).unwrap();
    let neg_a = unit(data(vh, 6, 1.0).iter().map(|v| -0.5 - v.abs()).collect());
    let dt_bias = unit(data(vh, 7, 0.2));
    let norm_w = unit(data(d, 8, 0.4).iter().map(|v| v + 1.0).collect());
    eprintln!("probe read {:.0} GB/s copy {:.0} GB/s, {layers} layers, k+1 = {k1}", wall.read_gbs, wall.copy_gbs);
    let seqs: Vec<usize> = std::env::var("SPECRT_GDN_SEQS").unwrap_or("1,8,16,32".into()).split(',').map(|v| v.parse().unwrap()).collect();
    for nseq in seqs {
        let rows = nseq * k1;
        let state = unit(data(nseq * state_seq, 9, 0.02));
        let mix = unit(data(rows * la.conv_dim(), 3, 1.0));
        let lin = unit(data(rows * la.in_proj_rows(), 10, 1.0));
        let ba = unit(data(rows * la.ba_rows(), 11, 1.0));
        let out = stream.alloc_zeros::<f32>(rows * la.value_dim()).unwrap();
        let klog = stream.alloc_zeros::<f32>(layers * rows * la.value_dim()).unwrap();
        let dlog = stream.alloc_zeros::<f32>(layers * rows * la.value_dim()).unwrap();
        let glog = stream.alloc_zeros::<f32>(layers * rows * vh).unwrap();
        let vslots = stream.clone_htod(&(0..rows as i32).map(|r| r / k1 as i32).collect::<Vec<i32>>()).unwrap();
        let sslots = stream.clone_htod(&(0..nseq as i32).collect::<Vec<i32>>()).unwrap();
        let accepted = stream.clone_htod(&(0..nseq as i32).map(|s| [1, 2, 1, 2, 0][s as usize % 5]).collect::<Vec<i32>>()).unwrap();
        let (seq_slots, k1_i, log_rows) = (layers as i32, k1 as i32, rows as i32);
        let run_verify = || {
            for l in 0..layers {
                let slot = l as i32;
                let (kl, dl, gl) = (klog.slice(l * rows * la.value_dim()..), dlog.slice(l * rows * la.value_dim()..), glog.slice(l * rows * vh..));
                let mut b = stream.launch_builder(&verify);
                b.arg(&mix).arg(&lin).arg(&ba).arg(&neg_a).arg(&dt_bias).arg(&norm_w).arg(&state).arg(&out).arg(&kl).arg(&dl).arg(&gl).arg(&slot).arg(&seq_slots).arg(&vslots).arg(&k1_i);
                unsafe { b.launch(LaunchConfig { grid_dim: (vh as u32, nseq as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
            }
        };
        let run_commit = || {
            let mut b = stream.launch_builder(&commit);
            b.arg(&state).arg(&klog).arg(&dlog).arg(&glog).arg(&seq_slots).arg(&vslots).arg(&k1_i).arg(&accepted).arg(&log_rows);
            unsafe { b.launch(LaunchConfig { grid_dim: (vh as u32, nseq as u32, layers as u32), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
        };
        let run_step = || {
            for l in 0..layers {
                let slot = l as i32;
                let mut b = stream.launch_builder(&step);
                b.arg(&mix).arg(&lin).arg(&ba).arg(&neg_a).arg(&dt_bias).arg(&norm_w).arg(&state).arg(&out).arg(&slot).arg(&seq_slots).arg(&sslots);
                unsafe { b.launch(LaunchConfig { grid_dim: (vh as u32, nseq as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
            }
        };
        let bytes = (nseq * state_seq * 4) as f64;
        let v = time_ms(&ctx, &stream, &run_verify);
        let c = time_ms(&ctx, &stream, &run_commit);
        let s = time_ms(&ctx, &stream, &run_step);
        let gbs = |b: f64, ms: f32| b / ms as f64 / 1e6;
        eprintln!(
            "seqs={nseq}: verify {v:.3} ms ({:.0} GB/s read, {:.0}% of probe); commit {c:.3} ms ({:.0} GB/s, {:.0}% of copy); verify+commit {:.3} ms; step {s:.3} ms ({:.0} GB/s, {:.0}% of copy)",
            gbs(bytes, v),
            100.0 * gbs(bytes, v) / wall.read_gbs,
            gbs(2.0 * bytes, c),
            100.0 * gbs(2.0 * bytes, c) / wall.copy_gbs,
            v + c,
            gbs(2.0 * bytes, s),
            100.0 * gbs(2.0 * bytes, s) / wall.copy_gbs,
        );
    }
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
    let tuning = kir::linear::LinearTuning::for_target(&specrt::codegen::cuda_target(&ctx).unwrap());
    let c = tuning.chunk;
    let cols = tuning.cols;
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
