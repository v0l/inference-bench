use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use specrt::codegen::{FP8_BLOCK, Module, Tile};
use specrt::jit::Jit;
use specrt::spec::Spec;
use specrt::weights::fp8_scales;
use std::path::Path;

fn spec(name: &str) -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{name}.toml"))).unwrap()
}

fn list(var: &str, default: &str) -> Vec<String> {
    std::env::var(var).unwrap_or_else(|_| default.into()).split(',').map(String::from).collect()
}

fn bytes(len: usize, seed: u32) -> Vec<u8> {
    (0..len).map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 13) as u8 & 0x77).collect()
}

fn variants(module: &Module) -> Vec<(String, Module)> {
    let base = "constexpr int RASTER_GROUP = 32;";
    assert!(module.source.contains(base));
    list("SPECRT_RASTER", "32")
        .into_iter()
        .map(|g| {
            let mut m = module.clone();
            m.source = m.source.replace(base, &format!("constexpr int RASTER_GROUP = {g};"));
            (format!("group {g}"), m)
        })
        .collect()
}

fn fp4_variants(spec: &Spec) -> Vec<(String, Spec, Option<String>)> {
    list("SPECRT_F4A_VARIANTS", "cur")
        .into_iter()
        .map(|v| {
            let mut parts = v.split(':');
            let src = parts.next().unwrap();
            let mut s = spec.clone();
            if let Some(tile) = parts.next().filter(|t| !t.is_empty()) {
                let t: Vec<usize> = tile.split('x').map(|x| x.parse().unwrap()).collect();
                s.runtime.fp4_tile = Some([t[0], t[1], t[2], t[3]]);
            }
            if let Some(st) = parts.next() {
                s.runtime.fp4_stages = Some(st.parse().unwrap());
            }
            let template = (src != "cur").then(|| std::fs::read_to_string(src).unwrap());
            (v.clone(), s, template)
        })
        .collect()
}

fn time_fp4(spec_name: &str) {
    let base = spec(spec_name);
    let ctx = CudaContext::new(0).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-gemm-speed-{}", std::process::id()));
    let jit = Jit::new(ctx.clone(), dir).unwrap();
    let stream = ctx.default_stream();
    let built: Vec<_> = fp4_variants(&base)
        .into_iter()
        .map(|(name, s, template)| {
            let mut module = Module::generate(&s, 256).unwrap();
            if let Some(t) = template {
                assert!(module.source.contains(specrt::codegen::FP4_ACT_TEMPLATE));
                module.source = module.source.replace(specrt::codegen::FP4_ACT_TEMPLATE, &t);
            }
            let loaded = jit.load(&module).unwrap();
            (name, s, module, loaded)
        })
        .collect();
    for linear in ["gate_up", "down"] {
        let (n, k, _, _) = specrt::codegen::linear_dims(&base, linear);
        let kernel = format!("{linear}_f4a_big");
        let fs: Vec<(CudaFunction, Tile, u32)> = built
            .iter()
            .map(|(_, s, module, l)| {
                let smem = module.smem[kernel.as_str()];
                let f = l.function(&kernel).unwrap();
                use cudarc::driver::sys::CUfunction_attribute as A;
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
                (f, Tile::fp4_variant("_big", &s.runtime), smem)
            })
            .collect();
        for m in list("SPECRT_GEMM_TOKENS", "1024,4096").iter().map(|t| t.parse::<usize>().unwrap()) {
            let w = stream.clone_htod(&bytes(n * k / 2, 1)).unwrap();
            let xq = stream.clone_htod(&bytes(m * k / 2, 2)).unwrap();
            let out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let planes = stream.alloc_zeros::<u16>(m * n / 2).unwrap();
            let planes2 = stream.alloc_zeros::<u16>(m * n / 2).unwrap();
            let mut s = vec![0u8; 16];
            s[..4].copy_from_slice(&1.0f32.to_le_bytes());
            s[4..8].copy_from_slice(&1.0f32.to_le_bytes());
            s.extend(std::iter::repeat_n(0x38u8, n * k / 16));
            let ws = stream.clone_htod(&s).unwrap();
            let xs = stream.clone_htod(&vec![0x38u8; m * k / 16]).unwrap();
            let tokens_i = m as i32;
            let ev = || {
                let e = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e.record(&stream).unwrap();
                e
            };
            let launch = |(f, tile, smem): &(CudaFunction, Tile, u32)| {
                let mut b = stream.launch_builder(f);
                b.arg(&w).arg(&ws).arg(&xq).arg(&xs).arg(&out).arg(&planes).arg(&planes2).arg(&tokens_i);
                let config = LaunchConfig {
                    grid_dim: (n.div_ceil(tile.bn) as u32, m.div_ceil(tile.bm) as u32, 1),
                    block_dim: (tile.threads(), 1, 1),
                    shared_mem_bytes: *smem,
                };
                unsafe { b.launch(config) }.unwrap();
            };
            let mut first = None;
            for (v, f) in fs.iter().enumerate() {
                if linear != "gate_up" {
                    break;
                }
                launch(f);
                stream.synchronize().unwrap();
                let got: Vec<u16> = stream.clone_dtoh(&planes).unwrap();
                match &first {
                    None => first = Some(got),
                    Some(r) => assert!(r == &got, "variant {} differs from the first", built[v].0),
                }
            }
            let mut times = vec![Vec::new(); fs.len()];
            for _ in 0..list("SPECRT_GEMM_REPS", "20")[0].parse::<usize>().unwrap() {
                for (v, f) in fs.iter().enumerate() {
                    let a = ev();
                    launch(f);
                    let b = ev();
                    stream.synchronize().unwrap();
                    times[v].push(a.elapsed_ms(&b).unwrap());
                }
            }
            for (v, mut t) in times.into_iter().enumerate() {
                t.sort_by(f32::total_cmp);
                let ms = t[t.len() / 2];
                let best = t[0];
                let tflops = 2.0 * (m * n * k) as f64 / ms as f64 / 1e9;
                eprintln!("{kernel} [{}] m={m}: median {ms:.3} ms, min {best:.3} ms, {tflops:.0} TFLOP/s", built[v].0);
            }
        }
    }
}

fn time(spec_name: &str, kernel_of: &dyn Fn(&str) -> (String, Tile), fp4: bool) {
    let spec = spec(spec_name);
    let ctx = CudaContext::new(0).unwrap();
    let dir = std::env::temp_dir().join(format!("specrt-gemm-speed-{}", std::process::id()));
    let jit = Jit::new(ctx.clone(), dir).unwrap();
    let module = Module::generate(&spec, 256).unwrap();
    let loaded: Vec<_> = variants(&module).into_iter().map(|(name, m)| (name, jit.load(&m).unwrap())).collect();
    let stream = ctx.default_stream();
    let per_token = spec.quant == specrt::spec::Quant::Fp8Channel;
    let linears: &[&str] = if fp4 { &["gate_up", "down"] } else { &["gate_up", "down", "lin_in"] };
    for &linear in linears {
        let (n, k, _, _) = specrt::codegen::linear_dims(&spec, linear);
        let (kernel, tile) = kernel_of(linear);
        let smem = module.smem[kernel.as_str()];
        let fs: Vec<CudaFunction> = loaded
            .iter()
            .map(|(_, l)| {
                let f = l.function(&kernel).unwrap();
                use cudarc::driver::sys::CUfunction_attribute as A;
                f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
                f
            })
            .collect();
        for m in list("SPECRT_GEMM_TOKENS", "2048,16384").iter().map(|t| t.parse::<usize>().unwrap()) {
            let w = stream.clone_htod(&bytes(if fp4 { n * k / 2 } else { n * k }, 1)).unwrap();
            let xq = stream.clone_htod(&bytes(if fp4 { m * k / 2 } else { m * k }, 2)).unwrap();
            let out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let planes = stream.alloc_zeros::<u16>(m * n / 2).unwrap();
            let planes2 = stream.alloc_zeros::<u16>(m * n / 2).unwrap();
            let bias = stream.alloc_zeros::<f32>(n).unwrap();
            let tokens_i = m as i32;
            let kb = k / FP8_BLOCK;
            let (ws_f, xs_f, ws_b, xs_b) = if fp4 {
                let mut s = vec![0u8; 16];
                s[..4].copy_from_slice(&1.0f32.to_le_bytes());
                s[4..8].copy_from_slice(&1.0f32.to_le_bytes());
                s.extend(std::iter::repeat_n(0x38u8, n * k / 16));
                (None, None, Some(stream.clone_htod(&s).unwrap()), Some(stream.clone_htod(&vec![0x38u8; m * k / 16]).unwrap()))
            } else {
                let rb = n.div_ceil(FP8_BLOCK);
                let (grid, map): (Vec<f32>, Vec<u32>) = if per_token {
                    (vec![0.01; n * kb], (0..n).map(|r| (r * kb) as u32).collect())
                } else {
                    (vec![0.01; rb * kb], (0..n).map(|r| (r / FP8_BLOCK * kb) as u32).collect())
                };
                (
                    Some(stream.clone_htod(&fp8_scales(&grid, &map)).unwrap()),
                    Some(stream.clone_htod(&vec![0.01f32; m * kb]).unwrap()),
                    None,
                    None,
                )
            };
            let ev = || {
                let e = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e.record(&stream).unwrap();
                e
            };
            let launch = |f: &CudaFunction| {
                let mut b = stream.launch_builder(f);
                b.arg(&w);
                if fp4 {
                    b.arg(ws_b.as_ref().unwrap()).arg(&xq).arg(xs_b.as_ref().unwrap());
                } else {
                    b.arg(ws_f.as_ref().unwrap()).arg(&bias).arg(&xq).arg(xs_f.as_ref().unwrap());
                }
                b.arg(&out).arg(&planes).arg(&planes2).arg(&tokens_i);
                let config = LaunchConfig {
                    grid_dim: (n.div_ceil(tile.bn) as u32, m.div_ceil(tile.bm) as u32, 1),
                    block_dim: (tile.threads(), 1, 1),
                    shared_mem_bytes: smem,
                };
                unsafe { b.launch(config) }.unwrap();
            };
            for f in &fs {
                launch(f);
                launch(f);
            }
            let mut times = vec![Vec::new(); fs.len()];
            for _ in 0..10 {
                for (v, f) in fs.iter().enumerate() {
                    let a = ev();
                    launch(f);
                    let b = ev();
                    stream.synchronize().unwrap();
                    times[v].push(a.elapsed_ms(&b).unwrap());
                }
            }
            for (v, mut t) in times.into_iter().enumerate() {
                t.sort_by(f32::total_cmp);
                let ms = t[t.len() / 2];
                let tflops = 2.0 * (m * n * k) as f64 / ms as f64 / 1e9;
                eprintln!("{spec_name} {kernel} {} m={m} n={n} k={k}: {ms:.3} ms, {tflops:.0} TFLOP/s", loaded[v].0);
            }
        }
    }
}

#[test]
#[ignore = "needs an sm_120 GPU; timing only"]
fn prefill_gemm_speed_fp8_block() {
    time("qwen3.8-27b", &|l| (format!("{l}_f8_huge"), Tile::FP8_HUGE), false);
}

#[test]
#[ignore = "needs an sm_120 GPU; timing only"]
fn prefill_gemm_speed_fp8_channel_and_fp4() {
    let s = spec("qwen3.8-27b-nvfp4");
    time("qwen3.8-27b-nvfp4", &|l| (format!("{l}_f8_huge"), Tile::FP8_HUGE), false);
    time("qwen3.8-27b-nvfp4", &|l| (format!("{l}_f4a_big"), Tile::fp4_variant("_big", &s.runtime)), true);
}

#[test]
#[ignore = "needs an sm_120 GPU; timing only"]
fn prefill_gemm_speed_fp4() {
    time_fp4("qwen3.8-27b-nvfp4-act");
}
