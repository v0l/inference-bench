use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

fn list(var: &str, default: &str) -> Vec<usize> {
    std::env::var(var).unwrap_or_else(|_| default.into()).split(',').map(|v| v.parse().unwrap()).collect()
}

fn pattern(len: usize, seed: u32) -> Vec<u16> {
    (0..len).map(|i| 0x3c00 + (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 13) & 0x3ff) as u16).collect()
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn flash_mma_prefill_attention_speed() {
    let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b.toml")).unwrap();
    let a = &spec.attention;
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-flash-speed-{}", std::process::id()))).unwrap();
    let stream = ctx.default_stream();
    let tokens = list("SPECRT_FLASH_TOKENS", "2048")[0];
    let max_ctx = list("SPECRT_FLASH_CTX", "32768")[0];
    let layers = spec.kv_layers();
    let blocks = max_ctx / 16;
    let module = Module::generate_kv(&spec, max_ctx, 1).unwrap();
    let base_text = format!("constexpr int FM_BK = {};", specrt::codegen::flash_keys(a.head_dim, 1));
    assert!(module.source.contains(&base_text));
    let qkv = stream.clone_htod(&vec![0.05f32; tokens * spec.qkv_rows()]).unwrap();
    let q_norm = stream.clone_htod(&vec![1.0f32; a.head_dim]).unwrap();
    let elems = blocks * layers * a.kv_heads * 16 * a.head_dim;
    let k = stream.clone_htod(&pattern(elems, 1)).unwrap();
    let v = stream.clone_htod(&pattern(elems, 2)).unwrap();
    let table = stream.clone_htod(&(0..blocks as i32).collect::<Vec<_>>()).unwrap();
    let hi = stream.alloc_zeros::<u16>(tokens * spec.q_dim()).unwrap();
    let lo = stream.alloc_zeros::<u16>(tokens * spec.q_dim()).unwrap();
    let mm_meta = stream.alloc_zeros::<i32>(4).unwrap();
    let mm_table = stream.alloc_zeros::<i32>(1).unwrap();
    let layer = 0i32;
    let slot = 0i32;
    let ntok = tokens as i32;
    for base in list("SPECRT_FLASH_BASES", "0,16384,30720") {
        let pos = stream.clone_htod(&[base as i32]).unwrap();
        for bk in list("SPECRT_FLASH_BK", &specrt::codegen::flash_keys(a.head_dim, 1).to_string()) {
            for stages in list("SPECRT_FLASH_STAGES", &specrt::codegen::flash_stages(a.head_dim, 1).to_string()) {
                for warps in list("SPECRT_FLASH_WARPS", &(specrt::codegen::flash_rows(a.head_dim) / 16).to_string()) {
                    let mut m = module.clone();
                    m.source = m
                        .source
                        .replace(&base_text, &format!("constexpr int FM_BK = {bk};"))
                        .replace(&format!("constexpr int FM_STAGES = {};", specrt::codegen::flash_stages(a.head_dim, 1)), &format!("constexpr int FM_STAGES = {stages};"))
                        .replace(&format!("constexpr int FM_WARPS = {};", specrt::codegen::flash_rows(a.head_dim) / 16), &format!("constexpr int FM_WARPS = {warps};"));
                    if std::env::var("SPECRT_FLASH_NOLOAD").is_ok() {
                        let patched = m.source.replace("cp_async16_zfill(dst, src + c * 8, tk < key_end);", "(void)dst; (void)src;");
                        assert_ne!(patched, m.source);
                        m.source = patched;
                    }
                    let smem = (stages * 2 * bk * a.head_dim * 2).max(warps * 16 * a.head_dim * 2) as u32;
                    let loaded = jit.load(&m).unwrap();
                    let f = loaded.function("flash_mma").unwrap();
                    use cudarc::driver::sys::CUfunction_attribute as A;
                    f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
                    let launch = || {
                        let mut b = stream.launch_builder(&f);
                        b.arg(&qkv).arg(&q_norm).arg(&k).arg(&v).arg(&table).arg(&pos).arg(&layer).arg(&hi).arg(&lo).arg(&ntok).arg(&mm_meta).arg(&mm_table).arg(&slot);
                        let config = LaunchConfig {
                            grid_dim: (tokens.div_ceil(warps * 16) as u32, a.heads as u32, 1),
                            block_dim: (warps as u32 * 32, 1, 1),
                            shared_mem_bytes: smem,
                        };
                        unsafe { b.launch(config) }.unwrap();
                    };
                    launch();
                    launch();
                    let mut times = Vec::new();
                    for _ in 0..10 {
                        let e0 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                        e0.record(&stream).unwrap();
                        launch();
                        let e1 = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                        e1.record(&stream).unwrap();
                        stream.synchronize().unwrap();
                        times.push(e0.elapsed_ms(&e1).unwrap());
                    }
                    times.sort_by(f32::total_cmp);
                    let keys = base as f64 + tokens as f64 / 2.0;
                    let flop = 4.0 * a.heads as f64 * tokens as f64 * keys * a.head_dim as f64;
                    eprintln!(
                        "flash_mma base={base} tokens={tokens} bk={bk} stages={stages} warps={warps}: min {:.3} ms median {:.3} ms, {:.0} TFLOP/s at min",
                        times[0], times[5], flop / times[0] as f64 / 1e9
                    );
                }
            }
        }
    }
}
