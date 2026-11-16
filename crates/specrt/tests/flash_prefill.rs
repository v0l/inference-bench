use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::{Module, cuda_target, flash_rows};
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    if exp == 0 {
        return sign * man * 2f32.powi(-24);
    }
    sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = (b >> 13) & 0x3ff;
    if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | man as u16 }
}

fn order_of(page: usize, blocks: usize) -> usize {
    blocks - 1 - page
}

fn run(heads: usize, kv_heads: usize, head_dim: usize, planes: usize, tokens: usize) -> f32 {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.5-122b-a10b-nvfp4.config.json");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(path).unwrap()).unwrap();
    spec.layers.count = spec.layers.pattern.len();
    spec.mtp = None;
    spec.attention.heads = heads;
    spec.attention.kv_heads = kv_heads;
    spec.attention.head_dim = head_dim;
    spec.attention.qk_norm = true;
    let ctx = CudaContext::new(0).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-flash-prefill-{}", std::process::id()))).unwrap();
    let stream = ctx.default_stream();
    let max_ctx = 256;
    let module = Module::generate_kv(&spec, max_ctx, planes, &cuda_target(&ctx).unwrap()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let f = loaded.function("flash_mma").unwrap();
    use cudarc::driver::sys::CUfunction_attribute as A;
    let smem = module.smem["flash_mma"];
    f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).unwrap();
    let qrow = spec.qkv_rows();
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    let qkv: Vec<f32> = (0..tokens * qrow).map(|_| rnd()).collect();
    let blocks = max_ctx / 16;
    let elems = blocks * kv_heads * 16 * head_dim * planes;
    let k = vec![0u16; elems];
    let v = vec![0u16; elems];
    let mut k = k;
    let mut v = v;
    let (mut kf, mut vf) = (vec![0f32; tokens * kv_heads * head_dim], vec![0f32; tokens * kv_heads * head_dim]);
    for t in 0..tokens {
        for h in 0..kv_heads {
            for d in 0..head_dim {
                let (kx, vx) = (rnd(), rnd());
                let base = order_of(t / 16, blocks) * kv_heads * 16 * head_dim * planes;
                let (kh, vh) = (half_bits(kx), half_bits(vx));
                let idx = base + (h * 16 + t % 16) * head_dim * planes + d;
                k[idx] = kh;
                v[idx] = vh;
                let (mut kv, mut vv) = (f16_value(kh), f16_value(vh));
                if planes == 2 {
                    let (kl, vl) = (half_bits(kx - kv), half_bits(vx - vv));
                    k[idx + head_dim] = kl;
                    v[idx + head_dim] = vl;
                    kv += f16_value(kl);
                    vv += f16_value(vl);
                }
                kf[(t * kv_heads + h) * head_dim + d] = kv;
                vf[(t * kv_heads + h) * head_dim + d] = vv;
            }
        }
    }
    let dq = stream.clone_htod(&qkv).unwrap();
    let dk = stream.clone_htod(&k).unwrap();
    let dv = stream.clone_htod(&v).unwrap();
    let order: Vec<i32> = (0..blocks as i32).rev().collect();
    let table = stream.clone_htod(&order).unwrap();
    let pos = stream.clone_htod(&[0i32]).unwrap();
    let hi = stream.alloc_zeros::<u16>(tokens * spec.q_dim()).unwrap();
    let lo = stream.alloc_zeros::<u16>(tokens * spec.q_dim()).unwrap();
    let nw: Vec<f32> = (0..head_dim).map(|d| 0.5 + (d % 7) as f32 * 0.25).collect();
    let norm = stream.clone_htod(&nw).unwrap();
    let mm_meta = stream.alloc_zeros::<i32>(4).unwrap();
    let mm_table = stream.alloc_zeros::<i32>(1).unwrap();
    let (layer, slot, ntok, window, stride) = (0i32, 0i32, tokens as i32, 0i32, max_ctx as i32);
    let rows = flash_rows(head_dim);
    let mut b = stream.launch_builder(&f);
    b.arg(&dq).arg(&norm).arg(&dk).arg(&dv).arg(&table).arg(&pos).arg(&layer).arg(&hi).arg(&lo).arg(&ntok).arg(&mm_meta).arg(&mm_table).arg(&stride).arg(&slot).arg(&window);
    let config = LaunchConfig { grid_dim: (tokens.div_ceil(rows) as u32, heads as u32, 1), block_dim: (rows as u32 * 2, 1, 1), shared_mem_bytes: smem };
    unsafe { b.launch(config) }.unwrap();
    stream.synchronize().unwrap();
    let got_hi = stream.clone_dtoh(&hi).unwrap();
    let got_lo = stream.clone_dtoh(&lo).unwrap();
    let group = heads / kv_heads;
    let gate_off = spec.q_dim() + 2 * spec.kv_dim();
    let mut worst = 0f32;
    let mut scale = 0f32;
    for t in 0..tokens {
        for head in 0..heads {
            let kvh = head / group;
            let mut q = qkv[t * qrow + head * head_dim..t * qrow + (head + 1) * head_dim].to_vec();
            let ss: f64 = q.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / head_dim as f64;
            let inv = 1.0 / (ss + spec.norm.eps).sqrt();
            for d in 0..head_dim {
                q[d] = (q[d] as f64 * inv * nw[d] as f64) as f32;
            }
            let rot = spec.rotary_dim();
            for d in 0..rot / 2 {
                let angle = t as f64 * (spec.rope.theta as f64).powf(-(2.0 * d as f64) / rot as f64);
                let (sn, c) = angle.sin_cos();
                let (x, y) = (q[d] as f64, q[d + rot / 2] as f64);
                q[d] = (x * c - y * sn) as f32;
                q[d + rot / 2] = (y * c + x * sn) as f32;
            }
            let scores: Vec<f64> = (0..=t).map(|s| q.iter().zip(&kf[(s * kv_heads + kvh) * head_dim..]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (head_dim as f64).sqrt()).collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = w.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = w.iter().enumerate().map(|(s, wt)| wt * vf[(s * kv_heads + kvh) * head_dim + d] as f64).sum();
                let gate = 1.0 / (1.0 + (-(qkv[t * qrow + gate_off + head * head_dim + d] as f64)).exp());
                let want = (acc / l * gate) as f32;
                let at = t * spec.q_dim() + head * head_dim + d;
                let got = bf16_value(got_hi[at]) + bf16_value(got_lo[at]);
                worst = worst.max((got - want).abs());
                scale = scale.max(want.abs());
            }
        }
    }
    eprintln!("heads {heads}/{kv_heads} dim {head_dim} planes {planes} tokens {tokens}: max err {worst:.2e} of {scale:.3}");
    worst / scale
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn flash_mma_matches_a_cpu_reference_at_gqa_32_over_2() {
    for planes in [1, 2] {
        for tokens in [1, 16, 17, 33, 96] {
            let e = run(32, 2, 256, planes, tokens);
            assert!(e < 5e-3, "planes {planes} tokens {tokens}: {e}");
        }
    }
}

#[test]
#[ignore = "needs an sm_120 GPU"]
fn flash_mma_matches_a_cpu_reference_at_gqa_6_over_2() {
    for tokens in [16, 17, 96] {
        let e = run(6, 2, 256, 2, tokens);
        assert!(e < 5e-3, "tokens {tokens}: {e}");
    }
}
