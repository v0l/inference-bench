use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::family::Registry as Families;
use specrt::jit::Jit;
use specrt::spec::Spec;
use std::path::Path;

const HEADS: usize = 4;
const DIM: usize = 64;
const TOPK: usize = 8;
const KEYS: usize = 96;
const PAGE: usize = 16;

fn spec() -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-glm-dsa");
    Spec::from_hf_config_serving_in(Families::embedded(), &std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

struct Lcg(u64);

impl Lcg {
    fn float(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 2000) as f32 / 1000.0 - 1.0
    }
}

impl Lcg {
    fn half_bits(&mut self) -> u16 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (self.0 >> 33) as u16;
        ((r & 1) << 15) | (0x3000 + (r >> 1) % 0xc00)
    }
}

fn reference(q: &[f32], w: &[f32], keys: &[f32], n: usize, scale: f32, w_scale: f32) -> Vec<i32> {
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|t| (0..HEADS).map(|h| w[h] * w_scale * ((0..DIM).map(|j| q[h * DIM + j] * keys[t * DIM + j]).sum::<f32>() * scale).max(0.0)).sum::<f32>())
        .zip(0..n)
        .collect();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<i32> = scores.iter().take(TOPK).map(|s| s.1 as i32).collect();
    out.resize(TOPK, -1);
    out
}

#[test]
fn the_index_description_builds_and_emits_cuda_without_a_gpu() {
    let module = Module::generate_kv(&spec(), 256, 1, &kir::Target::sm_120()).unwrap();
    assert!(module.kernels.contains(&"dsa_index_topk"));
    assert!(module.source.contains("dsa_index_topk(") && module.source.contains("constexpr int TOPK = 8;") && module.source.contains("fmaxf(d * scale, 0.0f)"));
    assert!(module.source.contains("attn_float(kr[j])"), "the keys are read from the paged bf16 state");
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn the_indexer_selects_the_reference_top_k_per_row_from_the_paged_key_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let spec = spec();
    let module = Module::generate_kv(&spec, 256, 1, &target).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-index-{}", std::process::id()))).unwrap();
    let loaded = jit.load(&module).unwrap();
    let stream = ctx.default_stream();
    let mut rng = Lcg(809);
    let rows = 3usize;
    let positions = [40i32, 5, 95];
    let (scale, w_scale) = (0.125f32, 0.5f32);
    let q_row = HEADS * DIM;
    let (w_row, w_offset) = (260usize, 256usize);
    let q: Vec<f32> = (0..rows * q_row).map(|_| rng.float()).collect();
    let w: Vec<f32> = (0..rows * w_row).map(|_| rng.float().abs()).collect();
    let weights = |r: usize| w[r * w_row + w_offset..r * w_row + w_offset + HEADS].to_vec();
    let key_bits: Vec<u16> = (0..KEYS * DIM).map(|_| rng.half_bits()).collect();
    let keys: Vec<f32> = key_bits.iter().map(|b| specrt::weights::f16_to_f32(*b)).collect();
    let pages = 256 / PAGE;
    let layers = 2usize;
    let layer = 1usize;
    let table: Vec<i32> = (0..pages as i32).rev().collect();
    let mut cache = vec![0u16; pages * layers * PAGE * DIM];
    for t in 0..KEYS {
        let block = table[t / PAGE] as usize;
        let at = ((block * layers + layer) * PAGE + t % PAGE) * DIM;
        for j in 0..DIM {
            cache[at + j] = key_bits[t * DIM + j];
        }
    }
    let (qd, wd, kd, pd) = (stream.clone_htod(&q).unwrap(), stream.clone_htod(&w).unwrap(), stream.clone_htod(&cache).unwrap(), stream.clone_htod(&positions).unwrap());
    let (td, sd) = (stream.clone_htod(&table).unwrap(), stream.clone_htod(&[0i32; 3]).unwrap());
    let mut idx: CudaSlice<i32> = stream.alloc_zeros(rows * TOPK).unwrap();
    let (layer_arg, pos_step, row_step, pages_arg) = (layer as i32, 1i32, 0i32, pages as i32);
    let f = loaded.function("dsa_index_topk").unwrap();
    let mut b = stream.launch_builder(&f);
    b.arg(&qd).arg(&wd).arg(&kd).arg(&mut idx).arg(&pd).arg(&layer_arg).arg(&pos_step).arg(&row_step).arg(&td).arg(&sd).arg(&pages_arg).arg(&scale).arg(&w_scale);
    unsafe { b.launch(LaunchConfig { grid_dim: (1, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&idx).unwrap();
    for r in 0..rows {
        let n = positions[r] as usize + 1;
        let want = reference(&q[r * q_row..(r + 1) * q_row], &weights(r), &keys, n, scale, w_scale);
        let mut a = got[r * TOPK..(r + 1) * TOPK].to_vec();
        let mut e = want.clone();
        a.sort();
        e.sort();
        assert_eq!(a, e, "row {r}");
        assert_eq!(got[r * TOPK..(r + 1) * TOPK], want[..], "row {r} keeps score order");
    }
    assert!(got[TOPK..2 * TOPK].iter().filter(|v| **v == -1).count() == 2, "a row with 6 live keys leaves 2 slots empty");
}
