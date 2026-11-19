use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use kir::desc::{Description, Shape};
use specrt::codegen::Module;
use specrt::jit::Jit;
use std::collections::BTreeMap;

const HEADS: usize = 4;
const DIM: usize = 16;
const TOPK: usize = 8;
const KEYS: usize = 96;

fn description() -> Description {
    Description::parse(include_str!("../specs/dsa_index_topk.toml")).unwrap()
}

fn shape() -> Shape {
    Shape::new()
        .num("index_heads", HEADS as f64)
        .num("index_head_dim", DIM as f64)
        .num("index_topk", TOPK as f64)
        .num("max_context", KEYS as f64)
        .num("index_q_row", (HEADS * DIM) as f64)
        .num("index_w_row", HEADS as f64)
}

struct Lcg(u64);

impl Lcg {
    fn float(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 2000) as f32 / 1000.0 - 1.0
    }
}

fn reference(q: &[f32], w: &[f32], keys: &[f32], n: usize) -> Vec<i32> {
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|t| (0..HEADS).map(|h| w[h] * (0..DIM).map(|j| q[h * DIM + j] * keys[t * DIM + j]).sum::<f32>().max(0.0)).sum::<f32>()).zip(0..n).collect();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<i32> = scores.iter().take(TOPK).map(|s| s.1 as i32).collect();
    out.resize(TOPK, -1);
    out
}

#[test]
fn the_index_description_builds_and_emits_cuda_without_a_gpu() {
    let k = description().build(&kir::Target::sm_120(), &shape()).unwrap();
    let src = kir::cuda::emit_module(&[k]).unwrap();
    assert!(src.contains("dsa_index_topk(") && src.contains("constexpr int TOPK = 8;") && src.contains("fmaxf(d, 0.0f)"));
    assert!(description().build(&kir::Target::wgpu_baseline(), &shape()).is_err());
    let too_many = Shape::new().num("index_heads", 4.0).num("index_head_dim", 16.0).num("index_topk", 200.0).num("max_context", 96.0).num("index_q_row", 64.0).num("index_w_row", 4.0);
    assert!(description().build(&kir::Target::sm_120(), &too_many).unwrap_err().contains("more keys"));
}

#[test]
#[ignore = "needs a CUDA GPU with NVRTC"]
fn the_indexer_selects_the_reference_top_k_per_row_into_an_index_buffer() {
    let ctx = CudaContext::new(0).unwrap();
    let target = specrt::codegen::cuda_target(&ctx).unwrap();
    let source = kir::cuda::emit_module(&[description().build(&target, &shape()).unwrap()]).unwrap();
    let jit = Jit::new(ctx.clone(), std::env::temp_dir().join(format!("specrt-index-{}", std::process::id()))).unwrap();
    let loaded = jit.load(&Module { source, kernels: vec![], smem: BTreeMap::new(), arch_features: false, kir_hash: String::new() }).unwrap();
    let stream = ctx.default_stream();
    let mut rng = Lcg(809);
    let rows = 3usize;
    let positions = [40i32, 5, 95];
    let q: Vec<f32> = (0..rows * HEADS * DIM).map(|_| rng.float()).collect();
    let w: Vec<f32> = (0..rows * HEADS).map(|_| rng.float().abs()).collect();
    let keys: Vec<f32> = (0..KEYS * DIM).map(|_| rng.float()).collect();
    let (qd, wd, kd, pd) = (stream.clone_htod(&q).unwrap(), stream.clone_htod(&w).unwrap(), stream.clone_htod(&keys).unwrap(), stream.clone_htod(&positions).unwrap());
    let mut idx: CudaSlice<i32> = stream.alloc_zeros(rows * TOPK).unwrap();
    let (pos_step, row_step) = (1i32, 0i32);
    let f = loaded.function("dsa_index_topk").unwrap();
    let mut b = stream.launch_builder(&f);
    b.arg(&qd).arg(&wd).arg(&kd).arg(&mut idx).arg(&pd).arg(&pos_step).arg(&row_step);
    unsafe { b.launch(LaunchConfig { grid_dim: (1, rows as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    stream.synchronize().unwrap();
    let got = stream.clone_dtoh(&idx).unwrap();
    for r in 0..rows {
        let n = positions[r] as usize + 1;
        let want = reference(&q[r * HEADS * DIM..(r + 1) * HEADS * DIM], &w[r * HEADS..(r + 1) * HEADS], &keys, n);
        let mut a = got[r * TOPK..(r + 1) * TOPK].to_vec();
        let mut e = want.clone();
        a.sort();
        e.sort();
        assert_eq!(a, e, "row {r}");
        assert_eq!(got[r * TOPK..(r + 1) * TOPK], want[..], "row {r} keeps score order");
    }
    assert!(got[TOPK..2 * TOPK].iter().filter(|v| **v == -1).count() == 2, "a row with 6 live keys leaves 2 slots empty");
}
