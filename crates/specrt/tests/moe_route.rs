use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::family::Registry as Families;
use specrt::jit::Jit;
use specrt::codegen::cuda_target;
use specrt::spec::Spec;
use std::path::Path;

fn spec(hash: bool) -> Spec {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-glm4-moe");
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut s = Spec::from_hf_config_serving_in(Families::embedded(), &json).unwrap();
    if hash {
        let m = s.moe.as_mut().unwrap();
        m.hash_table = Some("tid2eid".into());
        m.hash_layers = 1;
    }
    s
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn float(&mut self) -> f32 {
        (self.next() % 2000) as f32 / 1000.0 - 1.0
    }
}

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

struct Routed {
    idx: Vec<i32>,
    coef: Vec<f32>,
    ids: Vec<i32>,
    table: Vec<f32>,
    logits: Vec<f32>,
}

fn route(hash: bool) -> Routed {
    let spec = spec(hash);
    let (h, e, k, tokens) = (spec.dims.hidden, 8usize, 2usize, 5usize);
    let ctx = CudaContext::new(0).unwrap();
    let target = cuda_target(&ctx).unwrap();
    let module = Module::generate_kv(&spec, 256, 1, &target).unwrap();
    let jit = Jit::new(ctx.clone(), specrt::jit::default_cache_dir()).unwrap();
    let loaded = jit.load(&module).unwrap();
    let stream = ctx.default_stream();
    let mut rng = Lcg(809);
    let x: Vec<f32> = (0..tokens * h).map(|_| rng.float()).collect();
    let ln = vec![1.0f32; h];
    let router: Vec<u16> = (0..e * h).map(|_| bf16(rng.float() * 0.2)).collect();
    let bias: Vec<f32> = (0..e).map(|_| rng.float() * 0.3).collect();
    let ids: Vec<i32> = vec![3, 7, 11, 3, 200];
    let table: Vec<i32> = (0..512 * k).map(|i| ((i * 5 + i / 2) % e) as i32).collect::<Vec<_>>().chunks(k).flat_map(|c| if c[0] == c[1] { vec![c[0], (c[0] + 1) % e as i32] } else { c.to_vec() }).collect();
    let (xd, lnd, rd) = (stream.clone_htod(&x).unwrap(), stream.clone_htod(&ln).unwrap(), stream.clone_htod(&router).unwrap());
    let (gate, bd, td, idsd) = (stream.clone_htod(&[0f32]).unwrap(), stream.clone_htod(&bias).unwrap(), stream.clone_htod(&table).unwrap(), stream.clone_htod(&ids[..tokens]).unwrap());
    let mut xn: CudaSlice<f32> = stream.alloc_zeros(tokens * h).unwrap();
    let mut logits: CudaSlice<f32> = stream.alloc_zeros(tokens * (e + 1)).unwrap();
    let pairs = tokens * (k + 1);
    let mut idx: CudaSlice<i32> = stream.alloc_zeros(pairs).unwrap();
    let mut coef: CudaSlice<f32> = stream.alloc_zeros(pairs).unwrap();
    let logits_kernel = loaded.function("moe_route_logits").unwrap();
    let mut b = stream.launch_builder(&logits_kernel);
    b.arg(&xd).arg(&lnd).arg(&rd).arg(&gate).arg(&gate).arg(&mut xn).arg(&mut logits);
    unsafe { b.launch(LaunchConfig { grid_dim: ((e + 1).div_ceil(8) as u32, tokens as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    let (first, hashed) = (0i32, i32::from(hash));
    let route_kernel = loaded.function("moe_route").unwrap();
    let mut b = stream.launch_builder(&route_kernel);
    b.arg(&logits).arg(&bd).arg(&td).arg(&idsd).arg(&hashed).arg(&mut idx).arg(&mut coef).arg(&first);
    unsafe { b.launch(LaunchConfig { grid_dim: (tokens as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
    stream.synchronize().unwrap();
    Routed { idx: stream.clone_dtoh(&idx).unwrap(), coef: stream.clone_dtoh(&coef).unwrap(), ids, table: table.iter().map(|v| *v as f32).collect(), logits: stream.clone_dtoh(&logits).unwrap() }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn hash_layers_take_the_experts_from_the_token_table_and_weight_them_by_the_scores() {
    let Routed { idx, coef, ids, table, logits } = route(true);
    let (e, k, scale) = (8usize, 2usize, 1.5f32);
    for (t, id) in ids.iter().enumerate() {
        let want: Vec<i32> = (0..k).map(|j| table[*id as usize * k + j] as i32).collect();
        let got = &idx[t * (k + 1)..t * (k + 1) + k];
        assert_eq!(got, want.as_slice(), "token {t}");
        let scores: Vec<f32> = want.iter().map(|w| sigmoid(logits[t * (e + 1) + *w as usize])).collect();
        let sum: f32 = scores.iter().sum();
        for j in 0..k {
            let c = coef[t * (k + 1) + j];
            assert!((c - scores[j] / sum * scale).abs() < 1e-5, "token {t} slot {j}: {c}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn without_a_hash_table_the_same_kernel_selects_by_score_plus_bias() {
    let Routed { idx, ids, table, logits, .. } = route(false);
    let (e, k) = (8usize, 2usize);
    let differs = ids.iter().enumerate().any(|(t, id)| (0..k).any(|j| idx[t * (k + 1) + j] != table[*id as usize * k + j] as i32));
    assert!(differs, "learned routing should not equal the table for every token");
    for t in 0..ids.len() {
        let top = (0..e).max_by(|a, b| logits[t * (e + 1) + a].total_cmp(&logits[t * (e + 1) + b])).unwrap();
        assert!(idx[t * (k + 1)..t * (k + 1) + k].iter().all(|v| (0..e as i32).contains(v)), "{top}");
    }
}
