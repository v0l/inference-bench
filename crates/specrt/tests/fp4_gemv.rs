use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use specrt::codegen::Module;
use specrt::jit::Jit;
use specrt::spec::Spec;
use specrt::weights::{Checkpoint, Matrix};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn float(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 32) as u8).collect()
    }

    fn scales(&mut self, n: usize) -> Vec<u8> {
        self.bytes(n).into_iter().map(|b| if b & 0x7f == 0x7f { 0x38 } else { b & 0x7f }).collect()
    }
}

struct Case {
    gate_up: Vec<u8>,
    gate_up_scales: Vec<u8>,
    down: Vec<u8>,
    down_scales: Vec<u8>,
}

fn header(inv_global: f32) -> Vec<u8> {
    let mut h = inv_global.to_le_bytes().to_vec();
    h.extend_from_slice(&1.0f32.to_le_bytes());
    h.resize(16, 0);
    h
}

fn random_case(hidden: usize, inter: usize) -> Case {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let scales = |rows: usize, cols: usize, inv: f32, rng: &mut Rng| {
        let mut s = header(inv);
        s.extend(rng.scales(rows * cols / 16));
        s
    };
    Case {
        gate_up: rng.bytes(2 * inter * hidden / 2),
        gate_up_scales: scales(2 * inter, hidden, 1.0 / 2048.0, &mut rng),
        down: rng.bytes(hidden * inter / 2),
        down_scales: scales(hidden, inter, 1.0 / 4096.0, &mut rng),
    }
}

fn fixture_case(spec: &Spec) -> Case {
    let ck = Checkpoint::open(Path::new(&spec.repo)).unwrap();
    let layer = ck.layer(spec, 0).unwrap();
    let (Matrix::Fp4(gate_up), Matrix::Fp4(down)) = (&layer.gate_up, &layer.down) else { panic!("layer 0 is not nvfp4") };
    Case {
        gate_up: gate_up.packed.to_vec(),
        gate_up_scales: gate_up.device_scales(),
        down: down.packed.to_vec(),
        down_scales: down.device_scales(),
    }
}

fn tiny_spec() -> Spec {
    let dir = root().join("tests/fixtures/tiny-qwen3.5-nvfp4");
    let mut spec = Spec::from_hf_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    spec.name = "tiny-qwen3.5-nvfp4".into();
    spec.repo = dir.display().to_string();
    spec
}

fn big_spec() -> Spec {
    Spec::load(&root().join("tests/fixtures/qwen3.8-27b-nvfp4.toml")).unwrap()
}

fn compare(label: &str, spec: &Spec, case: &Case) -> bool {
    let Ok(ctx) = CudaContext::new(0) else { return false };
    let dir = std::env::temp_dir().join(format!("specrt-fp4gemv-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut module = Module::generate(spec, 256).unwrap();
    module.source.push_str(include_str!("fixtures/fp4_gemv_reference.cu"));
    let loaded = Jit::new(ctx.clone(), dir.clone()).unwrap().load(&module).unwrap();
    let stream = ctx.default_stream();
    let (hidden, inter) = (spec.dims.hidden, spec.dims.intermediate);
    let mut rng = Rng(42);
    let x: Vec<f32> = (0..hidden).map(|_| rng.float() * 8.0).collect();
    let nw: Vec<f32> = (0..hidden).map(|_| rng.float() + 1.0).collect();
    let act_in: Vec<f32> = (0..inter).map(|_| rng.float() * 4.0).collect();
    let resid: Vec<f32> = (0..hidden).map(|_| rng.float()).collect();
    let gw = stream.clone_htod(&case.gate_up).unwrap();
    let gs = stream.clone_htod(&case.gate_up_scales).unwrap();
    let dw = stream.clone_htod(&case.down).unwrap();
    let ds = stream.clone_htod(&case.down_scales).unwrap();
    let dx = stream.clone_htod(&x).unwrap();
    let dn = stream.clone_htod(&nw).unwrap();
    let da = stream.clone_htod(&act_in).unwrap();
    let dummy = stream.clone_htod(&[0.0f32]).unwrap();
    let cfg = |rows: usize, threads: u32| LaunchConfig {
        grid_dim: (rows.div_ceil(threads as usize / 32) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut gate_out = Vec::new();
    let mut down_out = Vec::new();
    for (name, threads) in [("gate_up_act_f4", 512u32), ("gate_up_act_f4_ref", 256)] {
        let out = stream.alloc_zeros::<f32>(inter).unwrap();
        let func = loaded.function(name).unwrap();
        let mut b = stream.launch_builder(&func);
        b.arg(&gw).arg(&gs).arg(&dx).arg(&dn).arg(&out);
        unsafe { b.launch(cfg(inter, threads)) }.unwrap();
        gate_out.push(stream.clone_dtoh(&out).unwrap());
    }
    for (name, threads) in [("down_proj_f4", 512u32), ("down_proj_f4_ref", 256)] {
        let out = stream.clone_htod(&resid).unwrap();
        let func = loaded.function(name).unwrap();
        let mut b = stream.launch_builder(&func);
        b.arg(&dw).arg(&ds).arg(&dummy).arg(&da).arg(&out);
        unsafe { b.launch(cfg(hidden, threads)) }.unwrap();
        down_out.push(stream.clone_dtoh(&out).unwrap());
    }
    std::fs::remove_dir_all(&dir).ok();
    let bits = |v: &Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert!(gate_out[0].iter().any(|v| *v != 0.0 && v.is_finite()), "gate/up output is degenerate");
    assert!(bits(&gate_out[0]) == bits(&gate_out[1]), "gate_up_act_f4 differs from the reference kernel");
    assert!(bits(&down_out[0]) == bits(&down_out[1]), "down_proj_f4 differs from the reference kernel");
    true
}

#[test]
fn tuned_fp4_gemvs_equal_the_reference_kernels_bit_for_bit_on_the_tiny_fixture() {
    let spec = tiny_spec();
    if !compare("tiny", &spec, &fixture_case(&spec)) {
        eprintln!("no CUDA device, skipped");
    }
}

#[test]
fn tuned_fp4_gemvs_equal_the_reference_kernels_bit_for_bit_on_a_random_5120_by_17408_matrix() {
    let spec = big_spec();
    assert_eq!((spec.dims.hidden, spec.dims.intermediate), (5120, 17408));
    if !compare("random", &spec, &random_case(5120, 17408)) {
        eprintln!("no CUDA device, skipped");
    }
}
