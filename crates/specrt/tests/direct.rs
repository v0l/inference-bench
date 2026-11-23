use specrt::direct::{DirectLink, DirectRegion};
use specrt::rdma::{LinkOptions, Transport, Verbs};
use specrt::runtime::{DecodeRow, Layout, Runtime, Sampling, StageRange};
use specrt::spec::Spec;
use specrt::stage::{bytes_f32, f32_bytes};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SLOTS: usize = 1;
const CTX: usize = 512;

fn fixture_dir(dir: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir)
}

fn spec_for(dir: &str) -> Spec {
    let json = std::fs::read_to_string(fixture_dir(dir).join("config.json")).unwrap();
    let mut spec = Spec::from_hf_config(&json).unwrap();
    spec.name = dir.into();
    spec.repo = fixture_dir(dir).display().to_string();
    spec.runtime.split_activations = Some(true);
    spec
}

fn gpu_allowed() -> bool {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    let first = visible.split(',').next().unwrap_or("").trim().to_string();
    if first.is_empty() || first == "0" {
        eprintln!("SKIPPED: CUDA_VISIBLE_DEVICES={visible:?} would use GPU0. Run under `~/.agents/bin/gpu-run 1 --`.");
        return false;
    }
    true
}

fn rdma_ip() -> Option<String> {
    if let Ok(ip) = std::env::var("SPECRT_RDMA_TEST_IP") {
        return Some(ip);
    }
    let text = std::process::Command::new("ip").args(["-4", "-o", "addr"]).output().ok()?;
    String::from_utf8_lossy(&text.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().nth(3).map(|a| a.split('/').next().unwrap().to_string()))
        .find(|a| a.starts_with("10.100.50."))
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 1024).collect()
}

fn load(spec: &Spec, start: usize, end: usize) -> (Runtime, specrt::runtime::Stepper) {
    let layout = Layout { kv_planes: 1, ..Layout::contiguous(CTX, SLOTS) };
    let mut rt = Runtime::load_stage(spec, layout, StageRange { device: 0, start, end }).unwrap();
    let pages = CTX.div_ceil(specrt::codegen::PAGE) as u32;
    rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
    let st = rt.stepper().unwrap();
    (rt, st)
}

trait Hop {
    fn carry(&mut self, from: &Runtime, to: &mut Runtime, rows: usize);
}

struct HostHop;

impl Hop for HostHop {
    fn carry(&mut self, from: &Runtime, to: &mut Runtime, rows: usize) {
        let bytes = f32_bytes(&from.export_hidden(rows).unwrap());
        to.import_hidden(&bytes_f32(&bytes).unwrap()).unwrap();
    }
}

struct DirectHop {
    a: DirectLink,
    b: DirectLink,
}

impl Hop for DirectHop {
    fn carry(&mut self, from: &Runtime, to: &mut Runtime, rows: usize) {
        let bytes = rows * from.hidden_row_bytes();
        from.export_hidden_device(rows, self.a.outbox.ptr()).unwrap();
        self.a.push(bytes, Duration::from_secs(5)).unwrap();
        self.b.wait(bytes, Duration::from_secs(5)).unwrap();
        to.import_hidden_device(rows, self.b.landing.ptr()).unwrap();
    }
}

fn pair(ctx: &Arc<cudarc::driver::CudaContext>, bytes: usize, ip: &str) -> Result<DirectHop, String> {
    let link = LinkOptions { transport: Transport::Rdma, library: None, gpu_direct: true };
    let ip = ip.parse().unwrap();
    let region = || DirectRegion::new(ctx, bytes).map(Arc::new).map_err(|e| format!("{e:#}"));
    let (out_a, land_a, out_b, land_b) = (region()?, region()?, region()?, region()?);
    let (pa, ia) = DirectLink::offer(&out_a, &land_a, &link, ip).map_err(|e| format!("{e:#}"))?;
    let (pb, ib) = DirectLink::offer(&out_b, &land_b, &link, ip).map_err(|e| format!("{e:#}"))?;
    let a = pa.connect(ib, land_b.len()).map_err(|e| format!("{e:#}"))?;
    let b = pb.connect(ia, land_a.len()).map_err(|e| format!("{e:#}"))?;
    Ok(DirectHop { a, b })
}

fn run_pipeline(spec: &Spec, hop: &mut dyn Hop, p: &[u32], new_tokens: usize) -> (Vec<u32>, Duration, Vec<Duration>) {
    let cut = 1;
    let (mut r0, mut s0) = load(spec, 0, cut);
    let (mut r1, mut s1) = load(spec, cut, spec.layers.count);
    let started = Instant::now();
    r0.prefill_slot(&mut s0, 0, p, 0, None).unwrap();
    hop.carry(&r0, &mut r1, p.len());
    let mut next = r1.prefill_slot(&mut s1, 0, &vec![0; p.len()], 0, Some((Sampling::GREEDY, 0))).unwrap().unwrap();
    let prefill = started.elapsed();
    let mut tokens = vec![next];
    let mut hops = Vec::new();
    let mut at = p.len();
    while tokens.len() < new_tokens {
        let row = DecodeRow { slot: 0, token: next, pos: at, sampling: Sampling::GREEDY, step: 0 };
        r0.decode(&mut s0, &[row]).unwrap();
        let t = Instant::now();
        hop.carry(&r0, &mut r1, 1);
        hops.push(t.elapsed());
        let row = DecodeRow { slot: 0, token: 0, pos: at, sampling: Sampling::GREEDY, step: 0 };
        next = r1.decode(&mut s1, &[row]).unwrap()[0];
        tokens.push(next);
        at += 1;
    }
    (tokens, prefill, hops)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[test]
#[ignore = "needs a CUDA GPU and a RoCE NIC, run under gpu-run 1"]
fn two_stage_pipeline_with_gpu_direct_matches_host_hops_and_one_stage() {
    if !gpu_allowed() {
        return;
    }
    if Verbs::load(specrt::rdma::DEFAULT_LIBRARY).is_err() {
        eprintln!("SKIPPED: no libibverbs");
        return;
    }
    let Some(ip) = rdma_ip() else {
        eprintln!("SKIPPED: no RoCE device with a 10.100.50.x address (set SPECRT_RDMA_TEST_IP)");
        return;
    };
    let spec = spec_for("tiny-qwen3.5");
    let p = prompt(100);
    let want = {
        let layout = Layout { kv_planes: 1, ..Layout::contiguous(CTX, SLOTS) };
        let mut rt = Runtime::load_layout(&spec, layout).unwrap();
        let pages = CTX.div_ceil(specrt::codegen::PAGE) as u32;
        rt.set_block_table(0, &(0..pages).collect::<Vec<_>>()).unwrap();
        rt.generate(&p, 32, Sampling::GREEDY).unwrap().tokens
    };
    let (host_tokens, host_prefill, host_hops) = run_pipeline(&spec, &mut HostHop, &p, 32);
    assert_eq!(host_tokens, want, "host hops");
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let mut direct = match pair(&ctx, 4 << 20, &ip) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU-DIRECT UNAVAILABLE, falling back to host hops: {e}");
            return;
        }
    };
    let (direct_tokens, direct_prefill, direct_hops) = run_pipeline(&spec, &mut direct, &p, 32);
    assert_eq!(direct_tokens, want, "gpu-direct hops");
    eprintln!("HOP decode ({} B) median: host {:?}, gpu-direct {:?}", spec.dims.hidden * 4, median(host_hops), median(direct_hops));
    eprintln!("HOP prefill pipeline (100 rows, {} B): host total {host_prefill:?}, gpu-direct total {direct_prefill:?}", 100 * spec.dims.hidden * 4);
}

#[test]
#[ignore = "needs a CUDA GPU and a RoCE NIC, run under gpu-run 1"]
fn raw_hop_timing_20kb_and_4mb() {
    if !gpu_allowed() {
        return;
    }
    let Some(ip) = rdma_ip() else {
        eprintln!("SKIPPED: no RoCE device");
        return;
    };
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let mut direct = match pair(&ctx, 4 << 20, &ip) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU-DIRECT UNAVAILABLE: {e}");
            return;
        }
    };
    let host_src = DirectRegion::new(&ctx, 4 << 20).unwrap();
    let host_dst = DirectRegion::new(&ctx, 4 << 20).unwrap();
    for (label, n, iters) in [("20KB", 20 * 1024usize, 500u32), ("4MB", 4 << 20, 50)] {
        let pattern: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        direct.a.outbox.write_host(&pattern).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            direct.a.push(n, Duration::from_secs(5)).unwrap();
            direct.b.wait(n, Duration::from_secs(5)).unwrap();
        }
        let d = t.elapsed() / iters;
        assert_eq!(direct.b.landing.read_host(n).unwrap(), pattern, "{label} arrived intact");
        host_src.write_host(&pattern).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let bytes = host_src.read_host(n).unwrap();
            host_dst.write_host(&bytes).unwrap();
        }
        let h = t.elapsed() / iters;
        eprintln!("HOP {label}: gpu-direct write {d:?}, host copy out and back (no NIC) {h:?}");
    }
}
