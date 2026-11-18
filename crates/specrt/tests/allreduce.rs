use cudarc::driver::{CudaSlice, CudaStream};
use kir::allreduce::Dtype;
use specrt::allreduce::{Comm, Fabric, PROBE_TIMEOUT, ProbeOutcome, Transport, contexts, peer_writes_land};
use std::sync::Arc;

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
}

fn to_bf16(v: f32) -> u16 {
    let u = v.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn gpu_ordinal() -> usize {
    std::env::var("SPECRT_TP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn cache() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("specrt-allreduce-{}", std::process::id()))
}

struct Group {
    fabric: Fabric,
    streams: Vec<Arc<CudaStream>>,
    comms: Vec<Comm>,
}

fn group(devices: &[usize], bytes: usize, force: Option<Transport>, norm: Option<(u32, f32)>) -> Group {
    let ctxs = contexts(devices).unwrap();
    let fabric = Fabric::new(&ctxs, bytes, force).unwrap();
    let streams: Vec<_> = ctxs.iter().map(|c| c.new_stream().unwrap()).collect();
    let comms = streams.iter().enumerate().map(|(r, s)| fabric.comm(r, s, cache(), norm).unwrap()).collect();
    Group { fabric, streams, comms }
}

fn raw_of(dtype: Dtype, v: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F32 => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
        Dtype::Bf16 => v.iter().flat_map(|f| to_bf16(*f).to_le_bytes()).collect(),
    }
}

fn run(devices: &[usize], force: Option<Transport>, dtype: Dtype, elements: usize, seed: u64) {
    let n = devices.len();
    let g = group(devices, elements * dtype.bytes(), force, None);
    let mut rng = Rng(seed);
    let parts: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..elements).map(|_| if dtype == Dtype::Bf16 { from_bf16(to_bf16(rng.float() * 4.0)) } else { rng.float() * 4.0 }).collect())
        .collect();
    let base: Vec<f32> = (0..elements).map(|_| rng.float()).collect();
    let mut inputs: Vec<CudaSlice<u8>> = Vec::new();
    let mut outputs: Vec<CudaSlice<u8>> = Vec::new();
    for (r, p) in parts.iter().enumerate() {
        inputs.push(g.streams[r].clone_htod(&raw_of(dtype, p)).unwrap());
        outputs.push(g.streams[r].clone_htod(&raw_of(dtype, &base)).unwrap());
    }
    for round in 0..3 {
        let add = round == 1;
        std::thread::scope(|s| {
            for (r, (inp, out)) in inputs.iter().zip(outputs.iter_mut()).enumerate() {
                let (comm, stream) = (&g.comms[r], &g.streams[r]);
                s.spawn(move || {
                    stream.context().bind_to_thread().unwrap();
                    comm.allreduce_raw(stream, dtype, inp, out, elements, add).unwrap();
                    stream.synchronize().unwrap();
                });
            }
        });
        for r in 0..n {
            let raw = g.streams[r].clone_dtoh(&outputs[r]).unwrap();
            for i in 0..elements {
                let mut acc = 0.0f32;
                for p in &parts {
                    acc += p[i];
                }
                let want = match dtype {
                    Dtype::F32 if add => (raw_prev(round, &base, &parts, i) + acc).to_bits(),
                    Dtype::F32 => acc.to_bits(),
                    Dtype::Bf16 if add => to_bf16(from_bf16(to_bf16(raw_prev(round, &base, &parts, i))) + acc) as u32,
                    Dtype::Bf16 => to_bf16(acc) as u32,
                };
                let got = match dtype {
                    Dtype::F32 => u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()),
                    Dtype::Bf16 => u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap()) as u32,
                };
                assert_eq!(got, want, "rank {r} element {i} round {round} of {elements} {dtype:?} {:?}", g.fabric.transport());
            }
        }
    }
    std::fs::remove_dir_all(cache()).ok();
}

fn raw_prev(round: usize, _base: &[f32], parts: &[Vec<f32>], i: usize) -> f32 {
    assert_eq!(round, 1);
    let mut acc = 0.0f32;
    for p in parts {
        acc += p[i];
    }
    acc
}

fn run_norm(ranks: usize, rows: usize, width: usize, seed: u64) {
    let d = gpu_ordinal();
    let eps = 1e-6f32;
    let g = group(&vec![d; ranks], rows * width * 4, None, Some((width as u32, eps)));
    let mut rng = Rng(seed);
    let parts: Vec<Vec<f32>> = (0..ranks).map(|_| (0..rows * width).map(|_| rng.float() * 2.0).collect()).collect();
    let x: Vec<f32> = (0..rows * width).map(|_| rng.float()).collect();
    let w: Vec<f32> = (0..width).map(|_| rng.float() + 1.0).collect();
    let mut bufs = Vec::new();
    for r in 0..ranks {
        let s = &g.streams[r];
        bufs.push((s.clone_htod(&parts[r]).unwrap(), s.clone_htod(&x).unwrap(), s.clone_htod(&w).unwrap(), s.alloc_zeros::<u16>(rows * width).unwrap(), s.alloc_zeros::<u16>(rows * width).unwrap()));
    }
    std::thread::scope(|s| {
        for (r, b) in bufs.iter().enumerate() {
            let (comm, stream) = (&g.comms[r], &g.streams[r]);
            s.spawn(move || {
                stream.context().bind_to_thread().unwrap();
                comm.allreduce_add_norm(stream, &b.0.as_view(), &b.1.as_view(), &b.2, &b.3.as_view(), &b.4.as_view(), rows).unwrap();
                stream.synchronize().unwrap();
            });
        }
    });
    for r in 0..ranks {
        let s = &g.streams[r];
        let (got_x, hi, lo) = (s.clone_dtoh(&bufs[r].1).unwrap(), s.clone_dtoh(&bufs[r].3).unwrap(), s.clone_dtoh(&bufs[r].4).unwrap());
        for row in 0..rows {
            let mut summed = vec![0f32; width];
            for i in 0..width {
                let mut acc = 0.0f32;
                for p in &parts {
                    acc += p[row * width + i];
                }
                summed[i] = x[row * width + i] + acc;
                assert_eq!(got_x[row * width + i].to_bits(), summed[i].to_bits(), "rank {r} row {row} col {i}: residual sum");
            }
            let ss: f64 = summed.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let inv = 1.0 / ((ss / width as f64) + eps as f64).sqrt();
            for i in 0..width {
                let want = w[i] as f64 * summed[i] as f64 * inv;
                let got = from_bf16(hi[row * width + i]) as f64 + from_bf16(lo[row * width + i]) as f64;
                assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "rank {r} row {row} col {i}: {got} vs {want}");
            }
        }
    }
    std::fs::remove_dir_all(cache()).ok();
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn virtual_ranks_sum_random_data_from_one_row_to_16k_rows() {
    let d = gpu_ordinal();
    for &n in &[2usize, 4] {
        for dtype in [Dtype::F32, Dtype::Bf16] {
            for rows in [1usize, 2, 8, 64, 513, 4096, 16384] {
                run(&vec![d; n], None, dtype, rows * 1024, 7 + rows as u64);
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn odd_sizes_do_not_leave_a_ragged_tail() {
    let d = gpu_ordinal();
    for n in [2usize, 3, 4] {
        for elements in [1usize, 3, 7, 1001, 66001] {
            run(&vec![d; n], None, Dtype::F32, elements, 99 + elements as u64);
            run(&vec![d; n], None, Dtype::Bf16, elements, 98 + elements as u64);
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_host_pinned_fallback_gives_the_same_sums() {
    let d = gpu_ordinal();
    for dtype in [Dtype::F32, Dtype::Bf16] {
        for rows in [1usize, 64, 4096] {
            run(&[d, d], Some(Transport::HostPinned), dtype, rows * 1024, 31 + rows as u64);
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn the_fused_add_rmsnorm_sums_exactly_and_normalises() {
    for ranks in [2usize, 4] {
        for rows in [1usize, 8, 64, 300] {
            run_norm(ranks, rows, 256, 11 + rows as u64);
        }
        run_norm(ranks, 4, 5120, 5);
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_peer_write_that_never_lands_falls_back_to_host_memory_within_the_timeout() {
    let d = gpu_ordinal();
    let ctxs = contexts(&[d, d]).unwrap();
    let landed = peer_writes_land(&ctxs, PROBE_TIMEOUT, true).unwrap();
    assert!(landed.iter().all(|p| p.2 == ProbeOutcome::Landed), "{landed:?}");
    let started = std::time::Instant::now();
    let fabric = Fabric::with_probe(&ctxs, 1 << 20, None, |c| peer_writes_land(c, PROBE_TIMEOUT, false)).unwrap();
    let waited = started.elapsed();
    eprintln!("dropped peer write: transport {:?} after {waited:?}", fabric.transport());
    assert_eq!(fabric.transport(), Transport::HostPinned);
    assert!(waited < std::time::Duration::from_secs(2), "the probe took {waited:?}");
    let trusted = Fabric::with_probe(&ctxs, 1 << 20, None, |c| peer_writes_land(c, PROBE_TIMEOUT, true)).unwrap();
    assert_eq!(trusted.transport(), Transport::SameDevice);
    drop((fabric, trusted));
    run(&[d, d], Some(Transport::HostPinned), Dtype::Bf16, 64 * 1024, 77);
}

#[test]
#[ignore = "needs two CUDA GPUs and the user's GPU 0 free"]
fn two_gpu_transport_report() {
    let ctxs = contexts(&[0, 1]).unwrap();
    for c in &ctxs {
        eprintln!("gpu{} iommu group type: {:?}", c.ordinal(), specrt::allreduce::iommu_mode(c));
    }
    eprintln!("peer writes: {:?}", peer_writes_land(&ctxs, PROBE_TIMEOUT, true).unwrap());
    let f = Fabric::new(&ctxs, 1 << 20, None).unwrap();
    eprintln!("auto transport for gpu0+gpu1: {:?}", f.transport());
}

#[test]
#[ignore = "needs two CUDA GPUs and the user's GPU 0 free"]
fn two_gpu_peer_group_sums_exactly() {
    for dtype in [Dtype::F32, Dtype::Bf16] {
        for rows in [1usize, 8, 64, 4096, 16384] {
            run(&[0, 1], None, dtype, rows * 5120, 5 + rows as u64);
        }
    }
}

#[test]
#[ignore = "needs two CUDA GPUs and the user's GPU 0 free"]
fn two_gpu_allreduce_latency_for_1_8_and_64_rows() {
    let width = 5120usize;
    for force in [None, Some(Transport::HostPinned)] {
        for rows in [1usize, 8, 64] {
            let bytes = rows * width * 2;
            let g = group(&[0, 1], bytes, force, None);
            let ins: Vec<CudaSlice<u8>> = (0..2).map(|r| g.streams[r].alloc_zeros::<u8>(bytes).unwrap()).collect();
            let mut outs: Vec<CudaSlice<u8>> = (0..2).map(|r| g.streams[r].alloc_zeros::<u8>(bytes).unwrap()).collect();
            let rounds = 500;
            let mut times = Vec::new();
            std::thread::scope(|s| {
                let handles: Vec<_> = ins.iter().zip(outs.iter_mut()).enumerate().map(|(r, (inp, out))| {
                    let (comm, stream) = (&g.comms[r], &g.streams[r]);
                    s.spawn(move || {
                        stream.context().bind_to_thread().unwrap();
                        for _ in 0..20 {
                            comm.allreduce_raw(stream, Dtype::Bf16, inp, out, rows * width, false).unwrap();
                        }
                        stream.synchronize().unwrap();
                        let t = std::time::Instant::now();
                        for _ in 0..rounds {
                            comm.allreduce_raw(stream, Dtype::Bf16, inp, out, rows * width, false).unwrap();
                        }
                        stream.synchronize().unwrap();
                        t.elapsed().as_secs_f64()
                    })
                }).collect();
                for h in handles {
                    times.push(h.join().unwrap());
                }
            });
            let secs = times.into_iter().fold(0.0, f64::max);
            eprintln!("allreduce {:?} rows {rows} bf16 width {width}: {:.1} us", g.fabric.transport(), secs / rounds as f64 * 1e6);
        }
    }
    std::fs::remove_dir_all(cache()).ok();
}
