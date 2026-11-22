use std::time::Instant;

const COLS: usize = 3072;
const ROWS_PER_EXPERT: usize = 3 * 1024;
const BYTES_PER_ROW: usize = COLS / 2;
const EXPERT_BYTES: usize = ROWS_PER_EXPERT * BYTES_PER_ROW;

struct Args {
    threads: usize,
    pool_mb: usize,
    seconds: f64,
}

fn parse() -> Args {
    let mut a = Args { threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1), pool_mb: 2048, seconds: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_default();
        match k.as_str() {
            "--threads" => a.threads = v.parse().unwrap(),
            "--pool-mb" => a.pool_mb = v.parse().unwrap(),
            "--seconds" => a.seconds = v.parse().unwrap(),
            _ => panic!("unknown flag {k}"),
        }
    }
    a
}

fn fill(buf: &mut [u8]) {
    let mut s = 0x9e3779b97f4a7c15u64;
    for b in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s >> 32) as u8;
    }
}

const E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn gemv_ref(w: &[u8], act: &[i8], out: &mut [i32]) {
    for (r, o) in out.iter_mut().enumerate() {
        let row = &w[r * BYTES_PER_ROW..(r + 1) * BYTES_PER_ROW];
        let mut acc = 0i32;
        for (c, chunk) in row.chunks(64).enumerate() {
            for (i, b) in chunk.iter().enumerate() {
                let lo = E2M1_X2[(b & 15) as usize] as i32;
                let hi = E2M1_X2[(b >> 4) as usize] as i32;
                acc += lo * act[c * 128 + i] as i32 + hi * act[c * 128 + 64 + i] as i32;
            }
        }
        *o = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn gemv_avx512(w: &[u8], act: &[i8], out: &mut [i32]) {
    use std::arch::x86_64::*;
    let lut_bytes: [u8; 16] = [12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0];
    let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(lut_bytes.as_ptr() as *const __m128i));
    let mask = _mm512_set1_epi8(0x0f);
    let ones = _mm512_set1_epi8(1);
    let chunks = BYTES_PER_ROW / 64;
    let mut asum = 0i32;
    for c in 0..chunks * 2 {
        let v = _mm512_loadu_si512(act.as_ptr().add(c * 64) as *const _);
        asum += _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, v));
    }
    for (r, o) in out.iter_mut().enumerate() {
        let row = w.as_ptr().add(r * BYTES_PER_ROW);
        let mut acc = _mm512_setzero_si512();
        for c in 0..chunks {
            let b = _mm512_loadu_si512(row.add(c * 64) as *const _);
            let lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(b, mask));
            let hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(b, 4), mask));
            let a0 = _mm512_loadu_si512(act.as_ptr().add(c * 128) as *const _);
            let a1 = _mm512_loadu_si512(act.as_ptr().add(c * 128 + 64) as *const _);
            acc = _mm512_dpbusd_epi32(acc, lo, a0);
            acc = _mm512_dpbusd_epi32(acc, hi, a1);
        }
        *o = _mm512_reduce_add_epi32(acc) - 12 * asum;
    }
}

fn run_gemv(w: &[u8], act: &[i8], out: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
        unsafe { gemv_avx512(w, act, out) };
        return;
    }
    gemv_ref(w, act, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn read_sum(buf: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let mut p = buf.as_ptr();
    let end = p.add(buf.len() & !255);
    while p < end {
        let a = _mm512_loadu_si512(p as *const _);
        let b = _mm512_loadu_si512(p.add(64) as *const _);
        let c = _mm512_loadu_si512(p.add(128) as *const _);
        let d = _mm512_loadu_si512(p.add(192) as *const _);
        acc = _mm512_xor_si512(acc, _mm512_xor_si512(_mm512_xor_si512(a, b), _mm512_xor_si512(c, d)));
        p = p.add(256);
    }
    _mm512_reduce_add_epi64(acc) as u64
}

fn timed<F: Fn(usize) + Sync>(threads: usize, seconds: f64, f: F) -> (f64, usize) {
    let start = Instant::now();
    let iters: usize = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let f = &f;
                s.spawn(move || {
                    let mut n = 0;
                    while start.elapsed().as_secs_f64() < seconds {
                        f(t);
                        n += 1;
                    }
                    n
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });
    (start.elapsed().as_secs_f64(), iters)
}

fn main() {
    let a = parse();
    let n_experts = (a.pool_mb << 20) / EXPERT_BYTES;
    let mut pool = vec![0u8; n_experts * EXPERT_BYTES];
    fill(&mut pool);
    let act: Vec<i8> = pool[..COLS].iter().map(|&b| b as i8 / 4).collect();
    println!("threads {} pool {} MB experts {} expert {:.2} MB", a.threads, a.pool_mb, n_experts, EXPERT_BYTES as f64 / 1e6);

    let mut o1 = vec![0i32; 64];
    let mut o2 = vec![0i32; 64];
    gemv_ref(&pool[..64 * BYTES_PER_ROW], &act, &mut o1);
    run_gemv(&pool[..64 * BYTES_PER_ROW], &act, &mut o2);
    assert_eq!(o1, o2, "avx512 gemv differs from reference");
    println!("gemv matches scalar reference");

    let share = pool.len() / a.threads / 256 * 256;
    let (secs, iters) = timed(a.threads, a.seconds, |t| {
        let s = unsafe { read_sum(&pool[t * share..(t + 1) * share]) };
        std::hint::black_box(s);
    });
    println!("read bandwidth {:.1} GB/s", (iters * share) as f64 / secs / 1e9);

    let src = &pool[..pool.len() / 2];
    let mut dst = vec![0u8; src.len()];
    let dshare = src.len() / a.threads;
    let dptr = dst.as_mut_ptr() as usize;
    let (secs, iters) = timed(a.threads, a.seconds, |t| unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr().add(t * dshare), (dptr as *mut u8).add(t * dshare), dshare);
    });
    println!("memcpy bandwidth {:.1} GB/s (read+write counted once)", (iters * dshare) as f64 / secs / 1e9);
    drop(dst);

    let per_thread_experts = n_experts.max(a.threads) / a.threads;
    let (secs, iters) = timed(a.threads, a.seconds, |t| {
        let mut out = vec![0i32; ROWS_PER_EXPERT];
        let mut k = 0;
        while k < per_thread_experts {
            let e = (t * per_thread_experts + k) % n_experts;
            run_gemv(&pool[e * EXPERT_BYTES..(e + 1) * EXPERT_BYTES], &act, &mut out);
            std::hint::black_box(&out);
            k += 1;
        }
    });
    let experts = iters * per_thread_experts;
    let bytes = experts as f64 * EXPERT_BYTES as f64;
    println!("cpu gemv, one expert per thread task, cold: {:.1} GB/s weights, {:.0} experts/s, {:.1} us per expert (whole machine)", bytes / secs / 1e9, experts as f64 / secs, secs / experts as f64 * 1e6);
    println!("pcie reference: 57.8 GB/s = {:.0} experts/s", 57.8e9 / EXPERT_BYTES as f64);

    let hot = &pool[..EXPERT_BYTES];
    let (secs, iters) = timed(a.threads, a.seconds, |_| {
        let mut out = vec![0i32; ROWS_PER_EXPERT];
        run_gemv(hot, &act, &mut out);
        std::hint::black_box(&out);
    });
    println!("cpu gemv same expert (L2/L3 resident, compute ceiling): {:.1} GB/s weights", iters as f64 * EXPERT_BYTES as f64 / secs / 1e9);
}
