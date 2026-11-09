use cudarc::driver::CudaContext;
use specrt::statepool::StatePool;

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_thousand_random_acquire_release_cycles_leave_no_memory_behind() {
    let ctx = CudaContext::new(0).unwrap();
    let stride = 3 * (1 << 20) + 4096;
    let mut pool = StatePool::new(&ctx, stride, 32).unwrap();
    assert_eq!(pool.mapped_bytes(), 0);
    let mut x = 99u64;
    let mut live: Vec<usize> = Vec::new();
    let mut peak = 0;
    for _ in 0..1000 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let slot = ((x >> 33) % 32) as usize;
        if live.contains(&slot) {
            pool.release(slot);
            live.retain(|&s| s != slot);
        } else {
            pool.acquire(slot).unwrap();
            live.push(slot);
        }
        peak = peak.max(pool.mapped_bytes());
        assert_eq!(pool.live_slots(), live.len());
        assert!(pool.mapped_bytes() <= live.len() * (stride.div_ceil(1 << 21) + 1) * (1 << 21));
    }
    assert!(pool.acquire(live.first().copied().unwrap_or(0)).is_err() || live.is_empty());
    for slot in live.drain(..) {
        pool.release(slot);
    }
    assert_eq!(pool.live_slots(), 0);
    assert_eq!(pool.mapped_bytes(), 0);
    assert_eq!(pool.outstanding_allocations(), 0);
    assert!(peak < 32 * stride);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_slot_can_be_acquired_released_and_reacquired_without_growth() {
    let ctx = CudaContext::new(0).unwrap();
    let mut pool = StatePool::new(&ctx, 5 << 20, 4).unwrap();
    for _ in 0..50 {
        pool.acquire(2).unwrap();
        let mapped = pool.mapped_bytes();
        pool.release(2);
        assert_eq!(pool.mapped_bytes(), 0);
        pool.acquire(2).unwrap();
        assert_eq!(pool.mapped_bytes(), mapped);
        pool.release(2);
    }
}
