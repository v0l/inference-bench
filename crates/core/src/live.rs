use crate::result::Generation;
use chrono::{DateTime, Utc};
use specrt::serve::{Histogram, LATENCY_BUCKETS, Stats};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const TTFT_BUCKETS: [f64; 16] = [0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0, 20.0, 40.0, 120.0];
pub const RECENT_REQUESTS: usize = 50;
pub const HISTORY_SAMPLES: usize = 900;
pub const LATENCY_WINDOW: Duration = Duration::from_secs(30);
pub const RATE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq)]
pub struct Ring<T> {
    cap: usize,
    items: VecDeque<T>,
}

impl<T> Ring<T> {
    pub fn new(cap: usize) -> Self {
        Self { cap: cap.max(1), items: VecDeque::new() }
    }

    pub fn push(&mut self, item: T) {
        if self.items.len() == self.cap {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.items.iter()
    }

    pub fn last(&self) -> Option<&T> {
        self.items.back()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestRecord {
    pub at: DateTime<Utc>,
    pub ttft_secs: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub decode_tok_s: f64,
}

struct MeterInner {
    generated: u64,
    prefilled: u64,
    requests: u64,
    ttft: Histogram,
    itl: Histogram,
    recent: Ring<RequestRecord>,
}

pub struct Meter {
    inner: Mutex<MeterInner>,
    in_flight: AtomicUsize,
    pending: AtomicUsize,
}

impl Default for Meter {
    fn default() -> Self {
        Self {
            inner: Mutex::new(MeterInner {
                generated: 0,
                prefilled: 0,
                requests: 0,
                ttft: Histogram::new(&TTFT_BUCKETS),
                itl: Histogram::new(&LATENCY_BUCKETS),
                recent: Ring::new(RECENT_REQUESTS),
            }),
            in_flight: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
        }
    }
}

impl Meter {
    pub fn submitted(&self) -> Instant {
        self.pending.fetch_add(1, Ordering::SeqCst);
        Instant::now()
    }

    pub fn begin(self: &Arc<Self>, queued_at: Instant) -> Tracked {
        self.leave_queue();
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        Tracked { meter: self.clone(), started: queued_at, first: None, last: None, pieces: 0 }
    }

    pub fn withdrawn(&self) {
        self.leave_queue();
    }

    fn leave_queue(&self) {
        let mut n = self.pending.load(Ordering::SeqCst);
        while n > 0 {
            match self.pending.compare_exchange(n, n - 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return,
                Err(c) => n = c,
            }
        }
    }

    pub fn running(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }

    pub fn recent(&self) -> Vec<RequestRecord> {
        self.inner.lock().unwrap().recent.iter().cloned().collect()
    }
}

pub struct Tracked {
    meter: Arc<Meter>,
    started: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    pieces: u64,
}

impl Tracked {
    pub fn piece(&mut self) {
        let now = Instant::now();
        let mut m = self.meter.inner.lock().unwrap();
        match self.last {
            Some(prev) => m.itl.observe(now.duration_since(prev).as_secs_f64()),
            None => {
                m.ttft.observe(now.duration_since(self.started).as_secs_f64());
                self.first = Some(now);
            }
        }
        m.generated += 1;
        self.last = Some(now);
        self.pieces += 1;
    }

    pub fn finish(self, g: &Generation) {
        let mut m = self.meter.inner.lock().unwrap();
        m.generated += g.completion_tokens.saturating_sub(self.pieces);
        m.prefilled += g.prompt_tokens.saturating_sub(g.cached_tokens);
        m.requests += 1;
        let decode_tok_s = match g.decode_secs > 0.0 {
            true => g.completion_tokens.saturating_sub(1) as f64 / g.decode_secs,
            false => 0.0,
        };
        let ttft_secs = self.first.map_or(g.ttft_secs, |f| f.duration_since(self.started).as_secs_f64());
        m.recent.push(RequestRecord { at: Utc::now(), ttft_secs, prompt_tokens: g.prompt_tokens, completion_tokens: g.completion_tokens, decode_tok_s });
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.meter.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone)]
pub struct Reading {
    pub generated: u64,
    pub prefilled: u64,
    pub requests: u64,
    pub running: usize,
    pub waiting: usize,
    pub max_seqs: usize,
    pub ttft: Histogram,
    pub itl: Histogram,
    pub serving: Option<Stats>,
}

impl Meter {
    pub fn read(&self, max_seqs: usize, serving: Option<Stats>) -> Reading {
        let m = self.inner.lock().unwrap();
        let (running, waiting) = match &serving {
            Some(s) => (s.running, s.waiting + self.pending()),
            None => (self.running(), self.pending()),
        };
        let (generated, prefilled, itl) = match &serving {
            Some(s) => (s.generated_tokens, s.prefill_tokens_per_step.sum as u64, s.inter_token_secs.clone()),
            None => (m.generated, m.prefilled, m.itl.clone()),
        };
        Reading { generated, prefilled, requests: m.requests, running, waiting, max_seqs, ttft: m.ttft.clone(), itl, serving }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Point {
    pub gen_tok_s: f64,
    pub prefill_tok_s: f64,
    pub ttft_p50: f64,
    pub ttft_p95: f64,
    pub itl_p50: f64,
    pub itl_p95: f64,
    pub running: usize,
    pub waiting: usize,
    pub kv_used_bytes: f64,
    pub kv_total_bytes: f64,
    pub requests: u64,
}

pub struct History {
    ring: Ring<(Instant, Point)>,
    prev: Option<(Instant, Reading)>,
    marks: VecDeque<(Instant, Histogram, Histogram)>,
    held: Point,
    latest: Option<Reading>,
}

impl History {
    pub fn new(cap: usize) -> Self {
        Self { ring: Ring::new(cap), prev: None, marks: VecDeque::new(), held: Point::default(), latest: None }
    }

    pub fn push(&mut self, now: Instant, r: Reading) {
        let mut p = Point { running: r.running, waiting: r.waiting, requests: r.requests, ttft_p50: self.held.ttft_p50, ttft_p95: self.held.ttft_p95, itl_p50: self.held.itl_p50, itl_p95: self.held.itl_p95, ..Point::default() };
        if let Some((at, prev)) = &self.prev {
            let dt = now.duration_since(*at).as_secs_f64().max(1e-3);
            p.gen_tok_s = r.generated.saturating_sub(prev.generated) as f64 / dt;
            p.prefill_tok_s = r.prefilled.saturating_sub(prev.prefilled) as f64 / dt;
        }
        self.marks.push_back((now, r.ttft.clone(), r.itl.clone()));
        while self.marks.len() > 1 && now.duration_since(self.marks[1].0) >= LATENCY_WINDOW {
            self.marks.pop_front();
        }
        if let Some((_, ttft0, itl0)) = self.marks.front() {
            let (ttft, itl) = (r.ttft.since(ttft0), r.itl.since(itl0));
            if let (Some(a), Some(b)) = (ttft.quantile(0.5), ttft.quantile(0.95)) {
                (p.ttft_p50, p.ttft_p95) = (a, b);
            }
            if let (Some(a), Some(b)) = (itl.quantile(0.5), itl.quantile(0.95)) {
                (p.itl_p50, p.itl_p95) = (a, b);
            }
        }
        if let Some(s) = &r.serving {
            p.kv_used_bytes = (s.kv_blocks_used * s.kv_block_bytes) as f64;
            p.kv_total_bytes = (s.kv_blocks_total * s.kv_block_bytes) as f64;
        }
        self.held = p.clone();
        self.ring.push((now, p));
        self.prev = Some((now, r.clone()));
        self.latest = Some(r);
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    pub fn points(&self) -> impl DoubleEndedIterator<Item = &Point> + ExactSizeIterator {
        self.ring.iter().map(|(_, p)| p)
    }

    pub fn latest(&self) -> Option<&Reading> {
        self.latest.as_ref()
    }

    pub fn last(&self) -> Option<&Point> {
        self.ring.last().map(|(_, p)| p)
    }

    pub fn series(&self, take: usize, f: impl Fn(&Point) -> f64) -> Vec<f32> {
        let skip = self.ring.len().saturating_sub(take);
        let mut out = vec![0.0f32; take.saturating_sub(self.ring.len())];
        out.extend(self.ring.iter().skip(skip).map(|(_, p)| f(p) as f32));
        out
    }

    pub fn requests_per_minute(&self) -> f64 {
        let Some((now, last)) = self.ring.iter().next_back() else { return 0.0 };
        let base = self.ring.iter().rev().find(|(at, _)| now.duration_since(*at) >= RATE_WINDOW).or_else(|| self.ring.iter().next());
        match base {
            Some((at, p)) if now.duration_since(*at) > Duration::ZERO => {
                let secs = now.duration_since(*at).as_secs_f64().min(RATE_WINDOW.as_secs_f64());
                last.requests.saturating_sub(p.requests) as f64 * 60.0 / secs.max(1.0)
            }
            _ => 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.ring.clear();
        self.prev = None;
        self.marks.clear();
        self.held = Point::default();
        self.latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(generated: u64, requests: u64) -> Reading {
        Meter::default().read(4, None).with(generated, requests)
    }

    impl Reading {
        fn with(mut self, generated: u64, requests: u64) -> Self {
            self.generated = generated;
            self.requests = requests;
            self
        }
    }

    #[test]
    fn the_ring_drops_the_oldest_samples() {
        let mut r = Ring::new(3);
        for i in 0..5 {
            r.push(i);
        }
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(r.last(), Some(&4));
    }

    #[test]
    fn the_history_keeps_only_its_capacity_and_pads_short_series() {
        let t0 = Instant::now();
        let mut h = History::new(4);
        for i in 0..10u64 {
            h.push(t0 + Duration::from_secs(i), reading(i * 10, 0));
        }
        assert_eq!(h.len(), 4);
        assert_eq!(h.series(6, |p| p.gen_tok_s), [0.0, 0.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn throughput_is_the_token_delta_over_the_interval() {
        let t0 = Instant::now();
        let mut h = History::new(10);
        h.push(t0, reading(0, 0));
        h.push(t0 + Duration::from_secs(2), reading(100, 0));
        assert_eq!(h.last().unwrap().gen_tok_s, 50.0);
        h.push(t0 + Duration::from_secs(3), reading(100, 0));
        assert_eq!(h.last().unwrap().gen_tok_s, 0.0);
    }

    #[test]
    fn requests_per_minute_counts_finished_requests_in_the_last_minute() {
        let t0 = Instant::now();
        let mut h = History::new(900);
        for i in 0..=120u64 {
            h.push(t0 + Duration::from_secs(i), reading(0, i / 10));
        }
        let rpm = h.requests_per_minute();
        assert!((rpm - 6.0).abs() < 0.5, "{rpm}");
    }

    #[test]
    fn latency_percentiles_cover_the_recent_window_and_hold_when_idle() {
        let meter = Arc::new(Meter::default());
        let t0 = Instant::now();
        let mut h = History::new(900);
        h.push(t0, meter.read(4, None));
        {
            let mut t = meter.begin(Instant::now());
            std::thread::sleep(Duration::from_millis(30));
            t.piece();
            t.piece();
            t.finish(&Generation { prompt_tokens: 5, completion_tokens: 2, decode_secs: 0.01, ..Generation::default() });
        }
        h.push(t0 + Duration::from_secs(1), meter.read(4, None));
        let p = h.last().unwrap().clone();
        assert!(p.ttft_p50 >= 0.025 && p.ttft_p95 >= p.ttft_p50, "{p:?}");
        assert!(p.itl_p50 > 0.0);
        h.push(t0 + Duration::from_secs(2), meter.read(4, None));
        assert_eq!(h.last().unwrap().ttft_p50, p.ttft_p50);
    }

    #[test]
    fn a_meter_counts_tokens_prompt_and_per_request_records() {
        let meter = Arc::new(Meter::default());
        let queued = meter.submitted();
        assert_eq!(meter.pending(), 1);
        let mut t = meter.begin(queued);
        assert_eq!((meter.pending(), meter.running()), (0, 1));
        t.piece();
        t.finish(&Generation { prompt_tokens: 9, completion_tokens: 3, decode_secs: 0.2, ..Generation::default() });
        assert_eq!(meter.running(), 0);
        let r = meter.read(1, None);
        assert_eq!((r.generated, r.prefilled, r.requests), (3, 9, 1));
        let rec = meter.recent();
        assert_eq!((rec[0].prompt_tokens, rec[0].completion_tokens), (9, 3));
        assert_eq!(rec[0].decode_tok_s, 10.0);
    }

    #[test]
    fn serving_stats_take_over_running_waiting_and_prefill() {
        let meter = Meter::default();
        let mut s = Stats { running: 3, waiting: 2, generated_tokens: 40, ..Stats::default() };
        s.prefill_tokens_per_step.observe(100.0);
        s.prefill_tokens_per_step.observe(50.0);
        let r = meter.read(4, Some(s));
        assert_eq!((r.running, r.waiting, r.prefilled, r.generated), (3, 2, 150, 40));
    }
}
