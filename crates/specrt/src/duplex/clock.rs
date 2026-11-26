use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

pub type SessionId = u64;

#[derive(Clone, Debug, PartialEq)]
pub struct Out {
    pub audio: Vec<f32>,
    pub token: Option<u32>,
    pub text: String,
    pub index: u64,
    pub late: bool,
    pub done: bool,
}

pub trait FrameEngine {
    fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> anyhow::Result<Vec<(SessionId, Out)>>;
}

pub enum Event {
    Open(SessionId),
    Audio(SessionId, Vec<f32>),
    Close(SessionId),
}

#[derive(Default)]
struct Uplink {
    pending: Vec<f32>,
    frames: VecDeque<Vec<f32>>,
    due: u64,
    skip: u64,
}

pub struct Clock {
    pub frame: Duration,
    pub samples: usize,
    pub max_queue: usize,
}

impl Clock {
    fn tick(&self, start: Instant, n: u64) -> Instant {
        start + self.frame * n as u32
    }
}

pub struct Stats {
    pub steps: u64,
    pub late_frames: u64,
    pub dropped_input: u64,
    pub step_secs: Vec<f64>,
}

pub fn run(engine: &mut dyn FrameEngine, clock: &Clock, events: Receiver<Event>, out: Sender<(SessionId, Out)>, until: Option<Instant>) -> anyhow::Result<Stats> {
    let start = Instant::now();
    let mut ups: BTreeMap<SessionId, Uplink> = BTreeMap::new();
    let mut stats = Stats { steps: 0, late_frames: 0, dropped_input: 0, step_secs: Vec::new() };
    let mut tick = 1u64;
    loop {
        if until.is_some_and(|u| Instant::now() >= u) {
            return Ok(stats);
        }
        let deadline = clock.tick(start, tick);
        loop {
            let wait = deadline.saturating_duration_since(Instant::now());
            match events.recv_timeout(wait) {
                Ok(Event::Open(id)) => {
                    ups.insert(id, Uplink { due: tick, ..Uplink::default() });
                }
                Ok(Event::Audio(id, pcm)) => {
                    if let Some(u) = ups.get_mut(&id) {
                        u.pending.extend(pcm);
                        while u.pending.len() >= clock.samples {
                            let f: Vec<f32> = u.pending.drain(..clock.samples).collect();
                            u.frames.push_back(f);
                            if u.frames.len() > clock.max_queue {
                                u.frames.pop_front();
                                stats.dropped_input += 1;
                            }
                        }
                    }
                }
                Ok(Event::Close(id)) => {
                    ups.remove(&id);
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return Ok(stats),
            }
        }
        let now_tick = tick;
        tick += 1;
        let mut batch = Vec::new();
        for (&id, u) in ups.iter_mut() {
            if u.due > now_tick {
                continue;
            }
            while u.skip > 0 && u.frames.len() > 1 {
                u.frames.pop_front();
                u.skip -= 1;
                stats.dropped_input += 1;
            }
            let frame = u.frames.pop_front().unwrap_or_else(|| vec![0.0; clock.samples]);
            batch.push((id, frame));
        }
        if batch.is_empty() {
            continue;
        }
        let began = Instant::now();
        let produced = engine.step(&batch)?;
        stats.step_secs.push(began.elapsed().as_secs_f64());
        stats.steps += 1;
        let behind = Instant::now().saturating_duration_since(clock.tick(start, tick)).as_nanos() / clock.frame.as_nanos().max(1);
        for (id, mut o) in produced {
            if behind > 0 {
                o.late = true;
                stats.late_frames += 1;
            }
            if o.done {
                ups.remove(&id);
            }
            if out.send((id, o)).is_err() {
                return Ok(stats);
            }
        }
        if behind > 0 {
            tick += behind as u64;
            for u in ups.values_mut() {
                u.skip += behind as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    struct Fake {
        delay: Duration,
        slow_at: Option<(u64, Duration)>,
        steps: u64,
        seen: Vec<(SessionId, f32)>,
    }

    impl FrameEngine for Fake {
        fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> anyhow::Result<Vec<(SessionId, Out)>> {
            self.steps += 1;
            let d = match self.slow_at {
                Some((at, d)) if at == self.steps => d,
                _ => self.delay,
            };
            std::thread::sleep(d);
            Ok(frames.iter().map(|(id, f)| {
                self.seen.push((*id, f[0]));
                (*id, Out { audio: f.clone(), token: None, text: String::new(), index: self.steps, late: false, done: false })
            }).collect())
        }
    }

    fn clock() -> Clock {
        Clock { frame: Duration::from_millis(20), samples: 4, max_queue: 4 }
    }

    #[test]
    fn every_open_session_gets_one_frame_per_tick_and_silence_when_its_input_is_missing() {
        let (tx, rx) = channel();
        let (otx, orx) = channel();
        tx.send(Event::Open(1)).unwrap();
        tx.send(Event::Open(2)).unwrap();
        tx.send(Event::Audio(1, vec![7.0; 8])).unwrap();
        let mut e = Fake { delay: Duration::ZERO, slow_at: None, steps: 0, seen: Vec::new() };
        let stats = run(&mut e, &clock(), rx, otx, Some(Instant::now() + Duration::from_millis(110))).unwrap();
        let outs: Vec<(SessionId, Out)> = orx.try_iter().collect();
        assert!(stats.steps >= 4, "{} steps", stats.steps);
        let ones: Vec<f32> = e.seen.iter().filter(|s| s.0 == 1).map(|s| s.1).collect();
        assert_eq!(&ones[..3], &[7.0, 7.0, 0.0], "two real frames then silence");
        assert!(e.seen.iter().filter(|s| s.0 == 2).all(|s| s.1 == 0.0));
        assert_eq!(outs.iter().filter(|o| o.0 == 1).count(), outs.iter().filter(|o| o.0 == 2).count());
        assert_eq!(stats.late_frames, 0);
    }

    #[test]
    fn a_late_engine_sends_what_it_has_marks_it_late_and_never_blocks_the_input_side() {
        let (tx, rx) = channel();
        let (otx, orx) = channel();
        tx.send(Event::Open(1)).unwrap();
        let mut e = Fake { delay: Duration::from_millis(1), slow_at: Some((2, Duration::from_millis(70))), steps: 0, seen: Vec::new() };
        let sender = std::thread::spawn(move || {
            let started = Instant::now();
            let mut worst = Duration::ZERO;
            for i in 0..12 {
                let t = Instant::now();
                tx.send(Event::Audio(1, vec![i as f32; 4])).unwrap();
                worst = worst.max(t.elapsed());
                std::thread::sleep(Duration::from_millis(20));
            }
            (started.elapsed(), worst)
        });
        let stats = run(&mut e, &clock(), rx, otx, Some(Instant::now() + Duration::from_millis(260))).unwrap();
        let (_, worst_send) = sender.join().unwrap();
        let outs: Vec<(SessionId, Out)> = orx.try_iter().collect();
        assert!(worst_send < Duration::from_millis(5), "sending input waited {worst_send:?}");
        assert!(stats.late_frames >= 1, "the slow step is reported late");
        assert!(outs.iter().any(|o| o.1.late));
        assert!(stats.dropped_input >= 1, "the backlog is cut so latency comes back");
        assert!(outs.iter().filter(|o| !o.1.late).count() >= 3, "frames after the slow step are on time again");
    }

    #[test]
    fn a_full_input_queue_drops_the_oldest_frame() {
        let (tx, rx) = channel();
        let (otx, orx) = channel();
        tx.send(Event::Open(9)).unwrap();
        tx.send(Event::Audio(9, (0..40).map(|i| (i / 4) as f32).collect())).unwrap();
        let mut e = Fake { delay: Duration::ZERO, slow_at: None, steps: 0, seen: Vec::new() };
        let stats = run(&mut e, &clock(), rx, otx, Some(Instant::now() + Duration::from_millis(30))).unwrap();
        let _ = orx.try_iter().count();
        assert_eq!(stats.dropped_input, 6);
        assert_eq!(e.seen[0], (9, 6.0), "the newest four frames are kept");
    }

    #[test]
    fn a_done_frame_closes_the_session() {
        struct Ends;
        impl FrameEngine for Ends {
            fn step(&mut self, frames: &[(SessionId, Vec<f32>)]) -> anyhow::Result<Vec<(SessionId, Out)>> {
                Ok(frames.iter().map(|(id, _)| (*id, Out { audio: Vec::new(), token: None, text: String::new(), index: 0, late: false, done: true })).collect())
            }
        }
        let (tx, rx) = channel();
        let (otx, orx) = channel();
        tx.send(Event::Open(3)).unwrap();
        run(&mut Ends, &clock(), rx, otx, Some(Instant::now() + Duration::from_millis(80))).unwrap();
        assert_eq!(orx.try_iter().count(), 1);
    }
}
