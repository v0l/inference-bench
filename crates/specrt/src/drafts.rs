use crate::spec::MAX_DRAFT_TOKENS;
use std::collections::HashMap;

const RING: usize = 8;
const PRIOR_WEIGHT: f64 = 2.0;
const SWITCH_GAIN: f64 = 1.03;
const FIRST_RETEST: u64 = 16;
const MAX_FAILED_RETESTS: u32 = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Positions {
    tried: [u32; MAX_DRAFT_TOKENS],
    accepted: [u32; MAX_DRAFT_TOKENS],
}

impl Positions {
    pub fn record(&mut self, k: usize, accepted: usize) {
        for j in 0..k.min(accepted + 1).min(MAX_DRAFT_TOKENS) {
            self.tried[j] += 1;
            if j < accepted {
                self.accepted[j] += 1;
            }
        }
    }

    pub fn tried(&self, position: usize) -> u32 {
        self.tried[position]
    }

    fn alpha(&self, position: usize, prior: f64) -> f64 {
        (self.accepted[position] as f64 + PRIOR_WEIGHT * prior) / (self.tried[position] as f64 + PRIOR_WEIGHT)
    }
}

#[derive(Debug, Default)]
struct Ring(Vec<f64>);

impl Ring {
    fn push(&mut self, v: f64) {
        if self.0.len() == RING {
            self.0.remove(0);
        }
        self.0.push(v);
    }

    fn median(&self) -> f64 {
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
}

#[derive(Debug)]
struct Retest {
    since: u64,
    interval: u64,
    failed: u32,
}

impl Default for Retest {
    fn default() -> Self {
        Retest { since: 0, interval: FIRST_RETEST, failed: 0 }
    }
}

#[derive(Debug, Clone, Copy)]
struct Trial {
    rows: usize,
    k: usize,
    left: u8,
}

#[derive(Debug)]
pub struct DraftChooser {
    kmax: usize,
    costs: HashMap<(usize, usize), Ring>,
    global: Positions,
    retests: HashMap<usize, Retest>,
    trial: Option<Trial>,
    last: usize,
    warm: bool,
}

impl DraftChooser {
    pub fn new(kmax: usize) -> Self {
        let kmax = kmax.min(MAX_DRAFT_TOKENS);
        DraftChooser { kmax, costs: HashMap::new(), global: Positions::default(), retests: HashMap::new(), trial: None, last: 0, warm: false }
    }

    pub fn global(&self) -> &Positions {
        &self.global
    }

    pub fn expected_tokens(&self, p: &Positions, k: usize) -> f64 {
        let mut chain = 1.0;
        let mut total = 1.0;
        for j in 0..k {
            chain *= p.alpha(j, self.global.alpha(j, 0.5));
            total += chain;
        }
        total
    }

    pub fn cost(&self, rows: usize, k: usize) -> Option<f64> {
        if let Some(r) = self.costs.get(&(rows, k)) {
            return Some(r.median());
        }
        let mut seen: Vec<(usize, f64)> = self.costs.iter().filter(|((_, kk), _)| *kk == k).map(|((n, _), r)| (*n, r.median())).collect();
        if seen.is_empty() {
            return None;
        }
        seen.sort_by_key(|s| s.0);
        let below = seen.iter().rev().find(|s| s.0 < rows);
        let above = seen.iter().find(|s| s.0 > rows);
        Some(match (below, above) {
            (Some(&(n0, t0)), Some(&(n1, t1))) => t0 + (t1 - t0) * (rows - n0) as f64 / (n1 - n0) as f64,
            (Some(&(_, t)), None) | (None, Some(&(_, t))) => t,
            (None, None) => unreachable!("a cost row exists"),
        })
    }

    fn measured(&self, k: usize) -> bool {
        self.costs.keys().any(|(_, kk)| *kk == k)
    }

    fn score(&self, rows: &[Positions], k: usize) -> Option<f64> {
        let cost = self.cost(rows.len(), k)?.max(1e-9);
        let tokens = if k == 0 { rows.len() as f64 } else { rows.iter().map(|p| self.expected_tokens(p, k)).sum() };
        Some(tokens / cost)
    }

    fn best(&self, rows: &[Positions]) -> usize {
        let mut best = (0, f64::MIN);
        for k in 0..=self.kmax {
            if let Some(s) = self.score(rows, k)
                && s > best.1
            {
                best = (k, s);
            }
        }
        best.0
    }

    fn best_drafting(&self, rows: &[Positions]) -> usize {
        (1..=self.kmax).filter_map(|k| self.score(rows, k).map(|s| (k, s))).max_by(|a, b| a.1.total_cmp(&b.1)).map_or(self.kmax, |b| b.0)
    }

    pub fn choose(&mut self, rows: &[Positions]) -> usize {
        let k = self.pick(rows);
        self.warm = k == 0 || k <= self.last;
        self.last = k;
        k
    }

    fn pick(&mut self, rows: &[Positions]) -> usize {
        if let Some(mut t) = self.trial {
            if t.left > 0 {
                t.left -= 1;
                self.trial = Some(t);
                return t.k;
            }
            self.trial = None;
            let failed = self.best(rows) == 0;
            let r = self.retests.entry(t.rows).or_default();
            if failed {
                r.failed += 1;
                r.interval *= 2;
            } else {
                *r = Retest::default();
            }
        }
        if let Some(k) = (0..=self.kmax).rev().find(|&k| !self.measured(k)) {
            return k;
        }
        let mut best = self.best(rows);
        if best != self.last
            && let (Some(now), Some(cur)) = (self.score(rows, best), self.score(rows, self.last))
            && now < cur * SWITCH_GAIN
        {
            best = self.last;
        }
        if best == 0 {
            let r = self.retests.entry(rows.len()).or_default();
            if r.failed < MAX_FAILED_RETESTS {
                r.since += 1;
                if r.since >= r.interval {
                    r.since = 0;
                    let k = self.best_drafting(rows);
                    self.costs.remove(&(rows.len(), k));
                    self.trial = Some(Trial { rows: rows.len(), k, left: 1 });
                    return k;
                }
            }
        }
        best
    }

    pub fn note_plain(&mut self) {
        self.last = 0;
    }

    pub fn observe(&mut self, rows: usize, k: usize, secs: f64) {
        if self.warm && secs > 0.0 {
            self.costs.entry((rows, k)).or_default().push(secs);
        }
    }

    pub fn observe_accepts(&mut self, k: usize, accepted: &[usize]) {
        for &a in accepted {
            self.global.record(k, a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, accept: usize, k: usize, times: usize) -> Vec<Positions> {
        let mut p = Positions::default();
        for _ in 0..times {
            p.record(k, accept);
        }
        vec![p; n]
    }

    fn feed(c: &mut DraftChooser, n: usize, k: usize, secs: f64, accept: usize, times: usize) {
        for _ in 0..times {
            c.costs.entry((n, k)).or_default().push(secs);
            c.observe_accepts(k, &vec![accept; n]);
        }
    }

    #[test]
    fn laplace_prior_gives_half_before_any_data() {
        let c = DraftChooser::new(3);
        let p = Positions::default();
        assert!((c.expected_tokens(&p, 3) - (1.0 + 0.5 + 0.25 + 0.125)).abs() < 1e-9);
    }

    #[test]
    fn positions_count_the_first_rejected_draft_as_tried() {
        let mut p = Positions::default();
        p.record(3, 1);
        assert_eq!((p.tried(0), p.tried(1), p.tried(2)), (1, 1, 0));
        assert_eq!((p.accepted[0], p.accepted[1]), (1, 0));
    }

    #[test]
    fn cost_interpolates_between_row_counts_and_stays_flat_past_the_deepest() {
        let mut c = DraftChooser::new(2);
        c.costs.entry((1, 2)).or_default().push(10.0);
        c.costs.entry((5, 2)).or_default().push(30.0);
        assert_eq!(c.cost(3, 2), Some(20.0));
        assert_eq!(c.cost(8, 2), Some(30.0));
        assert_eq!(c.cost(5, 1), None);
    }

    #[test]
    fn picks_no_drafts_when_verify_is_expensive() {
        let mut c = DraftChooser::new(3);
        feed(&mut c, 8, 0, 0.020, 0, 4);
        for k in 1..=3 {
            feed(&mut c, 8, k, 0.060 + 0.010 * k as f64, 1, 4);
        }
        assert_eq!(c.choose(&rows(8, 1, 3, 20)), 0);
    }

    #[test]
    fn picks_three_when_acceptance_is_high_and_verify_is_cheap() {
        let mut c = DraftChooser::new(3);
        feed(&mut c, 1, 0, 0.020, 0, 4);
        for k in 1..=3 {
            feed(&mut c, 1, k, 0.021 + 0.001 * k as f64, 3, 4);
        }
        assert_eq!(c.choose(&rows(1, 3, 3, 30)), 3);
    }

    #[test]
    fn explores_every_draft_count_before_deciding() {
        let mut c = DraftChooser::new(3);
        let r = rows(2, 0, 1, 0);
        let mut seen = Vec::new();
        for _ in 0..8 {
            let k = c.choose(&r);
            seen.push(k);
            c.observe(2, k, 0.01);
        }
        for k in 0..=3 {
            assert!(seen.contains(&k), "{seen:?}");
        }
    }

    fn simulate(c: &mut DraftChooser, n: usize, steps: usize, cost: [f64; 3], accept: usize) -> Vec<usize> {
        let mut ks = Vec::new();
        let mut seqs = vec![Positions::default(); n];
        for _ in 0..steps {
            let k = c.choose(&seqs);
            c.observe(n, k, cost[k]);
            if k > 0 {
                let a = accept.min(k);
                c.observe_accepts(k, &vec![a; n]);
                for p in &mut seqs {
                    p.record(k, a);
                }
            }
            ks.push(k);
        }
        ks
    }

    #[test]
    fn a_no_draft_choice_is_retested_on_a_doubling_schedule_then_retired() {
        let mut c = DraftChooser::new(2);
        let ks = simulate(&mut c, 4, 400, [0.010, 0.050, 0.060], 0);
        let late: Vec<usize> = (12..ks.len()).filter(|&i| ks[i] > 0 && ks[i - 1] == 0).collect();
        assert_eq!(late.len(), 3, "{ks:?}");
        assert!(late[1] - late[0] < late[2] - late[1], "{late:?}");
        assert!(ks[late[2] + 40..].iter().all(|&k| k == 0));
    }

    #[test]
    fn a_retest_notices_that_verify_became_cheap() {
        let mut c = DraftChooser::new(2);
        simulate(&mut c, 4, 40, [0.010, 0.050, 0.060], 2);
        let ks = simulate(&mut c, 4, 200, [0.010, 0.011, 0.012], 2);
        assert_eq!(*ks.last().unwrap(), 2, "{ks:?}");
    }
}
