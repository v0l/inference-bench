use crate::spec::TuneMode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const MAX_DISPERSION: f64 = 0.10;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeKey {
    pub kernel: String,
    pub rows: usize,
    pub width: usize,
    pub tokens: usize,
}

impl ShapeKey {
    pub fn new(kernel: &str, rows: usize, width: usize, tokens: usize) -> ShapeKey {
        ShapeKey { kernel: kernel.to_string(), rows, width, tokens }
    }

    fn label(&self) -> String {
        format!("{}:{}x{}:t{}", self.kernel, self.rows, self.width, self.tokens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub module_hash: String,
    pub gpu: String,
    pub driver: String,
}

impl Identity {
    pub fn file_name(&self) -> String {
        let mut h = Sha256::new();
        for part in [&self.module_hash, &self.gpu, &self.driver] {
            h.update(part.as_bytes());
            h.update([0]);
        }
        let hex: String = h.finalize().iter().take(12).map(|b| format!("{b:02x}")).collect();
        format!("tune-{hex}.json")
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Stored {
    module_hash: String,
    gpu: String,
    driver: String,
    choices: BTreeMap<String, usize>,
}

pub struct Tuner {
    mode: TuneMode,
    path: PathBuf,
    identity: Identity,
    choices: BTreeMap<String, usize>,
}

pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub fn dispersion(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n < 4 {
        return f64::INFINITY;
    }
    let m = median(&v);
    if m <= 0.0 {
        return f64::INFINITY;
    }
    let q1 = median(&v[..n / 2]);
    let q3 = median(&v[n.div_ceil(2)..]);
    (q3 - q1) / m
}

impl Tuner {
    pub fn open(mode: TuneMode, dir: &Path, identity: Identity) -> Tuner {
        let path = dir.join(identity.file_name());
        let choices = if mode == TuneMode::Off { BTreeMap::new() } else { Self::read(&path, &identity) };
        Tuner { mode, path, identity, choices }
    }

    fn read(path: &Path, identity: &Identity) -> BTreeMap<String, usize> {
        let Ok(text) = std::fs::read_to_string(path) else { return BTreeMap::new() };
        match serde_json::from_str::<Stored>(&text) {
            Ok(s) if s.module_hash == identity.module_hash && s.gpu == identity.gpu && s.driver == identity.driver => s.choices,
            _ => BTreeMap::new(),
        }
    }

    pub fn mode(&self) -> TuneMode {
        self.mode
    }

    pub fn cached(&self, key: &ShapeKey) -> Option<usize> {
        self.choices.get(&key.label()).copied()
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let stored = Stored {
            module_hash: self.identity.module_hash.clone(),
            gpu: self.identity.gpu.clone(),
            driver: self.identity.driver.clone(),
            choices: self.choices.clone(),
        };
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&stored).map_err(std::io::Error::other)?)?;
        std::fs::rename(&tmp, &self.path)
    }

    pub fn choose(&mut self, key: &ShapeKey, formula: usize, candidates: &[usize], mut measure: impl FnMut(usize) -> Vec<f64>) -> usize {
        if self.mode == TuneMode::Off {
            return formula;
        }
        if self.mode == TuneMode::Cached {
            if let Some(c) = self.cached(key).filter(|c| candidates.contains(c)) {
                return c;
            }
        }
        self.measure(key, formula, candidates, &mut measure)
    }

    fn measure(&mut self, key: &ShapeKey, formula: usize, candidates: &[usize], measure: &mut impl FnMut(usize) -> Vec<f64>) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for &c in candidates {
            let times = measure(c);
            if dispersion(&times) > MAX_DISPERSION {
                return formula;
            }
            let m = median(&times);
            if best.is_none_or(|(_, b)| m < b) {
                best = Some((c, m));
            }
        }
        let Some((winner, _)) = best else { return formula };
        self.choices.insert(key.label(), winner);
        winner
    }
}

pub fn candidate_splits(formula: usize, most: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [formula / 2, formula, formula * 2, formula + 1, formula.saturating_sub(1)]
        .into_iter()
        .filter(|&s| s >= 1 && s <= most.max(1))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity { module_hash: "m".into(), gpu: "gpu".into(), driver: "d".into() }
    }

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("specrt-tune-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn steady(base: f64) -> Vec<f64> {
        (0..8).map(|i| base + i as f64 * 0.001).collect()
    }

    #[test]
    fn cached_choice_is_reused() {
        let d = dir("reuse");
        let key = ShapeKey::new("gate_up_f4b", 17408, 5120, 4);
        let mut t = Tuner::open(TuneMode::Cached, &d, identity());
        let won = t.choose(&key, 4, &[2, 4, 8], |s| steady(if s == 8 { 1.0 } else { 2.0 }));
        assert_eq!(won, 8);
        t.save().unwrap();
        let mut again = Tuner::open(TuneMode::Cached, &d, identity());
        let got = again.choose(&key, 4, &[2, 4, 8], |_| panic!("measured again"));
        assert_eq!(got, 8);
        let mut forced = Tuner::open(TuneMode::Force, &d, identity());
        assert_eq!(forced.choose(&key, 4, &[2, 4, 8], |s| steady(if s == 2 { 1.0 } else { 2.0 })), 2);
        let mut off = Tuner::open(TuneMode::Off, &d, identity());
        assert_eq!(off.choose(&key, 4, &[2, 4, 8], |_| panic!("measured")), 4);
    }

    #[test]
    fn malformed_entry_is_measured_again() {
        let d = dir("malformed");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(identity().file_name()), "{ not json").unwrap();
        let key = ShapeKey::new("down_f4b", 5120, 17408, 8);
        let mut t = Tuner::open(TuneMode::Cached, &d, identity());
        assert!(t.is_empty());
        assert_eq!(t.choose(&key, 4, &[2, 4], |s| steady(if s == 2 { 1.0 } else { 2.0 })), 2);
        t.save().unwrap();
        let t = Tuner::open(TuneMode::Cached, &d, identity());
        assert_eq!(t.cached(&key), Some(2));
    }

    #[test]
    fn cached_value_outside_candidates_is_measured_again() {
        let d = dir("stale");
        let key = ShapeKey::new("k", 1, 1, 1);
        let mut t = Tuner::open(TuneMode::Cached, &d, identity());
        t.choose(&key, 4, &[8], |_| steady(1.0));
        t.save().unwrap();
        let mut t = Tuner::open(TuneMode::Cached, &d, identity());
        assert_eq!(t.choose(&key, 4, &[2, 4], |s| steady(s as f64)), 2);
    }

    #[test]
    fn other_gpu_or_driver_does_not_reuse() {
        let d = dir("other");
        let key = ShapeKey::new("k", 1, 1, 1);
        let mut t = Tuner::open(TuneMode::Cached, &d, identity());
        t.choose(&key, 4, &[2, 4], |s| steady(s as f64));
        t.save().unwrap();
        let other = Identity { driver: "e".into(), ..identity() };
        assert!(Tuner::open(TuneMode::Cached, &d, other).is_empty());
    }

    #[test]
    fn dispersed_measurement_keeps_the_formula() {
        let d = dir("noisy");
        let key = ShapeKey::new("k", 1, 1, 1);
        let mut t = Tuner::open(TuneMode::Force, &d, identity());
        let noisy = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.5, 1.0, 4.0];
        assert_eq!(t.choose(&key, 4, &[2, 4], |_| noisy.clone()), 4);
        assert!(t.is_empty());
    }

    #[test]
    fn candidates_stay_in_range() {
        assert_eq!(candidate_splits(4, 6), vec![2, 3, 4, 5]);
        assert_eq!(candidate_splits(1, 1), vec![1]);
    }
}
