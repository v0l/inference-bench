pub const REFUSE_SPREAD: f64 = 0.10;
pub const CONTROL_BAND: f64 = 0.02;
pub const PASS_AGREEMENT: f64 = 0.03;
pub const MIN_ROUNDS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub n: usize,
}

impl Summary {
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    pub fn spread(&self) -> f64 {
        self.iqr() / self.median
    }

    pub fn refused(&self) -> bool {
        let spread = self.spread();
        spread.is_nan() || spread > REFUSE_SPREAD
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

pub fn summarize(values: &[f64]) -> Summary {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    Summary { median: quantile(&v, 0.5), q1: quantile(&v, 0.25), q3: quantile(&v, 0.75), n: v.len() }
}

pub fn ratios(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| b / a).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Refused(String),
    NoDifference,
    BFaster,
    BSlower,
}

impl Verdict {
    pub fn label(&self) -> String {
        match self {
            Verdict::Refused(why) => format!("REFUSE ({why})"),
            Verdict::NoDifference => "PASS: no difference outside the A/A band".into(),
            Verdict::BFaster => "PASS: B faster".into(),
            Verdict::BSlower => "PASS: B slower".into(),
        }
    }
}

pub fn verdict(control: &[Summary], passes: &[Summary]) -> Verdict {
    for (i, c) in control.iter().enumerate() {
        if c.refused() {
            return Verdict::Refused(format!("A/A control {} spread {:.3} above {REFUSE_SPREAD}", i + 1, c.spread()));
        }
        if (c.median - 1.0).abs() > CONTROL_BAND {
            return Verdict::Refused(format!("A/A control {} median {:.4} outside 1 +/- {CONTROL_BAND}", i + 1, c.median));
        }
    }
    for (i, p) in passes.iter().enumerate() {
        if p.n < MIN_ROUNDS {
            return Verdict::Refused(format!("pass {} has {} rounds, need {MIN_ROUNDS}", i + 1, p.n));
        }
        if p.refused() {
            return Verdict::Refused(format!("pass {} spread {:.3} above {REFUSE_SPREAD}", i + 1, p.spread()));
        }
    }
    let (lo, hi) = passes.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p.median), hi.max(p.median)));
    if (hi - lo) / lo > PASS_AGREEMENT {
        return Verdict::Refused(format!("passes disagree: medians {lo:.4} and {hi:.4}"));
    }
    if lo > 1.0 + CONTROL_BAND {
        Verdict::BFaster
    } else if hi < 1.0 - CONTROL_BAND {
        Verdict::BSlower
    } else {
        Verdict::NoDifference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(median: f64, spread: f64) -> Summary {
        Summary { median, q1: median - spread * median / 2.0, q3: median + spread * median / 2.0, n: 20 }
    }

    #[test]
    fn quantiles_interpolate() {
        let x = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!((x.median, x.q1, x.q3), (3.0, 2.0, 4.0));
        assert!((x.iqr() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ratios_are_b_over_a() {
        assert_eq!(ratios(&[2.0, 4.0], &[1.0, 8.0]), vec![0.5, 2.0]);
    }

    #[test]
    fn same_arm_passes_as_no_difference() {
        let v = verdict(&[s(1.0, 0.01), s(1.005, 0.01)], &[s(1.001, 0.02), s(0.999, 0.02)]);
        assert_eq!(v, Verdict::NoDifference);
    }

    #[test]
    fn slower_arm_is_detected() {
        let v = verdict(&[s(1.0, 0.01), s(1.0, 0.01)], &[s(0.90, 0.02), s(0.91, 0.02)]);
        assert_eq!(v, Verdict::BSlower);
        let v = verdict(&[s(1.0, 0.01), s(1.0, 0.01)], &[s(1.10, 0.02), s(1.09, 0.02)]);
        assert_eq!(v, Verdict::BFaster);
    }

    #[test]
    fn noisy_pass_is_refused() {
        let v = verdict(&[s(1.0, 0.01)], &[s(1.2, 0.15), s(1.2, 0.02)]);
        assert!(matches!(v, Verdict::Refused(_)));
    }

    #[test]
    fn bad_control_is_refused() {
        assert!(matches!(verdict(&[s(1.05, 0.01)], &[s(1.2, 0.02), s(1.2, 0.02)]), Verdict::Refused(_)));
        assert!(matches!(verdict(&[s(1.0, 0.2)], &[s(1.2, 0.02), s(1.2, 0.02)]), Verdict::Refused(_)));
    }

    #[test]
    fn disagreeing_passes_are_refused() {
        assert!(matches!(verdict(&[s(1.0, 0.01)], &[s(1.1, 0.02), s(1.2, 0.02)]), Verdict::Refused(_)));
    }

    #[test]
    fn too_few_rounds_are_refused() {
        let mut few = s(1.2, 0.02);
        few.n = 6;
        assert!(matches!(verdict(&[s(1.0, 0.01)], &[few, few]), Verdict::Refused(_)));
    }
}
