use crate::result::RunResult;

#[derive(Debug, Clone, PartialEq)]
pub struct Grid<'a> {
    pub models: Vec<String>,
    pub backends: Vec<String>,
    cells: Vec<Option<&'a RunResult>>,
}

impl<'a> Grid<'a> {
    pub fn latest(runs: &'a [RunResult]) -> Self {
        let mut models: Vec<String> = Vec::new();
        let mut backends: Vec<String> = Vec::new();
        for r in runs {
            if !models.contains(&r.model) {
                models.push(r.model.clone());
            }
            if !backends.contains(&r.backend) {
                backends.push(r.backend.clone());
            }
        }
        let mut cells = vec![None; models.len() * backends.len()];
        for r in runs {
            let m = models.iter().position(|x| *x == r.model).unwrap();
            let b = backends.iter().position(|x| *x == r.backend).unwrap();
            let slot = &mut cells[m * backends.len() + b];
            if slot.is_none_or(|old: &RunResult| old.started <= r.started) {
                *slot = Some(r);
            }
        }
        Self { models, backends, cells }
    }

    pub fn cell(&self, model: usize, backend: usize) -> Option<&'a RunResult> {
        self.cells[model * self.backends.len() + backend]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    DecodeTokS,
    PrefillTokS,
    TtftMs,
}

impl Metric {
    pub const ALL: [Metric; 3] = [Self::DecodeTokS, Self::PrefillTokS, Self::TtftMs];

    pub fn label(self) -> &'static str {
        match self {
            Self::DecodeTokS => "decode tok/s",
            Self::PrefillTokS => "prefill tok/s",
            Self::TtftMs => "time to first token ms",
        }
    }

    pub fn value(self, r: &RunResult) -> Option<f64> {
        match self {
            Self::DecodeTokS => r.decode_tok_s,
            Self::PrefillTokS => r.prefill_tok_s,
            Self::TtftMs => r.ttft_secs.map(|s| s * 1000.0),
        }
    }

    pub fn higher_is_better(self) -> bool {
        !matches!(self, Self::TtftMs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub model: String,
    pub backend: String,
    pub value: Option<f64>,
    pub fraction: f32,
}

pub fn bars(grid: &Grid, metric: Metric) -> Vec<Bar> {
    let mut out = Vec::new();
    for (m, model) in grid.models.iter().enumerate() {
        for (b, backend) in grid.backends.iter().enumerate() {
            let value = grid.cell(m, b).and_then(|r| metric.value(r));
            out.push(Bar { model: model.clone(), backend: backend.clone(), value, fraction: 0.0 });
        }
    }
    let top = out.iter().filter_map(|b| b.value).fold(0.0, f64::max);
    for bar in &mut out {
        bar.fraction = match (bar.value, top > 0.0) {
            (Some(v), true) => (v / top) as f32,
            _ => 0.0,
        };
    }
    out
}

pub fn megabytes(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / 1048576.0)
}

pub fn millis(secs: f64) -> String {
    format!("{:.0} ms", secs * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{BackendError, Generation};
    use chrono::{Duration, Utc};

    fn ok(b: &str, m: &str, at: i64, text: &str) -> RunResult {
        RunResult::ok(
            b,
            m,
            Utc::now() + Duration::seconds(at),
            Generation { text: text.into(), ..Default::default() },
        )
    }

    #[test]
    fn the_grid_keeps_the_newest_run_of_each_cell() {
        let runs = vec![
            ok("vllm", "a", 5, "new"),
            ok("vllm", "a", 1, "old"),
            RunResult::failed("llama", "b", Utc::now(), BackendError::unsupported("x")),
        ];
        let g = Grid::latest(&runs);
        assert_eq!(g.models, vec!["a", "b"]);
        assert_eq!(g.backends, vec!["vllm", "llama"]);
        assert_eq!(g.cell(0, 0).unwrap().text, "new");
        assert!(g.cell(0, 1).is_none());
        assert!(g.cell(1, 1).unwrap().error.is_some());
        assert!(g.cell(1, 0).is_none());
    }

    #[test]
    fn bars_scale_to_the_best_cell_and_skip_failures() {
        let mut fast = ok("vllm", "a", 1, "");
        fast.decode_tok_s = Some(100.0);
        let mut slow = ok("llama", "a", 1, "");
        slow.decode_tok_s = Some(25.0);
        let dead = RunResult::failed("spec", "a", Utc::now(), BackendError::failed("x"));
        let runs = vec![fast, slow, dead];
        let b = bars(&Grid::latest(&runs), Metric::DecodeTokS);
        assert_eq!(b.iter().map(|x| x.fraction).collect::<Vec<_>>(), vec![1.0, 0.25, 0.0]);
        assert_eq!(b[2].value, None);
    }

    #[test]
    fn ttft_is_shown_in_milliseconds() {
        let mut r = ok("v", "a", 1, "");
        r.ttft_secs = Some(0.25);
        assert_eq!(Metric::TtftMs.value(&r), Some(250.0));
        assert!(!Metric::TtftMs.higher_is_better());
    }

    #[test]
    fn units_are_readable() {
        assert_eq!(megabytes(3 * 1048576), "3 MiB");
        assert_eq!(millis(0.1234), "123 ms");
    }
}
