use crate::target::{Target, Logits};
use crate::result::BackendError;
use serde::{Deserialize, Serialize};

pub const FIXED_PROMPT: &str = "The spec-driven runtime reads a TOML file describing the model, renders CUDA templates for every operator, and compiles them at load time. A paged key value cache keeps the attention state between decode steps, so";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitDiff {
    pub rows: usize,
    pub vocab: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub top1_agreement: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub top1: f32,
}

impl Tolerance {
    pub const BF16: Self = Self { max_abs: 1e-3, mean_abs: 1e-4, top1: 1.0 };
    pub const QUANTISED: Self = Self { max_abs: f32::INFINITY, mean_abs: 0.1, top1: 0.95 };
}

impl LogitDiff {
    pub fn within(&self, t: Tolerance) -> bool {
        self.max_abs <= t.max_abs && self.mean_abs <= t.mean_abs && self.top1_agreement >= t.top1
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

pub fn diff(ours: &Logits, reference: &Logits) -> Result<LogitDiff, BackendError> {
    if ours.tokens != reference.tokens {
        return Err(BackendError::failed(format!(
            "the backends tokenised the prompt differently: {} tokens against {}",
            ours.tokens.len(),
            reference.tokens.len()
        )));
    }
    if ours.vocab != reference.vocab || ours.values.len() != reference.values.len() {
        return Err(BackendError::failed(format!(
            "logit shapes differ: vocab {} against {}",
            ours.vocab, reference.vocab
        )));
    }
    let mut max_abs = 0f32;
    let mut sum = 0f64;
    for (a, b) in ours.values.iter().zip(&reference.values) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum += d as f64;
    }
    let rows = ours.rows();
    let agree = (0..rows).filter(|&r| argmax(ours.row(r)) == argmax(reference.row(r))).count();
    Ok(LogitDiff {
        rows,
        vocab: ours.vocab,
        max_abs,
        mean_abs: (sum / ours.values.len().max(1) as f64) as f32,
        top1_agreement: agree as f32 / rows.max(1) as f32,
    })
}

pub fn check(target: &mut dyn Target, reference: &Logits, prompt: &str) -> Result<LogitDiff, BackendError> {
    diff(&target.logits(prompt)?, reference)
}

pub fn read_reference(json: &std::path::Path) -> Result<Logits, BackendError> {
    let text = std::fs::read_to_string(json)
        .map_err(|e| BackendError::failed(format!("{}: {e}", json.display())))?;
    let meta: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| BackendError::failed(format!("{}: {e}", json.display())))?;
    let tokens: Vec<u32> = meta["tokens"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
        .unwrap_or_default();
    let vocab = meta["vocab"].as_u64().unwrap_or(0) as usize;
    let name = json.to_string_lossy();
    let bin = std::path::PathBuf::from(format!("{}.bin", name.strip_suffix(".json").unwrap_or(&name)));
    Logits::read(&bin, tokens, vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(values: Vec<f32>) -> Logits {
        Logits { tokens: vec![1, 2], vocab: 3, values }
    }

    #[test]
    fn identical_logits_have_no_difference() {
        let a = logits(vec![1., 2., 3., 6., 5., 4.]);
        let d = diff(&a, &a).unwrap();
        assert_eq!((d.max_abs, d.mean_abs, d.top1_agreement), (0.0, 0.0, 1.0));
        assert!(d.within(Tolerance::BF16));
    }

    #[test]
    fn a_flipped_argmax_lowers_top1_and_fails_the_tolerance() {
        let a = logits(vec![1., 2., 3., 6., 5., 4.]);
        let b = logits(vec![1., 2., 3., 4., 5., 6.]);
        let d = diff(&a, &b).unwrap();
        assert_eq!(d.top1_agreement, 0.5);
        assert_eq!(d.max_abs, 2.0);
        assert!(!d.within(Tolerance::QUANTISED));
    }

    #[test]
    fn different_tokenisation_is_an_error() {
        let a = logits(vec![0.; 6]);
        let b = Logits { tokens: vec![1, 3], ..a.clone() };
        assert!(diff(&a, &b).unwrap_err().message.contains("tokenised"));
    }

    #[test]
    fn check_asks_the_target_for_the_prompt_and_compares_with_the_reference() {
        struct Fixed(Logits);
        impl Target for Fixed {
            fn name(&self) -> &str {
                "fixed"
            }
            fn capabilities(&self) -> crate::target::Capabilities {
                Default::default()
            }
            fn load(&mut self, _: &str) -> Result<(), BackendError> {
                Ok(())
            }
            fn generate(
                &mut self,
                _: &crate::target::Request,
                _: &mut dyn FnMut(&str),
            ) -> Result<crate::result::Generation, BackendError> {
                unreachable!()
            }
            fn logits(&mut self, prompt: &str) -> Result<Logits, BackendError> {
                assert_eq!(prompt, FIXED_PROMPT);
                Ok(self.0.clone())
            }
        }
        let l = logits(vec![1., 2., 3., 6., 5., 4.]);
        let d = check(&mut Fixed(l.clone()), &l, FIXED_PROMPT).unwrap();
        assert_eq!(d.rows, 2);
    }

    #[test]
    fn a_reference_pair_from_reference_py_reads_back() {
        let dir = std::env::temp_dir().join(format!("ib-ref-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tiny.ref.json"), r#"{"tokens": [5, 6], "vocab": 2, "greedy": []}"#).unwrap();
        let bytes: Vec<u8> = [1f32, 2., 3., 4.].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("tiny.ref.bin"), bytes).unwrap();
        let l = read_reference(&dir.join("tiny.ref.json")).unwrap();
        assert_eq!((l.tokens.clone(), l.row(1)), (vec![5, 6], &[3.0, 4.0][..]));
    }
}
