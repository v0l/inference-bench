#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

pub const MAX_TOP: u32 = 20;

pub fn token_logprob(row: &[f32], token: u32, top: u32) -> TokenLogprob {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let norm = max + sum.ln() as f32;
    let at = |i: usize| row[i] - norm;
    let n = (top.min(MAX_TOP) as usize).min(row.len());
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(n + 1);
    if n > 0 {
        for (i, &v) in row.iter().enumerate() {
            if best.len() < n || v > best[best.len() - 1].1 {
                let pos = best.partition_point(|&(_, b)| b >= v);
                best.insert(pos, (i as u32, v));
                best.truncate(n);
            }
        }
    }
    TokenLogprob {
        token,
        logprob: row.get(token as usize).map_or(f32::NEG_INFINITY, |_| at(token as usize)),
        top: best.into_iter().map(|(i, _)| (i, at(i as usize))).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_are_a_log_softmax_and_top_is_sorted() {
        let row = [0.0f32, 2.0, 1.0, -1.0];
        let lp = token_logprob(&row, 2, 2);
        let z: f32 = row.iter().map(|v| v.exp()).sum();
        assert!((lp.logprob - (1.0f32.exp() / z).ln()).abs() < 1e-5);
        assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), [1, 2]);
        assert!((lp.top[0].1 - (2.0f32.exp() / z).ln()).abs() < 1e-5);
        let total: f32 = (0..4).map(|t| token_logprob(&row, t, 0).logprob.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(token_logprob(&row, 0, 0).top.is_empty());
        assert_eq!(token_logprob(&row, 0, 99).top.len(), 4);
    }
}
