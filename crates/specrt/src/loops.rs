use std::collections::HashMap;

pub const SPAN: usize = 40;
pub const REPEATS: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct LoopReport {
    pub repeat_ratio: f64,
    pub worst_repeats: usize,
    pub looping: bool,
}

pub fn detect(tokens: &[u32]) -> LoopReport {
    detect_span(tokens, SPAN, REPEATS)
}

pub fn detect_span(tokens: &[u32], span: usize, repeats: usize) -> LoopReport {
    if tokens.len() < span {
        return LoopReport { repeat_ratio: 0.0, worst_repeats: 0, looping: false };
    }
    let mut seen: HashMap<&[u32], usize> = HashMap::new();
    let mut again = 0usize;
    let mut worst = 0usize;
    for window in tokens.windows(span) {
        let n = seen.entry(window).or_insert(0);
        *n += 1;
        if *n > 1 {
            again += 1;
        }
        worst = worst.max(*n);
    }
    LoopReport {
        repeat_ratio: again as f64 / (tokens.len() - span + 1) as f64,
        worst_repeats: worst,
        looping: worst >= repeats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varied(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i.wrapping_mul(2654435761) >> 7).collect()
    }

    #[test]
    fn varied_text_is_not_a_loop() {
        let r = detect(&varied(2000));
        assert!(!r.looping);
        assert_eq!(r.repeat_ratio, 0.0);
    }

    #[test]
    fn a_paragraph_repeated_three_times_is_a_loop() {
        let para = varied(60);
        let mut t: Vec<u32> = varied(300).iter().map(|x| x + 7).collect();
        for _ in 0..3 {
            t.extend(&para);
            t.extend(varied(20).iter().map(|x| x + 1));
        }
        let r = detect(&t);
        assert!(r.looping);
        assert!(r.worst_repeats >= 3);
        assert!(r.repeat_ratio > 0.2);
    }

    #[test]
    fn two_repeats_are_not_enough() {
        let para = varied(60);
        let mut t: Vec<u32> = varied(300).iter().map(|x| x + 7).collect();
        t.extend(&para);
        t.extend(varied(50).iter().map(|x| x + 1));
        t.extend(&para);
        assert!(!detect(&t).looping);
    }

    #[test]
    fn a_short_repeat_is_ignored() {
        let mut t = Vec::new();
        for i in 0..5u32 {
            t.extend(varied(30));
            t.extend((0..20).map(|j| 9000 + i * 100 + j));
        }
        assert!(!detect(&t).looping);
    }
}
