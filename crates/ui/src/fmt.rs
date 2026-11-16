pub fn duration(secs: f64) -> String {
    let ms = (secs * 1000.0).round();
    if ms < 1000.0 {
        return format!("{ms:.0} ms");
    }
    if secs < 9.995 {
        return format!("{secs:.2} s");
    }
    if secs < 59.95 {
        return format!("{secs:.1} s");
    }
    let total = secs.round() as u64;
    format!("{}m {:02}s", total / 60, total % 60)
}

fn three_figures(v: f64) -> String {
    match v {
        v if v < 9.995 => format!("{v:.2}"),
        v if v < 99.95 => format!("{v:.1}"),
        v => format!("{v:.0}"),
    }
}

pub fn count(n: u64) -> String {
    let v = n as f64;
    match n {
        0..=999 => n.to_string(),
        1_000..=999_499 => format!("{}k", three_figures(v / 1e3)),
        999_500..=999_499_999 => format!("{}M", three_figures(v / 1e6)),
        _ => format!("{}B", three_figures(v / 1e9)),
    }
}

pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn rate(per_sec: f64) -> String {
    if per_sec < 99.95 {
        format!("{:.1} tok/s", per_sec.max(0.0))
    } else {
        format!("{} tok/s", thousands(per_sec.round() as u64))
    }
}

pub fn percent(fraction: f64) -> String {
    format!("{:.0}%", fraction * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_pick_a_readable_unit() {
        assert_eq!(duration(0.842), "842 ms");
        assert_eq!(duration(1.308), "1.31 s");
        assert_eq!(duration(5.6), "5.60 s");
        assert_eq!(duration(12.43), "12.4 s");
        assert_eq!(duration(65.0), "1m 05s");
        assert_eq!(duration(0.9996), "1.00 s");
        assert_eq!(duration(0.0), "0 ms");
    }

    #[test]
    fn token_counts_use_three_figures() {
        assert_eq!(count(625), "625");
        assert_eq!(count(1450), "1.45k");
        assert_eq!(count(14_500), "14.5k");
        assert_eq!(count(145_000), "145k");
        assert_eq!(count(1_200_000), "1.20M");
        assert_eq!(count(999_900), "1.00M");
        assert_eq!(count(7_000_000_000), "7.00B");
    }

    #[test]
    fn rates_get_a_thousands_separator() {
        assert_eq!(rate(35.44), "35.4 tok/s");
        assert_eq!(rate(2310.4), "2,310 tok/s");
        assert_eq!(rate(100.0), "100 tok/s");
        assert_eq!(rate(1_234_567.0), "1,234,567 tok/s");
    }

    #[test]
    fn percent_rounds_to_whole_numbers() {
        assert_eq!(percent(0.8), "80%");
    }
}
