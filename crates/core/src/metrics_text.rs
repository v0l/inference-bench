use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Sample {
    pub at: f64,
    pub preemptions: Option<f64>,
    pub kv_use: Option<f64>,
    pub running: Option<f64>,
    pub waiting: Option<f64>,
}

fn value(text: &str, names: &[&str]) -> Option<f64> {
    let mut total: Option<f64> = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (key, v) = match line.rsplit_once(' ') {
            Some(kv) => kv,
            None => continue,
        };
        let name = key.split('{').next().unwrap_or(key);
        if names.contains(&name)
            && let Ok(v) = v.trim().parse::<f64>()
        {
            *total.get_or_insert(0.0) += v;
        }
    }
    total
}

pub fn scrape(text: &str, at: f64) -> Sample {
    let kv_use = match (value(text, &["inference_kv_blocks_used"]), value(text, &["inference_kv_blocks_total"])) {
        (Some(u), Some(t)) if t > 0.0 => Some(u / t),
        _ => value(text, &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]),
    };
    Sample {
        at,
        preemptions: value(text, &["inference_preemptions_total", "vllm:num_preemptions_total"]),
        kv_use,
        running: value(text, &["inference_running_seqs", "vllm:num_requests_running"]),
        waiting: value(text, &["inference_waiting_seqs", "vllm:num_requests_waiting"]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specrt_and_vllm_metrics_read_the_same() {
        let ours = "inference_kv_blocks_used 25\ninference_kv_blocks_total 100\ninference_running_seqs 4\ninference_waiting_seqs 1\ninference_preemptions_total 2\n";
        let s = scrape(ours, 1.0);
        assert_eq!((s.kv_use, s.running, s.waiting, s.preemptions), (Some(0.25), Some(4.0), Some(1.0), Some(2.0)));
        let vllm = "# HELP x\nvllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.37\nvllm:num_requests_running{engine=\"0\"} 9.0\nvllm:num_preemptions_total{engine=\"0\"} 3.0\n";
        let s = scrape(vllm, 0.0);
        assert_eq!((s.kv_use, s.running, s.preemptions, s.waiting), (Some(0.37), Some(9.0), Some(3.0), None));
    }
}
