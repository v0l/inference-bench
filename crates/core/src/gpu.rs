use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuStats {
    pub name: String,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub util_percent: Option<u32>,
}

pub fn parse_nvidia_smi(output: &str) -> Vec<GpuStats> {
    output
        .lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 4 {
                return None;
            }
            let mib = |s: &str| s.parse::<u64>().ok().map(|v| v << 20);
            Some(GpuStats {
                name: f[0].to_string(),
                used_bytes: mib(f[1])?,
                total_bytes: mib(f[2])?,
                util_percent: f[3].parse().ok(),
            })
        })
        .collect()
}

pub fn sample() -> Vec<GpuStats> {
    Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_csv_is_parsed_into_bytes() {
        let out = "NVIDIA RTX PRO 6000 Blackwell, 85926, 97887, 96\nbroken line\nGPU 2, 10, 20, [N/A]\n";
        let g = parse_nvidia_smi(out);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].used_bytes, g[0].total_bytes, g[0].util_percent), (85926 << 20, 97887 << 20, Some(96)));
        assert_eq!(g[1].util_percent, None);
    }
}
