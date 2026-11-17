use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuStats {
    pub name: String,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub util_percent: Option<u32>,
    pub uuid: String,
    pub own_bytes: u64,
}

impl GpuStats {
    pub fn other_bytes(&self) -> u64 {
        self.used_bytes.saturating_sub(self.own_bytes)
    }

    pub fn available_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.other_bytes())
    }
}

pub fn used_total(stats: &[GpuStats]) -> Option<(u64, u64)> {
    (!stats.is_empty()).then(|| (stats.iter().map(|g| g.used_bytes).sum(), stats.iter().map(|g| g.total_bytes).sum()))
}

pub fn mean_util(stats: &[GpuStats]) -> Option<u32> {
    let utils: Vec<u32> = stats.iter().filter_map(|g| g.util_percent).collect();
    (!utils.is_empty()).then(|| utils.iter().sum::<u32>() / utils.len() as u32)
}

pub fn parse_compute_apps(output: &str) -> Vec<(String, u32, u64)> {
    output
        .lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 3 {
                return None;
            }
            Some((f[0].to_string(), f[1].parse().ok()?, f[2].parse::<u64>().ok()? << 20))
        })
        .collect()
}

pub fn attribute(stats: &mut [GpuStats], apps: &[(String, u32, u64)], own_pids: &[u32]) {
    for g in stats.iter_mut() {
        g.own_bytes = apps.iter().filter(|(u, p, _)| *u == g.uuid && own_pids.contains(p)).map(|a| a.2).sum();
    }
}

pub fn own_pids() -> Vec<u32> {
    let me = std::process::id();
    let mut pids = vec![me];
    let Ok(dir) = std::fs::read_dir("/proc") else { return pids };
    let mut parents: Vec<(u32, u32)> = Vec::new();
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else { continue };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        let after = stat.rsplit_once(')').map(|r| r.1).unwrap_or("");
        if let Some(ppid) = after.split_whitespace().nth(1).and_then(|p| p.parse().ok()) {
            parents.push((pid, ppid));
        }
    }
    loop {
        let before = pids.len();
        for &(pid, ppid) in &parents {
            if pids.contains(&ppid) && !pids.contains(&pid) {
                pids.push(pid);
            }
        }
        if pids.len() == before {
            return pids;
        }
    }
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
                uuid: f.get(4).map(|u| u.to_string()).unwrap_or_default(),
                own_bytes: 0,
            })
        })
        .collect()
}

pub fn sample() -> Vec<GpuStats> {
    let mut stats = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default();
    let apps = Command::new("nvidia-smi")
        .args(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_compute_apps(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default();
    attribute(&mut stats, &apps, &own_pids());
    stats
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

    #[test]
    fn used_memory_and_utilisation_cover_every_gpu() {
        let g = parse_nvidia_smi("RTX, 1000, 96000, 90\nRTX, 3000, 96000, 10\n");
        assert_eq!(used_total(&g), Some((4000 << 20, 192000 << 20)));
        assert_eq!(mean_util(&g), Some(50));
        assert_eq!(used_total(&[]), None);
    }

    #[test]
    fn this_apps_processes_are_not_counted_against_the_budget() {
        let mut g = parse_nvidia_smi("RTX, 87000, 97000, 90, GPU-aaa\nRTX, 1000, 97000, 0, GPU-bbb\n");
        let apps = parse_compute_apps("GPU-aaa, 100, 60000\nGPU-aaa, 200, 14000\nGPU-aaa, 300, 12000\nGPU-bbb, 100, 500\nbad\n");
        attribute(&mut g, &apps, &[100, 101]);
        assert_eq!((g[0].own_bytes, g[0].other_bytes()), (60000 << 20, 27000 << 20));
        assert_eq!(g[0].available_bytes(), 70000 << 20);
        assert_eq!((g[1].own_bytes, g[1].available_bytes()), (500 << 20, 96500 << 20));
    }
}
