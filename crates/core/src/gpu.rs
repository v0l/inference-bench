use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuStats {
    pub name: String,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub util_percent: Option<u32>,
    pub uuid: String,
    pub own_bytes: u64,
    pub display: bool,
    pub index: Option<u32>,
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

const COMPOSITORS: [&str; 14] = ["hyprland", "sway", "xwayland", "xorg", "x", "kwin_wayland", "kwin_x11", "gnome-shell", "mutter", "weston", "labwc", "river", "niri", "wayfire"];

pub fn parse_compositor_gpus(pmon: &str) -> Vec<u32> {
    let mut gpus: Vec<u32> = pmon
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let (gpu, kind, name) = (f.first()?.parse::<u32>().ok()?, f.get(2)?, f.last()?);
            (kind.contains('G') && COMPOSITORS.contains(&name.to_ascii_lowercase().as_str())).then_some(gpu)
        })
        .collect();
    gpus.sort_unstable();
    gpus.dedup();
    gpus
}

pub fn attribute(stats: &mut [GpuStats], apps: &[(String, u32, u64)], own_pids: &[u32]) {
    for g in stats.iter_mut() {
        g.own_bytes = apps.iter().filter(|(u, p, _)| *u == g.uuid && own_pids.contains(p)).map(|a| a.2).sum();
    }
}

pub const LANDED_MIN_BYTES: u64 = 256 << 20;

pub fn landed(before: &[GpuStats], after: &[GpuStats]) -> Vec<(String, u64)> {
    after
        .iter()
        .filter(|a| !a.uuid.is_empty())
        .filter_map(|a| {
            let was = before.iter().find(|b| b.uuid == a.uuid).map_or(0, |b| b.own_bytes);
            let grew = a.own_bytes.saturating_sub(was);
            (grew >= LANDED_MIN_BYTES).then(|| (a.uuid.clone(), grew))
        })
        .collect()
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
                display: f.get(5).is_some_and(|d| d.eq_ignore_ascii_case("enabled")),
                index: f.get(6).and_then(|i| i.parse().ok()),
            })
        })
        .collect()
}

pub fn sample() -> Vec<GpuStats> {
    let mut stats = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,uuid,display_active,index",
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
    let compositors = Command::new("nvidia-smi")
        .args(["pmon", "-c", "1"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_compositor_gpus(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default();
    for g in stats.iter_mut() {
        g.display |= g.index.is_some_and(|i| compositors.contains(&i));
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landed_names_only_the_gpus_whose_own_memory_grew() {
        let g = |uuid: &str, own: u64| GpuStats { name: "card".into(), used_bytes: own, total_bytes: 96 << 30, util_percent: None, uuid: uuid.into(), own_bytes: own, display: false, index: None };
        let before = [g("GPU-a", 0), g("GPU-b", 1 << 30)];
        let after = [g("GPU-a", 10 << 20), g("GPU-b", 40 << 30)];
        assert_eq!(landed(&before, &after), vec![("GPU-b".to_string(), 39 << 30)]);
        assert!(landed(&after, &after).is_empty());
    }

    #[test]
    fn nvidia_smi_csv_is_parsed_into_bytes() {
        let out = "NVIDIA RTX PRO 6000 Blackwell, 85926, 97887, 96\nbroken line\nGPU 2, 10, 20, [N/A]\n";
        let g = parse_nvidia_smi(out);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].used_bytes, g[0].total_bytes, g[0].util_percent), (85926 << 20, 97887 << 20, Some(96)));
        assert_eq!(g[1].util_percent, None);
        assert!(!g[0].display);
    }

    #[test]
    fn the_gpu_driving_a_display_is_marked() {
        let g = parse_nvidia_smi("RTX, 1, 96000, 0, GPU-a, Enabled\nRTX, 1, 96000, 0, GPU-b, Disabled\n");
        assert_eq!((g[0].display, g[1].display), (true, false));
    }

    #[test]
    fn the_gpu_holding_the_compositor_is_found_when_display_active_is_disabled() {
        let pmon = "# gpu pid type sm mem enc dec jpg ofa command\n# Idx # C/G % % % % % % name\n    0 451526 G - - - - - - alacritty\n    0 3711476 G - - - - - - Hyprland\n    0 3711554 G - - - - - - Xwayland\n    1 4113726 C 22 9 - - - - kg-trainer\n";
        assert_eq!(parse_compositor_gpus(pmon), vec![0]);
        assert!(parse_compositor_gpus("    1 5 C - - - - - - Hyprland\n").is_empty());
        let g = parse_nvidia_smi("RTX, 1, 96000, 0, GPU-a, Disabled, 0\nRTX, 1, 96000, 0, GPU-b, Disabled, 1\n");
        assert_eq!((g[0].index, g[1].index), (Some(0), Some(1)));
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
