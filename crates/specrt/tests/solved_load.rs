use specrt::options::Profile;
use specrt::runtime::Sampling;
use specrt::serve::{ServeOptions, Serving};
use specrt::solve::{solve, PlacementRequest, Target};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn gpu_used_mib() -> Vec<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse().ok()).collect()
}

#[test]
#[ignore = "needs the orcarouter/Qwen3.8-27B-Uncensored-NVFP4 checkpoint and two idle GPUs; SOLVE_MODE=max|low"]
fn the_engine_runs_exactly_the_solved_plan() {
    tracing_subscriber::fmt().with_env_filter("specrt=info").with_test_writer().init();
    let model = std::env::var("SOLVE_MODEL").unwrap_or_else(|_| "orcarouter/Qwen3.8-27B-Uncensored-NVFP4".into());
    let low = std::env::var("SOLVE_MODE").is_ok_and(|m| m == "low");
    let mode = if low { Profile::LowMemory } else { Profile::MaxPerformance };
    let context: usize = std::env::var("SOLVE_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);
    let memory = std::env::var("SOLVE_MEMORY_GIB").ok().and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as u64);

    let spec = specrt::engine::resolve_spec(&model).unwrap();
    let machine = specrt::hardware::probe().unwrap();
    let devices = machine.devices(&[]);
    for d in &devices {
        println!("DEVICE {} {:.1} GiB", d.name, d.memory_gib);
    }
    assert!(devices.len() >= 2, "{} GPUs", devices.len());
    let target = Target { context: Some(context), memory, slots: None };
    let solved = solve(&PlacementRequest { mode, target, pins: Default::default(), devices: &devices, spec: &spec }).unwrap();
    println!("SOLVED {}", solved.log_line());
    println!("STAGES {}", solved.placement.stages.iter().map(|s| format!("{}..{} tp{} gpus{:?}", s.start, s.end, s.tp, s.ordinals)).collect::<Vec<_>>().join(" | "));
    if low {
        assert_eq!((solved.tp, solved.placement.stages.len()), (1, 1), "{}", solved.strategy);
    } else {
        assert_eq!(solved.tp, 2, "{}", solved.strategy);
    }

    let opts = ServeOptions::from_solved(&solved, ServeOptions::default()).unwrap();
    let serving = Serving::load(&model, opts).unwrap();
    assert_eq!(serving.max_model_len(), solved.context);
    let stats = serving.stats();
    println!("STATS kv_blocks_total {} block_tokens {} block_bytes {}", stats.kv_blocks_total, stats.kv_block_tokens, stats.kv_block_bytes);
    let prompt = serving.tokens("The capital of France is").unwrap();
    let mut text = String::new();
    let done = serving
        .complete(prompt, 24, Sampling::GREEDY, false, String::new(), Arc::new(AtomicBool::new(false)), &mut |t| {
            text.push_str(t);
            true
        })
        .unwrap();
    println!("COMPLETION {:?} tokens {}", done.text, done.completion_tokens);
    let used = gpu_used_mib();
    println!("GPU_USED_MIB {used:?}");
    assert!(used[0] < 90 * 1024, "GPU0 holds {} MiB", used[0]);
    if low {
        assert!(used.iter().filter(|m| **m > 4096).count() == 1, "{used:?}");
    } else {
        assert!(used.iter().filter(|m| **m > 4096).count() == 2, "{used:?}");
    }
}
