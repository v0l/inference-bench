use serde::Serialize;
use serde_json::{Value, json};
use specrt::hardware::{GpuBackend, Machine};
use specrt::spec::{ActQuant, KvDtype, Spec};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Check {
        Check { name: name.into(), status: Status::Ok, detail: detail.into(), next: None }
    }

    fn warn(name: &str, detail: impl Into<String>, next: impl Into<String>) -> Check {
        Check { name: name.into(), status: Status::Warn, detail: detail.into(), next: Some(next.into()) }
    }

    fn fail(name: &str, detail: impl Into<String>, next: impl Into<String>) -> Check {
        Check { name: name.into(), status: Status::Fail, detail: detail.into(), next: Some(next.into()) }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ModelFeatures {
    pub quant: String,
    pub head_quant: String,
    pub act_quant: String,
    pub mlp_act_quant: String,
    pub mtp: Option<MtpFeature>,
    pub max_context: usize,
    pub kv_dtype: String,
    pub vision: bool,
    pub weights_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub state_bytes_per_seq: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MtpFeature {
    pub layers: usize,
    pub draft_tokens: usize,
    pub speculate: bool,
}

fn act_name(a: ActQuant) -> &'static str {
    match a {
        ActQuant::None => "none",
        ActQuant::Fp8 => "fp8",
        ActQuant::Fp4 => "fp4",
    }
}

fn kv_name(k: KvDtype) -> &'static str {
    match k {
        KvDtype::F16 => "f16",
        KvDtype::Fp8 => "fp8",
    }
}

fn quant_name(q: specrt::spec::Quant) -> String {
    serde_json::to_value(q).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_else(|| format!("{q:?}").to_lowercase())
}

fn config_context(spec: &Spec) -> Option<usize> {
    let path = specrt::weights::find_in_snapshot(&spec.repo, "config.json").ok()?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let text = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    text["max_position_embeddings"].as_u64().map(|c| c as usize)
}

pub fn features_of(spec: &Spec) -> ModelFeatures {
    let mtp_layers = spec.mtp.as_ref().map_or(0, |m| m.layers);
    let anatomy = specrt::anatomy::ModelAnatomy::new(spec, mtp_layers, config_context(spec));
    ModelFeatures {
        quant: quant_name(spec.quant),
        head_quant: quant_name(spec.lm_head_quant()),
        act_quant: act_name(spec.act_quant).to_string(),
        mlp_act_quant: act_name(spec.mlp.act_quant).to_string(),
        mtp: spec.mtp.as_ref().map(|m| MtpFeature { layers: m.layers, draft_tokens: m.draft_tokens, speculate: m.speculate }),
        max_context: anatomy.max_context,
        kv_dtype: kv_name(spec.runtime.kv_dtype).to_string(),
        vision: spec.vision.is_some(),
        weights_bytes: anatomy.total_bytes,
        kv_bytes_per_token: anatomy.kv_bytes_per_token,
        state_bytes_per_seq: anatomy.state_bytes_per_seq,
    }
}

pub fn features(model: &str) -> Result<(Spec, ModelFeatures), String> {
    let spec = specrt::engine::resolve_spec(model).map_err(|e| format!("{e:#}"))?;
    let f = features_of(&spec);
    Ok((spec, f))
}

fn weight_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            weight_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "safetensors" || x == "gguf") {
            out.push(p);
        }
    }
}

pub fn weights_on_disk(repo: &str, hub: &Path) -> Result<u64, String> {
    let direct = Path::new(repo);
    let mut files = Vec::new();
    if direct.is_file() {
        files.push(direct.to_path_buf());
    } else if direct.is_dir() {
        weight_files(direct, &mut files);
    } else {
        let snapshots = hub.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
        weight_files(&snapshots, &mut files);
    }
    if files.is_empty() {
        return Err(format!("no weight files for {repo}"));
    }
    Ok(files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum())
}

pub struct Environment {
    pub machine: Result<Machine, String>,
    pub driver: Option<(u32, u32)>,
    pub nvrtc: Option<(u32, u32)>,
    pub cache_dir: PathBuf,
    pub hub_dir: PathBuf,
}

impl Environment {
    pub fn detect() -> Environment {
        Environment {
            machine: specrt::hardware::probe().map_err(|e| format!("{e:#}")),
            driver: specrt::hardware::driver_version(),
            nvrtc: specrt::hardware::nvrtc_version(),
            cache_dir: specrt::jit::default_cache_dir(),
            hub_dir: specrt::weights::hub_dir(),
        }
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / crate::fit::GB)
}

fn version(v: Option<(u32, u32)>) -> String {
    v.map_or_else(|| "none".to_string(), |(a, b)| format!("{a}.{b}"))
}

fn host_checks(env: &Environment, checks: &mut Vec<Check>) {
    let machine = match &env.machine {
        Ok(m) if !m.gpus.is_empty() => m,
        Ok(_) => {
            checks.push(Check::fail("gpu", "no GPU found", "Install the NVIDIA driver (CUDA 13 capable) and check that `nvidia-smi` lists a card"));
            return;
        }
        Err(e) => {
            checks.push(Check::fail("gpu", e.clone(), "Install the NVIDIA driver (CUDA 13 capable) and check that `nvidia-smi` lists a card"));
            return;
        }
    };
    let names: Vec<String> = machine.gpus.iter().map(|g| format!("{} {} ({} free of {})", g.label(), g.arch(), gib(g.free_bytes), gib(g.total_bytes))).collect();
    checks.push(Check::ok("gpu", names.join("; ")));
    let cuda = machine.gpus.iter().any(|g| g.backend == GpuBackend::Cuda);
    if !cuda {
        return;
    }
    match env.driver {
        Some(_) => checks.push(Check::ok("driver", format!("CUDA driver {}", version(env.driver)))),
        None => checks.push(Check::fail("driver", "CUDA driver version unreadable", "Update the NVIDIA driver, then run `nvidia-smi`")),
    }
    match env.nvrtc {
        Some(_) => checks.push(Check::ok("nvrtc", format!("NVRTC {}", version(env.nvrtc)))),
        None => checks.push(Check::fail("nvrtc", "the NVRTC library was not found", specrt::hardware::nvrtc_missing_message())),
    }
    let old: Vec<String> = machine.gpus.iter().filter(|g| g.backend == GpuBackend::Cuda && g.compute.0 < 8).map(|g| format!("{} {}", g.label(), g.arch())).collect();
    if !old.is_empty() {
        checks.push(Check::fail("compute_capability", format!("below sm_80: {}", old.join(", ")), "Use a GPU with compute capability 8.0 or newer"));
    }
}

fn cache_check(dir: &Path, checks: &mut Vec<Check>) {
    let probe = dir.join(format!(".doctor-{}", std::process::id()));
    let writable = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&probe, b"ok")).and_then(|_| std::fs::remove_file(&probe));
    match writable {
        Ok(()) => checks.push(Check::ok("jit_cache", format!("{} is writable", dir.display()))),
        Err(e) => checks.push(Check::fail(
            "jit_cache",
            format!("{} is not writable: {e}", dir.display()),
            "Make the directory writable, or set [runtime] cache_dir in the model spec to a folder you own",
        )),
    }
}

fn memory_check(env: &Environment, f: &ModelFeatures, checks: &mut Vec<Check>) {
    let need = f.weights_bytes + f.state_bytes_per_seq + f.kv_bytes_per_token * crate::fit::TYPICAL_REQUEST.min(f.max_context) as u64;
    let free: u64 = match &env.machine {
        Ok(m) => m.gpus.iter().map(|g| g.free_bytes).sum(),
        Err(_) => 0,
    };
    if free == 0 {
        checks.push(Check::fail("memory", format!("needs about {} and no GPU memory is available", gib(need)), "Fix the gpu check first"));
        return;
    }
    let usable = (free as f64 * crate::fit::GPU_UTILIZATION) as u64;
    let detail = format!("needs about {} (weights {}, one {}-token request), {} free", gib(need), gib(f.weights_bytes), crate::fit::TYPICAL_REQUEST.min(f.max_context), gib(free));
    if need > usable {
        checks.push(Check::fail(
            "memory",
            detail,
            "Free GPU memory (`nvidia-smi` shows what holds it), pick a smaller quantisation, or add a GPU",
        ));
    } else if need as f64 > usable as f64 * 0.85 {
        checks.push(Check::warn("memory", detail, "The context will be short: use fp8 KV or free GPU memory for more"));
    } else {
        checks.push(Check::ok("memory", detail));
    }
}

fn model_checks(model: &str, env: &Environment, checks: &mut Vec<Check>) -> Option<ModelFeatures> {
    let (spec, f) = match features(model) {
        Ok(x) => x,
        Err(e) => {
            checks.push(Check::fail(
                "spec",
                e,
                "Check the name: pass a spec .toml, a HF repo that is downloaded, or a checkpoint folder with config.json",
            ));
            return None;
        }
    };
    checks.push(Check::ok("spec", format!("{} parses: quant {}, head_quant {}, act_quant {}, max context {}", spec.name, f.quant, f.head_quant, f.act_quant, f.max_context)));
    match weights_on_disk(&spec.repo, &env.hub_dir) {
        Ok(bytes) => checks.push(Check::ok("weights", format!("{} on disk for {}", gib(bytes), spec.repo))),
        Err(e) => checks.push(Check::fail(
            "weights",
            e,
            format!("Download {} in the app, or run `huggingface-cli download {}`", spec.repo, spec.repo),
        )),
    }
    memory_check(env, &f, checks);
    Some(f)
}

pub fn report(env: &Environment, model: Option<&str>) -> Value {
    let mut checks = Vec::new();
    host_checks(env, &mut checks);
    let cache = specrt::spec::Spec::load(Path::new(model.unwrap_or(""))).ok().and_then(|s| s.runtime.cache_dir).unwrap_or_else(|| env.cache_dir.clone());
    cache_check(&cache, &mut checks);
    let features = model.and_then(|m| model_checks(m, env, &mut checks));
    let exit = exit_code(&checks);
    let devices: Vec<Value> = env
        .machine
        .as_ref()
        .map(|m| {
            m.gpus
                .iter()
                .map(|g| json!({"ordinal": g.ordinal, "name": g.name, "arch": g.arch(), "total_bytes": g.total_bytes, "free_bytes": g.free_bytes}))
                .collect()
        })
        .unwrap_or_default();
    json!({
        "exit": exit,
        "devices": devices,
        "driver": env.driver.map(|(a, b)| format!("{a}.{b}")),
        "nvrtc": env.nvrtc.map(|(a, b)| format!("{a}.{b}")),
        "model": features,
        "checks": checks,
    })
}

pub fn exit_code(checks: &[Check]) -> i32 {
    if checks.iter().any(|c| c.status == Status::Fail) {
        2
    } else if checks.iter().any(|c| c.status == Status::Warn) {
        1
    } else {
        0
    }
}

pub fn describe(report: &Value) -> String {
    let mut out = String::new();
    for c in report["checks"].as_array().into_iter().flatten() {
        let mark = match c["status"].as_str() {
            Some("ok") => "ok  ",
            Some("warn") => "warn",
            _ => "FAIL",
        };
        out.push_str(&format!("{mark} {:<18} {}\n", c["name"].as_str().unwrap_or(""), c["detail"].as_str().unwrap_or("")));
        if let Some(next) = c["next"].as_str() {
            out.push_str(&format!("     next: {next}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use specrt::hardware::fake_gpus;

    fn fixture(name: &str) -> String {
        format!("{}/../specrt/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    fn env(machine: Result<Machine, String>, tag: &str) -> Environment {
        let dir = std::env::temp_dir().join(format!("ib-doctor-{tag}-{}", std::process::id()));
        Environment { machine, driver: Some((13, 0)), nvrtc: Some((13, 0)), cache_dir: dir.join("cache"), hub_dir: dir.join("hub") }
    }

    fn summary(report: &Value) -> Vec<(String, String)> {
        report["checks"].as_array().unwrap().iter().map(|c| (c["name"].as_str().unwrap().to_string(), c["status"].as_str().unwrap().to_string())).collect()
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    fn with_free(free: u64) -> Machine {
        let mut m = fake_gpus("test", 1, false);
        m.gpus[0].free_bytes = free;
        m.gpus[0].total_bytes = m.gpus[0].total_bytes.max(free);
        m
    }

    #[test]
    fn a_machine_with_no_gpu_fails_with_a_next_action() {
        let e = env(Err("no CUDA driver library on this machine".into()), "nogpu");
        let r = report(&e, None);
        assert_eq!(r["exit"], 2);
        assert_eq!(summary(&r), pairs(&[("gpu", "fail"), ("jit_cache", "ok")]));
        assert!(r["checks"][0]["next"].as_str().unwrap().contains("nvidia-smi"));
        assert_eq!(r["devices"], json!([]));
    }

    #[test]
    fn a_checkpoint_without_weights_fails_and_says_how_to_download() {
        let dir = std::env::temp_dir().join(format!("ib-doctor-noweights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(fixture("tiny-qwen3.5/config.json"), dir.join("config.json")).unwrap();
        let e = env(Ok(with_free(90 << 30)), "noweights");
        let r = report(&e, Some(dir.to_str().unwrap()));
        assert_eq!(r["exit"], 2);
        let s = summary(&r);
        assert!(s.contains(&("spec".into(), "ok".into())), "{s:?}");
        assert!(s.contains(&("weights".into(), "fail".into())), "{s:?}");
        let weights = r["checks"].as_array().unwrap().iter().find(|c| c["name"] == "weights").unwrap();
        assert!(weights["next"].as_str().unwrap().contains("huggingface-cli download"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn too_little_memory_fails_and_enough_passes() {
        let tiny = fixture("tiny-qwen3.5");
        let low = report(&env(Ok(with_free(1 << 10)), "low"), Some(&tiny));
        assert_eq!(low["exit"], 2);
        let mem = low["checks"].as_array().unwrap().iter().find(|c| c["name"] == "memory").unwrap();
        assert_eq!(mem["status"], "fail");
        assert!(mem["next"].as_str().unwrap().contains("Free GPU memory"));
        let high = report(&env(Ok(with_free(90 << 30)), "high"), Some(&tiny));
        assert_eq!(high["exit"], 0, "{high}");
        assert_eq!(summary(&high), pairs(&[("gpu", "ok"), ("driver", "ok"), ("nvrtc", "ok"), ("jit_cache", "ok"), ("spec", "ok"), ("weights", "ok"), ("memory", "ok")]));
        assert_eq!(high["model"]["kv_dtype"], "f16");
        assert!(high["model"]["max_context"].as_u64().unwrap() > 0);
    }

    #[test]
    fn a_missing_nvrtc_and_an_unwritable_cache_are_each_reported() {
        let mut e = env(Ok(with_free(90 << 30)), "nvrtc");
        e.nvrtc = None;
        let blocker = std::env::temp_dir().join(format!("ib-doctor-blocker-{}", std::process::id()));
        std::fs::write(&blocker, b"file").unwrap();
        e.cache_dir = blocker.join("cache");
        let r = report(&e, None);
        assert_eq!(r["exit"], 2);
        assert_eq!(summary(&r), pairs(&[("gpu", "ok"), ("driver", "ok"), ("nvrtc", "fail"), ("jit_cache", "fail")]));
        std::fs::remove_file(blocker).unwrap();
    }

    #[test]
    fn an_unknown_model_fails_the_spec_check() {
        let r = report(&env(Ok(with_free(90 << 30)), "unknown"), Some("/nonexistent/nothing"));
        assert_eq!(r["exit"], 2);
        assert!(summary(&r).contains(&("spec".into(), "fail".into())));
        assert!(r["model"].is_null());
    }

    #[test]
    fn mtp_fixture_reports_its_draft_head() {
        let (_, f) = features(&fixture("tiny-qwen3.5-mtp")).unwrap();
        assert!(f.mtp.is_some());
        let (_, plain) = features(&fixture("tiny-qwen3.5")).unwrap();
        assert!(plain.mtp.is_none());
    }
}
