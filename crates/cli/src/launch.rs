use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use inference_bench_core::fit::{self, LoadPlan, PlanInput};
use inference_bench_core::hf::{self, Hub, Progress, RepoFile};
use inference_bench_core::library::{self, Format, LocalModel, human_size};
use inference_bench_core::settings::Settings;
use inference_bench_server::config::ModelConfig;
use inference_bench_server::ServerConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Access {
    Local,
    Lan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Output {
    Text,
    Json,
}

#[derive(Args, Debug, Clone)]
pub struct LaunchArgs {
    #[arg(help = "A model folder, spec TOML or GGUF file, a model name from the library, or a Hugging Face repo to download")]
    pub model: Option<String>,
    #[arg(long, help = "Print what would be downloaded and run, and change nothing")]
    pub plan: bool,
    #[arg(long, short = 'y', help = "Download, load and serve without asking")]
    pub yes: bool,
    #[arg(long, help = "Never open a terminal interface or ask questions (there is none yet, kept for scripts)")]
    pub no_tui: bool,
    #[arg(long, help = "Report whether a launched server is running")]
    pub status: bool,
    #[arg(long, help = "Stop the launched server")]
    pub stop: bool,
    #[arg(long, help = "Run in the background and return once the model is loaded")]
    pub detach: bool,
    #[arg(long, value_enum, default_value = "text")]
    pub output: Output,
    #[arg(long, value_enum, default_value = "local", help = "local listens on 127.0.0.1; lan listens on every interface and generates an API key")]
    pub access: Access,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long, help = "API key, overrides the generated one")]
    pub api_key: Option<String>,
    #[arg(long, help = "Context length (default: the load dialog's default)")]
    pub ctx: Option<usize>,
    #[arg(long, help = "Parallel requests (default: the load dialog's default)")]
    pub seqs: Option<usize>,
    #[arg(long, help = "KV cache size in GB (default: the load dialog's default)")]
    pub kv_gb: Option<f64>,
    #[arg(long, help = "A file inside a Hugging Face repo, for repos with several GGUF quants")]
    pub file: Option<String>,
    #[arg(long, default_value = "xml_function", help = "Tool call format the model writes: xml_function or hermes")]
    pub tool_parser: String,
    #[arg(long, help = "Serve even when the estimate says it does not fit")]
    pub force: bool,
    #[arg(long, default_value = hf::HF_BASE, hide = true)]
    pub hf_base: String,
    #[arg(long, hide = true, help = "Pretend this much GPU memory is free, in MiB")]
    pub assume_free_mib: Option<u64>,
    #[arg(long, default_value_t = 3600, help = "Seconds --detach waits for the download and load")]
    pub wait_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct State {
    pub pid: u32,
    pub model: String,
    pub bind: String,
    pub urls: Vec<String>,
    pub api_key: Option<String>,
    pub started: u64,
    pub ready: bool,
}

impl State {
    pub fn path(data: &Path) -> PathBuf {
        data.join("launch.json")
    }

    pub fn read(data: &Path) -> Option<State> {
        serde_json::from_str(&std::fs::read_to_string(Self::path(data)).ok()?).ok()
    }

    pub fn write(&self, data: &Path) -> Result<()> {
        std::fs::create_dir_all(data)?;
        let path = Self::path(data);
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    pub fn clear(data: &Path) {
        let _ = std::fs::remove_file(Self::path(data));
    }

    pub fn alive(&self) -> bool {
        pid_alive(self.pid)
    }

    pub fn json(&self, health: Option<Value>) -> Value {
        json!({
            "running": self.alive(),
            "ready": self.ready,
            "pid": self.pid,
            "model": self.model,
            "urls": self.urls,
            "api_key": self.api_key,
            "health": health,
        })
    }
}

pub fn pid_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[derive(Debug, Clone)]
pub enum Source {
    Local(LocalModel),
    Remote { repo: String, files: Vec<RepoFile> },
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub model: String,
    pub source: &'static str,
    pub format: String,
    pub quant: String,
    pub download_bytes: u64,
    pub download_files: Vec<String>,
    pub context: usize,
    pub max_context: usize,
    pub parallel_requests: usize,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub state_bytes: u64,
    pub total_bytes: u64,
    pub free_gpu_bytes: Option<u64>,
    pub fits: Option<bool>,
    pub verdict: String,
    pub access: String,
    pub bind: String,
}

pub fn plan_text(p: &Plan) -> String {
    let mut o = String::new();
    let line = |o: &mut String, k: &str, v: String| o.push_str(&format!("{k:<14}{v}\n"));
    line(&mut o, "model", format!("{} ({}, {})", p.model, p.format, p.quant));
    match p.source {
        "download" => line(&mut o, "download", format!("{} in {} files", human_size(p.download_bytes), p.download_files.len())),
        _ => line(&mut o, "download", "nothing, the weights are on disk".into()),
    }
    line(&mut o, "context", format!("{} tokens (model maximum {})", p.context, p.max_context));
    line(&mut o, "parallel", format!("{} requests", p.parallel_requests));
    line(&mut o, "weights", human_size(p.weights_bytes));
    line(&mut o, "kv cache", human_size(p.kv_cache_bytes));
    line(&mut o, "state", human_size(p.state_bytes));
    line(&mut o, "total", human_size(p.total_bytes));
    line(&mut o, "free gpu", p.free_gpu_bytes.map(human_size).unwrap_or_else(|| "unknown, no GPU reading".into()));
    line(&mut o, "fit", p.verdict.clone());
    line(&mut o, "listen", format!("{} ({})", p.bind, p.access));
    o
}

fn free_gpu(args: &LaunchArgs) -> Option<u64> {
    if let Some(m) = args.assume_free_mib {
        return Some(m << 20);
    }
    inference_bench_core::gpu::sample().iter().map(|g| g.total_bytes.saturating_sub(g.used_bytes)).reduce(|a, b| a + b)
}

pub fn resolve(args: &LaunchArgs, data: &Path, settings: &Settings) -> Result<Source> {
    let name = args.model.as_deref().context("name a model: a folder, a spec, a library name or a Hugging Face repo")?;
    let path = Path::new(name);
    if path.exists() {
        let m = library::local_model_at(&std::fs::canonicalize(path)?).with_context(|| format!("{name} holds no model the engine can read"))?;
        return Ok(Source::Local(m));
    }
    let sources = library::Sources::standard(data, &settings.model_folders, &settings.spec_folders);
    let found = library::scan(&sources);
    if let Some(m) = found.iter().find(|m| m.name == name || m.repo == name || library::short_alias(&m.name).is_some_and(|a| a == name)) {
        return Ok(Source::Local(m.clone()));
    }
    if name.contains('/') {
        let hub = Hub::new(&args.hf_base, &settings.hf_token);
        let files = hub.files(name).map_err(anyhow::Error::msg)?;
        return Ok(Source::Remote { repo: name.to_string(), files });
    }
    bail!("{name} is not a path, a model in the library or a Hugging Face repo (org/name)")
}

const INDEX: &str = "model.safetensors.index.json";

pub fn download_set(repo: &str, files: &[RepoFile], pick: Option<&str>) -> Result<Vec<RepoFile>> {
    let mut chosen: Vec<RepoFile> = hf::companion_files(files).into_iter().cloned().collect();
    chosen.extend(files.iter().filter(|f| f.path == INDEX).cloned());
    let weights: Vec<&RepoFile> = files.iter().filter(|f| hf::is_weight_file(&f.path)).collect();
    match pick {
        Some(p) => {
            let file = files.iter().find(|f| f.path == p).with_context(|| format!("{repo} has no file {p}"))?;
            chosen.push(file.clone());
        }
        None => {
            let safetensors: Vec<&RepoFile> = weights.iter().copied().filter(|f| f.path.ends_with(".safetensors")).collect();
            if !safetensors.is_empty() {
                chosen.extend(safetensors.into_iter().cloned());
            } else {
                let names: Vec<&str> = weights.iter().map(|f| f.path.as_str()).collect();
                bail!("{repo} has only GGUF files; pick one with --file: {}", names.join(", "));
            }
        }
    }
    Ok(chosen)
}

pub fn build_plan(args: &LaunchArgs, data: &Path, settings: &Settings, source: &Source) -> Result<(Plan, Vec<RepoFile>)> {
    let free = free_gpu(args);
    let (name, described, format, quant, download) = match source {
        Source::Local(m) => {
            let d = fit::describe(m).map_err(anyhow::Error::msg)?;
            (m.name.clone(), d, format!("{:?}", m.format).to_lowercase(), m.quant.clone(), Vec::new())
        }
        Source::Remote { repo, files } => {
            let set = download_set(repo, files, args.file.as_deref())?;
            let hub = Hub::new(&args.hf_base, &settings.hf_token);
            let config = hub.text(repo, "config.json").map_err(anyhow::Error::msg)?;
            let d = fit::describe_config_json(repo, &config).map_err(anyhow::Error::msg)?;
            let format = if set.iter().any(|f| f.path.ends_with(".gguf")) { "gguf" } else { "safetensors" };
            let quant = serde_json::to_value(d.spec.quant).ok().and_then(|q| q.as_str().map(|s| s.to_uppercase())).unwrap_or_default();
            let missing: Vec<RepoFile> = set.into_iter().filter(|f| !hf::snapshot_path(&library::hub_dir(), repo, &f.path).exists()).collect();
            (repo.clone(), d, format.to_string(), quant, missing)
        }
    };
    let input = PlanInput { ctx: args.ctx, default_ctx: settings.default_ctx, seqs: args.seqs, default_seqs: settings.default_seqs, kv_gb: args.kv_gb };
    let lp: LoadPlan = fit::plan_for(&described.anatomy, &input, free);
    let fits = lp.fits();
    let verdict = match (lp.holds_one_request, fits) {
        (false, _) => "the KV cache cannot hold one request of this length: raise --kv-gb or lower --ctx".to_string(),
        (true, Some(true)) => "fits in free memory".to_string(),
        (true, Some(false)) => format!("does not fit in free memory, {} over", human_size(lp.estimate.total().saturating_sub(free.unwrap_or(0)))),
        (true, None) => "no GPU reading, so the estimate cannot be checked".to_string(),
    };
    let bind = bind_addr(args);
    let plan = Plan {
        model: name,
        source: if download.is_empty() && matches!(source, Source::Local(_)) { "disk" } else if download.is_empty() { "cache" } else { "download" },
        format,
        quant,
        download_bytes: download.iter().map(|f| f.size).sum(),
        download_files: download.iter().map(|f| f.path.clone()).collect(),
        context: lp.ctx,
        max_context: lp.max_ctx,
        parallel_requests: lp.seqs,
        weights_bytes: lp.estimate.weights,
        kv_cache_bytes: lp.estimate.kv,
        state_bytes: lp.estimate.state,
        total_bytes: lp.estimate.total(),
        free_gpu_bytes: free,
        fits,
        verdict,
        access: match args.access {
            Access::Local => "local only".into(),
            Access::Lan => "LAN, API key required".into(),
        },
        bind,
    };
    let _ = data;
    Ok((plan, download))
}

fn bind_addr(args: &LaunchArgs) -> String {
    match args.access {
        Access::Local => format!("127.0.0.1:{}", args.port),
        Access::Lan => format!("0.0.0.0:{}", args.port),
    }
}

pub fn generate_key() -> String {
    let bytes = specrt::pairing::random(24).expect("the system random source");
    format!("sk-ib-{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>())
}

fn lan_address() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    Some(socket.local_addr().ok()?.ip().to_string())
}

fn download(args: &LaunchArgs, settings: &Settings, repo: &str, files: &[RepoFile], quiet: bool) -> Result<()> {
    let hub = Hub::new(&args.hf_base, &settings.hf_token);
    let hub_dir = library::hub_dir();
    for f in files {
        let progress = Progress::default();
        let (done, total) = (progress.done.clone(), progress.total.clone());
        let started = Instant::now();
        let reporter = (!quiet).then(|| {
            let (done, total, path) = (done.clone(), total.clone(), f.path.clone());
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_secs(5));
                    let t = total.load(std::sync::atomic::Ordering::Relaxed);
                    if t == 0 {
                        continue;
                    }
                    let d = done.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!("downloading {path}: {} of {}", human_size(d), human_size(t));
                    if d >= t {
                        break;
                    }
                }
            })
        });
        hub.download(repo, &f.path, &hub_dir, &progress).map_err(anyhow::Error::msg)?;
        done.store(total.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
        if !quiet {
            eprintln!("downloaded {} ({}) in {:.0} s", f.path, human_size(f.size), started.elapsed().as_secs_f64());
        }
        drop(reporter);
    }
    Ok(())
}

fn server_config(args: &LaunchArgs, plan: &Plan, model: &LocalModel, key: &Option<String>) -> Result<(ServerConfig, ModelConfig)> {
    let id = model.repo.rsplit('/').next().filter(|s| !s.is_empty()).map(String::from).unwrap_or_else(|| model.name.replace(' ', "-"));
    let mut served = vec![model.name.clone()];
    if !model.repo.is_empty() {
        served.push(model.repo.clone());
    }
    served.retain(|s| s != &id);
    served.dedup();
    let mc = ModelConfig { id: id.clone(), model: Some(model.model_id(inference_bench_core::target::NATIVE)), max_ctx: Some(plan.context), served_names: served, ..Default::default() };
    let mut config = ServerConfig {
        bind: plan.bind.clone(),
        max_num_seqs: Some(plan.parallel_requests),
        max_model_len: Some(plan.context),
        kv_cache_gb: Some((plan.kv_cache_bytes as f64 / fit::GB * 100.0).ceil() / 100.0),
        default_model: Some(id),
        tool_parser: Some(args.tool_parser.clone()),
        models: vec![mc.clone()],
        ..ServerConfig::default()
    };
    config.specs.auto_update = false;
    if let Some(k) = key {
        config.api_keys = vec![k.clone()];
    }
    config.canonical_parser_ids();
    config.validate()?;
    Ok((config, mc))
}

fn urls(args: &LaunchArgs, addr: std::net::SocketAddr) -> Vec<String> {
    let mut v = vec![format!("http://127.0.0.1:{}", addr.port())];
    if args.access == Access::Lan
        && let Some(ip) = lan_address() {
            v.push(format!("http://{ip}:{}", addr.port()));
        }
    v
}

pub fn print_ready(state: &State, output: Output) {
    match output {
        Output::Json => println!("{}", serde_json::to_string_pretty(&state.json(None)).unwrap()),
        Output::Text => {
            println!("model    {}", state.model);
            for u in &state.urls {
                println!("address  {u}/v1");
            }
            if let Some(k) = &state.api_key {
                println!("api key  {k}");
            }
            println!("stop     inference-bench launch --stop");
        }
    }
}

fn confirm(plan: &Plan) -> Result<bool> {
    use std::io::{BufRead, IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        bail!("not a terminal: pass --yes to go ahead, or --plan to only see the plan");
    }
    eprint!("{}\nGo ahead? [y/N] ", plan_text(plan));
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

fn health(state: &State) -> Option<Value> {
    let url = format!("{}/health", state.urls.first()?);
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3))).build().into();
    agent.get(url).call().ok()?.body_mut().read_json().ok()
}

pub fn status(data: &Path, output: Output) -> Result<()> {
    let Some(state) = State::read(data) else {
        match output {
            Output::Json => println!("{}", json!({"running": false})),
            Output::Text => println!("not running"),
        }
        return Ok(());
    };
    let alive = state.alive();
    let health = alive.then(|| health(&state)).flatten();
    match output {
        Output::Json => println!("{}", serde_json::to_string_pretty(&state.json(health))?),
        Output::Text => match alive {
            false => println!("not running (a stale record of pid {} was found)", state.pid),
            true => {
                println!("running, pid {}", state.pid);
                print_ready(&state, Output::Text);
                if let Some(h) = health {
                    println!("health   {}", h["status"].as_str().unwrap_or("unknown"));
                }
            }
        },
    }
    if !alive {
        State::clear(data);
    }
    Ok(())
}

pub fn stop(data: &Path, output: Output) -> Result<()> {
    let Some(state) = State::read(data) else {
        match output {
            Output::Json => println!("{}", json!({"stopped": false, "reason": "not running"})),
            Output::Text => println!("not running"),
        }
        return Ok(());
    };
    let mut stopped = !state.alive();
    if !stopped {
        let _ = std::process::Command::new("kill").args(["-TERM", &state.pid.to_string()]).status();
        let deadline = Instant::now() + Duration::from_secs(90);
        while Instant::now() < deadline {
            if !state.alive() {
                stopped = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    if stopped {
        State::clear(data);
    }
    match output {
        Output::Json => println!("{}", json!({"stopped": stopped, "pid": state.pid})),
        Output::Text => println!("{}", if stopped { "stopped" } else { "still running after 90 s" }),
    }
    if !stopped {
        bail!("pid {} did not exit", state.pid);
    }
    Ok(())
}

fn detach(args: &LaunchArgs, data: &Path) -> Result<()> {
    let exe = std::env::current_exe()?;
    let forwarded: Vec<String> = std::env::args().skip(1).filter(|a| a != "--detach").collect();
    std::fs::create_dir_all(data)?;
    let log_path = data.join("launch.log");
    let log = std::fs::File::create(&log_path)?;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(&forwarded).stdin(std::process::Stdio::null()).stdout(log.try_clone()?).stderr(log);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn()?;
    let deadline = Instant::now() + Duration::from_secs(args.wait_secs);
    loop {
        if let Some(state) = State::read(data).filter(|s| s.pid == child.id() && s.ready) {
            print_ready(&state, args.output);
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            bail!("the server exited ({status}); see {}", log_path.display());
        }
        if Instant::now() > deadline {
            let _ = std::process::Command::new("kill").args(["-TERM", &child.id().to_string()]).status();
            bail!("not ready after {} s; see {}", args.wait_secs, log_path.display());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

pub fn run(args: &LaunchArgs, data: &Path) -> Result<()> {
    if args.status {
        return status(data, args.output);
    }
    if args.stop {
        return stop(data, args.output);
    }
    let settings = Settings::load(data);
    let source = resolve(args, data, &settings)?;
    let (plan, missing) = build_plan(args, data, &settings, &source)?;
    if args.plan {
        match args.output {
            Output::Json => println!("{}", serde_json::to_string_pretty(&plan)?),
            Output::Text => print!("{}", plan_text(&plan)),
        }
        return Ok(());
    }
    if let Some(s) = State::read(data).filter(|s| s.alive()) {
        bail!("already running as pid {} on {}: stop it with `inference-bench launch --stop`", s.pid, s.urls.join(", "));
    }
    if plan.fits == Some(false) && !args.force {
        bail!("{}: refusing to load. Lower --ctx or --seqs, free GPU memory, or pass --force", plan.verdict);
    }
    if !args.yes && !args.no_tui && !confirm(&plan)? {
        bail!("cancelled");
    }
    if !args.yes && args.no_tui {
        bail!("--no-tui needs --yes to go ahead");
    }
    if args.detach {
        return detach(args, data);
    }
    let quiet = args.output == Output::Json;
    let model = match &source {
        Source::Local(m) => m.clone(),
        Source::Remote { repo, .. } => {
            if !missing.is_empty() {
                download(args, &settings, repo, &missing, quiet)?;
            }
            let sources = library::Sources::standard(data, &settings.model_folders, &settings.spec_folders);
            library::scan(&sources).into_iter().find(|m| &m.repo == repo).with_context(|| format!("{repo} was downloaded but the library does not list it"))?
        }
    };
    let key = match (&args.api_key, args.access) {
        (Some(k), _) => Some(k.clone()),
        (None, Access::Lan) => Some(generate_key()),
        (None, Access::Local) => None,
    };
    if model.format == Format::Safetensors && model.repo.is_empty() {
        bail!("{} is a plain folder: the engine loads safetensors models by Hugging Face repo from the hub cache. Download it with `launch org/name` or give a spec TOML", model.path.display());
    }
    let (config, mc) = server_config(args, &plan, &model, &key)?;
    if matches!(model.format, Format::Spec | Format::Safetensors | Format::Gguf) {
        tracing::info!(model = %model.name, "loading");
    }
    let (args_out, data_dir, model_name) = (args.clone(), data.to_path_buf(), mc.id.clone());
    inference_bench_server::cli::run_config(config, Some(mc), move |handle| {
        let state = State {
            pid: std::process::id(),
            model: model_name,
            bind: handle.addr.to_string(),
            urls: urls(&args_out, handle.addr),
            api_key: key,
            started: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            ready: true,
        };
        state.write(&data_dir)?;
        print_ready(&state, args_out.output);
        Ok(())
    })?;
    State::clear(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../specrt/tests/fixtures").join(name)
    }

    fn args(model: &str) -> LaunchArgs {
        LaunchArgs {
            model: Some(model.to_string()),
            plan: true,
            yes: false,
            no_tui: true,
            status: false,
            stop: false,
            detach: false,
            output: Output::Json,
            access: Access::Local,
            port: 8080,
            api_key: None,
            ctx: None,
            seqs: None,
            kv_gb: None,
            file: None,
            tool_parser: "xml_function".into(),
            force: false,
            hf_base: hf::HF_BASE.into(),
            assume_free_mib: Some(24 * 1024),
            wait_secs: 10,
        }
    }

    #[test]
    fn the_plan_for_a_fixture_uses_the_load_dialogs_numbers() {
        let a = args(fixture("tiny-qwen3.5").to_str().unwrap());
        let settings = Settings::default();
        let data = std::env::temp_dir();
        let Source::Local(m) = resolve(&a, &data, &settings).unwrap() else { panic!("a folder is local") };
        let (plan, missing) = build_plan(&a, &data, &settings, &Source::Local(m.clone())).unwrap();
        assert!(missing.is_empty());
        assert_eq!(plan.source, "disk");
        assert_eq!(plan.format, "safetensors");
        assert_eq!(plan.context, 4096);
        assert_eq!(plan.fits, Some(true));
        let d = fit::describe(&m).unwrap();
        let lp = fit::plan_for(&d.anatomy, &PlanInput { ctx: None, default_ctx: settings.default_ctx, seqs: None, default_seqs: settings.default_seqs, kv_gb: None }, Some(24 << 30));
        assert_eq!(plan.total_bytes, lp.estimate.total());
        assert_eq!(plan.kv_cache_bytes, lp.kv_bytes);
        assert!(plan_text(&plan).contains("nothing, the weights are on disk"));
    }

    #[test]
    fn a_plan_that_does_not_fit_says_by_how_much() {
        let mut a = args(fixture("tiny-qwen3.5").to_str().unwrap());
        a.assume_free_mib = Some(0);
        let settings = Settings::default();
        let data = std::env::temp_dir();
        let src = resolve(&a, &data, &settings).unwrap();
        let (plan, _) = build_plan(&a, &data, &settings, &src).unwrap();
        assert_eq!(plan.fits, Some(false));
        assert!(plan.verdict.contains("does not fit"), "{}", plan.verdict);
    }

    #[test]
    fn lan_access_binds_every_interface() {
        let mut a = args(fixture("tiny-qwen3.5").to_str().unwrap());
        a.access = Access::Lan;
        a.port = 9000;
        assert_eq!(bind_addr(&a), "0.0.0.0:9000");
        let k = generate_key();
        assert!(k.starts_with("sk-ib-") && k.len() > 30);
        assert_ne!(k, generate_key());
    }

    #[test]
    fn a_remote_repo_downloads_weights_and_companions_but_a_gguf_only_repo_needs_a_file() {
        let f = |p: &str, size: u64| RepoFile { path: p.into(), size, kind: "file".into() };
        let files = vec![f("config.json", 10), f("tokenizer.json", 20), f("model-00001.safetensors", 100), f("README.md", 5), f(INDEX, 3)];
        let set = download_set("o/m", &files, None).unwrap();
        let names: Vec<&str> = set.iter().map(|f| f.path.as_str()).collect();
        assert!(names.contains(&"model-00001.safetensors") && names.contains(&"config.json") && names.contains(&INDEX));
        assert!(!names.contains(&"README.md"));
        let gguf = vec![f("a.Q4.gguf", 1), f("a.Q8.gguf", 2)];
        let err = download_set("o/g", &gguf, None).unwrap_err().to_string();
        assert!(err.contains("a.Q4.gguf") && err.contains("--file"), "{err}");
        assert_eq!(download_set("o/g", &gguf, Some("a.Q8.gguf")).unwrap().len(), 1);
    }

    #[test]
    fn state_round_trips_and_a_dead_pid_is_not_alive() {
        let dir = std::env::temp_dir().join(format!("ib-launch-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = State { pid: 4_000_000, model: "m".into(), bind: "127.0.0.1:1".into(), urls: vec!["http://127.0.0.1:1".into()], api_key: Some("k".into()), started: 1, ready: true };
        s.write(&dir).unwrap();
        assert_eq!(State::read(&dir), Some(s.clone()));
        assert!(!s.alive());
        status(&dir, Output::Json).unwrap();
        assert!(State::read(&dir).is_none(), "a stale record is dropped");
    }

    #[test]
    fn stop_ends_a_running_process_and_forgets_it() {
        let dir = std::env::temp_dir().join(format!("ib-launch-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut child = std::process::Command::new("sleep").arg("60").spawn().unwrap();
        let s = State { pid: child.id(), model: "m".into(), bind: "x".into(), urls: vec![], api_key: None, started: 1, ready: true };
        s.write(&dir).unwrap();
        assert!(s.alive());
        let waiter = std::thread::spawn(move || child.wait().unwrap());
        stop(&dir, Output::Json).unwrap();
        waiter.join().unwrap();
        assert!(State::read(&dir).is_none());
    }
}
