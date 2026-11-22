use crate::codegen::Module;
use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub struct Jit {
    ctx: Arc<CudaContext>,
    dir: PathBuf,
    arch: &'static str,
    options: Vec<String>,
    max_bytes: u64,
    jobs: usize,
}

pub const DEFAULT_CACHE_MAX_BYTES: u64 = 4 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolVersions {
    pub nvrtc: (i32, i32),
    pub driver: i32,
}

pub fn tool_versions() -> Result<ToolVersions> {
    static V: OnceLock<ToolVersions> = OnceLock::new();
    if let Some(v) = V.get() {
        return Ok(*v);
    }
    let (mut major, mut minor, mut driver) = (0, 0, 0);
    unsafe {
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor).result().map_err(|e| anyhow!("nvrtcVersion: {e:?}"))?;
        cudarc::driver::sys::cuDriverGetVersion(&mut driver).result().map_err(|e| anyhow!("cuDriverGetVersion: {e:?}"))?;
    }
    Ok(*V.get_or_init(|| ToolVersions { nvrtc: (major, minor), driver }))
}

pub fn default_compile_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)
}

pub struct Loaded {
    pub module: Arc<CudaModule>,
    pub cache_hit: bool,
    pub key: String,
    pub compile_secs: f64,
    pub module_secs: f64,
    pub binary: bool,
}

impl Loaded {
    pub fn function(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name).with_context(|| format!("kernel {name}"))
    }
}

pub fn cache_key(source: &str, arch: &str, options: &[String]) -> String {
    let mut h = Sha256::new();
    h.update(source.as_bytes());
    h.update([0]);
    h.update(arch.as_bytes());
    for o in options {
        h.update([0]);
        h.update(o.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn versioned_key(source: &str, arch: &str, options: &[String], versions: ToolVersions) -> String {
    let mut all = options.to_vec();
    all.push(format!("nvrtc={}.{}", versions.nvrtc.0, versions.nvrtc.1));
    all.push(format!("driver={}", versions.driver));
    cache_key(source, arch, &all)
}

pub fn evict_oldest(dir: &Path, max_bytes: u64, keep: Option<&Path>) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| matches!(e.path().extension().and_then(|x| x.to_str()), Some("ptx" | "cubin" | "wgsl")))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            m.is_file().then(|| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
    let mut removed = 0;
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        if keep == Some(path.as_path()) {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            total -= len;
            removed += len;
        }
    }
    removed
}

fn compile_cubin(source: &str, sm: &str, options: &[String], jobs: usize) -> Result<Vec<u8>> {
    use cudarc::nvrtc::{result, sys};
    let src = CString::new(source).map_err(|_| anyhow!("kernel source has a NUL"))?;
    let prog = result::create_program(&src, None).map_err(|e| anyhow!("nvrtc create: {e:?}"))?;
    let mut opts = vec![format!("--gpu-architecture={sm}")];
    opts.extend(options.iter().cloned());
    if jobs > 1 {
        opts.push(format!("--split-compile={jobs}"));
    }
    unsafe {
        let out = match result::compile_program(prog, &opts) {
            Ok(()) => {
                let mut size = 0usize;
                match sys::nvrtcGetCUBINSize(prog, &mut size).result() {
                    Ok(()) => {
                        let mut buf = vec![0u8; size];
                        match sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut _).result() {
                            Ok(()) => Ok(buf),
                            Err(e) => Err(anyhow!("nvrtcGetCUBIN: {e:?}")),
                        }
                    }
                    Err(e) => Err(anyhow!("nvrtcGetCUBINSize: {e:?}")),
                }
            }
            Err(e) => {
                let log = result::get_program_log(prog).ok().map(|l| CStr::from_ptr(l.as_ptr()).to_string_lossy().into_owned()).unwrap_or_default();
                Err(anyhow!("nvrtc cubin: {e:?}\n{log}"))
            }
        };
        let _ = result::destroy_program(prog);
        out
    }
}

pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("specrt")
}

impl Jit {
    pub fn new(ctx: Arc<CudaContext>, dir: PathBuf) -> Result<Self> {
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
        std::fs::create_dir_all(&dir)?;
        Ok(Jit { ctx, dir, arch, options: vec!["--std=c++17".to_string()], max_bytes: DEFAULT_CACHE_MAX_BYTES, jobs: default_compile_jobs() })
    }

    pub fn with_limits(mut self, max_bytes: u64, jobs: usize) -> Self {
        self.max_bytes = max_bytes;
        self.jobs = jobs;
        self
    }

    pub fn arch(&self) -> &str {
        self.arch
    }

    pub fn arch_for(&self, module: &Module) -> &'static str {
        if module.arch_features && matches!(self.arch, "compute_120" | "compute_121") { if self.arch == "compute_120" { "compute_120a" } else { "compute_121a" } } else { self.arch }
    }

    fn touch(path: &Path) {
        if let Ok(f) = std::fs::OpenOptions::new().append(true).open(path) {
            let _ = f.set_modified(std::time::SystemTime::now());
        }
    }

    fn store(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        evict_oldest(&self.dir, self.max_bytes, Some(path));
        Ok(())
    }

    fn compile_ptx(&self, source: &str, arch: &'static str, path: &Path) -> Result<String> {
        let opts = CompileOptions { arch: Some(arch), options: self.options.clone(), ..Default::default() };
        let ptx = compile_ptx_with_opts(source, opts).map_err(|e| anyhow!("nvrtc: {e:?}"))?.to_src();
        self.store(path, ptx.as_bytes())?;
        Ok(ptx)
    }

    pub fn load(&self, module: &Module) -> Result<Loaded> {
        let arch = self.arch_for(module);
        let mut options = self.options.clone();
        options.push(format!("kir={}", module.kir_hash));
        let key = versioned_key(&module.source, arch, &options, tool_versions()?);
        let cubin_path = self.dir.join(format!("{key}.cubin"));
        let ptx_path = self.dir.join(format!("{key}.ptx"));
        let compile_started = std::time::Instant::now();
        let mut cache_hit = true;
        let mut cubin = std::fs::read(&cubin_path).ok();
        let mut ptx = if cubin.is_none() { std::fs::read_to_string(&ptx_path).ok() } else { None };
        if cubin.is_none() && ptx.is_none() {
            cache_hit = false;
            let sm = arch.replacen("compute_", "sm_", 1);
            match compile_cubin(&module.source, &sm, &self.options, self.jobs) {
                Ok(bytes) => {
                    self.store(&cubin_path, &bytes)?;
                    cubin = Some(bytes);
                }
                Err(_) => ptx = Some(self.compile_ptx(&module.source, arch, &ptx_path)?),
            }
        } else {
            Self::touch(if cubin.is_some() { &cubin_path } else { &ptx_path });
        }
        let compile_secs = compile_started.elapsed().as_secs_f64();
        let module_started = std::time::Instant::now();
        let (loaded, binary) = match (cubin, ptx) {
            (Some(bytes), _) => match self.ctx.load_module(Ptx::from_binary(bytes)) {
                Ok(m) => (m, true),
                Err(_) => {
                    let _ = std::fs::remove_file(&cubin_path);
                    let text = self.compile_ptx(&module.source, arch, &ptx_path)?;
                    (self.ctx.load_module(Ptx::from_src(text))?, false)
                }
            },
            (None, Some(text)) => (self.ctx.load_module(Ptx::from_src(text))?, false),
            (None, None) => unreachable!(),
        };
        let module_secs = module_started.elapsed().as_secs_f64();
        Ok(Loaded { module: loaded, cache_hit, key, compile_secs, module_secs, binary })
    }
}
