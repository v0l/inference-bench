use crate::codegen::Module;
use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;

pub struct Jit {
    ctx: Arc<CudaContext>,
    dir: PathBuf,
    arch: &'static str,
    options: Vec<String>,
}

pub struct Loaded {
    pub module: Arc<CudaModule>,
    pub cache_hit: bool,
    pub key: String,
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

pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("SPECRT_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("specrt"))
}

impl Jit {
    pub fn new(ctx: Arc<CudaContext>, dir: PathBuf) -> Result<Self> {
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
        std::fs::create_dir_all(&dir)?;
        Ok(Jit { ctx, dir, arch, options: vec!["--std=c++17".to_string()] })
    }

    pub fn arch(&self) -> &str {
        self.arch
    }

    pub fn load(&self, module: &Module) -> Result<Loaded> {
        let key = cache_key(&module.source, self.arch, &self.options);
        let path = self.dir.join(format!("{key}.ptx"));
        let (ptx, cache_hit) = match std::fs::read_to_string(&path) {
            Ok(text) => (text, true),
            Err(_) => {
                let opts = CompileOptions {
                    arch: Some(self.arch),
                    options: self.options.clone(),
                    ..Default::default()
                };
                let ptx = compile_ptx_with_opts(&module.source, opts)
                    .map_err(|e| anyhow!("nvrtc: {e:?}"))?
                    .to_src();
                let tmp = path.with_extension("tmp");
                std::fs::write(&tmp, &ptx)?;
                std::fs::rename(&tmp, &path)?;
                (ptx, false)
            }
        };
        let module = self.ctx.load_module(Ptx::from_src(ptx))?;
        Ok(Loaded { module, cache_hit, key })
    }
}
