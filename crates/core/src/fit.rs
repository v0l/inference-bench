use specrt::anatomy::ModelAnatomy;

pub const MIN_CTX: usize = 512;
pub const STEP: usize = 512;
pub const MAX_SEQS: usize = crate::engine::MAX_PARALLEL;
pub const GPU_UTILIZATION: f64 = 0.9;
pub const TYPICAL_REQUEST: usize = 4096;
pub const GB: f64 = (1u64 << 30) as f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    pub weights: u64,
    pub kv: u64,
    pub state: u64,
    pub free: Option<u64>,
}

impl Estimate {
    pub fn total(&self) -> u64 {
        self.weights + self.kv + self.state
    }

    pub fn fits(&self) -> Option<bool> {
        self.free.map(|f| self.total() <= f)
    }
}

pub fn estimate(a: &ModelAnatomy, kv_bytes: u64, seqs: usize, free: Option<u64>) -> Estimate {
    let seqs = seqs.max(1) as u64;
    Estimate { weights: a.total_bytes, kv: kv_bytes, state: a.state_bytes_per_seq * seqs, free }
}

pub fn pool_bytes(a: &ModelAnatomy, ctx: usize, seqs: usize, kv_gb: f64) -> u64 {
    if seqs <= 1 {
        a.kv_bytes_per_token * ctx as u64
    } else {
        (kv_gb * GB) as u64
    }
}

pub fn pool_tokens(a: &ModelAnatomy, kv_bytes: u64) -> usize {
    kv_bytes.checked_div(a.kv_bytes_per_token).map_or(usize::MAX, |n| n as usize)
}

pub fn pool_holds_one(a: &ModelAnatomy, kv_bytes: u64, ctx: usize) -> bool {
    pool_tokens(a, kv_bytes) >= ctx
}

pub fn concurrency(a: &ModelAnatomy, kv_bytes: u64, per_request: usize) -> f64 {
    if per_request == 0 { 0.0 } else { pool_tokens(a, kv_bytes).min(usize::MAX / 2) as f64 / per_request as f64 }
}

pub fn room_bytes(a: &ModelAnatomy, free: Option<u64>, seqs: usize) -> Option<u64> {
    free.map(|f| f.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64))
}

pub fn default_kv_gb(a: &ModelAnatomy, free: Option<u64>, seqs: usize, ctx: usize) -> f64 {
    let needed = a.kv_bytes_per_token * ctx as u64;
    let bytes = match free {
        Some(f) => {
            let usable = (f as f64 * GPU_UTILIZATION) as u64;
            usable.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64)
        }
        None => needed,
    };
    round_gb(bytes.max(needed))
}

pub fn max_kv_gb(a: &ModelAnatomy, free: Option<u64>, seqs: usize, ctx: usize) -> f64 {
    let needed = a.kv_bytes_per_token * ctx as u64;
    let room = room_bytes(a, free, seqs).unwrap_or(needed * 4);
    round_gb(room.max(needed)).max(default_kv_gb(a, free, seqs, ctx)).max(0.25)
}

fn round_gb(bytes: u64) -> f64 {
    ((bytes as f64 / GB) * 4.0).ceil() / 4.0
}

pub fn kv_cost_at(spec: &specrt::spec::Spec, dtype: specrt::spec::KvDtype, ctx: usize) -> u64 {
    let mut s = spec.clone();
    s.runtime.kv_dtype = dtype;
    (specrt::runtime::kv_block_bytes(&s, 1) / specrt::codegen::PAGE) as u64 * ctx as u64
}

pub fn max_context(a: &ModelAnatomy) -> usize {
    a.max_context.max(1)
}

pub fn initial_ctx(remembered: Option<usize>, default: usize, max: usize) -> usize {
    let floor = MIN_CTX.min(max);
    remembered.unwrap_or(default).clamp(floor, max)
}

pub fn largest_seqs(a: &ModelAnatomy, free: Option<u64>) -> usize {
    let Some(free) = free else { return MAX_SEQS };
    if a.state_bytes_per_seq == 0 {
        return MAX_SEQS;
    }
    let room = free.saturating_sub(a.total_bytes);
    ((room / a.state_bytes_per_seq) as usize).clamp(1, MAX_SEQS)
}

pub fn initial_seqs(remembered: Option<usize>, default: usize, a: &ModelAnatomy, free: Option<u64>) -> usize {
    match remembered {
        Some(r) => r.clamp(1, MAX_SEQS),
        None => default.min(largest_seqs(a, free)).clamp(1, MAX_SEQS),
    }
}

pub struct Described {
    pub spec: specrt::spec::Spec,
    pub anatomy: ModelAnatomy,
    pub structure: Option<specrt::structure::Structure>,
    pub unread: Vec<(String, String)>,
    pub source: &'static str,
}

fn config_extras(path: &std::path::Path) -> (usize, Option<usize>) {
    let Some(v) = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else {
        return (0, None);
    };
    let text = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    let mtp = text["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    (mtp, text["max_position_embeddings"].as_u64().map(|c| c as usize))
}

fn finish(m: &crate::library::LocalModel, spec: specrt::spec::Spec, source: &'static str, mtp: usize, ctx: Option<usize>, with_header: bool) -> Described {
    let mut anatomy = ModelAnatomy::new(&spec, mtp, ctx);
    let mut spec = spec;
    let mut structure = None;
    if with_header
        && let Ok(st) = specrt::structure::Structure::read(&m.files) {
            spec.apply_structure(&st);
            anatomy = ModelAnatomy::new(&spec, anatomy.mtp_layers, Some(anatomy.max_context));
            structure = Some(st);
        }
    Described { spec, anatomy, structure, unread: Vec::new(), source }
}

pub fn describe(m: &crate::library::LocalModel) -> Result<Described, String> {
    use crate::library::Format;
    use specrt::spec::Spec;
    match m.format {
        Format::Gguf => {
            let g = specrt::gguf::Gguf::open(&m.path).map_err(|e| format!("{e:#}"))?;
            let mut spec = g.spec().map_err(|e| format!("{e:#}"))?;
            spec.name = m.name.clone();
            Ok(finish(m, spec, "GGUF metadata", 0, m.context_length.map(|c| c as usize), false))
        }
        Format::Spec => {
            let mut spec = Spec::load(&m.path).map_err(|e| format!("{e:#}"))?;
            let cfg = m.files.first().and_then(|f| f.parent()).map(|d| d.join("config.json"));
            if let Some(text) = cfg.as_deref().and_then(|p| std::fs::read_to_string(p).ok()) {
                let _ = spec.complete_from_checkpoint_config(&text);
            }
            let (mtp, ctx) = cfg.as_deref().map(config_extras).unwrap_or((0, None));
            Ok(finish(m, spec, "spec TOML", mtp, ctx.or(m.context_length.map(|c| c as usize)), true))
        }
        Format::Safetensors => {
            let path = m.path.join("config.json");
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let (mtp, ctx) = config_extras(&path);
            let mut d = describe_hf_config(&m.name, &text, mtp, ctx.or(m.context_length.map(|c| c as usize)))?;
            if let Ok(st) = specrt::structure::Structure::read(&m.files) {
                d.spec.apply_structure(&st);
                d.anatomy = ModelAnatomy::new(&d.spec, d.anatomy.mtp_layers, Some(d.anatomy.max_context));
                d.structure = Some(st);
            }
            Ok(d)
        }
    }
}

pub fn describe_hf_config(name: &str, text: &str, mtp: usize, ctx: Option<usize>) -> Result<Described, String> {
    let mut spec = specrt::spec::Spec::describe_hf_config(text).map_err(|e| format!("{e:#}"))?;
    spec.name = name.to_string();
    spec.vision = specrt::vision::Vision::from_hf_config(text).ok().flatten();
    let anatomy = ModelAnatomy::new(&spec, mtp, ctx);
    Ok(Described { spec, anatomy, structure: None, unread: specrt::spec::unread_hf_keys(text), source: "config.json" })
}

pub fn describe_spec_file(name: &str, path: &std::path::Path, config_text: Option<&str>) -> Result<Described, String> {
    let mut spec = specrt::spec::Spec::load(path).map_err(|e| format!("{e:#}"))?;
    let mut mtp = 0;
    let mut ctx = None;
    if let Some(text) = config_text {
        let _ = spec.complete_from_checkpoint_config(text);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
            let t = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
            mtp = t["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize;
            ctx = t["max_position_embeddings"].as_u64().map(|c| c as usize);
        }
    }
    let anatomy = ModelAnatomy::new(&spec, mtp, ctx);
    let _ = name;
    Ok(Described { spec, anatomy, structure: None, unread: Vec::new(), source: "spec TOML" })
}

pub fn describe_config_json(name: &str, text: &str) -> Result<Described, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let t = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    let mtp = t["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    describe_hf_config(name, text, mtp, t["max_position_embeddings"].as_u64().map(|c| c as usize))
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadPlan {
    pub ctx: usize,
    pub max_ctx: usize,
    pub seqs: usize,
    pub kv_gb: f64,
    pub kv_bytes: u64,
    pub estimate: Estimate,
    pub holds_one_request: bool,
}

impl LoadPlan {
    pub fn fits(&self) -> Option<bool> {
        match self.holds_one_request {
            true => self.estimate.fits(),
            false => Some(false),
        }
    }
}

pub struct PlanInput {
    pub ctx: Option<usize>,
    pub default_ctx: usize,
    pub seqs: Option<usize>,
    pub default_seqs: usize,
    pub kv_gb: Option<f64>,
}

pub fn plan_for(a: &ModelAnatomy, input: &PlanInput, free: Option<u64>) -> LoadPlan {
    let max_ctx = max_context(a);
    let seqs = initial_seqs(input.seqs, input.default_seqs, a, free);
    let ctx = initial_ctx(input.ctx, input.default_ctx, max_ctx);
    let kv_gb = input.kv_gb.filter(|g| *g > 0.0).unwrap_or_else(|| default_kv_gb(a, free, seqs, ctx));
    let kv_bytes = pool_bytes(a, ctx, seqs, kv_gb);
    LoadPlan { ctx, max_ctx, seqs, kv_gb, kv_bytes, estimate: estimate(a, kv_bytes, seqs, free), holds_one_request: pool_holds_one(a, kv_bytes, ctx) }
}
