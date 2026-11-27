use specrt::anatomy::ModelAnatomy;

pub const MIN_CTX: usize = 512;
pub const STEP: usize = 512;
pub const MAX_SEQS: usize = crate::engine::MAX_PARALLEL;
pub const GPU_RESERVE: u64 = (specrt::placement::DEFAULT_RESERVE_GIB * GB) as u64;
pub const TYPICAL_REQUEST: usize = 4096;
pub const GB: f64 = (1u64 << 30) as f64;

pub fn usable(free: u64) -> u64 {
    free.saturating_sub(GPU_RESERVE)
}

pub fn quantised_weight_bytes(spec: &specrt::spec::Spec, format: specrt::spec::Quant) -> Option<u64> {
    use specrt::spec::{LayerQuant, Quant, QuantPlan, QuantizeChoice};
    let covered = match format {
        Quant::Nvfp4G16 => true,
        Quant::Fp8Channel => spec.moe.is_none(),
        _ => false,
    };
    if !covered || spec.quant != Quant::Bf16 || spec.mlp.quant.is_some() {
        return None;
    }
    let mut s = spec.clone();
    s.runtime.quantize = QuantizeChoice::Off;
    let n = s.layers.count;
    s.quant_plan = QuantPlan::default();
    s.quant_plan.layers = (0..n).map(|i| if s.quant_keep.keeps_layer(i, n) { LayerQuant::BF16 } else { LayerQuant::new(format, true) }).collect();
    Some(ModelAnatomy::new(&s, 0, None).total_bytes)
}

pub fn quantised_fit(spec: &specrt::spec::Spec, e: &Estimate) -> Option<u64> {
    let free = e.free?;
    let plain = quantised_plain_bytes(spec)?;
    let quantised = quantised_weight_bytes(spec, specrt::spec::Quant::Nvfp4G16)?;
    let weights = (e.weights as f64 * quantised as f64 / plain as f64) as u64;
    let total = weights + e.kv + e.state;
    (total <= free).then_some(total)
}

fn quantised_plain_bytes(spec: &specrt::spec::Spec) -> Option<u64> {
    use specrt::spec::{Quant, QuantPlan, QuantizeChoice};
    if spec.quant != Quant::Bf16 || spec.mlp.quant.is_some() {
        return None;
    }
    let mut s = spec.clone();
    s.runtime.quantize = QuantizeChoice::Off;
    s.quant_plan = QuantPlan::default();
    Some(ModelAnatomy::new(&s, 0, None).total_bytes).filter(|b| *b > 0)
}

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
    let needed = round_gb(a.kv_bytes_per_token * ctx as u64);
    match free {
        Some(f) => {
            let usable = usable(f);
            let room = usable.saturating_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64);
            round_gb_down(room).max(needed)
        }
        None => needed,
    }
}

pub fn max_kv_gb(a: &ModelAnatomy, free: Option<u64>, seqs: usize, ctx: usize) -> f64 {
    let needed = a.kv_bytes_per_token * ctx as u64;
    let room = room_bytes(a, free, seqs).unwrap_or(needed * 4);
    round_gb(room.max(needed)).max(default_kv_gb(a, free, seqs, ctx)).max(0.25)
}

fn round_gb_down(bytes: u64) -> f64 {
    ((bytes as f64 / GB) * 4.0).floor() / 4.0
}

fn round_gb(bytes: u64) -> f64 {
    ((bytes as f64 / GB) * 4.0).ceil() / 4.0
}

pub fn kv_cost_at(spec: &specrt::spec::Spec, dtype: specrt::spec::KvDtype, ctx: usize) -> u64 {
    let mut s = spec.clone();
    s.runtime.kv_dtype = dtype;
    (specrt::runtime::kv_block_bytes(&s.for_kv_context(ctx), 1) / specrt::codegen::PAGE) as u64 * ctx as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKind {
    Auto,
    F16,
    Fp8,
    MixedFp8,
}

impl KvKind {
    pub const ALL: [KvKind; 4] = [KvKind::Auto, KvKind::F16, KvKind::Fp8, KvKind::MixedFp8];

    pub fn label(self) -> &'static str {
        match self {
            KvKind::Auto => "auto",
            KvKind::F16 => "fp16 / bf16",
            KvKind::Fp8 => "fp8",
            KvKind::MixedFp8 => "mixed fp8",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            KvKind::Auto => "The profile and the planner choose.",
            KvKind::F16 => "Full precision, the most memory.",
            KvKind::Fp8 => "One byte per value, about half the memory.",
            KvKind::MixedFp8 => "fp8 with the early layers kept at f16, the safest fp8.",
        }
    }
}

pub fn kv_bytes_per_token(spec: &specrt::spec::Spec, kind: KvKind, ctx: usize) -> u64 {
    use specrt::spec::KvDtype;
    let mut s = spec.clone();
    match kind {
        KvKind::Auto => {}
        KvKind::F16 => s.runtime.kv_dtype = KvDtype::F16,
        KvKind::Fp8 => {
            s.runtime.kv_dtype = KvDtype::Fp8;
            s.runtime.kv_f16_layers = Some(0);
        }
        KvKind::MixedFp8 => {
            s.runtime.kv_dtype = KvDtype::Fp8;
            s.runtime.kv_f16_layers = None;
        }
    }
    (specrt::runtime::kv_block_bytes(&s.for_kv_context(ctx), 1) / specrt::codegen::PAGE) as u64
}

pub fn kv_gb_for_tokens(bytes_per_token: u64, tokens: usize) -> f64 {
    bytes_per_token as f64 * tokens as f64 / GB
}

pub fn tokens_for_kv_gb(bytes_per_token: u64, gb: f64) -> usize {
    if bytes_per_token == 0 { usize::MAX } else { (gb.max(0.0) * GB / bytes_per_token as f64).floor() as usize }
}

pub fn fitting_context(a: &ModelAnatomy, free: Option<u64>, seqs: usize) -> Option<usize> {
    let usable = usable(free?);
    let room = usable.checked_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64)?;
    let tokens = pool_tokens(a, room).min(a.max_context.max(1));
    Some(tokens.max(1))
}

pub fn fitting_context_each(a: &ModelAnatomy, free: Option<u64>, seqs: usize) -> Option<usize> {
    let usable = usable(free?);
    let room = usable.checked_sub(a.total_bytes + a.state_bytes_per_seq * seqs.max(1) as u64)?;
    let tokens = (pool_tokens(a, room) / seqs.max(1)).min(a.max_context.max(1));
    Some(tokens.max(1))
}

pub fn kv_kind_effective(choice: &specrt::options::EngineChoice) -> KvKind {
    match choice.value("runtime.kv_dtype").as_str() {
        Some("fp8") if choice.value("runtime.kv_f16_layers").as_u64() == Some(0) => KvKind::Fp8,
        Some("fp8") => KvKind::MixedFp8,
        _ => KvKind::F16,
    }
}

pub fn kv_kind_of(choice: &specrt::options::EngineChoice) -> KvKind {
    if !choice.overrides.contains_key("runtime.kv_dtype") && !choice.overrides.contains_key("runtime.kv_f16_layers") {
        return KvKind::Auto;
    }
    kv_kind_effective(choice)
}

pub fn set_kv_kind(choice: &mut specrt::options::EngineChoice, kind: KvKind) {
    use serde_json::json;
    match kind {
        KvKind::Auto => {
            choice.reset("runtime.kv_dtype");
            choice.reset("runtime.kv_f16_layers");
        }
        KvKind::F16 => {
            choice.overrides.insert("runtime.kv_dtype".into(), json!("f16"));
            choice.reset("runtime.kv_f16_layers");
        }
        KvKind::Fp8 => {
            choice.set("runtime.kv_dtype", json!("fp8"));
            choice.set("runtime.kv_f16_layers", json!(0));
        }
        KvKind::MixedFp8 => {
            choice.set("runtime.kv_dtype", json!("fp8"));
            choice.reset("runtime.kv_f16_layers");
        }
    }
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
    let mtp = text["mtp_num_hidden_layers"].as_u64().or_else(|| text["num_nextn_predict_layers"].as_u64()).unwrap_or(0) as usize;
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

pub fn as_served(m: &crate::library::LocalModel, d: &mut Described) {
    if m.format != crate::library::Format::Safetensors {
        return;
    }
    let Some(served) = std::fs::read_to_string(m.path.join("config.json")).ok().and_then(|t| specrt::spec::Spec::from_hf_config_serving(&t).ok()) else { return };
    let mtp = if served.speculative().is_some() { d.anatomy.mtp_layers } else { 0 };
    d.spec.mtp = served.mtp;
    d.spec.mixer = served.mixer;
    d.anatomy = ModelAnatomy::new(&d.spec, mtp, Some(d.anatomy.max_context));
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
            mtp = t["mtp_num_hidden_layers"].as_u64().or_else(|| t["num_nextn_predict_layers"].as_u64()).unwrap_or(0) as usize;
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
    let mtp = t["mtp_num_hidden_layers"].as_u64().or_else(|| t["num_nextn_predict_layers"].as_u64()).unwrap_or(0) as usize;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> specrt::spec::Spec {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../specrt/tests/fixtures/qwen3.5-4b-base.config.json")).unwrap();
        specrt::spec::Spec::from_hf_config(&text).unwrap()
    }

    #[test]
    fn fp8_kv_costs_less_per_token_than_f16_and_mixed_sits_between() {
        let s = spec();
        let f16 = kv_bytes_per_token(&s, KvKind::F16, 8192);
        let fp8 = kv_bytes_per_token(&s, KvKind::Fp8, 8192);
        let mixed = kv_bytes_per_token(&s, KvKind::MixedFp8, 8192);
        assert!(fp8 < mixed && mixed < f16, "{fp8} {mixed} {f16}");
        assert_eq!(kv_bytes_per_token(&s, KvKind::Auto, 8192), f16);
    }

    #[test]
    fn context_to_gb_and_back_per_kv_type() {
        let s = spec();
        for kind in KvKind::ALL {
            let per = kv_bytes_per_token(&s, kind, 32768);
            let gb = kv_gb_for_tokens(per, 32768);
            let back = tokens_for_kv_gb(per, gb);
            assert!(back.abs_diff(32768) <= 1, "{kind:?} {back}");
            assert!(kv_gb_for_tokens(per, 65536) > gb);
        }
        assert_eq!(tokens_for_kv_gb(0, 1.0), usize::MAX);
        assert_eq!(tokens_for_kv_gb(1024, -1.0), 0);
    }

    #[test]
    fn kv_kind_round_trips_through_the_engine_choice() {
        let mut c = specrt::options::EngineChoice::default();
        assert_eq!(kv_kind_of(&c), KvKind::Auto);
        for kind in KvKind::ALL {
            set_kv_kind(&mut c, kind);
            assert_eq!(kv_kind_of(&c), kind);
        }
        set_kv_kind(&mut c, KvKind::Fp8);
        assert_eq!(c.value("runtime.kv_f16_layers"), json!(0));
        set_kv_kind(&mut c, KvKind::Auto);
        assert!(!c.is_custom());
    }

    #[test]
    fn the_fitting_context_shrinks_with_less_free_memory_and_is_capped_by_the_model() {
        let s = spec();
        let a = specrt::anatomy::ModelAnatomy::new(&s, 0, None);
        let big = fitting_context(&a, Some(96 << 30), 1).unwrap();
        let small = fitting_context(&a, Some(a.total_bytes + GPU_RESERVE + (2 << 30)), 1).unwrap();
        assert!(small < big);
        assert!(big <= a.max_context.max(1));
        assert_eq!(fitting_context(&a, Some(1 << 20), 1), None);
        assert_eq!(fitting_context(&a, None, 1), None);
        let tight = Some(a.total_bytes + GPU_RESERVE + a.state_bytes_per_seq * 4 + (2 << 30));
        let each = fitting_context_each(&a, tight, 4).unwrap();
        let pooled = fitting_context(&a, tight, 4).unwrap();
        assert_eq!(fitting_context_each(&a, tight, 1), fitting_context(&a, tight, 1));
        assert!(pooled < a.max_context && each.abs_diff(pooled / 4) <= 1, "{each} {pooled}");
    }
}
