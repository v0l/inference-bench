use crate::spec::{ActQuant, Activation, Quant, RuntimeOptions, Spec, TileChoice};
use anyhow::Result;
use kir::gemm::{self as kg, Epilogue, Fp4Gemv, Fp8Scale, GemmShape, KernelInstance, Policy, Request, Tuning, Variant};
use std::collections::BTreeMap;

pub use kg::{BatchShape, GemmPath, MmaShape, Tile, WeightFormat};
pub use kg::{BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, FP4_GEMV_THREADS, GGB_LINEARS, GGB_MIN_TOKENS, GGML_GEMV_THREADS, ROWS_MAX_TOKENS};
pub use kg::ggml_gemv_rows;

pub const MMA_LINEARS: [&str; 5] = ["qkv", "o", "gate_up", "down", "lm_head"];
pub const GDN_LINEARS: [&str; 3] = ["lin_in", "lin_out", "lin_ba"];
pub const MIX_LINEARS: [&str; 2] = ["mix_in", "mix_out"];
pub const FP8_ACT_LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
pub const FP4_LINEARS: [&str; 2] = ["gate_up", "down"];

pub const LATENT_LINEARS: [&str; 4] = ["latent_down", "latent_up", "shared_gate_up", "shared_down"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    Bf16,
    Q8,
    Fp8,
    Fp4,
    Ggml,
}

impl WeightKind {
    pub fn format(self) -> WeightFormat {
        match self {
            WeightKind::Bf16 => WeightFormat::Bf16,
            WeightKind::Q8 => WeightFormat::Q8,
            WeightKind::Fp8 => WeightFormat::Fp8,
            WeightKind::Fp4 => WeightFormat::Fp4,
            WeightKind::Ggml => WeightFormat::Ggml,
        }
    }
}

pub fn weight_format(quant: Quant) -> WeightFormat {
    match quant {
        Quant::Bf16 => WeightFormat::Bf16,
        Quant::Q8 => WeightFormat::Q8,
        Quant::Fp8Block128 | Quant::Fp8Channel => WeightFormat::Fp8,
        Quant::Gguf => WeightFormat::Ggml,
        Quant::Nvfp4G16 => WeightFormat::Fp4,
    }
}

pub fn head_format(spec: &Spec) -> WeightFormat {
    match spec.lm_head_quant() {
        Quant::Nvfp4G16 => WeightFormat::Bf16,
        q => weight_format(q),
    }
}

pub fn fp4_act_enabled(spec: &Spec) -> bool {
    spec.mlp.act_quant == ActQuant::Fp4
}

pub fn fp8_act_enabled(spec: &Spec) -> bool {
    spec.act_quant == ActQuant::Fp8
}

pub fn tuning(opts: &RuntimeOptions) -> Tuning {
    Tuning {
        tile: match opts.tile {
            TileChoice::Auto => kg::TileChoice::Auto,
            TileChoice::Small => kg::TileChoice::Small,
            TileChoice::Big => kg::TileChoice::Big,
            TileChoice::Narrow => kg::TileChoice::Narrow,
        },
        big: opts.big_tile.map(Tile::from_array),
        fp4_big: opts.fp4_tile.map(Tile::from_array),
        fp8_huge: opts.fp8_tile.map(Tile::from_array),
        fp4_stages: opts.fp4_stages,
    }
}

pub fn variant_tile(spec: &Spec, suffix: &str) -> Tile {
    let t = tuning(&spec.runtime);
    match suffix {
        "_big" => t.tile(Variant::Big),
        "_narrow" => t.tile(Variant::Narrow),
        "_huge" => t.tile(Variant::Huge),
        "_tiny" => t.tile(Variant::Tiny),
        "_skinny" => t.tile(Variant::Skinny),
        "_skinnywide" => t.tile(Variant::SkinnyWide),
        "_f4a_big" => t.fp4_tile(Variant::Big),
        _ => t.tile(Variant::Small),
    }
}

fn fp8_scale(spec: &Spec) -> Option<Fp8Scale> {
    match spec.quant {
        Quant::Fp8Block128 => Some(Fp8Scale::Block),
        Quant::Fp8Channel => Some(Fp8Scale::Channel),
        _ => None,
    }
}

pub fn policy(spec: &Spec) -> Policy {
    Policy {
        tuning: tuning(&spec.runtime),
        fp8_weights: fp8_scale(spec),
        fp8_act: spec.quant.is_fp8() && fp8_act_enabled(spec),
        fp4_act: spec.mlp.quant == Some(Quant::Nvfp4G16) && fp4_act_enabled(spec),
    }
}

fn act(spec: &Spec) -> kir::Activation {
    match spec.mlp.activation {
        Activation::Swiglu | Activation::Mapped => kir::Activation::Silu,
        Activation::Geglu => kir::Activation::GeluTanh,
    }
}

fn gate_up_epilogue(spec: &Spec) -> Epilogue {
    if spec.mlp.mapped.is_some() { Epilogue::Map } else { Epilogue::Act }
}

pub fn linear_shape(spec: &Spec, linear: &str) -> GemmShape {
    let d = &spec.dims;
    let la = spec.linear_attention.as_ref();
    let (n, k, epi, bias) = match linear {
        "qkv" => (spec.qkv_rows(), d.hidden, Epilogue::Store, spec.attention.qkv_bias),
        "o" => (d.hidden, spec.o_dim(), Epilogue::Residual, spec.attention.o_bias),
        "gate_up" => (spec.gate_up_rows(), d.hidden, gate_up_epilogue(spec), false),
        "gate_up_wide" => (spec.gate_up_rows_wide(), d.hidden, gate_up_epilogue(spec), false),
        "down" => (d.hidden, d.intermediate, Epilogue::Residual, false),
        "down_wide" => (d.hidden, 2 * d.intermediate, Epilogue::Residual, false),
        "lin_in" => (la.map_or(0, |l| l.in_proj_rows()), d.hidden, Epilogue::Store, false),
        "lin_ba" => (la.map_or(0, |l| l.ba_rows()), d.hidden, Epilogue::Store, false),
        "lin_out" => (d.hidden, la.map_or(0, |l| l.value_dim()), Epilogue::Residual, false),
        "mix_in" => (spec.mix_in_rows(), d.hidden, Epilogue::Store, false),
        "mix_out" => (d.hidden, spec.mix_out_cols(), Epilogue::Residual, false),
        "mtp_fc" => (d.hidden, 2 * d.hidden, Epilogue::Store, false),
        "mtp_qkv" => (spec.qkv_rows(), d.hidden, Epilogue::Store, false),
        "mtp_o" => (d.hidden, spec.q_dim(), Epilogue::Residual, false),
        "mtp_gate_up" => (2 * d.intermediate, d.hidden, Epilogue::Act, false),
        "mtp_down" => (d.hidden, d.intermediate, Epilogue::Residual, false),
        "latent_down" => (spec.moe.as_ref().map_or(0, |m| m.width(d.hidden)), d.hidden, Epilogue::Store, false),
        "latent_up" => (d.hidden, spec.moe.as_ref().map_or(0, |m| m.width(d.hidden)), Epilogue::Residual, false),
        "shared_gate_up" => (2 * spec.moe.as_ref().map_or(0, |m| m.shared_intermediate), d.hidden, Epilogue::Act, false),
        "shared_down" => (d.hidden, spec.moe.as_ref().map_or(0, |m| m.shared_intermediate), Epilogue::Residual, false),
        _ => (spec.head_rows().len(), d.hidden, Epilogue::Store, false),
    };
    GemmShape::new(n, k, epi, bias).with_act(act(spec))
}

pub fn linear_dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str, bool) {
    let s = linear_shape(spec, linear);
    (s.n, s.k, s.epi.cuda(), s.bias)
}

fn split(spec: &Spec, linear: &str, wf: WeightFormat) -> bool {
    wf != WeightFormat::Q8 && spec.split_activations() && !spec.runtime.unsplit.iter().any(|n| n == linear)
}

pub type KernelChoice = kg::Choice;

pub fn request<'a>(spec: &Spec, linear: &'a str, tokens: usize, weight: WeightKind) -> Request<'a> {
    let wf = weight.format();
    Request { linear, shape: linear_shape(spec, linear), tokens, weight: wf, split: split(spec, linear, wf) }
}

pub fn select(spec: &Spec, target: &kir::Target, linear: &str, tokens: usize, weight: WeightKind) -> Result<KernelChoice> {
    select_act(spec, target, linear, tokens, weight, true)
}

pub const GEMV_BLOCKS_PER_SM: u32 = kir::gemv::GEMV_BLOCKS_PER_SM;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemvLaunch {
    pub name: &'static str,
    pub rows: usize,
    pub threads: u32,
    pub rows_per_block: usize,
    pub blocks_per_sm: Option<u32>,
    pub bias: bool,
    pub act: bool,
}

impl GemvLaunch {
    pub fn grid(&self, sms: u32) -> u32 {
        let blocks = self.rows.div_ceil(self.rows_per_block) as u32;
        self.blocks_per_sm.map_or(blocks, |b| blocks.min(sms * b))
    }
}

fn gemv_name(linear: &str, normed: bool, fp4: bool, gdn: bool) -> Option<(&'static str, bool)> {
    let fp4_name = match (linear, normed) {
        ("gate_up", true) => Some("gate_up_act_f4"),
        ("lm_head", true) => Some("lm_head_norm_f4"),
        ("down", false) => Some("down_proj_f4"),
        _ => None,
    };
    if fp4 {
        return fp4_name.map(|n| (n, true));
    }
    let name = match (linear, normed) {
        ("qkv", true) => "qkv_proj_norm",
        ("gate_up", true) => "gate_up_act",
        ("lm_head", true) => "lm_head_norm",
        ("lin_in", true) if gdn => "lin_in_norm",
        ("lin_ba", true) if gdn => "lin_ba_norm",
        ("qkv", false) => "qkv_proj",
        ("o", false) => "o_proj",
        ("gate_up", false) => "gate_up_proj",
        ("down", false) => "down_proj",
        ("lin_out", false) if gdn => "lin_out_proj",
        ("lm_head", false) => "lm_head",
        _ => return None,
    };
    Some((name, false))
}

pub fn gemv(spec: &Spec, linear: &str, weight: WeightKind, normed: bool) -> Option<GemvLaunch> {
    gemv_into(spec, linear, weight, normed, false)
}

fn store_name(linear: &str) -> Option<&'static str> {
    match linear {
        "o" => Some("o_proj_store"),
        "down" => Some("down_proj_store"),
        _ => None,
    }
}

pub fn gemv_into(spec: &Spec, linear: &str, weight: WeightKind, normed: bool, store: bool) -> Option<GemvLaunch> {
    let s = linear_shape(spec, linear);
    if normed && s.epi == Epilogue::Map {
        return None;
    }
    if store {
        let plain = gemv_into(spec, linear, weight, normed, false)?;
        let name = store_name(linear).filter(|_| !normed && s.epi == Epilogue::Residual && weight != WeightKind::Fp4 && spec.runtime.store_epilogue && spec.uses_branch())?;
        return Some(GemvLaunch { name, ..plain });
    }
    let (name, fp4) = gemv_name(linear, normed, weight == WeightKind::Fp4, spec.linear_attention.is_some())?;
    let act = normed && s.epi == Epilogue::Act;
    let rows = if act { spec.gate_up_out(s.n) } else { s.n };
    let g = kir::gemv::gemv_geometry(weight.format(), rows, act, s.epi == Epilogue::Store, fp4);
    Some(GemvLaunch { name, rows, threads: g.threads, rows_per_block: g.rows_per_block, blocks_per_sm: g.blocks_per_sm, bias: !act, act })
}

pub fn select_act(spec: &Spec, target: &kir::Target, linear: &str, tokens: usize, weight: WeightKind, act: bool) -> Result<KernelChoice> {
    let policy = Policy { fp4_act: policy(spec).fp4_act && act, ..policy(spec) };
    kg::select(target, &policy, &request(spec, linear, tokens, weight)).map_err(anyhow::Error::msg)
}

pub fn batch_part_len(spec: &Spec, target: &kir::Target, fp4: bool) -> usize {
    let linears: Vec<&str> = if fp4 {
        FP4_LINEARS.iter().copied().chain(spec.requanted_linears()).collect()
    } else {
        kg::BATCH_LINEARS.iter().copied().filter(|l| spec.linear_attention.is_some() || !l.starts_with("lin_")).collect()
    };
    let most = linears
        .into_iter()
        .map(|l| linear_shape(spec, l))
        .chain(fp4.then(|| linear_shape(spec, "lm_head")))
        .map(|s| kg::batch_part_len(target, s.n, s.k))
        .max()
        .unwrap_or(1);
    most.max(BATCH_MAX_TOKENS)
}

pub fn ggml_part_len(spec: &Spec, target: &kir::Target) -> usize {
    GGB_LINEARS.iter().copied().chain(["lm_head"]).map(|l| linear_shape(spec, l)).map(|s| kg::ggml_part_len(target, s.n, s.k)).max().unwrap_or(1).max(1)
}

pub struct GemmSet {
    pub kernels: Vec<KernelInstance>,
    pub refused: Vec<String>,
    lenient: bool,
}

pub fn intern(name: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&n) = names.get(name) {
        return n;
    }
    let n: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.insert(n);
    n
}

pub const KERNEL_SELECT: &str = "kernel_select";

fn through_heavy(k: KernelInstance) -> Result<KernelInstance> {
    match kir::heavy::Heavy::lift(&k).map_err(anyhow::Error::msg)? {
        Some(h) => h.compile(&k.target, KERNEL_SELECT).map_err(anyhow::Error::msg),
        None => Ok(k),
    }
}

pub const PROLOGUE_ATTACH: &str = "prologue_attach";
pub const EPILOGUE_ATTACH: &str = "epilogue_attach";

pub fn normed(t: &kir::Target, name: &str, family: kir::heavy::Family, n: usize, k: usize, bias: bool, gated: bool) -> std::result::Result<kir::heavy::Heavy, kir::heavy::HeavyError> {
    use kir::heavy::{self, Heavy, HeavyError, Problem, Slot};
    let lower = |pass: &str, reason: String| HeavyError::Lower { pass: pass.into(), kernel: name.into(), reason };
    let mut h = Heavy::new(name, family, Problem { n, k, act: kir::Activation::Silu }, None);
    h.epilogue = heavy::epilogue(Epilogue::Store, bias, n).map_err(|e| lower(PROLOGUE_ATTACH, e))?;
    let h = h.attach(t, PROLOGUE_ATTACH, Slot::Prologue, heavy::norm_prologue())?;
    if !gated {
        return Ok(h);
    }
    let act = heavy::epilogue_into(Epilogue::Act, false, n, heavy::OUT).map_err(|e| lower(EPILOGUE_ATTACH, e))?;
    h.attach(t, EPILOGUE_ATTACH, Slot::Epilogue, act)
}

fn normed_fp4(t: &kir::Target, name: &str, shape: GemmShape, gated: bool) -> std::result::Result<KernelInstance, kir::heavy::HeavyError> {
    normed(t, name, kir::heavy::Family::Fp4Gemv, shape.n, shape.k, false, gated)?.compile(t, KERNEL_SELECT)
}

impl GemmSet {
    fn push_fused(&mut self, k: std::result::Result<KernelInstance, kir::heavy::HeavyError>) -> Result<()> {
        match k {
            Ok(k) => self.kernels.push(k),
            Err(kir::heavy::HeavyError::Gemm { error, .. }) if self.lenient && error.is_unsupported() => self.refused.push(error.to_string()),
            Err(e) => return Err(anyhow::Error::msg(e)),
        }
        Ok(())
    }
    fn push(&mut self, k: std::result::Result<KernelInstance, kg::GemmError>) -> Result<()> {
        match k {
            Ok(k) => self.kernels.push(through_heavy(k)?),
            Err(e) if self.lenient && e.is_unsupported() => self.refused.push(e.to_string()),
            Err(e) => return Err(anyhow::Error::msg(e)),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn tiles(&mut self, t: &kir::Target, names: &str, shape: GemmShape, wf: WeightFormat, split: bool, token_major: bool, tuning: &Tuning) -> Result<()> {
        let variants: &[Variant] = if t.language == kir::Language::Wgsl { &[Variant::Small] } else { &Variant::MMA };
        for &v in variants {
            let name = format!("{names}{}", v.suffix());
            self.push(KernelInstance::gemm(t, &name, shape, wf, tuning.mma_tile(v, wf, shape.n), split, token_major))?;
        }
        Ok(())
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.kernels.iter().map(|k| intern(&k.name)).collect()
    }

    pub fn smem(&self) -> BTreeMap<&'static str, u32> {
        self.kernels
            .iter()
            .filter(|k| k.tile().is_some() || k.batch_shape().is_some() || matches!(k.kind, kg::Kind::GgmlGemv { .. }))
            .map(|k| (intern(&k.name), k.smem_bytes))
            .collect()
    }
}

fn wide_of(spec: &Spec, base: &[&'static str]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for &l in base {
        out.push(l);
        if spec.mlp.wide_shared_tail && spec.layers.shared_tail > 0 {
            match l {
                "gate_up" => out.push("gate_up_wide"),
                "down" => out.push("down_wide"),
                _ => {}
            }
        }
    }
    out
}

pub fn gemm_kernels(spec: &Spec, t: &kir::Target) -> Result<GemmSet> {
    let tuning = tuning(&spec.runtime);
    let la = spec.linear_attention.is_some();
    let dequant = t.language == kir::Language::Wgsl;
    let mut set = GemmSet { kernels: Vec::new(), refused: Vec::new(), lenient: dequant };
    let gdn: &[&str] = if la { &GDN_LINEARS } else if spec.mixer.is_some() { &MIX_LINEARS } else { &[] };
    let mma = t.has(kir::Feature::Mma16816);
    if !mma && !dequant {
        return Ok(set);
    }
    for linear in wide_of(spec, &MMA_LINEARS).into_iter().chain(gdn.iter().copied()).chain(if spec.moe.as_ref().is_some_and(|m| m.latent.is_some()) { &LATENT_LINEARS[..] } else { &[] }.iter().copied()).collect::<Vec<_>>() {
        let wf = match linear {
            "lm_head" => head_format(spec),
            "lin_ba" => WeightFormat::Bf16,
            _ => weight_format(spec.quant),
        };
        set.tiles(t, &format!("{linear}_mma"), linear_shape(spec, linear), wf, split(spec, linear, wf), false, &tuning)?;
    }
    if dequant {
        if spec.mlp.quant == Some(Quant::Nvfp4G16) {
            for linear in wide_of(spec, &FP4_LINEARS) {
                let shape = linear_shape(spec, linear);
                set.tiles(t, &format!("{linear}_f4w"), shape, WeightFormat::Fp4, true, false, &tuning)?;
                set.push(KernelInstance::batch(t, &format!("{linear}_f4b8"), shape, WeightFormat::Fp4, 8))?;
                set.push(KernelInstance::fp4_act(t, &format!("{linear}_f4a"), shape, Tile::SMALL, &tuning))?;
            }
            let gu = linear_shape(spec, "gate_up");
            set.push_fused(normed_fp4(t, "gate_up_act_f4", gu, true))?;
        }
        if let Some(scale) = fp8_scale(spec) {
            let shape = linear_shape(spec, "qkv");
            set.push(KernelInstance::batch(t, "qkv_f8b8", shape, WeightFormat::Fp8, 8))?;
            set.push(KernelInstance::fp8_act(t, "qkv_f8", shape, Tile::SMALL, scale))?;
        }
        return Ok(set);
    }
    if spec.quant == Quant::Gguf {
        for linear in GGB_LINEARS {
            set.tiles(t, &format!("{linear}_ggb"), linear_shape(spec, linear), WeightFormat::Bf16, split(spec, linear, WeightFormat::Bf16), true, &tuning)?;
        }
        if mma {
            for linear in GGB_LINEARS.iter().copied().chain(["lm_head"]) {
                let shape = linear_shape(spec, linear);
                for tokens in kg::GGML_VARIANTS {
                    set.push(KernelInstance::ggml_batch(t, &format!("{linear}_ggr{tokens}"), shape, tokens, split(spec, linear, WeightFormat::Ggml)))?;
                }
                set.push(KernelInstance::batch_reduce(t, &format!("{linear}_ggr_reduce"), shape, WeightFormat::Ggml))?;
                if shape.k <= kg::GGML_GEMV_MAX_K {
                    set.push(KernelInstance::ggml_gemv(t, &format!("{linear}_ggv"), shape, split(spec, linear, WeightFormat::Ggml)))?;
                }
            }
        }
    }
    if spec.speculative().is_some() {
        for linear in crate::mtp::MTP_LINEARS {
            set.tiles(t, &format!("{linear}_mma"), linear_shape(spec, linear), WeightFormat::Bf16, split(spec, linear, WeightFormat::Bf16), false, &tuning)?;
        }
    }
    if let Some(scale) = fp8_scale(spec) {
        if t.has(kir::Feature::Fp8Mma) {
            set.push(KernelInstance::quant_fp8(t, Fp8Scale::Block))?;
            set.push(KernelInstance::quant_fp8(t, Fp8Scale::Channel))?;
            for linear in wide_of(spec, &FP8_ACT_LINEARS).into_iter().filter(|l| la || !l.starts_with("lin_")) {
                let shape = linear_shape(spec, linear);
                for v in Variant::FP8_ACT {
                    set.push(KernelInstance::fp8_act(t, &format!("{linear}_f8{}", v.suffix()), shape, tuning.tile(v), scale))?;
                }
            }
        }
        for linear in wide_of(spec, &kg::BATCH_LINEARS).into_iter().filter(|l| la || !l.starts_with("lin_")) {
            let shape = linear_shape(spec, linear);
            for tokens in BATCH_VARIANTS {
                set.push(KernelInstance::batch(t, &format!("{linear}_f8b{tokens}"), shape, WeightFormat::Fp8, tokens))?;
            }
            set.push(KernelInstance::batch_reduce(t, &format!("{linear}_f8b_reduce"), shape, WeightFormat::Fp8))?;
        }
    }
    let fp4_mlp = spec.mlp.quant == Some(Quant::Nvfp4G16) || spec.serves_fp4_mlp();
    if fp4_mlp {
        let (gu, dn) = (linear_shape(spec, "gate_up"), linear_shape(spec, "down"));
        set.push_fused(normed_fp4(t, "gate_up_act_f4", gu, true))?;
        set.push(KernelInstance::fp4_gemv(t, "down_proj_f4", Fp4Gemv::Down, dn.n, dn.k))?;
        for linear in wide_of(spec, &FP4_LINEARS) {
            let shape = linear_shape(spec, linear);
            set.tiles(t, &format!("{linear}_f4w"), shape, WeightFormat::Fp4, true, false, &tuning)?;
            for tokens in BATCH_VARIANTS {
                set.push(KernelInstance::batch(t, &format!("{linear}_f4b{tokens}"), shape, WeightFormat::Fp4, tokens))?;
            }
            set.push(KernelInstance::batch_reduce(t, &format!("{linear}_f4b_reduce"), shape, WeightFormat::Fp4))?;
        }
        if fp4_act_enabled(spec) && t.has(kir::Feature::BlockScaledFp4) {
            set.push(KernelInstance::quant_fp4(t))?;
            for linear in wide_of(spec, &FP4_LINEARS) {
                let shape = linear_shape(spec, linear);
                for v in Variant::FP4_ACT {
                    set.push(KernelInstance::fp4_act(t, &format!("{linear}_f4a{}", v.suffix()), shape, tuning.fp4_tile(v), &tuning))?;
                }
            }
        }
    }
    for linear in spec.requanted_linears() {
        let shape = linear_shape(spec, linear);
        set.tiles(t, &format!("{linear}_f4w"), shape, WeightFormat::Fp4, true, false, &tuning)?;
        for tokens in BATCH_VARIANTS {
            set.push(KernelInstance::batch(t, &format!("{linear}_f4b{tokens}"), shape, WeightFormat::Fp4, tokens))?;
        }
        set.push(KernelInstance::batch_reduce(t, &format!("{linear}_f4b_reduce"), shape, WeightFormat::Fp4))?;
    }
    if spec.lm_head_quant() == Quant::Nvfp4G16 {
        let shape = linear_shape(spec, "lm_head");
        set.push_fused(normed_fp4(t, "lm_head_norm_f4", shape, false))?;
        set.tiles(t, "lm_head_f4w", shape, WeightFormat::Fp4, true, false, &tuning)?;
    }
    if let Some(q) = spec.draft_head_quant() {
        let shape = linear_shape(spec, "lm_head");
        let (name, wf) = if q == Quant::Nvfp4G16 { ("mtp_head_f4w", WeightFormat::Fp4) } else { ("mtp_head_mma", weight_format(q)) };
        set.tiles(t, name, shape, wf, true, false, &tuning)?;
        if wf == WeightFormat::Fp4 {
            for tokens in BATCH_VARIANTS {
                set.push(KernelInstance::batch(t, &format!("mtp_head_f4b{tokens}"), shape, WeightFormat::Fp4, tokens))?;
            }
            set.push(KernelInstance::batch_reduce(t, "mtp_head_f4b_reduce", shape, WeightFormat::Fp4))?;
        }
    }
    Ok(set)
}
