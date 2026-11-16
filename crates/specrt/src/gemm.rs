use crate::spec::{ActQuant, Activation, Quant, RuntimeOptions, Spec, TileChoice};
use anyhow::Result;
use kir::gemm::{self as kg, Epilogue, Fp4Gemv, Fp8Scale, GemmShape, KernelInstance, Policy, Request, Tuning, Variant};
use std::collections::BTreeMap;

pub use kg::{BatchShape, GemmPath, MmaShape, Tile, WeightFormat};
pub use kg::{BATCH_MAX_TOKENS, BATCH_REDUCE_BLOCK, BATCH_VARIANTS, FP4_GEMV_THREADS, GGB_LINEARS, GGB_MIN_TOKENS, ROWS_MAX_TOKENS};

pub const MMA_LINEARS: [&str; 5] = ["qkv", "o", "gate_up", "down", "lm_head"];
pub const GDN_LINEARS: [&str; 3] = ["lin_in", "lin_out", "lin_ba"];
pub const FP8_ACT_LINEARS: [&str; 6] = ["qkv", "o", "gate_up", "down", "lin_in", "lin_out"];
pub const FP4_LINEARS: [&str; 2] = ["gate_up", "down"];

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
        Activation::Swiglu => kir::Activation::Silu,
        Activation::Geglu => kir::Activation::GeluTanh,
    }
}

pub fn linear_shape(spec: &Spec, linear: &str) -> GemmShape {
    let d = &spec.dims;
    let la = spec.linear_attention.as_ref();
    let (n, k, epi, bias) = match linear {
        "qkv" => (spec.qkv_rows(), d.hidden, Epilogue::Store, spec.attention.qkv_bias),
        "o" => (d.hidden, spec.q_dim(), Epilogue::Residual, false),
        "gate_up" => (2 * d.intermediate, d.hidden, Epilogue::Act, false),
        "down" => (d.hidden, d.intermediate, Epilogue::Residual, false),
        "lin_in" => (la.map_or(0, |l| l.in_proj_rows()), d.hidden, Epilogue::Store, false),
        "lin_ba" => (la.map_or(0, |l| l.ba_rows()), d.hidden, Epilogue::Store, false),
        "lin_out" => (d.hidden, la.map_or(0, |l| l.value_dim()), Epilogue::Residual, false),
        "mtp_fc" => (d.hidden, 2 * d.hidden, Epilogue::Store, false),
        "mtp_qkv" => (spec.qkv_rows(), d.hidden, Epilogue::Store, false),
        "mtp_o" => (d.hidden, spec.q_dim(), Epilogue::Residual, false),
        "mtp_gate_up" => (2 * d.intermediate, d.hidden, Epilogue::Act, false),
        "mtp_down" => (d.hidden, d.intermediate, Epilogue::Residual, false),
        _ => (d.vocab, d.hidden, Epilogue::Store, false),
    };
    GemmShape::new(n, k, epi, bias).with_act(act(spec))
}

pub fn linear_dims(spec: &Spec, linear: &str) -> (usize, usize, &'static str, bool) {
    let s = linear_shape(spec, linear);
    (s.n, s.k, s.epi.cuda(), s.bias)
}

fn split(spec: &Spec, linear: &str, wf: WeightFormat) -> bool {
    wf != WeightFormat::Q8 && !spec.runtime.unsplit.iter().any(|n| n == linear)
}

pub type KernelChoice = kg::Choice;

pub fn kernel_for(spec: &Spec, target: &kir::Target, linear: &str, tokens: usize, weight: WeightKind) -> Result<KernelChoice> {
    let wf = weight.format();
    let shape = linear_shape(spec, linear);
    let r = Request { linear, shape, tokens, weight: wf, split: split(spec, linear, wf) };
    kg::select(target, &policy(spec), &r).map_err(anyhow::Error::msg)
}

pub fn batch_part_len(spec: &Spec, target: &kir::Target, fp4: bool) -> usize {
    let linears: Vec<&str> = if fp4 {
        FP4_LINEARS.to_vec()
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

impl GemmSet {
    fn push(&mut self, k: std::result::Result<KernelInstance, kg::GemmError>) -> Result<()> {
        match k {
            Ok(k) => self.kernels.push(k),
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
            self.push(KernelInstance::gemm(t, &name, shape, wf, tuning.tile(v), split, token_major))?;
        }
        Ok(())
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.kernels.iter().map(|k| intern(&k.name)).collect()
    }

    pub fn smem(&self) -> BTreeMap<&'static str, u32> {
        self.kernels
            .iter()
            .filter(|k| k.tile().is_some() || k.batch_shape().is_some())
            .map(|k| (intern(&k.name), k.smem_bytes))
            .collect()
    }
}

pub fn gemm_kernels(spec: &Spec, t: &kir::Target) -> Result<GemmSet> {
    let tuning = tuning(&spec.runtime);
    let la = spec.linear_attention.is_some();
    let dequant = t.language == kir::Language::Wgsl;
    let mut set = GemmSet { kernels: Vec::new(), refused: Vec::new(), lenient: dequant };
    let gdn: &[&str] = if la { &GDN_LINEARS } else { &[] };
    let mma = t.has(kir::Feature::Mma16816);
    if !mma && !dequant {
        return Ok(set);
    }
    for &linear in MMA_LINEARS.iter().chain(gdn) {
        let wf = match linear {
            "lm_head" => head_format(spec),
            "lin_ba" => WeightFormat::Bf16,
            _ => weight_format(spec.quant),
        };
        set.tiles(t, &format!("{linear}_mma"), linear_shape(spec, linear), wf, split(spec, linear, wf), false, &tuning)?;
    }
    if dequant {
        if spec.mlp.quant == Some(Quant::Nvfp4G16) {
            for linear in FP4_LINEARS {
                let shape = linear_shape(spec, linear);
                set.tiles(t, &format!("{linear}_f4w"), shape, WeightFormat::Fp4, true, false, &tuning)?;
                set.push(KernelInstance::batch(t, &format!("{linear}_f4b8"), shape, WeightFormat::Fp4, 8))?;
                set.push(KernelInstance::fp4_act(t, &format!("{linear}_f4a"), shape, Tile::SMALL, &tuning))?;
            }
            let gu = linear_shape(spec, "gate_up");
            set.push(KernelInstance::fp4_gemv(t, "gate_up_act_f4", Fp4Gemv::GateUpAct, gu.n, gu.k))?;
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
    }
    if la && spec.speculative().is_some() {
        for linear in crate::mtp::MTP_LINEARS {
            set.tiles(t, &format!("{linear}_mma"), linear_shape(spec, linear), WeightFormat::Bf16, true, false, &tuning)?;
        }
    }
    if let Some(scale) = fp8_scale(spec) {
        if t.has(kir::Feature::Fp8Mma) {
            set.push(KernelInstance::quant_fp8(t, Fp8Scale::Block))?;
            set.push(KernelInstance::quant_fp8(t, Fp8Scale::Channel))?;
            for &linear in FP8_ACT_LINEARS.iter().filter(|l| la || !l.starts_with("lin_")) {
                let shape = linear_shape(spec, linear);
                for v in Variant::FP8_ACT {
                    set.push(KernelInstance::fp8_act(t, &format!("{linear}_f8{}", v.suffix()), shape, tuning.tile(v), scale))?;
                }
            }
        }
        for &linear in kg::BATCH_LINEARS.iter().filter(|l| la || !l.starts_with("lin_")) {
            let shape = linear_shape(spec, linear);
            for tokens in BATCH_VARIANTS {
                set.push(KernelInstance::batch(t, &format!("{linear}_f8b{tokens}"), shape, WeightFormat::Fp8, tokens))?;
            }
            set.push(KernelInstance::batch_reduce(t, &format!("{linear}_f8b_reduce"), shape, WeightFormat::Fp8))?;
        }
    }
    let fp4_mlp = spec.mlp.quant == Some(Quant::Nvfp4G16);
    if fp4_mlp {
        let (gu, dn) = (linear_shape(spec, "gate_up"), linear_shape(spec, "down"));
        set.push(KernelInstance::fp4_gemv(t, "gate_up_act_f4", Fp4Gemv::GateUpAct, gu.n, gu.k))?;
        set.push(KernelInstance::fp4_gemv(t, "down_proj_f4", Fp4Gemv::Down, dn.n, dn.k))?;
        for linear in FP4_LINEARS {
            let shape = linear_shape(spec, linear);
            set.tiles(t, &format!("{linear}_f4w"), shape, WeightFormat::Fp4, true, false, &tuning)?;
            for tokens in BATCH_VARIANTS {
                set.push(KernelInstance::batch(t, &format!("{linear}_f4b{tokens}"), shape, WeightFormat::Fp4, tokens))?;
            }
            set.push(KernelInstance::batch_reduce(t, &format!("{linear}_f4b_reduce"), shape, WeightFormat::Fp4))?;
        }
        if fp4_act_enabled(spec) && t.has(kir::Feature::BlockScaledFp4) {
            set.push(KernelInstance::quant_fp4(t))?;
            for linear in FP4_LINEARS {
                let shape = linear_shape(spec, linear);
                for v in Variant::FP4_ACT {
                    set.push(KernelInstance::fp4_act(t, &format!("{linear}_f4a{}", v.suffix()), shape, tuning.fp4_tile(v), &tuning))?;
                }
            }
        }
    }
    if spec.lm_head_quant() == Quant::Nvfp4G16 {
        let shape = linear_shape(spec, "lm_head");
        set.push(KernelInstance::fp4_head_norm(t, "lm_head_norm_f4", shape.n, shape.k))?;
        set.tiles(t, "lm_head_f4w", shape, WeightFormat::Fp4, true, false, &tuning)?;
    }
    if la && let Some(q) = spec.draft_head_quant() {
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
