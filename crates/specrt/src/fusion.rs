use crate::gemm::{self, GGML_GEMV_THREADS, GemmPath, ggml_gemv_rows};
use crate::plan::{Buf, Op, Phase, Plan, QuantSource, Weight, weight_kind};
use crate::signature::{Class, Count, Form, Place, Rows, Signer, Tensor, key, ranked, sections};
use crate::spec::{Quant, Spec};
use anyhow::Result;
use kir::gemm::Epilogue;
use kir::heavy::{BASE_REGS, acc_regs};
use std::collections::{BTreeMap, BTreeSet};

const CUDA_GBS: f64 = 1600.0;
const CUDA_LAUNCH_US: f64 = 2.0;
const CUDA_ROUND_TRIP_US: f64 = 0.8;
const CUDA_REREAD_RATE: f64 = 0.25;
const WGSL_GBS: f64 = 200.0;
const WGSL_LAUNCH_US: f64 = 5.0;
const WGSL_ROUND_TRIP_US: f64 = 2.0;
const FP4_ACT_GROUP: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub launch_bytes: u64,
    pub round_trip_bytes: u64,
    pub reread_rate: f64,
    pub reread_overlap: f64,
    pub smem_limit: u64,
    pub max_regs: u32,
}

fn us_bytes(us: f64, gbs: f64) -> u64 {
    (us * gbs * 1e3) as u64
}

impl Costs {
    pub fn for_target(t: &kir::Target) -> Costs {
        match t.language {
            kir::Language::Cuda => Costs {
                launch_bytes: us_bytes(CUDA_LAUNCH_US, CUDA_GBS),
                round_trip_bytes: us_bytes(CUDA_ROUND_TRIP_US, CUDA_GBS),
                reread_rate: CUDA_REREAD_RATE,
                reread_overlap: 1.0,
                smem_limit: t.smem_limit as u64,
                max_regs: t.max_regs,
            },
            kir::Language::Wgsl => Costs {
                launch_bytes: us_bytes(WGSL_LAUNCH_US, WGSL_GBS),
                round_trip_bytes: us_bytes(WGSL_ROUND_TRIP_US, WGSL_GBS),
                reread_rate: 1.0,
                reread_overlap: 0.0,
                smem_limit: t.smem_limit as u64,
                max_regs: t.max_regs,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Tile { cols: usize },
    Flat { cols: usize },
    Split { serial: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub col_blocks: usize,
    pub rows_block: usize,
    pub owner: Owner,
    pub smem: u64,
    pub regs: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Whole,
    Quant,
    Dequant,
    Main,
    Reduce,
    Residual,
    Act,
    QkNorm,
    Rope,
    KvWrite,
    Split,
    Combine,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub label: String,
    pub op: usize,
    pub part: Part,
    pub class: Class,
    pub reads: Vec<Tensor>,
    pub writes: Vec<Tensor>,
    pub launches: usize,
    pub geometry: Option<Geometry>,
    pub barrier: bool,
    pub bound: bool,
    pub row: usize,
}

impl Unit {
    fn new(label: impl Into<String>, op: usize, class: Class, reads: Vec<Tensor>, writes: Vec<Tensor>) -> Unit {
        let row = writes.first().map_or(1, |t| t.cols);
        Unit { label: label.into(), op, part: Part::Whole, class, reads, writes, launches: 1, geometry: None, barrier: false, bound: matches!(class, Class::Attention | Class::Scan), row }
    }

    fn with_part(self, part: Part) -> Unit {
        Unit { part, ..self }
    }

    fn with_row(self, row: usize) -> Unit {
        Unit { row, ..self }
    }

    fn with_geometry(self, g: Option<Geometry>) -> Unit {
        Unit { geometry: g, ..self }
    }

    fn heavy(&self) -> bool {
        matches!(self.class, Class::Gemm | Class::Attention | Class::Scan)
    }

    fn anchored(&self) -> bool {
        self.heavy() || self.geometry.is_some()
    }

    fn light(&self) -> bool {
        matches!(self.class, Class::Elementwise | Class::RowReduce)
    }

    fn opaque(&self) -> bool {
        self.launches != 1 || !(self.light() || self.heavy())
    }

    fn regs(&self) -> u32 {
        match (self.geometry, self.class) {
            (Some(g), _) => g.regs,
            (None, Class::RowReduce) => 8,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Prologue,
    Epilogue,
    Light,
    Pair,
    Fill,
    Split,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    SlotFit(Box<Reason>),
    Opaque,
    TwoHeavy,
    Barrier,
    Boundary,
    Rows,
    Schedule,
    Path,
    NotElementwise,
    Ownership { owned: usize, row: usize },
    Readers,
    Smem { need: u64, limit: u64 },
    Regs { need: u32, limit: u32 },
    Cost,
    Outscored { by: Vec<String>, saving: u64 },
    NoKernel,
}

impl Reason {
    pub fn describe(&self) -> String {
        match self {
            Reason::SlotFit(inner) => format!("slot fit split, the fused group {}", inner.describe()),
            Reason::Opaque => "a unit is several launches or has no kernel geometry".into(),
            Reason::TwoHeavy => "a group holds at most one op with its own block geometry".into(),
            Reason::Barrier => "the consumer waits on a device-wide barrier".into(),
            Reason::Boundary => "attention and scan units fuse only inside their own op".into(),
            Reason::Rows => "the consumer reads rows other blocks write".into(),
            Reason::Schedule => "the row reductions do not share a row schedule".into(),
            Reason::Path => "another group sits on a path between them".into(),
            Reason::NotElementwise => "an epilogue takes only elementwise and row ops".into(),
            Reason::Ownership { owned, row } => format!("a block owns {owned} columns of the accumulator, the consumer needs {row}"),
            Reason::Readers => "the intermediate has readers outside the group or is live after it".into(),
            Reason::Smem { need, limit } => format!("needs {need} bytes of shared memory, the target has {limit}"),
            Reason::Regs { need, limit } => format!("needs {need} registers per thread, the target has {limit}"),
            Reason::Cost => "the penalty is at least the saving".into(),
            Reason::Outscored { by, saving } => format!("{} saves more ({saving}) with the same units", by.join(" + ")),
            Reason::NoKernel => "no kernel in the library runs these ops as one".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub rule: Rule,
    pub producer: Vec<String>,
    pub consumers: Vec<Vec<String>>,
    pub saving: u64,
    pub penalty: u64,
    pub verdict: Result<(), Reason>,
}

impl Decision {
    pub fn describe(&self) -> String {
        let consumers: Vec<String> = self.consumers.iter().map(|c| c.join(" + ")).collect();
        let head = format!("{:?} {} -> {}: saves {} penalty {}", self.rule, self.producer.join(" + "), consumers.join(" | "), self.saving, self.penalty);
        match &self.verdict {
            Ok(()) => head,
            Err(r) => format!("{head}, {}", r.describe()),
        }
    }

    pub fn involves(&self, producer: &str, consumer: &str) -> bool {
        self.producer.iter().any(|p| p == producer) && self.consumers.iter().any(|c| c.iter().any(|u| u == consumer))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub units: Vec<String>,
    pub members: Vec<usize>,
    pub ops: Vec<usize>,
    pub launches: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SectionFusion {
    pub name: String,
    pub repeats: usize,
    pub units: Vec<Unit>,
    pub groups: Vec<Group>,
    pub fused: Vec<Decision>,
    pub rejected: Vec<Decision>,
    pub launches_before: usize,
    pub launches_after: usize,
}

impl SectionFusion {
    pub fn group_of(&self, label: &str) -> Vec<&Group> {
        self.groups.iter().filter(|g| g.units.iter().any(|u| u == label)).collect()
    }

    pub fn rejected_between(&self, producer: &str, consumer: &str) -> Option<&Decision> {
        self.rejected.iter().find(|d| d.involves(producer, consumer))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusionReport {
    pub phase: Phase,
    pub count: Count,
    pub costs: Costs,
    pub sections: Vec<SectionFusion>,
    pub capture: Vec<CaptureSection>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaptureSection {
    pub name: String,
    pub groups: Vec<Vec<(usize, String)>>,
}

impl FusionReport {
    pub fn section(&self, name: &str) -> Option<&SectionFusion> {
        self.sections.iter().find(|s| s.name == name)
    }

    pub fn render(&self) -> String {
        let c = &self.count;
        let mut out = format!("fusion {:?} tokens {} context {} launch cost {} bytes, dry run\n", self.phase, c.tokens, c.context, self.costs.launch_bytes);
        for s in &self.sections {
            out.push_str(&format!("\n== {} (x{}) launches {} -> {}\n", s.name, s.repeats, s.launches_before, s.launches_after));
            for g in &s.groups {
                out.push_str(&format!("  group [{}] launches {}\n", g.units.join(" + "), g.launches));
            }
            for d in &s.fused {
                out.push_str(&format!("  fused {}\n", d.describe()));
            }
            for d in &s.rejected {
                out.push_str(&format!("  kept apart {}\n", d.describe()));
            }
        }
        for c in &self.capture {
            out.push_str(&format!("\n== {} capture groups\n", c.name));
            for g in &c.groups {
                let members: Vec<String> = g.iter().map(|(i, l)| format!("{i} {l}")).collect();
                out.push_str(&format!("  one launch group [{}]\n", members.join(" | ")));
            }
        }
        out
    }
}

struct Lower<'a> {
    spec: &'a Spec,
    signer: &'a Signer<'a>,
    target: &'a kir::Target,
    count: Count,
    phase: Phase,
    layer: usize,
    units: Vec<Unit>,
    internal: BTreeSet<Place>,
}

impl Lower<'_> {
    fn local(&mut self, name: String, rows: Rows, cols: usize, form: Form) -> Tensor {
        let t = Tensor::new(Place::Local(name), rows, cols, form);
        self.internal.insert(t.place.clone());
        t
    }

    fn geometry(&self, weight: Weight, quant: QuantSource, tokens: usize, n: usize, k: usize) -> Option<(Geometry, GemmPath)> {
        if self.target.language == kir::Language::Wgsl {
            let (per, rows_block) = if tokens == 1 { (kir::wgsl::SG_ROWS as usize, 1) } else { (kir::wgsl::TILE_ROWS as usize, tokens.min(kir::wgsl::TILE_TOKENS as usize)) };
            return Some((Geometry { col_blocks: n.div_ceil(per), rows_block, owner: Owner::Tile { cols: per }, smem: 0, regs: BASE_REGS }, GemmPath::Mma));
        }
        let q = quant.resolve_for(self.spec, self.layer, weight);
        let choice = gemm::select(self.spec, self.target, weight.name(), tokens, weight_kind(q)).ok()?;
        let t = self.target;
        let gemv = tokens == 1 && self.phase == Phase::Decode && choice.path != GemmPath::Ggv;
        let g = match choice.path {
            _ if gemv => {
                let threads = if q == Quant::Nvfp4G16 { gemm::FP4_GEMV_THREADS } else { kir::gemv::GEMV_THREADS };
                let per = threads as usize / 32;
                return Some((Geometry { col_blocks: n.div_ceil(per), rows_block: 1, owner: Owner::Tile { cols: per }, smem: 0, regs: BASE_REGS }, GemmPath::Mma));
            }
            GemmPath::F8b | GemmPath::F4b | GemmPath::Ggr => {
                let kernel = choice.kernel.as_ref()?;
                let b = kernel.batch_shape()?;
                let splits = if choice.path == GemmPath::Ggr { b.ggml_splits(t, n, k) } else { b.splits(t, n, k) };
                let rows = b.rows();
                let owner = if splits > 1 { Owner::Split { serial: splits } } else { Owner::Tile { cols: rows } };
                Geometry { col_blocks: n.div_ceil(rows) * splits, rows_block: b.tokens.min(tokens), owner, smem: b.smem() as u64, regs: acc_regs(b.tokens * rows, b.threads()) }
            }
            GemmPath::Rows => Geometry { col_blocks: n, rows_block: tokens, owner: Owner::Tile { cols: 1 }, smem: 0, regs: BASE_REGS + tokens as u32 },
            GemmPath::Ggv => {
                let per = GGML_GEMV_THREADS as usize / 32 * ggml_gemv_rows(n);
                Geometry { col_blocks: n.div_ceil(per), rows_block: 1, owner: Owner::Tile { cols: per }, smem: 0, regs: BASE_REGS }
            }
            _ => {
                let tile = choice.tile;
                let kernel = choice.kernel.as_ref()?;
                Geometry { col_blocks: n.div_ceil(tile.bn), rows_block: tile.bm.min(tokens), owner: Owner::Tile { cols: tile.bn }, smem: kernel.smem_bytes as u64, regs: acc_regs(tile.bm * tile.bn, kernel.threads) }
            }
        };
        Some((g, choice.path))
    }

    fn linear(&mut self, i: usize, op: &Op, weight: Weight, quant: QuantSource, output: Buf) -> Result<()> {
        let sig = self.signer.signature(op, self.layer)?;
        let s = gemm::linear_shape(self.spec, weight.name());
        let out = sig.writes[0].clone();
        let tokens = self.count.rows(out.rows).max(1);
        let label = op.label();
        let residual = s.epi == Epilogue::Residual;
        let act = matches!(s.epi, Epilogue::Act | Epilogue::Map) && output.holds_activation();
        let Some((geo, path)) = self.geometry(weight, quant, tokens, s.n, s.k) else {
            let mut u = Unit::new(label, i, sig.class, sig.reads, sig.writes).with_part(Part::Main);
            u.launches = 0;
            self.units.push(u);
            return Ok(());
        };
        let mut reads: Vec<Tensor> = sig.reads.iter().filter(|t| !(residual && t.place == out.place)).cloned().collect();
        let quantised = match path {
            GemmPath::F4a => Some((Form::Weight(Quant::Nvfp4G16), FP4_ACT_GROUP)),
            GemmPath::F8a => Some(match gemm::policy(self.spec).fp8_weights {
                Some(kir::gemm::Fp8Scale::Block) => (Form::Weight(Quant::Fp8Block128), kir::gemm::FP8_BLOCK),
                _ => (Form::Weight(Quant::Fp8Channel), s.k),
            }),
            _ => None,
        };
        if let Some((form, row)) = quantised {
            let input = reads[0].clone();
            let t = self.local(format!("{label} act quant"), input.rows, s.k, form);
            self.units.push(Unit::new(format!("{label} quant"), i, Class::RowReduce, vec![input], vec![t.clone()]).with_row(row).with_part(Part::Quant));
            reads[0] = t;
        } else if path == GemmPath::Ggb {
            let w = reads[1].clone();
            let staged = self.local(format!("{label} dequant"), Rows::Fixed(s.n), s.k, Form::Bf16);
            self.units.push(Unit::new(format!("{label} dequant"), i, Class::Gather, vec![w], vec![staged.clone()]).with_part(Part::Dequant));
            reads[1] = staged;
        }
        let result = if residual || act { self.local(format!("{label} acc"), out.rows, s.n, Form::F32) } else { out.clone() };
        match geo.owner {
            Owner::Split { serial } => {
                let parts = self.local(format!("{label} parts"), out.rows, s.n * serial, Form::F32);
                self.units.push(Unit::new(label.clone(), i, Class::Gemm, reads, vec![parts.clone()]).with_geometry(Some(geo)).with_part(Part::Main));
                let block = gemm::BATCH_REDUCE_BLOCK;
                let reduce = Geometry { col_blocks: (tokens * s.n).div_ceil(block), rows_block: 1, owner: Owner::Flat { cols: block }, smem: 0, regs: BASE_REGS };
                self.units.push(Unit::new(format!("{label} reduce"), i, Class::Elementwise, vec![parts], vec![result.clone()]).with_geometry(Some(reduce)).with_part(Part::Reduce));
            }
            Owner::Tile { .. } | Owner::Flat { .. } => self.units.push(Unit::new(label.clone(), i, Class::Gemm, reads, vec![result.clone()]).with_geometry(Some(geo)).with_part(Part::Main)),
        }
        if residual {
            self.units.push(Unit::new(format!("{label} residual"), i, Class::Elementwise, vec![result, out.clone()], vec![out]).with_part(Part::Residual));
        } else if act {
            self.units.push(Unit::new(format!("{label} act"), i, Class::Elementwise, vec![result], vec![out]).with_part(Part::Act));
        }
        Ok(())
    }

    fn rope(&mut self, i: usize) {
        let a = &self.spec.attention;
        let qkv = self.signer.qkv();
        let mut src = qkv.clone();
        if a.qk_norm && !a.qk_norm_full {
            let normed = self.local("qk normed".into(), Rows::Tokens, qkv.cols, Form::F32);
            let mut reads = vec![qkv.clone()];
            reads.extend(self.signer.qk_weights());
            self.units.push(Unit::new("qk norm", i, Class::RowReduce, reads, vec![normed.clone()]).with_row(a.head_dim).with_part(Part::QkNorm));
            src = normed;
        }
        self.units.push(Unit::new("rope", i, Class::Elementwise, vec![src], vec![qkv.clone()]).with_part(Part::Rope));
        let caches = self.signer.caches(self.layer, Rows::Tokens);
        if !caches.is_empty() {
            self.units.push(Unit::new("kv write", i, Class::Elementwise, vec![qkv], caches).with_part(Part::KvWrite));
        }
    }

    fn decode_attention(&mut self, i: usize, op: &Op) -> Result<()> {
        self.rope(i);
        let a = &self.spec.attention;
        let splits = kir::attention::decode_splits(self.count.context.max(1));
        let group = (a.heads / a.kv_heads.max(1)).max(1);
        let parts = self.local("attention parts".into(), Rows::Tokens, a.heads * splits * (a.head_dim + 2), Form::F32);
        let mut reads = vec![self.signer.qkv()];
        reads.extend(self.signer.caches(self.layer, Rows::Context));
        let geo = Geometry { col_blocks: a.kv_heads * splits, rows_block: 1, owner: Owner::Split { serial: group }, smem: 0, regs: BASE_REGS };
        self.units.push(Unit::new("attention split", i, Class::Attention, reads, vec![parts.clone()]).with_geometry(Some(geo)).with_part(Part::Split));
        let out = self.signer.signature(op, self.layer)?.writes.last().cloned().expect("decode attention writes its output");
        let mut combine = Unit::new("attention combine", i, Class::Elementwise, vec![parts], vec![out]).with_row(a.head_dim).with_part(Part::Combine);
        combine.bound = true;
        self.units.push(combine);
        Ok(())
    }

    fn op(&mut self, i: usize, op: &Op) -> Result<()> {
        match *op {
            Op::Linear { weight, quant, output, .. } => self.linear(i, op, weight, quant, output),
            Op::KvRope | Op::QkRope | Op::DecodeRope => {
                self.rope(i);
                Ok(())
            }
            Op::DecodeAttention => self.decode_attention(i, op),
            _ => {
                let sig = self.signer.signature(op, self.layer)?;
                let mut u = Unit::new(op.label(), i, sig.class, sig.reads, sig.writes);
                u.launches = self.signer.launches(op, self.phase, self.layer, &self.count, self.target);
                u.barrier = matches!(op, Op::AllReduce | Op::AllReduceNorm { .. });
                self.units.push(u);
                Ok(())
            }
        }
    }
}

fn unique_labels(units: &mut [Unit]) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for u in units {
        let n = seen.entry(u.label.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            u.label = format!("{} {}", u.label, n);
        }
    }
}

struct Value {
    writer: usize,
    tensor: Tensor,
    readers: Vec<(usize, Tensor)>,
    live: bool,
}

fn tracked(p: &Place) -> bool {
    matches!(p, Place::Buf(_) | Place::Local(_) | Place::Cache(_) | Place::State(_))
}

fn values(units: &[Unit], internal: &BTreeSet<Place>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut current: BTreeMap<Place, usize> = BTreeMap::new();
    let mut overwritten: BTreeSet<usize> = BTreeSet::new();
    for (u, unit) in units.iter().enumerate() {
        for r in unit.reads.iter().filter(|r| tracked(&r.place)) {
            if let Some(&v) = current.get(&key(&r.place)) {
                out[v].readers.push((u, r.clone()));
            }
        }
        for w in unit.writes.iter().filter(|w| tracked(&w.place)) {
            if let Some(old) = current.insert(key(&w.place), out.len()) {
                overwritten.insert(old);
            }
            out.push(Value { writer: u, tensor: w.clone(), readers: Vec::new(), live: false });
        }
    }
    for (i, v) in out.iter_mut().enumerate() {
        let last = !overwritten.contains(&i);
        v.live = match &v.tensor.place {
            Place::Cache(_) | Place::State(_) => true,
            Place::Buf(Buf::X | Buf::Partial | Buf::Logits | Buf::MtpLogits) => last,
            Place::Local(_) => last && !internal.contains(&v.tensor.place),
            _ => false,
        };
    }
    out
}

pub type Library<'a> = &'a dyn Fn(&[&Unit]) -> Fit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Kernel,
    Part,
    None,
}

struct Engine<'a> {
    units: &'a [Unit],
    values: Vec<Value>,
    groups: Vec<BTreeSet<usize>>,
    costs: &'a Costs,
    count: Count,
    library: Option<Library<'a>>,
}

struct Candidate {
    decision: Decision,
    apply: Apply,
}

enum Apply {
    Prologue { producer: usize, consumers: Vec<usize> },
    Pair { producer: usize, consumer: usize },
}

fn constant(units: &[Unit], g: &BTreeSet<usize>) -> bool {
    g.iter().all(|&u| units[u].reads.is_empty() && units[u].class == Class::Elementwise && units[u].launches == 1)
}

impl Engine<'_> {
    fn bytes(&self, t: &Tensor) -> u64 {
        t.bytes(&self.count)
    }

    fn labels(&self, g: &BTreeSet<usize>) -> Vec<String> {
        g.iter().map(|&u| self.units[u].label.clone()).collect()
    }

    fn launches(&self, g: &BTreeSet<usize>) -> usize {
        if g.len() == 1 {
            return self.units[*g.first().unwrap()].launches;
        }
        match self.fit(g) {
            Fit::Part => g.iter().map(|&u| self.units[u].launches).sum(),
            Fit::Kernel | Fit::None => 1,
        }
    }

    fn fit(&self, g: &BTreeSet<usize>) -> Fit {
        let units: Vec<&Unit> = g.iter().map(|&u| &self.units[u]).collect();
        self.library.map_or(Fit::Kernel, |library| library(&units))
    }

    fn ops(&self, g: &BTreeSet<usize>) -> Vec<usize> {
        let ops: BTreeSet<usize> = g.iter().map(|&u| self.units[u].op).collect();
        ops.into_iter().collect()
    }

    fn realisable(&self, merged: &BTreeSet<usize>) -> Result<(), Reason> {
        match self.fit(merged) {
            Fit::None => Err(Reason::NoKernel),
            Fit::Kernel | Fit::Part => Ok(()),
        }
    }

    fn heavy(&self, g: &BTreeSet<usize>) -> Option<usize> {
        g.iter().copied().find(|&u| self.units[u].heavy())
    }

    fn anchor(&self, g: &BTreeSet<usize>) -> Option<usize> {
        g.iter().copied().filter(|&u| self.units[u].anchored()).last()
    }

    fn served(groups: &[BTreeSet<usize>], v: &Value, reader: usize) -> bool {
        groups.iter().filter(|g| g.contains(&reader)).all(|g| g.contains(&v.writer))
    }

    fn needs_memory(groups: &[BTreeSet<usize>], v: &Value) -> bool {
        v.live || v.readers.iter().any(|(r, _)| !Self::served(groups, v, *r))
    }

    fn edges(&self) -> BTreeSet<(usize, usize)> {
        let mut out = BTreeSet::new();
        for v in &self.values {
            for (r, _) in &v.readers {
                for (b, g2) in self.groups.iter().enumerate() {
                    if !g2.contains(r) || g2.contains(&v.writer) {
                        continue;
                    }
                    for (a, g1) in self.groups.iter().enumerate() {
                        if a != b && g1.contains(&v.writer) {
                            out.insert((a, b));
                        }
                    }
                }
            }
        }
        out
    }

    fn path_between(edges: &BTreeSet<(usize, usize)>, from: usize, to: usize) -> bool {
        let mut reach: BTreeSet<usize> = BTreeSet::new();
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            for &(_, b) in edges.iter().filter(|(a, _)| *a == n) {
                if b != to && b != from && reach.insert(b) {
                    stack.push(b);
                }
            }
        }
        reach.iter().any(|x| edges.contains(&(*x, to)))
    }

    fn flow<'v>(&'v self, from: &'v BTreeSet<usize>, to: &'v BTreeSet<usize>) -> impl Iterator<Item = (&'v Value, Vec<&'v Tensor>)> + 'v {
        self.values.iter().filter(move |v| from.contains(&v.writer) && !to.contains(&v.writer)).filter_map(move |v| {
            let reads: Vec<&Tensor> = v.readers.iter().filter(|(r, _)| to.contains(r)).map(|(_, t)| t).collect();
            (!reads.is_empty()).then_some((v, reads))
        })
    }

    fn rows_match(&self, from: &BTreeSet<usize>, to: &BTreeSet<usize>) -> bool {
        self.flow(from, to).all(|(v, reads)| reads.iter().all(|t| self.count.rows(t.rows) == self.count.rows(v.tensor.rows)))
    }

    fn boundary(&self, merged: &BTreeSet<usize>) -> bool {
        let ops: BTreeSet<usize> = merged.iter().map(|&u| self.units[u].op).collect();
        ops.len() > 1 && merged.iter().any(|&u| self.units[u].bound)
    }

    fn external_reads(&self, g: &BTreeSet<usize>) -> u64 {
        let inside: BTreeSet<Place> = self.values.iter().filter(|v| g.contains(&v.writer)).map(|v| key(&v.tensor.place)).collect();
        g.iter().flat_map(|&u| self.units[u].reads.iter()).filter(|t| !inside.contains(&key(&t.place))).map(|t| self.bytes(t)).sum()
    }

    fn decision(&self, rule: Rule, producer: &BTreeSet<usize>, consumers: &[&BTreeSet<usize>], saving: u64, penalty: u64, verdict: Result<(), Reason>) -> Decision {
        Decision { rule, producer: self.labels(producer), consumers: consumers.iter().map(|c| self.labels(c)).collect(), saving, penalty, verdict }
    }

    fn prologue(&self, edges: &BTreeSet<(usize, usize)>, a: usize) -> Option<Candidate> {
        let g1 = &self.groups[a];
        let consumers: Vec<usize> = edges.iter().filter(|(x, _)| *x == a).map(|(_, b)| *b).collect();
        if consumers.is_empty() || !consumers.iter().any(|&c| self.heavy(&self.groups[c]).is_some()) {
            return None;
        }
        let cs: Vec<&BTreeSet<usize>> = consumers.iter().map(|&c| &self.groups[c]).collect();
        let produced: Vec<&Value> = self.values.iter().filter(|v| g1.contains(&v.writer) && v.readers.iter().any(|(r, _)| !g1.contains(r))).collect();
        let mut saving = self.launches(g1) as u64 * self.costs.launch_bytes;
        for v in &produced {
            saving += self.bytes(&v.tensor) + v.readers.iter().filter(|(r, _)| !g1.contains(r)).map(|(_, t)| self.bytes(t)).sum::<u64>();
        }
        let row_bytes = self.external_reads(g1) as f64;
        let penalty: u64 = cs
            .iter()
            .filter_map(|c| self.heavy(c))
            .filter_map(|h| self.units[h].geometry.map(|g| (h, g)))
            .map(|(h, g)| {
                let rereads = g.col_blocks.saturating_sub(1) as f64 * row_bytes * self.costs.reread_rate;
                let hidden = self.external_reads(&BTreeSet::from([h])) as f64 * self.costs.reread_overlap;
                (rereads - hidden).max(0.0) as u64
            })
            .sum();
        let verdict = self.prologue_legal(edges, a, &consumers);
        let verdict = verdict.and_then(|()| if saving > penalty { Ok(()) } else { Err(Reason::Cost) });
        Some(Candidate { decision: self.decision(Rule::Prologue, g1, &cs, saving, penalty, verdict), apply: Apply::Prologue { producer: a, consumers } })
    }

    fn prologue_legal(&self, edges: &BTreeSet<(usize, usize)>, a: usize, consumers: &[usize]) -> Result<(), Reason> {
        let g1 = &self.groups[a];
        if g1.iter().any(|&u| self.units[u].opaque()) {
            return Err(Reason::Opaque);
        }
        if g1.iter().any(|&u| self.units[u].barrier) {
            return Err(Reason::Barrier);
        }
        let mut heavies = Vec::new();
        for &c in consumers {
            let g2 = &self.groups[c];
            let Some(h) = self.heavy(g2) else { return Err(Reason::Readers) };
            if g2.iter().any(|&u| self.units[u].opaque()) {
                return Err(Reason::Opaque);
            }
            let merged: BTreeSet<usize> = g1.union(g2).copied().collect();
            if self.boundary(&merged) {
                return Err(Reason::Boundary);
            }
            self.realisable(&merged)?;
            heavies.push((c, h));
        }
        let produced: Vec<&Value> = self.values.iter().filter(|v| g1.contains(&v.writer) && v.readers.iter().any(|(r, _)| !g1.contains(r))).collect();
        if produced.iter().any(|v| v.live) {
            return Err(Reason::Readers);
        }
        for &(c, _) in &heavies {
            let g2 = &self.groups[c];
            if !self.rows_match(g1, g2) {
                return Err(Reason::Rows);
            }
            if Self::path_between(edges, a, c) {
                return Err(Reason::Path);
            }
        }
        Ok(())
    }

    fn budget(&self, g: &BTreeSet<usize>) -> Result<(), Reason> {
        let Some(h) = self.anchor(g) else { return Ok(()) };
        let Some(geo) = self.units[h].geometry else { return Ok(()) };
        let staged: usize = self.values.iter().filter(|v| g.contains(&v.writer) && !self.units[v.writer].anchored() && v.readers.iter().any(|(r, _)| *r == h)).map(|v| v.tensor.cols).sum();
        let need = geo.smem + (geo.rows_block * staged * 4) as u64;
        if need > self.costs.smem_limit {
            return Err(Reason::Smem { need, limit: self.costs.smem_limit });
        }
        let regs = geo.regs + g.iter().filter(|&&u| u != h).map(|&u| self.units[u].regs()).sum::<u32>();
        if regs > self.costs.max_regs {
            return Err(Reason::Regs { need: regs, limit: self.costs.max_regs });
        }
        Ok(())
    }

    fn whole_ops(&self, g: &BTreeSet<usize>) -> bool {
        let ops: BTreeSet<usize> = g.iter().map(|&u| self.units[u].op).collect();
        ops.len() <= 1 || self.units.iter().enumerate().filter(|(_, u)| ops.contains(&u.op)).all(|(i, _)| g.contains(&i))
    }

    fn cut_cost(&self, left: &BTreeSet<usize>, right: &BTreeSet<usize>) -> u64 {
        self.costs.launch_bytes + self.flow(left, right).map(|(v, reads)| self.bytes(&v.tensor) + reads.iter().map(|t| self.bytes(t)).sum::<u64>()).sum::<u64>()
    }

    fn slot_fit(&self, g: &BTreeSet<usize>, splits: &mut Vec<Decision>) -> Vec<BTreeSet<usize>> {
        let Err(reason) = self.budget(g) else { return vec![g.clone()] };
        let members: Vec<usize> = g.iter().copied().collect();
        let mut best: Option<(bool, u64, BTreeSet<usize>, BTreeSet<usize>)> = None;
        for k in 1..members.len() {
            let left: BTreeSet<usize> = members[..k].iter().copied().collect();
            let right: BTreeSet<usize> = members[k..].iter().copied().collect();
            if !self.whole_ops(&left) || !self.whole_ops(&right) || self.fit(&left) == Fit::None || self.fit(&right) == Fit::None {
                continue;
            }
            let fits = self.budget(&left).is_ok() && self.budget(&right).is_ok();
            let cost = self.cut_cost(&left, &right);
            if best.as_ref().is_none_or(|(f, c, _, _)| (!fits, cost) < (!*f, *c)) {
                best = Some((fits, cost, left, right));
            }
        }
        let Some((_, cost, left, right)) = best else {
            return members.iter().map(|&u| BTreeSet::from([u])).collect();
        };
        splits.push(self.decision(Rule::Split, &left, &[&right], 0, cost, Err(Reason::SlotFit(Box::new(reason)))));
        let mut out = self.slot_fit(&left, splits);
        out.extend(self.slot_fit(&right, splits));
        out
    }

    fn split_over_budget(&mut self) -> Vec<Decision> {
        let mut splits = Vec::new();
        let groups = std::mem::take(&mut self.groups);
        for g in &groups {
            let pieces = self.slot_fit(g, &mut splits);
            self.groups.extend(pieces);
        }
        splits
    }

    fn pair(&self, edges: &BTreeSet<(usize, usize)>, a: usize, b: usize) -> Candidate {
        let (g1, g2) = (&self.groups[a], &self.groups[b]);
        let merged: BTreeSet<usize> = g1.union(g2).copied().collect();
        let rule = match (self.anchor(g1), self.heavy(g2)) {
            (Some(_), Some(_)) => Rule::Pair,
            (Some(_), None) => Rule::Epilogue,
            _ => Rule::Light,
        };
        let anchor = self.anchor(g1);
        let split = anchor.and_then(|h| self.units[h].geometry).and_then(|g| match g.owner {
            Owner::Split { serial } => Some(serial),
            Owner::Tile { .. } | Owner::Flat { .. } => None,
        });
        let after: Vec<BTreeSet<usize>> = self.groups.iter().enumerate().filter(|(i, _)| *i != a && *i != b).map(|(_, g)| g.clone()).chain([merged.clone()]).collect();
        let mut saving = (self.launches(g1) + self.launches(g2) - 1) as u64 * self.costs.launch_bytes;
        let mut held = false;
        for (v, reads) in self.flow(g1, g2) {
            let from_split = split.is_some() && Some(v.writer) == anchor;
            if Self::needs_memory(&after, v) {
                held = true;
            } else if !from_split {
                saving += self.bytes(&v.tensor);
            }
            if !from_split {
                saving += reads.iter().map(|t| self.bytes(t)).sum::<u64>();
            }
        }
        let penalty = match (rule, split) {
            (Rule::Epilogue, Some(serial)) => (serial as u64 + 1) * self.costs.round_trip_bytes,
            _ => 0,
        };
        let verdict = self.pair_legal(edges, a, b, rule, &merged, held);
        let verdict = verdict.and_then(|()| if saving > penalty { Ok(()) } else { Err(Reason::Cost) });
        Candidate { decision: self.decision(rule, g1, &[g2], saving, penalty, verdict), apply: Apply::Pair { producer: a, consumer: b } }
    }

    fn pair_legal(&self, edges: &BTreeSet<(usize, usize)>, a: usize, b: usize, rule: Rule, merged: &BTreeSet<usize>, held: bool) -> Result<(), Reason> {
        let (g1, g2) = (&self.groups[a], &self.groups[b]);
        if merged.iter().any(|&u| self.units[u].opaque()) {
            return Err(Reason::Opaque);
        }
        self.realisable(merged)?;
        if rule == Rule::Pair {
            return Err(Reason::TwoHeavy);
        }
        if g2.iter().any(|&u| self.units[u].barrier) {
            return Err(Reason::Barrier);
        }
        if self.boundary(merged) {
            return Err(Reason::Boundary);
        }
        if !self.rows_match(g1, g2) {
            return Err(Reason::Rows);
        }
        if Self::path_between(edges, a, b) {
            return Err(Reason::Path);
        }
        if rule == Rule::Light {
            let rows: Vec<usize> = merged.iter().filter(|&&u| self.units[u].class == Class::RowReduce).map(|&u| self.units[u].row.max(1)).collect();
            let widest = rows.iter().copied().max().unwrap_or(1);
            if rows.iter().any(|r| !widest.is_multiple_of(*r)) {
                return Err(Reason::Schedule);
            }
            return Ok(());
        }
        let h = self.anchor(g1).expect("an epilogue has an anchored producer");
        let Some(geo) = self.units[h].geometry else { return Err(Reason::Opaque) };
        if g2.iter().any(|&u| !self.units[u].light()) {
            return Err(Reason::NotElementwise);
        }
        if let Owner::Split { .. } = geo.owner
            && g2.iter().any(|&u| self.units[u].class == Class::RowReduce)
        {
            return Err(Reason::Ownership { owned: 0, row: g2.iter().map(|&u| self.units[u].row).max().unwrap_or(0) });
        }
        if let Owner::Tile { cols } | Owner::Flat { cols } = geo.owner {
            for &u in g2 {
                let unit = &self.units[u];
                let narrows = unit.reads.first().zip(unit.writes.first()).is_some_and(|(r, w)| r.cols != w.cols);
                if unit.class == Class::RowReduce && cols < unit.row {
                    return Err(Reason::Ownership { owned: cols, row: unit.row });
                }
                if narrows && !cols.is_multiple_of(2) {
                    return Err(Reason::Ownership { owned: cols, row: 2 });
                }
            }
        }
        if held {
            return Err(Reason::Readers);
        }
        Ok(())
    }

    fn fill(&self, edges: &BTreeSet<(usize, usize)>, a: usize, b: usize) -> Candidate {
        let (g1, g2) = (&self.groups[a], &self.groups[b]);
        let merged: BTreeSet<usize> = g1.union(g2).copied().collect();
        let after: Vec<BTreeSet<usize>> = self.groups.iter().enumerate().filter(|(i, _)| *i != a && *i != b).map(|(_, g)| g.clone()).chain([merged.clone()]).collect();
        let mut saving = self.launches(g1) as u64 * self.costs.launch_bytes;
        let mut held = false;
        for (v, reads) in self.flow(g1, g2) {
            held |= Self::needs_memory(&after, v);
            saving += self.bytes(&v.tensor) + reads.iter().map(|t| self.bytes(t)).sum::<u64>();
        }
        let verdict = self.fill_legal(edges, a, b, &merged, held);
        Candidate { decision: self.decision(Rule::Fill, g1, &[g2], saving, 0, verdict), apply: Apply::Pair { producer: b, consumer: a } }
    }

    fn fill_legal(&self, edges: &BTreeSet<(usize, usize)>, a: usize, b: usize, merged: &BTreeSet<usize>, held: bool) -> Result<(), Reason> {
        let (g1, g2) = (&self.groups[a], &self.groups[b]);
        if merged.iter().any(|&u| self.units[u].opaque()) {
            return Err(Reason::Opaque);
        }
        self.realisable(merged)?;
        if g2.iter().any(|&u| self.units[u].barrier) {
            return Err(Reason::Barrier);
        }
        let anchor = self.anchor(g2);
        let readers: Vec<usize> = self.values.iter().filter(|v| g1.contains(&v.writer)).flat_map(|v| v.readers.iter().map(|(r, _)| *r)).filter(|r| g2.contains(r)).collect();
        if readers.iter().any(|&r| Some(r) == anchor || !self.units[r].light()) {
            return Err(Reason::NotElementwise);
        }
        if !self.rows_match(g1, g2) {
            return Err(Reason::Rows);
        }
        if Self::path_between(edges, a, b) {
            return Err(Reason::Path);
        }
        if held {
            return Err(Reason::Readers);
        }
        Ok(())
    }

    fn candidates(&self) -> Vec<Candidate> {
        let edges = self.edges();
        let mut out = Vec::new();
        let mut prologued = BTreeSet::new();
        for &(a, b) in &edges {
            if constant(self.units, &self.groups[a]) {
                if self.anchor(&self.groups[b]).is_some() {
                    out.push(self.fill(&edges, a, b));
                }
                continue;
            }
            let light = self.anchor(&self.groups[a]).is_none();
            if light && self.heavy(&self.groups[b]).is_some() {
                if prologued.insert(a)
                    && let Some(c) = self.prologue(&edges, a)
                {
                    out.push(c);
                }
            } else {
                out.push(self.pair(&edges, a, b));
            }
        }
        out
    }

    fn apply(&mut self, apply: &Apply) {
        match apply {
            Apply::Prologue { producer, consumers } => {
                let units = self.groups[*producer].clone();
                for &c in consumers {
                    self.groups[c].extend(units.iter().copied());
                }
                self.groups.remove(*producer);
            }
            Apply::Pair { producer, consumer } => {
                let units = self.groups[*consumer].clone();
                self.groups[*producer].extend(units);
                self.groups.remove(*consumer);
            }
        }
    }

    fn run(&mut self) -> (Vec<Decision>, Vec<Decision>) {
        let mut fused = Vec::new();
        let mut seen = Vec::new();
        loop {
            let mut changed = false;
            for phase in PHASES {
                loop {
                    let candidates = self.candidates();
                    let Some(best) = Self::best(&candidates, phase) else {
                        seen.extend(candidates.into_iter().map(|c| c.decision));
                        break;
                    };
                    let mut candidates = candidates;
                    let chosen = candidates.swap_remove(best);
                    let taken = self.apply_set(&chosen.apply);
                    let net = chosen.decision.saving - chosen.decision.penalty;
                    let by: Vec<String> = chosen.decision.producer.iter().chain(chosen.decision.consumers.iter().flatten()).cloned().collect();
                    for c in candidates {
                        let mut d = c.decision;
                        if d.verdict.is_ok() && self.touches(&c.apply, &taken) {
                            d.verdict = Err(Reason::Outscored { by: by.clone(), saving: net });
                        }
                        seen.push(d);
                    }
                    self.apply(&chosen.apply);
                    fused.push(chosen.decision);
                    changed = true;
                }
            }
            if !changed {
                let splits = self.split_over_budget();
                let kept = |d: &Decision| self.groups.iter().any(|g| d.producer.iter().chain(d.consumers.iter().flatten()).all(|u| g.iter().any(|&m| self.units[m].label == *u)));
                let fused = fused.into_iter().filter(|d| kept(d)).collect();
                seen.extend(splits);
                return (fused, seen);
            }
        }
    }

    fn apply_set(&self, apply: &Apply) -> BTreeSet<usize> {
        match apply {
            Apply::Prologue { producer, consumers } => std::iter::once(producer).chain(consumers).flat_map(|&g| self.groups[g].iter().copied()).collect(),
            Apply::Pair { producer, consumer } => self.groups[*producer].union(&self.groups[*consumer]).copied().collect(),
        }
    }

    fn touches(&self, apply: &Apply, taken: &BTreeSet<usize>) -> bool {
        !self.apply_set(apply).is_disjoint(taken)
    }

    fn best(candidates: &[Candidate], phase: &[Rule]) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| phase.contains(&c.decision.rule) && c.decision.verdict.is_ok())
            .max_by_key(|(i, c)| (c.decision.saving - c.decision.penalty, std::cmp::Reverse(*i)))
            .map(|(i, _)| i)
    }
}

const PHASES: [&[Rule]; 2] = [&[Rule::Epilogue, Rule::Light, Rule::Fill], &[Rule::Prologue]];

fn fuse_section(units: Vec<Unit>, internal: &BTreeSet<Place>, costs: &Costs, count: Count, name: String, repeats: usize, library: Option<Library>) -> SectionFusion {
    let vals = values(&units, internal);
    let mut engine = Engine { units: &units, values: vals, groups: (0..units.len()).map(|u| BTreeSet::from([u])).collect(), costs, count, library };
    let launches_before = units.iter().map(|u| u.launches).sum();
    let (fused, seen) = engine.run();
    let mut ordered = engine.groups.clone();
    ordered.sort_by_key(|g| (g.iter().copied().find(|&u| units[u].heavy()).unwrap_or(*g.first().unwrap()), g.len()));
    let groups: Vec<Group> = ordered.iter().map(|g| Group { units: engine.labels(g), members: g.iter().copied().collect(), ops: engine.ops(g), launches: engine.launches(g) }).collect();
    let together = |d: &Decision| groups.iter().any(|g| d.producer.iter().chain(d.consumers.iter().flatten()).all(|u| g.units.contains(u)));
    let mut rejected: Vec<Decision> = Vec::new();
    for d in seen.into_iter().filter(|d| d.verdict.is_err() && !together(d)) {
        if !rejected.iter().any(|r| r.rule == d.rule && r.producer == d.producer && r.consumers == d.consumers) {
            rejected.push(d);
        }
    }
    let launches_after = groups.iter().map(|g| g.launches).sum();
    drop(engine);
    SectionFusion { name, repeats, units, groups, fused, rejected, launches_before, launches_after }
}

pub struct Section<'a> {
    pub name: String,
    pub repeats: usize,
    pub phase: Phase,
    pub layer: usize,
    pub ops: &'a [Op],
}

#[allow(clippy::too_many_arguments)]
pub fn fuse_ops(spec: &Spec, signer: &Signer, target: &kir::Target, count: Count, costs: &Costs, section: Section, library: Option<Library>) -> Result<SectionFusion> {
    let mut lower = Lower { spec, signer, target, count, phase: section.phase, layer: section.layer, units: Vec::new(), internal: BTreeSet::new() };
    for (i, op) in section.ops.iter().enumerate() {
        lower.op(i, op)?;
    }
    let Lower { mut units, internal, .. } = lower;
    unique_labels(&mut units);
    Ok(fuse_section(units, &internal, costs, count, section.name, section.repeats, library))
}

pub fn fusion_report(spec: &Spec, plan: &Plan, count: Count, target: &kir::Target, costs: &Costs) -> Result<FusionReport> {
    let signer = Signer::new(spec)?.ranked(ranked(plan));
    let mut out = Vec::new();
    for s in sections(spec, plan).into_iter().filter(|s| !s.ops.is_empty()) {
        let section = Section { name: s.name, repeats: s.repeats, phase: plan.phase, layer: s.layer, ops: s.ops };
        out.push(fuse_ops(spec, &signer, target, count, costs, section, None)?);
    }
    Ok(FusionReport { phase: plan.phase, count, costs: *costs, sections: out, capture: Vec::new() })
}
