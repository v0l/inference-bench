use crate::spec::{Spec, VHeadOrder};
use anyhow::{Result, bail, ensure};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shard {
    Column,
    Row,
    Vocab,
    Replicate,
}

impl Shard {
    pub fn parse(s: &str) -> Result<Shard> {
        Ok(match s {
            "column" => Shard::Column,
            "row" => Shard::Row,
            "vocab" => Shard::Vocab,
            "replicate" => Shard::Replicate,
            other => bail!("unknown shard {other}"),
        })
    }

    pub fn needs_allreduce(self) -> bool {
        self == Shard::Row
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub rows: Vec<usize>,
    pub cols: usize,
}

pub fn layout(spec: &Spec, weight: &str) -> Result<Layout> {
    let d = &spec.dims;
    let a = &spec.attention;
    let la = spec.linear_attention.as_ref();
    let need_la = || la.ok_or_else(|| anyhow::anyhow!("{weight} needs [linear_attention]"));
    Ok(match weight {
        "qkv" => {
            let mut rows = vec![spec.q_dim(), spec.kv_dim(), spec.kv_dim()];
            if a.output_gate {
                rows.push(spec.q_dim());
            }
            Layout { rows, cols: d.hidden }
        }
        "o" => Layout { rows: vec![d.hidden], cols: spec.q_dim() },
        "gate_up" => Layout { rows: vec![spec.gate_up_rows()], cols: d.hidden },
        "down" => Layout { rows: vec![d.hidden], cols: d.intermediate },
        "lin_in" => {
            let l = need_la()?;
            let key = l.key_heads * l.key_head_dim;
            Layout { rows: vec![key, key, l.value_dim(), l.value_dim()], cols: d.hidden }
        }
        "lin_ba" => {
            let l = need_la()?;
            Layout { rows: vec![l.value_heads, l.value_heads], cols: d.hidden }
        }
        "lin_out" => Layout { rows: vec![d.hidden], cols: need_la()?.value_dim() },
        "lm_head" => Layout { rows: vec![d.vocab], cols: d.hidden },
        other => bail!("no shard layout for {other}"),
    })
}

fn split_even(len: usize, tp: usize, rank: usize) -> Result<Range<usize>> {
    ensure!(len.is_multiple_of(tp), "{len} does not divide by tp {tp}");
    let each = len / tp;
    Ok(rank * each..(rank + 1) * each)
}

pub const VOCAB_ROW_ALIGN: usize = 16;

pub fn vocab_each(vocab: usize, tp: usize) -> Result<usize> {
    ensure!(tp >= 1, "tp must be at least 1");
    let each = vocab.div_ceil(tp).next_multiple_of(VOCAB_ROW_ALIGN);
    ensure!(tp == 1 || each * (tp - 1) < vocab, "a vocabulary of {vocab} leaves the last of {tp} ranks no rows in slices of {each}");
    Ok(each)
}

pub fn vocab_span(vocab: usize, tp: usize, rank: usize) -> Result<Range<usize>> {
    ensure!(rank < tp, "rank {rank} outside tp {tp}");
    let each = vocab_each(vocab, tp)?;
    let lo = (rank * each).min(vocab);
    Ok(lo..(lo + each).min(vocab))
}

pub fn head_shard() -> Result<Shard> {
    let shards = crate::graph::Graph::decoder().shards()?;
    let head = shards.get("lm_head").copied().unwrap_or(Shard::Replicate);
    let draft = shards.get("mtp_head").copied().unwrap_or(head);
    ensure!(head == draft, "lm_head shards as {head:?} and mtp_head as {draft:?}; the draft head reads the same rows, so they must match");
    Ok(head)
}

pub fn head_split(spec: &Spec, tp: usize) -> Result<Option<usize>> {
    if tp <= 1 || head_shard()? != Shard::Vocab {
        return Ok(None);
    }
    let Ok(each) = vocab_each(spec.dims.vocab, tp) else { return Ok(None) };
    let last = spec.dims.vocab - each * (tp - 1);
    let fp4 = spec.lm_head_quant() == crate::spec::Quant::Nvfp4G16 || spec.draft_head_quant() == Some(crate::spec::Quant::Nvfp4G16);
    let per_row = if fp4 { FP4_GROUP } else { 2 };
    Ok(last.is_multiple_of(per_row).then_some(each))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub rows: Vec<Range<usize>>,
    pub cols: Range<usize>,
}

fn one(r: Range<usize>) -> Vec<Range<usize>> {
    std::iter::once(r).collect()
}

impl Slice {
    pub fn row_count(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }
}

pub fn slice_of(spec: &Spec, weight: &str, shard: Shard, tp: usize, rank: usize) -> Result<Slice> {
    ensure!(tp >= 1 && rank < tp, "rank {rank} outside tp {tp}");
    let l = layout(spec, weight)?;
    let total: usize = l.rows.iter().sum();
    if tp == 1 || shard == Shard::Replicate {
        return Ok(Slice { rows: one(0..total), cols: 0..l.cols });
    }
    if let Some(la) = &spec.linear_attention {
        ensure!(la.v_heads == VHeadOrder::Grouped, "tp needs grouped delta-net value heads, {} has tiled ones", spec.name);
    }
    match shard {
        Shard::Column => {
            let mut rows = Vec::new();
            let mut base = 0;
            for seg in &l.rows {
                let r = split_even(*seg, tp, rank).map_err(|e| anyhow::anyhow!("{weight}: {e}"))?;
                rows.push(base + r.start..base + r.end);
                base += seg;
            }
            Ok(Slice { rows, cols: 0..l.cols })
        }
        Shard::Row => {
            let cols = split_even(l.cols, tp, rank).map_err(|e| anyhow::anyhow!("{weight}: {e}"))?;
            Ok(Slice { rows: one(0..total), cols })
        }
        Shard::Vocab => Ok(Slice { rows: vec![vocab_span(total, tp, rank)?], cols: 0..l.cols }),
        Shard::Replicate => unreachable!(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heads {
    pub attention: usize,
    pub kv: usize,
    pub key: usize,
    pub value: usize,
}

pub fn heads_per_rank(spec: &Spec, tp: usize) -> Result<Heads> {
    ensure!(tp >= 1, "tp must be at least 1");
    let div = |what: &str, n: usize| -> Result<usize> {
        ensure!(n.is_multiple_of(tp), "tp {tp} does not divide the {n} {what} of {}", spec.name);
        Ok(n / tp)
    };
    let (key, value) = match &spec.linear_attention {
        Some(l) => (div("delta-net key heads", l.key_heads)?, div("delta-net value heads", l.value_heads)?),
        None => (0, 0),
    };
    Ok(Heads { attention: div("attention heads", spec.attention.heads)?, kv: div("kv heads", spec.attention.kv_heads)?, key, value })
}

pub fn supports(spec: &Spec, tp: usize) -> Result<()> {
    heads_per_rank(spec, tp)?;
    ensure!(spec.dims.intermediate.is_multiple_of(tp), "tp {tp} does not divide the intermediate size {} of {}", spec.dims.intermediate, spec.name);
    if tp > 1 {
        crate::allreduce::bail_unsupported(tp)?;
        if let Some(m) = &spec.moe {
            moe_split(spec, m, tp)?;
        }
        ensure!(spec.runtime.backend != crate::spec::BackendChoice::Wgpu, "tp runs on CUDA only");
        ensure!(spec.mixer.is_none(), "tp does not split described mixer layers yet ({})", spec.name);
        ensure!(!spec.layers.split && spec.mlp.mapped.is_none(), "tp does not split mlp-only layers or non-gated mlps yet ({})", spec.name);
        ensure!(spec.quant != crate::spec::Quant::Gguf, "tp does not slice GGUF weights yet ({})", spec.name);
        ensure!(!spec.uses_branch(), "tp does not cover sandwich or scaled residual norms yet ({})", spec.name);
        ensure!(!spec.attention.qk_norm_full, "tp does not split full-width qk norms yet ({})", spec.name);
        if let Some(l) = &spec.linear_attention {
            ensure!(!l.fused_interleaved, "tp does not split fused interleaved delta-net projections yet ({})", spec.name);
            ensure!(l.v_heads == VHeadOrder::Grouped, "tp needs grouped delta-net value heads, {} has tiled ones", spec.name);
        }
        let (inter, cols) = (spec.dims.intermediate / tp, spec.q_dim() / tp);
        for (what, n) in [("intermediate", inter), ("attention", cols)] {
            ensure!(n % 16 == 0, "tp {tp} leaves {n} {what} columns per rank, which is not a multiple of 16");
        }
        for q in [spec.quant, spec.mlp.quant.unwrap_or(spec.quant)] {
            if q.is_fp8() && q == crate::spec::Quant::Fp8Block128 {
                ensure!(inter % FP8_BLOCK == 0 && cols % FP8_BLOCK == 0, "tp {tp} cuts fp8 128 blocks in {}", spec.name);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeSplit {
    Experts,
    Columns,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoePlan {
    pub split: MoeSplit,
    pub reason: String,
}

const MIN_COLUMNS_PER_RANK: usize = 256;

fn columns_fit(spec: &Spec, m: &crate::spec::Moe, tp: usize) -> Result<()> {
    for (what, n) in [("expert intermediate size", m.expert_intermediate), ("shared expert intermediate size", m.shared_intermediate)] {
        ensure!(n % tp == 0, "tp {tp} does not divide the {what} {n} of {}", spec.name);
        ensure!((n / tp).is_multiple_of(16), "tp {tp} leaves {} {what} columns per rank in {}, which is not a multiple of 16", n / tp, spec.name);
    }
    ensure!(m.shared_slots() <= 1, "the {} shared experts of {} are wider than a routed expert, so a column split is not supported", m.shared_slots(), spec.name);
    let per_rank = m.expert_intermediate / tp;
    match spec.mlp.quant.unwrap_or(spec.quant) {
        crate::spec::Quant::Nvfp4G16 => ensure!(per_rank.is_multiple_of(256), "tp {tp} leaves {per_rank} nvfp4 expert columns per rank in {}, which must be a multiple of 256", spec.name),
        crate::spec::Quant::Fp8Block128 => ensure!(per_rank.is_multiple_of(FP8_BLOCK), "tp {tp} cuts fp8 128 blocks in the experts of {}", spec.name),
        _ => {}
    }
    Ok(())
}

fn experts_fit(spec: &Spec, m: &crate::spec::Moe, tp: usize) -> Result<()> {
    ensure!(m.experts.is_multiple_of(tp), "tp {tp} does not divide the {} experts of {}", m.experts, spec.name);
    Ok(())
}

pub fn plan_moe(spec: &Spec, m: &crate::spec::Moe, tp: usize) -> Result<MoePlan> {
    use crate::spec::TpSplit;
    let (columns, experts) = (columns_fit(spec, m, tp), experts_fit(spec, m, tp));
    let plan = |split, reason: String| Ok(MoePlan { split, reason });
    match m.tp_split {
        TpSplit::Experts => {
            experts?;
            plan(MoeSplit::Experts, "forced by [moe] tp_split".into())
        }
        TpSplit::Columns => {
            columns?;
            plan(MoeSplit::Columns, "forced by [moe] tp_split".into())
        }
        TpSplit::Auto => match (columns, experts) {
            (Err(c), Err(e)) => anyhow::bail!("{c}; {e}"),
            (Ok(()), Err(e)) => plan(MoeSplit::Columns, format!("the experts cannot be split evenly ({e})")),
            (Err(c), Ok(())) => plan(MoeSplit::Experts, format!("columns cannot be split ({c}); each rank owns {} of {} experts", m.experts / tp, m.experts)),
            (Ok(()), Ok(())) => {
                let width = m.expert_intermediate / tp;
                if width < MIN_COLUMNS_PER_RANK {
                    plan(MoeSplit::Experts, format!("a column split leaves {width} columns per expert per rank, under {MIN_COLUMNS_PER_RANK}; each rank owns {} of {} experts, {} active per token", m.experts / tp, m.experts, m.top_k()))
                } else {
                    plan(MoeSplit::Columns, format!("{width} columns per expert per rank keep the kernels wide and every rank works on every token's {} active experts; the shared expert is split too", m.top_k()))
                }
            }
        },
    }
}

fn moe_split(spec: &Spec, m: &crate::spec::Moe, tp: usize) -> Result<()> {
    plan_moe(spec, m, tp).map(|_| ())
}

pub fn rank_spec(spec: &Spec, tp: usize) -> Result<Spec> {
    rank_spec_of(spec, tp, 0)
}

pub fn rank_spec_of(spec: &Spec, tp: usize, rank: usize) -> Result<Spec> {
    ensure!(rank < tp.max(1), "rank {rank} outside tp {tp}");
    supports(spec, tp)?;
    if tp == 1 {
        return Ok(spec.clone());
    }
    let h = heads_per_rank(spec, tp)?;
    let mut s = spec.clone();
    s.attention.heads = h.attention;
    s.attention.kv_heads = h.kv;
    s.dims.intermediate = spec.dims.intermediate / tp;
    if head_split(spec, tp)?.is_some() {
        s.dims.head_rows = Some(vocab_span(spec.dims.vocab, tp, rank)?);
    }
    if let Some(m) = s.moe.as_mut() {
        match plan_moe(spec, m, tp)?.split {
            MoeSplit::Columns => {
                m.expert_intermediate /= tp;
                m.shared_intermediate /= tp;
            }
            MoeSplit::Experts => m.local = Some(crate::spec::ExpertRange { first: rank * (m.experts / tp), count: m.experts / tp }),
        }
    }
    if let Some(l) = s.linear_attention.as_mut() {
        l.key_heads = h.key;
        l.value_heads = h.value;
    }
    Ok(s)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    Rows(Vec<Range<usize>>),
    Cols(Range<usize>),
}

#[derive(Clone, Debug, Default)]
pub struct RankCut {
    pub rank: usize,
    pub tp: usize,
    pub rules: Vec<(String, Cut)>,
}

fn expert_pattern(rest: &str) -> String {
    if let Some(tail) = rest.strip_prefix("mlp.experts.")
        && let Some((index, after)) = tail.split_once('.')
        && !index.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit())
    {
        return format!("mlp.experts.*.{after}");
    }
    rest.to_string()
}

fn heads_cut(heads: usize, head_rows: usize, tp: usize, rank: usize) -> Cut {
    let each = heads / tp * head_rows;
    Cut::Rows(one(rank * each..(rank + 1) * each))
}

impl RankCut {
    pub fn new(spec: &Spec, graph: &crate::graph::Graph, tp: usize, rank: usize) -> Result<RankCut> {
        supports(spec, tp)?;
        ensure!(rank < tp, "rank {rank} outside tp {tp}");
        let shards = graph.shards()?;
        let mut rules: Vec<(String, Cut)> = Vec::new();
        if tp == 1 {
            return Ok(RankCut { rank, tp, rules });
        }
        let a = &spec.attention;
        let hd = a.head_dim;
        let d = &spec.dims;
        let rowcut = |n: usize| Cut::Rows(one(rank * n / tp..(rank + 1) * n / tp));
        let colcut = |n: usize| Cut::Cols(rank * n / tp..(rank + 1) * n / tp);
        let shard = |w: &str| shards.get(w).copied().unwrap_or(Shard::Replicate);
        let mut add = |key: &str, c: Cut| rules.push((key.to_string(), c));
        if shard("qkv") == Shard::Column && a.fused_qkv {
            let (q, kv) = (spec.q_dim(), spec.kv_dim());
            let (qe, ke) = (q / tp, kv / tp);
            add("self_attn.qkv_proj", Cut::Rows(vec![rank * qe..(rank + 1) * qe, q + rank * ke..q + (rank + 1) * ke, q + kv + rank * ke..q + kv + (rank + 1) * ke]));
        } else if shard("qkv") == Shard::Column {
            let q_rows = if a.output_gate { 2 * hd } else { hd };
            add("self_attn.q_proj", heads_cut(a.heads, q_rows, tp, rank));
            add("self_attn.k_proj", heads_cut(a.kv_heads, hd, tp, rank));
            add("self_attn.v_proj", heads_cut(a.kv_heads, hd, tp, rank));
        }
        if shard("o") == Shard::Row {
            add("self_attn.o_proj", colcut(spec.q_dim()));
        }
        if shard("gate_up") == Shard::Column && spec.mlp.fused_gate_up {
            let (i, e) = (d.intermediate, d.intermediate / tp);
            add("mlp.gate_up_proj", Cut::Rows(vec![rank * e..(rank + 1) * e, i + rank * e..i + (rank + 1) * e]));
        } else if shard("gate_up") == Shard::Column {
            add("mlp.gate_proj", rowcut(d.intermediate));
            add("mlp.up_proj", rowcut(d.intermediate));
        }
        if shard("down") == Shard::Row {
            add("mlp.down_proj", colcut(d.intermediate));
        }
        if let Some(m) = &spec.moe
            && plan_moe(spec, m, tp)?.split == MoeSplit::Columns
        {
            let inter = m.expert_intermediate;
            for expert in ["mlp.experts.*".to_string(), format!("mlp.{}", m.shared_module)] {
                add(&format!("{expert}.gate_proj"), rowcut(inter));
                add(&format!("{expert}.up_proj"), rowcut(inter));
                add(&format!("{expert}.down_proj"), colcut(inter));
            }
        }
        if let Some(l) = &spec.linear_attention {
            let key = l.key_heads * l.key_head_dim;
            let value = l.value_dim();
            if shard("lin_in") == Shard::Column {
                let (k, v) = (key / tp, value / tp);
                let qkv = vec![rank * k..(rank + 1) * k, key + rank * k..key + (rank + 1) * k, 2 * key + rank * v..2 * key + (rank + 1) * v];
                add("linear_attn.in_proj_qkv", Cut::Rows(qkv.clone()));
                add("linear_attn.conv1d", Cut::Rows(qkv));
                add("linear_attn.in_proj_z", rowcut(value));
            }
            if shard("lin_ba") == Shard::Column {
                for m in ["linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.A_log", "linear_attn.dt_bias"] {
                    add(m, rowcut(l.value_heads));
                }
            }
            if shard("lin_out") == Shard::Row {
                add("linear_attn.out_proj", colcut(value));
            }
        }
        Ok(RankCut { rank, tp, rules })
    }

    fn find(&self, key: &str) -> Option<&Cut> {
        self.rules.iter().find(|(k, _)| k == key).map(|(_, c)| c)
    }

    pub fn rule(&self, name: &str, shape: &[usize]) -> Result<Option<Cut>> {
        if self.tp <= 1 {
            return Ok(None);
        }
        let Some(rest) = name.split_once(".layers.").and_then(|(_, r)| r.split_once('.')).map(|(_, r)| r) else { return Ok(None) };
        let rest = &expert_pattern(rest);
        if let Some(c) = self.find(rest) {
            return Ok(Some(c.clone()));
        }
        let Some((stem, field)) = rest.rsplit_once('.') else { return Ok(None) };
        let Some(cut) = self.find(stem) else { return Ok(None) };
        if shape.iter().product::<usize>() <= 1 {
            return Ok(None);
        }
        let aligned = |v: usize, unit: usize| -> Result<usize> {
            ensure!(v.is_multiple_of(unit), "{name}: tp {} cuts at {v}, inside a {unit} wide scale block", self.tp);
            Ok(v / unit)
        };
        Ok(match cut {
            Cut::Rows(r) if field == "weight_scale_inv" => {
                Some(Cut::Rows(r.iter().map(|r| Ok(aligned(r.start, FP8_BLOCK)?..aligned(r.end, FP8_BLOCK)?)).collect::<Result<_>>()?))
            }
            Cut::Rows(r) => Some(Cut::Rows(r.clone())),
            Cut::Cols(c) => {
                let full = c.len() * self.tp;
                let Some(&width) = shape.get(1) else { return Ok(None) };
                match field {
                    "weight_scale" if width == 1 => None,
                    "weight_scale_inv" | "weight" | "weight_packed" | "weight_scale" => {
                        let per = full / width;
                        ensure!(per * width == full, "{name}: {width} columns do not tile the {full} wide input");
                        Some(Cut::Cols(aligned(c.start, per)?..aligned(c.end, per)?))
                    }
                    _ => None,
                }
            }
        })
    }
}

pub fn cut_bytes<'a>(data: &'a [u8], shape: &[usize], elem: usize, rule: &Cut) -> Result<(Vec<usize>, std::borrow::Cow<'a, [u8]>)> {
    ensure!(!shape.is_empty(), "cannot slice a scalar");
    let rows = shape[0];
    let row_bytes = data.len() / rows.max(1);
    match rule {
        Cut::Rows(ranges) => {
            ensure!(ranges.iter().all(|r| r.end <= rows), "row cut {ranges:?} past {rows} rows");
            let mut shape = shape.to_vec();
            shape[0] = ranges.iter().map(|r| r.len()).sum();
            if let [r] = ranges.as_slice() {
                return Ok((shape, std::borrow::Cow::Borrowed(&data[r.start * row_bytes..r.end * row_bytes])));
            }
            Ok((shape, std::borrow::Cow::Owned(take_rows(data, row_bytes, ranges))))
        }
        Cut::Cols(c) => {
            ensure!(shape.len() == 2, "column cut of a {}-d tensor", shape.len());
            ensure!(c.end <= shape[1], "column cut {c:?} past {} columns", shape[1]);
            let mut out = Vec::with_capacity(rows * c.len() * elem);
            for row in data.chunks_exact(row_bytes) {
                out.extend_from_slice(&row[c.start * elem..c.end * elem]);
            }
            Ok((vec![rows, c.len()], std::borrow::Cow::Owned(out)))
        }
    }
}

pub fn take_rows<T: Copy>(data: &[T], row_width: usize, rows: &[Range<usize>]) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.iter().map(|r| r.len()).sum::<usize>() * row_width);
    for r in rows {
        out.extend_from_slice(&data[r.start * row_width..r.end * row_width]);
    }
    out
}

pub fn take_cols<T: Copy>(data: &[T], row_width: usize, cols: Range<usize>) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len() / row_width * cols.len());
    for row in data.chunks_exact(row_width) {
        out.extend_from_slice(&row[cols.clone()]);
    }
    out
}

pub const FP4_GROUP: usize = 16;
pub const FP8_BLOCK: usize = 128;
pub const Q8_GROUP: usize = 32;

pub fn fp4_rank(packed: &[u8], scales: &[u8], cols: usize, slice: &Slice) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(slice.cols.start.is_multiple_of(FP4_GROUP) && slice.cols.end.is_multiple_of(FP4_GROUP), "an nvfp4 column slice must sit on {FP4_GROUP} element groups");
    let p = take_rows(packed, cols / 2, &slice.rows);
    let s = take_rows(scales, cols / FP4_GROUP, &slice.rows);
    let (pw, sw) = (cols / 2, cols / FP4_GROUP);
    let p = if slice.cols == (0..cols) { p } else { take_cols(&p, pw, slice.cols.start / 2..slice.cols.end / 2) };
    let s = if slice.cols == (0..cols) { s } else { take_cols(&s, sw, slice.cols.start / FP4_GROUP..slice.cols.end / FP4_GROUP) };
    Ok((p, s))
}

pub fn q8_rank(values: &[i8], scales: &[f32], cols: usize, slice: &Slice) -> Result<(Vec<i8>, Vec<f32>)> {
    ensure!(slice.cols.start.is_multiple_of(Q8_GROUP) && slice.cols.end.is_multiple_of(Q8_GROUP), "a q8 column slice must sit on {Q8_GROUP} element groups");
    let v = take_rows(values, cols, &slice.rows);
    let s = take_rows(scales, cols / Q8_GROUP, &slice.rows);
    let v = if slice.cols == (0..cols) { v } else { take_cols(&v, cols, slice.cols.clone()) };
    let s = if slice.cols == (0..cols) { s } else { take_cols(&s, cols / Q8_GROUP, slice.cols.start / Q8_GROUP..slice.cols.end / Q8_GROUP) };
    Ok((v, s))
}

pub fn fp8_rank(values: &[u8], grid: &[f32], map: &[u32], cols: usize, slice: &Slice) -> Result<(Vec<u8>, Vec<f32>, Vec<u32>)> {
    ensure!(slice.cols.start.is_multiple_of(FP8_BLOCK) && (slice.cols.end.is_multiple_of(FP8_BLOCK) || slice.cols.end == cols), "an fp8 column slice must sit on {FP8_BLOCK} wide blocks");
    let kb = cols.div_ceil(FP8_BLOCK);
    let v = take_rows(values, cols, &slice.rows);
    let v = if slice.cols == (0..cols) { v } else { take_cols(&v, cols, slice.cols.clone()) };
    let rows: Vec<u32> = slice.rows.iter().flat_map(|r| map[r.clone()].iter().copied()).collect();
    let mut blocks: Vec<u32> = rows.iter().map(|m| m / kb as u32).collect();
    blocks.sort_unstable();
    blocks.dedup();
    let lo = slice.cols.start / FP8_BLOCK;
    let hi = slice.cols.end.div_ceil(FP8_BLOCK);
    let mut new_grid = Vec::new();
    let mut at = std::collections::HashMap::new();
    for b in &blocks {
        at.insert(*b, new_grid.len() as u32);
        new_grid.extend_from_slice(&grid[*b as usize * kb + lo..*b as usize * kb + hi]);
    }
    let new_map = rows.iter().map(|m| at[&(m / kb as u32)]).collect();
    Ok((v, new_grid, new_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(path: &str) -> Spec {
        let json = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(path)).unwrap();
        Spec::from_hf_config_serving(&json).unwrap()
    }

    #[test]
    fn rank_snapshots_stack_and_come_back_whole() {
        use crate::prefix::StateSnapshot;
        let parts = vec![StateSnapshot::new(vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0]), StateSnapshot::new(vec![6.0], vec![7.0, 8.0], vec![])];
        let stacked = StateSnapshot::stacked(parts.clone());
        assert_eq!(stacked.bytes(), (1 + 6 + 8) * 4);
        assert_eq!(stacked.unstack(2).unwrap(), parts);
        assert!(stacked.unstack(4).is_err());
    }

    #[test]
    fn rank_views_of_the_27b_fixture_are_the_matching_cuts_of_the_full_tensors() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-27b");
        let s = spec("tiny-qwen3.5-27b/config.json");
        let full = crate::weights::Checkpoint::open(&dir).unwrap();
        let graph = crate::graph::Graph::decoder();
        let root = full.root();
        let names: Vec<String> = full.names_under(&format!("{root}.layers.")).into_iter().chain(full.names_under("mtp.layers.")).collect();
        for tp in [2usize] {
            let mut cut_any = 0;
            let ranks: Vec<_> = (0..tp).map(|r| full.for_rank(RankCut::new(&s, graph, tp, r).unwrap()).unwrap()).collect();
            for name in &names {
                let whole = full.view(name).unwrap();
                let parts: Vec<_> = ranks.iter().map(|c| c.view(name).unwrap()).collect();
                let same = parts.iter().all(|p| p.data == whole.data);
                if same {
                    continue;
                }
                cut_any += 1;
                let elem = whole.dtype.bitsize() / 8;
                if parts[0].shape[0] == whole.shape[0] {
                    let width = whole.shape[1] * elem;
                    for row in 0..whole.shape[0] {
                        let joined: Vec<u8> = parts.iter().flat_map(|p| p.data[row * p.shape[1] * elem..(row + 1) * p.shape[1] * elem].to_vec()).collect();
                        assert_eq!(joined, whole.data[row * width..(row + 1) * width], "{name} row {row}: column cuts do not rebuild the tensor");
                    }
                } else {
                    let total: usize = parts.iter().map(|p| p.data.len()).sum();
                    assert_eq!(total, whole.data.len(), "{name}: row cuts cover every byte once");
                    assert_eq!(parts.iter().map(|p| p.shape[0]).sum::<usize>(), whole.shape[0], "{name}");
                }
            }
            assert!(cut_any >= 20, "only {cut_any} tensors were sliced");
        }
        let q = ranks_view(&full, &s, 2, 1, &format!("{root}.layers.3.self_attn.q_proj.weight"));
        let hd = s.attention.head_dim;
        assert_eq!(q.0[0], s.attention.heads / 2 * 2 * hd, "rank 1 owns half the gated q heads");
        let k = full.view(&format!("{root}.layers.3.self_attn.k_proj.weight")).unwrap();
        let k1 = ranks_view(&full, &s, 2, 1, &format!("{root}.layers.3.self_attn.k_proj.weight"));
        assert_eq!(k1.1, k.data[k.data.len() / 2..], "rank 1 owns the second kv head");
        let a_log = ranks_view(&full, &s, 2, 0, &format!("{root}.layers.0.linear_attn.A_log"));
        assert_eq!(a_log.0, vec![s.linear_attention.as_ref().unwrap().value_heads / 2]);
    }

    fn ranks_view(full: &crate::weights::Checkpoint, s: &Spec, tp: usize, rank: usize, name: &str) -> (Vec<usize>, Vec<u8>) {
        let c = full.for_rank(RankCut::new(s, crate::graph::Graph::decoder(), tp, rank).unwrap()).unwrap();
        let v = c.view(name).unwrap();
        (v.shape.clone(), v.data.to_vec())
    }

    #[test]
    fn each_rank_loads_the_matching_rows_of_the_head_and_the_draft_head() {
        use crate::spec::Quant;
        use crate::weights::Matrix;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.5-mtp");
        let mut s = spec("tiny-qwen3.5-mtp/config.json");
        s.repo = dir.display().to_string();
        let ck = crate::weights::Checkpoint::for_repo(&s.repo).unwrap().renamed(&s.tensors);
        let (_, whole) = ck.head(&s).unwrap();
        let Matrix::Bf16(whole) = whole else { panic!("the fixture head is bf16") };
        let fp4_whole = ck.draft_head(&s, Quant::Nvfp4G16).unwrap();
        let Matrix::Fp4(fp4_whole) = fp4_whole else { panic!("nvfp4 draft head") };
        let h = s.dims.hidden;
        for tp in [2usize] {
            for rank in 0..tp {
                let r = rank_spec_of(&s, tp, rank).unwrap();
                let rows = r.head_rows();
                let (_, part) = ck.head(&r).unwrap();
                let Matrix::Bf16(part) = part else { panic!("bf16") };
                assert_eq!(part.as_ref(), &whole[rows.start * h..rows.end * h], "tp {tp} rank {rank}: lm_head rows {rows:?}");
                let Matrix::Fp4(f) = ck.draft_head(&r, Quant::Nvfp4G16).unwrap() else { panic!("nvfp4") };
                assert_eq!(f.global, fp4_whole.global, "tp {tp} rank {rank}: the global scale comes from the whole head");
                assert_eq!(f.packed.as_ref(), &fp4_whole.packed[rows.start * h / 2..rows.end * h / 2], "tp {tp} rank {rank}: packed rows");
                assert_eq!(f.scales.as_ref(), &fp4_whole.scales[rows.start * h / 16..rows.end * h / 16], "tp {tp} rank {rank}: group scales");
            }
        }
    }

    #[test]
    fn moe_fixtures_cut_every_expert_and_leave_the_router_whole() {
        for f in ["tiny-qwen3.5-moe", "tiny-qwen3.5-moe-fp8"] {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f);
            let mut s = spec(&format!("{f}/config.json"));
            s.moe.as_mut().unwrap().tp_split = crate::spec::TpSplit::Columns;
            let full = crate::weights::Checkpoint::open(&dir).unwrap();
            let root = full.root();
            let names = full.names_under(&format!("{root}.layers."));
            let ranks: Vec<_> = (0..2).map(|r| full.for_rank(RankCut::new(&s, crate::graph::Graph::decoder(), 2, r).unwrap()).unwrap()).collect();
            let (mut cut, mut whole) = (0, 0);
            for name in names.iter().filter(|n| n.contains(".mlp.")) {
                let w = full.view(name).unwrap();
                let parts: Vec<_> = ranks.iter().map(|c| c.view(name).unwrap()).collect();
                let expert = name.contains(".experts.") || name.contains(".shared_expert.");
                if expert && w.shape.iter().product::<usize>() > 1 {
                    cut += 1;
                    let bytes: usize = parts.iter().map(|p| p.data.len()).sum();
                    assert_eq!(bytes, w.data.len(), "{name}: the two ranks cover every byte once");
                    assert!(parts.iter().all(|p| p.data.len() * 2 == w.data.len()), "{name}");
                } else {
                    whole += 1;
                    assert!(parts.iter().all(|p| p.data == w.data), "{name} stays whole on every rank");
                }
            }
            assert!(cut >= 6 * 9 && whole >= 2, "{f}: cut {cut} whole {whole}");
        }
    }

    #[test]
    fn the_planner_picks_the_moe_split_from_the_expert_widths() {
        use crate::spec::TpSplit;
        for f in ["tiny-qwen3.5-moe", "tiny-qwen3.5-moe-fp8"] {
            let s = spec(&format!("{f}/config.json"));
            let m = s.moe.as_ref().unwrap();
            let plan = plan_moe(&s, m, 2).unwrap();
            assert_eq!(plan.split, MoeSplit::Experts, "{f}: {}", plan.reason);
            assert!(plan.reason.contains("4 of 8 experts"), "{}", plan.reason);
            let r = rank_spec_of(&s, 2, 1).unwrap().moe.unwrap();
            assert_eq!((r.first_expert(), r.local_experts(), r.expert_intermediate), (4, 4, m.expert_intermediate));
            let mut forced = m.clone();
            forced.tp_split = TpSplit::Columns;
            assert_eq!(plan_moe(&s, &forced, 2).unwrap().split, MoeSplit::Columns);
        }
        let big = spec("qwen3.5-122b-a10b-nvfp4.config.json");
        let m = big.moe.as_ref().unwrap();
        let plan = plan_moe(&big, m, 2).unwrap();
        assert_eq!(plan.split, MoeSplit::Columns, "{}", plan.reason);
        let wide = plan_moe(&big, m, 4).unwrap();
        assert_eq!(wide.split, MoeSplit::Columns, "{}", wide.reason);
        let eight = plan_moe(&big, m, 8).unwrap();
        assert_eq!(eight.split, MoeSplit::Experts, "{}", eight.reason);
        let mut odd = m.clone();
        odd.experts = 255;
        odd.tp_split = TpSplit::Experts;
        assert!(plan_moe(&big, &odd, 2).unwrap_err().to_string().contains("does not divide"));
    }

    #[test]
    fn the_real_27b_nvfp4_checkpoint_cuts_cleanly_at_tp_2() {
        let s = Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3.8-27b-nvfp4-exact.toml")).unwrap();
        rank_spec(&s, 2).unwrap();
        let Ok(dir) = crate::weights::find_checkpoint(&s.repo) else { return };
        let ck = crate::weights::Checkpoint::open(&dir).unwrap();
        let root = ck.root();
        for rank in 0..2 {
            let cut = RankCut::new(&s, crate::graph::Graph::decoder(), 2, rank).unwrap();
            let mut cuts = 0;
            for name in ck.names_under(&format!("{root}.layers.")) {
                let shape = ck.shape_of(&name).unwrap();
                if let Some(c) = cut.rule(&name, &shape).unwrap() {
                    cuts += 1;
                    let (rows, cols) = (shape[0], shape.get(1).copied().unwrap_or(1));
                    match c {
                        Cut::Rows(r) => assert!(r.iter().all(|r| r.end <= rows) && r.iter().map(|r| r.len()).sum::<usize>() * 2 == rows, "{name} {r:?} of {rows}"),
                        Cut::Cols(c) => assert!(c.end <= cols && c.len() * 2 == cols, "{name} {c:?} of {cols}"),
                    }
                }
            }
            assert!(cuts > 64 * 5, "rank {rank}: only {cuts} tensors cut");
            eprintln!("27B nvfp4 rank {rank}: {cuts} tensors cut");
        }
    }

    #[test]
    fn every_linear_node_declares_how_it_shards() {
        let shards = crate::graph::Graph::decoder().shards().unwrap();
        for (w, s) in [("qkv", Shard::Column), ("o", Shard::Row), ("gate_up", Shard::Column), ("down", Shard::Row), ("lin_in", Shard::Column), ("lin_ba", Shard::Column), ("lin_out", Shard::Row), ("lm_head", Shard::Vocab), ("mtp_qkv", Shard::Column), ("mtp_down", Shard::Row), ("mtp_fc", Shard::Replicate), ("mtp_head", Shard::Vocab)] {
            assert_eq!(shards[w], s, "{w}");
        }
    }

    #[test]
    fn a_missing_shard_is_refused() {
        let text = crate::graph::DECODER.replacen("shard = \"column\"\n", "", 1);
        assert!(crate::graph::Graph::parse(&text).unwrap().shards().is_err());
    }

    #[test]
    fn the_27b_splits_at_tp_2_and_4_and_each_rank_owns_whole_heads() {
        let s = spec("qwen3.8-27b.config.json");
        for tp in [2usize, 4] {
            supports(&s, tp).unwrap();
            let h = heads_per_rank(&s, tp).unwrap();
            assert_eq!(h.attention * tp, s.attention.heads);
            let mut covered = vec![0usize; s.qkv_rows()];
            for rank in 0..tp {
                let sl = slice_of(&s, "qkv", Shard::Column, tp, rank).unwrap();
                assert_eq!(sl.row_count() * tp, s.qkv_rows());
                for r in &sl.rows {
                    for i in r.clone() {
                        covered[i] += 1;
                    }
                }
                assert_eq!(sl.rows[0].len(), h.attention * s.attention.head_dim);
                assert_eq!(sl.rows[1].len(), h.kv * s.attention.head_dim);
            }
            assert!(covered.iter().all(|&c| c == 1), "qkv rows are covered exactly once");
        }
        assert!(supports(&s, 3).is_err());
        assert!(supports(&s, 8).is_err());
    }

    #[test]
    fn the_hybrid_in_projection_follows_key_and_value_heads() {
        let s = spec("qwen3.8-27b.config.json");
        let la = s.linear_attention.clone().unwrap();
        let sl = slice_of(&s, "lin_in", Shard::Column, 2, 1).unwrap();
        assert_eq!(sl.rows.len(), 4);
        assert_eq!(sl.rows[0].len(), la.key_heads / 2 * la.key_head_dim);
        assert_eq!(sl.rows[2].len(), la.value_heads / 2 * la.value_head_dim);
        let ba = slice_of(&s, "lin_ba", Shard::Column, 2, 0).unwrap();
        assert_eq!(ba.rows.iter().map(|r| r.len()).collect::<Vec<_>>(), vec![la.value_heads / 2; 2]);
    }

    #[test]
    fn row_parallel_weights_split_the_input_columns() {
        let s = spec("qwen3.8-27b.config.json");
        let o = slice_of(&s, "o", Shard::Row, 4, 3).unwrap();
        assert_eq!(o.rows, one(0..s.dims.hidden));
        assert_eq!(o.cols, 3 * s.q_dim() / 4..s.q_dim());
        let down = slice_of(&s, "down", Shard::Row, 2, 0).unwrap();
        assert_eq!(down.cols, 0..s.dims.intermediate / 2);
    }

    #[test]
    fn the_vocabulary_splits_with_a_short_last_rank() {
        let s = spec("qwen3.8-27b.config.json");
        let parts: Vec<_> = (0..3).map(|r| slice_of(&s, "lm_head", Shard::Vocab, 3, r).unwrap().row_count()).collect();
        assert_eq!(parts.iter().sum::<usize>(), s.dims.vocab);
        assert!(parts[2] <= parts[0]);
    }

    #[test]
    fn vocab_slices_are_aligned_tile_the_vocabulary_and_refuse_an_empty_rank() {
        assert_eq!(vocab_each(248320, 2).unwrap(), 124160);
        let spans: Vec<_> = (0..3).map(|r| vocab_span(1024, 3, r).unwrap()).collect();
        assert_eq!(spans, vec![0..352, 352..704, 704..1024]);
        let spans: Vec<_> = (0..3).map(|r| vocab_span(1000, 3, r).unwrap()).collect();
        assert_eq!(spans, vec![0..336, 336..672, 672..1000]);
        assert!(vocab_each(32, 3).is_err());
        assert_eq!(vocab_span(512, 1, 0).unwrap(), 0..512);
    }

    #[test]
    fn rank_specs_own_their_vocabulary_rows_and_keep_the_full_vocabulary() {
        let s = spec("tiny-qwen3.5-27b/config.json");
        assert_eq!(s.head_rows(), 0..s.dims.vocab);
        let r: Vec<_> = (0..2).map(|k| rank_spec_of(&s, 2, k).unwrap()).collect();
        assert_eq!(r[0].head_rows(), 0..512);
        assert_eq!(r[1].head_rows(), 512..1024);
        for k in &r {
            assert_eq!(k.dims.vocab, s.dims.vocab);
            assert_eq!(crate::gemm::linear_shape(k, "lm_head").n, 512);
            assert_eq!(crate::gemm::linear_shape(k, "mtp_head").n, 512);
            assert_eq!(crate::plan::Weight::LmHead.rows(k), 512);
        }
        assert_eq!(rank_spec_of(&s, 1, 0).unwrap().dims.head_rows, None);
        let mut odd = s.clone();
        odd.mtp = None;
        odd.dims.vocab = 1000;
        assert_eq!(head_split(&odd, 3).unwrap(), Some(336));
        odd.dims.vocab = 1001;
        assert_eq!(head_split(&odd, 3).unwrap(), None, "an odd last slice of 329 rows keeps the head whole");
        odd.dims.vocab = 1000;
        odd.head_quant = Some(crate::spec::Quant::Nvfp4G16);
        assert_eq!(head_split(&odd, 3).unwrap(), None, "a last nvfp4 slice of 328 rows keeps the head whole");
        let mut phi = spec("tiny-phi3/config.json");
        phi.attention.heads = 6;
        phi.attention.kv_heads = 3;
        phi.dims.intermediate = 288;
        phi.dims.vocab = 1024;
        assert_eq!(rank_spec_of(&phi, 3, 2).unwrap().dims.head_rows, Some(704..1024));
    }

    #[test]
    fn tiny_fixtures_support_tp_2_and_the_moe_fixture_is_split_per_expert() {
        for f in ["tiny-qwen3.5-27b", "tiny-qwen3.5"] {
            let s = spec(&format!("{f}/config.json"));
            supports(&s, 2).unwrap_or_else(|e| panic!("{f}: {e}"));
        }
        let moe = spec("tiny-qwen3.5-moe/config.json");
        assert!(supports(&moe, 1).is_ok());
        assert!(supports(&moe, 2).is_ok());
    }

    #[test]
    fn column_shards_rebuild_the_matrix_and_row_shards_sum_to_the_product() {
        let (rows, cols, tp) = (8usize, 12usize, 2usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) - 5.0).collect();
        let full: Vec<f32> = (0..rows).map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum()).collect();
        let mut stacked = Vec::new();
        for rank in 0..tp {
            stacked.extend(take_rows(&w, cols, &[split_even(rows, tp, rank).unwrap()]));
        }
        assert_eq!(stacked, w);
        let mut summed = vec![0.0f32; rows];
        for rank in 0..tp {
            let c = split_even(cols, tp, rank).unwrap();
            let part = take_cols(&w, cols, c.clone());
            for r in 0..rows {
                summed[r] += (0..c.len()).map(|k| part[r * c.len() + k] * x[c.start + k]).sum::<f32>();
            }
        }
        assert_eq!(summed, full);
    }

    #[test]
    fn nvfp4_slices_keep_packed_bytes_and_group_scales_aligned() {
        let (rows, cols) = (4usize, 64usize);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|i| i as u8).collect();
        let scales: Vec<u8> = (0..rows * cols / FP4_GROUP).map(|i| 100 + i as u8).collect();
        let slice = Slice { rows: one(2..4), cols: 32..64 };
        let (p, s) = fp4_rank(&packed, &scales, cols, &slice).unwrap();
        assert_eq!(p.len(), 2 * 16);
        assert_eq!(&p[..16], &packed[2 * 32 + 16..2 * 32 + 32]);
        assert_eq!(&s[..2], &scales[2 * 4 + 2..2 * 4 + 4]);
        let bad = Slice { rows: one(0..4), cols: 8..64 };
        assert!(fp4_rank(&packed, &scales, cols, &bad).is_err());
    }

    #[test]
    fn q8_slices_follow_the_group_scales() {
        let (rows, cols) = (2usize, 128usize);
        let v: Vec<i8> = (0..rows * cols).map(|i| (i % 127) as i8).collect();
        let s: Vec<f32> = (0..rows * cols / Q8_GROUP).map(|i| i as f32).collect();
        let (pv, ps) = q8_rank(&v, &s, cols, &Slice { rows: one(1..2), cols: 64..128 }).unwrap();
        assert_eq!(pv, v[cols + 64..cols + 128]);
        assert_eq!(ps, vec![6.0, 7.0]);
    }

    #[test]
    fn fp8_block_scales_follow_row_and_column_slices() {
        let (rows, cols) = (256usize, 256usize);
        let kb = cols / FP8_BLOCK;
        let values: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
        let grid: Vec<f32> = (0..(rows / FP8_BLOCK) * kb).map(|i| i as f32 + 1.0).collect();
        let map: Vec<u32> = (0..rows).map(|r| (r / FP8_BLOCK * kb) as u32).collect();
        let slice = Slice { rows: one(128..256), cols: 128..256 };
        let (v, g, m) = fp8_rank(&values, &grid, &map, cols, &slice).unwrap();
        assert_eq!(v.len(), 128 * 128);
        assert_eq!(g, vec![4.0]);
        assert!(m.iter().all(|&i| g[i as usize] == 4.0));
        let whole = Slice { rows: one(0..256), cols: 0..128 };
        let (_, g, m) = fp8_rank(&values, &grid, &map, cols, &whole).unwrap();
        assert_eq!(g, vec![1.0, 3.0]);
        assert_eq!(g[m[0] as usize], 1.0);
        assert_eq!(g[m[200] as usize], 3.0);
    }
}
