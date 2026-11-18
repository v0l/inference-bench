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
        "gate_up" => Layout { rows: vec![2 * d.intermediate], cols: d.hidden },
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

fn split_ceil(len: usize, tp: usize, rank: usize) -> Range<usize> {
    let each = len.div_ceil(tp);
    (rank * each).min(len)..((rank + 1) * each).min(len)
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
        Shard::Vocab => Ok(Slice { rows: vec![split_ceil(total, tp, rank)], cols: 0..l.cols }),
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
        ensure!(spec.moe.is_none(), "tp over mixture of experts models is not built yet ({})", spec.name);
        ensure!(spec.runtime.backend != crate::spec::BackendChoice::Wgpu, "tp runs on CUDA only");
        ensure!(spec.quant != crate::spec::Quant::Gguf, "tp does not slice GGUF weights yet ({})", spec.name);
        ensure!(!spec.uses_branch(), "tp does not cover sandwich or scaled residual norms yet ({})", spec.name);
        ensure!(!spec.attention.qk_norm_full, "tp does not split full-width qk norms yet ({})", spec.name);
        if let Some(l) = &spec.linear_attention {
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

pub fn rank_spec(spec: &Spec, tp: usize) -> Result<Spec> {
    supports(spec, tp)?;
    if tp == 1 {
        return Ok(spec.clone());
    }
    let h = heads_per_rank(spec, tp)?;
    let mut s = spec.clone();
    s.attention.heads = h.attention;
    s.attention.kv_heads = h.kv;
    s.dims.intermediate = spec.dims.intermediate / tp;
    if let Some(l) = s.linear_attention.as_mut() {
        l.key_heads = h.key;
        l.value_heads = h.value;
    }
    s.vision = None;
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
        for (w, s) in [("qkv", Shard::Column), ("o", Shard::Row), ("gate_up", Shard::Column), ("down", Shard::Row), ("lin_in", Shard::Column), ("lin_ba", Shard::Column), ("lin_out", Shard::Row), ("lm_head", Shard::Replicate), ("mtp_qkv", Shard::Column), ("mtp_down", Shard::Row), ("mtp_fc", Shard::Replicate), ("mtp_head", Shard::Replicate)] {
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
    fn tiny_fixtures_support_tp_2_and_the_moe_fixture_is_refused() {
        for f in ["tiny-qwen3.5-27b", "tiny-qwen3.5"] {
            let s = spec(&format!("{f}/config.json"));
            supports(&s, 2).unwrap_or_else(|e| panic!("{f}: {e}"));
        }
        let moe = spec("tiny-qwen3.5-moe/config.json");
        assert!(supports(&moe, 1).is_ok());
        assert!(supports(&moe, 2).unwrap_err().to_string().contains("mixture of experts"));
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
