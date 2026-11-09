use super::{BatchBuffers, Runtime, Sampling, cfg};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, PushKernelArg};
use std::collections::HashMap;

pub const GRAPH_SIZES: [usize; 6] = [1, 2, 4, 8, 16, 32];

pub fn pad_rows(rows: usize) -> usize {
    GRAPH_SIZES.iter().copied().find(|&s| s >= rows).unwrap_or(rows)
}

pub fn round8(rows: usize) -> usize {
    rows.div_ceil(8) * 8
}

pub fn row_capacity(slots: usize) -> usize {
    pad_rows(slots).max(round8(slots))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeRow {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
    pub sampling: Sampling,
    pub step: u32,
}

pub struct Stepper {
    pub(super) bb: BatchBuffers,
    pub(super) params: CudaSlice<i32>,
    graphs: HashMap<usize, CudaGraph>,
    use_graphs: bool,
}

impl Stepper {
    pub fn set_graphs(&mut self, on: bool) {
        self.use_graphs = on;
    }

    pub fn captured_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }
}

pub(super) fn params(sampling: Sampling, step: u32) -> [i32; 6] {
    [
        sampling.inverse_temperature().to_bits() as i32,
        sampling.seed as i32,
        step as i32,
        sampling.top_k as i32,
        sampling.top_p.to_bits() as i32,
        sampling.min_p.to_bits() as i32,
    ]
}

impl Runtime {
    pub fn stepper(&self) -> Result<Stepper> {
        let rows = row_capacity(self.slots) + 1;
        let parts = rows * self.spec.attention.heads * crate::codegen::BATCH_DECODE_SPLITS;
        let bb = BatchBuffers {
            tokens: self.stream.alloc_zeros(rows)?,
            logits: self.stream.alloc_zeros(rows * self.spec.dims.vocab)?,
            history: self.stream.alloc_zeros(1)?,
            part_acc: self.stream.alloc_zeros(parts * self.spec.attention.head_dim)?,
            part_ml: self.stream.alloc_zeros(parts * 2)?,
            pos: self.stream.alloc_zeros(rows)?,
            slots: self.stream.alloc_zeros(rows)?,
        };
        Ok(Stepper {
            bb,
            params: self.stream.alloc_zeros(6 * rows)?,
            graphs: HashMap::new(),
            use_graphs: self.spec.runtime.graphs,
        })
    }

    pub fn prefill_chunk_limit(&self) -> usize {
        self.prefill_limit
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    pub fn prefill_slot(
        &mut self,
        st: &mut Stepper,
        slot: usize,
        tokens: &[u32],
        start: usize,
        sample: Option<(Sampling, u32)>,
    ) -> Result<Option<u32>> {
        ensure!(self.use_mma, "serving needs the tensor core path");
        ensure!(slot < self.slots, "slot {slot} out of {}", self.slots);
        ensure!(!tokens.is_empty() && tokens.len() <= self.prefill_limit, "prefill chunk of {} tokens", tokens.len());
        ensure!(start + tokens.len() <= self.max_ctx, "prefill past max_ctx");
        if start == 0 {
            self.reset_slot(slot)?;
        }
        self.seq = slot as i32;
        let ran = self.prefill_chunk(tokens, start, None);
        self.seq = 0;
        ran?;
        self.mtp_after_prefill(slot, tokens, start)?;
        let Some((sampling, step)) = sample else { return Ok(None) };
        self.stream.memcpy_htod(&params(sampling, step), &mut st.params.slice_mut(..6))?;
        let mut b = self.stream.launch_builder(self.kernel("sample_rows"));
        b.arg(&self.buf.logits).arg(&st.bb.tokens).arg(&st.params);
        unsafe { b.launch(cfg(1, 1024)) }?;
        let token = self.stream.clone_dtoh(&st.bb.tokens.slice(..1))?;
        Ok(Some(token[0] as u32))
    }

    fn decode_and_sample(&self, st: &Stepper, n: usize) -> Result<()> {
        self.decode_rows(&st.bb, n)?;
        let mut b = self.stream.launch_builder(self.kernel("sample_rows"));
        b.arg(&st.bb.logits).arg(&st.bb.tokens).arg(&st.params);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        Ok(())
    }

    fn capture_decode(&self, st: &Stepper, n: usize) -> Result<CudaGraph> {
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.decode_and_sample(st, n);
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        graph.context("empty decode graph")
    }

    pub fn decode(&mut self, st: &mut Stepper, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        ensure!(self.use_mma, "serving needs the tensor core path");
        let live = rows.len();
        ensure!(live >= 1 && live <= self.slots, "{live} decode rows for {} slots", self.slots);
        ensure!(rows.iter().all(|r| r.slot < self.slots && r.pos < self.max_ctx), "decode row out of range");
        let n = if st.use_graphs { pad_rows(live) } else { live };
        let pad = DecodeRow { slot: self.slots, token: 0, pos: 0, sampling: Sampling::GREEDY, step: 0 };
        let padded: Vec<DecodeRow> = rows.iter().copied().chain(std::iter::repeat(pad)).take(n).collect();
        let tokens: Vec<i32> = padded.iter().map(|r| r.token as i32).collect();
        let pos: Vec<i32> = padded.iter().map(|r| r.pos as i32).collect();
        let slots: Vec<i32> = padded.iter().map(|r| r.slot as i32).collect();
        let p: Vec<i32> = padded.iter().flat_map(|r| params(r.sampling, r.step)).collect();
        self.stream.memcpy_htod(&tokens, &mut st.bb.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&pos, &mut st.bb.pos.slice_mut(..n))?;
        self.stream.memcpy_htod(&slots, &mut st.bb.slots.slice_mut(..n))?;
        self.stream.memcpy_htod(&p, &mut st.params.slice_mut(..6 * n))?;
        if st.use_graphs {
            if !st.graphs.contains_key(&n) {
                let graph = self.capture_decode(st, n)?;
                graph.upload()?;
                st.graphs.insert(n, graph);
            }
            st.graphs[&n].launch()?;
        } else {
            self.decode_and_sample(st, n)?;
        }
        let out = self.stream.clone_dtoh(&st.bb.tokens.slice(..live))?;
        self.mtp_after_decode(rows)?;
        Ok(out.into_iter().map(|t| t as u32).collect())
    }
}
