use super::{BatchBuffers, Runtime, Sampling, cfg};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, PushKernelArg};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeRow {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
    pub sampling: Sampling,
    pub step: u32,
}

pub struct Stepper {
    bb: BatchBuffers,
    params: CudaSlice<i32>,
    rows: usize,
}

fn params(sampling: Sampling, step: u32) -> [i32; 3] {
    [sampling.inverse_temperature().to_bits() as i32, sampling.seed as i32, step as i32]
}

impl Runtime {
    pub fn stepper(&self) -> Result<Stepper> {
        let rows = self.slots;
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
        Ok(Stepper { bb, params: self.stream.alloc_zeros(3 * rows)?, rows })
    }

    pub fn prefill_chunk_limit(&self) -> usize {
        self.chunk
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
        ensure!(!tokens.is_empty() && tokens.len() <= self.chunk, "prefill chunk of {} tokens", tokens.len());
        ensure!(start + tokens.len() <= self.max_ctx, "prefill past max_ctx");
        if start == 0 {
            self.reset_slot(slot)?;
        }
        self.seq = slot as i32;
        let ran = self.prefill_chunk(tokens, start, None);
        self.seq = 0;
        ran?;
        let Some((sampling, step)) = sample else { return Ok(None) };
        self.stream.memcpy_htod(&params(sampling, step), &mut st.params.slice_mut(..3))?;
        let mut b = self.stream.launch_builder(self.kernel("sample_rows"));
        b.arg(&self.buf.logits).arg(&st.bb.tokens).arg(&st.params);
        unsafe { b.launch(cfg(1, 1024)) }?;
        let token = self.stream.clone_dtoh(&st.bb.tokens.slice(..1))?;
        Ok(Some(token[0] as u32))
    }

    pub fn decode(&mut self, st: &mut Stepper, rows: &[DecodeRow]) -> Result<Vec<u32>> {
        ensure!(self.use_mma, "serving needs the tensor core path");
        let n = rows.len();
        ensure!(n >= 1 && n <= st.rows, "{n} decode rows for {} slots", st.rows);
        ensure!(rows.iter().all(|r| r.slot < self.slots && r.pos < self.max_ctx), "decode row out of range");
        let tokens: Vec<i32> = rows.iter().map(|r| r.token as i32).collect();
        let pos: Vec<i32> = rows.iter().map(|r| r.pos as i32).collect();
        let slots: Vec<i32> = rows.iter().map(|r| r.slot as i32).collect();
        let p: Vec<i32> = rows.iter().flat_map(|r| params(r.sampling, r.step)).collect();
        self.stream.memcpy_htod(&tokens, &mut st.bb.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&pos, &mut st.bb.pos.slice_mut(..n))?;
        self.stream.memcpy_htod(&slots, &mut st.bb.slots.slice_mut(..n))?;
        self.stream.memcpy_htod(&p, &mut st.params.slice_mut(..3 * n))?;
        self.decode_rows(&st.bb, n)?;
        let mut b = self.stream.launch_builder(self.kernel("sample_rows"));
        b.arg(&st.bb.logits).arg(&st.bb.tokens).arg(&st.params);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        let out = self.stream.clone_dtoh(&st.bb.tokens.slice(..n))?;
        Ok(out.into_iter().map(|t| t as u32).collect())
    }
}
