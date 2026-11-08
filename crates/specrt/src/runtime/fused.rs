use super::step::{DecodeRow, Stepper, params};
use super::{DeviceMixer, GemmOut, Runtime, Sampling, cfg, grid};
use anyhow::{Result, ensure};
use cudarc::driver::{LaunchConfig, PushKernelArg};

pub struct PrefillPart<'a> {
    pub slot: usize,
    pub tokens: &'a [u32],
    pub start: usize,
    pub sample: Option<(Sampling, u32)>,
}

impl Runtime {
    pub fn fused_step(
        &mut self,
        st: &mut Stepper,
        rows: &[DecodeRow],
        part: PrefillPart<'_>,
    ) -> Result<(Vec<u32>, Option<u32>)> {
        ensure!(self.use_mma, "serving needs the tensor core path");
        let nd = rows.len();
        let np = part.tokens.len();
        let n = nd + np;
        ensure!(nd >= 1 && nd <= self.slots, "{nd} decode rows for {} slots", self.slots);
        ensure!(np >= 1 && n <= self.chunk, "fused step of {n} rows");
        ensure!(part.slot < self.slots, "slot {} out of {}", part.slot, self.slots);
        ensure!(part.start + np <= self.max_ctx, "prefill past max_ctx");
        ensure!(rows.iter().all(|r| r.slot < self.slots && r.pos < self.max_ctx), "decode row out of range");
        ensure!(rows.iter().all(|r| r.slot != part.slot), "decode row in the prefilling slot");
        if part.start == 0 {
            self.reset_slot(part.slot)?;
        }
        let ids: Vec<i32> =
            rows.iter().map(|r| r.token as i32).chain(part.tokens.iter().map(|&t| t as i32)).collect();
        let pos: Vec<i32> = rows.iter().map(|r| r.pos as i32).collect();
        let slots: Vec<i32> = rows.iter().map(|r| r.slot as i32).collect();
        let mut p: Vec<i32> = rows.iter().flat_map(|r| params(r.sampling, r.step)).collect();
        let sampled = part.sample.is_some();
        if let Some((sampling, step)) = part.sample {
            p.extend(params(sampling, step));
        }
        self.stream.memcpy_htod(&ids, &mut self.pre.tokens.slice_mut(..n))?;
        self.stream.memcpy_htod(&pos, &mut st.bb.pos.slice_mut(..nd))?;
        self.stream.memcpy_htod(&slots, &mut st.bb.slots.slice_mut(..nd))?;
        self.stream.memcpy_htod(&[part.start as i32], &mut self.buf.pos)?;
        self.stream.memcpy_htod(&p, &mut st.params.slice_mut(..p.len()))?;
        self.seq = part.slot as i32;
        let ran = self.fused_layers(st, nd, np, sampled);
        self.seq = 0;
        ran?;
        let out_rows = nd + usize::from(sampled);
        let mut b = self.stream.launch_builder(self.kernel("sample_rows"));
        b.arg(&st.bb.logits).arg(&st.bb.tokens).arg(&st.params);
        unsafe { b.launch(cfg(out_rows as u32, 1024)) }?;
        let out = self.stream.clone_dtoh(&st.bb.tokens.slice(..out_rows))?;
        let decoded = out[..nd].iter().map(|&t| t as u32).collect();
        Ok((decoded, sampled.then(|| out[nd] as u32)))
    }

    fn fused_layers(&self, st: &Stepper, nd: usize, np: usize, sampled: bool) -> Result<()> {
        let s = &self.spec;
        let hidden = s.dims.hidden;
        let n = nd + np;
        let pre = &self.pre;
        let bb = &st.bb;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&pre.tokens).arg(&pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        for layer in &self.layers {
            self.rms_split(&pre.x.as_view(), &layer.ln1, &pre.xn_planes, n)?;
            match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
                    let la = &gb.la;
                    self.tc("lin_in", &g.in_proj, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_lin), la.in_proj_rows(), n)?;
                    self.tc("lin_ba", &g.ba, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_ba), la.ba_rows(), n)?;
                    self.gdn_decode_rows(g, &bb.slots, nd)?;
                    self.gdn_prefill_rows(g, np, nd)?;
                    self.tc("lin_out", &g.out, None, gb.planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                }
                DeviceMixer::Attention(l) => {
                    self.tc(
                        "qkv",
                        &l.qkv,
                        l.qkv_bias.as_ref(),
                        pre.xn_planes.all(),
                        GemmOut::F32(&pre.qkv),
                        s.qkv_rows(),
                        n,
                    )?;
                    self.attn_decode_rows(l, bb, nd)?;
                    self.attn_prefill_rows(l, np, nd)?;
                    self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                }
            }
            self.mlp_prefill(layer, n)?;
        }
        let (hi, lo) = (pre.xn_planes.hi.as_view(), pre.xn_planes.lo.as_view());
        self.rms_split_into(&pre.x.as_view(), &self.norm, &hi, &lo, nd)?;
        let head_rows = if sampled {
            let last = pre.x.slice((n - 1) * hidden..n * hidden);
            let hi = pre.xn_planes.hi.slice(nd * hidden..(nd + 1) * hidden);
            let lo = pre.xn_planes.lo.slice(nd * hidden..(nd + 1) * hidden);
            self.rms_split_into(&last, &self.norm, &hi, &lo, 1)?;
            nd + 1
        } else {
            nd
        };
        self.tc(
            "lm_head",
            &self.lm_head,
            None,
            pre.xn_planes.all(),
            GemmOut::F32(&bb.logits),
            s.dims.vocab,
            head_rows,
        )
    }

    pub fn step_logits(&self, st: &Stepper, from: usize, rows: usize) -> Result<Vec<f32>> {
        let vocab = self.spec.dims.vocab;
        Ok(self.stream.clone_dtoh(&st.bb.logits.slice(from * vocab..(from + rows) * vocab))?)
    }

    pub fn prefill_logits(&self) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(&self.buf.logits)?)
    }
}
