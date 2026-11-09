use super::{
    BatchBuffers, DeviceAttn, DeviceLayer, DeviceMatrix, DeviceMixer, GemmOut, Planes, Runtime, Sampling, cfg, grid,
};
use crate::codegen::LA_KSPLIT;
use crate::spec::Spec;
use crate::weights::{Checkpoint, Mixer};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub(super) struct DeviceMtp {
    fc: DeviceMatrix,
    norm_embed: CudaSlice<f32>,
    norm_hidden: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    layer: DeviceLayer,
}

impl DeviceMtp {
    pub(super) fn load(stream: &Arc<CudaStream>, checkpoint: &Checkpoint, spec: &Spec) -> Result<Option<DeviceMtp>> {
        if spec.speculative().is_none() {
            return Ok(None);
        }
        let w = checkpoint.mtp(spec).context("MTP layer")?;
        let up = |v: &Option<Vec<f32>>| -> Result<Option<CudaSlice<f32>>> {
            Ok(match v {
                Some(v) => Some(stream.clone_htod(v)?),
                None => None,
            })
        };
        let Mixer::Attention(a) = &w.layer.mixer else { unreachable!("the MTP layer is full attention") };
        let layer = DeviceLayer {
            ln1: stream.clone_htod(&w.layer.ln1)?,
            ln2: stream.clone_htod(&w.layer.ln2)?,
            mixer: DeviceMixer::Attention(DeviceAttn {
                qkv: DeviceMatrix::upload(stream, &a.qkv)?,
                qkv_bias: up(&a.qkv_bias)?,
                q_norm: up(&a.q_norm)?,
                k_norm: up(&a.k_norm)?,
                o: DeviceMatrix::upload(stream, &a.o)?,
                kv: spec.attention_layers() as i32,
            }),
            gate_up: DeviceMatrix::upload(stream, &w.layer.gate_up)?,
            down: DeviceMatrix::upload(stream, &w.layer.down)?,
        };
        let m = DeviceMtp {
            fc: DeviceMatrix::upload(stream, &w.fc)?,
            norm_embed: stream.clone_htod(&w.norm_embed)?,
            norm_hidden: stream.clone_htod(&w.norm_hidden)?,
            norm: stream.clone_htod(&w.norm)?,
            layer,
        };
        stream.synchronize()?;
        Ok(Some(m))
    }

    pub(super) fn bytes(&self) -> u64 {
        self.fc.bytes()
            + self.layer.bytes()
            + (self.norm_embed.num_bytes() + self.norm_hidden.num_bytes() + self.norm.num_bytes()) as u64
    }
}

pub(super) struct MtpState {
    w: DeviceMtp,
    k: usize,
    cap: usize,
    rows: usize,
    vb: BatchBuffers,
    fc_planes: Planes,
    vx: CudaSlice<f32>,
    pending: CudaSlice<f32>,
    cg: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    drafts: CudaSlice<i32>,
    ctok: CudaSlice<i32>,
    hidx: CudaSlice<i32>,
    sidx: CudaSlice<i32>,
    didx: CudaSlice<i32>,
    vout: CudaSlice<i32>,
    accepted: CudaSlice<i32>,
    max_accept: CudaSlice<i32>,
    params: CudaSlice<i32>,
    klog: CudaSlice<f32>,
    dlog: CudaSlice<f32>,
    glog: CudaSlice<f32>,
    clog: CudaSlice<f32>,
    host_drafts: Vec<Vec<u32>>,
    stats: SpecStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub steps: u64,
    pub rows: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub position_accepted: [u64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecRow {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
    pub sampling: Sampling,
    pub step: u32,
    pub max_accept: usize,
}

enum Attn {
    Contiguous,
    Rows,
}

impl MtpState {
    pub(super) fn new(rt: &Runtime, w: DeviceMtp) -> Result<MtpState> {
        let s = &rt.spec;
        let m = s.speculative().context("speculation is off")?;
        let k = m.draft_tokens;
        let cap = m.max_batch.unwrap_or(rt.slots).min(rt.slots);
        let rows = cap * (k + 1);
        ensure!(rows <= rt.chunk, "{rows} verify rows do not fit a step of {}", rt.chunk);
        let la = s.linear_attention.as_ref().context("MTP needs linear attention")?;
        let stream = &rt.stream;
        let h = s.dims.hidden;
        let vb_rows = rows.max(rt.slots) + 1;
        let wide = rt.chunk.max(vb_rows);
        let parts = vb_rows * s.attention.heads * crate::codegen::BATCH_DECODE_SPLITS;
        let layers = s.linear_layers();
        let vd = la.value_heads * la.key_head_dim;
        let z = |n: usize| stream.alloc_zeros::<f32>(n.max(1));
        let zi = |n: usize| stream.alloc_zeros::<i32>(n.max(1));
        Ok(MtpState {
            w,
            k,
            cap,
            rows,
            vb: BatchBuffers {
                tokens: zi(vb_rows)?,
                logits: z(rows * s.dims.vocab)?,
                history: zi(1)?,
                part_acc: z(parts * s.attention.head_dim)?,
                part_ml: z(parts * 2)?,
                pos: zi(vb_rows)?,
                slots: zi(vb_rows)?,
            },
            fc_planes: Planes::new(stream, wide * 2 * h)?,
            vx: z(wide * h)?,
            pending: z((rt.slots + 1) * h)?,
            cg: z(cap * h)?,
            logits: z(cap * s.dims.vocab)?,
            drafts: zi(cap * k)?,
            ctok: zi(cap)?,
            hidx: zi(wide)?,
            sidx: zi(vb_rows)?,
            didx: zi(vb_rows)?,
            vout: zi(rows)?,
            accepted: zi(cap)?,
            max_accept: zi(cap)?,
            params: zi(3 * cap)?,
            klog: z(layers * rows * vd)?,
            dlog: z(layers * rows * vd)?,
            glog: z(layers * rows * la.value_heads)?,
            clog: z(layers * rows * la.conv_dim())?,
            host_drafts: vec![Vec::new(); rt.slots + 1],
            stats: SpecStats::default(),
        })
    }
}

impl Runtime {
    fn mtp(&self) -> &MtpState {
        self.mtp.as_ref().expect("an MTP spec")
    }

    pub fn draft_tokens(&self) -> Option<usize> {
        self.mtp.as_ref().map(|m| m.k)
    }

    pub fn speculation_cap(&self) -> usize {
        self.mtp.as_ref().map_or(0, |m| m.cap)
    }

    pub fn spec_stats(&self) -> SpecStats {
        self.mtp.as_ref().map(|m| m.stats).unwrap_or_default()
    }

    pub fn drafts(&self, slot: usize) -> Vec<u32> {
        self.mtp.as_ref().map(|m| m.host_drafts[slot].clone()).unwrap_or_default()
    }

    pub fn set_drafts(&mut self, slot: usize, drafts: &[u32]) -> Result<()> {
        let m = self.mtp.as_mut().context("no MTP layer")?;
        ensure!(drafts.len() == m.k, "{} drafts for k = {}", drafts.len(), m.k);
        m.host_drafts[slot] = drafts.to_vec();
        Ok(())
    }

    pub(super) fn mtp_forget(&mut self, slot: usize) {
        if let Some(m) = self.mtp.as_mut() {
            m.host_drafts[slot].clear();
        }
    }

    fn up_i32(&self, host: &[i32], dst: &CudaSlice<i32>) -> Result<()> {
        use cudarc::driver::DevicePtr;
        ensure!(host.len() <= dst.len(), "{} values for a device buffer of {}", host.len(), dst.len());
        if host.is_empty() {
            return Ok(());
        }
        let (ptr, _sync) = dst.device_ptr(&self.stream);
        unsafe { cudarc::driver::result::memcpy_htod_async(ptr, host, self.stream.cu_stream()) }?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn copy_hidden(&self, src: &CudaSlice<f32>, sidx: &[i32], dst: &CudaSlice<f32>, didx: &[i32]) -> Result<()> {
        if sidx.is_empty() {
            return Ok(());
        }
        let m = self.mtp();
        self.up_i32(sidx, &m.sidx)?;
        self.up_i32(didx, &m.didx)?;
        let mut b = self.stream.launch_builder(self.kernel("mtp_copy"));
        b.arg(src).arg(&m.sidx).arg(dst).arg(&m.didx);
        let config = LaunchConfig {
            grid_dim: (grid(self.spec.dims.hidden, 256), sidx.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn mtp_layer(&self, tokens: &CudaView<i32>, xsrc: &CudaSlice<f32>, pre: &CudaSlice<f32>, n: usize, attn: Attn) -> Result<()> {
        let m = self.mtp();
        let s = &self.spec;
        let h = s.dims.hidden;
        let p = &self.pre;
        let mut b = self.stream.launch_builder(self.kernel("mtp_fc_in"));
        b.arg(&self.embed)
            .arg(tokens)
            .arg(xsrc)
            .arg(&m.pending)
            .arg(&m.hidx)
            .arg(pre)
            .arg(&m.w.norm_embed)
            .arg(&m.w.norm_hidden)
            .arg(&m.fc_planes.hi)
            .arg(&m.fc_planes.lo);
        unsafe { b.launch(cfg(n as u32, 256)) }?;
        self.tc("mtp_fc", &m.w.fc, None, m.fc_planes.all(), GemmOut::F32(&p.x), h, n)?;
        let layer = &m.w.layer;
        let DeviceMixer::Attention(l) = &layer.mixer else { unreachable!("the MTP layer is full attention") };
        self.rms_split(&p.x.as_view(), &layer.ln1, &p.xn_planes, n)?;
        self.tc("mtp_qkv", &l.qkv, None, p.xn_planes.all(), GemmOut::F32(&p.qkv), s.qkv_rows(), n)?;
        match attn {
            Attn::Contiguous => self.attn_prefill_rows(l, n, 0)?,
            Attn::Rows => self.attn_decode_rows(l, &m.vb, n)?,
        }
        self.tc("mtp_o", &l.o, None, p.attn_planes.all(), GemmOut::F32(&p.x), h, n)?;
        self.rms_split(&p.x.as_view(), &layer.ln2, &p.xn_planes, n)?;
        let inter = s.dims.intermediate;
        self.tc("mtp_gate_up", &layer.gate_up, None, p.xn_planes.all(), GemmOut::Planes(&p.act_planes), 2 * inter, n)?;
        self.tc("mtp_down", &layer.down, None, p.act_planes.all(), GemmOut::F32(&p.x), h, n)
    }

    fn mtp_heads(&self, sidx: &[i32], col: usize) -> Result<()> {
        let m = self.mtp();
        let n = sidx.len();
        let ident: Vec<i32> = (0..n as i32).collect();
        self.copy_hidden(&self.pre.x, sidx, &m.cg, &ident)?;
        self.rms_split(&m.cg.as_view(), &m.w.norm, &self.pre.xn_planes, n)?;
        self.tc("lm_head", &self.lm_head, None, self.pre.xn_planes.all(), GemmOut::F32(&m.logits), self.spec.dims.vocab, n)?;
        let (col, k) = (col as i32, m.k as i32);
        let mut b = self.stream.launch_builder(self.kernel("mtp_draft"));
        b.arg(&m.logits).arg(&m.drafts).arg(&col).arg(&k).arg(&m.ctok);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        Ok(())
    }

    fn mtp_rows(&self, tokens: &[i32], hidx: &[i32], pos: &[i32], slots: &[i32], pre: &CudaSlice<f32>) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let m = self.mtp();
        self.up_i32(tokens, &m.vb.tokens)?;
        self.up_i32(hidx, &m.hidx)?;
        self.up_i32(pos, &m.vb.pos)?;
        self.up_i32(slots, &m.vb.slots)?;
        self.mtp_layer(&m.vb.tokens.as_view(), &m.vx, pre, tokens.len(), Attn::Rows)
    }

    fn mtp_chain(&self, heads: &[i32], q0: &[usize], slots: &[i32]) -> Result<Vec<Vec<u32>>> {
        let m = self.mtp();
        let n = heads.len();
        self.mtp_heads(heads, 0)?;
        let ident: Vec<i32> = (0..n as i32).collect();
        for i in 1..m.k {
            let pos: Vec<i32> = q0.iter().map(|&q| (q + i) as i32).collect();
            self.up_i32(&ident, &m.hidx)?;
            self.up_i32(&pos, &m.vb.pos)?;
            self.up_i32(slots, &m.vb.slots)?;
            self.mtp_layer(&m.ctok.slice(..n), &m.cg, &m.w.norm, n, Attn::Rows)?;
            self.mtp_heads(&ident, i)?;
        }
        let d = self.stream.clone_dtoh(&m.drafts.slice(..n * m.k))?;
        Ok(d.chunks(m.k).map(|c| c.iter().map(|&t| t as u32).collect()).collect())
    }

    pub(super) fn mtp_after_prefill(&mut self, slot: usize, tokens: &[u32], start: usize) -> Result<()> {
        if self.mtp.is_none() {
            return Ok(());
        }
        let n = tokens.len();
        let h = self.spec.dims.hidden;
        let m = self.mtp();
        self.copy_rows(&self.pre.x.slice(..n * h), &m.vx.slice(..n * h), n * h)?;
        let (from, mut hidx) = if start == 0 { (1, Vec::new()) } else { (0, vec![-1 - slot as i32]) };
        hidx.extend((0..n.saturating_sub(1)) .map(|i| i as i32));
        let rows = hidx.len();
        if rows > 0 {
            self.up_i32(&hidx, &m.hidx)?;
            self.stream.memcpy_htod(&[(start + from - 1) as i32], &mut self.buf.pos)?;
            self.seq = slot as i32;
            let ran = self.mtp_layer(&self.pre.tokens.slice(from..from + rows), &self.mtp().vx, &self.norm, rows, Attn::Contiguous);
            self.seq = 0;
            ran?;
        }
        let m = self.mtp();
        self.copy_hidden(&m.vx, &[n as i32 - 1], &m.pending, &[slot as i32])?;
        self.mtp_forget(slot);
        Ok(())
    }

    pub(super) fn mtp_after_decode(&mut self, rows: &[super::DecodeRow]) -> Result<()> {
        if self.mtp.is_none() {
            return Ok(());
        }
        let n = rows.len();
        let h = self.spec.dims.hidden;
        let m = self.mtp();
        self.copy_rows(&self.pre.x.slice(..n * h), &m.vx.slice(..n * h), n * h)?;
        let live: Vec<&super::DecodeRow> = rows.iter().filter(|r| r.pos >= 1).collect();
        let tokens: Vec<i32> = live.iter().map(|r| r.token as i32).collect();
        let hidx: Vec<i32> = live.iter().map(|r| -1 - r.slot as i32).collect();
        let pos: Vec<i32> = live.iter().map(|r| r.pos as i32 - 1).collect();
        let slots: Vec<i32> = live.iter().map(|r| r.slot as i32).collect();
        self.mtp_rows(&tokens, &hidx, &pos, &slots, &self.norm)?;
        let src: Vec<i32> = (0..n as i32).collect();
        let dst: Vec<i32> = rows.iter().map(|r| r.slot as i32).collect();
        let m = self.mtp();
        self.copy_hidden(&m.vx, &src, &m.pending, &dst)?;
        for r in rows {
            self.mtp_forget(r.slot);
        }
        Ok(())
    }

    fn prime(&mut self, rows: &[SpecRow]) -> Result<()> {
        let need: Vec<&SpecRow> = rows.iter().filter(|r| self.mtp().host_drafts[r.slot].is_empty()).collect();
        if need.is_empty() {
            return Ok(());
        }
        ensure!(need.iter().all(|r| r.pos >= 1), "a speculative row needs a computed token before it");
        let tokens: Vec<i32> = need.iter().map(|r| r.token as i32).collect();
        let hidx: Vec<i32> = need.iter().map(|r| -1 - r.slot as i32).collect();
        let pos: Vec<i32> = need.iter().map(|r| r.pos as i32 - 1).collect();
        let slots: Vec<i32> = need.iter().map(|r| r.slot as i32).collect();
        self.mtp_rows(&tokens, &hidx, &pos, &slots, &self.norm)?;
        let heads: Vec<i32> = (0..need.len() as i32).collect();
        let q0: Vec<usize> = need.iter().map(|r| r.pos - 1).collect();
        let drafts = self.mtp_chain(&heads, &q0, &slots)?;
        let m = self.mtp.as_mut().expect("an MTP spec");
        for (r, d) in need.iter().zip(drafts) {
            m.host_drafts[r.slot] = d;
        }
        Ok(())
    }

    fn verify_rows(&self, nseq: usize) -> Result<()> {
        let m = self.mtp();
        let s = &self.spec;
        let k1 = m.k + 1;
        let n = nseq * k1;
        let hidden = s.dims.hidden;
        let pre = &self.pre;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(&m.vb.tokens).arg(&pre.x);
        let config = LaunchConfig { grid_dim: (grid(hidden, 256), n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let seq_slots = s.linear_layers() as i32;
        let (vd, rows_i, k1_i) = (la.value_dim(), m.rows, k1 as i32);
        for layer in &self.layers {
            self.rms_split(&pre.x.as_view(), &layer.ln1, &pre.xn_planes, n)?;
            match &layer.mixer {
                DeviceMixer::Linear(g) => {
                    self.tc("lin_in", &g.in_proj, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_lin), la.in_proj_rows(), n)?;
                    self.tc("lin_ba", &g.ba, None, pre.xn_planes.all(), GemmOut::F32(&gb.pre_ba), la.ba_rows(), n)?;
                    let l = g.slot as usize;
                    let clog = m.clog.slice(l * rows_i * la.conv_dim()..);
                    let mut b = self.stream.launch_builder(self.kernel("gdn_conv_verify"));
                    b.arg(&gb.pre_lin)
                        .arg(&g.conv)
                        .arg(&*gb.conv_state)
                        .arg(&gb.pre_mix)
                        .arg(&clog)
                        .arg(&g.slot)
                        .arg(&seq_slots)
                        .arg(&m.vb.slots)
                        .arg(&k1_i);
                    let config = LaunchConfig {
                        grid_dim: (grid(la.conv_dim(), 256), nseq as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b.launch(config) }?;
                    let klog = m.klog.slice(l * rows_i * vd..);
                    let dlog = m.dlog.slice(l * rows_i * vd..);
                    let glog = m.glog.slice(l * rows_i * la.value_heads..);
                    let mut b = self.stream.launch_builder(self.kernel("gdn_verify"));
                    b.arg(&gb.pre_mix)
                        .arg(&gb.pre_lin)
                        .arg(&gb.pre_ba)
                        .arg(&g.neg_a)
                        .arg(&g.dt_bias)
                        .arg(&g.norm)
                        .arg(&*gb.ssm_state)
                        .arg(&gb.o)
                        .arg(&klog)
                        .arg(&dlog)
                        .arg(&glog)
                        .arg(&g.slot)
                        .arg(&seq_slots)
                        .arg(&m.vb.slots)
                        .arg(&k1_i);
                    let config = LaunchConfig {
                        grid_dim: (la.value_heads as u32, nseq as u32, 1),
                        block_dim: ((la.value_head_dim * LA_KSPLIT) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b.launch(config) }?;
                    let mut b = self.stream.launch_builder(self.kernel("split_rows"));
                    let count = (n * vd) as i32;
                    b.arg(&gb.o).arg(&gb.planes.hi).arg(&gb.planes.lo).arg(&count);
                    unsafe { b.launch(cfg(grid(n * vd, 512), 256)) }?;
                    self.tc("lin_out", &g.out, None, gb.planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                }
                DeviceMixer::Attention(l) => {
                    self.tc("qkv", &l.qkv, l.qkv_bias.as_ref(), pre.xn_planes.all(), GemmOut::F32(&pre.qkv), s.qkv_rows(), n)?;
                    self.attn_decode_rows(l, &m.vb, n)?;
                    self.tc("o", &l.o, None, pre.attn_planes.all(), GemmOut::F32(&pre.x), hidden, n)?;
                }
            }
            self.mlp_prefill(layer, n)?;
        }
        self.rms_split(&pre.x.as_view(), &self.norm, &pre.xn_planes, n)?;
        self.tc("lm_head", &self.lm_head, None, pre.xn_planes.all(), GemmOut::F32(&m.vb.logits), s.dims.vocab, n)
    }

    fn commit(&self, nseq: usize) -> Result<()> {
        let m = self.mtp();
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let layers = self.spec.linear_layers() as u32;
        let (seq_slots, k1, log_rows) = (layers as i32, (m.k + 1) as i32, m.rows as i32);
        let mut b = self.stream.launch_builder(self.kernel("gdn_conv_commit"));
        b.arg(&*gb.conv_state).arg(&m.clog).arg(&seq_slots).arg(&m.vb.slots).arg(&k1).arg(&m.accepted).arg(&log_rows);
        let config = LaunchConfig {
            grid_dim: (grid(la.conv_dim(), 256), nseq as u32, layers),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("gdn_commit"));
        b.arg(&*gb.ssm_state)
            .arg(&m.klog)
            .arg(&m.dlog)
            .arg(&m.glog)
            .arg(&seq_slots)
            .arg(&m.vb.slots)
            .arg(&k1)
            .arg(&m.accepted)
            .arg(&log_rows);
        let config = LaunchConfig {
            grid_dim: (la.value_heads as u32, nseq as u32, layers),
            block_dim: ((la.value_head_dim * LA_KSPLIT) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    pub fn speculate(&mut self, rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
        ensure!(self.mtp.is_some(), "speculation needs an [mtp] spec with speculate = true");
        ensure!(self.use_mma, "speculation needs the tensor core path");
        let (k, cap) = (self.mtp().k, self.mtp().cap);
        let n = rows.len();
        ensure!(n >= 1 && n <= cap, "{n} speculative rows for a cap of {cap}");
        ensure!(
            rows.iter().all(|r| r.slot < self.slots && r.pos + 2 * k < self.max_ctx),
            "speculative row out of range"
        );
        let mut seen = vec![false; self.slots];
        ensure!(rows.iter().all(|r| !std::mem::replace(&mut seen[r.slot], true)), "a slot twice in one step");
        self.prime(rows)?;
        let k1 = k + 1;
        let m = self.mtp();
        let mut vtok = Vec::with_capacity(n * k1);
        let mut pos = Vec::with_capacity(n * k1);
        let mut slots = Vec::with_capacity(n * k1);
        for r in rows {
            vtok.push(r.token as i32);
            vtok.extend(m.host_drafts[r.slot].iter().map(|&t| t as i32));
            pos.extend((0..k1).map(|j| (r.pos + j) as i32));
            slots.extend(std::iter::repeat_n(r.slot as i32, k1));
        }
        self.up_i32(&vtok, &m.vb.tokens)?;
        self.up_i32(&pos, &m.vb.pos)?;
        self.up_i32(&slots, &m.vb.slots)?;
        let p: Vec<i32> = rows.iter().flat_map(|r| super::step::params(r.sampling, r.step)).collect();
        self.up_i32(&p, &m.params)?;
        let caps: Vec<i32> = rows.iter().map(|r| r.max_accept.min(k) as i32).collect();
        self.up_i32(&caps, &m.max_accept)?;
        self.verify_rows(n)?;
        let h = self.spec.dims.hidden;
        self.copy_rows(&self.pre.x.slice(..n * k1 * h), &m.vx.slice(..n * k1 * h), n * k1 * h)?;
        let k_i = k as i32;
        let mut b = self.stream.launch_builder(self.kernel("spec_accept"));
        b.arg(&m.vb.logits).arg(&m.vb.tokens).arg(&m.params).arg(&m.max_accept).arg(&k_i).arg(&m.vout).arg(&m.accepted);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        self.commit(n)?;
        let out = self.stream.clone_dtoh(&m.vout.slice(..n * k1))?;
        let accepted = self.stream.clone_dtoh(&m.accepted.slice(..n))?;
        let mut tokens = Vec::new();
        let mut hidx = Vec::new();
        let mut mpos = Vec::new();
        let mut mslots = Vec::new();
        let mut heads = Vec::new();
        let mut q0 = Vec::new();
        let mut emitted = Vec::with_capacity(n);
        for (s, r) in rows.iter().enumerate() {
            let a = accepted[s] as usize;
            let row = &out[s * k1..s * k1 + a + 1];
            emitted.push(row.iter().map(|&t| t as u32).collect::<Vec<u32>>());
            for (j, &t) in row.iter().enumerate() {
                tokens.push(t);
                hidx.push((s * k1 + j) as i32);
                mpos.push((r.pos + j) as i32);
                mslots.push(r.slot as i32);
            }
            heads.push(tokens.len() as i32 - 1);
            q0.push(r.pos + a);
        }
        self.mtp_rows(&tokens, &hidx, &mpos, &mslots, &self.norm)?;
        let chain_slots: Vec<i32> = rows.iter().map(|r| r.slot as i32).collect();
        let drafts = self.mtp_chain(&heads, &q0, &chain_slots)?;
        let src: Vec<i32> = accepted.iter().enumerate().map(|(s, &a)| (s * k1) as i32 + a).collect();
        let m = self.mtp();
        self.copy_hidden(&m.vx, &src, &m.pending, &chain_slots)?;
        let m = self.mtp.as_mut().expect("an MTP spec");
        for (r, d) in rows.iter().zip(drafts) {
            m.host_drafts[r.slot] = d;
        }
        m.stats.steps += 1;
        m.stats.rows += n as u64;
        m.stats.drafted += (n * k) as u64;
        m.stats.accepted += accepted.iter().map(|&a| a as u64).sum::<u64>();
        for &a in &accepted {
            for j in 0..(a as usize).min(4) {
                m.stats.position_accepted[j] += 1;
            }
        }
        Ok(emitted)
    }

    pub fn mtp_prompt_logits(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<u32>)> {
        let (logits, drafts, _) = self.mtp_prompt_trace(tokens)?;
        Ok((logits, drafts))
    }

    pub fn mtp_prompt_trace(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<u32>, Vec<f32>)> {
        ensure!(self.mtp.is_some(), "no MTP layer");
        ensure!(!tokens.is_empty() && tokens.len() + 2 * self.mtp().k < self.max_ctx, "prompt too long");
        self.reset_slot(0)?;
        self.prefill_chunk(tokens, 0, None)?;
        self.mtp_after_prefill(0, tokens, 0)?;
        let first = super::argmax_host(&self.stream.clone_dtoh(&self.buf.logits)?);
        let row = SpecRow { slot: 0, token: first, pos: tokens.len(), sampling: Sampling::GREEDY, step: 0, max_accept: 0 };
        self.prime(&[row])?;
        let vocab = self.spec.dims.vocab;
        let logits = self.stream.clone_dtoh(&self.mtp().logits.slice(..vocab))?;
        let n = tokens.len() * self.spec.dims.hidden;
        let hidden = self.stream.clone_dtoh(&self.mtp().vx.slice(..n))?;
        Ok((logits, self.mtp().host_drafts[0].clone(), hidden))
    }
}
