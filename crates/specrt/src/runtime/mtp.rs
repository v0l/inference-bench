use super::{
    BatchBuffers, DeviceAttn, DeviceLayer, DeviceMatrix, DeviceMixer, Planes, Runtime, Sampling, cfg, grid,
};
use crate::spec::Spec;
use crate::weights::{Checkpoint, Mixer};
use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, CudaView, LaunchConfig};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

pub(super) struct DeviceMtp {
    fc: DeviceMatrix,
    norm_embed: CudaSlice<f32>,
    norm_hidden: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    layer: DeviceLayer,
    head: Option<DeviceMatrix>,
}

impl DeviceMtp {
    pub(super) fn set_head(&mut self, head: Option<DeviceMatrix>) {
        self.head = head;
    }

    pub(super) fn load(stream: &Arc<CudaStream>, checkpoint: &Checkpoint, spec: &Spec, with_head: bool) -> Result<Option<DeviceMtp>> {
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
            ln_pre_mlp: None,
            ln_post_mlp: None,
            gated: Vec::new(),
            prelude: None,
            prelude_stored: Default::default(),
            ple: crate::runtime::PleWeights::new(),
            ple_at: -1,
            residual: Vec::new(),
            latent: None,
            global: spec.layers.count,
            mixer: DeviceMixer::Attention(DeviceAttn {
                qkv: DeviceMatrix::upload(stream, &a.qkv)?,
                qkv_bias: up(&a.qkv_bias)?,
                o_bias: up(&a.o_bias)?,
                q_norm: up(&a.q_norm)?,
                k_norm: up(&a.k_norm)?,
                o: DeviceMatrix::upload(stream, &a.o)?,
                kv: spec.attention_layers() as i32,
                kind_kv: spec.attention_layers() as i32,
                window: 0,
                nope: false,
                extra: a.extra.iter().map(|(n, v)| Ok((n.clone(), stream.clone_htod(v)?))).collect::<Result<_>>()?,
            }),
            gate_up: DeviceMatrix::upload(stream, &w.layer.gate_up)?,
            down: DeviceMatrix::upload(stream, &w.layer.down)?,
            moe: w.layer.moe.as_ref().map(super::moe::DeviceMoe::upload_with(stream)).transpose()?,
            hyper: Vec::new(),
        };
        let m = DeviceMtp {
            fc: DeviceMatrix::upload(stream, &w.fc)?,
            norm_embed: stream.clone_htod(&w.norm_embed)?,
            norm_hidden: stream.clone_htod(&w.norm_hidden)?,
            norm: stream.clone_htod(&w.norm)?,
            layer,
            head: match spec.draft_head_quant().filter(|_| with_head) {
                Some(q) => Some(DeviceMatrix::upload(stream, &checkpoint.draft_head(spec, q)?)?),
                None => None,
            },
        };
        stream.synchronize()?;
        Ok(Some(m))
    }

    pub(super) fn bytes(&self) -> u64 {
        self.fc.bytes()
            + self.head.as_ref().map_or(0, DeviceMatrix::bytes)
            + self.layer.bytes()
            + (self.norm_embed.num_bytes() + self.norm_hidden.num_bytes() + self.norm.num_bytes()) as u64
    }
}

pub(super) struct MtpState {
    w: DeviceMtp,
    k: usize,
    cap: usize,
    pcap: usize,
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
    iota: CudaSlice<i32>,
    zeros: CudaSlice<i32>,
    sidx: CudaSlice<i32>,
    didx: CudaSlice<i32>,
    rin: CudaSlice<i32>,
    res: CudaSlice<i32>,
    klog: CudaSlice<f32>,
    dlog: CudaSlice<f32>,
    glog: CudaSlice<f32>,
    clog: CudaSlice<f32>,
    drafted: Vec<usize>,
    graphs: HashMap<(usize, usize), CudaGraph>,
    capture_secs: Vec<(usize, f64)>,
    use_graphs: bool,
    syncs: Cell<u64>,
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

pub(super) struct MtpArgs<'a> {
    pub tokens: &'a CudaView<'a, i32>,
    pub hidx: &'a CudaView<'a, i32>,
    pub xsrc: &'a CudaSlice<f32>,
    pub pre: &'a CudaSlice<f32>,
    pub src: &'a CudaView<'a, i32>,
    pub col: usize,
    pub slots: &'a CudaView<'a, i32>,
}

enum Attn {
    Contiguous,
    Rows(usize),
}

impl MtpState {
    pub(super) fn new(rt: &Runtime, w: DeviceMtp) -> Result<MtpState> {
        let s = &rt.spec;
        let m = s.speculative().context("speculation is off")?;
        let k = m.draft_tokens;
        let cap = m.max_batch.unwrap_or(rt.slots).min(rt.slots);
        let k1 = k + 1;
        let padded = super::step::pad_rows(cap);
        let acap = if padded * k1 <= rt.chunk { padded } else { cap };
        let rows = acap * k1;
        ensure!(rows <= rt.chunk, "{rows} verify rows do not fit a step of {}", rt.chunk);
        let stream = &rt.stream;
        let h = s.dims.hidden;
        let vb_rows = rows.max(rt.slots) + 1;
        let wide = rt.chunk.max(vb_rows);
        let parts = vb_rows * s.attention.heads * crate::codegen::decode_splits(rt.max_ctx);
        let layers = s.linear_layers();
        let (vd, value_heads, conv_dim) = s.linear_attention.as_ref().map_or((0, 0, 0), |la| (la.value_heads * la.key_head_dim, la.value_heads, la.conv_dim()));
        let z = |n: usize| stream.alloc_zeros::<f32>(n.max(1));
        let zi = |n: usize| stream.alloc_zeros::<i32>(n.max(1));
        Ok(MtpState {
            w,
            k,
            cap,
            pcap: acap,
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
            cg: z(acap * h)?,
            logits: z(acap * s.dims.vocab)?,
            drafts: zi((rt.slots + 1) * k)?,
            ctok: zi(acap)?,
            hidx: zi(wide)?,
            iota: stream.clone_htod(&(0..wide as i32).collect::<Vec<i32>>())?,
            zeros: zi(acap)?,
            sidx: zi(vb_rows)?,
            didx: zi(vb_rows)?,
            rin: zi(10 * acap)?,
            res: zi(rows + acap)?,
            klog: z(layers * rows * vd)?,
            dlog: z(layers * rows * vd)?,
            glog: z(layers * rows * value_heads)?,
            clog: z(layers * rows * conv_dim)?,
            drafted: vec![0; rt.slots + 1],
            graphs: HashMap::new(),
            capture_secs: Vec::new(),
            use_graphs: s.runtime.graphs && m.graph,
            syncs: Cell::new(0),
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
        let Some(m) = self.mtp.as_ref().filter(|m| m.drafted[slot] > 0) else { return Vec::new() };
        let d = self.stream.clone_dtoh(&m.drafts.slice(slot * m.k..slot * m.k + m.drafted[slot])).expect("read drafts");
        d.into_iter().map(|t| t as u32).collect()
    }

    pub fn set_drafts(&mut self, slot: usize, drafts: &[u32]) -> Result<()> {
        let m = self.mtp.as_ref().context("no MTP layer")?;
        ensure!(drafts.len() == m.k, "{} drafts for k = {}", drafts.len(), m.k);
        let host: Vec<i32> = drafts.iter().map(|&t| t as i32).collect();
        self.up_i32(&host, &m.drafts.slice(slot * m.k..(slot + 1) * m.k))?;
        self.mtp.as_mut().expect("an MTP spec").drafted[slot] = drafts.len();
        Ok(())
    }

    pub fn set_speculative_graphs(&mut self, on: bool) {
        if let Some(m) = self.mtp.as_mut() {
            m.use_graphs = on;
        }
    }

    pub fn speculation_capture_secs(&self) -> Vec<(usize, f64)> {
        self.mtp.as_ref().map(|m| m.capture_secs.clone()).unwrap_or_default()
    }

    pub fn captured_speculation_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.mtp.as_ref().map(|m| m.graphs.keys().map(|k| k.0).collect()).unwrap_or_default();
        sizes.dedup();
        sizes.sort();
        sizes
    }

    pub fn stream_syncs(&self) -> u64 {
        self.mtp.as_ref().map_or(0, |m| m.syncs.get())
    }

    pub(super) fn mtp_forget(&mut self, slot: usize) {
        if let Some(m) = self.mtp.as_mut() {
            m.drafted[slot] = 0;
        }
    }

    pub(super) fn mtp_pending_host(&self, slot: usize) -> Result<Vec<f32>> {
        let h = self.spec.dims.hidden;
        Ok(self.stream.clone_dtoh(&self.mtp().pending.slice(slot * h..(slot + 1) * h))?)
    }

    pub(super) fn mtp_pending_to(&self, slot: usize, dst: &mut cudarc::driver::CudaViewMut<f32>) -> Result<()> {
        let h = self.spec.dims.hidden;
        self.stream.memcpy_dtod(&self.mtp().pending.slice(slot * h..(slot + 1) * h), dst)?;
        Ok(())
    }

    pub(super) fn mtp_pending_from(&mut self, slot: usize, host: &[f32]) -> Result<()> {
        let h = self.spec.dims.hidden;
        ensure!(host.len() == h, "pending hidden of {} values for a hidden size of {h}", host.len());
        let stream = self.stream.clone();
        let m = self.mtp.as_mut().context("no MTP layer")?;
        stream.memcpy_htod(host, &mut m.pending.slice_mut(slot * h..(slot + 1) * h))?;
        m.drafted[slot] = 0;
        Ok(())
    }

    fn up_i32(&self, host: &[i32], dst: &impl cudarc::driver::DevicePtr<i32>) -> Result<()> {
        ensure!(host.len() <= dst.len(), "{} values for a device buffer of {}", host.len(), dst.len());
        if host.is_empty() {
            return Ok(());
        }
        let (ptr, _sync) = dst.device_ptr(&self.stream);
        unsafe { cudarc::driver::result::memcpy_htod_async(ptr, host, self.stream.cu_stream()) }?;
        Ok(())
    }

    fn copy_hidden(&self, src: &CudaSlice<f32>, sidx: &[i32], dst: &CudaSlice<f32>, didx: &[i32]) -> Result<()> {
        if sidx.is_empty() {
            return Ok(());
        }
        let m = self.mtp();
        self.up_i32(sidx, &m.sidx)?;
        self.up_i32(didx, &m.didx)?;
        let mut b = self.launcher(self.kernel("mtp_copy"));
        b.arg(src).arg(&m.sidx).arg(dst).arg(&m.didx);
        let config = LaunchConfig {
            grid_dim: (grid(self.spec.dims.hidden, 256), sidx.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn mtp_run<'a>(&'a self, n: usize, qlen: usize, args: MtpArgs<'a>, ops: &[crate::plan::Op], contiguous: bool) -> Result<()> {
        let m = self.mtp();
        let bb = if contiguous { None } else { Some(&m.vb) };
        let run = super::exec::Run { n, tokens: &m.vb.tokens, logits: &m.logits, rows: 0..n, bb, start: None, qlen, mtp: Some(args) };
        for op in ops {
            self.run_mtp_node(op, &run)?;
        }
        Ok(())
    }

    fn mtp_layer(&self, tokens: &CudaView<i32>, hidx: &CudaView<i32>, xsrc: &CudaSlice<f32>, pre: &CudaSlice<f32>, n: usize, attn: Attn) -> Result<()> {
        let iota = self.mtp().iota.as_view();
        let args = MtpArgs { tokens, hidx, xsrc, pre, src: &iota, col: 0, slots: &iota };
        match attn {
            Attn::Contiguous => self.mtp_run(n, 1, args, &self.plans.prefill.mtp, true),
            Attn::Rows(qlen) => self.mtp_run(n, qlen, args, &self.plans.decode.mtp, false),
        }
    }

    fn mtp_heads(&self, src: &CudaView<i32>, n: usize, col: usize, slots: &CudaView<i32>) -> Result<()> {
        let m = self.mtp();
        let iota = m.iota.as_view();
        let args = MtpArgs { tokens: &iota, hidx: &iota, xsrc: &m.vx, pre: &self.norm, src, col, slots };
        self.mtp_run(n, 1, args, &self.plans.decode.mtp_head, false)
    }

    pub(super) fn mtp_logits(&self) -> &CudaSlice<f32> {
        &self.mtp().logits
    }

    pub(super) fn mtp_fc_planes(&self) -> &Planes {
        &self.mtp().fc_planes
    }

    pub(super) fn mtp_weight(&self, w: crate::plan::Weight) -> &DeviceMatrix {
        let m = self.mtp();
        match w {
            crate::plan::Weight::MtpFc => &m.w.fc,
            _ => m.w.head.as_ref().unwrap_or(&self.lm_head),
        }
    }

    pub(super) fn mtp_head_kernel(&self) -> &'static str {
        if self.mtp().w.head.is_some() { "mtp_head" } else { "lm_head" }
    }

    pub(super) fn run_mtp_node(&self, op: &crate::plan::Op, run: &super::exec::Run) -> Result<()> {
        self.run_one(op, Some(&self.mtp().w.layer), run)
    }

    pub(super) fn mtp_op(&self, op: &crate::plan::Op, run: &super::exec::Run) -> Result<()> {
        use crate::plan::Op;
        let m = self.mtp();
        let a = run.mtp.as_ref().expect("mtp ops carry their arguments");
        let n = run.n;
        match op {
            Op::MtpIn => {
                let mut b = self.launcher(self.kernel("mtp_fc_in"));
                b.arg(&self.embed)
                    .arg(a.tokens)
                    .arg(a.xsrc)
                    .arg(&m.pending)
                    .arg(a.hidx)
                    .arg(a.pre)
                    .arg(&m.w.norm_embed)
                    .arg(&m.w.norm_hidden)
                    .arg(&m.fc_planes.hi)
                    .arg(&m.fc_planes.lo);
                unsafe { b.launch(cfg(n as u32, 256)) }?;
                Ok(())
            }
            Op::MtpGather => self.copy_rows_by_index(&self.pre.x, a.src, &m.cg, &m.iota.as_view(), n),
            Op::MtpNorm => self.rms_split(&m.cg.as_view(), &m.w.norm, &self.pre.xn_planes, n),
            Op::MtpDraft => {
                let (col, k) = (a.col as i32, m.k as i32);
                let mut b = self.launcher(self.kernel("mtp_draft"));
                b.arg(&m.logits).arg(&m.drafts).arg(&col).arg(&k).arg(&m.ctok).arg(a.slots);
                unsafe { b.launch(cfg(n as u32, 1024)) }?;
                Ok(())
            }
            _ => unreachable!("not an MTP op"),
        }
    }

    pub(super) fn gdn_verify_op(&self, g: &super::DeviceGdn, run: &super::exec::Run) -> Result<()> {
        let m = self.mtp();
        let n = run.n;
        let k1 = run.qlen;
        let nseq = n / k1;
        let gb = self.gdn.as_ref().expect("linear layers have gdn buffers");
        let la = &gb.la;
        let seq_slots = self.spec.linear_layers() as i32;
        let (vd, rows_i, k1_i) = (la.value_dim(), m.rows, k1 as i32);
        let l = g.slot as usize;
        let clog = m.clog.slice(l * rows_i * la.conv_dim()..);
        let mut b = self.launcher(self.kernel("gdn_conv_verify"));
        b.arg(&gb.pre_lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.pre_mix).arg(&clog).arg(&g.slot).arg(&seq_slots).arg(&m.vb.slots).arg(&k1_i);
        let config = LaunchConfig { grid_dim: (grid(la.conv_dim(), 256), nseq as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        let klog = m.klog.slice(l * rows_i * vd..);
        let dlog = m.dlog.slice(l * rows_i * vd..);
        let glog = m.glog.slice(l * rows_i * la.value_heads..);
        let mut b = self.launcher(self.kernel("gdn_verify"));
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
            block_dim: ((la.value_head_dim * self.linear_tuning.ksplit) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.launcher(self.kernel("split_rows"));
        let count = (n * vd) as i32;
        b.arg(&gb.o).arg(&gb.planes.hi).arg(&gb.planes.lo).arg(&count);
        unsafe { b.launch(cfg(grid(n * vd, 512), 256)) }?;
        Ok(())
    }

    fn copy_rows_by_index(&self, src: &CudaSlice<f32>, sidx: &CudaView<i32>, dst: &CudaSlice<f32>, didx: &CudaView<i32>, n: usize) -> Result<()> {
        let mut b = self.launcher(self.kernel("mtp_copy"));
        b.arg(src).arg(sidx).arg(dst).arg(didx);
        let config = LaunchConfig {
            grid_dim: (grid(self.spec.dims.hidden, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
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
        self.mtp_layer(&m.vb.tokens.as_view(), &m.hidx.as_view(), &m.vx, pre, tokens.len(), Attn::Rows(1))
    }

    fn mtp_chain(&self, n: usize, first: &CudaView<i32>, rin: &CudaView<i32>, accepted: &CudaView<i32>) -> Result<()> {
        let m = self.mtp();
        let slots = rin.slice(2 * n..3 * n);
        self.mtp_heads(first, n, 0, &slots)?;
        let n_i = n as i32;
        for i in 1..m.k {
            let step = i as i32;
            let mut b = self.launcher(self.kernel("mtp_chain_rows"));
            b.arg(rin).arg(&n_i).arg(accepted).arg(&step).arg(&m.vb.pos).arg(&m.vb.slots);
            unsafe { b.launch(cfg(1, n as u32)) }?;
            self.mtp_layer(&m.ctok.slice(..n), &m.iota.slice(..n), &m.cg, &m.w.norm, n, Attn::Rows(1))?;
            self.mtp_heads(&m.iota.slice(..n), n, i, &slots)?;
        }
        Ok(())
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
            let ran = self.mtp_layer(&self.pre.tokens.slice(from..from + rows), &self.mtp().hidx.as_view(), &self.mtp().vx, &self.norm, rows, Attn::Contiguous);
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
        let need: Vec<&SpecRow> = rows.iter().filter(|r| self.mtp().drafted[r.slot] == 0).collect();
        if need.is_empty() {
            return Ok(());
        }
        ensure!(need.iter().all(|r| r.pos >= 1), "a speculative row needs a computed token before it");
        let n = need.len();
        let tokens: Vec<i32> = need.iter().map(|r| r.token as i32).collect();
        let hidx: Vec<i32> = need.iter().map(|r| -1 - r.slot as i32).collect();
        let pos: Vec<i32> = need.iter().map(|r| r.pos as i32 - 1).collect();
        let slots: Vec<i32> = need.iter().map(|r| r.slot as i32).collect();
        self.mtp_rows(&tokens, &hidx, &pos, &slots, &self.norm)?;
        let m = self.mtp();
        let rin: Vec<i32> = tokens.iter().map(|_| 0).chain(pos.iter().copied()).chain(slots.iter().copied()).collect();
        self.up_i32(&rin, &m.rin)?;
        self.mtp_chain(n, &m.iota.slice(..n), &m.rin.slice(..), &m.zeros.slice(..n))?;
        let m = self.mtp.as_mut().expect("an MTP spec");
        for r in need {
            m.drafted[r.slot] = m.k;
        }
        Ok(())
    }

    fn verify_rows(&self, nseq: usize, k: usize) -> Result<()> {
        let m = self.mtp();
        let n = nseq * (k + 1);
        let run = super::exec::Run { n, tokens: &m.vb.tokens, logits: &m.vb.logits, rows: 0..n, bb: Some(&m.vb), start: None, qlen: k + 1, mtp: None };
        crate::executor::run_plan(self, &self.plans.verify, &run, &mut |_| Ok(()))
    }

    fn commit(&self, nseq: usize, k: usize) -> Result<()> {
        let Some(gb) = self.gdn.as_ref() else { return Ok(()) };
        let m = self.mtp();
        let accepted = m.res.slice(..nseq);
        let la = &gb.la;
        let layers = self.spec.linear_layers() as u32;
        let (seq_slots, k1, log_rows) = (layers as i32, (k + 1) as i32, m.rows as i32);
        let mut b = self.launcher(self.kernel("gdn_conv_commit"));
        b.arg(&*gb.conv_state).arg(&m.clog).arg(&seq_slots).arg(&m.vb.slots).arg(&k1).arg(&accepted).arg(&log_rows);
        let config = LaunchConfig {
            grid_dim: (grid(la.conv_dim(), 256), nseq as u32, layers),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.launcher(self.kernel("gdn_commit"));
        b.arg(&*gb.ssm_state)
            .arg(&m.klog)
            .arg(&m.dlog)
            .arg(&m.glog)
            .arg(&seq_slots)
            .arg(&m.vb.slots)
            .arg(&k1)
            .arg(&accepted)
            .arg(&log_rows);
        let config = LaunchConfig {
            grid_dim: (la.value_heads as u32, nseq as u32, layers),
            block_dim: ((la.value_head_dim * self.linear_tuning.ksplit) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn round_body(&self, n: usize, k: usize) -> Result<()> {
        let m = self.mtp();
        let k1 = k + 1;
        let h = self.spec.dims.hidden;
        let (n_i, k_i, stride_i) = (n as i32, k as i32, m.k as i32);
        let rin = m.rin.slice(..);
        let accepted = m.res.slice(..n);
        let vout = m.res.slice(m.pcap..m.pcap + n * k1);
        let mut b = self.launcher(self.kernel("mtp_round_in"));
        b.arg(&rin).arg(&m.drafts).arg(&n_i).arg(&k_i).arg(&stride_i).arg(&m.vb.tokens).arg(&m.vb.pos).arg(&m.vb.slots);
        unsafe { b.launch(cfg(n as u32, k1 as u32)) }?;
        self.verify_rows(n, k)?;
        self.copy_rows(&self.pre.x.slice(..n * k1 * h), &m.vx.slice(..n * k1 * h), n * k1 * h)?;
        let maxa = rin.slice(3 * n..4 * n);
        let params = rin.slice(4 * n..10 * n);
        let mut b = self.launcher(self.kernel("spec_accept"));
        b.arg(&m.vb.logits).arg(&m.vb.tokens).arg(&params).arg(&maxa).arg(&k_i).arg(&vout).arg(&accepted);
        unsafe { b.launch(cfg(n as u32, 1024)) }?;
        self.commit(n, k)?;
        let mut b = self.launcher(self.kernel("mtp_pick_rows"));
        b.arg(&accepted).arg(&k_i).arg(&m.sidx);
        unsafe { b.launch(cfg(1, n as u32)) }?;
        self.mtp_layer(&vout, &m.iota.slice(..n * k1), &m.vx, &self.norm, n * k1, Attn::Rows(k1))?;
        let heads = m.sidx.slice(..n);
        self.mtp_chain(n, &heads, &rin, &accepted)?;
        self.copy_rows_by_index(&m.vx, &heads, &m.pending, &rin.slice(2 * n..3 * n), n)
    }

    fn capture_round(&self, n: usize, k: usize) -> Result<CudaGraph> {
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.round_body(n, k);
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0))?;
        recorded?;
        graph.context("empty speculative graph")
    }

    pub fn speculate(&mut self, rows: &[SpecRow]) -> Result<Vec<Vec<u32>>> {
        let k = self.mtp.as_ref().map_or(0, |m| m.k);
        self.speculate_k(rows, k)
    }

    pub fn speculate_k(&mut self, rows: &[SpecRow], k: usize) -> Result<Vec<Vec<u32>>> {
        ensure!(self.mtp.is_some(), "speculation needs an [mtp] spec with speculate = true");
        ensure!(self.use_mma, "speculation needs the tensor core path");
        let (kmax, cap, pcap) = (self.mtp().k, self.mtp().cap, self.mtp().pcap);
        ensure!((1..=kmax).contains(&k), "{k} draft tokens for a maximum of {kmax}");
        let live = rows.len();
        ensure!(live >= 1 && live <= cap, "{live} speculative rows for a cap of {cap}");
        ensure!(
            rows.iter().all(|r| r.slot < self.slots && r.pos + 2 * k < self.max_ctx),
            "speculative row out of range"
        );
        let mut seen = vec![false; self.slots];
        ensure!(rows.iter().all(|r| !std::mem::replace(&mut seen[r.slot], true)), "a slot twice in one step");
        self.prime(rows)?;
        let k1 = k + 1;
        let graphs = self.mtp().use_graphs && super::step::pad_rows(live) <= pcap;
        let n = if graphs { super::step::pad_rows(live) } else { live };
        let pad = SpecRow { slot: self.slots, token: 0, pos: 0, sampling: Sampling::GREEDY, step: 0, max_accept: 0 };
        let padded: Vec<SpecRow> = rows.iter().copied().chain(std::iter::repeat(pad)).take(n).collect();
        let mut rin: Vec<i32> = Vec::with_capacity(10 * n);
        rin.extend(padded.iter().map(|r| r.token as i32));
        rin.extend(padded.iter().map(|r| r.pos as i32));
        rin.extend(padded.iter().map(|r| r.slot as i32));
        rin.extend(padded.iter().map(|r| r.max_accept.min(k) as i32));
        rin.extend(padded.iter().flat_map(|r| super::step::params(r.sampling, r.step)));
        let m = self.mtp();
        self.up_i32(&rin, &m.rin)?;
        if graphs {
            if !m.graphs.contains_key(&(n, k)) {
                let started = std::time::Instant::now();
                let graph = self.capture_round(n, k)?;
                graph.upload()?;
                self.stream.synchronize()?;
                let secs = started.elapsed().as_secs_f64();
                tracing::info!(rows = n, k, capture_secs = secs, "speculation graph captured");
                let m = self.mtp.as_mut().expect("an MTP spec");
                m.capture_secs.push((n, secs));
                m.graphs.insert((n, k), graph);
            }
            self.mtp().graphs[&(n, k)].launch()?;
        } else {
            self.round_body(n, k)?;
        }
        let m = self.mtp();
        m.syncs.set(m.syncs.get() + 1);
        let res = self.stream.clone_dtoh(&m.res.slice(..m.pcap + n * k1))?;
        let accepted = &res[..live];
        let out = &res[m.pcap..];
        let emitted: Vec<Vec<u32>> = (0..live)
            .map(|s| out[s * k1..s * k1 + accepted[s] as usize + 1].iter().map(|&t| t as u32).collect())
            .collect();
        let m = self.mtp.as_mut().expect("an MTP spec");
        for r in rows {
            m.drafted[r.slot] = m.k;
        }
        m.stats.steps += 1;
        m.stats.rows += live as u64;
        m.stats.drafted += (live * k) as u64;
        m.stats.accepted += accepted.iter().map(|&a| a as u64).sum::<u64>();
        for &a in accepted {
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
        Ok((logits, self.drafts(0), hidden))
    }
}
