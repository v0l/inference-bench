use super::{BatchBuffers, DeviceAttn, DeviceLayer, DeviceMatrix, DeviceMixer, GemmOut, Planes, Runtime, grid};
use crate::executor::PlanBackend;
use crate::plan::{Buf, NormKind, Op, Phase, Plan, Weight};
use crate::spec::{Block, Spec};
use anyhow::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::ops::Range;

pub(super) struct Plans {
    pub prefill: Plan,
    pub decode: Plan,
    pub verify: Plan,
}

impl Plans {
    pub fn build(spec: &Spec, tensor_core: bool) -> Self {
        Plans {
            prefill: Plan::build(spec, Phase::Prefill, tensor_core),
            decode: Plan::build(spec, Phase::Decode, tensor_core),
            verify: Plan::build(spec, Phase::Verify, tensor_core),
        }
    }
}

impl Plans {
    pub fn for_tp(self, shards: &std::collections::BTreeMap<String, crate::tp::Shard>, fuse: bool) -> Self {
        Plans { prefill: self.prefill.for_tp(shards, fuse), decode: self.decode.for_tp(shards, fuse), verify: self.verify.for_tp(shards, fuse) }
    }

    pub fn for_stage(mut self, role: super::StageRole) -> Self {
        for plan in [&mut self.prefill, &mut self.decode, &mut self.verify] {
            if !role.first {
                plan.head.clear();
            }
            if !role.last {
                plan.tail.clear();
            }
        }
        self
    }
}

pub(super) struct Run<'a> {
    pub n: usize,
    pub tokens: &'a CudaSlice<i32>,
    pub logits: &'a CudaSlice<f32>,
    pub rows: Range<usize>,
    pub bb: Option<&'a BatchBuffers>,
    pub start: Option<usize>,
    pub qlen: usize,
    pub mtp: Option<super::mtp::MtpArgs<'a>>,
}

impl PlanBackend<Run<'_>> for Runtime {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn block(&self, layer: usize) -> Block {
        self.spec.block(layer)
    }

    fn sparse(&self, layer: usize) -> bool {
        self.layers[layer].moe.is_some()
    }

    fn run_op(&self, op: &Op, layer: Option<usize>, run: &Run<'_>) -> Result<()> {
        self.run_one(op, layer.map(|i| &self.layers[i]), run)
    }
}

impl Runtime {
    pub(super) fn run_one(&self, op: &Op, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let n = run.n;
        match *op {
            Op::Embed => self.embed_rows(run),
            Op::Splice => match run.start {
                Some(start) => self.splice_media(self.seq as usize, start, run.n, 0),
                None => Ok(()),
            },
            Op::Norm { which, into } => self.norm_op(which, into, layer, run),
            Op::NormAdd { which, from } => self.norm_add_op(which, from, layer, run),
            Op::Clear { buf } => self.clear_op(buf, run),
            Op::LogitSoftcap => self.softcap_op(run, "logit_softcap"),
            Op::LogitScale => self.softcap_op(run, "logit_scale"),
            Op::Split { from, into } => self.split_op(from, into, run),
            Op::QkNorm => self.qk_norm_op(self.attn_of(layer), run.n),
            Op::ScaledAdd { from } => self.scaled_add_op(from, run),
            Op::Linear { weight, input, output, .. } => self.linear_op(weight, input, output, layer, run),
            Op::KvRope => self.attn_rope_rows(self.attn_of(layer), n, 0),
            Op::QkRope => self.qk_rope_legacy(self.attn_of(layer), n),
            Op::Flash => self.attn_flash_rows(self.attn_of(layer), n, 0),
            Op::FlashLegacy => self.flash_legacy(self.attn_of(layer), n),
            Op::DecodeAttention => self.attn_decode_rows(self.attn_of(layer), run.bb.expect("decode runs carry batch buffers"), n, run.qlen),
            Op::Swiglu => self.swiglu_legacy(n),
            Op::GdnPrefill => self.gdn_prefill_rows(self.gdn_of(layer), n, 0),
            Op::GdnDecode => self.gdn_decode_rows(self.gdn_of(layer), &run.bb.expect("decode runs carry batch buffers").slots, n),
            Op::GdnVerify => self.gdn_verify_op(self.gdn_of(layer), run),
            Op::MtpIn | Op::MtpGather | Op::MtpNorm | Op::MtpDraft => self.mtp_op(op, run),
            Op::MoeRoute => self.moe_route(layer.expect("moe runs inside a layer"), &self.pre.x, n),
            Op::MoeExperts => self.moe_experts(layer.expect("moe runs inside a layer"), n),
            Op::MoeCombine => self.moe_combine(&self.pre.x, n),
            Op::AllReduce => self.allreduce_op(n),
            Op::AllReduceNorm { which } => self.allreduce_norm_op(which, layer, n),
        }
    }

    fn comm(&self) -> &crate::allreduce::Comm {
        self.comm.as_ref().expect("allreduce ops run on a tensor parallel rank")
    }

    fn allreduce_op(&self, rows: usize) -> Result<()> {
        let n = rows * self.spec.dims.hidden;
        self.comm().add_into(&self.stream, &self.partial.slice(..n), &self.pre.x.slice(..n), n)
    }

    fn allreduce_norm_op(&self, which: NormKind, layer: Option<&DeviceLayer>, rows: usize) -> Result<()> {
        let n = rows * self.spec.dims.hidden;
        let weight = self.norm_weight(which, layer);
        let planes = &self.pre.xn_planes;
        self.comm().allreduce_add_norm(&self.stream, &self.partial.slice(..n), &self.pre.x.slice(..n), weight, &planes.hi.slice(..n), &planes.lo.slice(..n), rows)
    }

    fn attn_of<'a>(&self, layer: Option<&'a DeviceLayer>) -> &'a DeviceAttn {
        match &layer.expect("attention ops run inside a layer").mixer {
            DeviceMixer::Attention(a) => a,
            DeviceMixer::Linear(_) => unreachable!("attention op on a linear layer"),
        }
    }

    fn gdn_of<'a>(&self, layer: Option<&'a DeviceLayer>) -> &'a super::DeviceGdn {
        match &layer.expect("linear ops run inside a layer").mixer {
            DeviceMixer::Linear(g) => g,
            DeviceMixer::Attention(_) => unreachable!("linear op on an attention layer"),
        }
    }

    fn embed_rows(&self, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let mut b = self.stream.launch_builder(self.kernel("embed"));
        b.arg(&self.embed).arg(run.tokens).arg(&self.pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), run.n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn norm_weight<'a>(&'a self, which: NormKind, layer: Option<&'a DeviceLayer>) -> &'a CudaSlice<f32> {
        let l = || layer.expect("layer norm runs inside a layer");
        match which {
            NormKind::Input => &l().ln1,
            NormKind::PostAttention => &l().ln2,
            NormKind::PreMlp => l().ln_pre_mlp.as_ref().expect("a sandwich layer has a pre mlp norm"),
            NormKind::PostMlp => l().ln_post_mlp.as_ref().expect("a sandwich layer has a post mlp norm"),
            NormKind::Final => &self.norm,
        }
    }

    fn norm_op(&self, which: NormKind, into: Buf, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let pre = &self.pre;
        let hidden = self.spec.dims.hidden;
        let weight = self.norm_weight(which, layer);
        match into {
            Buf::XnPlanes if which == NormKind::Final => {
                let x = pre.x.slice(run.rows.start * hidden..run.rows.end * hidden);
                self.rms_split(&x, weight, &pre.xn_planes, run.rows.len())
            }
            Buf::XnPlanes => self.rms_split(&pre.x.as_view(), weight, &pre.xn_planes, run.n),
            Buf::Xn => self.rmsnorm_rows(&pre.x, weight, &pre.xn, run.n),
            other => unreachable!("norm into {}", other.label()),
        }
    }

    fn norm_add_op(&self, which: NormKind, from: Buf, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let weight = self.norm_weight(which, layer);
        let mut b = self.stream.launch_builder(self.kernel("norm_add"));
        let delta = self.f32_buf(from, run);
        b.arg(delta).arg(weight).arg(&self.pre.x).arg(&self.pre.x);
        unsafe { b.launch(super::cfg(run.n as u32, 1024)) }?;
        Ok(())
    }

    fn clear_op(&self, buf: Buf, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let target = self.f32_buf(buf, run);
        let mut b = self.stream.launch_builder(self.kernel("clear_branch"));
        b.arg(target).arg(target);
        let config = LaunchConfig { grid_dim: (grid(hidden, 256), run.n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn split_op(&self, from: Buf, into: Buf, run: &Run) -> Result<()> {
        let planes = self.planes(into);
        let source = self.f32_buf(from, run);
        let count = (run.n * self.spec.dims.hidden) as i32;
        let mut b = self.stream.launch_builder(self.kernel("split_rows"));
        b.arg(source).arg(&planes.hi).arg(&planes.lo).arg(&count);
        unsafe { b.launch(super::cfg(super::grid(count as usize, 512), 256)) }?;
        Ok(())
    }

    fn qk_norm_op(&self, l: &DeviceAttn, n: usize) -> Result<()> {
        let (q, k) = (l.q_norm.as_ref().expect("full-width qk norm has weights"), l.k_norm.as_ref().expect("full-width qk norm has weights"));
        for (name, w) in [("q_norm_rows", q), ("k_norm_rows", k)] {
            let mut b = self.stream.launch_builder(self.kernel(name));
            b.arg(&self.pre.qkv).arg(w).arg(&self.pre.qkv);
            unsafe { b.launch(super::cfg(n as u32, 1024)) }?;
        }
        Ok(())
    }

    fn scaled_add_op(&self, from: Buf, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let delta = self.f32_buf(from, run);
        let mut b = self.stream.launch_builder(self.kernel("residual_add"));
        b.arg(&self.pre.x).arg(delta).arg(&self.pre.x);
        let config = LaunchConfig { grid_dim: (grid(hidden, 256), run.n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn softcap_op(&self, run: &Run, name: &str) -> Result<()> {
        let vocab = self.spec.dims.vocab;
        let mut b = self.stream.launch_builder(self.kernel(name));
        b.arg(run.logits).arg(run.logits);
        let config = LaunchConfig { grid_dim: (grid(vocab, 256), run.rows.len() as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn planes(&self, b: Buf) -> &Planes {
        match b {
            Buf::XnPlanes => &self.pre.xn_planes,
            Buf::AttnPlanes => &self.pre.attn_planes,
            Buf::ActPlanes => &self.pre.act_planes,
            Buf::GdnPlanes => &self.gdn.as_ref().expect("linear layers have gdn buffers").planes,
            Buf::FcPlanes => self.mtp_fc_planes(),
            other => unreachable!("{} is not a plane pair", other.label()),
        }
    }

    fn f32_buf<'a>(&'a self, b: Buf, run: &Run<'a>) -> &'a CudaSlice<f32> {
        let gdn = || self.gdn.as_ref().expect("linear layers have gdn buffers");
        match b {
            Buf::X => &self.pre.x,
            Buf::Xn => &self.pre.xn,
            Buf::Branch => &self.pre.branch,
            Buf::Qkv => &self.pre.qkv,
            Buf::Attn => &self.pre.attn,
            Buf::GateUp => &self.pre.gate_up,
            Buf::Act => &self.pre.act,
            Buf::PreLin => &gdn().pre_lin,
            Buf::PreBa => &gdn().pre_ba,
            Buf::Logits => run.logits,
            Buf::MtpLogits => self.mtp_logits(),
            Buf::Partial => &self.partial,
            other => unreachable!("{} is not an f32 buffer", other.label()),
        }
    }

    fn is_planes(b: Buf) -> bool {
        matches!(b, Buf::XnPlanes | Buf::AttnPlanes | Buf::ActPlanes | Buf::GdnPlanes | Buf::FcPlanes)
    }

    fn weight_of<'a>(&'a self, weight: Weight, layer: Option<&'a DeviceLayer>) -> (&'a DeviceMatrix, Option<&'a CudaSlice<f32>>) {
        let attn = || self.attn_of(layer);
        let gdn = || self.gdn_of(layer);
        let layer_of = || layer.expect("layer weights need a layer");
        match weight {
            Weight::Qkv => (&attn().qkv, attn().qkv_bias.as_ref()),
            Weight::O => (&attn().o, None),
            Weight::GateUp => (&layer_of().gate_up, None),
            Weight::Down => (&layer_of().down, None),
            Weight::LinIn => (&gdn().in_proj, None),
            Weight::LinBa => (&gdn().ba, None),
            Weight::LinOut => (&gdn().out, None),
            Weight::LmHead => (&self.lm_head, None),
            Weight::MtpQkv => (&attn().qkv, None),
            Weight::MtpO => (&attn().o, None),
            Weight::MtpGateUp => (&layer_of().gate_up, None),
            Weight::MtpDown => (&layer_of().down, None),
            Weight::MtpFc | Weight::MtpHead => (self.mtp_weight(weight), None),
        }
    }

    fn linear_op(&self, weight: Weight, input: Buf, output: Buf, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let (w, bias) = self.weight_of(weight, layer);
        let rows = weight.rows(&self.spec);
        let hidden = self.spec.dims.hidden;
        let head = weight == Weight::LmHead;
        let tokens = if head { run.rows.len() } else { run.n };
        if Self::is_planes(input) {
            let planes = self.planes(input);
            let x = if head { planes.rows(0, tokens, hidden) } else { planes.all() };
            let out = match output {
                Buf::ActPlanes => GemmOut::Planes(&self.pre.act_planes),
                other => GemmOut::F32(self.f32_buf(other, run)),
            };
            let name = if weight == Weight::MtpHead { self.mtp_head_kernel() } else { weight.name() };
            return self.tc(name, w, bias, x, out, rows, tokens);
        }
        let out = self.f32_buf(output, run);
        let name = format!("{}_gemm", weight.name());
        let x = self.f32_buf(input, run);
        if head {
            let view = x.slice(run.rows.start * hidden..run.rows.end * hidden);
            return self.gemm(&name, w, bias, &view, out, rows, tokens);
        }
        self.gemm(&name, w, bias, &x.as_view(), out, rows, tokens)
    }

    fn qk_rope_legacy(&self, l: &DeviceAttn, n: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let (pages, mm_stride) = (self.pages as i32, self.max_ctx as i32);
        let mut b = self.stream.launch_builder(self.kernel("qk_norm_rope"));
        b.arg(&pre.qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(&buf.k_cache)
            .arg(&buf.v_cache)
            .arg(&table)
            .arg(&buf.pos)
            .arg(&buf.zero_slots)
            .arg(&l.kv)
            .arg(&0i32)
            .arg(&pages)
            .arg(&mm_stride)
            .arg(&self.mm_meta)
            .arg(&self.mm_table)
            .arg(&self.seq);
        let config = LaunchConfig {
            grid_dim: ((a.heads + 2 * a.kv_heads) as u32, n as u32, 1),
            block_dim: (a.head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn flash_legacy(&self, l: &DeviceAttn, n: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let ntok = n as i32;
        let mut b = self.stream.launch_builder(self.kernel("flash_attention"));
        b.arg(&pre.qkv)
            .arg(&buf.k_cache)
            .arg(&buf.v_cache)
            .arg(&table)
            .arg(&buf.pos)
            .arg(&l.kv)
            .arg(&pre.attn)
            .arg(&ntok)
            .arg(&l.window);
        let config = LaunchConfig {
            grid_dim: (grid(n, 16), a.heads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: self.smem["flash_attention"],
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn swiglu_legacy(&self, n: usize) -> Result<()> {
        let inter = self.spec.dims.intermediate;
        let pre = &self.pre;
        let mut b = self.stream.launch_builder(self.kernel("swiglu"));
        b.arg(&pre.gate_up).arg(&pre.act);
        let config = LaunchConfig {
            grid_dim: (grid(inter, 256), n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }
}
