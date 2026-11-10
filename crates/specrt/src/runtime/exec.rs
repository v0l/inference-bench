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
}

impl Plans {
    pub fn build(spec: &Spec, tensor_core: bool) -> Self {
        Plans {
            prefill: Plan::build(spec, Phase::Prefill, tensor_core),
            decode: Plan::build(spec, Phase::Decode, tensor_core),
        }
    }
}

pub(super) struct Run<'a> {
    pub n: usize,
    pub tokens: &'a CudaSlice<i32>,
    pub logits: &'a CudaSlice<f32>,
    pub rows: Range<usize>,
    pub bb: Option<&'a BatchBuffers>,
    pub start: Option<usize>,
}

impl PlanBackend<Run<'_>> for Runtime {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn block(&self, layer: usize) -> Block {
        self.spec.block(layer)
    }

    fn run_op(&self, op: &Op, layer: Option<usize>, run: &Run<'_>) -> Result<()> {
        self.run_one(op, layer.map(|i| &self.layers[i]), run)
    }
}

impl Runtime {
    fn run_one(&self, op: &Op, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let n = run.n;
        match *op {
            Op::Embed => {
                self.embed_rows(run)?;
                match run.start {
                    Some(start) => self.splice_media(self.seq as usize, start, run.n, 0),
                    None => Ok(()),
                }
            }
            Op::Norm { which, into } => self.norm_op(which, into, layer, run),
            Op::Linear { weight, input, output } => self.linear_op(weight, input, output, layer, run),
            Op::KvRope => self.attn_rope_rows(self.attn_of(layer), n, 0),
            Op::QkRope => self.qk_rope_legacy(self.attn_of(layer), n),
            Op::Flash => self.attn_flash_rows(self.attn_of(layer), n, 0),
            Op::FlashLegacy => self.flash_legacy(self.attn_of(layer), n),
            Op::DecodeAttention => self.attn_decode_rows(self.attn_of(layer), run.bb.expect("decode runs carry batch buffers"), n),
            Op::Swiglu => self.swiglu_legacy(n),
            Op::GdnPrefill => self.gdn_prefill_rows(self.gdn_of(layer), n, 0),
            Op::GdnDecode => self.gdn_decode_rows(self.gdn_of(layer), &run.bb.expect("decode runs carry batch buffers").slots, n),
        }
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

    fn norm_op(&self, which: NormKind, into: Buf, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let pre = &self.pre;
        let hidden = self.spec.dims.hidden;
        let weight = match which {
            NormKind::Input => &layer.expect("layer norm runs inside a layer").ln1,
            NormKind::PostAttention => &layer.expect("layer norm runs inside a layer").ln2,
            NormKind::Final => &self.norm,
        };
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

    fn planes(&self, b: Buf) -> &Planes {
        match b {
            Buf::XnPlanes => &self.pre.xn_planes,
            Buf::AttnPlanes => &self.pre.attn_planes,
            Buf::ActPlanes => &self.pre.act_planes,
            Buf::GdnPlanes => &self.gdn.as_ref().expect("linear layers have gdn buffers").planes,
            other => unreachable!("{} is not a plane pair", other.label()),
        }
    }

    fn f32_buf<'a>(&'a self, b: Buf, run: &Run<'a>) -> &'a CudaSlice<f32> {
        let gdn = || self.gdn.as_ref().expect("linear layers have gdn buffers");
        match b {
            Buf::X => &self.pre.x,
            Buf::Xn => &self.pre.xn,
            Buf::Qkv => &self.pre.qkv,
            Buf::Attn => &self.pre.attn,
            Buf::GateUp => &self.pre.gate_up,
            Buf::Act => &self.pre.act,
            Buf::PreLin => &gdn().pre_lin,
            Buf::PreBa => &gdn().pre_ba,
            Buf::Logits => run.logits,
            other => unreachable!("{} is not an f32 buffer", other.label()),
        }
    }

    fn is_planes(b: Buf) -> bool {
        matches!(b, Buf::XnPlanes | Buf::AttnPlanes | Buf::ActPlanes | Buf::GdnPlanes)
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
            return self.tc(weight.name(), w, bias, x, out, rows, tokens);
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
            .arg(&ntok);
        let config = LaunchConfig {
            grid_dim: (grid(n, 16), a.heads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
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
