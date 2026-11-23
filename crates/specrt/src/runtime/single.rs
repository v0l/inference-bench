use super::{DeviceLayer, DeviceMatrix, Runtime, cfg, grid};
use crate::executor::PlanBackend;
use crate::plan::{Buf, NormKind, Op, QuantSource, Weight, weight_kind};
use crate::spec::Block;
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaSlice, LaunchConfig};

pub(super) struct Single;

impl PlanBackend<Single> for Runtime {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn block(&self, layer: usize) -> Block {
        self.spec.block(layer)
    }

    fn sparse(&self, layer: usize) -> bool {
        self.layers[layer].moe.is_some()
    }

    fn run_op(&self, op: &Op, layer: Option<usize>, _: &Single) -> Result<()> {
        self.single_op(op, layer, layer.map(|i| &self.layers[i]))
    }

    fn enter_layer(&self, layer: usize) -> Result<()> {
        Runtime::enter_layer(self, layer)
    }

    fn leave_layer(&self, layer: usize) -> Result<()> {
        Runtime::leave_layer(self, layer)
    }
}

impl Runtime {
    fn row(&self, b: Buf) -> Result<&CudaSlice<f32>> {
        let gdn = || self.gdn.as_ref().context("linear layers have gdn buffers");
        Ok(match b {
            Buf::X => &self.buf.x,
            Buf::Qkv => &self.buf.qkv,
            Buf::Attn => &self.buf.attn,
            Buf::GateUp => &self.buf.gate_up,
            Buf::Act => &self.buf.act,
            Buf::Logits => &self.buf.logits,
            Buf::Branch => &self.pre.branch,
            Buf::PreLin => &gdn()?.lin,
            Buf::PreBa => &gdn()?.ba,
            Buf::Gdn => &gdn()?.out,
            other => bail!("{} has no single row buffer", other.label()),
        })
    }

    fn single_op(&self, op: &Op, index: Option<usize>, layer: Option<&DeviceLayer>) -> Result<()> {
        let s = &self.spec;
        let hidden = s.dims.hidden;
        match *op {
            Op::Embed => {
                let mut b = self.launcher(self.kernel("embed"));
                b.arg(&self.embed).arg(&self.buf.token).arg(&self.buf.x);
                unsafe { b.launch(cfg(grid(hidden, 256), 256)) }?;
            }
            Op::Gemv { norm, weight, quant, input, output } => self.gemv_op(norm, weight, quant, input, output, index.unwrap_or(s.layers.count.saturating_sub(1)), layer, false)?,
            Op::GemvStore { weight, quant, input, output } => self.gemv_op(None, weight, quant, input, output, index.unwrap_or(s.layers.count.saturating_sub(1)), layer, true)?,
            Op::Swiglu => {
                let mut b = self.launcher(self.kernel("swiglu"));
                b.arg(&self.buf.gate_up).arg(&self.buf.act);
                unsafe { b.launch(cfg(grid(s.dims.intermediate, 256), 256)) }?;
            }
            Op::QkNorm => {
                let l = self.attn_of(layer);
                for (name, w) in [("q_norm_rows", &l.q_norm), ("k_norm_rows", &l.k_norm)] {
                    let mut b = self.launcher(self.kernel(name));
                    b.arg(&self.buf.qkv).arg(w.as_ref().context("full-width qk norm has weights")?).arg(&self.buf.qkv);
                    unsafe { b.launch(cfg(1, 1024)) }?;
                }
            }
            Op::StepAttention => self.step_attention(layer)?,
            Op::Clear { buf } => {
                let target = self.row(buf)?;
                let mut b = self.launcher(self.kernel("clear_branch"));
                b.arg(target).arg(target);
                unsafe { b.launch(cfg(grid(hidden, 256), 256)) }?;
            }
            Op::NormAdd { which, from } => {
                let mut b = self.launcher(self.kernel("norm_add"));
                b.arg(self.row(from)?).arg(self.norm_weight(which, layer)).arg(&self.buf.x).arg(&self.buf.x);
                unsafe { b.launch(cfg(1, 1024)) }?;
            }
            Op::ScaledAdd { from } => {
                let mut b = self.launcher(self.kernel("residual_add"));
                b.arg(&self.buf.x).arg(self.row(from)?).arg(&self.buf.x);
                unsafe { b.launch(cfg(grid(hidden, 256), 256)) }?;
            }
            Op::GdnStep => self.gdn_step(layer)?,
            Op::MoeRoute => self.moe_route(layer.context("moe runs inside a layer")?, &self.buf.x, &self.buf.token, 1)?,
            Op::MoeExperts => self.moe_experts(layer.context("moe runs inside a layer")?, 1)?,
            Op::MoeCombine => self.moe_combine(&self.buf.x, 1)?,
            Op::LogitScale | Op::LogitSoftcap => {
                let name = if *op == Op::LogitScale { "logit_scale" } else { "logit_softcap" };
                let mut b = self.launcher(self.kernel(name));
                b.arg(&self.buf.logits).arg(&self.buf.logits);
                unsafe { b.launch(cfg(grid(s.dims.vocab, 256), 256)) }?;
            }
            ref other => bail!("{} is not a single row op", other.label()),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_op(&self, norm: Option<NormKind>, weight: Weight, quant: QuantSource, input: Buf, output: Buf, index: usize, layer: Option<&DeviceLayer>, store: bool) -> Result<()> {
        let (w, bias): (&DeviceMatrix, _) = self.weight_of(weight, layer);
        let kind = weight_kind(quant.resolve_for(&self.spec, index, weight));
        ensure!(w.kind() == kind, "{} is planned as {kind:?} but loaded as {:?}", weight.name(), w.kind());
        let g = crate::gemm::gemv_into(&self.spec, weight.name(), kind, norm.is_some(), store).with_context(|| format!("no gemv for {}", weight.name()))?;
        let mut b = self.launcher(self.kernel(g.name));
        w.args(&mut b, &self.buf.dummy);
        if g.bias {
            b.arg(bias.unwrap_or(&self.buf.dummy));
        }
        b.arg(self.row(input)?);
        if let Some(which) = norm {
            b.arg(self.norm_weight(which, layer));
        }
        b.arg(self.row(output)?);
        unsafe { b.launch(cfg(g.grid(self.sms), g.threads)) }?;
        Ok(())
    }

    fn step_attention(&self, layer: Option<&DeviceLayer>) -> Result<()> {
        let l = self.attn_of(layer);
        let a = &self.spec.attention;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let (pages, mm_stride) = (self.pages as i32, self.max_ctx as i32);
        let mut b = self.launcher(self.kernel("qk_norm_rope"));
        b.arg(&buf.qkv)
            .arg(l.q_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(l.k_norm.as_ref().unwrap_or(&buf.dummy))
            .arg(buf.k_cache())
            .arg(buf.v_cache())
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
        unsafe { b.launch(cfg((a.heads + 2 * a.kv_heads) as u32, a.head_dim.max(32) as u32)) }?;
        let mut b = self.launcher(self.kernel("attention_split"));
        b.arg(&buf.qkv).arg(buf.k_cache()).arg(buf.v_cache()).arg(&table).arg(&buf.pos).arg(&l.kv).arg(&buf.attn_part).arg(&l.window);
        unsafe { b.launch(LaunchConfig { grid_dim: (a.kv_heads as u32, super::ATTN_SPLITS as u32, 1), block_dim: (super::ATTN_BLOCK, 1, 1), shared_mem_bytes: 0 }) }?;
        let mut b = self.launcher(self.kernel("attention_reduce"));
        b.arg(&buf.attn_part).arg(&buf.pos).arg(&buf.qkv).arg(&buf.attn).arg(&l.window);
        unsafe { b.launch(cfg(a.heads as u32, a.head_dim.max(32) as u32)) }?;
        Ok(())
    }

    fn gdn_step(&self, layer: Option<&DeviceLayer>) -> Result<()> {
        let g = self.gdn_of(layer);
        let gb = self.gdn.as_ref().context("linear layers have gdn buffers")?;
        let la = &gb.la;
        let zero = 0i32;
        let mut b = self.launcher(self.kernel("gdn_conv_step"));
        b.arg(&gb.lin).arg(&g.conv).arg(&*gb.conv_state).arg(&gb.mix).arg(&g.slot).arg(&zero).arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(grid(la.conv_dim(), 256), 256)) }?;
        let mut b = self.launcher(self.kernel("gdn_step"));
        b.arg(&gb.mix)
            .arg(&gb.lin)
            .arg(&gb.ba)
            .arg(&g.neg_a)
            .arg(&g.dt_bias)
            .arg(&g.norm)
            .arg(&*gb.ssm_state)
            .arg(&gb.out)
            .arg(&g.slot)
            .arg(&zero)
            .arg(&self.buf.zero_slots);
        unsafe { b.launch(cfg(la.value_heads as u32, (la.value_head_dim * self.linear_tuning.ksplit) as u32)) }?;
        Ok(())
    }
}
