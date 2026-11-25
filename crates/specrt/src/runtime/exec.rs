use super::{BatchBuffers, DeviceAttn, DeviceLayer, DeviceMatrix, DeviceMixer, GemmOut, Planes, Runtime, grid};
use crate::executor::PlanBackend;
use crate::plan::{Buf, NormKind, Op, Phase, Plan, Weight};
use crate::graph::Bind;
use crate::spec::{Block, Spec};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaSlice, LaunchConfig};
use std::ops::Range;

pub(super) struct Plans {
    pub prefill: Plan,
    pub decode: Plan,
    pub verify: Plan,
    pub single: std::result::Result<Plan, String>,
}

impl Plans {
    pub fn compile(spec: &Spec, tensor_core: bool, target: &kir::Target, opts: &crate::compile::Options, role: super::StageRole) -> Result<Self> {
        let build = |phase| -> Result<Plan> { Ok(Self::staged(crate::compile::compile(spec, Plan::build(spec, phase, tensor_core), target, opts)?, role)) };
        let single = crate::compile::single(spec, target, opts.context).map(|p| Self::staged(p, role)).map_err(|e| format!("{e:#}"));
        Ok(Plans { prefill: build(Phase::Prefill)?, decode: build(Phase::Decode)?, verify: build(Phase::Verify)?, single })
    }

    fn staged(mut plan: Plan, role: super::StageRole) -> Plan {
        if !role.first {
            plan.head.clear();
        }
        if !role.last {
            plan.tail.clear();
        }
        plan
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

    fn tail(&self, layer: usize) -> bool {
        self.spec.shares_kv(layer)
    }

    fn run_op(&self, op: &Op, layer: Option<usize>, run: &Run<'_>) -> Result<()> {
        self.run_one(op, layer.map(|i| &self.layers[i]), run)
    }

    fn enter_layer(&self, layer: usize) -> Result<()> {
        if let Some(hb) = &self.gated
            && let Some(t) = hb.trace.lock().unwrap().as_ref()
            && let Some(delta) = t.inject.get(&layer)
        {
            let mut b = self.launcher(self.kernel("residual_add"));
            let hidden = self.spec.dims.hidden;
            b.arg(&hb.streams).arg(delta).arg(&hb.streams);
            let config = LaunchConfig { grid_dim: (grid(hidden, 256), (delta.len() / hidden) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(config) }?;
        }
        match &self.streamer {
            Some(s) => s.enter(layer, &self.stream),
            None => Ok(()),
        }
    }

    fn leave_layer(&self, layer: usize) -> Result<()> {
        match &self.streamer {
            Some(s) => s.leave(layer, &self.stream),
            None => Ok(()),
        }
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
            Op::NormAdd { which, from, into } => self.norm_add_op(which, from, into, layer, run),
            Op::Clear { buf } => self.clear_op(buf, run),
            Op::LogitSoftcap => self.softcap_op(run, "logit_softcap"),
            Op::LogitScale => self.softcap_op(run, "logit_scale"),
            Op::Split { from, into } if layer.is_none() && self.gated.is_some() => self.split_rows_op(from, into, run),
            Op::Split { from, into } => self.split_op(from, into, run),
            Op::QkNorm => self.qk_norm_op(self.attn_of(layer), run.n),
            Op::ScaledAdd { from } => self.scaled_add_op(from, run),
            Op::Linear { weight, input, output, .. } => self.linear_op(weight, input, output, layer, run),
            Op::KvRope => self.attn_rope_rows(self.attn_of(layer), n, 0),
            Op::QkRope => self.qk_rope_legacy(self.attn_of(layer), n),
            Op::Flash => self.attn_flash_rows(self.attn_of(layer), n, 0),
            Op::FlashLegacy => self.flash_legacy(self.attn_of(layer), n),
            Op::DecodeAttention => self.attn_decode_rows(self.attn_of(layer), run.bb.expect("decode runs carry batch buffers"), n, run.qlen),
            Op::DecodeRope => self.decode_rope_rows(self.attn_of(layer), run.bb.expect("decode runs carry batch buffers"), n),
            Op::Kernel { node } => self.kernel_op(node, self.attn_of(layer), run),
            Op::MixKernel { node } => self.mix_kernel_op(node, layer.expect("mixer ops run inside a layer"), run),
            Op::Gated { node, site } => self.gated_op(node, site, layer, run),
            Op::Swiglu => self.swiglu_legacy(n),
            Op::GdnPrefill => self.gdn_prefill_rows(self.gdn_of(layer), n, 0),
            Op::GdnDecode => self.gdn_decode_rows(self.gdn_of(layer), &run.bb.expect("decode runs carry batch buffers").slots, n),
            Op::GdnVerify => self.gdn_verify_op(self.gdn_of(layer), run),
            Op::MtpIn | Op::MtpGather | Op::MtpNorm | Op::MtpDraft => self.mtp_op(op, run),
            Op::MoeRoute => self.moe_route(layer.expect("moe runs inside a layer"), &self.pre.x, run.tokens, n),
            Op::MoeExperts => self.moe_experts(layer.expect("moe runs inside a layer"), n),
            Op::MoeCombine => self.moe_combine(if self.comm.is_some() { &self.partial } else { &self.pre.x }, n),
            Op::MoeCombineInto { into } => self.moe_combine(self.f32_buf(into, run), n),
            Op::AllReduce => self.allreduce_op(n),
            Op::AllReduceNorm { which } => self.allreduce_norm_op(which, layer, n),
            Op::Gemv { .. } | Op::GemvStore { .. } | Op::StepAttention | Op::GdnStep => unreachable!("{} runs on the single row plan", op.label()),
            Op::HyperExpand | Op::HyperPre { .. } | Op::HyperPost | Op::HyperMean => self.hyper_op(op, layer, n),
            Op::PerLayer { node, stage } => self.per_layer_op(node, stage, layer, run),
        }
    }

    fn per_layer_op(&self, node: u8, stage: u8, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let pb = self.ple.as_ref().context("the spec has no [per_layer]")?;
        let p = self.spec.per_layer().expect("per layer buffers need a [per_layer]");
        let launch = &pb.launch[node as usize];
        let spec_node = &p.node[node as usize];
        let head = stage == crate::plan::PLE_HEAD;
        let weights = if head { &pb.head } else { &layer.expect("a per layer node runs inside a layer").ple };
        let at: i32 = if head { 0 } else { layer.expect("a per layer node runs inside a layer").ple_at };
        let numbers: Vec<i32> = launch
            .params
            .iter()
            .map(|p| match spec_node.bind.get(p.name) {
                Some(Bind::Number(v)) => *v as i32,
                _ => 0,
            })
            .collect();
        let mut b = self.launcher(self.kernel(launch.name));
        for (i, param) in launch.params.iter().enumerate() {
            match spec_node.bind.get(param.name) {
                Some(Bind::Weight { weight, .. }) => {
                    b.arg(weights.get(weight).with_context(|| format!("no per layer weight {weight}"))?);
                }
                Some(Bind::Buffer { buffer }) if buffer == "x" => {
                    b.arg(&self.pre.x);
                }
                Some(Bind::Buffer { buffer }) if buffer == "tokens" => {
                    b.arg(run.tokens);
                }
                Some(Bind::Buffer { buffer }) => {
                    b.arg(pb.scratch.get(buffer).with_context(|| format!("no per layer scratch {buffer:?}"))?);
                }
                Some(Bind::Name(n)) if n == "layer" => {
                    b.arg(&at);
                }
                Some(Bind::Number(_)) => {
                    b.arg(&numbers[i]);
                }
                _ => bail!("param {:?} of {} is not bound to a buffer, weight or number", param.name, launch.name),
            }
        }
        let dim = |symbol: &String| -> Result<u32> {
            Ok(match symbol.as_str() {
                "rows" => run.n as u32,
                other => crate::state::dim(&self.spec, other)? as u32,
            })
        };
        let dims = spec_node.grid.iter().map(dim).collect::<Result<Vec<_>>>()?;
        let config = LaunchConfig { grid_dim: (dims[0], dims.get(1).copied().unwrap_or(1), dims.get(2).copied().unwrap_or(1)), block_dim: (launch.threads, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn hyper_op(&self, op: &Op, layer: Option<&DeviceLayer>, n: usize) -> Result<()> {
        let (name, threads) = match *op {
            Op::HyperExpand => ("hyper_expand", 128),
            Op::HyperPre { .. } => ("hyper_pre", 256),
            Op::HyperPost => ("hyper_post", 128),
            _ => ("hyper_mean", 128),
        };
        let mut b = self.launcher(self.kernel(name));
        match *op {
            Op::HyperExpand => b.arg(&self.pre.x).arg(&self.pre.streams),
            Op::HyperPre { site } => {
                let w = &layer.context("hyper pre runs inside a layer")?.hyper;
                let at = 3 * site as usize;
                ensure!(w.len() >= at + 3, "the layer has no hyper weights for site {site}");
                b.arg(&self.pre.streams).arg(&w[at]).arg(&w[at + 1]).arg(&w[at + 2]).arg(&self.pre.x).arg(&self.pre.hyper_mix)
            }
            Op::HyperPost => b.arg(&self.pre.x).arg(&self.pre.hyper_mix).arg(&self.pre.streams),
            _ => b.arg(&self.pre.streams).arg(&self.pre.x),
        };
        unsafe { b.launch(LaunchConfig { grid_dim: (1, n as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }) }?;
        Ok(())
    }

    fn kernel_op(&self, node: u8, l: &DeviceAttn, run: &Run) -> Result<()> {
        let launch = self.described.get(node as usize).and_then(Option::as_ref).context("attention node has no described kernel")?;
        let spec_node = &self.spec.attention.node[node as usize];
        let (table, slots, pos, pos_step, row_step) = match run.bb {
            Some(bb) => (self.buf.block_table.as_view(), &bb.slots, &bb.pos, 1i32, 0i32),
            None => (self.slot_table(self.seq as usize), &self.buf.zero_slots, &self.buf.pos, 0, 1),
        };
        let numbers = |bind: &Bind| -> Result<f64> {
            Ok(match bind {
                Bind::Number(v) => *v,
                Bind::Name(n) => match n.as_str() {
                    "window" => l.window as f64,
                    "nope" => f64::from(u8::from(l.nope)),
                    "kind_layer" => f64::from(l.kind_kv),
                    other => *self.spec.attention.params.get(other).with_context(|| format!("{other:?} is not window, nope, kind_layer or an [attention.params] entry"))?,
                },
                Bind::Weight { .. } | Bind::Buffer { .. } | Bind::State { .. } => bail!("a weight, buffer or state binding is not a number"),
            })
        };
        let mut ints = vec![0i32; launch.params.len()];
        let mut floats = vec![0f32; launch.params.len()];
        for (i, p) in launch.params.iter().enumerate().filter(|(_, p)| p.role == kir::Role::Value) {
            let v = match p.name {
                "layer" => match spec_node.bind.get("layer") {
                    Some(b) => numbers(b)?,
                    None => f64::from(l.kv),
                },
                "pages" => self.pages as f64,
                "pos_step" => f64::from(pos_step),
                "row_step" => f64::from(row_step),
                name => numbers(spec_node.bind.get(name).with_context(|| format!("param {name:?} is not bound"))?)?,
            };
            ints[i] = v as i32;
            floats[i] = v as f32;
        }
        let a = &self.spec.attention;
        let tiled = launch.tile && !self.scan_tile_off.load(std::sync::atomic::Ordering::Relaxed) && crate::codegen::described_tiles(&self.spec, spec_node);
        let tile_name = tiled.then(|| format!("{}{}", launch.name, if run.bb.is_none() { "_mma" } else { "_mma_dec" }));
        let which = match (tiled, run.bb.is_none()) {
            (true, true) => 0,
            (true, false) => 1,
            _ => 2,
        };
        self.scan_launches[which].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut b = match &tile_name {
            Some(n) => self.launcher(self.kernel(n)),
            None => self.launcher(self.kernel(launch.name)),
        };
        for (i, p) in launch.params.iter().enumerate() {
            match (p.role, p.name) {
                (kir::Role::Value, _) => match p.ty {
                    kir::Scalar::F32 => b.arg(&floats[i]),
                    _ => b.arg(&ints[i]),
                },
                (kir::Role::Weight, name) => match spec_node.bind.get(name) {
                    Some(Bind::Weight { weight, .. }) => b.arg(l.extra.get(weight).with_context(|| format!("layer has no {weight} tensor"))?),
                    _ => bail!("param {name:?} is not bound to a weight"),
                },
                (_, name) if spec_node.bind.contains_key(name) => match &spec_node.bind[name] {
                    Bind::Buffer { buffer } => match buffer.as_str() {
                        "qkv" => b.arg(&self.pre.qkv),
                        "attn" => b.arg(&self.pre.attn),
                        other => match self.int_scratch.get(other) {
                            Some(ints) => b.arg(ints),
                            None => b.arg(self.scratch.get(other).with_context(|| format!("no scratch buffer {other:?}"))?),
                        },
                    },
                    Bind::State { state } => {
                        let at = crate::state::positional(&self.spec).position(|s| s.name == *state).with_context(|| format!("no per token state {state:?}"))?;
                        b.arg(&self.buf.caches[at])
                    }
                    _ => bail!("param {name:?} of {} is bound to a number", launch.name),
                },
                (_, "qkv") => b.arg(&self.pre.qkv),
                (_, "k_cache") => b.arg(self.buf.k_cache()),
                (_, "v_cache") => b.arg(self.buf.v_cache()),
                (_, "block_table") => b.arg(&table),
                (_, "pos") => b.arg(pos),
                (_, "slots") => b.arg(slots),
                (_, "out_hi") => b.arg(&self.pre.attn_planes.hi),
                (_, "out_lo") => b.arg(&self.pre.attn_planes.lo),
                (_, name) => bail!("kernel {} has an unknown buffer param {name:?}", launch.name),
            };
        }
        if tiled {
            let rows = crate::codegen::flash_rows(a.head_dim, self.kv_planes);
            let tokens = run.n as i32;
            let splits = crate::codegen::decode_splits(self.max_ctx) as i32;
            let config = if run.bb.is_none() {
                b.arg(&tokens);
                LaunchConfig { grid_dim: (grid(run.n, rows), a.heads as u32, 1), block_dim: (rows as u32 * 2, 1, 1), shared_mem_bytes: self.smem["flash_mma"] }
            } else {
                let bb = run.bb.expect("tile decode runs on batch buffers");
                b.arg(&bb.part_acc).arg(&bb.part_ml).arg(&splits);
                LaunchConfig { grid_dim: (a.kv_heads as u32, run.n as u32, splits as u32), block_dim: (128, 1, 1), shared_mem_bytes: self.smem["flash_mma"] }
            };
            unsafe { b.launch(config) }?;
            if let Some(bb) = run.bb {
                let mut c = self.launcher(self.kernel("batch_decode_combine"));
                c.arg(&bb.part_acc).arg(&bb.part_ml).arg(&self.pre.qkv).arg(&bb.pos).arg(&1i32).arg(&splits).arg(&self.pre.attn_planes.hi).arg(&self.pre.attn_planes.lo);
                unsafe { c.launch(LaunchConfig { grid_dim: (a.heads as u32, run.n as u32, 1), block_dim: (a.head_dim as u32 / 2, 1, 1), shared_mem_bytes: 0 }) }?;
            }
            return Ok(());
        }
        let dim = |symbol: &String| -> Result<u32> {
            let a = &self.spec.attention;
            Ok(match symbol.as_str() {
                "heads" => a.heads as u32,
                "kv_heads" => a.kv_heads as u32,
                "rows" => run.n as u32,
                other => match a.params.get(other) {
                    Some(v) => *v as u32,
                    None => other.parse()?,
                },
            })
        };
        let dims = spec_node.grid.iter().map(dim).collect::<Result<Vec<_>>>()?;
        let config = LaunchConfig { grid_dim: (dims[0], dims.get(1).copied().unwrap_or(1), dims.get(2).copied().unwrap_or(1)), block_dim: (launch.threads, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn mix_kernel_op(&self, node: u8, layer: &DeviceLayer, run: &Run) -> Result<()> {
        let DeviceMixer::Described(d) = &layer.mixer else { unreachable!("mixer op on a layer without a described mixer") };
        let mb = self.mixer.as_ref().expect("a described mixer has buffers");
        let m = self.spec.mixer.as_ref().expect("a described mixer has a spec");
        let launch = &self.mixer_launch[node as usize];
        let spec_node = &m.node[node as usize];
        let (seqs, rows_per_seq, slot_base, slot_step, slots) = match run.bb {
            Some(bb) if run.n > 0 => (run.n / run.qlen.max(1), run.qlen.max(1), 0, 1, &bb.slots),
            _ => (1, run.n, self.seq, 0, &self.buf.zero_slots),
        };
        let number = |b: &Bind| -> Result<f64> {
            Ok(match b {
                Bind::Number(v) => *v,
                Bind::Name(n) => *m.params.get(n).with_context(|| format!("{n:?} is not a [mixer.params] entry"))?,
                _ => bail!("a weight, buffer or state binding is not a number"),
            })
        };
        let mut ints = vec![0i32; launch.params.len()];
        let mut floats = vec![0f32; launch.params.len()];
        for (i, p) in launch.params.iter().enumerate().filter(|(_, p)| p.role == kir::Role::Value) {
            let v = match p.name {
                "layer" => f64::from(d.slot),
                "slot_base" => f64::from(slot_base),
                "slot_step" => f64::from(slot_step),
                "rows_per_seq" => rows_per_seq as f64,
                name => number(spec_node.bind.get(name).with_context(|| format!("param {name:?} is not bound"))?)?,
            };
            ints[i] = v as i32;
            floats[i] = v as f32;
        }
        let launch_one = |name: &str, config: LaunchConfig, chunked: bool| -> Result<()> {
            let mut b = self.launcher(self.kernel(name));
            for (i, p) in launch.params.iter().enumerate() {
                match (p.role, p.name) {
                    (kir::Role::Value, _) => match p.ty {
                        kir::Scalar::F32 => b.arg(&floats[i]),
                        _ => b.arg(&ints[i]),
                    },
                    (_, "slots") => b.arg(slots),
                    (_, "out_hi") => b.arg(&mb.planes.hi),
                    (_, "out_lo") => b.arg(&mb.planes.lo),
                    (_, name) => match spec_node.bind.get(name) {
                        Some(Bind::Weight { weight, .. }) => b.arg(d.extra.get(weight).with_context(|| format!("layer has no {weight} tensor"))?),
                        Some(Bind::Buffer { buffer }) if buffer == "mix" => b.arg(&mb.mix),
                        Some(Bind::Buffer { buffer }) => b.arg(mb.scratch.get(buffer).with_context(|| format!("no mixer scratch {buffer:?}"))?),
                        Some(Bind::State { state }) => b.arg(&*mb.states.iter().find(|s| s.name == *state).with_context(|| format!("no sequence state {state:?}"))?.slice),
                        _ => bail!("param {name:?} of {} is not bound to a buffer", launch.name),
                    },
                };
            }
            if chunked {
                b.arg(&mb.chunk_state).arg(&mb.chunk_decay);
            }
            unsafe { b.launch(config) }?;
            Ok(())
        };
        let dim = |symbol: &String| -> Result<u32> {
            Ok(match symbol.as_str() {
                "rows" => run.n as u32,
                "seqs" => seqs as u32,
                other => match m.params.get(other) {
                    Some(v) => *v as u32,
                    None => other.parse()?,
                },
            })
        };
        let dims = spec_node.grid.iter().map(dim).collect::<Result<Vec<_>>>()?;
        if let Some(f) = launch.chunk.as_ref().filter(|f| rows_per_seq >= f.min_rows as usize) {
            let block = (f.threads, 1, 1);
            let tiles = LaunchConfig { grid_dim: (dims[0], rows_per_seq.div_ceil(f.rows as usize) as u32, seqs as u32), block_dim: block, shared_mem_bytes: f.state_smem };
            launch_one(f.state, tiles, true)?;
            launch_one(f.carry, LaunchConfig { grid_dim: (dims[0], seqs as u32, f.state_floats.div_ceil(f.threads)), block_dim: block, shared_mem_bytes: 0 }, true)?;
            return launch_one(f.out, LaunchConfig { grid_dim: (dims[0], tiles.grid_dim.1 * f.subs, seqs as u32), shared_mem_bytes: f.out_smem, ..tiles }, true);
        }
        let config = LaunchConfig { grid_dim: (dims[0], dims.get(1).copied().unwrap_or(1), dims.get(2).copied().unwrap_or(1)), block_dim: (launch.threads, 1, 1), shared_mem_bytes: 0 };
        launch_one(launch.name, config, false)
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

    pub(super) fn attn_of<'a>(&self, layer: Option<&'a DeviceLayer>) -> &'a DeviceAttn {
        match &layer.expect("attention ops run inside a layer").mixer {
            DeviceMixer::Attention(a) => a,
            _ => unreachable!("attention op on a linear layer"),
        }
    }

    pub(super) fn gdn_of<'a>(&self, layer: Option<&'a DeviceLayer>) -> &'a super::DeviceGdn {
        match &layer.expect("linear ops run inside a layer").mixer {
            DeviceMixer::Linear(g) => g,
            _ => unreachable!("linear op on an attention layer"),
        }
    }

    fn described_of<'a>(&self, layer: Option<&'a DeviceLayer>) -> &'a super::DeviceDescribed {
        match &layer.expect("mixer ops run inside a layer").mixer {
            DeviceMixer::Described(d) => d,
            _ => unreachable!("mixer op on a layer without a described mixer"),
        }
    }

    fn embed_rows(&self, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let mut b = self.launcher(self.kernel("embed"));
        b.arg(&self.embed).arg(run.tokens).arg(&self.pre.x);
        let config = LaunchConfig {
            grid_dim: (grid(hidden, 256), run.n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    pub(super) fn norm_weight<'a>(&'a self, which: NormKind, layer: Option<&'a DeviceLayer>) -> &'a CudaSlice<f32> {
        let l = || layer.expect("layer norm runs inside a layer");
        match which {
            NormKind::Input => &l().ln1,
            NormKind::PostAttention => &l().ln2,
            NormKind::PreMlp => l().ln_pre_mlp.as_ref().expect("a sandwich layer has a pre mlp norm"),
            NormKind::PostMlp => l().ln_post_mlp.as_ref().expect("a sandwich layer has a post mlp norm"),
            NormKind::PostDense => l().moe_norms().0,
            NormKind::PostExperts => l().moe_norms().1,
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

    fn norm_add_op(&self, which: NormKind, from: Buf, into: Buf, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let weight = self.norm_weight(which, layer);
        let mut b = self.launcher(self.kernel("norm_add"));
        let delta = self.f32_buf(from, run);
        let target = self.f32_buf(into, run);
        b.arg(delta).arg(weight).arg(target).arg(target);
        unsafe { b.launch(super::cfg(run.n as u32, 1024)) }?;
        Ok(())
    }

    fn clear_op(&self, buf: Buf, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let target = self.f32_buf(buf, run);
        let mut b = self.launcher(self.kernel("clear_branch"));
        b.arg(target).arg(target);
        let config = LaunchConfig { grid_dim: (grid(hidden, 256), run.n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn split_rows_op(&self, from: Buf, into: Buf, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let planes = self.planes(into);
        let source = self.f32_buf(from, run).slice(run.rows.start * hidden..run.rows.end * hidden);
        let count = (run.rows.len() * hidden) as i32;
        let mut b = self.launcher(self.kernel("split_rows"));
        b.arg(&source).arg(&planes.hi).arg(&planes.lo).arg(&count);
        unsafe { b.launch(super::cfg(super::grid(count as usize, 512), 256)) }?;
        Ok(())
    }

    fn gated_op(&self, node: u8, site: u8, layer: Option<&DeviceLayer>, run: &Run) -> Result<()> {
        let hb = self.gated.as_ref().expect("a gated residual has buffers");
        let h = self.spec.gated_residual().expect("a gated residual");
        let launch = &hb.launch[node as usize];
        let spec_node = &h.node[node as usize];
        let weights = match layer {
            Some(l) if site != crate::plan::GATED_MODULE => &l.gated[site as usize],
            _ => &hb.head,
        };
        let mut b = self.launcher(self.kernel(launch.name));
        for p in &launch.params {
            match spec_node.bind.get(p.name) {
                Some(Bind::Weight { weight, .. }) => b.arg(weights.get(weight).with_context(|| format!("no hyper weight {weight}"))?),
                Some(Bind::Buffer { buffer }) if buffer == "x" => b.arg(&self.pre.x),
                Some(Bind::Buffer { buffer }) if buffer == "streams" => b.arg(&hb.streams),
                Some(Bind::Buffer { buffer }) => b.arg(hb.scratch.get(buffer).with_context(|| format!("no hyper scratch {buffer:?}"))?),
                _ => bail!("param {:?} of {} is not bound to a buffer or weight", p.name, launch.name),
            };
        }
        let dim = |symbol: &String| -> Result<u32> {
            Ok(match symbol.as_str() {
                "rows" => run.n as u32,
                other => crate::state::dim(&self.spec, other)? as u32,
            })
        };
        let dims = spec_node.grid.iter().map(dim).collect::<Result<Vec<_>>>()?;
        let config = LaunchConfig { grid_dim: (dims[0], dims.get(1).copied().unwrap_or(1), dims.get(2).copied().unwrap_or(1)), block_dim: (launch.threads, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn split_op(&self, from: Buf, into: Buf, run: &Run) -> Result<()> {
        let planes = self.planes(into);
        let source = self.f32_buf(from, run);
        let count = (run.n * self.spec.dims.hidden) as i32;
        let mut b = self.launcher(self.kernel("split_rows"));
        b.arg(source).arg(&planes.hi).arg(&planes.lo).arg(&count);
        unsafe { b.launch(super::cfg(super::grid(count as usize, 512), 256)) }?;
        Ok(())
    }

    fn qk_norm_op(&self, l: &DeviceAttn, n: usize) -> Result<()> {
        let (q, k) = (l.q_norm.as_ref().expect("full-width qk norm has weights"), l.k_norm.as_ref().expect("full-width qk norm has weights"));
        for (name, w) in [("q_norm_rows", q), ("k_norm_rows", k)] {
            let mut b = self.launcher(self.kernel(name));
            b.arg(&self.pre.qkv).arg(w).arg(&self.pre.qkv);
            unsafe { b.launch(super::cfg(n as u32, 1024)) }?;
        }
        Ok(())
    }

    fn scaled_add_op(&self, from: Buf, run: &Run) -> Result<()> {
        let hidden = self.spec.dims.hidden;
        let delta = self.f32_buf(from, run);
        let mut b = self.launcher(self.kernel("residual_add"));
        b.arg(&self.pre.x).arg(delta).arg(&self.pre.x);
        let config = LaunchConfig { grid_dim: (grid(hidden, 256), run.n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn softcap_op(&self, run: &Run, name: &str) -> Result<()> {
        let vocab = self.spec.dims.vocab;
        let mut b = self.launcher(self.kernel(name));
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
            Buf::MixPlanes => &self.mixer.as_ref().expect("a described mixer has buffers").planes,
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
            Buf::Merged => &self.pre.merged,
            Buf::Qkv => &self.pre.qkv,
            Buf::Attn => &self.pre.attn,
            Buf::GateUp => &self.pre.gate_up,
            Buf::Act => &self.pre.act,
            Buf::PreLin => &gdn().pre_lin,
            Buf::PreBa => &gdn().pre_ba,
            Buf::Mix => &self.mixer.as_ref().expect("a described mixer has buffers").mix,
            Buf::Logits => run.logits,
            Buf::MtpLogits => self.mtp_logits(),
            Buf::Partial => &self.partial,
            other => unreachable!("{} is not an f32 buffer", other.label()),
        }
    }

    pub(super) fn weight_of<'a>(&'a self, weight: Weight, layer: Option<&'a DeviceLayer>) -> (&'a DeviceMatrix, Option<&'a CudaSlice<f32>>) {
        let attn = || self.attn_of(layer);
        let gdn = || self.gdn_of(layer);
        let layer_of = || layer.expect("layer weights need a layer");
        match weight {
            Weight::Qkv => (&attn().qkv, attn().qkv_bias.as_ref()),
            Weight::O => (&attn().o, attn().o_bias.as_ref()),
            Weight::GateUp | Weight::GateUpWide => (&layer_of().gate_up, None),
            Weight::Down | Weight::DownWide => (&layer_of().down, None),
            Weight::LinIn => (&gdn().in_proj, None),
            Weight::LinBa => (&gdn().ba, None),
            Weight::LinOut => (&gdn().out, None),
            Weight::MixIn => (&self.described_of(layer).in_proj, None),
            Weight::MixOut => (&self.described_of(layer).out, None),
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
        if matches!(weight, Weight::LmHead | Weight::MtpHead) && let Some(part) = &self.head_part {
            return self.sharded_head(weight, w, input, output, part, run);
        }
        let rows = weight.rows(&self.spec);
        let hidden = self.spec.dims.hidden;
        let head = weight == Weight::LmHead;
        let tokens = if head { run.rows.len() } else { run.n };
        if input.is_planes() {
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

    fn sharded_head(&self, weight: Weight, w: &DeviceMatrix, input: Buf, output: Buf, part: &CudaSlice<f32>, run: &Run) -> Result<()> {
        anyhow::ensure!(input.is_planes(), "a vocabulary sharded head reads the tensor core planes");
        let comm = self.comm();
        let (hidden, vocab, own) = (self.spec.dims.hidden, self.spec.dims.vocab, self.spec.head_rows().len());
        let each = comm.gather_each().context("a vocabulary sharded head needs the gather kernel")?;
        let tokens = if weight == Weight::LmHead { run.rows.len() } else { run.n };
        let batch = (part.len() / each).min(comm.capacity_bytes() / (each * 4)).max(1);
        let planes = self.planes(input);
        let out = self.f32_buf(output, run);
        let name = if weight == Weight::MtpHead { self.mtp_head_kernel() } else { weight.name() };
        let mut at = 0;
        while at < tokens {
            let n = (tokens - at).min(batch);
            self.tc(name, w, None, planes.rows(at, at + n, hidden), GemmOut::F32(part), own, n)?;
            comm.gather_rows(&self.stream, &part.slice(..n * own), &out.slice(at * vocab..(at + n) * vocab), n)?;
            at += n;
        }
        Ok(())
    }

    fn qk_rope_legacy(&self, l: &DeviceAttn, n: usize) -> Result<()> {
        let a = &self.spec.attention;
        let pre = &self.pre;
        let buf = &self.buf;
        let table = self.slot_table(self.seq as usize);
        let (pages, mm_stride) = (self.pages as i32, self.max_ctx as i32);
        let mut b = self.launcher(self.kernel("qk_norm_rope"));
        b.arg(&pre.qkv)
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
        let mut b = self.launcher(self.kernel("flash_attention"));
        b.arg(&pre.qkv)
            .arg(buf.k_cache())
            .arg(buf.v_cache())
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
        let mut b = self.launcher(self.kernel("swiglu"));
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
