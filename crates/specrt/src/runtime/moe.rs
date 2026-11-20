use super::{DeviceLayer, Runtime, cfg};
use crate::weights::{MoeWeights, Stacked};
use anyhow::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub(super) enum StackWeights {
    Bf16(CudaSlice<u16>),
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8>, inv_global: CudaSlice<f32>, input_global: f32 },
    Fp8 { values: CudaSlice<u8>, grid: CudaSlice<f32> },
    Block { packed: CudaSlice<u8>, scales: CudaSlice<u8> },
}

pub(super) struct DeviceMoe {
    router: CudaSlice<u16>,
    shared_gate: CudaSlice<f32>,
    score_bias: CudaSlice<f32>,
    router_bias: CudaSlice<f32>,
    gate_up_bias: CudaSlice<f32>,
    down_bias: CudaSlice<f32>,
    hash_table: CudaSlice<i32>,
    hashed: i32,
    gate: StackWeights,
    up: StackWeights,
    down: StackWeights,
}

impl StackWeights {
    fn upload(stream: &Arc<cudarc::driver::CudaStream>, s: &Stacked) -> Result<Self> {
        Ok(match s {
            Stacked::Bf16(w) => StackWeights::Bf16(stream.clone_htod(w)?),
            Stacked::Fp8 { values, grid } => StackWeights::Fp8 { values: stream.clone_htod(values)?, grid: stream.clone_htod(grid)? },
            Stacked::Block { packed, scales } => StackWeights::Block { packed: stream.clone_htod(packed)?, scales: stream.clone_htod(scales)? },
            Stacked::Fp4 { packed, scales, inv_global, input_global } => StackWeights::Fp4 {
                input_global: *input_global,
                packed: stream.clone_htod(packed)?,
                scales: stream.clone_htod(scales)?,
                inv_global: stream.clone_htod(inv_global)?,
            },
        })
    }

    fn bytes(&self) -> u64 {
        (match self {
            StackWeights::Bf16(w) => w.num_bytes(),
            StackWeights::Fp8 { values, grid } => values.num_bytes() + grid.num_bytes(),
            StackWeights::Block { packed, scales } => packed.num_bytes() + scales.num_bytes(),
            StackWeights::Fp4 { packed, scales, inv_global, .. } => packed.num_bytes() + scales.num_bytes() + inv_global.num_bytes(),
        }) as u64
    }

    fn mma_ready(&self) -> bool {
        matches!(self, StackWeights::Fp4 { .. } | StackWeights::Block { .. })
    }

    fn suffix(&self) -> &'static str {
        match self {
            StackWeights::Bf16(_) => "bf16",
            StackWeights::Fp4 { .. } => "fp4",
            StackWeights::Fp8 { .. } => "fp8",
            StackWeights::Block { .. } => "mx",
        }
    }
}

impl DeviceMoe {
    pub(super) fn upload(stream: &Arc<cudarc::driver::CudaStream>, m: &MoeWeights) -> Result<Self> {
        Ok(DeviceMoe {
            router: stream.clone_htod(&m.router)?,
            shared_gate: stream.clone_htod(m.shared_gate.as_deref().unwrap_or(&[0.0]))?,
            score_bias: stream.clone_htod(m.score_bias.as_deref().unwrap_or(&[0.0]))?,
            router_bias: stream.clone_htod(m.router_bias.as_deref().unwrap_or(&[0.0]))?,
            gate_up_bias: stream.clone_htod(m.gate_up_bias.as_deref().unwrap_or(&[0.0]))?,
            down_bias: stream.clone_htod(m.down_bias.as_deref().unwrap_or(&[0.0]))?,
            hash_table: stream.clone_htod(m.hash_table.as_deref().unwrap_or(&[0]))?,
            hashed: i32::from(m.hash_table.is_some()),
            gate: StackWeights::upload(stream, &m.gate)?,
            up: StackWeights::upload(stream, &m.up)?,
            down: StackWeights::upload(stream, &m.down)?,
        })
    }

    pub(super) fn upload_with<'a>(stream: &'a Arc<cudarc::driver::CudaStream>) -> impl Fn(&MoeWeights) -> Result<Self> + 'a {
        move |m| Self::upload(stream, m)
    }

    pub(super) fn bytes(&self) -> u64 {
        (self.router.num_bytes() + self.shared_gate.num_bytes() + self.score_bias.num_bytes() + self.router_bias.num_bytes() + self.gate_up_bias.num_bytes() + self.down_bias.num_bytes() + self.hash_table.num_bytes()) as u64 + self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

pub(super) struct MoeBuffers {
    xn: CudaSlice<f32>,
    idx: CudaSlice<i32>,
    coef: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    act: CudaSlice<f32>,
    out: CudaSlice<f32>,
    offsets: CudaSlice<i32>,
    sorted: CudaSlice<i32>,
    mma: Option<MmaBuffers>,
    mma8: Option<Mma8Buffers>,
    mmb: Option<MmbBuffers>,
}

struct MmbBuffers {
    xh: CudaSlice<u16>,
    xl: CudaSlice<u16>,
    ah: CudaSlice<u16>,
    al: CudaSlice<u16>,
    tiles: CudaSlice<i32>,
}

struct Mma8Buffers {
    xq: CudaSlice<u8>,
    xs: CudaSlice<f32>,
    aq: CudaSlice<u8>,
    as_: CudaSlice<f32>,
    tiles: CudaSlice<i32>,
}

struct MmaBuffers {
    xq: CudaSlice<u8>,
    xs: CudaSlice<u8>,
    xq2: CudaSlice<u8>,
    xs2: CudaSlice<u8>,
    aq: CudaSlice<u8>,
    as_: CudaSlice<u8>,
    aq2: CudaSlice<u8>,
    as2: CudaSlice<u8>,
    tiles: CudaSlice<i32>,
}

pub const MMA_MIN_TOKENS: usize = 8;

pub const GROUP_MIN_TOKENS: usize = 17;

impl MoeBuffers {
    pub(super) fn new(stream: &Arc<cudarc::driver::CudaStream>, rows: usize, spec: &crate::spec::Spec, mma: bool, mma8: bool, mmb: bool) -> Result<Self> {
        let m = spec.moe.as_ref().expect("moe buffers need [moe]");
        let pairs = rows * m.pairs();
        Ok(MoeBuffers {
            xn: stream.alloc_zeros(rows * spec.dims.hidden)?,
            idx: stream.alloc_zeros(pairs)?,
            coef: stream.alloc_zeros(pairs)?,
            logits: stream.alloc_zeros(rows * (m.experts + 1))?,
            act: stream.alloc_zeros(pairs * m.expert_intermediate)?,
            out: stream.alloc_zeros(pairs * spec.dims.hidden)?,
            offsets: stream.alloc_zeros(m.experts + 2)?,
            sorted: stream.alloc_zeros(pairs)?,
            mma: if mma {
                let pad = crate::dequant::DEVICE_TILE;
                let (h, i) = (spec.dims.hidden.next_multiple_of(pad), m.expert_intermediate.next_multiple_of(pad));
                let tiles = m.experts + 1 + pairs / crate::codegen::MOE_MMA_BM_SMALL;
                Some(MmaBuffers {
                    xq: stream.alloc_zeros(rows * h / 2)?,
                    xs: stream.alloc_zeros(rows * h / 16)?,
                    xq2: stream.alloc_zeros(rows * h / 2)?,
                    xs2: stream.alloc_zeros(rows * h / 16)?,
                    aq: stream.alloc_zeros(pairs * i / 2)?,
                    as_: stream.alloc_zeros(pairs * i / 16)?,
                    aq2: stream.alloc_zeros(pairs * i / 2)?,
                    as2: stream.alloc_zeros(pairs * i / 16)?,
                    tiles: stream.alloc_zeros(2 * tiles)?,
                })
            } else {
                None
            },
            mmb: if mmb {
                let (h, i) = (spec.dims.hidden, m.expert_intermediate);
                let tiles = m.experts + 1 + pairs / crate::codegen::MOE_MMA_BM_SMALL;
                Some(MmbBuffers {
                    xh: stream.alloc_zeros(rows * h)?,
                    xl: stream.alloc_zeros(rows * h)?,
                    ah: stream.alloc_zeros(pairs * i)?,
                    al: stream.alloc_zeros(pairs * i)?,
                    tiles: stream.alloc_zeros(2 * tiles)?,
                })
            } else {
                None
            },
            mma8: if mma8 {
                let (h, i) = (spec.dims.hidden, m.expert_intermediate);
                let tiles = m.experts + 1 + pairs / crate::codegen::MOE_MMA_BM_SMALL;
                Some(Mma8Buffers {
                    xq: stream.alloc_zeros(rows * h)?,
                    xs: stream.alloc_zeros(rows * h / 32)?,
                    aq: stream.alloc_zeros(pairs * i)?,
                    as_: stream.alloc_zeros(pairs * i / 32)?,
                    tiles: stream.alloc_zeros(2 * tiles)?,
                })
            } else {
                None
            },
        })
    }
}

const ROWS_PER_BLOCK: usize = 8;

impl Runtime {
    pub(super) fn moe_of<'a>(&self, layer: &'a DeviceLayer) -> &'a DeviceMoe {
        layer.moe.as_ref().expect("moe op on a dense layer")
    }

    pub(super) fn moe_route(&self, layer: &DeviceLayer, x: &CudaSlice<f32>, tokens: &CudaSlice<i32>, n: usize) -> Result<()> {
        let w = self.moe_of(layer);
        let mb = self.moe.as_ref().expect("moe buffers");
        let experts = self.spec.moe.as_ref().expect("moe spec").experts;
        let n_arg = n as i32;
        let mut b = self.stream.launch_builder(self.kernel("moe_route_logits"));
        b.arg(x).arg(&layer.ln2).arg(&w.router).arg(&w.shared_gate).arg(&w.router_bias).arg(&mb.xn).arg(&mb.logits).arg(&n_arg);
        let config = LaunchConfig { grid_dim: ((experts + 1).div_ceil(8) as u32, n.div_ceil(4) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel("moe_route"));
        let first = self.spec.moe.as_ref().expect("moe spec").first_expert() as i32;
        b.arg(&mb.logits).arg(&w.score_bias).arg(&w.hash_table).arg(tokens).arg(&w.hashed).arg(&mb.idx).arg(&mb.coef).arg(&first);
        unsafe { b.launch(cfg(n as u32, 32)) }?;
        Ok(())
    }

    fn stack_args<'a>(&'a self, b: &mut cudarc::driver::LaunchArgs<'a>, s: &'a StackWeights) {
        match s {
            StackWeights::Bf16(w) => b.arg(w).arg(&self.pre.dummy_u8).arg(&self.buf.dummy),
            StackWeights::Fp8 { values, grid } => b.arg(values).arg(&self.pre.dummy_u8).arg(grid),
            StackWeights::Fp4 { packed, scales, inv_global, .. } => b.arg(packed).arg(scales).arg(inv_global),
            StackWeights::Block { packed, scales } => b.arg(packed).arg(scales).arg(&self.buf.dummy),
        };
    }

    pub(super) fn moe_experts(&self, layer: &DeviceLayer, n: usize) -> Result<()> {
        let w = self.moe_of(layer);
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mb = self.moe.as_ref().expect("moe buffers");
        let pairs = (n * m.pairs()) as u32;
        if mb.mma.is_some() && w.gate.mma_ready() && n >= MMA_MIN_TOKENS {
            return self.moe_grouped_mma(w, mb, n);
        }
        if mb.mma8.is_some() && matches!(w.gate, StackWeights::Fp8 { .. }) && n >= MMA_MIN_TOKENS {
            return self.moe_grouped_mma8(w, mb, n);
        }
        if mb.mmb.is_some() && matches!(w.gate, StackWeights::Bf16(_)) && n >= MMA_MIN_TOKENS {
            return self.moe_grouped_mmb(w, mb, n);
        }
        if n >= GROUP_MIN_TOKENS {
            return self.moe_grouped(w, mb, n);
        }
        let kind = w.gate.suffix();
        let up = format!("moe_up_{kind}");
        let down = format!("moe_down_{kind}");
        let mut b = self.stream.launch_builder(self.kernel(&up));
        self.stack_args(&mut b, &w.gate);
        self.stack_args(&mut b, &w.up);
        b.arg(&w.gate_up_bias).arg(&mb.xn).arg(&mb.idx).arg(&mb.act);
        let config = LaunchConfig {
            grid_dim: (pairs, m.expert_intermediate.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel(&down));
        self.stack_args(&mut b, &w.down);
        b.arg(&w.down_bias).arg(&mb.act).arg(&mb.idx).arg(&mb.out);
        let config = LaunchConfig {
            grid_dim: (pairs, self.spec.dims.hidden.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn moe_permute(&self, mb: &MoeBuffers, n: usize) -> Result<()> {
        let m = self.spec.moe.as_ref().expect("moe spec");
        let count = (n * m.pairs()) as i32;
        let mut b = self.stream.launch_builder(self.kernel("moe_permute"));
        b.arg(&mb.idx).arg(&mb.offsets).arg(&mb.sorted).arg(&count);
        unsafe { b.launch(cfg(1, 1024)) }?;
        Ok(())
    }

    fn moe_grouped_mma(&self, w: &DeviceMoe, mb: &MoeBuffers, n: usize) -> Result<()> {
        use crate::codegen::{MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, moe_mma_smem};
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mma = mb.mma.as_ref().expect("mma buffers");
        let block = matches!(w.gate, StackWeights::Block { .. });
        let (gw, gs, gg, gi, uw, us, ug, ui, dw, ds, dg, di) = match (&w.gate, &w.up, &w.down) {
            (
                StackWeights::Fp4 { packed: gw, scales: gs, inv_global: gg, input_global: gi },
                StackWeights::Fp4 { packed: uw, scales: us, inv_global: ug, input_global: ui },
                StackWeights::Fp4 { packed: dw, scales: ds, inv_global: dg, input_global: di },
            ) => (gw, gs, gg, *gi, uw, us, ug, *ui, dw, ds, dg, *di),
            (StackWeights::Block { packed: gw, scales: gs }, StackWeights::Block { packed: uw, scales: us }, StackWeights::Block { packed: dw, scales: ds }) => {
                (gw, gs, &self.buf.dummy, 1.0, uw, us, &self.buf.dummy, 1.0, dw, ds, &self.buf.dummy, 1.0)
            }
            _ => unreachable!("the mma path needs NVFP4 or block scaled experts"),
        };
        let stem = if block { "moe_mma_mx" } else { "moe_mma" };
        let quant_kernel = if block { "moe_quant_mx" } else { "moe_quant_f32" };
        let (h, inter) = (self.spec.dims.hidden, m.expert_intermediate);
        let slots = m.local_experts() + m.shared_slots();
        let pairs = n * m.pairs();
        let small = pairs / slots < MOE_MMA_BM / 2;
        let bm = if small { MOE_MMA_BM_SMALL } else { MOE_MMA_BM };
        let max_tiles = slots + pairs / bm;
        self.moe_permute(mb, n)?;
        let (bm_i, max_i) = (bm as i32, max_tiles as i32);
        let mut b = self.stream.launch_builder(self.kernel("moe_tiles"));
        b.arg(&mb.offsets).arg(&mma.tiles).arg(&bm_i).arg(&max_i);
        unsafe { b.launch(cfg(1, 32)) }?;
        let igs_up = gi.min(ui);
        let (up_planes, down_planes) = (self.spec.mlp.up_act_planes, self.spec.mlp.down_act_planes);
        let quant = |x: &CudaSlice<f32>, q: &CudaSlice<u8>, s: &CudaSlice<u8>, q2: &CudaSlice<u8>, s2: &CudaSlice<u8>, planes: u8, rows: usize, width: usize, igs: f32| -> Result<()> {
            let width = width as i32;
            let none = 0u64;
            let mut b = self.stream.launch_builder(self.kernel(quant_kernel));
            b.arg(x).arg(q).arg(s);
            if block {
                if planes == 2 {
                    b.arg(q2).arg(s2);
                } else {
                    b.arg(&none).arg(&none);
                }
            }
            b.arg(&width);
            if !block {
                b.arg(&igs);
            }
            unsafe { b.launch(cfg(rows as u32, 256)) }?;
            Ok(())
        };
        quant(&mb.xn, &mma.xq, &mma.xs, &mma.xq2, &mma.xs2, up_planes, n, h, igs_up)?;
        let inv_up = 1.0 / igs_up;
        let tag = |planes: u8| if planes == 2 { "2" } else { "" };
        let name = format!("{stem}_up{}{}", tag(up_planes), if small { "_s" } else { "" });
        let mut b = self.stream.launch_builder(self.kernel(&name));
        b.arg(gw).arg(gs).arg(gg).arg(uw).arg(us).arg(ug).arg(&w.gate_up_bias).arg(&inv_up).arg(&mma.xq).arg(&mma.xs);
        if block {
            b.arg(&mma.xq2).arg(&mma.xs2);
        }
        b.arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.act);
        let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(bm, 2) };
        unsafe { b.launch(config) }?;
        quant(&mb.act, &mma.aq, &mma.as_, &mma.aq2, &mma.as2, down_planes, pairs, inter, di)?;
        let inv_down = 1.0 / di;
        let name = format!("{stem}_down{}{}", tag(down_planes), if small { "_s" } else { "" });
        let mut b = self.stream.launch_builder(self.kernel(&name));
        b.arg(dw).arg(ds).arg(dg).arg(dw).arg(ds).arg(dg).arg(&w.down_bias).arg(&inv_down).arg(&mma.aq).arg(&mma.as_);
        if block {
            b.arg(&mma.aq2).arg(&mma.as2);
        }
        b.arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.out);
        let config = LaunchConfig { grid_dim: ((h / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(bm, 1) };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn moe_grouped_mmb(&self, w: &DeviceMoe, mb: &MoeBuffers, n: usize) -> Result<()> {
        use crate::codegen::{MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, moe_mma_bf16_smem};
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mma = mb.mmb.as_ref().expect("bf16 mma buffers");
        let (StackWeights::Bf16(gw), StackWeights::Bf16(uw), StackWeights::Bf16(dw)) = (&w.gate, &w.up, &w.down) else {
            unreachable!("the bf16 mma path needs bf16 experts")
        };
        let (h, inter) = (self.spec.dims.hidden, m.expert_intermediate);
        let mapped = self.spec.mlp.mapped.is_some();
        let up_weights = if mapped { 1 } else { 2 };
        let (gw, uw) = if mapped { (uw, uw) } else { (gw, uw) };
        let slots = m.local_experts() + m.shared_slots();
        let pairs = n * m.pairs();
        let small = pairs / slots < MOE_MMA_BM / 2;
        let bm = if small { MOE_MMA_BM_SMALL } else { MOE_MMA_BM };
        let max_tiles = slots + pairs / bm;
        self.moe_permute(mb, n)?;
        let (bm_i, max_i) = (bm as i32, max_tiles as i32);
        let mut b = self.stream.launch_builder(self.kernel("moe_tiles_b16"));
        b.arg(&mb.offsets).arg(&mma.tiles).arg(&bm_i).arg(&max_i);
        unsafe { b.launch(cfg(1, 32)) }?;
        let split = |x: &CudaSlice<f32>, hi: &CudaSlice<u16>, lo: &CudaSlice<u16>, rows: usize, width: usize| -> Result<()> {
            let width = width as i32;
            let mut b = self.stream.launch_builder(self.kernel("moe_split_bf16"));
            b.arg(x).arg(hi).arg(lo).arg(&width);
            unsafe { b.launch(cfg(rows as u32, 256)) }?;
            Ok(())
        };
        split(&mb.xn, &mma.xh, &mma.xl, n, h)?;
        let tag = if small { "_s" } else { "" };
        let mut b = self.stream.launch_builder(self.kernel(&format!("moe_mma_up_b16{tag}")));
        b.arg(gw).arg(uw).arg(&w.gate_up_bias).arg(&mma.xh).arg(&mma.xl).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.act);
        let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_bf16_smem(bm, up_weights) };
        unsafe { b.launch(config) }?;
        split(&mb.act, &mma.ah, &mma.al, pairs, inter)?;
        let mut b = self.stream.launch_builder(self.kernel(&format!("moe_mma_down_b16{tag}")));
        b.arg(dw).arg(dw).arg(&w.down_bias).arg(&mma.ah).arg(&mma.al).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.out);
        let config = LaunchConfig { grid_dim: ((h / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_bf16_smem(bm, 1) };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn moe_grouped_mma8(&self, w: &DeviceMoe, mb: &MoeBuffers, n: usize) -> Result<()> {
        use crate::codegen::{FP8_BLOCK, MOE_MMA_BM, MOE_MMA_BM_SMALL, MOE_MMA_BN, moe_mma8_smem};
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mma = mb.mma8.as_ref().expect("fp8 mma buffers");
        let (StackWeights::Fp8 { values: gw, grid: gg }, StackWeights::Fp8 { values: uw, grid: ug }, StackWeights::Fp8 { values: dw, grid: dg }) = (&w.gate, &w.up, &w.down) else {
            unreachable!("the fp8 mma path needs FP8 experts")
        };
        let (h, inter) = (self.spec.dims.hidden, m.expert_intermediate);
        let slots = m.local_experts() + m.shared_slots();
        let pairs = n * m.pairs();
        let small = pairs / slots < MOE_MMA_BM / 2;
        let bm = if small { MOE_MMA_BM_SMALL } else { MOE_MMA_BM };
        let max_tiles = slots + pairs / bm;
        self.moe_permute(mb, n)?;
        let (bm_i, max_i) = (bm as i32, max_tiles as i32);
        let mut b = self.stream.launch_builder(self.kernel("moe_tiles_f8"));
        b.arg(&mb.offsets).arg(&mma.tiles).arg(&bm_i).arg(&max_i);
        unsafe { b.launch(cfg(1, 32)) }?;
        let quant = |x: &CudaSlice<f32>, q: &CudaSlice<u8>, s: &CudaSlice<f32>, rows: usize, width: usize| -> Result<()> {
            let w = width as i32;
            let mut b = self.stream.launch_builder(self.kernel("moe_quant_f8"));
            b.arg(x).arg(q).arg(s).arg(&w);
            let config = LaunchConfig { grid_dim: (rows as u32, (width / FP8_BLOCK) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { b.launch(config) }?;
            Ok(())
        };
        quant(&mb.xn, &mma.xq, &mma.xs, n, h)?;
        let name = if small { "moe_mma_up_f8_s" } else { "moe_mma_up_f8" };
        let mut b = self.stream.launch_builder(self.kernel(name));
        b.arg(gw).arg(gg).arg(uw).arg(ug).arg(&mma.xq).arg(&mma.xs).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.act);
        let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma8_smem(bm, 2) };
        unsafe { b.launch(config) }?;
        quant(&mb.act, &mma.aq, &mma.as_, pairs, inter)?;
        let name = if small { "moe_mma_down_f8_s" } else { "moe_mma_down_f8" };
        let mut b = self.stream.launch_builder(self.kernel(name));
        b.arg(dw).arg(dg).arg(dw).arg(dg).arg(&mma.aq).arg(&mma.as_).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.out);
        let config = LaunchConfig { grid_dim: ((h / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma8_smem(bm, 1) };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn moe_grouped(&self, w: &DeviceMoe, mb: &MoeBuffers, n: usize) -> Result<()> {
        let m = self.spec.moe.as_ref().expect("moe spec");
        let slots = (m.local_experts() + m.shared_slots()) as u32;
        let count = (n * m.pairs()) as i32;
        let mut b = self.stream.launch_builder(self.kernel("moe_permute"));
        b.arg(&mb.idx).arg(&mb.offsets).arg(&mb.sorted).arg(&count);
        unsafe { b.launch(cfg(1, 1024)) }?;
        let kind = w.gate.suffix();
        let (up, down) = (format!("moe_group_up_{kind}"), format!("moe_group_down_{kind}"));
        let mut b = self.stream.launch_builder(self.kernel(&up));
        self.stack_args(&mut b, &w.gate);
        self.stack_args(&mut b, &w.up);
        b.arg(&w.gate_up_bias).arg(&mb.xn).arg(&mb.offsets).arg(&mb.sorted).arg(&mb.act);
        let config = LaunchConfig {
            grid_dim: (slots, m.expert_intermediate.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel(&down));
        self.stack_args(&mut b, &w.down);
        b.arg(&w.down_bias).arg(&mb.act).arg(&mb.offsets).arg(&mb.sorted).arg(&mb.out);
        let config = LaunchConfig {
            grid_dim: (slots, self.spec.dims.hidden.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    pub(super) fn moe_combine(&self, x: &CudaSlice<f32>, n: usize) -> Result<()> {
        let mb = self.moe.as_ref().expect("moe buffers");
        let mut b = self.stream.launch_builder(self.kernel("moe_combine"));
        b.arg(x).arg(&mb.out).arg(&mb.coef);
        let config = LaunchConfig {
            grid_dim: (self.spec.dims.hidden.div_ceil(256) as u32, n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    pub(super) fn moe_block(&self, layer: &DeviceLayer, x: &CudaSlice<f32>, tokens: &CudaSlice<i32>, n: usize) -> Result<()> {
        self.moe_route(layer, x, tokens, n)?;
        self.moe_experts(layer, n)?;
        self.moe_combine(x, n)
    }

    pub(super) fn mlp_rows(&self, layer: &DeviceLayer, n: usize) -> Result<()> {
        if layer.moe.is_some() { self.moe_block(layer, &self.pre.x, &self.pre.tokens, n) } else { self.mlp_prefill(layer, n) }
    }
}
