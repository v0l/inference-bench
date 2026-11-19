use super::{DeviceLayer, Runtime, cfg};
use crate::weights::{MoeWeights, Stacked};
use anyhow::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub(super) enum StackWeights {
    Bf16(CudaSlice<u16>),
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8>, inv_global: CudaSlice<f32>, input_global: f32 },
    Fp8 { values: CudaSlice<u8>, grid: CudaSlice<f32> },
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
            StackWeights::Fp4 { packed, scales, inv_global, .. } => packed.num_bytes() + scales.num_bytes() + inv_global.num_bytes(),
        }) as u64
    }

    fn fp4(&self) -> bool {
        matches!(self, StackWeights::Fp4 { .. })
    }

    fn suffix(&self) -> &'static str {
        match self {
            StackWeights::Bf16(_) => "bf16",
            StackWeights::Fp4 { .. } => "fp4",
            StackWeights::Fp8 { .. } => "fp8",
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
}

struct MmaBuffers {
    xq: CudaSlice<u8>,
    xs: CudaSlice<u8>,
    aq: CudaSlice<u8>,
    as_: CudaSlice<u8>,
    tiles: CudaSlice<i32>,
}

pub const MMA_MIN_TOKENS: usize = 8;

pub const GROUP_MIN_TOKENS: usize = 17;

impl MoeBuffers {
    pub(super) fn new(stream: &Arc<cudarc::driver::CudaStream>, rows: usize, spec: &crate::spec::Spec, mma: bool) -> Result<Self> {
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
                let (h, i) = (spec.dims.hidden, m.expert_intermediate);
                let tiles = m.experts + 1 + pairs / crate::codegen::MOE_MMA_BM_SMALL;
                Some(MmaBuffers {
                    xq: stream.alloc_zeros(rows * h / 2)?,
                    xs: stream.alloc_zeros(rows * h / 16)?,
                    aq: stream.alloc_zeros(pairs * i / 2)?,
                    as_: stream.alloc_zeros(pairs * i / 16)?,
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
        let mut b = self.stream.launch_builder(self.kernel("moe_route_logits"));
        b.arg(x).arg(&layer.ln2).arg(&w.router).arg(&w.shared_gate).arg(&w.router_bias).arg(&mb.xn).arg(&mb.logits);
        let config = LaunchConfig { grid_dim: ((experts + 1).div_ceil(8) as u32, n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
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
        };
    }

    pub(super) fn moe_experts(&self, layer: &DeviceLayer, n: usize) -> Result<()> {
        let w = self.moe_of(layer);
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mb = self.moe.as_ref().expect("moe buffers");
        let pairs = (n * m.pairs()) as u32;
        let fp4 = w.gate.fp4();
        if mb.mma.is_some() && fp4 && n >= MMA_MIN_TOKENS {
            return self.moe_grouped_mma(w, mb, n);
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
        let (StackWeights::Fp4 { packed: gw, scales: gs, inv_global: gg, input_global: gi }, StackWeights::Fp4 { packed: uw, scales: us, inv_global: ug, input_global: ui }, StackWeights::Fp4 { packed: dw, scales: ds, inv_global: dg, input_global: di }) =
            (&w.gate, &w.up, &w.down)
        else {
            unreachable!("the mma path needs NVFP4 experts")
        };
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
        let igs_up = gi.min(*ui);
        let quant = |x: &CudaSlice<f32>, q: &CudaSlice<u8>, s: &CudaSlice<u8>, rows: usize, width: usize, igs: f32| -> Result<()> {
            let width = width as i32;
            let mut b = self.stream.launch_builder(self.kernel("moe_quant_f32"));
            b.arg(x).arg(q).arg(s).arg(&width).arg(&igs);
            unsafe { b.launch(cfg(rows as u32, 256)) }?;
            Ok(())
        };
        quant(&mb.xn, &mma.xq, &mma.xs, n, h, igs_up)?;
        let inv_up = 1.0 / igs_up;
        let name = if small { "moe_mma_up_s" } else { "moe_mma_up" };
        let mut b = self.stream.launch_builder(self.kernel(name));
        b.arg(gw).arg(gs).arg(gg).arg(uw).arg(us).arg(ug).arg(&inv_up).arg(&mma.xq).arg(&mma.xs).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.act);
        let config = LaunchConfig { grid_dim: ((inter / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(bm, 2) };
        unsafe { b.launch(config) }?;
        quant(&mb.act, &mma.aq, &mma.as_, pairs, inter, *di)?;
        let inv_down = 1.0 / *di;
        let name = if small { "moe_mma_down_s" } else { "moe_mma_down" };
        let mut b = self.stream.launch_builder(self.kernel(name));
        b.arg(dw).arg(ds).arg(dg).arg(dw).arg(ds).arg(dg).arg(&inv_down).arg(&mma.aq).arg(&mma.as_).arg(&mb.offsets).arg(&mb.sorted).arg(&mma.tiles).arg(&mb.out);
        let config = LaunchConfig { grid_dim: ((h / MOE_MMA_BN) as u32, max_tiles as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: moe_mma_smem(bm, 1) };
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
