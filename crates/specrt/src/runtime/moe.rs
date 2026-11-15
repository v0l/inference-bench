use super::{DeviceLayer, Runtime, cfg};
use crate::weights::{MoeWeights, Stacked};
use anyhow::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

pub(super) enum StackWeights {
    Bf16(CudaSlice<u16>),
    Fp4 { packed: CudaSlice<u8>, scales: CudaSlice<u8>, inv_global: CudaSlice<f32> },
}

pub(super) struct DeviceMoe {
    router: CudaSlice<u16>,
    shared_gate: CudaSlice<f32>,
    gate: StackWeights,
    up: StackWeights,
    down: StackWeights,
}

impl StackWeights {
    fn upload(stream: &Arc<cudarc::driver::CudaStream>, s: &Stacked) -> Result<Self> {
        Ok(match s {
            Stacked::Bf16(w) => StackWeights::Bf16(stream.clone_htod(w)?),
            Stacked::Fp4 { packed, scales, inv_global } => StackWeights::Fp4 {
                packed: stream.clone_htod(packed)?,
                scales: stream.clone_htod(scales)?,
                inv_global: stream.clone_htod(inv_global)?,
            },
        })
    }

    fn bytes(&self) -> u64 {
        (match self {
            StackWeights::Bf16(w) => w.num_bytes(),
            StackWeights::Fp4 { packed, scales, inv_global } => packed.num_bytes() + scales.num_bytes() + inv_global.num_bytes(),
        }) as u64
    }

    fn fp4(&self) -> bool {
        matches!(self, StackWeights::Fp4 { .. })
    }
}

impl DeviceMoe {
    pub(super) fn upload(stream: &Arc<cudarc::driver::CudaStream>, m: &MoeWeights) -> Result<Self> {
        Ok(DeviceMoe {
            router: stream.clone_htod(&m.router)?,
            shared_gate: stream.clone_htod(m.shared_gate.as_deref().unwrap_or(&[0.0]))?,
            gate: StackWeights::upload(stream, &m.gate)?,
            up: StackWeights::upload(stream, &m.up)?,
            down: StackWeights::upload(stream, &m.down)?,
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        (self.router.num_bytes() + self.shared_gate.num_bytes()) as u64 + self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

pub(super) struct MoeBuffers {
    xn: CudaSlice<f32>,
    idx: CudaSlice<i32>,
    coef: CudaSlice<f32>,
    act: CudaSlice<f32>,
    out: CudaSlice<f32>,
    offsets: CudaSlice<i32>,
    sorted: CudaSlice<i32>,
}

pub const GROUP_MIN_TOKENS: usize = 17;

impl MoeBuffers {
    pub(super) fn new(stream: &Arc<cudarc::driver::CudaStream>, rows: usize, spec: &crate::spec::Spec) -> Result<Self> {
        let m = spec.moe.as_ref().expect("moe buffers need [moe]");
        let pairs = rows * m.pairs();
        Ok(MoeBuffers {
            xn: stream.alloc_zeros(rows * spec.dims.hidden)?,
            idx: stream.alloc_zeros(pairs)?,
            coef: stream.alloc_zeros(pairs)?,
            act: stream.alloc_zeros(pairs * m.expert_intermediate)?,
            out: stream.alloc_zeros(pairs * spec.dims.hidden)?,
            offsets: stream.alloc_zeros(m.experts + 2)?,
            sorted: stream.alloc_zeros(pairs)?,
        })
    }
}

const ROWS_PER_BLOCK: usize = 8;

impl Runtime {
    pub(super) fn moe_of<'a>(&self, layer: &'a DeviceLayer) -> &'a DeviceMoe {
        layer.moe.as_ref().expect("moe op on a dense layer")
    }

    pub(super) fn moe_route(&self, layer: &DeviceLayer, x: &CudaSlice<f32>, n: usize) -> Result<()> {
        let w = self.moe_of(layer);
        let mb = self.moe.as_ref().expect("moe buffers");
        let mut b = self.stream.launch_builder(self.kernel("moe_route"));
        b.arg(x).arg(&layer.ln2).arg(&w.router).arg(&w.shared_gate).arg(&mb.xn).arg(&mb.idx).arg(&mb.coef);
        unsafe { b.launch(cfg(n as u32, 256)) }?;
        Ok(())
    }

    fn stack_args<'a>(&'a self, b: &mut cudarc::driver::LaunchArgs<'a>, s: &'a StackWeights) {
        match s {
            StackWeights::Bf16(w) => b.arg(w).arg(&self.pre.dummy_u8).arg(&self.buf.dummy),
            StackWeights::Fp4 { packed, scales, inv_global } => b.arg(packed).arg(scales).arg(inv_global),
        };
    }

    pub(super) fn moe_experts(&self, layer: &DeviceLayer, n: usize) -> Result<()> {
        let w = self.moe_of(layer);
        let m = self.spec.moe.as_ref().expect("moe spec");
        let mb = self.moe.as_ref().expect("moe buffers");
        let pairs = (n * m.pairs()) as u32;
        let fp4 = w.gate.fp4();
        if n >= GROUP_MIN_TOKENS {
            return self.moe_grouped(w, mb, n, fp4);
        }
        let up = if fp4 { "moe_up_fp4" } else { "moe_up_bf16" };
        let down = if fp4 { "moe_down_fp4" } else { "moe_down_bf16" };
        let mut b = self.stream.launch_builder(self.kernel(up));
        self.stack_args(&mut b, &w.gate);
        self.stack_args(&mut b, &w.up);
        b.arg(&mb.xn).arg(&mb.idx).arg(&mb.act);
        let config = LaunchConfig {
            grid_dim: (pairs, m.expert_intermediate.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel(down));
        self.stack_args(&mut b, &w.down);
        b.arg(&mb.act).arg(&mb.idx).arg(&mb.out);
        let config = LaunchConfig {
            grid_dim: (pairs, self.spec.dims.hidden.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        Ok(())
    }

    fn moe_grouped(&self, w: &DeviceMoe, mb: &MoeBuffers, n: usize, fp4: bool) -> Result<()> {
        let m = self.spec.moe.as_ref().expect("moe spec");
        let slots = (m.experts + usize::from(m.shared_intermediate > 0)) as u32;
        let count = (n * m.pairs()) as i32;
        let mut b = self.stream.launch_builder(self.kernel("moe_permute"));
        b.arg(&mb.idx).arg(&mb.offsets).arg(&mb.sorted).arg(&count);
        unsafe { b.launch(cfg(1, 1024)) }?;
        let (up, down) = if fp4 { ("moe_group_up_fp4", "moe_group_down_fp4") } else { ("moe_group_up_bf16", "moe_group_down_bf16") };
        let mut b = self.stream.launch_builder(self.kernel(up));
        self.stack_args(&mut b, &w.gate);
        self.stack_args(&mut b, &w.up);
        b.arg(&mb.xn).arg(&mb.offsets).arg(&mb.sorted).arg(&mb.act);
        let config = LaunchConfig {
            grid_dim: (slots, m.expert_intermediate.div_ceil(ROWS_PER_BLOCK) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }?;
        let mut b = self.stream.launch_builder(self.kernel(down));
        self.stack_args(&mut b, &w.down);
        b.arg(&mb.act).arg(&mb.offsets).arg(&mb.sorted).arg(&mb.out);
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

    pub(super) fn moe_block(&self, layer: &DeviceLayer, x: &CudaSlice<f32>, n: usize) -> Result<()> {
        self.moe_route(layer, x, n)?;
        self.moe_experts(layer, n)?;
        self.moe_combine(x, n)
    }

    pub(super) fn mlp_rows(&self, layer: &DeviceLayer, n: usize) -> Result<()> {
        if layer.moe.is_some() { self.moe_block(layer, &self.pre.x, n) } else { self.mlp_prefill(layer, n) }
    }
}
