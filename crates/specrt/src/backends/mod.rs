use crate::wgsl::WgslKernel;
use anyhow::Result;

#[cfg(feature = "wgpu")]
pub mod dense;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_buffer_bytes: u64,
    pub max_binding_bytes: u64,
    pub max_storage_buffers: u32,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_storage: u32,
    pub subgroup_size: u32,
    pub shader_f16: bool,
    pub coop_matrix: bool,
}

impl DeviceLimits {
    pub fn target(&self) -> kir::Target {
        let t = kir::Target::wgpu(self.max_workgroup_invocations, self.max_workgroup_storage);
        let t = if self.subgroup_size > 0 { t.with_subgroups(self.subgroup_size) } else { t };
        let t = if self.shader_f16 { t.with_feature(kir::Feature::ShaderF16) } else { t };
        if self.coop_matrix { t.with_feature(kir::Feature::CoopMatrix) } else { t }
    }
}

pub trait Backend {
    type Buffer;

    fn description(&self) -> String;
    fn limits(&self) -> DeviceLimits;
    fn alloc(&self, bytes: u64) -> Result<Self::Buffer>;
    fn upload(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);
    fn download(&self, buffer: &Self::Buffer, offset: u64, len: u64) -> Result<Vec<u8>>;
    fn compile(&mut self, kernels: &[WgslKernel]) -> Result<()>;
    fn dispatch(&mut self, kernel: &str, buffers: &[&Self::Buffer], params: [u32; 8], grid: [u32; 3]) -> Result<()>;
    fn submit(&mut self) -> Result<()>;
}
