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
