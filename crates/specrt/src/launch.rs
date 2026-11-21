use cudarc::driver::sys::{self, CUdeviceptr};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DriverError, LaunchConfig};
use std::ffi::c_void;

pub struct Launch<'a> {
    stream: &'a CudaStream,
    func: &'a CudaFunction,
    args: Vec<*mut c_void>,
    #[allow(clippy::vec_box)]
    ptrs: Vec<Box<CUdeviceptr>>,
    pdl: bool,
    small_grid: u32,
}

impl<'a> Launch<'a> {
    pub fn new(stream: &'a CudaStream, func: &'a CudaFunction, pdl: bool, small_grid: u32) -> Self {
        Launch { stream, func, args: Vec::new(), ptrs: Vec::new(), pdl, small_grid }
    }

    fn push_ptr(&mut self, ptr: CUdeviceptr) {
        let boxed = Box::new(ptr);
        self.args.push(&*boxed as *const CUdeviceptr as *mut c_void);
        self.ptrs.push(boxed);
    }

    /// # Safety
    /// The arguments must match the kernel signature.
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut attr: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
        attr.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
        let programmatic = self.pdl && cfg.grid_dim.0 * cfg.grid_dim.1 * cfg.grid_dim.2 <= self.small_grid;
        let config = sys::CUlaunchConfig {
            gridDimX: cfg.grid_dim.0,
            gridDimY: cfg.grid_dim.1,
            gridDimZ: cfg.grid_dim.2,
            blockDimX: cfg.block_dim.0,
            blockDimY: cfg.block_dim.1,
            blockDimZ: cfg.block_dim.2,
            sharedMemBytes: cfg.shared_mem_bytes,
            hStream: self.stream.cu_stream(),
            attrs: if programmatic { &mut attr } else { std::ptr::null_mut() },
            numAttrs: programmatic as u32,
        };
        unsafe { sys::cuLaunchKernelEx(&config, self.func.cu_function(), self.args.as_mut_ptr(), std::ptr::null_mut()) }.result()
    }
}

pub trait LaunchArg<'b> {
    fn push(self, l: &mut Launch<'_>);
}

macro_rules! scalar {
    ($($t:ty),*) => {$(
        impl<'b> LaunchArg<'b> for &'b $t {
            fn push(self, l: &mut Launch<'_>) {
                l.args.push(self as *const $t as *mut c_void);
            }
        }
    )*};
}

scalar!(i32, u32, i64, u64, f32, usize, u8, u16);

impl<'b, T> LaunchArg<'b> for &'b CudaSlice<T> {
    fn push(self, l: &mut Launch<'_>) {
        let (ptr, _guard) = self.device_ptr(l.stream);
        l.push_ptr(ptr);
    }
}

impl<'b, T> LaunchArg<'b> for &'b mut CudaSlice<T> {
    fn push(self, l: &mut Launch<'_>) {
        let (ptr, _guard) = self.device_ptr(l.stream);
        l.push_ptr(ptr);
    }
}

impl<'b, 'c, T> LaunchArg<'b> for &'b CudaView<'c, T> {
    fn push(self, l: &mut Launch<'_>) {
        let (ptr, _guard) = self.device_ptr(l.stream);
        l.push_ptr(ptr);
    }
}

impl<'b, 'c, T> LaunchArg<'b> for &'b mut CudaViewMut<'c, T> {
    fn push(self, l: &mut Launch<'_>) {
        let (ptr, _guard) = self.device_ptr(l.stream);
        l.push_ptr(ptr);
    }
}

impl<'a> Launch<'a> {
    pub fn arg<'b: 'a, A: LaunchArg<'b>>(&mut self, a: A) -> &mut Self {
        a.push(self);
        self
    }
}
