use super::{Backend, DeviceLimits};
use crate::wgsl::{Access, WgslKernel};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::PathBuf;

const PARAM_SLOT: u64 = 256;
const PARAM_SLOTS: u64 = 8192;

pub struct Buffer {
    pub raw: wgpu::Buffer,
    pub bytes: u64,
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bindings: Vec<Access>,
}

pub struct Wgpu {
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: DeviceLimits,
    pipelines: HashMap<&'static str, Pipeline>,
    params: wgpu::Buffer,
    slot: u64,
    encoder: Option<wgpu::CommandEncoder>,
    cache_dir: Option<PathBuf>,
    cache: Option<wgpu::PipelineCache>,
    cache_path: Option<PathBuf>,
}

pub fn adapters() -> Vec<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())).iter().map(|a| a.get_info()).collect()
}

impl Wgpu {
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::select(cache_dir, false)
    }

    pub fn allow_software(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::select(cache_dir, true)
    }

    fn select(cache_dir: Option<PathBuf>, software: bool) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut found = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        found.retain(|a| software || a.get_info().device_type != wgpu::DeviceType::Cpu);
        found.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        });
        let adapter = found.into_iter().next().ok_or_else(|| anyhow!("no GPU adapter is available to wgpu (Vulkan, Metal or DX12)"))?;
        let info = adapter.get_info();
        let have = adapter.limits();
        let mut required = wgpu::Limits::downlevel_defaults();
        required.max_buffer_size = have.max_buffer_size;
        required.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size;
        required.max_storage_buffers_per_shader_stage = have.max_storage_buffers_per_shader_stage.min(16);
        required.max_compute_workgroup_storage_size = have.max_compute_workgroup_storage_size;
        required.max_compute_invocations_per_workgroup = have.max_compute_invocations_per_workgroup;
        required.max_compute_workgroup_size_x = have.max_compute_workgroup_size_x;
        required.max_compute_workgroups_per_dimension = have.max_compute_workgroups_per_dimension;
        let wanted_cache = adapter.features() & wgpu::Features::PIPELINE_CACHE;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("specrt"),
            required_features: wanted_cache,
            required_limits: required.clone(),
            ..Default::default()
        }))
        .with_context(|| format!("wgpu device on {}", info.name))?;
        let limits = DeviceLimits {
            max_buffer_bytes: required.max_buffer_size,
            max_binding_bytes: required.max_storage_buffer_binding_size as u64,
            max_storage_buffers: required.max_storage_buffers_per_shader_stage,
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_SLOT * PARAM_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (cache, cache_path) = match (&cache_dir, wanted_cache.is_empty()) {
            (Some(dir), false) => {
                let _ = std::fs::create_dir_all(dir);
                let path = dir.join(format!("wgpu-{:x}-{:x}.cache", info.vendor, info.device));
                let data = std::fs::read(&path).ok();
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("specrt"), data: data.as_deref(), fallback: true })
                };
                (Some(cache), Some(path))
            }
            _ => (None, None),
        };
        Ok(Wgpu { info, device, queue, limits, pipelines: HashMap::new(), params, slot: 0, encoder: None, cache_dir, cache, cache_path })
    }

    pub fn info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    fn save_cache(&self) {
        if let (Some(cache), Some(path)) = (&self.cache, &self.cache_path)
            && let Some(data) = cache.get_data()
        {
            let _ = std::fs::write(path, data);
        }
    }

    fn source_path(&self, kernel: &WgslKernel) -> Option<PathBuf> {
        let dir = self.cache_dir.as_ref()?;
        let options = vec![self.info.name.clone(), self.info.driver.clone()];
        Some(dir.join(format!("{}.wgsl", crate::jit::cache_key(&kernel.source, "wgsl", &options))))
    }
}

impl Drop for Wgpu {
    fn drop(&mut self) {
        self.save_cache();
    }
}

impl Backend for Wgpu {
    type Buffer = Buffer;

    fn description(&self) -> String {
        format!("{} ({:?}, {:?})", self.info.name, self.info.backend, self.info.device_type)
    }

    fn limits(&self) -> DeviceLimits {
        self.limits
    }

    fn alloc(&self, bytes: u64) -> Result<Buffer> {
        let padded = bytes.max(16).next_multiple_of(4);
        if padded > self.limits.max_buffer_bytes {
            bail!("a {padded} byte buffer exceeds max_buffer_size {} on {}", self.limits.max_buffer_bytes, self.info.name);
        }
        if padded > self.limits.max_binding_bytes {
            bail!("a {padded} byte buffer exceeds max_storage_buffer_binding_size {} on {}", self.limits.max_binding_bytes, self.info.name);
        }
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Buffer { raw, bytes: padded })
    }

    fn upload(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        if data.len().is_multiple_of(4) {
            self.queue.write_buffer(&buffer.raw, offset, data);
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().next_multiple_of(4), 0);
            self.queue.write_buffer(&buffer.raw, offset, &padded);
        }
    }

    fn download(&self, buffer: &Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        let padded = len.next_multiple_of(4);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&buffer.raw, offset, &staging, 0, padded);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("wgpu poll: {e:?}"))?;
        rx.recv()??;
        let data = staging.slice(..).get_mapped_range().map_err(|e| anyhow!("map: {e:?}"))?;
        let out = data[..len as usize].to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    fn compile(&mut self, kernels: &[WgslKernel]) -> Result<()> {
        for k in kernels {
            let storage = k.bindings.iter().filter(|a| **a != Access::Uniform).count() as u32;
            if storage > self.limits.max_storage_buffers {
                bail!("kernel {} needs {storage} storage buffers, this adapter allows {}", k.name, self.limits.max_storage_buffers);
            }
            if let Some(path) = self.source_path(k)
                && !path.exists()
            {
                let _ = std::fs::create_dir_all(path.parent().expect("cache file has a parent"));
                let _ = std::fs::write(&path, &k.source);
            }
            let entries: Vec<wgpu::BindGroupLayoutEntry> = k
                .bindings
                .iter()
                .enumerate()
                .map(|(i, a)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match a {
                            Access::Read => wgpu::BufferBindingType::Storage { read_only: true },
                            Access::Write => wgpu::BufferBindingType::Storage { read_only: false },
                            Access::Uniform => wgpu::BufferBindingType::Uniform,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(k.name), entries: &entries });
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(k.name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(k.name),
                source: wgpu::ShaderSource::Wgsl(k.source.as_str().into()),
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k.name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: self.cache.as_ref(),
            });
            if let Some(e) = pollster::block_on(scope.pop()) {
                bail!("kernel {} failed to compile on {}: {e}", k.name, self.info.name);
            }
            self.pipelines.insert(k.name, Pipeline { pipeline, layout, bindings: k.bindings.clone() });
        }
        self.save_cache();
        Ok(())
    }

    fn dispatch(&mut self, kernel: &str, buffers: &[&Buffer], params: [u32; 8], grid: [u32; 3]) -> Result<()> {
        if self.slot == PARAM_SLOTS {
            self.submit()?;
        }
        let slot = self.slot;
        self.slot += 1;
        let bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&self.params, slot * PARAM_SLOT, &bytes);
        let p = self.pipelines.get(kernel).with_context(|| format!("kernel {kernel} was not compiled"))?;
        let mut data = buffers.iter();
        let entries: Vec<wgpu::BindGroupEntry> = p
            .bindings
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let resource = if *a == Access::Uniform {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: slot * PARAM_SLOT,
                        size: std::num::NonZeroU64::new(32),
                    })
                } else {
                    data.next().expect("a buffer for each storage binding").raw.as_entire_binding()
                };
                wgpu::BindGroupEntry { binding: i as u32, resource }
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.layout, entries: &entries });
        let pipeline = p.pipeline.clone();
        let device = &self.device;
        let encoder = self.encoder.get_or_insert_with(|| device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("specrt") }));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("specrt"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        if let Some(e) = self.encoder.take() {
            self.queue.submit([e.finish()]);
        }
        self.slot = 0;
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("wgpu poll: {e:?}"))?;
        Ok(())
    }
}
