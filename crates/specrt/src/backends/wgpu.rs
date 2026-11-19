use super::{Backend, DeviceLimits};
use crate::wgsl::{Access, WgslKernel};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::PathBuf;

const PARAM_SLOT: u64 = 256;
const PARAM_SLOTS: u64 = 8192;
const UPLOAD_FLUSH: u64 = 128 << 20;

pub struct Buffer {
    pub raw: wgpu::Buffer,
    pub bytes: u64,
    id: u64,
}

fn next_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bindings: Vec<Access>,
    index: u64,
}

pub struct Wgpu {
    info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: DeviceLimits,
    pipelines: HashMap<&'static str, Pipeline>,
    params: wgpu::Buffer,
    slot: u64,
    staged: Vec<u8>,
    groups: HashMap<Vec<u64>, wgpu::BindGroup>,
    dispatched: u64,
    counts: std::collections::BTreeMap<String, u64>,
    pending_upload: std::cell::Cell<u64>,
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
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
        let wanted_subgroups = adapter.features() & wgpu::Features::SUBGROUP;
        let apple = info.backend == wgpu::Backend::Metal && info.subgroup_min_size <= 32 && info.subgroup_max_size >= 32;
        let subgroups = !wanted_subgroups.is_empty() && info.subgroup_min_size > 0 && (info.subgroup_min_size == info.subgroup_max_size || apple);
        let f16 = adapter.features() & wgpu::Features::SHADER_F16;
        let coop = adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter.cooperative_matrix_properties().iter().any(|c| {
                c.m_size == 8 && c.n_size == 8 && c.k_size == 8 && c.ab_type == wgpu::CooperativeScalarType::F32 && c.cr_type == wgpu::CooperativeScalarType::F32
            });
        let coop = coop && std::env::var_os("SPECRT_NO_COOP").is_none();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("specrt"),
            required_features: wanted_cache | f16 | if coop { wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX } else { wgpu::Features::empty() } | if subgroups { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() },
            required_limits: required.clone(),
            experimental_features: if coop { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
            ..Default::default()
        }))
        .with_context(|| format!("wgpu device on {}", info.name))?;
        device.set_device_lost_callback(|r, m| tracing::warn!("wgpu device lost {r:?}: {m}"));
        device.on_uncaptured_error(std::sync::Arc::new(|e| tracing::error!("wgpu: {e}")));
        let limits = DeviceLimits {
            max_buffer_bytes: required.max_buffer_size,
            max_binding_bytes: required.max_storage_buffer_binding_size,
            max_storage_buffers: required.max_storage_buffers_per_shader_stage,
            max_workgroup_invocations: required.max_compute_invocations_per_workgroup,
            max_workgroup_storage: required.max_compute_workgroup_storage_size,
            subgroup_size: if !subgroups { 0 } else if apple { 32 } else { info.subgroup_min_size },
            shader_f16: !f16.is_empty(),
            coop_matrix: coop,
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
        Ok(Wgpu { info, device, queue, limits, pipelines: HashMap::new(), params, slot: 0, staged: Vec::new(), groups: HashMap::new(), dispatched: 0, counts: Default::default(), pending_upload: Default::default(), encoder: None, pass: None, cache_dir, cache, cache_path })
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
        if std::env::var_os("SPECRT_COUNT").is_some() {
            eprintln!("dispatches {} {:?} groups {}", self.dispatched, self.counts, self.groups.len());
        }
        self.save_cache();
    }
}

const COPY_PIECE: usize = 8 << 20;

struct Dst(*mut u8);

unsafe impl Sync for Dst {}

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
        Ok(Buffer { raw, bytes: padded, id: next_id() })
    }

    fn alloc_filled(&self, parts: &[&[u8]]) -> Result<Buffer> {
        let bytes: u64 = parts.iter().map(|p| p.len() as u64).sum();
        let padded = bytes.max(16).next_multiple_of(4);
        if padded > self.limits.max_buffer_bytes || padded > self.limits.max_binding_bytes {
            bail!("a {padded} byte buffer exceeds this adapter's buffer limits on {}", self.info.name);
        }
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = raw.slice(..).get_mapped_range_mut().map_err(|e| anyhow!("map: {e:?}"))?;
            let mut whole = view.slice(..);
            let dst = Dst(whole.as_raw_ptr().as_ptr().cast::<u8>());
            let mut jobs: Vec<(usize, &[u8])> = Vec::new();
            let mut at = 0usize;
            for part in parts {
                for (i, piece) in part.chunks(COPY_PIECE).enumerate() {
                    jobs.push((at + i * COPY_PIECE, piece));
                }
                at += part.len();
            }
            let next = std::sync::atomic::AtomicUsize::new(0);
            let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| {
                        let dst = &dst;
                        loop {
                            let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((offset, piece)) = jobs.get(j) else { break };
                            unsafe { std::ptr::copy_nonoverlapping(piece.as_ptr(), dst.0.add(*offset), piece.len()) };
                        }
                    });
                }
            });
        }
        raw.unmap();
        self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(Buffer { raw, bytes: padded, id: next_id() })
    }

    fn upload(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        let pending = self.pending_upload.get() + data.len() as u64;
        if pending >= UPLOAD_FLUSH {
            self.queue.submit([]);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.pending_upload.set(0);
        } else {
            self.pending_upload.set(pending);
        }
        if data.len().is_multiple_of(4) {
            self.queue.write_buffer(&buffer.raw, offset, data);
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().next_multiple_of(4), 0);
            self.queue.write_buffer(&buffer.raw, offset, &padded);
        }
    }

    fn download(&self, buffer: &Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        let padded = len.next_multiple_of(4).max(16);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&buffer.raw, offset, &staging, 0, padded.min(buffer.bytes - offset));
        self.queue.submit([encoder.finish()]);
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("reading back {padded} bytes on {} failed: {e} (max_buffer_size {})", self.info.name, self.limits.max_buffer_bytes);
        }
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
                        has_dynamic_offset: *a == Access::Uniform,
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
            let descriptor = wgpu::ShaderModuleDescriptor {
                label: Some(k.name),
                source: wgpu::ShaderSource::Wgsl(k.source.as_str().into()),
            };
            let module = if std::env::var_os("SPECRT_CHECKED").is_some() {
                self.device.create_shader_module(descriptor)
            } else {
                unsafe { self.device.create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked()) }
            };
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k.name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: false, ..Default::default() },
                cache: self.cache.as_ref(),
            });
            if let Some(e) = pollster::block_on(scope.pop()) {
                bail!("kernel {} failed to compile on {}: {e}", k.name, self.info.name);
            }
            self.pipelines.insert(k.name, Pipeline { pipeline, layout, bindings: k.bindings.clone(), index: self.pipelines.len() as u64 });
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
        self.dispatched += 1;
        *self.counts.entry(kernel.to_string()).or_insert(0) += 1;
        self.staged.extend(params.iter().flat_map(|v| v.to_le_bytes()));
        self.staged.resize((slot + 1) as usize * PARAM_SLOT as usize, 0);
        let p = self.pipelines.get(kernel).with_context(|| format!("kernel {kernel} was not compiled"))?;
        let mut key = Vec::with_capacity(buffers.len() + 1);
        key.push(p.index);
        key.extend(buffers.iter().map(|b| b.id));
        let group = self.groups.entry(key).or_insert_with(|| {
            let mut data = buffers.iter();
            let entries: Vec<wgpu::BindGroupEntry> = p
                .bindings
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let resource = if *a == Access::Uniform {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: std::num::NonZeroU64::new(32) })
                    } else {
                        data.next().expect("a buffer for each storage binding").raw.as_entire_binding()
                    };
                    wgpu::BindGroupEntry { binding: i as u32, resource }
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.layout, entries: &entries })
        });
        if self.pass.is_none() {
            let device = &self.device;
            let encoder = self.encoder.get_or_insert_with(|| device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("specrt") }));
            self.pass = Some(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("specrt"), timestamp_writes: None }).forget_lifetime());
        }
        let pass = self.pass.as_mut().expect("a compute pass is open");
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &*group, &[(slot * PARAM_SLOT) as u32]);
        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.pass = None;
        if !self.staged.is_empty() {
            self.queue.write_buffer(&self.params, 0, &self.staged);
            self.staged.clear();
        }
        if let Some(e) = self.encoder.take() {
            self.queue.submit([e.finish()]);
        }
        self.slot = 0;
        if let Some(e) = pollster::block_on(scope.pop()) {
            bail!("submitting work on {} failed: {e}", self.info.name);
        }
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("wgpu poll: {e:?}"))?;
        Ok(())
    }
}
