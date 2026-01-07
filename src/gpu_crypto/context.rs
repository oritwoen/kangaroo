//! GPU context and device management

use anyhow::{Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
}

impl GpuContext {
    pub async fn new(device_index: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let mut adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        // Prefer discrete > virtual > integrated > cpu/other
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::VirtualGpu => 1,
            wgpu::DeviceType::IntegratedGpu => 2,
            wgpu::DeviceType::Cpu => 3,
            _ => 4,
        });

        if adapters.is_empty() {
            anyhow::bail!("No GPU adapters found");
        }

        let adapter = adapters
            .into_iter()
            .nth(device_index as usize)
            .context("GPU device index out of range")?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpu-crypto"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                // memory_hints is a newer feature (wgpu 22+), ensuring we use recent version
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("Failed to create GPU device")?;

        let limits = device.limits();

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    pub fn max_workgroup_size(&self) -> u32 {
        self.limits.max_compute_workgroup_size_x
    }

    pub fn max_workgroups(&self) -> u32 {
        self.limits.max_compute_workgroups_per_dimension
    }

    /// Optimal batch size heuristic
    pub fn optimal_batch_size(&self) -> u32 {
        // Heuristic: use large batches for better GPU utilization
        let workgroup_size = 64u32;
        // Conservative limit to prevent TDR (Timeout Detection and Recovery) on Windows
        // or just freezing the screen on Linux
        let workgroups = self.max_workgroups().min(65535).min(4096);
        workgroup_size * workgroups
    }

    pub fn compute_units(&self) -> u32 {
        self.max_workgroups()
    }

    pub fn optimal_kangaroos(&self) -> u32 {
        self.optimal_batch_size()
    }

    pub fn optimal_steps_per_call(&self) -> u32 {
        // Extremely conservative default for heavy shaders
        16
    }

    /// Create an uninitialized buffer of type T with count elements
    pub fn create_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        count: u64,
    ) -> wgpu::Buffer {
        let size = count * std::mem::size_of::<T>() as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a buffer initialized with data
    pub fn create_buffer_init<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        data: &[T],
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Create a shader module from multiple source strings
    pub fn create_shader_module(&self, label: &str, sources: &[&str]) -> wgpu::ShaderModule {
        let source = sources.join("\n\n");
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    /// Read data from a mappable buffer (async)
    pub async fn read_buffer<T: bytemuck::Pod + Clone>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        count: u64,
    ) -> Result<Vec<T>> {
        let size = count * std::mem::size_of::<T>() as u64;
        let slice = buffer.slice(offset..offset + size);

        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.await??;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(result)
    }
}
