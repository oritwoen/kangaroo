//! GPU context and device management

use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use std::sync::Arc;
use tracing::{debug, info};
use wgpu::util::DeviceExt;

/// GPU backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum GpuBackend {
    /// Automatically select best available backend (Vulkan → Metal → DX12 → GL)
    #[default]
    Auto,
    /// Vulkan backend (Linux, Windows, Android)
    Vulkan,
    /// DirectX 12 backend (Windows only)
    Dx12,
    /// Metal backend (macOS, iOS)
    Metal,
    /// OpenGL backend (fallback)
    Gl,
}

impl GpuBackend {
    /// Convert to wgpu::Backends bitflag
    pub fn to_wgpu_backends(self) -> wgpu::Backends {
        match self {
            GpuBackend::Auto => wgpu::Backends::all(),
            GpuBackend::Vulkan => wgpu::Backends::VULKAN,
            GpuBackend::Dx12 => wgpu::Backends::DX12,
            GpuBackend::Metal => wgpu::Backends::METAL,
            GpuBackend::Gl => wgpu::Backends::GL,
        }
    }

    /// Fallback order for Auto mode
    pub fn fallback_order() -> &'static [GpuBackend] {
        &[
            GpuBackend::Vulkan,
            GpuBackend::Metal,
            GpuBackend::Dx12,
            GpuBackend::Gl,
        ]
    }

    /// Human-readable name for logging
    pub fn name(self) -> &'static str {
        match self {
            GpuBackend::Auto => "auto",
            GpuBackend::Vulkan => "Vulkan",
            GpuBackend::Dx12 => "DX12",
            GpuBackend::Metal => "Metal",
            GpuBackend::Gl => "OpenGL",
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
}

impl GpuContext {
    /// Create GPU context with specified backend
    pub async fn new(device_index: u32, backend: GpuBackend) -> Result<Self> {
        match backend {
            GpuBackend::Auto => Self::new_with_fallback(device_index).await,
            _ => Self::new_with_backend(device_index, backend).await,
        }
    }

    /// Create GPU context trying backends in fallback order
    async fn new_with_fallback(device_index: u32) -> Result<Self> {
        // First pass: try to find hardware GPU, skip software renderers
        for &backend in GpuBackend::fallback_order() {
            debug!("Trying {} backend (hardware only)...", backend);
            match Self::try_backend(device_index, backend, true).await {
                Ok(ctx) => {
                    info!("Using {} backend: {}", backend, ctx.device_name());
                    return Ok(ctx);
                }
                Err(e) => {
                    debug!("{} backend failed (hardware): {}", backend, e);
                }
            }
        }

        // Second pass: accept software renderers as fallback
        debug!("No hardware GPU found, trying software renderers...");
        for &backend in GpuBackend::fallback_order() {
            debug!("Trying {} backend (including software)...", backend);
            match Self::try_backend(device_index, backend, false).await {
                Ok(ctx) => {
                    info!(
                        "Using {} backend (software): {}",
                        backend,
                        ctx.device_name()
                    );
                    return Ok(ctx);
                }
                Err(e) => {
                    debug!("{} backend failed: {}", backend, e);
                }
            }
        }

        Err(anyhow!("No GPU backends available"))
    }

    /// Check if adapter is a software renderer
    fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
        if info.device_type == wgpu::DeviceType::Cpu {
            return true;
        }
        let name_lower = info.name.to_lowercase();
        name_lower.contains("llvmpipe")
            || name_lower.contains("swiftshader")
            || name_lower.contains("software")
            || name_lower.contains("lavapipe")
            || name_lower.contains("mesa software")
    }

    /// Try to create context with specific backend
    async fn try_backend(
        device_index: u32,
        backend: GpuBackend,
        hardware_only: bool,
    ) -> Result<Self> {
        let backends = backend.to_wgpu_backends();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let mut adapters: Vec<_> = instance.enumerate_adapters(backends).await;

        if hardware_only {
            adapters.retain(|a| !Self::is_software_adapter(&a.get_info()));
        }

        if adapters.is_empty() {
            if hardware_only {
                anyhow::bail!("No hardware {} adapters found", backend);
            } else {
                anyhow::bail!("No {} adapters found", backend);
            }
        }

        // Sort by device type (discrete > virtual > integrated > cpu)
        // and by backend priority (Vulkan > Metal > DX12 > GL)
        adapters.sort_by_key(|a| {
            let info = a.get_info();
            let device_priority = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::VirtualGpu => 1,
                wgpu::DeviceType::IntegratedGpu => 2,
                wgpu::DeviceType::Cpu => 3,
                _ => 4,
            };
            let backend_priority = match info.backend {
                wgpu::Backend::Vulkan => 0,
                wgpu::Backend::Metal => 1,
                wgpu::Backend::Dx12 => 2,
                wgpu::Backend::Gl => 3,
                _ => 4,
            };
            (device_priority, backend_priority)
        });

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

    /// Create GPU context with a specific backend (accepts software renderers)
    async fn new_with_backend(device_index: u32, backend: GpuBackend) -> Result<Self> {
        Self::try_backend(device_index, backend, false).await
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
