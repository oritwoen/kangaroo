//! GPU context and device management

use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use std::sync::Arc;
use tracing::{debug, info, warn};
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

    /// Convert from wgpu::Backend to GpuBackend
    pub fn from_wgpu_backend(backend: wgpu::Backend) -> Self {
        match backend {
            wgpu::Backend::Vulkan => GpuBackend::Vulkan,
            wgpu::Backend::Metal => GpuBackend::Metal,
            wgpu::Backend::Dx12 => GpuBackend::Dx12,
            wgpu::Backend::Gl => GpuBackend::Gl,
            _ => GpuBackend::Auto,
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Information about a discovered GPU device
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Human-readable adapter name
    pub name: String,
    /// Device type (Discrete, Integrated, Virtual, CPU)
    pub device_type: wgpu::DeviceType,
    /// Backend (Vulkan, Metal, DX12, GL)
    pub backend: wgpu::Backend,
    /// Sequential index for CLI selection
    pub index: u32,
}

/// Check if an adapter is a software renderer
pub fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
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

struct AdapterEntry {
    adapter: wgpu::Adapter,
    info: wgpu::AdapterInfo,
    software: bool,
    device_priority: u32,
    backend_priority: u32,
}

fn device_priority(device_type: wgpu::DeviceType) -> u32 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::VirtualGpu => 1,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    }
}

fn backend_priority(backend: wgpu::Backend) -> u32 {
    match backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    }
}

async fn enumerate_visible_adapters(backend: GpuBackend) -> Result<Vec<AdapterEntry>> {
    let backends = backend.to_wgpu_backends();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let adapters: Vec<_> = instance.enumerate_adapters(backends).await;
    if adapters.is_empty() {
        anyhow::bail!("No GPU adapters found");
    }

    let mut entries: Vec<AdapterEntry> = adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            AdapterEntry {
                software: is_software_adapter(&info),
                device_priority: device_priority(info.device_type),
                backend_priority: backend_priority(info.backend),
                adapter,
                info,
            }
        })
        .collect();

    entries.sort_by_key(|e| (e.device_priority, e.backend_priority));

    let has_hardware = entries.iter().any(|e| !e.software && e.device_priority < 3);
    if has_hardware {
        entries.retain(|e| !e.software && e.device_priority < 3);
    } else {
        warn!("No hardware GPU found, listing software renderers");
    }

    let mut best_backend_by_name: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    for e in &entries {
        let current_best = best_backend_by_name
            .entry(e.info.name.clone())
            .or_insert(e.backend_priority);
        if e.backend_priority < *current_best {
            *current_best = e.backend_priority;
        }
    }
    entries.retain(|e| e.backend_priority == *best_backend_by_name.get(&e.info.name).unwrap());

    Ok(entries)
}

/// Enumerate available GPU devices, sorted by priority
///
/// Filters software renderers unless no hardware GPU is found.
/// Deduplicates adapters by name (same GPU under multiple backends).
/// Sorts by device type (Discrete > Virtual > Integrated > CPU)
/// and backend priority (Vulkan > Metal > DX12 > GL).
pub async fn enumerate_gpus(backend: GpuBackend) -> Result<Vec<GpuDeviceInfo>> {
    let devices: Vec<GpuDeviceInfo> = enumerate_visible_adapters(backend)
        .await?
        .into_iter()
        .enumerate()
        .map(|(i, entry)| GpuDeviceInfo {
            name: entry.info.name,
            device_type: entry.info.device_type,
            backend: entry.info.backend,
            index: i as u32,
        })
        .collect();

    Ok(devices)
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

    /// Create GPU context using the same global index ordering as `enumerate_gpus`.
    /// This avoids enumerating once for selection and then again for device creation.
    pub async fn new_from_global_index(device_index: u32, backend: GpuBackend) -> Result<Self> {
        let entry = enumerate_visible_adapters(backend)
            .await?
            .into_iter()
            .nth(device_index as usize)
            .context("GPU device index out of range")?;
        Self::from_adapter(entry.adapter, entry.info).await
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
            adapters.retain(|a| !is_software_adapter(&a.get_info()));
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
        Self::from_adapter(adapter, adapter_info).await
    }

    /// Create GPU context with a specific backend (accepts software renderers)
    async fn new_with_backend(device_index: u32, backend: GpuBackend) -> Result<Self> {
        Self::try_backend(device_index, backend, false).await
    }

    async fn from_adapter(adapter: wgpu::Adapter, adapter_info: wgpu::AdapterInfo) -> Result<Self> {
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
        let workgroup_size = self.max_workgroup_size().min(128);
        // Conservative cap: very large grids make even the minimum 16-step dispatch too slow
        // for our 50ms target and explode CPU-side initialization cost. Keep the default launch
        // size smaller, then let solver calibration tune steps_per_call on top.
        let workgroups = self.max_workgroups().min(65535).min(512);
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
    ) -> Result<wgpu::Buffer> {
        let element_size = std::mem::size_of::<T>() as u64;
        let size = count.checked_mul(element_size).ok_or_else(|| {
            anyhow!(
                "Buffer size overflow for '{label}': count={count}, element_size={element_size}"
            )
        })?;

        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }))
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
}
