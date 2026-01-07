//! GPU buffer management

use super::{
    GpuAffinePoint, GpuConfig, GpuContext, GpuDistinguishedPoint, GpuKangaroo, KangarooPipeline,
};
use anyhow::Result;
use wgpu::{BindGroup, Buffer, BufferUsages};

/// GPU buffer collection
pub struct GpuBuffers {
    pub config_buffer: Buffer,
    #[allow(dead_code)]
    jump_points_buffer: Buffer,
    #[allow(dead_code)]
    jump_distances_buffer: Buffer,
    pub kangaroos_buffer: Buffer,
    pub dp_buffer: Buffer,
    pub dp_count_buffer: Buffer,
    pub staging_buffer: Buffer,
    pub bind_group: BindGroup,
}

impl GpuBuffers {
    /// Create GPU buffers
    pub fn new(
        ctx: &GpuContext,
        pipeline: &KangarooPipeline,
        config: &GpuConfig,
        jump_points: &[GpuAffinePoint],
        jump_distances: &[[u32; 8]],
        num_kangaroos: u32,
        max_dps: u32,
    ) -> Result<Self> {
        // Config buffer (uniform)
        let config_buffer = ctx.create_buffer_init(
            "Config Buffer",
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            &[*config],
        );

        // Jump points buffer (storage)
        let jump_points_buffer =
            ctx.create_buffer_init("Jump Points Buffer", BufferUsages::STORAGE, jump_points);

        // Jump distances buffer (storage)
        let jump_distances_buffer = ctx.create_buffer_init(
            "Jump Distances Buffer",
            BufferUsages::STORAGE,
            jump_distances,
        );

        // Kangaroos buffer
        let kangaroos_buffer = ctx.create_buffer::<GpuKangaroo>(
            "Kangaroos Buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            num_kangaroos as u64,
        );

        // DP buffer
        let dp_buffer = ctx.create_buffer::<GpuDistinguishedPoint>(
            "DP Buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            max_dps as u64,
        );

        // DP count buffer (atomic u32)
        let dp_count_buffer = ctx.create_buffer_init(
            "DP Count Buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            &[0u32],
        );

        // Staging buffer for readback
        // Must be large enough to hold either kangaroos (for normalization) or DPs
        let kangaroos_size = (num_kangaroos as usize) * std::mem::size_of::<GpuKangaroo>();
        let dp_size = (max_dps as usize) * std::mem::size_of::<GpuDistinguishedPoint>();
        let staging_size = std::cmp::max(kangaroos_size, dp_size) as u64 + 4;

        let staging_buffer = ctx.create_buffer::<u8>(
            "Staging Buffer",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            staging_size,
        );

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Kangaroo Bind Group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: jump_points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: jump_distances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kangaroos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dp_count_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            config_buffer,
            jump_points_buffer,
            jump_distances_buffer,
            kangaroos_buffer,
            dp_buffer,
            dp_count_buffer,
            staging_buffer,
            bind_group,
        })
    }
}
