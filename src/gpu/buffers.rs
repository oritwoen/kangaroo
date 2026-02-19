//! GPU buffer management
//!
//! Uses double-buffered DP slots so the CPU can read back results from
//! the previous dispatch while the GPU is already executing the next one.

use super::{
    GpuAffinePoint, GpuConfig, GpuContext, GpuDistinguishedPoint, GpuKangaroo, KangarooPipeline,
};
use anyhow::Result;
use wgpu::{BindGroup, Buffer, BufferUsages};

/// Number of DP buffer slots for double buffering
const NUM_SLOTS: usize = 2;

/// One slot of DP-related buffers (dp_buffer + dp_count + staging + bind_group)
struct DpSlot {
    dp_buffer: Buffer,
    dp_count_buffer: Buffer,
    staging_buffer: Buffer,
    bind_group: BindGroup,
}

/// GPU buffer collection with double-buffered DP slots
pub struct GpuBuffers {
    pub config_buffer: Buffer,
    #[allow(dead_code)]
    jump_points_buffer: Buffer,
    #[allow(dead_code)]
    jump_distances_buffer: Buffer,
    pub kangaroos_buffer: Buffer,
    slots: [DpSlot; NUM_SLOTS],
}

impl GpuBuffers {
    /// Create GPU buffers with double-buffered DP slots
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

        let kangaroos_buffer = ctx.create_buffer::<GpuKangaroo>(
            "Kangaroos Buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            num_kangaroos as u64,
        )?;

        let kangaroos_size = (num_kangaroos as usize) * std::mem::size_of::<GpuKangaroo>();
        let dp_size = (max_dps as usize) * std::mem::size_of::<GpuDistinguishedPoint>();
        let staging_size = std::cmp::max(kangaroos_size, dp_size) as u64 + 4;

        let make_slot = |i: usize| -> Result<DpSlot> {
            let label_suffix = if i == 0 { "A" } else { "B" };

            let dp_buffer = ctx.create_buffer::<GpuDistinguishedPoint>(
                &format!("DP Buffer {label_suffix}"),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                max_dps as u64,
            )?;

            let dp_count_buffer = ctx.create_buffer_init(
                &format!("DP Count Buffer {label_suffix}"),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                &[0u32],
            );

            let staging_buffer = ctx.create_buffer::<u8>(
                &format!("Staging Buffer {label_suffix}"),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                staging_size,
            )?;

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(if i == 0 {
                    "Kangaroo Bind Group A"
                } else {
                    "Kangaroo Bind Group B"
                }),
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

            Ok(DpSlot {
                dp_buffer,
                dp_count_buffer,
                staging_buffer,
                bind_group,
            })
        };

        let slots = [make_slot(0)?, make_slot(1)?];

        Ok(Self {
            config_buffer,
            jump_points_buffer,
            jump_distances_buffer,
            kangaroos_buffer,
            slots,
        })
    }

    /// Get the bind group for a given slot
    pub fn bind_group(&self, slot: usize) -> &BindGroup {
        &self.slots[slot].bind_group
    }

    /// Get the DP buffer for a given slot
    pub fn dp_buffer(&self, slot: usize) -> &Buffer {
        &self.slots[slot].dp_buffer
    }

    /// Get the DP count buffer for a given slot
    pub fn dp_count_buffer(&self, slot: usize) -> &Buffer {
        &self.slots[slot].dp_count_buffer
    }

    /// Get the staging buffer for a given slot
    pub fn staging_buffer(&self, slot: usize) -> &Buffer {
        &self.slots[slot].staging_buffer
    }
}
