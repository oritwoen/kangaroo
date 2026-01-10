//! GPU compute module

mod buffers;
mod pipeline;

pub use crate::gpu_crypto::{GpuAffinePoint, GpuContext};
pub use buffers::GpuBuffers;
pub use pipeline::KangarooPipeline;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuConfig {
    pub dp_mask_lo: [u32; 4],
    pub dp_mask_hi: [u32; 4],
    pub num_kangaroos: u32,
    pub steps_per_call: u32,
    pub jump_table_size: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuKangaroo {
    pub x: [u32; 8],
    pub y: [u32; 8],
    pub dist: [u32; 8],
    pub ktype: u32,
    pub is_active: u32,
    pub _padding: [u32; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDistinguishedPoint {
    pub x: [u32; 8],
    pub dist: [u32; 8],
    pub ktype: u32,
    pub kangaroo_id: u32,
    pub _padding: [u32; 6],
}
