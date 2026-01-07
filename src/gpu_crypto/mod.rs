pub mod context;
pub mod shaders;

use bytemuck::{Pod, Zeroable};
pub use context::GpuContext;

/// GPU Affine Point (x, y coordinates in 32-bit limbs)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuAffinePoint {
    pub x: [u32; 8],
    pub y: [u32; 8],
}
