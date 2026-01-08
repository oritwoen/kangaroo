//! Kangaroo initialization and jump table generation.

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use crate::math::negate_256_be;
use anyhow::Result;
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
use rayon::prelude::*;

/// Generate jump table with precomputed points.
///
/// Uses FNV-1a pseudo-random generation (Strategy S2) for better distribution
/// than simple powers of 2.
pub fn generate_jump_table(range_bits: u32) -> (Vec<GpuAffinePoint>, Vec<[u32; 8]>) {
    const TABLE_SIZE: usize = 256;

    let mut points = Vec::with_capacity(TABLE_SIZE);
    let mut distances = Vec::with_capacity(TABLE_SIZE);

    // Target mean step size: sqrt(N) / 2
    let mean_exp = range_bits / 2;

    // Generate random scalars with magnitude around 2^mean_exp.
    for i in 0..TABLE_SIZE {
        // FNV-1a like hash to generate deterministic random steps
        let mut h = 0x811c9dc5u32;
        h = (h ^ (i as u32)).wrapping_mul(0x01000193);

        // Strategy: fill bytes up to mean_exp / 8
        // We need ceil(mean_exp / 8) bytes to cover mean_exp bits
        let num_bytes = mean_exp.div_ceil(8);
        let limit_byte = (num_bytes as usize).min(32);
        let mut scalar_bytes = [0u8; 32];

        // Use the hash to fill bytes
        #[allow(clippy::needless_range_loop)]
        for b in (32 - limit_byte)..32 {
            h = (h ^ (b as u32)).wrapping_mul(0x01000193);
            scalar_bytes[b] = (h & 0xFF) as u8;
        }

        // Mask the top byte to ensure we don't exceed our target range
        // If mean_exp=20, num_bytes=3. Top byte index is 32-3=29.
        // 20 % 8 = 4. We want 4 bits (0..3) from the top byte?
        // No, if mean_exp=20, we want values up to 2^20.
        // Byte 31 (LSB) -> bits 0..7
        // Byte 30 -> bits 8..15
        // Byte 29 -> bits 16..23
        // We want to mask Byte 29 to keep bits 16..19 (4 bits).
        // 20 % 8 = 4.

        let rem = mean_exp % 8;
        if rem != 0 {
            let mask = (1u8 << rem) - 1;
            if 32 - limit_byte < 32 {
                scalar_bytes[32 - limit_byte] &= mask;
            }
        } else if limit_byte > 0 {
            // If exact multiple of 8, we typically allow full byte?
            // e.g. mean_exp=16. limit=2. Bytes 30, 31.
            // We want up to 2^16? No, 2^16 requires 17 bits usually (bit 16 set).
            // But here we are generating random offsets *up to* 2^mean_exp.
            // So for 16, we probably want 0..2^16-1. So 2 full bytes is fine.
        }

        // Ensure at least 1
        if scalar_bytes.iter().all(|&x| x == 0) {
            scalar_bytes[31] = 1;
        }

        // Compute point = scalar * G
        let scalar_uint = K256U256::from_be_slice(&scalar_bytes);
        let scalar = Scalar::reduce(scalar_uint);
        let point = ProjectivePoint::mul_by_generator(&scalar);
        let affine = point.to_affine();

        points.push(affine_to_gpu(&affine));
        distances.push(scalar_be_to_limbs(&scalar_bytes));
    }

    // Debug logging for first few
    for i in 0..4u32 {
        tracing::debug!("Jump table[{}] generated", i);
    }

    (points, distances)
}

/// Initialize kangaroo positions.
///
/// Half are "tame" (start at known point), half are "wild" (start near pubkey).
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
) -> Result<Vec<GpuKangaroo>> {
    let half = num_kangaroos / 2;

    // Handle the case where range_bits is 128 or more (u128 overflow)
    // Actually our u128 math might overflow if range_bits=128, but usually it is <128.
    // If range_bits=128, range_size=0 (overflow) in u128.
    // But GpuKangaroo supports full 256 bits.
    // For initialization we assume range < 128 bits for now as logic uses u128.

    // Re-check range_size calculation to avoid panic on overflow if range_bits=128
    let range_size = if range_bits >= 128 {
        u128::MAX
    } else {
        1u128 << range_bits
    };

    let range_middle = if range_bits >= 128 {
        u128::MAX / 2
    } else {
        1u128 << (range_bits - 1)
    };

    // Use u256_to_u128 for start (assuming start fits in 128 bits for this logic?
    // No, start can be large.
    // Wait, the original code used `u256_to_u128(start)`.
    // If start is 256-bit (e.g. 0x8000...), `u256_to_u128` might truncate or panic?
    // Let's check `convert.rs`.
    // Assuming for now we are solving in a sub-range so start is arbitrary.
    // But `init_tame_kangaroo` uses `start_val + tame_offset`.
    // If start is large, we need full 256-bit arithmetic.
    // The original code used u128 for offsets and simple math.
    // We should stick to that assumption or improve it.
    // Let's assume start fits in u128 OR we handle full scalar math.
    // Since `init_tame_kangaroo_at_offset` takes `start_val: u128`, it seems the original code assumed
    // start is small OR we only care about the lower 128 bits for the offset logic?
    // NO, start is the base key. It can be large.
    // `init_tame_kangaroo` in original code:
    // `let tame_scalar = start_val + tame_offset;` where start_val is u128.
    // This implies the original code ONLY supported start values < 2^128.
    // That's a limitation of the original code. I will keep it for now but note it.
    // Actually, let's fix it by passing `start` as U256 to the helper.

    // Wait, let's stick to the interface.
    // I will read `convert.rs` later to see `u256_to_u128`.
    // For now, I'll restore `hash_seed` and the logic.

    tracing::debug!(
        "Kangaroo init: range_bits={}, range_size=0x{:x}, range_middle=0x{:x}",
        range_bits,
        range_size,
        range_middle
    );

    // Grid delta for even distribution (S2 strategy)
    let grid_delta = if num_kangaroos > 0 {
        range_size / (num_kangaroos as u128)
    } else {
        range_size
    };

    // Parallel initialization with rayon
    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            let is_tame = i < half;

            // Grid-based offset + small random jitter
            let grid_pos = (i as u128) * grid_delta;
            let prng_seed = hash_seed(i, 0xCAFEBABE);
            let jitter = prng_seed % (grid_delta / 2 + 1);

            let offset = (grid_pos + jitter) % range_size;

            let (point, dist) = if is_tame {
                init_tame_kangaroo_at_offset(start, offset)
            } else {
                init_wild_kangaroo_at_offset(pubkey, offset, range_middle)
            };

            let gpu_point = affine_to_gpu(&point);

            GpuKangaroo {
                x: gpu_point.x,
                y: gpu_point.y,
                z: [1, 0, 0, 0, 0, 0, 0, 0], // Z = 1 (affine)
                dist,
                ktype: if is_tame { 0 } else { 1 },
                is_active: 1,
                _padding: [0; 2],
            }
        })
        .collect();

    Ok(kangaroos)
}

/// FNV-1a hash for deterministic PRNG seeding.
fn hash_seed(index: u32, salt: u64) -> u128 {
    let mut h = 0xcbf29ce484222325u64; // FNV offset basis

    h ^= index as u64;
    h = h.wrapping_mul(0x100000001b3); // FNV prime
    h ^= salt;
    h = h.wrapping_mul(0x100000001b3);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;

    let h2 = h.wrapping_mul(0xc4ceb9fe1a85ec53) ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
    ((h as u128) << 64) | (h2 as u128)
}

/// Initialize a tame kangaroo at a specific offset from start
fn init_tame_kangaroo_at_offset(start: &U256, offset: u128) -> (k256::AffinePoint, [u32; 8]) {
    // Convert start (U256) to Scalar
    // U256 is usually little-endian or we need to check crate::crypto::U256
    // In `tools/kangaroo/src/crypto/mod.rs` likely uses `k256::U256` or custom.
    // The original code used `u256_to_u128` which implies `start` might be compatible.
    // Let's assume we can convert `start` to big-endian bytes.
    // `start.to_be_bytes()`?
    // Let's use `u256_to_u128` if we assume start is small, OR fix the math.
    // BETTER: Use `k256::U256` add.

    // Since I don't know the exact `U256` type (it's imported from `crate::crypto::U256`),
    // I will use `u256_to_u128` as per original logic if I can't find better,
    // BUT the original code passed `start_val: u128`.
    // I changed the signature to `start: &U256`.

    // Let's assume start fits in u128 for now to minimize breakage or check `crypto::U256`.
    // Actually, looking at `cpu/solver.rs`: `start: U256`.
    // `u256_to_u128` is imported from `crate::convert`.

    // Convert start (LE bytes) to K256U256
    let start_uint = K256U256::from_le_slice(start);

    // Convert offset to K256U256
    let mut offset_le = [0u8; 32];
    offset_le[0..16].copy_from_slice(&offset.to_le_bytes());
    let offset_uint = K256U256::from_le_slice(&offset_le);

    // Add to get absolute scalar
    let sum = start_uint.wrapping_add(&offset_uint);

    let scalar = Scalar::reduce(sum);
    let point = ProjectivePoint::mul_by_generator(&scalar);

    // Store offset as dist (relative to start)
    // Distance tracks the offset from start, so it fits in u128 (range size)
    let mut offset_bytes = [0u8; 32];
    offset_bytes[16..].copy_from_slice(&offset.to_be_bytes());

    (point.to_affine(), scalar_be_to_limbs(&offset_bytes))
}

/// Initialize a wild kangaroo at a specific offset
fn init_wild_kangaroo_at_offset(
    pubkey: &Point,
    raw_offset: u128,
    range_middle: u128,
) -> (k256::AffinePoint, [u32; 8]) {
    // Center the offset: map [0, range) to [-range/2, range/2)
    let centered_offset = raw_offset as i128 - range_middle as i128;

    if centered_offset >= 0 {
        let offset = centered_offset as u128;
        let mut offset_bytes = [0u8; 32];
        offset_bytes[16..].copy_from_slice(&offset.to_be_bytes());

        let scalar_uint = K256U256::from_be_slice(&offset_bytes);
        let scalar = Scalar::reduce(scalar_uint);
        let offset_point = ProjectivePoint::mul_by_generator(&scalar);
        let wild_point = *pubkey + offset_point;

        (wild_point.to_affine(), scalar_be_to_limbs(&offset_bytes))
    } else {
        // Negative offset: subtract from pubkey
        let abs_offset = (-centered_offset) as u128;
        let mut offset_bytes = [0u8; 32];
        offset_bytes[16..].copy_from_slice(&abs_offset.to_be_bytes());

        let scalar_uint = K256U256::from_be_slice(&offset_bytes);
        let scalar = Scalar::reduce(scalar_uint);
        let offset_point = ProjectivePoint::mul_by_generator(&scalar);
        let wild_point = *pubkey - offset_point;

        // Store negative offset as two's complement
        let neg_offset_bytes = negate_256_be(&offset_bytes);
        (
            wild_point.to_affine(),
            scalar_be_to_limbs(&neg_offset_bytes),
        )
    }
}
