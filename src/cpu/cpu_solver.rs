//! Pure CPU implementation of Kangaroo algorithm
//!
//! Used for performance comparison with GPU implementation.

use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Pure CPU Kangaroo solver using k256
pub struct CpuKangarooSolver {
    pubkey: ProjectivePoint,
    start: Scalar, // Use Scalar for full 256-bit arithmetic
    range_bits: u32,
    dp_mask: u128,                     // DP check only needs low bits
    tame_table: HashMap<u128, Scalar>, // x_low -> distance (Scalar)
    wild_table: HashMap<u128, Scalar>,
    ops: u64,
}

impl CpuKangarooSolver {
    #[allow(dead_code)]
    pub fn new(pubkey: ProjectivePoint, start_u128: u128, range_bits: u32, dp_bits: u32) -> Self {
        // NOTE: The original interface took u128 start. We should ideally take U256.
        // For now, we convert the u128 to Scalar.
        // To support full P135, we need to change the constructor signature or rely on `run` passing a full scalar.
        // But `run` in lib.rs calls this. We need to update `lib.rs` too.
        // Let's fix this struct first.

        let mut bytes = [0u8; 32];
        bytes[16..].copy_from_slice(&start_u128.to_be_bytes());
        let start_uint = K256U256::from_be_slice(&bytes);
        let start = Scalar::reduce(start_uint);

        let dp_mask = (1u128 << dp_bits) - 1;
        Self {
            pubkey,
            start,
            range_bits,
            dp_mask,
            tame_table: HashMap::new(),
            wild_table: HashMap::new(),
            ops: 0,
        }
    }

    // New constructor that takes full 256-bit start (big-endian bytes)
    pub fn new_full(
        pubkey: ProjectivePoint,
        start_bytes: [u8; 32],
        range_bits: u32,
        dp_bits: u32,
    ) -> Self {
        let start_uint = K256U256::from_be_slice(&start_bytes);
        let start = Scalar::reduce(start_uint);
        let dp_mask = (1u128 << dp_bits) - 1;
        Self {
            pubkey,
            start,
            range_bits,
            dp_mask,
            tame_table: HashMap::new(),
            wild_table: HashMap::new(),
            ops: 0,
        }
    }

    pub fn solve(&mut self, timeout: Duration) -> Option<Vec<u8>> {
        let start_time = Instant::now();

        // Calculate middle of range: 2^(range_bits - 1)
        // Construct scalar for range_middle
        let mut mid_bytes = [0u8; 32];
        // Bit position: range_bits - 1.
        // e.g. range_bits=135 -> bit 134.
        // Byte index (from end): 134 // 8 = 16. remainder 6.
        // mid_bytes[31 - 16] |= 1 << 6.
        // Be careful with indices.
        // Let's use U256 bit setting if possible, or manual.
        // range_bits is usually small (<256).

        // Actually, simpler: range_middle = 1 << (range_bits - 1).
        // Convert this to Scalar.
        let bit = self.range_bits - 1;
        let byte_idx = 31 - (bit / 8) as usize;
        let bit_idx = bit % 8;
        mid_bytes[byte_idx] |= 1 << bit_idx;

        let range_middle_uint = K256U256::from_be_slice(&mid_bytes);
        let range_middle = Scalar::reduce(range_middle_uint);

        // mid = start + range_middle
        let mid = self.start + range_middle;

        // Initialize tame kangaroo at mid
        // tame_scalar = mid
        // tame_dist = 0 (relative to mid)
        let mut tame_pos = ProjectivePoint::mul_by_generator(&mid);
        let mut tame_dist = Scalar::ZERO;

        // Initialize wild kangaroo at pubkey
        let mut wild_pos = self.pubkey;
        let mut wild_dist = Scalar::ZERO;

        // Jump table (Scalar distances)
        let jump_distances: Vec<Scalar> = (0..16)
            .map(|i| {
                let mut h = 0x811c9dc5u32;
                h = (h ^ (i as u32)).wrapping_mul(0x01000193);
                let mean_exp = (self.range_bits / 2).saturating_sub(2).max(8);

                // Generate u128 approximation for jump (usually jumps are small < 2^128)
                // If range_bits is huge (e.g. 256), jumps should be ~2^128.
                // CPU solver is mostly for smaller ranges, but let's support up to 128-bit jumps.
                let base = 1u128 << (mean_exp - 1);
                // Cap mask to avoid overflow if mean_exp is large
                let mask = if mean_exp >= 128 {
                    u128::MAX
                } else {
                    (1u128 << mean_exp) - 1
                };
                let val = base + (h as u128 & mask);
                let val = if i == 0 { val | 1 } else { val };

                // Convert u128 to Scalar
                let mut bytes = [0u8; 32];
                bytes[16..].copy_from_slice(&val.to_be_bytes());
                let uint = K256U256::from_be_slice(&bytes);
                Scalar::reduce(uint)
            })
            .collect();

        let jump_points: Vec<ProjectivePoint> = jump_distances
            .iter()
            .map(ProjectivePoint::mul_by_generator)
            .collect();

        loop {
            if start_time.elapsed() > timeout {
                return None;
            }

            // Tame step
            let tame_x = get_x_low(&tame_pos);
            let jump_idx = (tame_x & 15) as usize;
            tame_pos += jump_points[jump_idx];
            tame_dist += jump_distances[jump_idx];
            self.ops += 1;

            // Check DP
            if (tame_x & self.dp_mask) == 0 {
                if let Some(&wild_d) = self.wild_table.get(&tame_x) {
                    tracing::info!("Collision Tame->Wild: x={:x}", tame_x);
                    // Collision!
                    // Tame pos: mid + tame_dist
                    // Wild pos: key + wild_dist
                    // mid + tame_dist = key + wild_dist
                    // key = mid + tame_dist - wild_dist
                    let key = mid + tame_dist - wild_d;
                    let key_bytes = key.to_bytes();
                    let key_vec = key_bytes.to_vec();
                    // Verify? (Optional, caller does it)
                    return Some(key_vec);
                }
                self.tame_table.insert(tame_x, tame_dist);
            }

            // Wild step
            let wild_x = get_x_low(&wild_pos);
            let jump_idx = (wild_x & 15) as usize;
            wild_pos += jump_points[jump_idx];
            wild_dist += jump_distances[jump_idx];
            self.ops += 1;

            // Check DP
            if (wild_x & self.dp_mask) == 0 {
                if let Some(&tame_d) = self.tame_table.get(&wild_x) {
                    tracing::info!("Collision Wild->Tame: x={:x}", wild_x);
                    // Collision!
                    // key = mid + tame_dist - wild_dist
                    let key = mid + tame_d - wild_dist;
                    let key_bytes = key.to_bytes();
                    return Some(key_bytes.to_vec());
                }
                self.wild_table.insert(wild_x, wild_dist);
            }
        }
    }

    pub fn total_ops(&self) -> u64 {
        self.ops
    }
}

fn get_x_low(point: &ProjectivePoint) -> u128 {
    let affine = point.to_affine();
    let encoded = affine.to_encoded_point(false);
    let x_bytes = encoded.x().unwrap();
    // Get low 128 bits (last 16 bytes)
    let mut low = [0u8; 16];
    low.copy_from_slice(&x_bytes[16..32]);
    u128::from_be_bytes(low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::parse_pubkey;

    #[test]
    fn test_cpu_solver_simple() {
        // Key: 0x12345
        // Pubkey for 0x12345
        let pubkey_str = "02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482";
        let pubkey = parse_pubkey(pubkey_str).unwrap();

        // Range: [0x10000, 0x18000] (contains 0x12345)
        // Range bits: 15 (size 32768)
        // Start: 0x10000
        let start = 0x10000;
        let range_bits = 15;
        let dp_bits = 4; // Frequent DPs for small range

        let mut solver = CpuKangarooSolver::new(pubkey, start, range_bits, dp_bits);
        let result = solver.solve(Duration::from_secs(10));

        assert!(result.is_some());
        let key = result.unwrap();
        let hex_key = hex::encode(key);
        let trimmed = hex_key.trim_start_matches('0');
        // Handle "012345" -> "12345"
        assert_eq!(trimmed, "12345");
    }
}
