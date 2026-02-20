//! Pure CPU implementation of Kangaroo algorithm
//!
//! Used for performance comparison with GPU implementation.

use super::dp_table::compute_candidate_scalars;
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::elliptic_curve::point::AffineCoordinates;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::U256 as K256U256;
use k256::{AffinePoint, ProjectivePoint, Scalar};
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
    pub fn new(
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

        if self.range_bits <= 24 {
            return self.bruteforce_fallback(timeout);
        }

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

        let mut tame_cycle_counter: u32 = 0;
        let mut wild_cycle_counter: u32 = 0;
        let mut tame_last_jump: usize = 16;
        let mut wild_last_jump: usize = 16;
        let mut tame_repeat: u32 = 0;
        let mut wild_repeat: u32 = 0;

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
            let elapsed = start_time.elapsed();
            if elapsed >= timeout {
                return None;
            }

            // Tame step
            let tame_affine = tame_pos.to_affine();
            let tame_x = get_x_low_from_affine(&tame_affine);
            let tame_dist_before = tame_dist;
            let jump_idx = (tame_x & 15) as usize;
            let y_is_odd: bool = bool::from(tame_affine.y_is_odd());
            if y_is_odd {
                tame_pos -= jump_points[jump_idx];
                tame_dist -= jump_distances[jump_idx];
            } else {
                tame_pos += jump_points[jump_idx];
                tame_dist += jump_distances[jump_idx];
            }
            self.ops += 1;

            tame_cycle_counter += 1;
            if jump_idx == tame_last_jump {
                tame_repeat += 1;
            } else {
                tame_repeat = 0;
            }
            tame_last_jump = jump_idx;
            if tame_cycle_counter > 1024 || tame_repeat > 4 {
                let escape_idx = ((tame_cycle_counter as usize)
                    .wrapping_mul(31)
                    .wrapping_add(jump_idx)
                    .wrapping_add(self.ops as usize))
                    & 15;
                tame_pos += jump_points[escape_idx];
                tame_dist += jump_distances[escape_idx];
                tame_cycle_counter = 0;
                tame_repeat = 0;
                tame_last_jump = escape_idx;
            }

            // Check DP
            if (tame_x & self.dp_mask) == 0 {
                if let Some(&wild_d) = self.wild_table.get(&tame_x) {
                    tracing::info!("Collision Tame->Wild: x={:x}", tame_x);
                    if let Some(key) = try_candidates(mid, tame_dist_before, wild_d, &self.pubkey) {
                        return Some(key);
                    }
                }
                self.tame_table.insert(tame_x, tame_dist_before);
                tame_cycle_counter = 0;
                tame_repeat = 0;
            }

            // Wild step
            let wild_affine = wild_pos.to_affine();
            let wild_x = get_x_low_from_affine(&wild_affine);
            let wild_dist_before = wild_dist;
            let jump_idx = (wild_x & 15) as usize;
            let y_is_odd: bool = bool::from(wild_affine.y_is_odd());
            if y_is_odd {
                wild_pos -= jump_points[jump_idx];
                wild_dist -= jump_distances[jump_idx];
            } else {
                wild_pos += jump_points[jump_idx];
                wild_dist += jump_distances[jump_idx];
            }
            self.ops += 1;

            wild_cycle_counter += 1;
            if jump_idx == wild_last_jump {
                wild_repeat += 1;
            } else {
                wild_repeat = 0;
            }
            wild_last_jump = jump_idx;
            if wild_cycle_counter > 1024 || wild_repeat > 4 {
                let escape_idx = ((wild_cycle_counter as usize)
                    .wrapping_mul(31)
                    .wrapping_add(jump_idx)
                    .wrapping_add(self.ops as usize))
                    & 15;
                wild_pos += jump_points[escape_idx];
                wild_dist += jump_distances[escape_idx];
                wild_cycle_counter = 0;
                wild_repeat = 0;
                wild_last_jump = escape_idx;
            }

            // Check DP
            if (wild_x & self.dp_mask) == 0 {
                if let Some(&tame_d) = self.tame_table.get(&wild_x) {
                    tracing::info!("Collision Wild->Tame: x={:x}", wild_x);
                    if let Some(key) = try_candidates(mid, tame_d, wild_dist_before, &self.pubkey) {
                        return Some(key);
                    }
                }
                self.wild_table.insert(wild_x, wild_dist_before);
                wild_cycle_counter = 0;
                wild_repeat = 0;
            }
        }
    }

    pub fn total_ops(&self) -> u64 {
        self.ops
    }

    fn bruteforce_fallback(&self, timeout: Duration) -> Option<Vec<u8>> {
        if self.range_bits > 24 {
            return None;
        }

        if timeout.is_zero() {
            return None;
        }

        let started = Instant::now();

        let limit = 1u64.checked_shl(self.range_bits)?;

        let mut candidate = self.start;
        for _ in 0..limit {
            if started.elapsed() > timeout {
                return None;
            }

            let key_bytes = candidate.to_bytes();
            let first_nonzero = key_bytes.iter().position(|&x| x != 0).unwrap_or(31);
            let trimmed = &key_bytes[first_nonzero..];
            if crate::crypto::verify_key(trimmed, &self.pubkey) {
                return Some(trimmed.to_vec());
            }
            candidate += Scalar::ONE;
        }

        None
    }
}

fn get_x_low_from_affine(affine: &AffinePoint) -> u128 {
    let encoded = affine.to_encoded_point(false);
    let x_bytes = encoded.x().unwrap();
    let mut low = [0u8; 16];
    low.copy_from_slice(&x_bytes[16..32]);
    u128::from_be_bytes(low)
}

fn try_candidates(
    mid: Scalar,
    tame_dist: Scalar,
    wild_dist: Scalar,
    pubkey: &ProjectivePoint,
) -> Option<Vec<u8>> {
    use crate::crypto::verify_key;

    let candidates = compute_candidate_scalars(mid, tame_dist, wild_dist);

    for key_scalar in &candidates {
        let key_bytes = key_scalar.to_bytes();
        let first_nonzero = key_bytes.iter().position(|&x| x != 0).unwrap_or(31);
        let trimmed = &key_bytes[first_nonzero..];
        if verify_key(trimmed, pubkey) {
            return Some(trimmed.to_vec());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::parse_pubkey;

    #[test]
    fn test_cpu_solver_simple() {
        let pubkey_str = "02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482";
        let pubkey = parse_pubkey(pubkey_str).unwrap();

        let range_bits = 15;
        let dp_bits = 4;

        let mut start_bytes = [0u8; 32];
        start_bytes[29..32].copy_from_slice(&0x10000u32.to_be_bytes()[1..4]);

        let mut solver = CpuKangarooSolver::new(pubkey, start_bytes, range_bits, dp_bits);
        let result = solver.solve(Duration::from_secs(10));

        assert!(result.is_some());
        let key = result.unwrap();
        let hex_key = hex::encode(key);
        let trimmed = hex_key.trim_start_matches('0');
        assert_eq!(trimmed, "12345");
    }

    #[test]
    fn test_cpu_solver_respects_timeout_budget() {
        let pubkey_str = "02e963ffdfe34e63b68aeb42a5826e08af087660e0dac1c3e79f7625ca4e6ae482";
        let pubkey = parse_pubkey(pubkey_str).unwrap();

        let mut start_bytes = [0u8; 32];
        start_bytes[29..32].copy_from_slice(&0x10000u32.to_be_bytes()[1..4]);

        let mut solver = CpuKangarooSolver::new(pubkey, start_bytes, 24, 4);
        let timeout = Duration::from_millis(1);

        let started = Instant::now();
        let _ = solver.solve(timeout);
        let elapsed = started.elapsed();

        assert!(
            elapsed <= Duration::from_millis(100),
            "solver exceeded timeout budget too much: {:?}",
            elapsed
        );
    }
}
