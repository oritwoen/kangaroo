//! Kangaroo initialization and jump table generation.

use crate::convert::{affine_to_gpu, scalar_be_to_limbs};
use crate::crypto::{Point, U256};
use crate::gpu::{GpuAffinePoint, GpuKangaroo};
use crate::math::negate_256_be;
use anyhow::Result;
use k256::elliptic_curve::ops::Reduce;
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
use rayon::prelude::*;
use std::ops::Neg;

/// Generate jump table with precomputed points.
///
/// Uses FNV-1a pseudo-random generation (Strategy S2) for better distribution
/// than simple powers of 2.
pub fn generate_jump_table(
    range_bits: u32,
    base_point: &ProjectivePoint,
) -> (Vec<GpuAffinePoint>, Vec<[u32; 8]>) {
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
        let rem = mean_exp % 8;
        if rem != 0 {
            let mask = (1u8 << rem) - 1;
            if 32 - limit_byte < 32 {
                scalar_bytes[32 - limit_byte] &= mask;
            }
        }

        // Ensure at least 1
        if scalar_bytes.iter().all(|&x| x == 0) {
            scalar_bytes[31] = 1;
        }

        // Compute point = scalar * G
        let scalar_uint = K256U256::from_be_slice(&scalar_bytes);
        let scalar = Scalar::reduce(scalar_uint);
        let point = *base_point * scalar;
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
/// Split into three sets: tame (ktype=0), wild_1 (ktype=1), wild_2 (ktype=2).
/// Wild_2 uses the negated public key for cross-wild collision detection.
///
/// `kangaroo_offset` shifts global indices so multiple GPU workers get unique positions.
/// For single-GPU, pass 0. For multi-GPU, GPU N gets offset = N * num_kangaroos.
///
/// All range/offset math uses full 256-bit arithmetic to correctly handle
/// range_bits >= 128 (fixes silent degradation from u128 clamping).
pub fn initialize_kangaroos(
    pubkey: &Point,
    start: &U256,
    range_bits: u32,
    num_kangaroos: u32,
    base_point: &ProjectivePoint,
    kangaroo_offset: u32,
    global_kangaroo_count: u32,
) -> Result<Vec<GpuKangaroo>> {
    anyhow::ensure!(
        num_kangaroos >= 3,
        "Multi-set requires at least 3 kangaroos"
    );
    anyhow::ensure!(
        range_bits > 0 && range_bits <= 255,
        "range_bits must be 1..=255"
    );

    let one_third = num_kangaroos / 3;

    // Full 256-bit range arithmetic
    let range_size = K256U256::ONE.shl_vartime(range_bits as usize);
    let range_middle = K256U256::ONE.shl_vartime((range_bits - 1) as usize);

    tracing::debug!(
        "Kangaroo init: range_bits={}, num_kangaroos={}",
        range_bits,
        num_kangaroos
    );

    // Grid delta for even distribution (full 256-bit division)
    let total_k = K256U256::from(global_kangaroo_count.max(1) as u64);
    let grid_delta = div_u256(&range_size, &total_k);
    let grid_delta = if grid_delta == K256U256::ZERO {
        K256U256::ONE
    } else {
        grid_delta
    };

    // Pre-compute NonZero range_size for modulo (always non-zero for range_bits > 0)
    let nz_range = nonzero_u256(range_size);

    let neg_pubkey = pubkey.neg();

    // Parallel initialization with rayon
    let kangaroos: Vec<GpuKangaroo> = (0..num_kangaroos)
        .into_par_iter()
        .map(|i| {
            let ktype = if i < one_third {
                0 // tame
            } else if i < 2 * one_third {
                1 // wild_1
            } else {
                2 // wild_2
            };

            // Grid-based offset + small random jitter (all 256-bit)
            let global_i = kangaroo_offset + i;
            let i_uint = K256U256::from(global_i as u64);
            let grid_pos = i_uint.wrapping_mul(&grid_delta);

            let prng_seed = hash_seed(global_i, 0xCAFEBABE);
            let jitter_span = grid_delta.shr_vartime(1);
            let jitter_span = if jitter_span == K256U256::ZERO {
                K256U256::ONE
            } else {
                jitter_span
            };

            let seed_uint = u128_to_u256(prng_seed);
            let jitter = rem_u256(&seed_uint, &jitter_span);

            let sum = grid_pos.wrapping_add(&jitter);
            let (_, offset) = sum.div_rem(&nz_range);

            let (point, dist) = match ktype {
                0 => init_tame_kangaroo_at_offset(start, &offset, base_point),
                1 => init_wild_kangaroo_at_offset(pubkey, &offset, &range_middle, base_point),
                _ => init_wild_kangaroo_at_offset(&neg_pubkey, &offset, &range_middle, base_point),
            };

            let gpu_point = affine_to_gpu(&point);

            GpuKangaroo {
                x: gpu_point.x,
                y: gpu_point.y,
                dist,
                ktype,
                is_active: 1,
                cycle_counter: 0,
                repeat_count: 0,
                last_jump: 0xFFFFFFFF,
                _padding: [0; 3],
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

/// Convert u128 to K256U256.
fn u128_to_u256(val: u128) -> K256U256 {
    let mut le = [0u8; 32];
    le[0..16].copy_from_slice(&val.to_le_bytes());
    K256U256::from_le_slice(&le)
}

/// Extract big-endian [u8; 32] from K256U256 via limb decomposition.
fn u256_to_be_bytes(val: &K256U256) -> [u8; 32] {
    let limbs = val.as_limbs();
    let n = limbs.len();
    let mut bytes = [0u8; 32];
    for i in 0..n {
        let be = limbs[n - 1 - i].0.to_be_bytes();
        let sz = be.len();
        bytes[i * sz..(i + 1) * sz].copy_from_slice(&be);
    }
    bytes
}

/// Divide two K256U256 values. Returns zero if divisor is zero.
fn div_u256(a: &K256U256, b: &K256U256) -> K256U256 {
    if *b == K256U256::ZERO {
        return K256U256::ZERO;
    }
    let nz = nonzero_u256(*b);
    let (q, _) = a.div_rem(&nz);
    q
}

/// Remainder of K256U256 division. Returns `a` if divisor is zero.
fn rem_u256(a: &K256U256, b: &K256U256) -> K256U256 {
    if *b == K256U256::ZERO {
        return *a;
    }
    let nz = nonzero_u256(*b);
    let (_, r) = a.div_rem(&nz);
    r
}

/// Create NonZero<K256U256>. Panics on zero.
fn nonzero_u256(val: K256U256) -> crypto_bigint::NonZero<K256U256> {
    Option::from(crypto_bigint::NonZero::new(val)).expect("value must be non-zero")
}

/// Initialize a tame kangaroo at a specific offset from start.
fn init_tame_kangaroo_at_offset(
    start: &U256,
    offset: &K256U256,
    base_point: &ProjectivePoint,
) -> (k256::AffinePoint, [u32; 8]) {
    let start_uint = K256U256::from_le_slice(start);
    let sum = start_uint.wrapping_add(offset);
    let scalar = Scalar::reduce(sum);
    let point = *base_point * scalar;

    // Distance = offset relative to start (full 256-bit)
    let offset_be = u256_to_be_bytes(offset);
    (point.to_affine(), scalar_be_to_limbs(&offset_be))
}

/// Initialize a wild kangaroo at a specific offset, centered around the range midpoint.
///
/// Maps raw offset in `[0, range)` to centered offset in `[-range/2, range/2)`.
/// Uses `sbb` (subtract-with-borrow) for sign detection in full 256-bit space.
fn init_wild_kangaroo_at_offset(
    pubkey: &Point,
    raw_offset: &K256U256,
    range_middle: &K256U256,
    base_point: &ProjectivePoint,
) -> (k256::AffinePoint, [u32; 8]) {
    // Detect sign: sbb returns borrow != 0 when raw_offset < range_middle
    let (diff, borrow) = raw_offset.sbb(range_middle, k256::elliptic_curve::bigint::Limb::ZERO);
    let is_negative = borrow != k256::elliptic_curve::bigint::Limb::ZERO;

    if !is_negative {
        // raw_offset >= range_middle: positive direction
        // diff = raw_offset - range_middle (exact, no wrap)
        let scalar = Scalar::reduce(diff);
        let offset_point = *base_point * scalar;
        let wild_point = *pubkey + offset_point;

        let delta_be = u256_to_be_bytes(&diff);
        (wild_point.to_affine(), scalar_be_to_limbs(&delta_be))
    } else {
        // raw_offset < range_middle: negative direction
        let delta = range_middle.wrapping_sub(raw_offset);
        let scalar = Scalar::reduce(delta);
        let offset_point = *base_point * scalar;
        let wild_point = *pubkey - offset_point;

        // Store negative offset as two's complement
        let delta_be = u256_to_be_bytes(&delta);
        let neg_bytes = negate_256_be(&delta_be);
        (wild_point.to_affine(), scalar_be_to_limbs(&neg_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wild2_init_with_negated_pubkey() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let neg_pubkey = pubkey.neg();

        let range_middle = K256U256::ONE.shl_vartime(19);
        let offset = K256U256::from(1000u64);

        let (point, dist) = init_wild_kangaroo_at_offset(
            &neg_pubkey,
            &offset,
            &range_middle,
            &ProjectivePoint::GENERATOR,
        );

        let gpu_point = affine_to_gpu(&point);

        assert!(
            gpu_point.x.iter().any(|&x| x != 0),
            "GPU point x should have at least one non-zero element"
        );
        assert!(
            dist.iter().any(|&d| d != 0),
            "Distance should have at least one non-zero element"
        );
    }

    #[test]
    fn test_three_set_distribution() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 20u32;
        let num_kangaroos = 4096u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();
        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        let tame = kangaroos.iter().filter(|k| k.ktype == 0).count();
        let wild1 = kangaroos.iter().filter(|k| k.ktype == 1).count();
        let wild2 = kangaroos.iter().filter(|k| k.ktype == 2).count();

        assert_eq!(tame, 1365);
        assert_eq!(wild1, 1365);
        assert_eq!(wild2, 1366);
        assert_eq!(tame + wild1 + wild2, num_kangaroos as usize);
    }

    #[test]
    fn test_minimum_kangaroo_count() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result =
            initialize_kangaroos(&pubkey, &start, 20, 2, &ProjectivePoint::GENERATOR, 0, 2);
        assert!(result.is_err(), "Should fail for num_kangaroos < 3");
        assert!(result.unwrap_err().to_string().contains("at least 3"));
    }

    #[test]
    fn test_multi_gpu_workers_have_disjoint_initial_states() {
        use std::collections::HashSet;

        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 24u32;
        let per_gpu_k = 512u32;
        let total_k = per_gpu_k * 2;

        let gpu0 = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            per_gpu_k,
            &ProjectivePoint::GENERATOR,
            0,
            total_k,
        )
        .unwrap();

        let gpu1 = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            per_gpu_k,
            &ProjectivePoint::GENERATOR,
            per_gpu_k,
            total_k,
        )
        .unwrap();

        let gpu0_states: HashSet<([u32; 8], [u32; 8], [u32; 8], u32)> = gpu0
            .iter()
            .map(|k| (k.x, k.y, k.dist, k.ktype as u32))
            .collect();

        for k in &gpu1 {
            let state = (k.x, k.y, k.dist, k.ktype as u32);
            assert!(
                !gpu0_states.contains(&state),
                "GPU workers must not share initial kangaroo states"
            );
        }
    }

    /// Verify initialization works correctly for range_bits >= 128.
    /// This is the core regression test for issue #70.
    #[test]
    fn test_initialize_kangaroos_large_range() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 135u32;
        let num_kangaroos = 6u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();

        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        for k in &kangaroos {
            assert!(
                k.x.iter().any(|&v| v != 0),
                "kangaroo should have non-zero position"
            );
        }

        assert!(kangaroos.iter().any(|k| k.ktype == 0));
        assert!(kangaroos.iter().any(|k| k.ktype == 1));
        assert!(kangaroos.iter().any(|k| k.ktype == 2));
    }

    /// Verify initialization at range_bits = 200 (well above u128 limit).
    #[test]
    fn test_initialize_kangaroos_200bit_range() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let range_bits = 200u32;
        let num_kangaroos = 9u32;

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &ProjectivePoint::GENERATOR,
            0,
            num_kangaroos,
        )
        .unwrap();

        assert_eq!(kangaroos.len(), num_kangaroos as usize);

        for k in &kangaroos {
            assert!(
                k.x.iter().any(|&v| v != 0),
                "kangaroo should have non-zero position"
            );
            assert_eq!(k.is_active, 1);
        }
    }

    /// range_bits > 255 must fail (2^256 overflows U256).
    #[test]
    fn test_range_bits_overflow_rejected() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result =
            initialize_kangaroos(&pubkey, &start, 256, 6, &ProjectivePoint::GENERATOR, 0, 6);
        assert!(result.is_err());
    }

    /// range_bits = 0 must fail.
    #[test]
    fn test_range_bits_zero_rejected() {
        let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
        let pubkey = crate::crypto::parse_pubkey(pubkey_hex).expect("Failed to parse pubkey");
        let start = [0u8; 32];
        let result = initialize_kangaroos(&pubkey, &start, 0, 6, &ProjectivePoint::GENERATOR, 0, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_u256_be_bytes_roundtrip() {
        let val = K256U256::from(0xDEAD_BEEF_u64);
        let be = u256_to_be_bytes(&val);
        let recovered = K256U256::from_be_slice(&be);
        assert_eq!(val, recovered);

        let large = K256U256::ONE.shl_vartime(200);
        let be2 = u256_to_be_bytes(&large);
        let recovered2 = K256U256::from_be_slice(&be2);
        assert_eq!(large, recovered2);
    }
}
