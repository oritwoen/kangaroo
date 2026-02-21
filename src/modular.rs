//! Modular constraint filtering for the Kangaroo ECDLP solver
//!
//! When the private key k is known to satisfy k ≡ R (mod M), the search space
//! can be reduced by a factor of M. Instead of searching for k directly,
//! we search for j such that k = R + M*j, transforming:
//!   - Base point: G → H = M*G
//!   - Public key: P → Q = P - R*G
//!   - Range: 2^range_bits → 2^range_bits / M

use anyhow::{anyhow, Result};
use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};

/// Modular constraint parameters for transformed ECDLP search.
///
/// If k ≡ R (mod M), substitute k = R + M*j and solve j*H = Q
/// where H = M*G and Q = P - R*G.
pub struct ModConstraint {
    /// H = M * G — the new base point
    pub base_point: ProjectivePoint,
    /// Q = P - R * G — the transformed public key
    pub transformed_pubkey: ProjectivePoint,
    /// Starting index j = ceil((start - R) / M) as LE bytes
    pub j_start: [u8; 32],
    /// Bits needed for reduced range: ~range_bits - log2(M)
    pub effective_range_bits: u32,
    /// M as Scalar
    pub mod_step: Scalar,
    /// R as Scalar (0 ≤ R < M)
    pub mod_start: Scalar,
}

impl ModConstraint {
    /// Create a new modular constraint.
    ///
    /// # Arguments
    /// - `mod_step_hex`: M as hex string (e.g. "7" for M=7). Must be ≥ 1.
    /// - `mod_start_hex`: R as hex string (e.g. "0" for R=0). Must be 0 ≤ R < M.
    /// - `pubkey`: original public key P
    /// - `start`: search range start as LE bytes [u8; 32]
    /// - `range_bits`: original range bits
    ///
    /// # Returns
    /// - `Ok(None)` if M=1 and R=0 (no constraint, caller uses default path)
    /// - `Ok(Some(constraint))` if M > 1
    /// - `Err(...)` if M=0, R >= M, or hex parse fails
    pub fn new(
        mod_step_hex: &str,
        mod_start_hex: &str,
        pubkey: &ProjectivePoint,
        start: &[u8; 32],
        range_bits: u32,
    ) -> Result<Option<Self>> {
        let m_u64 = u64::from_str_radix(mod_step_hex.trim_start_matches("0x"), 16)
            .map_err(|e| anyhow!("Invalid hex for mod_step M: {e}"))?;
        let r_u64 = u64::from_str_radix(mod_start_hex.trim_start_matches("0x"), 16)
            .map_err(|e| anyhow!("Invalid hex for mod_start R: {e}"))?;

        if m_u64 == 0 {
            return Err(anyhow!("mod_step M must be >= 1, got 0"));
        }
        if r_u64 >= m_u64 {
            return Err(anyhow!(
                "mod_start R must be < M, got R={r_u64} >= M={m_u64}"
            ));
        }

        // M=1, R=0 → identity constraint, no transformation needed
        if m_u64 == 1 {
            return Ok(None);
        }

        let mod_step = <Scalar as Reduce<K256U256>>::reduce(K256U256::from(m_u64));
        let mod_start = <Scalar as Reduce<K256U256>>::reduce(K256U256::from(r_u64));

        // H = M * G
        let base_point = ProjectivePoint::mul_by_generator(&mod_step);

        // Q = P - R * G
        let transformed_pubkey = if r_u64 == 0 {
            *pubkey
        } else {
            let r_g = ProjectivePoint::mul_by_generator(&mod_start);
            *pubkey - r_g
        };

        // j_start = ceil((start - R) / M) using full 256-bit arithmetic
        // start is LE [u8; 32], M and R fit in u64
        let diff = sub_u64_from_u256_le(start, r_u64);
        let (quotient, remainder) = div_u256_le_by_u64(&diff, m_u64);
        // ceil: if remainder > 0, add 1
        let j_start = if remainder > 0 {
            add_one_u256_le(&quotient)
        } else {
            quotient
        };

        // effective_range_bits = range_bits - floor(log2(M)), minimum 1
        let log2_m = 63u32 - m_u64.leading_zeros();
        let effective_range_bits = range_bits.saturating_sub(log2_m).max(1);

        Ok(Some(Self {
            base_point,
            transformed_pubkey,
            j_start,
            effective_range_bits,
            mod_step,
            mod_start,
        }))
    }
}

/// Subtract a u64 from a 256-bit LE number, saturating at zero.
fn sub_u64_from_u256_le(le_bytes: &[u8; 32], val: u64) -> [u8; 32] {
    let mut result = *le_bytes;
    let mut borrow = val as u128;
    for chunk in 0..4 {
        if borrow == 0 {
            break;
        }
        let offset = chunk * 8;
        let limb = u64::from_le_bytes(result[offset..offset + 8].try_into().unwrap()) as u128;
        if limb >= borrow {
            result[offset..offset + 8].copy_from_slice(&((limb - borrow) as u64).to_le_bytes());
            borrow = 0;
        } else {
            let diff = (1u128 << 64) + limb - borrow;
            result[offset..offset + 8].copy_from_slice(&(diff as u64).to_le_bytes());
            borrow = 1;
        }
    }
    result
}

/// Divide a 256-bit LE number by a u64. Returns (quotient_le, remainder).
fn div_u256_le_by_u64(le_bytes: &[u8; 32], divisor: u64) -> ([u8; 32], u64) {
    let mut result = [0u8; 32];
    let mut remainder: u128 = 0;
    let d = divisor as u128;
    // Process from most significant limb to least significant
    for chunk in (0..4).rev() {
        let offset = chunk * 8;
        let limb = u64::from_le_bytes(le_bytes[offset..offset + 8].try_into().unwrap()) as u128;
        let combined = (remainder << 64) | limb;
        result[offset..offset + 8].copy_from_slice(&((combined / d) as u64).to_le_bytes());
        remainder = combined % d;
    }
    (result, remainder as u64)
}

/// Add 1 to a 256-bit LE number.
fn add_one_u256_le(le_bytes: &[u8; 32]) -> [u8; 32] {
    let mut result = *le_bytes;
    for chunk in 0..4 {
        let offset = chunk * 8;
        let limb = u64::from_le_bytes(result[offset..offset + 8].try_into().unwrap());
        let (sum, overflow) = limb.overflowing_add(1);
        result[offset..offset + 8].copy_from_slice(&sum.to_le_bytes());
        if !overflow {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::parse_pubkey;

    #[test]
    fn test_mod_constraint_identity() {
        // M=1, R=0 → returns None (no constraint)
        let pubkey =
            parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c")
                .unwrap();
        let start = [0u8; 32];
        let result = ModConstraint::new("1", "0", &pubkey, &start, 20).unwrap();
        assert!(result.is_none(), "M=1,R=0 should return None");
    }

    #[test]
    fn test_mod_constraint_m7_r0() {
        // M=7, R=0, puzzle 20 pubkey
        // H = 7*G, Q = P - 0*G = P
        let pubkey =
            parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c")
                .unwrap();
        // start = 0x80000 = 524288. LE: [0x00, 0x00, 0x08, 0x00, ...]
        let mut start = [0u8; 32];
        start[2] = 0x08;

        let constraint = ModConstraint::new("7", "0", &pubkey, &start, 20).unwrap();
        assert!(constraint.is_some(), "M=7,R=0 should return Some");
        let c = constraint.unwrap();

        // H = 7*G
        let seven = <Scalar as Reduce<K256U256>>::reduce(K256U256::from(7u64));
        let expected_h = ProjectivePoint::mul_by_generator(&seven);
        assert_eq!(c.base_point, expected_h, "base_point should be 7*G");

        // Q = P (since R=0)
        assert_eq!(
            c.transformed_pubkey, pubkey,
            "transformed_pubkey should equal pubkey when R=0"
        );

        // effective_range_bits: floor(log2(7)) = 2, so 20 - 2 = 18
        assert_eq!(
            c.effective_range_bits, 18,
            "effective_range_bits should be 18 for M=7"
        );
    }

    #[test]
    fn test_mod_constraint_invalid_r_ge_m() {
        let pubkey =
            parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c")
                .unwrap();
        let start = [0u8; 32];
        let result = ModConstraint::new("3", "3", &pubkey, &start, 20);
        assert!(result.is_err(), "R >= M should return Err");
    }

    #[test]
    fn test_mod_constraint_m0_invalid() {
        let pubkey =
            parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c")
                .unwrap();
        let start = [0u8; 32];
        let result = ModConstraint::new("0", "0", &pubkey, &start, 20);
        assert!(result.is_err(), "M=0 should return Err");
    }
}
