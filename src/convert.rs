//! Byte and limb conversion utilities for GPU/CPU data exchange.
//!
//! GPU uses little-endian [u32; 8] limbs, while k256 uses big-endian [u8; 32].

use crate::gpu::GpuAffinePoint;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::AffinePoint;

/// Convert GPU limbs (little-endian u32) to big-endian bytes.
///
/// GPU format: limbs[0] contains least significant 32 bits
/// Output: bytes[0] contains most significant byte
pub fn limbs_to_be_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let be = limbs[7 - i].to_be_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&be);
    }
    bytes
}

/// Convert big-endian bytes to GPU limbs (little-endian u32).
///
/// Input: bytes[0] contains most significant byte
/// GPU format: limbs[0] contains least significant 32 bits
pub fn be_bytes_to_limbs(bytes: &[u8; 32]) -> [u32; 8] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_be_bytes([
            bytes[(7 - i) * 4],
            bytes[(7 - i) * 4 + 1],
            bytes[(7 - i) * 4 + 2],
            bytes[(7 - i) * 4 + 3],
        ]);
    }
    limbs
}

/// Convert [u32; 8] limbs to little-endian bytes.
pub fn limbs_to_le_bytes(arr: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, &val) in arr.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert 32-byte big-endian scalar to [u32; 8] little-endian limbs.
pub fn scalar_be_to_limbs(bytes: &[u8; 32]) -> [u32; 8] {
    be_bytes_to_limbs(bytes)
}

/// Convert k256 `AffinePoint` to GPU format.
pub fn affine_to_gpu(point: &AffinePoint) -> GpuAffinePoint {
    let encoded = point.to_encoded_point(false);
    let x_bytes = encoded.x().unwrap();
    let y_bytes = encoded.y().unwrap();

    let mut x_arr = [0u8; 32];
    let mut y_arr = [0u8; 32];
    x_arr.copy_from_slice(x_bytes);
    y_arr.copy_from_slice(y_bytes);

    GpuAffinePoint {
        x: be_bytes_to_limbs(&x_arr),
        y: be_bytes_to_limbs(&y_arr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limbs_to_be_bytes_roundtrip() {
        let original: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let bytes = limbs_to_be_bytes(&original);
        let recovered = be_bytes_to_limbs(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_limbs_to_be_bytes_value() {
        // limbs[0] = 0x01020304 (LE: least significant limb)
        // Should appear at end of big-endian output
        let mut limbs = [0u32; 8];
        limbs[0] = 0x01020304;

        let bytes = limbs_to_be_bytes(&limbs);

        // In BE: this value should be at bytes[28..32]
        assert_eq!(bytes[28], 0x01);
        assert_eq!(bytes[29], 0x02);
        assert_eq!(bytes[30], 0x03);
        assert_eq!(bytes[31], 0x04);
    }

    #[test]
    fn test_scalar_be_to_limbs() {
        let mut bytes = [0u8; 32];
        bytes[31] = 0x42; // Least significant byte in BE

        let limbs = scalar_be_to_limbs(&bytes);

        // Should be in limbs[0] as least significant
        assert_eq!(limbs[0] & 0xFF, 0x42);
    }
}
