//! 256-bit arithmetic utilities for kangaroo algorithm.

/// Add two 256-bit values (little-endian bytes).
#[allow(dead_code)]
pub fn add_256(a: &[u8], b: &[u8], result: &mut [u8]) {
    debug_assert!(a.len() >= 32 && b.len() >= 32 && result.len() >= 32);

    let mut carry = 0u16;
    for i in 0..32 {
        let sum = u16::from(a[i]) + u16::from(b[i]) + carry;
        result[i] = sum as u8;
        carry = sum >> 8;
    }
}

/// Subtract two 256-bit values: result = a - b (little-endian bytes).
#[allow(dead_code)]
pub fn subtract_256(a: &[u8], b: &[u8], result: &mut [u8]) {
    debug_assert!(a.len() >= 32 && b.len() >= 32 && result.len() >= 32);

    let mut borrow = 0i16;
    for i in 0..32 {
        let diff = i16::from(a[i]) - i16::from(b[i]) - borrow;
        if diff < 0 {
            result[i] = (diff + 256) as u8;
            borrow = 1;
        } else {
            result[i] = diff as u8;
            borrow = 0;
        }
    }
}

/// Negate a 256-bit value using two's complement.
/// Returns -value mod 2^256 (big-endian input/output).
pub fn negate_256_be(bytes: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 1u16;

    // Two's complement: invert all bits and add 1
    // Process big-endian: index 31 is least significant
    for i in 0..32 {
        let idx = 31 - i;
        let inverted = !bytes[idx];
        let sum = inverted as u16 + carry;
        result[idx] = sum as u8;
        carry = sum >> 8;
    }

    result
}

/// Create DP mask with `dp_bits` trailing zeros required.
/// Returns mask in little-endian [u32; 8] format.
#[allow(dead_code)]
pub fn create_dp_mask(dp_bits: u32) -> [u32; 8] {
    let mut mask = [0u32; 8];

    let full_limbs = (dp_bits / 32) as usize;
    let remaining_bits = dp_bits % 32;

    // Set full limbs from the bottom (index 0 up)
    for limb in mask.iter_mut().take(full_limbs.min(8)) {
        *limb = 0xFFFF_FFFF;
    }

    // Partial limb
    if remaining_bits > 0 && full_limbs < 8 {
        mask[full_limbs] = (1u32 << remaining_bits) - 1;
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_256_simple() {
        let a = [1u8; 32];
        let b = [1u8; 32];
        let mut result = [0u8; 32];
        add_256(&a, &b, &mut result);
        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_add_256_carry() {
        let mut a = [0u8; 32];
        a[0] = 0xFF;
        let mut b = [0u8; 32];
        b[0] = 0x01;
        let mut result = [0u8; 32];
        add_256(&a, &b, &mut result);
        assert_eq!(result[0], 0x00);
        assert_eq!(result[1], 0x01);
    }

    #[test]
    fn test_subtract_256_simple() {
        let mut a = [0u8; 32];
        a[0] = 5;
        let mut b = [0u8; 32];
        b[0] = 3;
        let mut result = [0u8; 32];
        subtract_256(&a, &b, &mut result);
        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_subtract_256_borrow() {
        let mut a = [0u8; 32];
        a[1] = 1; // 256 in LE
        let mut b = [0u8; 32];
        b[0] = 1;
        let mut result = [0u8; 32];
        subtract_256(&a, &b, &mut result);
        assert_eq!(result[0], 0xFF);
        assert_eq!(result[1], 0x00);
    }

    #[test]
    fn test_negate_256_be() {
        // Negate 1 should give -1 (all 0xFF in two's complement)
        let mut one = [0u8; 32];
        one[31] = 1; // big-endian: LSB at end
        let neg = negate_256_be(&one);
        assert!(neg.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_create_dp_mask_8_bits() {
        let mask = create_dp_mask(8);
        assert_eq!(mask[0], 0xFF);
        assert_eq!(mask[1], 0);
    }

    #[test]
    fn test_create_dp_mask_32_bits() {
        let mask = create_dp_mask(32);
        assert_eq!(mask[0], 0xFFFF_FFFF);
        assert_eq!(mask[1], 0);
    }

    #[test]
    fn test_create_dp_mask_40_bits() {
        let mask = create_dp_mask(40);
        assert_eq!(mask[0], 0xFFFF_FFFF);
        assert_eq!(mask[1], 0xFF);
        assert_eq!(mask[2], 0);
    }
}
