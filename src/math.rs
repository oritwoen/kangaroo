//! 256-bit arithmetic utilities for kangaroo algorithm.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negate_256_be() {
        // Negate 1 should give -1 (all 0xFF in two's complement)
        let mut one = [0u8; 32];
        one[31] = 1; // big-endian: LSB at end
        let neg = negate_256_be(&one);
        assert!(neg.iter().all(|&b| b == 0xFF));
    }
}
