//! Cryptographic types and utilities

use anyhow::{Context, Result};
use bitcoin::hashes::{hash160, Hash};
use bitcoin::{Address, Network, PubkeyHash};
use k256::elliptic_curve::ops::MulByGenerator;
use k256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use k256::elliptic_curve::PrimeField;
use k256::{AffinePoint, EncodedPoint, ProjectivePoint, Scalar};

/// 256-bit unsigned integer
pub type U256 = [u8; 32];

/// Elliptic curve point
pub type Point = ProjectivePoint;

/// Parse compressed public key from hex
pub fn parse_pubkey(hex_str: &str) -> Result<Point> {
    let bytes = hex::decode(hex_str.trim_start_matches("0x")).context("Invalid hex in pubkey")?;

    let encoded = EncodedPoint::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("Invalid encoded point: {e}"))?;

    let affine = AffinePoint::from_encoded_point(&encoded);

    if affine.is_some().into() {
        Ok(ProjectivePoint::from(affine.unwrap()))
    } else {
        anyhow::bail!("Point not on curve")
    }
}

/// Parse hex string to U256
pub fn parse_hex_u256(hex_str: &str) -> Result<U256> {
    let hex_clean = hex_str.trim_start_matches("0x");

    if hex_clean.len() > 64 {
        anyhow::bail!(
            "Invalid U256 length: {} hex chars (max 64)",
            hex_clean.len()
        );
    }

    let padded = format!("{:0>64}", hex_clean);

    let bytes = hex::decode(&padded).context("Invalid hex")?;

    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);

    // Convert to little-endian for internal use
    result.reverse();

    Ok(result)
}

/// Verify that private key produces the public key with arbitrary base point
pub fn verify_key_with_base(private_key: &[u8], public_key: &Point, base_point: &Point) -> bool {
    if private_key.is_empty() || private_key.len() > 32 {
        return false;
    }

    // Pad to 32 bytes if shorter (key is big-endian with leading zeros trimmed)
    let mut key_be = [0u8; 32];
    let offset = 32 - private_key.len();
    key_be[offset..].copy_from_slice(private_key);

    let scalar = match Scalar::from_repr_vartime(key_be.into()) {
        Some(s) => s,
        None => return false,
    };

    let computed = *base_point * scalar;
    computed == *public_key
}

/// Verify that private key produces the public key (using generator as base)
pub fn verify_key(private_key: &[u8], public_key: &Point) -> bool {
    verify_key_with_base(private_key, public_key, &ProjectivePoint::GENERATOR)
}

/// Compute compressed public key from private key bytes
pub fn privkey_to_pubkey(private_key: &[u8]) -> Result<Vec<u8>> {
    if private_key.is_empty() || private_key.len() > 32 {
        anyhow::bail!("Invalid private key length");
    }

    // Pad to 32 bytes
    let mut key_be = [0u8; 32];
    let offset = 32 - private_key.len();
    key_be[offset..].copy_from_slice(private_key);

    let scalar = Scalar::from_repr_vartime(key_be.into())
        .ok_or_else(|| anyhow::anyhow!("Invalid scalar"))?;

    let point = ProjectivePoint::mul_by_generator(&scalar);
    let affine = point.to_affine();
    let encoded = affine.to_encoded_point(true); // compressed

    Ok(encoded.as_bytes().to_vec())
}

/// Compute Hash160 (RIPEMD160(SHA256(data)))
pub fn compute_hash160(data: &[u8]) -> [u8; 20] {
    let hash = hash160::Hash::hash(data);
    let mut result = [0u8; 20];
    result.copy_from_slice(hash.as_ref());
    result
}

/// Compute Bitcoin address from Hash160
pub fn pubkey_hash_to_address(hash: &[u8; 20]) -> String {
    let pubkey_hash = PubkeyHash::from_slice(hash).expect("Invalid hash160");
    Address::p2pkh(pubkey_hash, Network::Bitcoin).to_string()
}

/// Full verification: private key -> pubkey -> hash160 -> address
#[derive(Debug)]
pub struct FullVerification {
    pub pubkey_hex: String,
    pub hash160_hex: String,
    pub address: String,
    pub pubkey_match: bool,
    pub hash160_match: bool,
    pub address_match: bool,
}

/// Verify private key produces expected pubkey, hash160, and address
pub fn full_verify(
    private_key: &[u8],
    expected_pubkey: &str,
    expected_hash160: &str,
    expected_address: &str,
) -> Result<FullVerification> {
    // Compute pubkey
    let pubkey = privkey_to_pubkey(private_key)?;
    let pubkey_hex = hex::encode(&pubkey);

    // Compute hash160
    let h160 = compute_hash160(&pubkey);
    let hash160_hex = hex::encode(h160);

    // Compute address
    let address = pubkey_hash_to_address(&h160);

    Ok(FullVerification {
        pubkey_hex: pubkey_hex.clone(),
        hash160_hex: hash160_hex.clone(),
        address: address.clone(),
        pubkey_match: pubkey_hex == expected_pubkey,
        hash160_match: hash160_hex == expected_hash160,
        address_match: address == expected_address,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_key_with_base_generator() {
        // Test that verify_key_with_base with generator base matches verify_key
        // Using puzzle 20 test vector: k=0x1234 (4660 decimal)
        let key_bytes = [0x12u8, 0x34u8];

        // Compute pubkey: Q = k*G
        let mut key_be = [0u8; 32];
        let offset = 32 - key_bytes.len();
        key_be[offset..].copy_from_slice(&key_bytes);
        let scalar = Scalar::from_repr_vartime(key_be.into()).unwrap();
        let pubkey = ProjectivePoint::mul_by_generator(&scalar);

        // Both should return true
        assert!(
            verify_key(&key_bytes, &pubkey),
            "verify_key should return true"
        );
        assert!(
            verify_key_with_base(&key_bytes, &pubkey, &ProjectivePoint::GENERATOR),
            "verify_key_with_base with generator should return true"
        );
    }

    #[test]
    fn test_verify_key_with_base_custom() {
        // Test verify_key_with_base with custom base point H = 3*G
        // Given: H = 3*G, j = 5, Q = j*H = 5*H = 15*G
        // verify_key_with_base(&[5], &Q, &H) should be true
        // verify_key(&[5], &Q) should be false (because 5*G ≠ 15*G)

        let three = Scalar::from(3u64);
        let five = Scalar::from(5u64);

        // H = 3*G
        let h = ProjectivePoint::GENERATOR * three;

        // Q = 5*H = 15*G
        let q = h * five;

        // j_bytes = [5]
        let j_bytes = [5u8];

        // verify_key_with_base(&[5], &Q, &H) should be true
        assert!(
            verify_key_with_base(&j_bytes, &q, &h),
            "verify_key_with_base with custom base H should return true"
        );

        // verify_key(&[5], &Q) should be false
        assert!(
            !verify_key(&j_bytes, &q),
            "verify_key with generator should return false (5*G ≠ 15*G)"
        );
    }

    #[test]
    fn test_parse_hex_u256_rejects_too_long_input() {
        let too_long = "11".repeat(33);
        let err = parse_hex_u256(&too_long).unwrap_err();
        assert!(err.to_string().contains("Invalid U256 length"));
    }

    #[test]
    fn test_parse_hex_u256_accepts_short_hex() {
        let parsed = parse_hex_u256("abcde").unwrap();
        assert_eq!(parsed[0], 0xde);
        assert_eq!(parsed[1], 0xbc);
        assert_eq!(parsed[2], 0x0a);
        assert!(parsed[3..].iter().all(|&b| b == 0));
    }
}
