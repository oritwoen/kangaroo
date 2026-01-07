//! Cryptographic types and utilities

use anyhow::{Context, Result};
use bitcoin::hashes::{hash160, Hash};
use bitcoin::{Address, CompressedPublicKey, Network, PubkeyHash};
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
    let padded = format!("{:0>64}", hex_clean);

    let bytes = hex::decode(&padded).context("Invalid hex")?;

    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);

    // Convert to little-endian for internal use
    result.reverse();

    Ok(result)
}

/// Convert U256 (little-endian [u8; 32]) back to little-endian bytes
#[allow(dead_code)]
pub fn u256_to_le_bytes(val: &U256) -> [u8; 32] {
    *val
}

/// Verify that private key produces the public key
pub fn verify_key(private_key: &[u8], public_key: &Point) -> bool {
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

    let computed = ProjectivePoint::mul_by_generator(&scalar);
    computed == *public_key
}

/// Compute compressed public key from private key bytes
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn compute_hash160(data: &[u8]) -> [u8; 20] {
    let hash = hash160::Hash::hash(data);
    let mut result = [0u8; 20];
    result.copy_from_slice(hash.as_ref());
    result
}

/// Compute Bitcoin address from compressed public key
#[allow(dead_code)]
pub fn pubkey_to_address(pubkey: &[u8]) -> String {
    let compressed =
        CompressedPublicKey::from_slice(pubkey).expect("Invalid compressed public key");
    Address::p2pkh(compressed, Network::Bitcoin).to_string()
}

/// Compute Bitcoin address from Hash160
#[allow(dead_code)]
pub fn pubkey_hash_to_address(hash: &[u8; 20]) -> String {
    let pubkey_hash = PubkeyHash::from_slice(hash).expect("Invalid hash160");
    Address::p2pkh(pubkey_hash, Network::Bitcoin).to_string()
}

/// Full verification: private key -> pubkey -> hash160 -> address
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Convert U256 to [u32; 8] for GPU
#[allow(dead_code)]
pub fn u256_to_u32_array(val: &U256) -> [u32; 8] {
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] =
            u32::from_le_bytes([val[i * 4], val[i * 4 + 1], val[i * 4 + 2], val[i * 4 + 3]]);
    }
    result
}
