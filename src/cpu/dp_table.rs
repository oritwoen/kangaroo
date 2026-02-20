//! Distinguished Point hash table for collision detection

use crate::crypto::verify_key;
use crate::gpu::GpuDistinguishedPoint;
use k256::elliptic_curve::ops::Reduce;
use k256::{ProjectivePoint, Scalar, U256 as K256U256};
use std::collections::HashMap;

const MAX_DISTINGUISHED_POINTS: usize = 65_536;

/// Stored DP with full affine X for proper verification
#[derive(Clone)]
struct StoredDP {
    affine_x: [u8; 32],
    dist: Vec<u8>,
    ktype: u32,
}

pub struct DPTable {
    table: HashMap<u64, Vec<StoredDP>>,
    start: [u8; 32], // search range start for key computation
    pubkey: ProjectivePoint,
    total_dps: usize,
}

impl DPTable {
    pub fn new(start: [u8; 32], pubkey: ProjectivePoint) -> Self {
        Self {
            table: HashMap::new(),
            start,
            pubkey,
            total_dps: 0,
        }
    }

    /// Insert DP and check for collision
    /// Returns private key if collision found between tame and wild
    pub fn insert_and_check(&mut self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
        let dist_bytes = u32_array_to_bytes(&dp.dist);

        // X is already in affine coordinates (no Z conversion needed)
        let affine_x = u32_array_to_be_bytes(&dp.x);

        // Debug: log first few DPs (only with RUST_LOG=debug)
        let total = self.total_dps();
        if total < 20 {
            let ktype_str = if dp.ktype == 0 { "tame" } else { "wild" };
            tracing::debug!(
                "DP[{}] {}: x[0..2]=[{:08x},{:08x}] dist[0..2]=[{:08x},{:08x}] affine_x[0..4]={}",
                total,
                ktype_str,
                dp.x[0],
                dp.x[1],
                dp.dist[0],
                dp.dist[1],
                hex::encode(&affine_x[..4])
            );
        }

        // Use first 8 bytes of affine X as hash key
        let hash_key = u64::from_le_bytes([
            affine_x[0],
            affine_x[1],
            affine_x[2],
            affine_x[3],
            affine_x[4],
            affine_x[5],
            affine_x[6],
            affine_x[7],
        ]);

        // Check for existing DPs with same hash
        if let Some(existing_list) = self.table.get_mut(&hash_key) {
            for existing in existing_list.iter() {
                // Verify full affine X match (not just hash)
                if existing.affine_x != affine_x {
                    continue;
                }

                // Same affine X - check if tame vs wild collision
                if existing.ktype == dp.ktype {
                    // Same type collision - log for debugging
                    let ktype_str = if dp.ktype == 0 {
                        "tame-tame"
                    } else {
                        "wild-wild"
                    };
                    tracing::debug!(
                        "Same-type collision ({}): affine_x={}",
                        ktype_str,
                        hex::encode(&affine_x[..8])
                    );
                    return None;
                }
                let (tame_dist_bytes, wild_dist_bytes) = if existing.ktype == 0 {
                    (existing.dist.as_slice(), dist_bytes.as_ref())
                } else {
                    (dist_bytes.as_ref(), existing.dist.as_slice())
                };

                let candidates =
                    compute_candidate_keys(&self.start, tame_dist_bytes, wild_dist_bytes);
                for candidate in &candidates {
                    if verify_key(candidate, &self.pubkey) {
                        tracing::info!("Collision found! Key: 0x{}", hex::encode(candidate));
                        return Some(candidate.clone());
                    }
                }

                tracing::debug!("Spurious collision: no candidate key verifies");
                return None;
            }
            // No collision, add to list
            if self.total_dps >= MAX_DISTINGUISHED_POINTS {
                return None;
            }
            existing_list.push(StoredDP {
                affine_x,
                dist: dist_bytes.to_vec(),
                ktype: dp.ktype,
            });
            self.total_dps += 1;
        } else {
            if self.total_dps >= MAX_DISTINGUISHED_POINTS {
                return None;
            }
            // New hash key
            self.table.insert(
                hash_key,
                vec![StoredDP {
                    affine_x,
                    dist: dist_bytes.to_vec(),
                    ktype: dp.ktype,
                }],
            );
            self.total_dps += 1;
        }

        None
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn total_dps(&self) -> usize {
        self.total_dps
    }

    pub fn count_by_type(&self) -> (usize, usize) {
        let mut tame = 0;
        let mut wild = 0;
        for list in self.table.values() {
            for dp in list {
                if dp.ktype == 0 {
                    tame += 1;
                } else {
                    wild += 1;
                }
            }
        }
        (tame, wild)
    }
}

fn u32_array_to_be_bytes(arr: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let limb_bytes = arr[7 - i].to_be_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&limb_bytes);
    }
    bytes
}

fn u32_array_to_bytes(arr: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, &val) in arr.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    bytes
}

#[allow(dead_code)]
fn compute_private_key_legacy(
    start: &[u8; 32],
    dist1: &[u8],
    dist2: &[u8],
    type1: u32,
    _type2: u32,
) -> Vec<u8> {
    // k = start + tame_dist - wild_dist
    // When collision occurs:
    // - Tame position: (start + tame_dist) * G
    // - Wild position: (k + wild_dist) * G
    // - Equal means: start + tame_dist = k + wild_dist
    // - Therefore: k = start + tame_dist - wild_dist
    let mut diff = vec![0u8; 32];

    if type1 == 0 {
        // existing is tame, new is wild
        subtract_256(dist1, dist2, &mut diff);
    } else {
        // existing is wild, new is tame
        subtract_256(dist2, dist1, &mut diff);
    }

    let mut result = vec![0u8; 32];
    add_256(start, &diff, &mut result);

    // Convert to big-endian and trim leading zeros
    result.reverse();
    let first_nonzero = result.iter().position(|&x| x != 0).unwrap_or(31);
    result[first_nonzero..].to_vec()
}

/// Compute 4 candidate private keys for negation map collision resolution.
///
/// When using the negation map, the tame/wild distances can have mixed signs.
/// We try all 4 combinations of (±tame_dist, ±wild_dist) and verify each.
///
/// The canonical formula is: k = start + tame_dist - wild_dist
/// But with negation map, the actual formula might be any of:
///   k = start + tame_dist - wild_dist
///   k = start - tame_dist - wild_dist  (tame walked in negated direction)
///   k = start + tame_dist + wild_dist  (wild walked in negated direction)
///   k = start - tame_dist + wild_dist  (both walked in negated direction)
fn compute_candidate_keys(start: &[u8; 32], tame_dist: &[u8], wild_dist: &[u8]) -> Vec<Vec<u8>> {
    let start_uint = K256U256::from_le_slice(start);
    let start_scalar = Scalar::reduce(start_uint);

    let tame_uint = K256U256::from_le_slice(&pad_to_32(tame_dist));
    let tame_scalar = Scalar::reduce(tame_uint);

    let wild_uint = K256U256::from_le_slice(&pad_to_32(wild_dist));
    let wild_scalar = Scalar::reduce(wild_uint);

    let mut candidates = Vec::with_capacity(4);

    let k1 = start_scalar + tame_scalar - wild_scalar;
    candidates.push(scalar_to_key_bytes(&k1));

    let k2 = start_scalar - tame_scalar - wild_scalar;
    candidates.push(scalar_to_key_bytes(&k2));

    let k3 = start_scalar + tame_scalar + wild_scalar;
    candidates.push(scalar_to_key_bytes(&k3));

    let k4 = start_scalar - tame_scalar + wild_scalar;
    candidates.push(scalar_to_key_bytes(&k4));

    candidates
}

fn pad_to_32(bytes: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let len = bytes.len().min(32);
    result[..len].copy_from_slice(&bytes[..len]);
    result
}

fn scalar_to_key_bytes(scalar: &Scalar) -> Vec<u8> {
    let bytes = scalar.to_bytes();
    let first_nonzero = bytes.iter().position(|&x| x != 0).unwrap_or(31);
    bytes[first_nonzero..].to_vec()
}

fn add_256(a: &[u8], b: &[u8], result: &mut [u8]) {
    let mut carry = 0u16;
    for i in 0..32 {
        let sum = u16::from(a[i]) + u16::from(b[i]) + carry;
        result[i] = sum as u8;
        carry = sum >> 8;
    }
}

fn subtract_256(a: &[u8], b: &[u8], result: &mut [u8]) {
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

#[cfg(test)]
mod tests {
    use super::DPTable;
    use super::MAX_DISTINGUISHED_POINTS;
    use crate::gpu::GpuDistinguishedPoint;
    use k256::ProjectivePoint;

    fn make_dp(x: u32, dist: u32, ktype: u32) -> GpuDistinguishedPoint {
        let mut x_words = [0u32; 8];
        x_words[7] = x;

        let mut dist_words = [0u32; 8];
        dist_words[0] = dist;

        GpuDistinguishedPoint {
            x: x_words,
            dist: dist_words,
            ktype,
            kangaroo_id: 0,
            _padding: [0u32; 6],
        }
    }

    #[test]
    fn insert_and_check_caps_at_maximum() {
        let mut table = DPTable::new([0u8; 32], ProjectivePoint::GENERATOR);

        for i in 0..MAX_DISTINGUISHED_POINTS {
            let dp = make_dp(i as u32, i as u32, 0);
            assert!(table.insert_and_check(dp).is_none());
        }

        assert_eq!(table.total_dps(), MAX_DISTINGUISHED_POINTS);

        let overflow_dp = make_dp(u32::MAX, u32::MAX, 1);
        assert!(table.insert_and_check(overflow_dp).is_none());
        assert_eq!(table.total_dps(), MAX_DISTINGUISHED_POINTS);
    }
}
