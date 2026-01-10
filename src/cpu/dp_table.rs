//! Distinguished Point hash table for collision detection

use crate::gpu::GpuDistinguishedPoint;
use dashmap::DashMap;

/// Stored DP with full affine X for proper verification
#[derive(Clone)]
struct StoredDP {
    affine_x: [u8; 32],
    dist: Vec<u8>,
    ktype: u32,
}

/// Thread-safe DP table for collision detection
pub struct DPTable {
    table: DashMap<u64, Vec<StoredDP>>,
    start: [u8; 32], // search range start for key computation
}

impl DPTable {
    pub fn new(start: [u8; 32]) -> Self {
        Self {
            table: DashMap::new(),
            start,
        }
    }

    /// Insert DP and check for collision
    /// Returns private key if collision found between tame and wild
    pub fn insert_and_check(&self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
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
        if let Some(mut existing_list) = self.table.get_mut(&hash_key) {
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
                let key = compute_private_key(
                    &self.start,
                    &existing.dist,
                    &dist_bytes,
                    existing.ktype,
                    dp.ktype,
                );
                tracing::info!("Collision found! Key: 0x{}", hex::encode(&key));
                return Some(key);
            }
            // No collision, add to list
            existing_list.push(StoredDP {
                affine_x,
                dist: dist_bytes.to_vec(),
                ktype: dp.ktype,
            });
        } else {
            // New hash key
            self.table.insert(
                hash_key,
                vec![StoredDP {
                    affine_x,
                    dist: dist_bytes.to_vec(),
                    ktype: dp.ktype,
                }],
            );
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
        self.table.iter().map(|entry| entry.value().len()).sum()
    }

    pub fn count_by_type(&self) -> (usize, usize) {
        let mut tame = 0;
        let mut wild = 0;
        for entry in &self.table {
            for dp in entry.value() {
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

fn compute_private_key(
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
