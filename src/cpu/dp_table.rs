//! Distinguished Point hash table for collision detection

use crate::crypto::verify_key_with_base;
use crate::gpu::GpuDistinguishedPoint;
use k256::elliptic_curve::ops::Reduce;
use k256::{ProjectivePoint, Scalar, U256 as K256U256};
use std::collections::HashMap;

const MAX_DISTINGUISHED_POINTS: usize = 65_536;

/// SCALAR_HALF = (n+1)/2 where n is secp256k1 order
/// Property: 2 × SCALAR_HALF ≡ 1 (mod n)
/// Used for resolving wild₁↔wild₂ collisions via k = (d₂ - d₁) × SCALAR_HALF
pub(crate) fn scalar_half() -> Scalar {
    let bytes = [
        0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d, 0xdf, 0xe9, 0x2f, 0x46, 0x68, 0x1b,
        0x20, 0xa1,
    ];
    Scalar::reduce(K256U256::from_be_slice(&bytes))
}

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
    base_point: ProjectivePoint,
    total_dps: usize,
}

impl DPTable {
    pub fn new(start: [u8; 32], pubkey: ProjectivePoint, base_point: ProjectivePoint) -> Self {
        Self {
            table: HashMap::new(),
            start,
            pubkey,
            base_point,
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
            let ktype_str = match dp.ktype {
                0 => "tame",
                1 => "wild1",
                2 => "wild2",
                _ => {
                    tracing::warn!("DP[{}] unknown ktype={}, skipping", total, dp.ktype);
                    return None;
                }
            };
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
                    let ktype_str = match dp.ktype {
                        0 => "tame-tame",
                        1 => "wild1-wild1",
                        2 => "wild2-wild2",
                        _ => "unknown-unknown",
                    };
                    tracing::debug!(
                        "Same-type collision ({}): affine_x={}",
                        ktype_str,
                        hex::encode(&affine_x[..8])
                    );
                    return None;
                }

                if existing.ktype != 0 && dp.ktype != 0 && existing.ktype != dp.ktype {
                    let (d1, d2) = if existing.ktype == 1 {
                        (existing.dist.as_slice(), dist_bytes.as_ref())
                    } else {
                        (dist_bytes.as_ref(), existing.dist.as_slice())
                    };
                    if let Some(key) =
                        compute_candidate_keys_cross_wild(d1, d2, &self.pubkey, &self.base_point)
                    {
                        tracing::info!("Cross-wild collision found! Key: 0x{}", hex::encode(&key));
                        return Some(key);
                    }
                    tracing::debug!("Spurious cross-wild collision: no candidate key verifies");
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
                    if verify_key_with_base(candidate, &self.pubkey, &self.base_point) {
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

    pub fn count_by_type(&self) -> (usize, usize, usize) {
        let mut tame = 0usize;
        let mut w1 = 0usize;
        let mut w2 = 0usize;
        for list in self.table.values() {
            for dp in list {
                match dp.ktype {
                    0 => tame += 1,
                    1 => w1 += 1,
                    2 => w2 += 1,
                    _ => {} // silently skip unknown types
                }
            }
        }
        (tame, w1, w2)
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

pub(crate) fn compute_candidate_scalars(
    base: Scalar,
    tame_d: Scalar,
    wild_d: Scalar,
) -> [Scalar; 8] {
    let neg_base = Scalar::ZERO - base;
    [
        base + tame_d - wild_d,
        base - tame_d - wild_d,
        base + tame_d + wild_d,
        base - tame_d + wild_d,
        neg_base + tame_d - wild_d,
        neg_base - tame_d - wild_d,
        neg_base + tame_d + wild_d,
        neg_base - tame_d + wild_d,
    ]
}

fn compute_candidate_keys(start: &[u8; 32], tame_dist: &[u8], wild_dist: &[u8]) -> Vec<Vec<u8>> {
    let start_uint = K256U256::from_le_slice(start);
    let start_scalar = Scalar::reduce(start_uint);

    let tame_pair = distance_scalar_pair(&pad_to_32(tame_dist));
    let wild_pair = distance_scalar_pair(&pad_to_32(wild_dist));

    let mut keys = Vec::with_capacity(32);
    for &td in &tame_pair {
        for &wd in &wild_pair {
            for candidate in &compute_candidate_scalars(start_scalar, td, wd) {
                keys.push(scalar_to_key_bytes(candidate));
            }
        }
    }
    keys
}

fn compute_candidate_keys_cross_wild(
    d1_bytes: &[u8],
    d2_bytes: &[u8],
    pubkey: &ProjectivePoint,
    base_point: &ProjectivePoint,
) -> Option<Vec<u8>> {
    let d1_pair = distance_scalar_pair(&pad_to_32(d1_bytes));
    let d2_pair = distance_scalar_pair(&pad_to_32(d2_bytes));
    let half = scalar_half();

    for &d1 in &d1_pair {
        for &d2 in &d2_pair {
            let k_diff = (d2 - d1) * half;
            let candidates = [k_diff, Scalar::ZERO - k_diff];

            for candidate in &candidates {
                let key_bytes = scalar_to_key_bytes(candidate);
                if verify_key_with_base(&key_bytes, pubkey, base_point) {
                    return Some(key_bytes);
                }
            }
        }
    }

    None
}

fn pad_to_32(bytes: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let len = bytes.len().min(32);
    result[..len].copy_from_slice(&bytes[..len]);
    result
}

/// Two scalar interpretations of a GPU distance: direct and negated.
///
/// GPU distances use unsigned 256-bit wrapping (mod 2^256), but we need
/// mod n scalars. We can't distinguish a positive distance from a wrapped
/// negative without extra metadata, so we return both interpretations and
/// let candidate key verification determine which is correct.
///
/// - `[0]` = `Scalar::reduce(v)` — correct when distance didn't wrap
/// - `[1]` = `-(Scalar::reduce(2^256 - v))` — correct when distance wrapped negative
fn distance_scalar_pair(dist_le_bytes: &[u8; 32]) -> [Scalar; 2] {
    let uint = K256U256::from_le_slice(dist_le_bytes);
    let direct = <Scalar as Reduce<K256U256>>::reduce(uint);

    let neg_bytes = negate_u256_bytes(dist_le_bytes);
    let neg_uint = K256U256::from_le_slice(&neg_bytes);
    let negated = Scalar::ZERO - <Scalar as Reduce<K256U256>>::reduce(neg_uint);

    [direct, negated]
}

/// Two's complement negation of a 256-bit LE value: returns `2^256 - value`.
fn negate_u256_bytes(bytes: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 1u16;
    for i in 0..32 {
        let val = (!bytes[i]) as u16 + carry;
        result[i] = val as u8;
        carry = val >> 8;
    }
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
    use crate::crypto::{parse_pubkey, verify_key};
    use crate::gpu::GpuDistinguishedPoint;
    use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    use k256::{ProjectivePoint, Scalar, U256 as K256U256};

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
        let mut table = DPTable::new(
            [0u8; 32],
            ProjectivePoint::GENERATOR,
            ProjectivePoint::GENERATOR,
        );

        for i in 0..MAX_DISTINGUISHED_POINTS {
            let dp = make_dp(i as u32, i as u32, 0);
            assert!(table.insert_and_check(dp).is_none());
        }

        assert_eq!(table.total_dps(), MAX_DISTINGUISHED_POINTS);

        let overflow_dp = make_dp(u32::MAX, u32::MAX, 1);
        assert!(table.insert_and_check(overflow_dp).is_none());
        assert_eq!(table.total_dps(), MAX_DISTINGUISHED_POINTS);
    }

    // --- Helpers for crypto-aware DP tests ---

    fn scalar_from_u64(val: u64) -> Scalar {
        let mut le_bytes = [0u8; 32];
        le_bytes[..8].copy_from_slice(&val.to_le_bytes());
        let uint = K256U256::from_le_slice(&le_bytes);
        <Scalar as Reduce<K256U256>>::reduce(uint)
    }

    fn scalar_to_le_bytes(s: &Scalar) -> [u8; 32] {
        let be = s.to_bytes();
        let mut le = [0u8; 32];
        for i in 0..32 {
            le[i] = be[31 - i];
        }
        le
    }

    fn scalar_to_dist_u32(s: &Scalar) -> [u32; 8] {
        let le = scalar_to_le_bytes(s);
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] =
                u32::from_le_bytes([le[i * 4], le[i * 4 + 1], le[i * 4 + 2], le[i * 4 + 3]]);
        }
        result
    }

    fn point_to_x_u32(p: &ProjectivePoint) -> [u32; 8] {
        let affine = p.to_affine();
        let encoded = affine.to_encoded_point(false);
        let x_bytes = encoded.x().unwrap();
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[7 - i] = u32::from_be_bytes([
                x_bytes[i * 4],
                x_bytes[i * 4 + 1],
                x_bytes[i * 4 + 2],
                x_bytes[i * 4 + 3],
            ]);
        }
        result
    }

    fn make_real_dp(
        collision_point: &ProjectivePoint,
        dist: &Scalar,
        ktype: u32,
    ) -> GpuDistinguishedPoint {
        GpuDistinguishedPoint {
            x: point_to_x_u32(collision_point),
            dist: scalar_to_dist_u32(dist),
            ktype,
            kangaroo_id: 0,
            _padding: [0u32; 6],
        }
    }

    fn assert_solves_with_dps(
        start_s: Scalar,
        k: Scalar,
        tame_dist: Scalar,
        wild_dist: Scalar,
        tame_point: ProjectivePoint,
        wild_point: ProjectivePoint,
    ) {
        let pubkey = ProjectivePoint::mul_by_generator(&k);
        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&tame_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&wild_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(result.is_some(), "Collision should resolve to a valid key");
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    // --- Four-formula collision tests ---

    #[test]
    fn test_four_formula_collision_case1() {
        // Formula: k = start + tame_dist - wild_dist
        // k=66, start=50, tame_dist=30, wild_dist=14 → 50+30-14=66 ✓
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(50);
        let tame_dist = scalar_from_u64(30);
        let wild_dist = scalar_from_u64(14);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Collision: tame lands at (start+tame_dist)*G, wild at (k+wild_dist)*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(k + wild_dist))
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&collision_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should find key via k = start + tame_dist - wild_dist"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    #[test]
    fn test_four_formula_collision_case2() {
        // Formula: k = start - tame_dist - wild_dist (tame negated)
        // k=66, start=100, tame_dist=20, wild_dist=14 → 100-20-14=66 ✓
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(100);
        let tame_dist = scalar_from_u64(20);
        let wild_dist = scalar_from_u64(14);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Tame walked negated: (start-tame_dist)*G = 80*G
        // Wild walked normal: (k+wild_dist)*G = 80*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(k + wild_dist))
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&collision_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should find key via k = start - tame_dist - wild_dist"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    #[test]
    fn test_four_formula_collision_case3() {
        // Formula: k = start + tame_dist + wild_dist (wild negated)
        // k=66, start=30, tame_dist=20, wild_dist=16 → 30+20+16=66 ✓
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(30);
        let tame_dist = scalar_from_u64(20);
        let wild_dist = scalar_from_u64(16);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Tame walked normal: (start+tame_dist)*G = 50*G
        // Wild walked negated: (k-wild_dist)*G = 50*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(k - wild_dist))
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&collision_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should find key via k = start + tame_dist + wild_dist"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    #[test]
    fn test_four_formula_collision_case4() {
        // Formula: k = start - tame_dist + wild_dist (tame negated, wild negated)
        // k=66, start=80, tame_dist=30, wild_dist=16 → 80-30+16=66 ✓
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(80);
        let tame_dist = scalar_from_u64(30);
        let wild_dist = scalar_from_u64(16);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Tame walked negated: (start-tame_dist)*G = 50*G
        // Wild walked negated: (k-wild_dist)*G = 50*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(k - wild_dist))
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&collision_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should find key via k = start - tame_dist + wild_dist"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    #[test]
    fn test_eight_formula_case1() {
        let start_s = scalar_from_u64(50);
        let tame_dist = scalar_from_u64(30);
        let wild_dist = scalar_from_u64(14);
        let k = start_s + tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case2() {
        let start_s = scalar_from_u64(100);
        let tame_dist = scalar_from_u64(20);
        let wild_dist = scalar_from_u64(14);
        let k = start_s - tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case3() {
        let start_s = scalar_from_u64(30);
        let tame_dist = scalar_from_u64(20);
        let wild_dist = scalar_from_u64(16);
        let k = start_s + tame_dist + wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k - wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case4() {
        let start_s = scalar_from_u64(80);
        let tame_dist = scalar_from_u64(30);
        let wild_dist = scalar_from_u64(16);
        let k = start_s - tame_dist + wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k - wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case5() {
        let start_s = scalar_from_u64(90);
        let tame_dist = scalar_from_u64(11);
        let wild_dist = scalar_from_u64(7);
        let neg_start = Scalar::ZERO - start_s;
        let k = neg_start + tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, -wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case6() {
        let start_s = scalar_from_u64(95);
        let tame_dist = scalar_from_u64(19);
        let wild_dist = scalar_from_u64(9);
        let neg_start = Scalar::ZERO - start_s;
        let k = neg_start - tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, -wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case7() {
        let start_s = scalar_from_u64(104);
        let tame_dist = scalar_from_u64(15);
        let wild_dist = scalar_from_u64(6);
        let neg_start = Scalar::ZERO - start_s;
        let k = neg_start + tame_dist + wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s - tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k - wild_dist));
        assert_eq!(tame_point, -wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_eight_formula_case8() {
        let start_s = scalar_from_u64(77);
        let tame_dist = scalar_from_u64(13);
        let wild_dist = scalar_from_u64(5);
        let neg_start = Scalar::ZERO - start_s;
        let k = neg_start - tame_dist + wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k - wild_dist));
        assert_eq!(tame_point, -wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_xonly_negation_via_dptable() {
        let start_s = scalar_from_u64(120);
        let tame_dist = scalar_from_u64(21);
        let wild_dist = scalar_from_u64(8);
        let neg_start = Scalar::ZERO - start_s;
        let k = neg_start - tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, -wild_point);

        let tame_x = point_to_x_u32(&tame_point);
        let wild_x = point_to_x_u32(&wild_point);
        assert_eq!(tame_x, wild_x, "x-only DP collision must match");

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_candidate_zero_tame_distance() {
        let start_s = scalar_from_u64(66);
        let tame_dist = Scalar::ZERO;
        let wild_dist = scalar_from_u64(17);
        let k = start_s - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_candidate_zero_wild_distance() {
        let start_s = scalar_from_u64(66);
        let tame_dist = scalar_from_u64(9);
        let wild_dist = Scalar::ZERO;
        let k = start_s + tame_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_candidate_equal_distances() {
        let start_s = scalar_from_u64(73);
        let tame_dist = scalar_from_u64(15);
        let wild_dist = scalar_from_u64(15);
        let k = start_s;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_candidate_zero_start() {
        let start_s = Scalar::ZERO;
        let tame_dist = scalar_from_u64(23);
        let wild_dist = scalar_from_u64(11);
        let k = tame_dist - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(start_s, k, tame_dist, wild_dist, tame_point, wild_point);
    }

    #[test]
    fn test_virtual_dp_flipped_distance() {
        let start_s = scalar_from_u64(100);
        let walk_dist = scalar_from_u64(20);
        let jump_dist = scalar_from_u64(7);
        let tame_dist_virtual = walk_dist + jump_dist;
        let wild_dist = scalar_from_u64(61);
        let k = start_s + tame_dist_virtual - wild_dist;

        let tame_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist_virtual));
        let wild_point = ProjectivePoint::mul_by_generator(&(k + wild_dist));
        assert_eq!(tame_point, wild_point);

        assert_solves_with_dps(
            start_s,
            k,
            tame_dist_virtual,
            wild_dist,
            tame_point,
            wild_point,
        );
    }

    #[test]
    fn test_signed_distance_wrapping() {
        // GPU-style wrapped negative distance via unsigned 256-bit subtraction.
        // GPU produces 2^256 - d (NOT n - d) when distance underflows.
        // k = start + tame_dist - wild_dist = 100 + (-10) - 24 = 66
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(100);
        let wild_dist = scalar_from_u64(24);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Collision at (start - 10)*G = 90*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s - scalar_from_u64(10)));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&scalar_from_u64(90))
        );

        // GPU unsigned wrapping: 2^256 - 10 (bit 255 set → negative in signed interp)
        let tame_dist_u32 = gpu_wrapped_neg_u64(10);
        assert!(
            tame_dist_u32[7] & 0x80000000 != 0,
            "GPU-wrapped distance should have bit 255 set"
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = GpuDistinguishedPoint {
            x: point_to_x_u32(&collision_point),
            dist: tame_dist_u32,
            ktype: 0,
            kangaroo_id: 0,
            _padding: [0u32; 6],
        };
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should resolve collision with GPU unsigned wrapped distance"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    /// Two's complement negate of u64 as `[u32; 8]` LE limbs: returns `2^256 - d`.
    fn gpu_wrapped_neg_u64(d: u64) -> [u32; 8] {
        let mut d_limbs = [0u32; 8];
        d_limbs[0] = d as u32;
        d_limbs[1] = (d >> 32) as u32;

        let mut result = [0u32; 8];
        let mut carry = 1u64;
        for i in 0..8 {
            let val = (!d_limbs[i]) as u64 + carry;
            result[i] = val as u32;
            carry = val >> 32;
        }
        result
    }

    #[test]
    fn test_gpu_unsigned_wrap_both_distances() {
        // Both tame and wild have GPU-wrapped negative distances.
        // k = start + tame_dist - wild_dist = 200 + (-30) - (-8) = 200 - 30 + 8 = 178
        let k = scalar_from_u64(178);
        let start_s = scalar_from_u64(200);

        let tame_dist_u32 = gpu_wrapped_neg_u64(30);
        let wild_dist_u32 = gpu_wrapped_neg_u64(8);

        let pubkey = ProjectivePoint::mul_by_generator(&k);
        // Collision point: (start - 30)*G = 170*G, wild at (k - 8)*G = 170*G
        let collision_point = ProjectivePoint::mul_by_generator(&(start_s - scalar_from_u64(30)));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(k - scalar_from_u64(8)))
        );

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = GpuDistinguishedPoint {
            x: point_to_x_u32(&collision_point),
            dist: tame_dist_u32,
            ktype: 0,
            kangaroo_id: 0,
            _padding: [0u32; 6],
        };
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = GpuDistinguishedPoint {
            x: point_to_x_u32(&collision_point),
            dist: wild_dist_u32,
            ktype: 1,
            kangaroo_id: 0,
            _padding: [0u32; 6],
        };
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_some(),
            "Should resolve collision with both distances GPU-wrapped"
        );
        assert!(verify_key(&result.unwrap(), &pubkey));
    }

    #[test]
    fn test_distance_scalar_pair_unit() {
        // Zero: both interpretations give Scalar::ZERO
        let zero = [0u8; 32];
        let pair = super::distance_scalar_pair(&zero);
        assert_eq!(pair[0], Scalar::ZERO);
        assert_eq!(pair[1], Scalar::ZERO);

        // Positive 42: direct = 42, negated ≠ 42
        let mut pos_le = [0u8; 32];
        pos_le[0] = 42;
        let pair = super::distance_scalar_pair(&pos_le);
        assert_eq!(pair[0], scalar_from_u64(42));
        assert_ne!(pair[1], scalar_from_u64(42));

        // GPU wrapped -10 (2^256 - 10): negated interpretation = n - 10
        let mut neg_ten = [0xFFu8; 32];
        neg_ten[0] = 0xF6;
        let pair = super::distance_scalar_pair(&neg_ten);
        let expected_neg = Scalar::ZERO - scalar_from_u64(10);
        assert_ne!(pair[0], expected_neg); // direct gives wrong result for wrapped
        assert_eq!(pair[1], expected_neg); // negated gives correct n - 10
    }

    #[test]
    fn test_four_formula_no_false_match() {
        // Collision with wrong pubkey — no formula produces a valid key.
        let wrong_pubkey = ProjectivePoint::mul_by_generator(&scalar_from_u64(999));

        let start_s = scalar_from_u64(50);
        let tame_dist = scalar_from_u64(30);
        let wild_dist = scalar_from_u64(14);

        let collision_point = ProjectivePoint::mul_by_generator(&(start_s + tame_dist));

        let start = scalar_to_le_bytes(&start_s);
        let mut table = DPTable::new(start, wrong_pubkey, ProjectivePoint::GENERATOR);

        let tame_dp = make_real_dp(&collision_point, &tame_dist, 0);
        assert!(table.insert_and_check(tame_dp).is_none());

        let wild_dp = make_real_dp(&collision_point, &wild_dist, 1);
        let result = table.insert_and_check(wild_dp);
        assert!(
            result.is_none(),
            "Should return None when no formula produces valid key"
        );
    }

    #[test]
    fn test_scalar_half_inverse_of_two() {
        // Verify that 2 × SCALAR_HALF ≡ 1 (mod n)
        let half = super::scalar_half();
        let two = scalar_from_u64(2);
        let product = two * half;
        assert_eq!(
            product,
            Scalar::ONE,
            "2 × SCALAR_HALF should equal 1 (mod n)"
        );
    }

    #[test]
    fn test_scalar_half_value() {
        // Verify SCALAR_HALF has the correct byte representation
        let half = super::scalar_half();
        let bytes = half.to_bytes();
        let expected = [
            0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d, 0xdf, 0xe9, 0x2f, 0x46,
            0x68, 0x1b, 0x20, 0xa1,
        ];
        assert_eq!(
            bytes.as_slice(),
            &expected,
            "SCALAR_HALF bytes should match expected value"
        );
    }

    #[test]
    fn test_cross_wild_collision_resolves() {
        let pubkey =
            parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c")
                .unwrap();
        let k = scalar_from_u64(0x0d2c55);

        let d1 = scalar_from_u64(123_456);
        let d2 = d1 + (k + k);
        let collision_point = ProjectivePoint::mul_by_generator(&(k + d1));
        assert_eq!(
            collision_point,
            ProjectivePoint::mul_by_generator(&(-k + d2))
        );

        let mut table = DPTable::new([0u8; 32], pubkey, ProjectivePoint::GENERATOR);

        let wild1_dp = make_real_dp(&collision_point, &d1, 1);
        assert!(table.insert_and_check(wild1_dp).is_none());

        let wild2_dp = make_real_dp(&collision_point, &d2, 2);
        let result = table.insert_and_check(wild2_dp);
        assert!(
            result.is_some(),
            "wild1↔wild2 collision should resolve to a valid key"
        );
        assert!(verify_key(&result.clone().unwrap(), &table.pubkey));
        assert_eq!(result.unwrap(), super::scalar_to_key_bytes(&k));
    }

    #[test]
    fn test_cross_wild_no_false_positive() {
        let wrong_pubkey =
            parse_pubkey("031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e")
                .unwrap();
        let k = scalar_from_u64(0x0d2c55);

        let d1 = scalar_from_u64(123_456);
        let d2 = d1 + (k + k);
        let collision_point = ProjectivePoint::mul_by_generator(&(k + d1));

        let mut table = DPTable::new([0u8; 32], wrong_pubkey, ProjectivePoint::GENERATOR);

        let wild1_dp = make_real_dp(&collision_point, &d1, 1);
        assert!(table.insert_and_check(wild1_dp).is_none());

        let wild2_dp = make_real_dp(&collision_point, &d2, 2);
        let result = table.insert_and_check(wild2_dp);
        assert!(
            result.is_none(),
            "cross-wild candidate formulas should not accept wrong pubkey"
        );
    }
}
