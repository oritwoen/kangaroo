//! SSD persistence feature by Cryptosapien34
//! https://github.com/Cryptosapien34
//!
//! Distinguished Point table with SSD persistence
//!
//! # Architecture
//! - **RAM**: lightweight index only (~16 bytes per DP: hash_key + file offset + ktype)
//! - **SSD**: full DP data (68 bytes per DP: 32B affine_x + 32B dist + 4B ktype)
//! - File header (37 bytes): 4B magic + 33B compressed pubkey for validation
//! - On startup: validates pubkey, builds index from existing SSD file
//! - On new DP: write to SSD, add to index, check for collisions
//! - On collision check: read full DP from SSD only when hash_key matches
//!
//! # Persistence
//! DPs are stored in `~/Desktop/kangaroo_dps/dps_range{bits}.bin`
//! Each puzzle gets its own file based on the search range bits.
//! The file header contains the target pubkey — if the pubkey changes,
//! the old file is rejected and a fresh one is created.
//! No DP is ever lost between sessions for the same puzzle.
//!
//! # Memory usage
//! | DPs stored  | Index RAM | SSD size |
//! |-------------|-----------|----------|
//! | 1 million   | ~16 Mo    | ~68 Mo   |
//! | 100 million | ~1.6 Go   | ~6.8 Go  |
//! | 1 billion   | ~16 Go    | ~68 Go   |

use crate::convert::{limbs_to_be_bytes, limbs_to_le_bytes};
use crate::crypto::verify_key_with_base;
use crate::gpu::GpuDistinguishedPoint;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar, U256 as K256U256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

/// Size of one DP record on disk: 32 (affine_x) + 32 (dist) + 4 (ktype) = 68 bytes
const DP_RECORD_SIZE: u64 = 68;

/// File header: 4 bytes magic + 32 bytes compressed pubkey = 36 bytes
const HEADER_MAGIC: &[u8; 4] = b"KDP1";
const HEADER_SIZE: u64 = 37;

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

/// Compressed pubkey bytes (33 bytes: 02/03 prefix + 32 bytes X)
fn pubkey_to_bytes(pubkey: &ProjectivePoint) -> [u8; 33] {
    let affine = pubkey.to_affine();
    let encoded = affine.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    let mut result = [0u8; 33];
    result.copy_from_slice(&bytes[..33]);
    result
}

/// Full DP data — only used temporarily when reading from disk for collision verification
#[derive(Clone, Copy)]
struct StoredDP {
    affine_x: [u8; 32],
    dist: [u8; 32],
    ktype: u32,
}

/// Lightweight disk reference — this is what lives in RAM
/// Only 12 bytes per DP instead of 68
#[derive(Clone, Copy)]
struct DiskRef {
    offset: u64,
    ktype: u32,
}

pub struct DPTable {
    /// Index: hash_key → list of disk references. NO full DP data in RAM.
    index: HashMap<u64, Vec<DiskRef>>,
    start: [u8; 32],
    pubkey: ProjectivePoint,
    base_point: ProjectivePoint,
    total_dps: usize,
    tame_count: usize,
    wild1_count: usize,
    wild2_count: usize,
    /// Buffered writer for appending new DPs to disk
    dp_writer: Option<BufWriter<File>>,
    /// Persistent reader for collision checks (avoids reopening file each time)
    dp_reader: Option<File>,
    /// Path to the DP file on disk
    dp_file_path: String,
    /// Next write offset in the file
    next_offset: u64,
}

impl DPTable {
    /// Create a new DPTable with SSD persistence.
    ///
    /// `range_bits` determines the puzzle-specific file name:
    /// - Puzzle #135 (range_bits=135) → `dps_range135.bin`
    /// - Puzzle #140 (range_bits=140) → `dps_range140.bin`
    /// - etc.
    ///
    /// The file header contains the target pubkey. If the pubkey doesn't match
    /// the existing file (different puzzle with same bit range), the old file
    /// is renamed as backup and a fresh file is created.
    pub fn new(start: [u8; 32], pubkey: ProjectivePoint, base_point: ProjectivePoint, range_bits: u32) -> Self {
        let dp_dir = format!(
            "{}/Desktop/kangaroo_dps",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
        );
        if let Err(e) = fs::create_dir_all(&dp_dir) {
            tracing::error!("Failed to create DP directory {}: {}", dp_dir, e);
        }
        let dp_file_path = format!("{}/dps_range{}.bin", dp_dir, range_bits);

        let mut tbl = Self {
            index: HashMap::new(),
            start,
            pubkey,
            base_point,
            total_dps: 0,
            tame_count: 0,
            wild1_count: 0,
            wild2_count: 0,
            dp_writer: None,
            dp_reader: None,
            dp_file_path: dp_file_path.clone(),
            next_offset: HEADER_SIZE,
        };

        // Validate header and build index, or create fresh file
        tbl.init_file();

        tbl
    }

    /// Initialize the DP file: validate header, build index, open handles.
    fn init_file(&mut self) {
        let pubkey_bytes = pubkey_to_bytes(&self.pubkey);
        let file_exists = std::path::Path::new(&self.dp_file_path).exists();

        if file_exists {
            match self.validate_header(&pubkey_bytes) {
                Ok(true) => {
                    // Header matches — build index from existing data
                    self.build_index();
                }
                Ok(false) => {
                    // Header mismatch — different puzzle, backup and start fresh
                    let backup = format!("{}.backup.{}", self.dp_file_path,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs()
                    );
                    tracing::warn!(
                        "Pubkey mismatch in {} — backing up to {} and starting fresh",
                        self.dp_file_path, backup
                    );
                    let _ = fs::rename(&self.dp_file_path, &backup);
                    self.write_header(&pubkey_bytes);
                }
                Err(e) => {
                    tracing::warn!("Cannot read DP file header: {} — starting fresh", e);
                    self.write_header(&pubkey_bytes);
                }
            }
        } else {
            self.write_header(&pubkey_bytes);
        }

        // Open persistent reader for collision checks
        if let Ok(f) = File::open(&self.dp_file_path) {
            self.dp_reader = Some(f);
        }

        // Open writer for appending new DPs
        if let Ok(f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.dp_file_path)
        {
            tracing::info!("DP persistence: {}", self.dp_file_path);
            self.dp_writer = Some(BufWriter::new(f));
        } else {
            tracing::error!("Cannot open DP file for writing: {}", self.dp_file_path);
        }
    }

    /// Validate the file header: check magic and pubkey match
    fn validate_header(&self, expected_pubkey: &[u8; 33]) -> Result<bool, std::io::Error> {
        let mut file = File::open(&self.dp_file_path)?;
        let mut header = [0u8; 37];

        // File too small for header
        let file_size = file.metadata()?.len();
        if file_size < HEADER_SIZE {
            return Ok(false);
        }

        file.read_exact(&mut header)?;

        // Check magic
        if &header[0..4] != HEADER_MAGIC {
            // Old format file without header — treat as mismatch
            tracing::warn!("DP file has no header (old format) — will backup and recreate");
            return Ok(false);
        }

        let stored_pubkey = &header[4..37];
                
        if stored_pubkey != &expected_pubkey[0..33] {
            return Ok(false);
        }

        Ok(true)
    }

    /// Write file header with magic and pubkey
    fn write_header(&self, pubkey_bytes: &[u8; 33]) {
        match File::create(&self.dp_file_path) {
            Ok(mut f) => {
                let mut header = [0u8; 37];
                header[0..4].copy_from_slice(HEADER_MAGIC);
                header[4..37].copy_from_slice(&pubkey_bytes[0..33]); // Full compressed pubkey (prefix + X)
                if let Err(e) = f.write_all(&header) {
                    tracing::error!("Failed to write DP file header: {}", e);
                }
            }
            Err(e) => {
                tracing::error!("Failed to create DP file: {}", e);
            }
        }
    }

    /// Build lightweight index by scanning the existing DP file.
    /// Only reads hash_key (8 bytes) and ktype (4 bytes) per record.
    /// Full DP data stays on disk.
    fn build_index(&mut self) {
        let file = match File::open(&self.dp_file_path) {
            Ok(f) => f,
            Err(_) => return,
        };

        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_size <= HEADER_SIZE {
            return;
        }

        let data_size = file_size - HEADER_SIZE;
        let expected = data_size / DP_RECORD_SIZE;

        // Check for truncated file
        let remainder = data_size % DP_RECORD_SIZE;
        if remainder != 0 {
            tracing::warn!(
                "DP file may be corrupted: {} trailing bytes (not a multiple of {} byte records). {} complete records will be loaded.",
                remainder, DP_RECORD_SIZE, expected
            );
        }

        if expected == 0 {
            return;
        }

        tracing::info!(
            "Loading index from {} (~{} DPs)...",
            self.dp_file_path, expected
        );

        let mut reader = BufReader::new(file);
        // Skip header
        if reader.seek(SeekFrom::Start(HEADER_SIZE)).is_err() {
            tracing::error!("Failed to seek past header in DP file");
            return;
        }

        let mut buf = [0u8; 68];
        let mut offset: u64 = HEADER_SIZE;

        loop {
            match reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(_) => break,
            }

            let hash_key = u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]);
            let ktype = u32::from_le_bytes([buf[64], buf[65], buf[66], buf[67]]);

            self.index
                .entry(hash_key)
                .or_insert_with(Vec::new)
                .push(DiskRef { offset, ktype });

            match ktype {
                0 => self.tame_count += 1,
                1 => self.wild1_count += 1,
                2 => self.wild2_count += 1,
                _ => {}
            }

            self.total_dps += 1;
            offset += DP_RECORD_SIZE;
        }

        self.next_offset = offset;

        if self.total_dps > 0 {
            let index_mb = (self.index.len() * 40 + self.total_dps * 16) / (1024 * 1024);
            tracing::info!(
                "Index ready: {} DPs, ~{} Mo RAM (tame:{}, wild1:{}, wild2:{})",
                self.total_dps, index_mb, self.tame_count, self.wild1_count, self.wild2_count
            );
        }
    }

    /// Read a full DP record from disk at the given offset.
    /// Uses persistent file handle to avoid reopening on every read.
    fn read_dp_from_disk(&mut self, offset: u64) -> Option<StoredDP> {
        let reader = self.dp_reader.as_mut()?;
        let mut buf = [0u8; 68];
        if reader.seek(SeekFrom::Start(offset)).is_err() {
            tracing::warn!("Failed to seek to offset {} in DP file", offset);
            return None;
        }
        if reader.read_exact(&mut buf).is_err() {
            tracing::warn!("Failed to read DP at offset {} from disk", offset);
            return None;
        }
        let mut affine_x = [0u8; 32];
        let mut dist = [0u8; 32];
        affine_x.copy_from_slice(&buf[0..32]);
        dist.copy_from_slice(&buf[32..64]);
        let ktype = u32::from_le_bytes([buf[64], buf[65], buf[66], buf[67]]);
        Some(StoredDP { affine_x, dist, ktype })
    }

    /// Append a new DP to the SSD file. Logs errors instead of silently discarding.
    fn write_dp_to_disk(&mut self, affine_x: &[u8; 32], dist: &[u8; 32], ktype: u32) -> u64 {
        let offset = self.next_offset;
        if let Some(ref mut writer) = self.dp_writer {
            let mut buf = [0u8; 68];
            buf[0..32].copy_from_slice(affine_x);
            buf[32..64].copy_from_slice(dist);
            buf[64..68].copy_from_slice(&ktype.to_le_bytes());
            if let Err(e) = writer.write_all(&buf) {
                tracing::error!("Failed to write DP to disk: {} — DP may be lost!", e);
                return offset;
            }
            if self.total_dps % 1000 == 0 {
                if let Err(e) = writer.flush() {
                    tracing::error!("Failed to flush DP file: {}", e);
                }
            }
        }
        self.next_offset += DP_RECORD_SIZE;
        offset
    }

    /// Check if a new DP collides with any existing DP on disk.
    /// Only reads full DP data from SSD when the hash_key matches AND ktype differs.
    fn check_collision(
        &mut self, hash_key: u64, affine_x: &[u8; 32], dist_bytes: &[u8; 32], dp_ktype: u32,
    ) -> Option<Vec<u8>> {
        // Clone the refs to avoid borrow conflict with self
        let refs = match self.index.get(&hash_key) {
            Some(r) => r.clone(),
            None => return None,
        };

        for disk_ref in &refs {
            // Same type cannot produce a valid collision
            if disk_ref.ktype == dp_ktype {
                continue;
            }

            // Read full DP from disk only when we have a potential collision
            let existing = match self.read_dp_from_disk(disk_ref.offset) {
                Some(dp) => dp,
                None => continue,
            };

            // Verify full affine X match (not just hash)
            if existing.affine_x != *affine_x {
                continue;
            }

            tracing::info!("Collision candidate (offset={})", disk_ref.offset);

            // Cross-wild collision (wild1 ↔ wild2)
            if existing.ktype != 0 && dp_ktype != 0 && existing.ktype != dp_ktype {
                let (d1, d2) = if existing.ktype == 1 {
                    (existing.dist.as_slice(), dist_bytes.as_ref())
                } else {
                    (dist_bytes.as_ref(), existing.dist.as_slice())
                };
                if let Some(key) = compute_candidate_keys_cross_wild(d1, d2, &self.pubkey, &self.base_point) {
                    tracing::info!("Cross-wild collision! Key: 0x{}", hex::encode(&key));
                    return Some(key);
                }
                return None;
            }

            // Tame ↔ wild collision
            let (tame_dist, wild_dist) = if existing.ktype == 0 {
                (existing.dist.as_slice(), dist_bytes.as_ref())
            } else {
                (dist_bytes.as_ref(), existing.dist.as_slice())
            };

            let candidates = compute_candidate_keys(&self.start, tame_dist, wild_dist);
            for candidate in &candidates {
                if verify_key_with_base(candidate, &self.pubkey, &self.base_point) {
                    tracing::info!("Collision found! Key: 0x{}", hex::encode(candidate));
                    return Some(candidate.clone());
                }
            }
            return None;
        }
        None
    }

    /// Insert a new DP and check for collisions against ALL stored DPs.
    ///
    /// Flow:
    /// 1. Compute hash_key from affine X
    /// 2. Check index for potential collisions (RAM lookup)
    /// 3. If match found, read full DP from SSD and verify
    /// 4. Write new DP to SSD (append)
    /// 5. Add to index (RAM)
    ///
    /// Returns the private key if a valid collision is found.
    pub fn insert_and_check(&mut self, dp: GpuDistinguishedPoint) -> Option<Vec<u8>> {
        let dist_bytes = limbs_to_le_bytes(&dp.dist);
        let affine_x = limbs_to_be_bytes(&dp.x);

        let total = self.total_dps();
        if total < 20 {
            let ktype_str = match dp.ktype {
                0 => "tame",
                1 => "wild1",
                2 => "wild2",
                _ => {
                    tracing::warn!("DP[{}] unknown ktype={}", total, dp.ktype);
                    return None;
                }
            };
            tracing::debug!(
                "DP[{}] {}: x[0..2]=[{:08x},{:08x}] dist[0..2]=[{:08x},{:08x}] affine_x[0..4]={}",
                total, ktype_str, dp.x[0], dp.x[1], dp.dist[0], dp.dist[1],
                hex::encode(&affine_x[..4])
            );
        }

        let hash_key = u64::from_le_bytes([
            affine_x[0], affine_x[1], affine_x[2], affine_x[3],
            affine_x[4], affine_x[5], affine_x[6], affine_x[7],
        ]);

        // Check for collision against ALL DPs via index
        if let Some(key) = self.check_collision(hash_key, &affine_x, &dist_bytes, dp.ktype) {
            return Some(key);
        }

        // No collision — write to SSD and add to index
        let offset = self.write_dp_to_disk(&affine_x, &dist_bytes, dp.ktype);
        self.index
            .entry(hash_key)
            .or_insert_with(Vec::new)
            .push(DiskRef { offset, ktype: dp.ktype });
        self.total_dps += 1;
        self.increment_type_counter(dp.ktype);

        None
    }

    pub fn total_dps(&self) -> usize {
        self.total_dps
    }

    pub fn count_by_type(&self) -> (usize, usize, usize) {
        (self.tame_count, self.wild1_count, self.wild2_count)
    }

    fn increment_type_counter(&mut self, ktype: u32) {
        match ktype {
            0 => self.tame_count += 1,
            1 => self.wild1_count += 1,
            2 => self.wild2_count += 1,
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════
// Collision resolution — unchanged from original
// ═══════════════════════════════════════════════════

pub(crate) fn compute_candidate_scalars(base: Scalar, tame_d: Scalar, wild_d: Scalar) -> [Scalar; 8] {
    let neg_base = -base;
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
    d1_bytes: &[u8], d2_bytes: &[u8], pubkey: &ProjectivePoint, base_point: &ProjectivePoint,
) -> Option<Vec<u8>> {
    let d1_pair = distance_scalar_pair(&pad_to_32(d1_bytes));
    let d2_pair = distance_scalar_pair(&pad_to_32(d2_bytes));
    let half = scalar_half();
    for &d1 in &d1_pair {
        for &d2 in &d2_pair {
            let k_diff = (d2 - d1) * half;
            for candidate in &[k_diff, -k_diff] {
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
fn distance_scalar_pair(dist_le_bytes: &[u8; 32]) -> [Scalar; 2] {
    let uint = K256U256::from_le_slice(dist_le_bytes);
    let direct = <Scalar as Reduce<K256U256>>::reduce(uint);
    let neg_bytes = negate_u256_bytes(dist_le_bytes);
    let neg_uint = K256U256::from_le_slice(&neg_bytes);
    let negated = -<Scalar as Reduce<K256U256>>::reduce(neg_uint);
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

// ═══════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::DPTable;
    use crate::crypto::{parse_pubkey, verify_key};
    use crate::gpu::GpuDistinguishedPoint;
    use k256::elliptic_curve::ops::{MulByGenerator, Reduce};
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    use k256::{ProjectivePoint, Scalar, U256 as K256U256};

    fn make_dp(x: u32, dist: u32, ktype: u32) -> GpuDistinguishedPoint {
        let mut xw = [0u32; 8]; xw[7] = x;
        let mut dw = [0u32; 8]; dw[0] = dist;
        GpuDistinguishedPoint { x: xw, dist: dw, ktype, kangaroo_id: 0, _padding: [0u32; 6] }
    }

    fn scalar_from_u64(val: u64) -> Scalar {
        let mut le = [0u8; 32];
        le[..8].copy_from_slice(&val.to_le_bytes());
        <Scalar as Reduce<K256U256>>::reduce(K256U256::from_le_slice(&le))
    }

    fn scalar_to_le_bytes(s: &Scalar) -> [u8; 32] {
        let be = s.to_bytes();
        let mut le = [0u8; 32];
        for i in 0..32 { le[i] = be[31 - i]; }
        le
    }

    fn scalar_to_dist_u32(s: &Scalar) -> [u32; 8] {
        let le = scalar_to_le_bytes(s);
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = u32::from_le_bytes([le[i * 4], le[i * 4 + 1], le[i * 4 + 2], le[i * 4 + 3]]);
        }
        r
    }

    fn point_to_x_u32(p: &ProjectivePoint) -> [u32; 8] {
        let a = p.to_affine();
        let e = a.to_encoded_point(false);
        let x = e.x().unwrap();
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[7 - i] = u32::from_be_bytes([x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3]]);
        }
        r
    }

    fn make_real_dp(cp: &ProjectivePoint, dist: &Scalar, ktype: u32) -> GpuDistinguishedPoint {
        GpuDistinguishedPoint {
            x: point_to_x_u32(cp), dist: scalar_to_dist_u32(dist),
            ktype, kangaroo_id: 0, _padding: [0u32; 6],
        }
    }

    fn cleanup_test_file(range_bits: u32) {
        let _ = std::fs::remove_file(format!(
            "{}/Desktop/kangaroo_dps/dps_range{}.bin",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string()),
            range_bits
        ));
    }

    #[test]
    fn insert_stores_on_disk() {
        cleanup_test_file(40);
        let mut table = DPTable::new([0u8; 32], ProjectivePoint::GENERATOR, ProjectivePoint::GENERATOR, 40);
        for i in 0..1000u32 {
            assert!(table.insert_and_check(make_dp(i, i, 0)).is_none());
        }
        assert_eq!(table.total_dps(), 1000);
        cleanup_test_file(40);
    }

    #[test]
    fn test_collision_case1() {
        cleanup_test_file(41);
        let k = scalar_from_u64(66);
        let start_s = scalar_from_u64(50);
        let td = scalar_from_u64(30);
        let wd = scalar_from_u64(14);
        let pk = ProjectivePoint::mul_by_generator(&k);
        let cp = ProjectivePoint::mul_by_generator(&(start_s + td));
        let mut table = DPTable::new(scalar_to_le_bytes(&start_s), pk, ProjectivePoint::GENERATOR, 41);
        assert!(table.insert_and_check(make_real_dp(&cp, &td, 0)).is_none());
        let r = table.insert_and_check(make_real_dp(&cp, &wd, 1));
        assert!(r.is_some());
        assert!(verify_key(&r.unwrap(), &pk));
        cleanup_test_file(41);
    }

    #[test]
    fn test_scalar_half() {
        assert_eq!(scalar_from_u64(2) * super::scalar_half(), Scalar::ONE);
    }

    #[test]
    fn test_cross_wild() {
        cleanup_test_file(42);
        let pk = parse_pubkey("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c").unwrap();
        let k = scalar_from_u64(0x0d2c55);
        let d1 = scalar_from_u64(123_456);
        let d2 = d1 + (k + k);
        let cp = ProjectivePoint::mul_by_generator(&(k + d1));
        let mut table = DPTable::new([0u8; 32], pk, ProjectivePoint::GENERATOR, 42);
        assert!(table.insert_and_check(make_real_dp(&cp, &d1, 1)).is_none());
        let r = table.insert_and_check(make_real_dp(&cp, &d2, 2));
        assert!(r.is_some());
        assert!(verify_key(&r.clone().unwrap(), &table.pubkey));
        cleanup_test_file(42);
    }

    #[test]
    fn test_persistence_across_sessions() {
        cleanup_test_file(99);

        // Session 1: insert a tame DP
        {
            let k = scalar_from_u64(66);
            let start_s = scalar_from_u64(50);
            let td = scalar_from_u64(30);
            let pk = ProjectivePoint::mul_by_generator(&k);
            let cp = ProjectivePoint::mul_by_generator(&(start_s + td));
            let mut table = DPTable::new(scalar_to_le_bytes(&start_s), pk, ProjectivePoint::GENERATOR, 99);
            assert!(table.insert_and_check(make_real_dp(&cp, &td, 0)).is_none());
            assert_eq!(table.total_dps(), 1);
        }

        // Session 2: insert a wild DP at the same point → should find collision from disk
        {
            let k = scalar_from_u64(66);
            let start_s = scalar_from_u64(50);
            let td = scalar_from_u64(30);
            let wd = scalar_from_u64(14);
            let pk = ProjectivePoint::mul_by_generator(&k);
            let cp = ProjectivePoint::mul_by_generator(&(start_s + td));
            let mut table = DPTable::new(scalar_to_le_bytes(&start_s), pk, ProjectivePoint::GENERATOR, 99);
            assert_eq!(table.total_dps(), 1);
            let r = table.insert_and_check(make_real_dp(&cp, &wd, 1));
            assert!(r.is_some(), "Should find collision with DP from previous session");
            assert!(verify_key(&r.unwrap(), &pk));
        }

        cleanup_test_file(99);
    }

    #[test]
    fn test_pubkey_mismatch_creates_new_file() {
        cleanup_test_file(98);

        // Session 1: puzzle A
        {
            let pk_a = ProjectivePoint::mul_by_generator(&scalar_from_u64(100));
            let mut table = DPTable::new([0u8; 32], pk_a, ProjectivePoint::GENERATOR, 98);
            assert!(table.insert_and_check(make_dp(1, 1, 0)).is_none());
            assert_eq!(table.total_dps(), 1);
        }

        // Session 2: puzzle B (different pubkey, same range_bits)
        {
            let pk_b = ProjectivePoint::mul_by_generator(&scalar_from_u64(200));
            let table = DPTable::new([0u8; 32], pk_b, ProjectivePoint::GENERATOR, 98);
            // Should have started fresh — old DPs rejected due to pubkey mismatch
            assert_eq!(table.total_dps(), 0);
        }

        cleanup_test_file(98);
        // Also clean up the backup file
        let dp_dir = format!(
            "{}/Desktop/kangaroo_dps",
            std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
        );
        if let Ok(entries) = std::fs::read_dir(&dp_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("dps_range98.bin.backup") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}
