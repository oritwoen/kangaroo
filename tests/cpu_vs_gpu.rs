//! CPU vs GPU performance comparison benchmark
//!
//! Run with: cargo test --release cpu_vs_gpu -- --nocapture --ignored

use k256::elliptic_curve::ops::MulByGenerator;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::PrimeField;
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
use kangaroo::{parse_hex_u256, parse_pubkey, verify_key, GpuContext, KangarooSolver};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Pure CPU Kangaroo solver using k256
struct CpuKangarooSolver {
    pubkey: ProjectivePoint,
    start: u128,
    range_bits: u32,
    dp_mask: u128,
    tame_table: HashMap<u128, u128>, // x_low -> distance
    wild_table: HashMap<u128, u128>,
    ops: u64,
}

impl CpuKangarooSolver {
    fn new(pubkey: ProjectivePoint, start: u128, range_bits: u32, dp_bits: u32) -> Self {
        let dp_mask = (1u128 << dp_bits) - 1;
        Self {
            pubkey,
            start,
            range_bits,
            dp_mask,
            tame_table: HashMap::new(),
            wild_table: HashMap::new(),
            ops: 0,
        }
    }

    fn solve(&mut self, timeout: Duration) -> Option<Vec<u8>> {
        let start_time = Instant::now();
        let range_middle = 1u128 << (self.range_bits - 1);
        let mid = self.start + range_middle;

        // Initialize tame kangaroo at mid
        let tame_scalar = scalar_from_u128(mid);
        let mut tame_pos = ProjectivePoint::mul_by_generator(&tame_scalar);
        let mut tame_dist: u128 = 0;

        // Initialize wild kangaroo at pubkey
        let mut wild_pos = self.pubkey;
        let mut wild_dist: u128 = 0;

        // Jump table (simple: use powers of 2)
        let mean_exp = (self.range_bits / 2).saturating_sub(2).max(8);
        let jump_scalars: Vec<Scalar> = (0..8)
            .map(|i| {
                let exp = mean_exp - 4 + i;
                let exp = exp.clamp(6, 20);
                scalar_from_u128(1u128 << exp)
            })
            .collect();
        let jump_points: Vec<ProjectivePoint> = jump_scalars
            .iter()
            .map(|s| ProjectivePoint::mul_by_generator(s))
            .collect();
        let jump_distances: Vec<u128> = (0..8)
            .map(|i| {
                let exp = mean_exp - 4 + i;
                let exp = exp.clamp(6, 20);
                1u128 << exp
            })
            .collect();

        loop {
            if start_time.elapsed() > timeout {
                return None;
            }

            // Tame step
            let tame_x = get_x_low(&tame_pos);
            let jump_idx = (tame_x & 7) as usize;
            tame_pos = tame_pos + jump_points[jump_idx];
            tame_dist = tame_dist.wrapping_add(jump_distances[jump_idx]);
            self.ops += 1;

            // Check DP
            if (tame_x & self.dp_mask) == 0 {
                if let Some(&wild_d) = self.wild_table.get(&tame_x) {
                    // Collision! k = mid + tame_dist - wild_dist
                    let key = mid.wrapping_add(tame_dist).wrapping_sub(wild_d);
                    let key_bytes = key_to_bytes(key);
                    if verify_key_bytes(&key_bytes, &self.pubkey) {
                        return Some(key_bytes);
                    }
                }
                self.tame_table.insert(tame_x, tame_dist);
            }

            // Wild step
            let wild_x = get_x_low(&wild_pos);
            let jump_idx = (wild_x & 7) as usize;
            wild_pos = wild_pos + jump_points[jump_idx];
            wild_dist = wild_dist.wrapping_add(jump_distances[jump_idx]);
            self.ops += 1;

            // Check DP
            if (wild_x & self.dp_mask) == 0 {
                if let Some(&tame_d) = self.tame_table.get(&wild_x) {
                    // Collision! k = mid + tame_dist - wild_dist
                    let key = mid.wrapping_add(tame_d).wrapping_sub(wild_dist);
                    let key_bytes = key_to_bytes(key);
                    if verify_key_bytes(&key_bytes, &self.pubkey) {
                        return Some(key_bytes);
                    }
                }
                self.wild_table.insert(wild_x, wild_dist);
            }
        }
    }

    fn total_ops(&self) -> u64 {
        self.ops
    }
}

fn scalar_from_u128(val: u128) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[16..].copy_from_slice(&val.to_be_bytes());
    let uint = K256U256::from_be_slice(&bytes);
    Scalar::reduce(uint)
}

fn get_x_low(point: &ProjectivePoint) -> u128 {
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let affine = point.to_affine();
    let encoded = affine.to_encoded_point(false);
    let x_bytes = encoded.x().unwrap();
    // Get low 128 bits
    let mut low = [0u8; 16];
    low.copy_from_slice(&x_bytes[16..32]);
    u128::from_be_bytes(low)
}

fn key_to_bytes(key: u128) -> Vec<u8> {
    let bytes = key.to_be_bytes();
    // Trim leading zeros
    let mut start = 0;
    while start < bytes.len() - 1 && bytes[start] == 0 {
        start += 1;
    }
    bytes[start..].to_vec()
}

fn verify_key_bytes(key: &[u8], pubkey: &ProjectivePoint) -> bool {
    if key.is_empty() || key.len() > 32 {
        return false;
    }
    let mut key_be = [0u8; 32];
    let offset = 32 - key.len();
    key_be[offset..].copy_from_slice(key);

    if let Some(scalar) = Scalar::from_repr_vartime(key_be.into()) {
        let computed = ProjectivePoint::mul_by_generator(&scalar);
        computed == *pubkey
    } else {
        false
    }
}

#[test]
#[ignore]
fn cpu_vs_gpu_benchmark() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              CPU vs GPU PERFORMANCE COMPARISON                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let puzzles = [
        (
            20,
            "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
            "0x80000",
            "d2c55",
        ),
        (
            21,
            "031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e",
            "0x100000",
            "1ba534",
        ),
        (
            22,
            "023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43",
            "0x200000",
            "2de40f",
        ),
    ];

    println!("\n--- CPU-only Solver ---\n");

    let mut cpu_results = Vec::new();
    for (puzzle_num, pubkey_hex, start_hex, _expected) in &puzzles {
        let pubkey = parse_pubkey(pubkey_hex).expect("valid pubkey");
        let start_bytes = parse_hex_u256(start_hex).expect("valid start");
        let start = u128::from_le_bytes(start_bytes[0..16].try_into().unwrap());
        let range_bits = *puzzle_num as u32;
        let dp_bits = (range_bits / 2).saturating_sub(2).clamp(8, 20);

        let mut solver = CpuKangarooSolver::new(pubkey, start, range_bits, dp_bits);
        let start_time = Instant::now();
        let found = solver.solve(Duration::from_secs(120));
        let elapsed = start_time.elapsed();
        let ops = solver.total_ops();
        let ops_per_sec = ops as f64 / elapsed.as_secs_f64();

        let status = if found.is_some() { "FOUND" } else { "TIMEOUT" };
        println!(
            "  Puzzle {}: {:>6.2}s | {:>10} ops | {:>10.0} ops/s | {}",
            puzzle_num,
            elapsed.as_secs_f64(),
            ops,
            ops_per_sec,
            status
        );
        cpu_results.push((elapsed, ops, ops_per_sec, found.is_some()));
    }

    println!("\n--- GPU Hybrid Solver ---\n");

    let mut gpu_results = Vec::new();
    for (puzzle_num, pubkey_hex, start_hex, _expected) in &puzzles {
        let pubkey = parse_pubkey(pubkey_hex).expect("valid pubkey");
        let start = parse_hex_u256(start_hex).expect("valid start");
        let range_bits = *puzzle_num as u32;
        let dp_bits = (range_bits / 2).saturating_sub(2).clamp(8, 20);

        let ctx = pollster::block_on(GpuContext::new(0)).expect("GPU context");
        let mut solver = KangarooSolver::new(ctx, pubkey.clone(), start, range_bits, dp_bits, 4096)
            .expect("solver");

        let start_time = Instant::now();
        let timeout = Duration::from_secs(120);
        let mut found = false;

        loop {
            if start_time.elapsed() > timeout {
                break;
            }
            match solver.step() {
                Ok(Some(key)) => {
                    if verify_key(&key, &pubkey) {
                        found = true;
                    }
                    break;
                }
                Ok(None) => continue,
                Err(_) => break,
            }
        }

        let elapsed = start_time.elapsed();
        let ops = solver.total_operations();
        let ops_per_sec = ops as f64 / elapsed.as_secs_f64();

        let status = if found { "FOUND" } else { "TIMEOUT" };
        println!(
            "  Puzzle {}: {:>6.2}s | {:>10} ops | {:>10.0} ops/s | {}",
            puzzle_num,
            elapsed.as_secs_f64(),
            ops,
            ops_per_sec,
            status
        );
        gpu_results.push((elapsed, ops, ops_per_sec, found));
    }

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      COMPARISON SUMMARY                          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Puzzle │  CPU Time  │  GPU Time  │ Speedup │ CPU ops/s │GPU ops/s║");
    println!("╠════════╪════════════╪════════════╪═════════╪═══════════╪═════════╣");

    for (i, (puzzle_num, _, _, _)) in puzzles.iter().enumerate() {
        let (cpu_time, _, cpu_ops, _) = cpu_results[i];
        let (gpu_time, _, gpu_ops, _) = gpu_results[i];
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "║   {:>2}   │ {:>8.2}s  │ {:>8.2}s  │ {:>5.1}×  │ {:>9.0} │ {:>7.0} ║",
            puzzle_num,
            cpu_time.as_secs_f64(),
            gpu_time.as_secs_f64(),
            speedup,
            cpu_ops,
            gpu_ops
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Calculate totals
    let cpu_total_time: f64 = cpu_results.iter().map(|(t, _, _, _)| t.as_secs_f64()).sum();
    let gpu_total_time: f64 = gpu_results.iter().map(|(t, _, _, _)| t.as_secs_f64()).sum();
    let cpu_total_ops: u64 = cpu_results.iter().map(|(_, o, _, _)| *o).sum();
    let gpu_total_ops: u64 = gpu_results.iter().map(|(_, o, _, _)| *o).sum();

    let overall_speedup = cpu_total_time / gpu_total_time;
    let cpu_avg_ops = cpu_total_ops as f64 / cpu_total_time;
    let gpu_avg_ops = gpu_total_ops as f64 / gpu_total_time;
    let throughput_ratio = gpu_avg_ops / cpu_avg_ops;

    println!("\nTotals:");
    println!(
        "  CPU: {:.2}s total, {:.0} avg ops/s",
        cpu_total_time, cpu_avg_ops
    );
    println!(
        "  GPU: {:.2}s total, {:.0} avg ops/s",
        gpu_total_time, gpu_avg_ops
    );
    println!("\n  Time speedup: {:.1}× faster", overall_speedup);
    println!("  Throughput:   {:.1}× higher", throughput_ratio);
}
