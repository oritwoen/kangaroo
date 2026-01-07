//! Benchmark: Measure solving performance using Criterion
//!
//! Run with: cargo bench
//!
//! This suite covers:
//! 1. Complexity scaling (increasing puzzle difficulty)
//! 2. Kangaroo count scaling (tuning optimization)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kangaroo::{parse_hex_u256, parse_pubkey, verify_key, GpuContext, KangarooSolver};
use std::time::Duration;

/// Test configuration for a specific puzzle
#[derive(Clone, Copy)]
struct PuzzleConfig {
    id: u32,
    pubkey: &'static str,
    start: &'static str,
    range_bits: u32,
    expected_key: &'static str,
}

const PUZZLES: &[PuzzleConfig] = &[
    PuzzleConfig {
        id: 20,
        pubkey: "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
        start: "0x80000",
        range_bits: 20,
        expected_key: "d2c55",
    },
    PuzzleConfig {
        id: 21,
        pubkey: "031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e",
        start: "0x100000",
        range_bits: 21,
        expected_key: "1ba534",
    },
    // We stop at 22 for regular benchmarking to keep runtime reasonable (~seconds)
    PuzzleConfig {
        id: 22,
        pubkey: "023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43",
        start: "0x200000",
        range_bits: 22,
        expected_key: "2de40f",
    },
];

/// Helper function to run a single solve attempt
fn run_solve(puzzle: &PuzzleConfig, num_kangaroos: u32) {
    let pubkey = parse_pubkey(puzzle.pubkey).unwrap();
    let start = parse_hex_u256(puzzle.start).unwrap();
    // Auto-calculate DP bits based on range to match main logic
    let dp_bits = (puzzle.range_bits / 2).saturating_sub(2).clamp(8, 20);

    // Initialize context (blocking for benchmark)
    let ctx = pollster::block_on(GpuContext::new(0)).unwrap();

    let mut solver = KangarooSolver::new(
        ctx,
        pubkey.clone(),
        start,
        puzzle.range_bits,
        dp_bits,
        num_kangaroos,
    )
    .unwrap();

    // Loop until solved (no strict timeout here, Criterion handles the measurement time)
    loop {
        match solver.step() {
            Ok(Some(key)) => {
                if verify_key(&key, &pubkey) {
                    break;
                }
            }
            Ok(None) => continue,
            Err(e) => panic!("Solver error: {}", e),
        }
    }
}

/// Benchmark 1: Performance across different puzzle difficulties
fn bench_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("puzzle_complexity");

    // ECDLP is probabilistic and takes time.
    // We use a small sample size but longer measurement time.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Use a fixed kangaroo count for complexity comparison to be fair
    let fixed_kangaroos = 16384;

    for puzzle in PUZZLES {
        // Estimate workload as sqrt(range) for throughput reporting
        let range_size = 1u64 << puzzle.range_bits;
        let expected_ops = (range_size as f64).sqrt() as u64;

        group.throughput(Throughput::Elements(expected_ops));

        group.bench_with_input(
            BenchmarkId::new("solve", format!("{}bit", puzzle.range_bits)),
            puzzle,
            |b, &p| {
                b.iter(|| run_solve(&p, fixed_kangaroos));
            },
        );
    }
    group.finish();
}

/// Benchmark 2: Kangaroo count scaling (tuning)
fn bench_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("kangaroo_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Use Puzzle 20 as the base for tuning tests (fastest to iterate)
    let puzzle = PUZZLES[0];
    let kangaroo_counts = [4096, 8192, 16384, 32768];

    for &count in &kangaroo_counts {
        group.bench_with_input(BenchmarkId::new("kangaroos", count), &count, |b, &k| {
            b.iter(|| run_solve(&puzzle, k));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_complexity, bench_tuning);
criterion_main!(benches);
