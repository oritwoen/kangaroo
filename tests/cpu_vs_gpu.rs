//! CPU vs GPU performance comparison benchmark
//!
//! Run with: cargo test --release cpu_vs_gpu -- --nocapture --ignored

use kangaroo::{
    parse_hex_u256, parse_pubkey, verify_key, CpuKangarooSolver, GpuBackend, GpuContext,
    KangarooSolver,
};
use std::time::{Duration, Instant};

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
        let range_bits = *puzzle_num as u32;
        let dp_bits = (range_bits / 2).saturating_sub(2).clamp(8, 20);

        let mut solver = CpuKangarooSolver::new(pubkey, start_bytes, range_bits, dp_bits);
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

        let ctx = pollster::block_on(GpuContext::new(0, GpuBackend::Auto)).expect("GPU context");
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
