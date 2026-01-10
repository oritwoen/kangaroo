//! Kangaroo: Pollard's Kangaroo ECDLP solver using Vulkan/Metal/DX12 compute
//!
//! GPU-accelerated implementation for solving the Elliptic Curve Discrete
//! Logarithm Problem on secp256k1 within a known range.
//!
//! Supports AMD, NVIDIA, Intel GPUs via wgpu (Vulkan/Metal/DX12).

mod cli;
mod convert;
mod cpu;
mod crypto;
mod gpu;
mod gpu_crypto;
mod math;
mod provider;
mod solver;

pub use crypto::{full_verify, parse_hex_u256, parse_pubkey, verify_key, Point};
pub use gpu_crypto::GpuContext;
pub use solver::KangarooSolver;

use anyhow::anyhow;
use clap::Parser;
use indicatif::ProgressBar;
#[cfg(feature = "boha")]
use num_bigint::BigUint;
use serde::Serialize;
use std::time::Instant;
use tracing::{error, info};

/// Pollard's Kangaroo ECDLP solver for secp256k1
///
/// Finds private key k such that P = k*G, given that k is in range [start, start + 2^range_bits]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Public key to solve (compressed hex, 33 bytes)
    #[arg(short, long)]
    pubkey: Option<String>,

    /// Start of search range (hex, without 0x prefix)
    #[arg(short, long)]
    start: Option<String>,

    /// Bit range to search (key is in [start, start + 2^range])
    #[arg(short, long)]
    range: Option<u32>,

    /// Data provider target (e.g., boha:b1000/135)
    #[arg(short, long)]
    target: Option<String>,

    /// List available puzzles from providers
    #[arg(long)]
    list_providers: bool,

    /// Distinguished point bits (auto-calculated if not set)
    #[arg(short, long)]
    dp_bits: Option<u32>,

    /// Number of kangaroos (default: auto based on GPU)
    #[arg(short, long)]
    kangaroos: Option<u32>,

    /// GPU device index
    #[arg(long, default_value = "0")]
    gpu: u32,

    /// Output file for result (hex private key)
    #[arg(short, long)]
    output: Option<String>,

    /// Quiet mode - minimal output, just print found key
    #[arg(short, long)]
    quiet: bool,

    /// Maximum operations before giving up (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_ops: u64,

    /// Use CPU solver instead of GPU (slow, for benchmarking)
    #[arg(long)]
    cpu: bool,

    /// Output benchmark results in JSON format to stdout
    #[arg(long)]
    json: bool,

    /// Run benchmark suite and print results
    #[arg(long)]
    benchmark: bool,
}

#[derive(Serialize)]
struct BenchmarkResult {
    metric: String,
    value: f64,
    unit: String,
    metadata: Metadata,
}

#[derive(Serialize)]
struct Metadata {
    device: String,
    range_bits: u32,
    algorithm: String,
    total_ops: u64,
    time_seconds: f64,
}

pub fn run_from_args<I, S>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    let args = Args::parse_from(args);
    run(args)
}

struct ResolvedParams {
    pubkey_str: String,
    start_str: String,
    range_bits: u32,
}

fn resolve_params(args: &Args) -> anyhow::Result<ResolvedParams> {
    let provider_result = if let Some(ref target) = args.target {
        provider::resolve(target)?
    } else {
        None
    };

    let (pubkey_str, start_str, range_bits) = match provider_result {
        Some(ref pr) => {
            let pubkey_str = match (&args.pubkey, &pr.pubkey) {
                (Some(p), _) => p.clone(),
                (None, Some(p)) => p.clone(),
                (None, None) => {
                    return Err(anyhow!(
                        "Puzzle '{}' has no public key. Cannot solve without pubkey.",
                        pr.id
                    ))
                }
            };

            let start_str = args
                .start
                .clone()
                .or_else(|| pr.start.clone())
                .unwrap_or_else(|| "0".to_string());

            let range_bits = match (args.range, pr.range_bits) {
                (Some(user_range), Some(provider_range)) => {
                    validate_range_override(user_range, provider_range, &pr.id)?;
                    user_range
                }
                (Some(user_range), None) => user_range,
                (None, Some(provider_range)) => provider_range,
                (None, None) => 32,
            };

            validate_search_bounds(&start_str, range_bits, pr)?;

            (pubkey_str, start_str, range_bits)
        }
        None => {
            let pubkey_str = args
                .pubkey
                .clone()
                .ok_or_else(|| anyhow!("--pubkey is required when not using --target"))?;
            let start_str = args.start.clone().unwrap_or_else(|| "0".to_string());
            let range_bits = args.range.unwrap_or(32);
            (pubkey_str, start_str, range_bits)
        }
    };

    Ok(ResolvedParams {
        pubkey_str,
        start_str,
        range_bits,
    })
}

fn validate_range_override(
    user_range: u32,
    provider_range: u32,
    puzzle_id: &str,
) -> anyhow::Result<()> {
    if user_range > provider_range {
        return Err(anyhow!(
            "Range {} bits exceeds puzzle '{}' maximum of {} bits",
            user_range,
            puzzle_id,
            provider_range
        ));
    }
    Ok(())
}

#[cfg(feature = "boha")]
fn validate_search_bounds(
    start: &str,
    range_bits: u32,
    provider: &provider::ProviderResult,
) -> anyhow::Result<()> {
    let (Some(ref provider_start), Some(ref provider_end)) = (&provider.start, &provider.end)
    else {
        return Ok(());
    };

    let start_val = BigUint::parse_bytes(start.as_bytes(), 16)
        .ok_or_else(|| anyhow!("Invalid hex start value: {}", start))?;
    let provider_start_val = BigUint::parse_bytes(provider_start.as_bytes(), 16)
        .ok_or_else(|| anyhow!("Invalid provider start hex"))?;
    let provider_end_val = BigUint::parse_bytes(provider_end.as_bytes(), 16)
        .ok_or_else(|| anyhow!("Invalid provider end hex"))?;

    if start_val < provider_start_val {
        return Err(anyhow!(
            "Start 0x{} is below puzzle '{}' minimum 0x{}",
            start,
            provider.id,
            provider_start
        ));
    }

    let search_end = &start_val + (BigUint::from(1u64) << range_bits);
    if search_end > provider_end_val {
        return Err(anyhow!(
            "Search range [0x{}..0x{:x}] exceeds puzzle '{}' maximum 0x{}",
            start,
            search_end,
            provider.id,
            provider_end
        ));
    }

    Ok(())
}

#[cfg(not(feature = "boha"))]
fn validate_search_bounds(
    _start: &str,
    _range_bits: u32,
    _provider: &provider::ProviderResult,
) -> anyhow::Result<()> {
    Ok(())
}

fn print_providers_list() {
    let providers = provider::supported_providers();
    if providers.is_empty() {
        println!("No providers available. Rebuild with --features boha");
        return;
    }

    println!("Available puzzles:");
    println!(
        "{:<20} {:<45} {:>6} {:>8}",
        "ID", "Address", "Bits", "Pubkey"
    );
    println!("{}", "-".repeat(85));

    for (provider_name, id, address, bits, has_pubkey) in provider::list_available() {
        let bits_str = bits
            .map(|b| b.to_string())
            .unwrap_or_else(|| "?".to_string());
        let pubkey_str = if has_pubkey { "yes" } else { "no" };
        println!(
            "{:<20} {:<45} {:>6} {:>8}",
            format!("{}:{}", provider_name, id),
            address,
            bits_str,
            pubkey_str
        );
    }
}

struct BenchmarkCase {
    name: &'static str,
    pubkey: &'static str,
    start: &'static str,
    range_bits: u32,
}

const BENCHMARK_CASES: &[BenchmarkCase] = &[
    BenchmarkCase {
        name: "32-bit",
        pubkey: "03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7",
        start: "80000000",
        range_bits: 32,
    },
    BenchmarkCase {
        name: "40-bit",
        pubkey: "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
        start: "8000000000",
        range_bits: 40,
    },
    BenchmarkCase {
        name: "48-bit",
        pubkey: "026864513503daca97ffae5d13d784192f932f304677b9a67a48a41af53f88ad19",
        start: "800000000000",
        range_bits: 48,
    },
];

fn run_benchmark(gpu_index: u32) -> anyhow::Result<()> {
    println!("Kangaroo Benchmark Suite");
    println!("========================\n");

    let gpu_context = pollster::block_on(gpu_crypto::GpuContext::new(gpu_index))?;
    println!("GPU: {}", gpu_context.device_name());
    println!("Compute units: {}\n", gpu_context.compute_units());

    println!(
        "{:<10} {:>12} {:>12} {:>14}",
        "Range", "Time", "Ops", "Rate"
    );
    println!("{}", "-".repeat(52));

    let num_k = gpu_context.optimal_kangaroos();

    for case in BENCHMARK_CASES {
        let pubkey = crypto::parse_pubkey(case.pubkey)?;
        let start = crypto::parse_hex_u256(case.start)?;

        let dp_bits = (case.range_bits / 2)
            .saturating_sub((num_k as f64).log2() as u32 / 2)
            .clamp(8, 40);

        let mut solver = solver::KangarooSolver::new(
            gpu_context.clone(),
            pubkey,
            start,
            case.range_bits,
            dp_bits,
            num_k,
        )?;

        let start_time = Instant::now();
        loop {
            if solver.step()?.is_some() {
                break;
            }
        }
        let duration = start_time.elapsed();
        let total_ops = solver.total_operations();
        let rate = total_ops as f64 / duration.as_secs_f64();

        println!(
            "{:<10} {:>10.2}s {:>12} {:>12.2}M/s",
            case.name,
            duration.as_secs_f64(),
            total_ops,
            rate / 1_000_000.0
        );
    }

    println!("\n[Copy above results to BENCHMARK.md]");
    Ok(())
}

pub fn run(args: Args) -> anyhow::Result<()> {
    cli::init_tracing(false, args.quiet || args.json || args.benchmark);

    if args.list_providers {
        print_providers_list();
        return Ok(());
    }

    if args.benchmark {
        return run_benchmark(args.gpu);
    }

    let params = resolve_params(&args)?;

    if !args.quiet && !args.json {
        info!("Kangaroo ECDLP Solver");
        info!("=====================");
        if let Some(ref target) = args.target {
            info!("Target: {}", target);
        }
        info!("Pubkey: {}", params.pubkey_str);
        info!(
            "Search range: {} bits from 0x{}",
            params.range_bits, params.start_str
        );
    }

    let pubkey = crypto::parse_pubkey(&params.pubkey_str)?;
    let start = crypto::parse_hex_u256(&params.start_str)?;
    let range_bits = params.range_bits;

    if args.cpu {
        if !args.quiet && !args.json {
            info!("Mode: CPU (Software Solver)");
        }

        let dp_bits = args
            .dp_bits
            .unwrap_or_else(|| (range_bits / 2).saturating_sub(2).clamp(8, 20));

        if !args.quiet && !args.json {
            info!("DP bits: {}", dp_bits);
        }

        let mut start_be = start;
        start_be.reverse();

        let mut solver = cpu::CpuKangarooSolver::new_full(pubkey, start_be, range_bits, dp_bits);

        let expected_ops = (1u128 << (range_bits / 2)) as u64;
        let pb = if args.quiet || args.json {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(expected_ops);
            pb.set_style(cli::default_progress_style_with_msg());
            pb
        };

        let start_time = Instant::now();
        let result = solver.solve(std::time::Duration::from_secs(3600));
        let duration = start_time.elapsed();

        if let Some(private_key) = result {
            pb.finish_with_message("FOUND!");
            let key_hex = hex::encode(&private_key);
            let key_hex_trimmed = key_hex.trim_start_matches('0');
            let key_hex_display = if key_hex_trimmed.is_empty() {
                "0"
            } else {
                key_hex_trimmed
            };

            if args.json {
                let total_ops = solver.total_ops();
                let time_seconds = duration.as_secs_f64();
                let rate = total_ops as f64 / time_seconds;

                let result = BenchmarkResult {
                    metric: "hash_rate".to_string(),
                    value: rate,
                    unit: "ops/s".to_string(),
                    metadata: Metadata {
                        device: "cpu".to_string(),
                        range_bits,
                        algorithm: "pollard_kangaroo".to_string(),
                        total_ops,
                        time_seconds,
                    },
                };
                println!("{}", serde_json::to_string(&result)?);
            } else if args.quiet {
                println!("{}", key_hex_display);
            } else {
                info!("Private key found: 0x{}", key_hex_display);
                info!("Verification: SUCCESS");
                info!("Total operations: {}", solver.total_ops());
                info!("Time elapsed: {:.2}s", duration.as_secs_f64());
            }

            if let Some(ref output) = args.output {
                std::fs::write(output, &key_hex)?;
            }

            return Ok(());
        } else {
            pb.finish_with_message("TIMEOUT");
            return Err(anyhow!("Key not found within timeout"));
        }
    }

    let gpu_context = pollster::block_on(gpu_crypto::GpuContext::new(args.gpu))?;
    let device_name = gpu_context.device_name().to_string();
    if !args.quiet && !args.json {
        info!("GPU: {}", device_name);
        info!("Compute units: {}", gpu_context.compute_units());
    }

    let num_k = args.kangaroos.unwrap_or(gpu_context.optimal_kangaroos());
    let dp_bits = args.dp_bits.unwrap_or_else(|| {
        let auto_dp = (range_bits / 2).saturating_sub((num_k as f64).log2() as u32 / 2);
        auto_dp.clamp(8, 40)
    });

    if !args.quiet && !args.json {
        info!("DP bits: {}", dp_bits);
        info!("Kangaroos: {}", num_k);
    }

    let mut solver =
        solver::KangarooSolver::new(gpu_context, pubkey, start, range_bits, dp_bits, num_k)?;

    let expected_ops = (1u128 << (range_bits / 2)) as u64;
    let pb = if args.quiet || args.json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(expected_ops);
        pb.set_style(cli::default_progress_style());
        pb
    };

    if !args.quiet && !args.json {
        info!("Starting search...");
    }

    let max_ops = if args.max_ops == 0 {
        u64::MAX
    } else {
        args.max_ops
    };

    let start_time = Instant::now();

    loop {
        let result = solver.step()?;
        let total_ops = solver.total_operations();
        pb.set_position(total_ops);

        if let Some(private_key) = result {
            let duration = start_time.elapsed();
            pb.finish_with_message("FOUND!");
            let key_hex = hex::encode(&private_key);
            let key_hex_trimmed = key_hex.trim_start_matches('0');
            let key_hex_display = if key_hex_trimmed.is_empty() {
                "0"
            } else {
                key_hex_trimmed
            };

            if !crypto::verify_key(&private_key, &pubkey) {
                error!("Verification FAILED - this is a bug!");
                continue;
            }

            if args.json {
                let time_seconds = duration.as_secs_f64();
                let rate = total_ops as f64 / time_seconds;

                let result = BenchmarkResult {
                    metric: "hash_rate".to_string(),
                    value: rate,
                    unit: "ops/s".to_string(),
                    metadata: Metadata {
                        device: device_name,
                        range_bits,
                        algorithm: "pollard_kangaroo".to_string(),
                        total_ops,
                        time_seconds,
                    },
                };
                println!("{}", serde_json::to_string(&result)?);
            } else if args.quiet {
                println!("{}", key_hex_display);
            } else {
                info!("Private key found: 0x{}", key_hex_display);
                info!("Verification: SUCCESS");
                info!("Total operations: {}", total_ops);
                info!("Time elapsed: {:.2}s", duration.as_secs_f64());
            }

            if let Some(ref output) = args.output {
                std::fs::write(output, &key_hex)?;
                if !args.quiet && !args.json {
                    info!("Result written to: {}", output);
                }
            }

            return Ok(());
        }

        if total_ops >= max_ops {
            pb.finish_with_message("LIMIT REACHED");
            if !args.quiet && !args.json {
                info!(
                    "Maximum operations reached ({}) without finding key",
                    max_ops
                );
            }
            return Err(anyhow!("Key not found within {} operations", max_ops));
        }
    }
}

#[cfg(all(test, feature = "boha"))]
mod tests {
    use super::*;

    fn make_provider_result(start: &str, end: &str) -> provider::ProviderResult {
        provider::ProviderResult {
            id: "test/1".to_string(),
            pubkey: None,
            start: Some(start.to_string()),
            end: Some(end.to_string()),
            range_bits: Some(66),
        }
    }

    #[test]
    fn test_validate_search_bounds_valid() {
        let pr = make_provider_result("20000000000000000", "40000000000000000");
        assert!(validate_search_bounds("20000000000000000", 64, &pr).is_ok());
    }

    #[test]
    fn test_validate_search_bounds_start_below_minimum() {
        let pr = make_provider_result("20000000000000000", "40000000000000000");
        let result = validate_search_bounds("10000000000000000", 64, &pr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("below"));
    }

    #[test]
    fn test_validate_search_bounds_range_exceeds_end() {
        let pr = make_provider_result("20000000000000000", "40000000000000000");
        // start at 0x30... with range 66 bits would exceed 0x40...
        let result = validate_search_bounds("30000000000000000", 66, &pr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn test_validate_search_bounds_exact_fit() {
        // Range [0x20..., 0x40...) is exactly 2^65 wide
        // Starting at 0x20... with range 65 bits should fit exactly
        let pr = make_provider_result("20000000000000000", "40000000000000000");
        assert!(validate_search_bounds("20000000000000000", 65, &pr).is_ok());
    }
}
