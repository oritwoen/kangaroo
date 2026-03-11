//! Kangaroo: Pollard's Kangaroo ECDLP solver using Vulkan/Metal/DX12 compute
//!
//! GPU-accelerated implementation for solving the Elliptic Curve Discrete
//! Logarithm Problem on secp256k1 within a known range.
//!
//! Supports AMD, NVIDIA, Intel GPUs via wgpu (Vulkan/Metal/DX12).

mod benchmark;
mod cli;
mod convert;
mod cpu;
mod crypto;
mod gpu;
mod gpu_crypto;
mod math;
mod modular;
mod provider;
mod solver;

pub use cpu::CpuKangarooSolver;
pub use crypto::{
    full_verify, parse_hex_u256, parse_pubkey, verify_key, verify_key_with_base, Point,
};
pub use gpu_crypto::{enumerate_gpus, GpuBackend, GpuContext, GpuDeviceInfo};
pub use modular::ModConstraint;
pub use solver::KangarooSolver;

use anyhow::anyhow;
use clap::Parser;
use indicatif::ProgressBar;
use k256::elliptic_curve::ops::Reduce;
use k256::U256 as K256U256;
use k256::{ProjectivePoint, Scalar};
#[cfg(feature = "boha")]
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info};

/// Pollard's Kangaroo ECDLP solver for secp256k1
///
/// Finds private key k such that P = k*G, given that k is in range [start, start + 2^range_bits - 1]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Public key to solve (compressed hex, 33 bytes)
    #[arg(short, long)]
    pubkey: Option<String>,

    /// Start of search range (hex, without 0x prefix)
    #[arg(short, long)]
    start: Option<String>,

    /// Bit range to search (key is in [start, start + 2^range - 1])
    #[arg(short, long)]
    range: Option<u32>,

    /// Data provider target (e.g., boha:b1000/135)
    #[arg(short, long)]
    target: Option<String>,

    /// List available puzzles from providers
    #[arg(long)]
    list_providers: bool,

    /// List available GPU devices
    #[arg(long)]
    list_gpus: bool,

    /// Distinguished point bits (auto-calculated if not set)
    #[arg(short, long)]
    dp_bits: Option<u32>,

    /// Number of kangaroos (default: auto based on GPU)
    #[arg(short, long)]
    kangaroos: Option<u32>,

    /// GPU device selection (index, comma-separated indices, or "all")
    #[arg(long, default_value = "0")]
    gpu: String,

    /// Include integrated GPUs in `--gpu all` selection
    #[arg(long)]
    include_integrated: bool,

    /// GPU backend to use
    #[arg(long, value_enum, default_value = "auto")]
    backend: gpu_crypto::GpuBackend,

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

    /// Save benchmark results to BENCHMARKS.md (opt-in)
    #[arg(long)]
    save_benchmarks: bool,

    /// Modular step (hex): search only for x ≡ mod_start (mod mod_step) [e.g. 3c = 60]
    #[arg(long, default_value = "1")]
    mod_step: String,

    /// Modular residue (hex): class residue for constraint (0 ≤ R < M) [e.g. 25 = 37]
    #[arg(long, default_value = "0")]
    mod_start: String,
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
    k_factor: f64,
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

            let range_bits = match args.range {
                Some(user_range) => {
                    // User provided explicit range - validate it
                    validate_search_bounds(&start_str, user_range, pr)?;
                    user_range
                }
                None => {
                    // No user range - calculate from provider bounds if available
                    calculate_range_bits_from_provider(&start_str, pr)?
                }
            };

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

#[cfg(feature = "boha")]
fn calculate_range_bits_from_provider(
    start: &str,
    provider: &provider::ProviderResult,
) -> anyhow::Result<u32> {
    let (Some(ref provider_start), Some(ref provider_end)) = (&provider.start, &provider.end)
    else {
        return provider.range_bits.ok_or_else(|| {
            anyhow!(
                "Provider '{}' has no range information. Use --range to specify search range.",
                provider.id
            )
        });
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

    if start_val > provider_end_val {
        return Err(anyhow!(
            "Start 0x{} exceeds puzzle '{}' maximum 0x{}",
            start,
            provider.id,
            provider_end
        ));
    }

    let range_size = &provider_end_val - &start_val + BigUint::from(1u32);
    let bits = range_size.bits() as u32;

    Ok(bits)
}

#[cfg(not(feature = "boha"))]
fn calculate_range_bits_from_provider(
    _start: &str,
    provider: &provider::ProviderResult,
) -> anyhow::Result<u32> {
    provider.range_bits.ok_or_else(|| {
        anyhow!(
            "Provider '{}' has no range information. Use --range to specify search range.",
            provider.id
        )
    })
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

    let search_end = &start_val + (BigUint::from(1u64) << range_bits) - BigUint::from(1u32);
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

/// Parse GPU selection string into sorted, deduplicated list of indices
///
/// Supports: "all", single index ("0"), comma-separated ("0,1,2").
/// Deduplicates and validates against available GPU count.
fn parse_gpu_selection(gpu_str: &str, available_count: usize) -> anyhow::Result<Vec<u32>> {
    if available_count == 0 {
        return Err(anyhow!("No GPU devices available"));
    }

    let trimmed = gpu_str.trim();

    if trimmed.eq_ignore_ascii_case("all") {
        return Ok((0..available_count as u32).collect());
    }

    let mut indices: Vec<u32> = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let idx: u32 = part.parse().map_err(|_| {
            anyhow!(
                "Invalid GPU index '{}'. Use a number, comma-separated numbers, or 'all'",
                part
            )
        })?;
        if idx as usize >= available_count {
            return Err(anyhow!(
                "GPU index {} out of range. Available GPUs: 0..{} ({} device{})",
                idx,
                available_count - 1,
                available_count,
                if available_count == 1 { "" } else { "s" }
            ));
        }
        indices.push(idx);
    }

    if indices.is_empty() {
        return Err(anyhow!("Empty GPU selection"));
    }

    indices.sort();
    indices.dedup();
    Ok(indices)
}

fn filter_integrated_from_all_selection(
    selected: Vec<u32>,
    gpu_devices: &[gpu_crypto::GpuDeviceInfo],
    gpu_arg_raw: &str,
    include_integrated: bool,
) -> Vec<u32> {
    if include_integrated || !gpu_arg_raw.trim().eq_ignore_ascii_case("all") {
        return selected;
    }

    let selected_infos: Vec<&gpu_crypto::GpuDeviceInfo> = selected
        .iter()
        .filter_map(|idx| gpu_devices.iter().find(|d| d.index == *idx))
        .collect();

    let has_discrete = selected_infos
        .iter()
        .any(|d| d.device_type == wgpu::DeviceType::DiscreteGpu);
    if !has_discrete {
        return selected;
    }

    let filtered: Vec<u32> = selected_infos
        .into_iter()
        .filter(|d| d.device_type != wgpu::DeviceType::IntegratedGpu)
        .map(|d| d.index)
        .collect();

    if filtered.is_empty() {
        selected
    } else {
        filtered
    }
}

fn gpu_weight_for_device_type(device_type: wgpu::DeviceType) -> u32 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 8,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::Cpu => 1,
        _ => 1,
    }
}

fn allocate_weighted_kangaroos(total_k: u32, weights: &[u32], min_per_gpu: u32) -> Vec<u32> {
    let mut allocation = vec![min_per_gpu; weights.len()];
    if weights.is_empty() {
        return allocation;
    }

    let min_total = min_per_gpu.saturating_mul(weights.len() as u32);
    let remaining = total_k.saturating_sub(min_total);
    if remaining == 0 {
        return allocation;
    }

    let normalized_weights: Vec<u64> = weights.iter().map(|&w| u64::from(w.max(1))).collect();
    let weight_sum: u64 = normalized_weights.iter().sum::<u64>().max(1);

    let mut assigned_extra = 0u32;
    let mut remainders = Vec::with_capacity(weights.len());

    for (idx, w) in normalized_weights.iter().enumerate() {
        let numer = u64::from(remaining) * *w;
        let extra = (numer / weight_sum) as u32;
        let rem = numer % weight_sum;
        allocation[idx] = allocation[idx].saturating_add(extra);
        assigned_extra = assigned_extra.saturating_add(extra);
        remainders.push((rem, idx));
    }

    let mut leftovers = remaining.saturating_sub(assigned_extra);
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for &(_, idx) in &remainders {
        if leftovers == 0 {
            break;
        }
        allocation[idx] = allocation[idx].saturating_add(1);
        leftovers -= 1;
    }

    allocation
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct CalibrationCache {
    version: String,
    entries: HashMap<String, u32>,
}

fn calibration_cache_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(
            PathBuf::from(path)
                .join("kangaroo")
                .join("gpu-calibration.json"),
        );
    }
    std::env::var_os("HOME").map(PathBuf::from).map(|p| {
        p.join(".cache")
            .join("kangaroo")
            .join("gpu-calibration.json")
    })
}

fn load_calibration_cache() -> CalibrationCache {
    let Some(path) = calibration_cache_path() else {
        return CalibrationCache::default();
    };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return CalibrationCache::default();
    };
    let Ok(cache) = serde_json::from_str::<CalibrationCache>(&raw) else {
        return CalibrationCache::default();
    };
    if cache.version != env!("CARGO_PKG_VERSION") {
        return CalibrationCache::default();
    }
    cache
}

fn save_calibration_cache(cache: &CalibrationCache) {
    let Some(path) = calibration_cache_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(raw) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(path, raw);
    }
}

fn auto_calibrate_gpu_weights(
    gpu_contexts: &[(u32, gpu_crypto::GpuContext, u32, wgpu::DeviceType)],
    solve_pubkey: &Point,
    solve_start: &crate::crypto::U256,
    solve_range_bits: u32,
    dp_bits: u32,
    solve_base_point: &ProjectivePoint,
    probe_k_hint: u32,
) -> Vec<u32> {
    let mut measured_weights = Vec::with_capacity(gpu_contexts.len());
    let mut cache = load_calibration_cache();
    let mut cache_dirty = false;

    for (gpu_index, ctx, fallback_weight, _) in gpu_contexts {
        let probe_k = probe_k_hint.clamp(3, 131_072);
        let cache_key = format!(
            "{}|{}|{}|{}",
            ctx.device_name(),
            ctx.backend() as u32,
            dp_bits,
            probe_k
        );
        if let Some(cached_weight) = cache.entries.get(&cache_key) {
            let weight = (*cached_weight).max(1);
            tracing::info!("Calibration cache GPU {}: weight {}", gpu_index, weight);
            measured_weights.push(weight);
            continue;
        }

        let solver = solver::KangarooSolver::new_with_base_no_dp_table(
            ctx.clone(),
            *solve_pubkey,
            *solve_start,
            solve_range_bits,
            dp_bits,
            probe_k,
            probe_k,
            *solve_base_point,
            0,
        );

        let mut solver = match solver {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "Calibration failed to initialize GPU {}: {}. Falling back to heuristic weight {}",
                    gpu_index,
                    e,
                    fallback_weight
                );
                measured_weights.push((*fallback_weight).max(1));
                continue;
            }
        };

        if let Err(e) = solver.step_collect() {
            tracing::warn!(
                "Calibration warmup error on GPU {}: {}. Falling back to heuristic weight {}",
                gpu_index,
                e,
                fallback_weight
            );
            measured_weights.push((*fallback_weight).max(1));
            continue;
        }

        let probe_end = Instant::now() + Duration::from_millis(450);
        let start_t = Instant::now();
        let mut ops: u64 = 0;
        while Instant::now() < probe_end {
            match solver.step_collect() {
                Ok((_, delta)) => ops = ops.saturating_add(delta),
                Err(e) => {
                    tracing::warn!(
                        "Calibration probe error on GPU {}: {}. Falling back to heuristic weight {}",
                        gpu_index,
                        e,
                        fallback_weight
                    );
                    ops = 0;
                    break;
                }
            }
        }

        if ops == 0 {
            measured_weights.push((*fallback_weight).max(1));
            continue;
        }

        let elapsed = start_t.elapsed().as_secs_f64();
        let ops_per_sec = if elapsed > 0.0 {
            ops as f64 / elapsed
        } else {
            0.0
        };
        let measured = (ops_per_sec / 1_000_000.0).round() as u32;
        let weight = measured.max(1);
        tracing::info!(
            "Calibration GPU {}: {:.2}M ops/s -> weight {}",
            gpu_index,
            ops_per_sec / 1_000_000.0,
            weight
        );
        measured_weights.push(weight);
        cache.entries.insert(cache_key, weight);
        cache_dirty = true;
    }

    if cache_dirty {
        cache.version = env!("CARGO_PKG_VERSION").to_string();
        save_calibration_cache(&cache);
    }

    measured_weights
}

fn print_gpu_list(devices: &[gpu_crypto::GpuDeviceInfo]) {
    if devices.is_empty() {
        println!("No GPU devices found.");
        return;
    }

    println!("Available GPUs:");
    println!(
        "{:>5}  {:<40} {:<12} {:<8}",
        "Index", "Name", "Type", "Backend"
    );
    println!("{}", "-".repeat(70));

    for dev in devices {
        let type_str = match dev.device_type {
            wgpu::DeviceType::DiscreteGpu => "Discrete",
            wgpu::DeviceType::IntegratedGpu => "Integrated",
            wgpu::DeviceType::VirtualGpu => "Virtual",
            wgpu::DeviceType::Cpu => "CPU",
            _ => "Other",
        };
        let backend_str = match dev.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "GL",
            _ => "Other",
        };
        println!(
            "{:>5}  {:<40} {:<12} {:<8}",
            dev.index, dev.name, type_str, backend_str
        );
    }
}

fn recover_key_from_j(j_bytes: &[u8], mod_step: Scalar, mod_start: Scalar) -> Vec<u8> {
    debug_assert!(j_bytes.len() <= 32, "j_bytes too long: {}", j_bytes.len());
    let mut j_be = [0u8; 32];
    let len = j_bytes.len().min(32);
    j_be[32 - len..].copy_from_slice(&j_bytes[..len]);
    let j_uint = K256U256::from_be_slice(&j_be);
    let j_scalar = Scalar::reduce(j_uint);

    let k_scalar = mod_start + mod_step * j_scalar;
    let k_be = k_scalar.to_bytes();
    let first_nonzero = k_be.iter().position(|&x| x != 0).unwrap_or(k_be.len() - 1);
    k_be[first_nonzero..].to_vec()
}

pub fn run(args: Args) -> anyhow::Result<()> {
    cli::init_tracing(false, args.quiet || args.json || args.benchmark);

    if args.list_gpus {
        let devices = pollster::block_on(gpu_crypto::enumerate_gpus(args.backend))?;
        print_gpu_list(&devices);
        return Ok(());
    }

    if args.list_providers {
        print_providers_list();
        return Ok(());
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

    let constraint = crate::modular::ModConstraint::new(
        &args.mod_step,
        &args.mod_start,
        &pubkey,
        &start,
        range_bits,
    )
    .map_err(|e| anyhow!("Invalid modular constraint: {e}"))?;

    let effective_range = constraint
        .as_ref()
        .map(|c| c.effective_range_bits)
        .unwrap_or(range_bits);

    if args.cpu {
        if !args.quiet && !args.json {
            info!("Mode: CPU (Software Solver)");
        }

        let dp_bits = args
            .dp_bits
            .map(|v| v.clamp(8, 20))
            .unwrap_or_else(|| (effective_range / 2).saturating_sub(2).clamp(8, 20));

        if !args.quiet && !args.json {
            info!("DP bits: {}", dp_bits);
        }

        let (solve_pubkey, solve_start_be, solve_range_bits, solve_base_point) = match &constraint {
            Some(c) => {
                let mut j_start_be = c.j_start;
                j_start_be.reverse();
                (
                    c.transformed_pubkey,
                    j_start_be,
                    c.effective_range_bits,
                    c.base_point,
                )
            }
            None => {
                let mut start_be = start;
                start_be.reverse();
                (pubkey, start_be, range_bits, ProjectivePoint::GENERATOR)
            }
        };

        let mut solver = cpu::CpuKangarooSolver::new(
            solve_pubkey,
            solve_start_be,
            solve_range_bits,
            dp_bits,
            solve_base_point,
        );

        let expected_ops = 1u128
            .checked_shl((effective_range / 2) as u32)
            .unwrap_or(u64::MAX as u128)
            .min(u64::MAX as u128) as u64;
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

        if let Some(j_or_key) = result {
            let private_key = match &constraint {
                Some(c) => recover_key_from_j(&j_or_key, c.mod_step, c.mod_start),
                None => j_or_key,
            };
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
                let k_factor = total_ops as f64 / (2.0_f64).powf(effective_range as f64 / 2.0);

                let result = BenchmarkResult {
                    metric: "hash_rate".to_string(),
                    value: rate,
                    unit: "ops/s".to_string(),
                    metadata: Metadata {
                        device: "cpu".to_string(),
                        range_bits: effective_range,
                        algorithm: "pollard_kangaroo".to_string(),
                        total_ops,
                        time_seconds,
                        k_factor,
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
                info!(
                    "K-factor: {:.3}",
                    solver.total_ops() as f64 / (2.0_f64).powf(effective_range as f64 / 2.0)
                );
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

    // Parse GPU selection and validate against available devices
    let gpu_devices = pollster::block_on(gpu_crypto::enumerate_gpus(args.backend))?;
    let gpu_indices = filter_integrated_from_all_selection(
        parse_gpu_selection(&args.gpu, gpu_devices.len())?,
        &gpu_devices,
        &args.gpu,
        args.include_integrated,
    );

    if args.benchmark {
        if gpu_indices.len() > 1 {
            return Err(anyhow!(
                "Benchmark mode only supports a single GPU. Use --gpu N to select one."
            ));
        }
        return benchmark::run(gpu_indices[0], args.backend, args.save_benchmarks);
    }

    if gpu_indices.len() == 1 {
        let single_backend = gpu_devices
            .iter()
            .find(|d| d.index == gpu_indices[0])
            .map(|d| gpu_crypto::GpuBackend::from_wgpu_backend(d.backend))
            .unwrap_or(args.backend);
        let gpu_context =
            pollster::block_on(gpu_crypto::GpuContext::new(gpu_indices[0], single_backend))?;
        let device_name = gpu_context.device_name().to_string();
        if !args.quiet && !args.json {
            info!("GPU: {}", device_name);
            info!("Compute units: {}", gpu_context.compute_units());
        }

        let requested_num_k = args.kangaroos.unwrap_or(gpu_context.optimal_kangaroos());
        let max_k_for_range = if effective_range >= 32 {
            u32::MAX
        } else {
            (1u32 << effective_range).max(3)
        };
        let num_k = requested_num_k.min(max_k_for_range);
        if !args.quiet && !args.json && num_k != requested_num_k {
            info!(
                "Capping kangaroos from {} to {} for {}-bit range",
                requested_num_k, num_k, effective_range
            );
        }
        let dp_bits = args.dp_bits.map(|v| v.clamp(8, 40)).unwrap_or_else(|| {
            let auto_dp = (effective_range / 2).saturating_sub((num_k as f64).log2() as u32 / 2);
            auto_dp.clamp(8, 40)
        });

        if !args.quiet && !args.json {
            info!("DP bits: {}", dp_bits);
            info!("Kangaroos: {}", num_k);
        }

        let mut solver = match &constraint {
            Some(c) => solver::KangarooSolver::new_with_base(
                gpu_context,
                c.transformed_pubkey,
                c.j_start,
                c.effective_range_bits,
                dp_bits,
                num_k,
                c.base_point,
            )?,
            None => {
                solver::KangarooSolver::new(gpu_context, pubkey, start, range_bits, dp_bits, num_k)?
            }
        };

        let expected_ops = 1u128
            .checked_shl((effective_range / 2) as u32)
            .unwrap_or(u64::MAX as u128)
            .min(u64::MAX as u128) as u64;
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

            if let Some(j_or_key) = result {
                let private_key = match &constraint {
                    Some(c) => recover_key_from_j(&j_or_key, c.mod_step, c.mod_start),
                    None => j_or_key,
                };
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
                    let k_factor = total_ops as f64 / (2.0_f64).powf(effective_range as f64 / 2.0);

                    let result = BenchmarkResult {
                        metric: "hash_rate".to_string(),
                        value: rate,
                        unit: "ops/s".to_string(),
                        metadata: Metadata {
                            device: device_name,
                            range_bits: effective_range,
                            algorithm: "pollard_kangaroo".to_string(),
                            total_ops,
                            time_seconds,
                            k_factor,
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
                    info!(
                        "K-factor: {:.3}",
                        total_ops as f64 / (2.0_f64).powf(effective_range as f64 / 2.0)
                    );
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

    let mut gpu_contexts = Vec::new();
    for &gpu_index in &gpu_indices {
        // Use the specific backend from enumeration to ensure index consistency
        let backend = gpu_devices
            .iter()
            .find(|d| d.index == gpu_index)
            .map(|d| gpu_crypto::GpuBackend::from_wgpu_backend(d.backend))
            .unwrap_or(args.backend);
        let device_type = gpu_devices
            .iter()
            .find(|d| d.index == gpu_index)
            .map(|d| d.device_type)
            .unwrap_or(wgpu::DeviceType::IntegratedGpu);
        let weight = gpu_weight_for_device_type(device_type);

        match pollster::block_on(gpu_crypto::GpuContext::new(gpu_index, backend)) {
            Ok(ctx) => gpu_contexts.push((gpu_index, ctx, weight, device_type)),
            Err(e) => tracing::warn!("Failed to initialize GPU {}: {}", gpu_index, e),
        }
    }
    if gpu_contexts.is_empty() {
        return Err(anyhow!("Failed to initialize any selected GPU"));
    }

    let num_gpus = gpu_contexts.len() as u32;
    let device_name = gpu_contexts
        .iter()
        .map(|(_, ctx, _, _)| ctx.device_name().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if !args.quiet && !args.json {
        info!("GPUs: {}", device_name);
        info!("GPU workers: {}", num_gpus);
    }

    let max_k_for_range = if effective_range >= 32 {
        u32::MAX
    } else {
        (1u32 << effective_range).max(3)
    };
    let requested_total_k = args.kangaroos.unwrap_or_else(|| {
        gpu_contexts
            .iter()
            .map(|(_, ctx, _, _)| ctx.optimal_kangaroos())
            .fold(0u32, |acc, k| acc.saturating_add(k))
    });
    let total_requested_capped = requested_total_k.min(max_k_for_range);
    let min_per_gpu = 3u32;
    let min_total_required = min_per_gpu.saturating_mul(num_gpus);
    if total_requested_capped < min_total_required {
        return Err(anyhow!(
            "Need at least {} kangaroos per selected GPU ({} GPUs selected, minimum total {}). Increase --kangaroos or select fewer GPUs.",
            min_per_gpu,
            num_gpus,
            min_total_required
        ));
    }
    let total_k = total_requested_capped;
    if !args.quiet && !args.json && total_k != requested_total_k {
        info!(
            "Capping kangaroos from {} to {} for {}-bit range",
            requested_total_k, total_k, effective_range
        );
    }
    let dp_bits = args.dp_bits.map(|v| v.clamp(8, 40)).unwrap_or_else(|| {
        let auto_dp = (effective_range / 2).saturating_sub((total_k as f64).log2() as u32 / 2);
        auto_dp.clamp(8, 40)
    });

    let (solve_pubkey, solve_start, solve_range_bits, solve_base_point) = match &constraint {
        Some(c) => (
            c.transformed_pubkey,
            c.j_start,
            c.effective_range_bits,
            c.base_point,
        ),
        None => (pubkey, start, range_bits, ProjectivePoint::GENERATOR),
    };

    let calibrated_weights = auto_calibrate_gpu_weights(
        &gpu_contexts,
        &solve_pubkey,
        &solve_start,
        solve_range_bits,
        dp_bits,
        &solve_base_point,
        total_requested_capped / num_gpus,
    );
    let per_gpu_k_allocation =
        allocate_weighted_kangaroos(total_requested_capped, &calibrated_weights, min_per_gpu);

    if !args.quiet && !args.json {
        info!("DP bits: {}", dp_bits);
        info!("Total kangaroos: {}", total_k);
        for (((gpu_index, ctx, _, device_type), calibrated_weight), per_gpu_k) in gpu_contexts
            .iter()
            .zip(calibrated_weights.iter())
            .zip(per_gpu_k_allocation.iter())
        {
            info!(
                "GPU {} ({}, {:?}, weight={}): {} kangaroos",
                gpu_index,
                ctx.device_name(),
                device_type,
                calibrated_weight,
                per_gpu_k
            );
        }
    }

    let mut solvers = Vec::with_capacity(gpu_contexts.len());
    let mut kangaroo_offset = 0u32;
    for ((gpu_index, ctx, _, _), per_gpu_k) in gpu_contexts
        .into_iter()
        .zip(per_gpu_k_allocation.into_iter())
    {
        let solver = solver::KangarooSolver::new_with_base_no_dp_table(
            ctx,
            solve_pubkey,
            solve_start,
            solve_range_bits,
            dp_bits,
            per_gpu_k,
            total_k,
            solve_base_point,
            kangaroo_offset,
        )
        .map_err(|e| anyhow!("Failed to initialize solver for GPU {}: {}", gpu_index, e))?;
        solvers.push((gpu_index, solver));
        kangaroo_offset = kangaroo_offset.saturating_add(per_gpu_k);
    }

    let expected_ops = 1u128
        .checked_shl((effective_range / 2) as u32)
        .unwrap_or(u64::MAX as u128)
        .min(u64::MAX as u128) as u64;
    let pb = if args.quiet || args.json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(expected_ops);
        pb.set_style(cli::default_progress_style());
        pb
    };

    if !args.quiet && !args.json {
        info!("Starting multi-GPU search...");
    }

    let max_ops = if args.max_ops == 0 {
        u64::MAX
    } else {
        args.max_ops
    };
    let start_time = Instant::now();

    let queue_capacity = (solvers.len() * 2).max(1);
    let (tx, rx) = mpsc::sync_channel::<Vec<gpu::GpuDistinguishedPoint>>(queue_capacity);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::with_capacity(solvers.len());
    for (gpu_index, mut solver) in solvers {
        let tx = tx.clone();
        let stop_flag = Arc::clone(&stop_flag);
        let total_ops = Arc::clone(&total_ops);
        let handle = thread::spawn(move || loop {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            match solver.step_collect() {
                Ok((dps, ops_delta)) => {
                    total_ops.fetch_add(ops_delta, Ordering::Relaxed);
                    if dps.is_empty() {
                        continue;
                    }
                    if tx.send(dps).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("GPU worker {} stopped: {}", gpu_index, e);
                    break;
                }
            }
        });
        handles.push(handle);
    }
    drop(tx);

    let mut dp_table = cpu::DPTable::new(solve_start, solve_pubkey, solve_base_point);
    let mut found_key: Option<Vec<u8>> = None;

    let mut last_log_ops: u64 = 0;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(dps) => {
                for dp in dps {
                    if let Some(j_or_key) = dp_table.insert_and_check(dp) {
                        // Verify candidate key before stopping all workers
                        let candidate = match &constraint {
                            Some(c) => recover_key_from_j(&j_or_key, c.mod_step, c.mod_start),
                            None => j_or_key,
                        };
                        if crypto::verify_key(&candidate, &pubkey) {
                            found_key = Some(candidate);
                            stop_flag.store(true, Ordering::Relaxed);
                            break;
                        } else {
                            tracing::warn!(
                                "Collision candidate failed verification, continuing search"
                            );
                        }
                    }
                }
                if found_key.is_some() {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let current_ops = total_ops.load(Ordering::Relaxed);
        pb.set_position(current_ops);

        // Periodic DP stats logging (every ~10M ops, matching single-GPU behavior)
        if current_ops.saturating_sub(last_log_ops) >= 10_000_000 {
            last_log_ops = current_ops;
            let (tame, w1, w2) = dp_table.count_by_type();
            tracing::info!(
                "Ops: {}M | DPs: {} ({} tame, {} wild1, {} wild2)",
                current_ops / 1_000_000,
                dp_table.total_dps(),
                tame,
                w1,
                w2
            );
        }

        if current_ops >= max_ops {
            stop_flag.store(true, Ordering::Relaxed);
            break;
        }
    }

    stop_flag.store(true, Ordering::Relaxed);
    drop(rx);
    for handle in handles {
        if let Err(e) = handle.join() {
            tracing::warn!("GPU worker thread panicked: {:?}", e);
        }
    }

    let final_total_ops = total_ops.load(Ordering::Relaxed);

    if let Some(private_key) = found_key {
        let duration = start_time.elapsed();
        pb.finish_with_message("FOUND!");
        let key_hex = hex::encode(&private_key);
        let key_hex_trimmed = key_hex.trim_start_matches('0');
        let key_hex_display = if key_hex_trimmed.is_empty() {
            "0"
        } else {
            key_hex_trimmed
        };

        if args.json {
            let time_seconds = duration.as_secs_f64();
            let rate = final_total_ops as f64 / time_seconds;
            let k_factor = final_total_ops as f64 / (2.0_f64).powf(effective_range as f64 / 2.0);

            let result = BenchmarkResult {
                metric: "hash_rate".to_string(),
                value: rate,
                unit: "ops/s".to_string(),
                metadata: Metadata {
                    device: device_name,
                    range_bits: effective_range,
                    algorithm: "pollard_kangaroo".to_string(),
                    total_ops: final_total_ops,
                    time_seconds,
                    k_factor,
                },
            };
            println!("{}", serde_json::to_string(&result)?);
        } else if args.quiet {
            println!("{}", key_hex_display);
        } else {
            info!("Private key found: 0x{}", key_hex_display);
            info!("Verification: SUCCESS");
            info!("Total operations: {}", final_total_ops);
            info!("Time elapsed: {:.2}s", duration.as_secs_f64());
            info!(
                "K-factor: {:.3}",
                final_total_ops as f64 / (2.0_f64).powf(effective_range as f64 / 2.0)
            );
        }

        if let Some(ref output) = args.output {
            std::fs::write(output, &key_hex)?;
            if !args.quiet && !args.json {
                info!("Result written to: {}", output);
            }
        }

        return Ok(());
    }

    if final_total_ops >= max_ops {
        pb.finish_with_message("LIMIT REACHED");
        if !args.quiet && !args.json {
            info!(
                "Maximum operations reached ({}) without finding key",
                max_ops
            );
        }
        return Err(anyhow!("Key not found within {} operations", max_ops));
    }

    pb.finish_with_message("STOPPED");
    Err(anyhow!("All GPU workers stopped before finding key"))
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
        let pr = make_provider_result("20000000000000000", "40000000000000000");
        assert!(validate_search_bounds("20000000000000000", 65, &pr).is_ok());
    }

    #[test]
    fn test_validate_search_bounds_exact_fit_8_bit_inclusive() {
        let pr = make_provider_result("0", "ff");
        assert!(validate_search_bounds("0", 8, &pr).is_ok());
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn test_cli_mod_flags_default() {
        // Verify that mod_step and mod_start fields exist with correct defaults
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
        ])
        .expect("Failed to parse args");

        assert_eq!(args.mod_step, "1", "mod_step should default to '1'");
        assert_eq!(args.mod_start, "0", "mod_start should default to '0'");
    }

    #[test]
    fn test_cli_mod_flags_custom() {
        // Verify that custom mod_step and mod_start values are parsed correctly
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
            "--mod-step",
            "7",
            "--mod-start",
            "3",
        ])
        .expect("Failed to parse args");

        assert_eq!(args.mod_step, "7", "mod_step should be '7'");
        assert_eq!(args.mod_start, "3", "mod_start should be '3'");
    }

    #[test]
    fn test_cli_gpu_default() {
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
        ])
        .expect("Failed to parse args");
        assert_eq!(args.gpu, "0");
    }

    #[test]
    fn test_cli_gpu_custom() {
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
            "--gpu",
            "1,2",
        ])
        .expect("Failed to parse args");
        assert_eq!(args.gpu, "1,2");
    }

    #[test]
    fn test_cli_gpu_all() {
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
            "--gpu",
            "all",
        ])
        .expect("Failed to parse args");
        assert_eq!(args.gpu, "all");
    }

    #[test]
    fn test_cli_include_integrated_flag() {
        let args = Args::try_parse_from([
            "kangaroo",
            "--pubkey",
            "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
            "--range",
            "20",
            "--gpu",
            "all",
            "--include-integrated",
        ])
        .expect("Failed to parse args");
        assert!(args.include_integrated);
    }

    #[test]
    fn test_cli_list_gpus_flag() {
        let args = Args::try_parse_from(["kangaroo", "--list-gpus"]).expect("Failed to parse args");
        assert!(args.list_gpus);
    }

    #[test]
    fn test_parse_gpu_selection_single() {
        assert_eq!(parse_gpu_selection("0", 4).unwrap(), vec![0]);
        assert_eq!(parse_gpu_selection("2", 4).unwrap(), vec![2]);
        assert_eq!(parse_gpu_selection("3", 4).unwrap(), vec![3]);
    }

    #[test]
    fn test_parse_gpu_selection_all() {
        assert_eq!(parse_gpu_selection("all", 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_gpu_selection("ALL", 2).unwrap(), vec![0, 1]);
        assert_eq!(parse_gpu_selection("All", 1).unwrap(), vec![0]);
    }

    #[test]
    fn test_parse_gpu_selection_comma_separated() {
        assert_eq!(parse_gpu_selection("0,1,2", 4).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_gpu_selection("2,0,1", 3).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_gpu_selection_dedup() {
        assert_eq!(parse_gpu_selection("0,0", 2).unwrap(), vec![0]);
        assert_eq!(parse_gpu_selection("1,1,1", 3).unwrap(), vec![1]);
        assert_eq!(parse_gpu_selection("2,1,2,0,1", 3).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_gpu_selection_out_of_range() {
        assert!(parse_gpu_selection("4", 4).is_err());
        assert!(parse_gpu_selection("1", 1).is_err());
        assert!(parse_gpu_selection("0,5", 4).is_err());
    }

    #[test]
    fn test_parse_gpu_selection_invalid() {
        assert!(parse_gpu_selection("abc", 4).is_err());
        assert!(parse_gpu_selection("-1", 4).is_err());
        assert!(parse_gpu_selection("", 4).is_err());
    }

    #[test]
    fn test_parse_gpu_selection_no_gpus() {
        assert!(parse_gpu_selection("0", 0).is_err());
        assert!(parse_gpu_selection("all", 0).is_err());
    }

    fn mk_gpu(index: u32, device_type: wgpu::DeviceType) -> gpu_crypto::GpuDeviceInfo {
        gpu_crypto::GpuDeviceInfo {
            name: format!("gpu-{index}"),
            device_type,
            backend: wgpu::Backend::Vulkan,
            index,
        }
    }

    #[test]
    fn test_filter_integrated_from_all_selection_drops_integrated_when_discrete_present() {
        let devices = vec![
            mk_gpu(0, wgpu::DeviceType::DiscreteGpu),
            mk_gpu(1, wgpu::DeviceType::IntegratedGpu),
        ];
        let selected = vec![0, 1];
        let out = filter_integrated_from_all_selection(selected, &devices, "all", false);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn test_filter_integrated_from_all_selection_keeps_integrated_when_flag_set() {
        let devices = vec![
            mk_gpu(0, wgpu::DeviceType::DiscreteGpu),
            mk_gpu(1, wgpu::DeviceType::IntegratedGpu),
        ];
        let selected = vec![0, 1];
        let out = filter_integrated_from_all_selection(selected, &devices, "all", true);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn test_filter_integrated_from_all_selection_keeps_manual_selection() {
        let devices = vec![
            mk_gpu(0, wgpu::DeviceType::DiscreteGpu),
            mk_gpu(1, wgpu::DeviceType::IntegratedGpu),
        ];
        let selected = vec![1];
        let out = filter_integrated_from_all_selection(selected, &devices, "1", false);
        assert_eq!(out, vec![1]);
    }
}
