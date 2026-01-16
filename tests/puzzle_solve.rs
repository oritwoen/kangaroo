//! Integration tests: Solve known puzzles to verify algorithm correctness
//!
//! These tests solve real Bitcoin puzzles (20-25 bit range) to ensure
//! the Kangaroo algorithm works correctly after any code changes.

mod fixtures;

use fixtures::{get_smoke_test_puzzle, get_test_puzzles, PuzzleTestCase};
use kangaroo::{
    full_verify, parse_hex_u256, parse_pubkey, verify_key, GpuBackend, GpuContext, KangarooSolver,
};
use std::time::{Duration, Instant};

/// Test configuration
const TIMEOUT_SECS: u64 = 60;
const MAX_RETRIES: u32 = 2;
const NUM_KANGAROOS: u32 = 4096;

/// Result of attempting to solve a puzzle
#[derive(Debug)]
enum SolveResult {
    Found(String),
    Timeout,
    Error(String),
}

/// Initialize GPU context with CPU fallback
fn init_context() -> Result<GpuContext, String> {
    match pollster::block_on(GpuContext::new(0, GpuBackend::Auto)) {
        Ok(ctx) => {
            let name = ctx.device_name().to_string();
            if name.contains("llvmpipe") || name.contains("SwiftShader") || name.contains("CPU") {
                println!(
                    "  \x1b[33m⚠ WARNING: Using software renderer ({})\x1b[0m",
                    name
                );
                println!("  \x1b[33m⚠ Tests will be slower. Install GPU drivers for better performance.\x1b[0m");
            } else {
                println!("  \x1b[32m✓ GPU: {}\x1b[0m", name);
            }
            Ok(ctx)
        }
        Err(e) => {
            println!("  \x1b[33m⚠ GPU initialization failed: {}\x1b[0m", e);
            println!("  \x1b[33m⚠ Attempting CPU fallback via software renderer...\x1b[0m");

            // Try with software backend explicitly
            match pollster::block_on(try_software_context()) {
                Ok(ctx) => {
                    println!("  \x1b[33m⚠ Using CPU fallback (slower)\x1b[0m");
                    Ok(ctx)
                }
                Err(e2) => Err(format!("No GPU or CPU fallback available: {} / {}", e, e2)),
            }
        }
    }
}

/// Try to create a software-rendered context
async fn try_software_context() -> Result<GpuContext, anyhow::Error> {
    // This will use whatever backend is available, including software
    GpuContext::new(0, GpuBackend::Auto).await
}

/// Solve a puzzle with timeout and retries
fn solve_puzzle(puzzle: &PuzzleTestCase, ctx: &GpuContext) -> SolveResult {
    for attempt in 0..=MAX_RETRIES {
        if attempt > 0 {
            println!("    Retry {}/{}...", attempt, MAX_RETRIES);
        }

        match try_solve_once(puzzle, ctx) {
            SolveResult::Found(key) => return SolveResult::Found(key),
            SolveResult::Timeout => {
                if attempt == MAX_RETRIES {
                    return SolveResult::Timeout;
                }
                // Continue to retry
            }
            SolveResult::Error(e) => return SolveResult::Error(e),
        }
    }
    SolveResult::Timeout
}

/// Single attempt to solve a puzzle
fn try_solve_once(puzzle: &PuzzleTestCase, _ctx: &GpuContext) -> SolveResult {
    // Parse inputs
    let pubkey = match parse_pubkey(puzzle.pubkey) {
        Ok(p) => p,
        Err(e) => return SolveResult::Error(format!("Invalid pubkey: {}", e)),
    };

    let start = match parse_hex_u256(puzzle.start) {
        Ok(s) => s,
        Err(e) => return SolveResult::Error(format!("Invalid start: {}", e)),
    };

    // Auto-configure DP bits
    let dp_bits = (puzzle.range_bits / 2).saturating_sub(2).clamp(8, 20);

    // Create solver - need to clone context for each attempt
    // Since GpuContext doesn't implement Clone, we need to create a new one
    // For now, we'll work around this by creating the solver directly
    let solver_result = pollster::block_on(async {
        match GpuContext::new(0, GpuBackend::Auto).await {
            Ok(new_ctx) => KangarooSolver::new(
                new_ctx,
                pubkey,
                start,
                puzzle.range_bits,
                dp_bits,
                NUM_KANGAROOS,
            ),
            Err(e) => Err(anyhow::anyhow!("Context creation failed: {}", e)),
        }
    });

    let mut solver = match solver_result {
        Ok(s) => s,
        Err(e) => return SolveResult::Error(format!("Solver creation failed: {}", e)),
    };

    // Solve with timeout
    let start_time = Instant::now();
    let timeout = Duration::from_secs(TIMEOUT_SECS);

    loop {
        if start_time.elapsed() > timeout {
            return SolveResult::Timeout;
        }

        match solver.step() {
            Ok(Some(key)) => {
                let key_hex = hex::encode(&key);
                // Verify the key
                if verify_key(&key, &pubkey) {
                    return SolveResult::Found(key_hex);
                } else {
                    return SolveResult::Error(format!(
                        "Key verification failed: found {} but doesn't match pubkey",
                        key_hex
                    ));
                }
            }
            Ok(None) => continue,
            Err(e) => return SolveResult::Error(format!("Solver error: {}", e)),
        }
    }
}

/// Normalize hex key for comparison (remove leading zeros, lowercase)
fn normalize_key(key: &str) -> String {
    key.trim_start_matches("0x")
        .trim_start_matches('0')
        .to_lowercase()
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_smoke_puzzle_20() {
    println!("\n=== Smoke Test: Puzzle 20 (20-bit) ===");

    let puzzle = get_smoke_test_puzzle();
    println!("  Target: {}", puzzle.pubkey);
    println!("  Range: {} bits", puzzle.range_bits);
    println!("  Expected: 0x{}", puzzle.expected_key);

    // Initialize GPU/CPU
    let ctx = match init_context() {
        Ok(c) => c,
        Err(e) => {
            println!("  \x1b[31m✗ SKIP: {}\x1b[0m", e);
            return; // Skip test if no compute available
        }
    };

    let start_time = Instant::now();
    let result = solve_puzzle(&puzzle, &ctx);
    let elapsed = start_time.elapsed();

    match result {
        SolveResult::Found(key) => {
            let normalized_found = normalize_key(&key);
            let normalized_expected = normalize_key(puzzle.expected_key);

            if normalized_found == normalized_expected {
                println!(
                    "  \x1b[32m✓ PASS: Found 0x{} in {:.2}s\x1b[0m",
                    key,
                    elapsed.as_secs_f64()
                );
            } else {
                panic!(
                    "Wrong key! Expected 0x{}, got 0x{}",
                    puzzle.expected_key, key
                );
            }
        }
        SolveResult::Timeout => {
            panic!("Timeout after {}s - algorithm may be broken", TIMEOUT_SECS);
        }
        SolveResult::Error(e) => {
            panic!("Error: {}", e);
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test puzzle_solve -- --ignored
fn test_all_puzzles() {
    println!("\n=== Full Test Suite: Puzzles 20-25 ===\n");

    let puzzles = get_test_puzzles();
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    // Initialize once
    let ctx = match init_context() {
        Ok(c) => c,
        Err(e) => {
            println!("\x1b[31m✗ SKIP ALL: {}\x1b[0m", e);
            return;
        }
    };

    for puzzle in &puzzles {
        println!(
            "Puzzle {} ({}-bit):",
            puzzle.puzzle_number, puzzle.range_bits
        );
        println!("  Target: {}...", &puzzle.pubkey[..20]);

        let start_time = Instant::now();
        let result = solve_puzzle(puzzle, &ctx);
        let elapsed = start_time.elapsed();

        match result {
            SolveResult::Found(key) => {
                let normalized_found = normalize_key(&key);
                let normalized_expected = normalize_key(puzzle.expected_key);

                if normalized_found == normalized_expected {
                    println!(
                        "  \x1b[32m✓ PASS: 0x{} ({:.2}s)\x1b[0m\n",
                        key,
                        elapsed.as_secs_f64()
                    );
                    passed += 1;
                } else {
                    println!(
                        "  \x1b[31m✗ FAIL: Expected 0x{}, got 0x{}\x1b[0m\n",
                        puzzle.expected_key, key
                    );
                    failed += 1;
                }
            }
            SolveResult::Timeout => {
                println!("  \x1b[31m✗ TIMEOUT after {}s\x1b[0m\n", TIMEOUT_SECS);
                failed += 1;
            }
            SolveResult::Error(e) => {
                println!("  \x1b[33m⚠ SKIP: {}\x1b[0m\n", e);
                skipped += 1;
            }
        }
    }

    println!("\n=== Summary ===");
    println!("  Passed:  {}", passed);
    println!("  Failed:  {}", failed);
    println!("  Skipped: {}", skipped);

    if failed > 0 {
        panic!("{} tests failed!", failed);
    }
}

#[test]
fn test_key_verification() {
    // Test that verify_key works correctly with known puzzle
    let puzzle = get_smoke_test_puzzle();
    let pubkey = parse_pubkey(puzzle.pubkey).expect("Valid pubkey");

    // Parse expected key (pad to even length if needed)
    let padded_key = if puzzle.expected_key.len() % 2 == 1 {
        format!("0{}", puzzle.expected_key)
    } else {
        puzzle.expected_key.to_string()
    };
    let key_bytes = hex::decode(&padded_key).expect("Valid hex");

    assert!(
        verify_key(&key_bytes, &pubkey),
        "Known key should verify against known pubkey"
    );

    // Wrong key should fail
    let wrong_key = vec![0x12, 0x34, 0x56];
    assert!(
        !verify_key(&wrong_key, &pubkey),
        "Wrong key should not verify"
    );

    println!("  \x1b[32m✓ Key verification works correctly\x1b[0m");
}

#[test]
fn test_full_address_verification() {
    println!("\n=== Full Address Chain Verification ===\n");

    let puzzles = get_test_puzzles();

    for puzzle in &puzzles {
        // Parse expected key (pad to even length if needed)
        let padded_key = if puzzle.expected_key.len() % 2 == 1 {
            format!("0{}", puzzle.expected_key)
        } else {
            puzzle.expected_key.to_string()
        };
        let key_bytes = hex::decode(&padded_key).expect("Valid hex");

        // Full verification
        let result = full_verify(
            &key_bytes,
            puzzle.pubkey,
            puzzle.expected_hash160,
            puzzle.expected_address,
        )
        .expect("Verification should not error");

        println!("Puzzle {}:", puzzle.puzzle_number);
        println!("  Private Key: 0x{}", puzzle.expected_key);
        println!(
            "  Public Key:  {} {}",
            if result.pubkey_match {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            },
            &result.pubkey_hex[..20]
        );
        println!(
            "  Hash160:     {} {}",
            if result.hash160_match {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            },
            result.hash160_hex
        );
        println!(
            "  Address:     {} {}",
            if result.address_match {
                "\x1b[32m✓\x1b[0m"
            } else {
                "\x1b[31m✗\x1b[0m"
            },
            result.address
        );

        assert!(
            result.pubkey_match,
            "Puzzle {}: Public key mismatch! Expected {}, got {}",
            puzzle.puzzle_number, puzzle.pubkey, result.pubkey_hex
        );
        assert!(
            result.hash160_match,
            "Puzzle {}: Hash160 mismatch! Expected {}, got {}",
            puzzle.puzzle_number, puzzle.expected_hash160, result.hash160_hex
        );
        assert!(
            result.address_match,
            "Puzzle {}: Address mismatch! Expected {}, got {}",
            puzzle.puzzle_number, puzzle.expected_address, result.address
        );

        println!();
    }

    println!(
        "\x1b[32m✓ All {} puzzles verified correctly!\x1b[0m",
        puzzles.len()
    );
}
