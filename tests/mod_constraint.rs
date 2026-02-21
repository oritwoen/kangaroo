//! Integration tests: ModConstraint + CpuKangarooSolver + key recovery
//!
//! CPU-only — no GPU required.

use k256::ProjectivePoint;
use kangaroo::{parse_hex_u256, parse_pubkey, verify_key, CpuKangarooSolver, ModConstraint};
use std::time::Duration;

/// Solve via modular constraint pipeline and verify the result.
/// M > 1: solver finds j in reduced problem (j*H = Q), recovers k = R + M*j.
/// M = 1: solver works on the original problem directly.
fn solve_with_constraint(
    pubkey_hex: &str,
    start_hex: &str,
    range_bits: u32,
    expected_key_hex: &str,
    mod_step_hex: &str,
    mod_start_hex: &str,
) {
    let pubkey = parse_pubkey(pubkey_hex).expect("parse pubkey");
    let start_le = parse_hex_u256(start_hex).expect("parse start");

    let constraint =
        ModConstraint::new(mod_step_hex, mod_start_hex, &pubkey, &start_le, range_bits)
            .expect("create constraint");

    // When M > 1, constraint MUST be Some — verifies that constraint path is active
    if mod_step_hex != "1" {
        assert!(
            constraint.is_some(),
            "M={} > 1 should produce a constraint (got None)",
            mod_step_hex
        );
    }

    let (solve_pubkey, solve_start_be, solve_range_bits, solve_base) = match constraint {
        Some(ref c) => {
            let mut j_be = c.j_start;
            j_be.reverse(); // LE → BE for CpuKangarooSolver
            (
                c.transformed_pubkey,
                j_be,
                c.effective_range_bits,
                c.base_point,
            )
        }
        None => {
            let mut start_be = start_le;
            start_be.reverse(); // LE → BE
            (pubkey, start_be, range_bits, ProjectivePoint::GENERATOR)
        }
    };

    let mut solver = CpuKangarooSolver::new(
        solve_pubkey,
        solve_start_be,
        solve_range_bits,
        8,
        solve_base,
    );

    let result = solver.solve(Duration::from_secs(60));
    assert!(result.is_some(), "Should find key within timeout");

    let j_or_key = result.unwrap();

    // k = R + M*j when constraint is active
    let recovered_key = match constraint {
        Some(ref c) => {
            use k256::elliptic_curve::ops::Reduce;
            use k256::U256 as K256U256;

            let mut j_be_arr = [0u8; 32];
            let len = j_or_key.len().min(32);
            j_be_arr[32 - len..].copy_from_slice(&j_or_key[..len]);

            let j_scalar = k256::Scalar::reduce(K256U256::from_be_slice(&j_be_arr));
            let k_scalar = c.mod_start + c.mod_step * j_scalar;
            let k_be = k_scalar.to_bytes();
            let first_nonzero = k_be.iter().position(|&x| x != 0).unwrap_or(31);
            k_be[first_nonzero..].to_vec()
        }
        None => j_or_key,
    };

    assert!(
        verify_key(&recovered_key, &pubkey),
        "Recovered key must verify against original pubkey"
    );

    let found_hex = hex::encode(&recovered_key);
    let found_trimmed = found_hex.trim_start_matches('0');
    assert_eq!(
        found_trimmed, expected_key_hex,
        "Found key must match expected"
    );
}

// Puzzle 20: k=863317 (0xd2c55). M=1,R=0 → identity, no transformation.
#[test]
fn test_mod_constraint_identity_solve_puzzle20() {
    solve_with_constraint(
        "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
        "80000",
        20,
        "d2c55",
        "1",
        "0",
    );
}

// Puzzle 20: k=863317. 863317 mod 7 = 0 → M=7,R=0, j=123331, eff_bits=18
#[test]
fn test_mod_constraint_m7r0_solve_puzzle20() {
    solve_with_constraint(
        "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
        "80000",
        20,
        "d2c55",
        "7",
        "0",
    );
}

// Puzzle 20: k=863317. 863317 mod 3 = 1 → M=3,R=1, j=287772, eff_bits=19
#[test]
fn test_mod_constraint_m3r1_solve_puzzle20() {
    solve_with_constraint(
        "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
        "80000",
        20,
        "d2c55",
        "3",
        "1",
    );
}

// Puzzle 21: k=1811764 (0x1ba534). 1811764 mod 4 = 0 → M=4,R=0, j=452941, eff_bits=19
#[test]
fn test_mod_constraint_m4r0_solve_puzzle21() {
    solve_with_constraint(
        "031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e",
        "100000",
        21,
        "1ba534",
        "4",
        "0",
    );
}
