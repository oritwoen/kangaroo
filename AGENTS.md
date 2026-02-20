# KANGAROO - PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-10
**Commit:** 078d304
**Branch:** main

## OVERVIEW

GPU-accelerated Pollard's Kangaroo algorithm for ECDLP on secp256k1. Cross-platform via wgpu (Vulkan/Metal/DX12). Rust + WGSL compute shaders.

## STRUCTURE

```
kangaroo/
├── src/
│   ├── main.rs              # CLI entry (thin wrapper)
│   ├── lib.rs               # Library entry + Args + run()
│   ├── solver.rs            # GPU solver coordination
│   ├── cli.rs               # Tracing + progress bar utilities
│   ├── cpu/                  
│   │   ├── cpu_solver.rs    # Pure CPU solver (testing/comparison)
│   │   ├── dp_table.rs      # Distinguished Points collision detection
│   │   └── init.rs          # Kangaroo initialization + jump tables
│   ├── crypto/mod.rs        # k256 wrappers: parse_pubkey, verify_key
│   ├── gpu/
│   │   ├── pipeline.rs      # wgpu compute pipeline setup
│   │   └── buffers.rs       # GPU buffer management
│   ├── gpu_crypto/
│   │   ├── context.rs       # GpuContext: device, queue, capabilities
│   │   └── shaders/         # WGSL library modules
│   │       ├── field.wgsl   # secp256k1 field arithmetic
│   │       └── curve.wgsl   # Jacobian point operations
│   ├── shaders/
│   │   └── kangaroo_affine.wgsl  # Main compute shader
│   └── provider/
│       ├── mod.rs           # Provider trait + resolution
│       └── boha.rs          # boha puzzle data (feature-gated)
├── tests/
│   ├── puzzle_solve.rs      # Integration: solve known puzzles
│   ├── cpu_vs_gpu.rs        # Performance comparison benchmark
│   └── fixtures.rs          # Test data (Bitcoin puzzles 20-25 bit)
├── justfile                 # Task runner (test, build, release)
└── PKGBUILD                 # Arch Linux packaging
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add CLI flag | `src/lib.rs` (Args struct) | Uses clap derive |
| GPU compute logic | `src/shaders/kangaroo_affine.wgsl` | Main algorithm |
| Collision detection | `src/cpu/dp_table.rs` | Distinguished Points table |
| secp256k1 field ops | `src/gpu_crypto/shaders/field.wgsl` | 256-bit modular arithmetic |
| Key verification | `src/crypto/mod.rs` | `verify_key`, `full_verify` |
| Add data provider | `src/provider/` | Implement `resolve()` pattern |
| GPU context setup | `src/gpu_crypto/context.rs` | wgpu adapter/device creation |
| Solver state machine | `src/solver.rs` | `step()` method, calibration |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `KangarooSolver` | struct | solver.rs | Main GPU solver orchestration |
| `GpuContext` | struct | gpu_crypto/context.rs | Device + queue + capabilities |
| `KangarooPipeline` | struct | gpu/pipeline.rs | Compute pipeline + bind group |
| `DPTable` | struct | cpu/dp_table.rs | Collision detection via Distinguished Points |
| `CpuKangarooSolver` | struct | cpu/cpu_solver.rs | Reference CPU implementation |
| `verify_key` | fn | crypto/mod.rs | Check private key matches pubkey |
| `parse_pubkey` | fn | crypto/mod.rs | Hex string -> Point |
| `resolve` | fn | provider/mod.rs | Target string -> puzzle params |
| `compute_candidate_keys` | fn | cpu/dp_table.rs | 4-formula collision candidates for negation map |

## CONVENTIONS

- **Shaders**: WGSL files in `src/gpu_crypto/shaders/` (library) and `src/shaders/` (main compute)
- **Field arithmetic**: 256-bit represented as `array<u32, 8>` in WGSL, little-endian limbs
- **DP bits**: Auto-calculated from range_bits, clamped 8-40
- **Feature flags**: `boha` enables puzzle data provider
- **Release profile**: LTO enabled, codegen-units=1, panic=abort, strip=true

## ANTI-PATTERNS

- **Never** use Jacobian coords in final DP storage (must convert to affine)
- **Never** trust DP buffer count without clamping to MAX_DISTINGUISHED_POINTS
- **TODO**: Double buffering for async readback not yet implemented (see solver.rs:355)
- **Never** add `y_parity` or `symClass` to `GpuDistinguishedPoint` (collision resolved by 4-formula CPU-side via `compute_candidate_keys()`)
- **Never** split jump table for negation map (use add/sub with same 256-entry table)

## COMMANDS

```bash
# Development
just test              # cargo test --all-features
just build             # cargo build --release --all-features
just clippy            # cargo clippy --all-features -- -D warnings

# Release
just release 0.4.0     # Bump version, changelog, tag

# Benchmarks
cargo test --release cpu_vs_gpu -- --nocapture --ignored
kangaroo --benchmark   # Built-in benchmark suite

# With boha provider
cargo build --release --features boha
kangaroo --target boha:b1000/66
```

## GPU ARCHITECTURE NOTES

- **Workgroup size**: 64 threads (hardcoded in shader)
- **Steps calibration**: Auto-calibrated to stay under 50ms TDR threshold
- **DP buffer**: 65,536 max entries, capped at 90% capacity per dispatch
- **Jump table**: 256 precomputed points, power-of-2 distances
- **Supported backends**: Vulkan (AMD/NVIDIA/Intel), Metal (Apple), DX12 (Windows)
- **Negation map**: Y-parity determines jump direction (add/sub). Cycle detection via iteration cap (4096) + repeat counter (threshold 8). ~1.29× practical speedup.

## TEST STRATEGY

- `test_smoke_puzzle_20`: Quick sanity check (20-bit puzzle)
- `test_all_puzzles`: Full suite 20-25 bit (use `--ignored`)
- `test_key_verification`: Crypto correctness
- `test_full_address_verification`: Full chain (key -> pubkey -> hash160 -> address)
- Timeout: 60s per puzzle, 2 retries

## NOTES

- `src/kangaroo-0.1.0/`: Legacy packaging artifact, can be ignored
- Software renderer fallback (llvmpipe/SwiftShader) detected and warned
- Progress bar hidden in JSON/quiet mode
- Hex parsing expects no `0x` prefix for start values

<!-- skilld -->
Before modifying code, evaluate each installed skill against the current task.
For each skill, determine YES/NO relevance and invoke all YES skills before proceeding.
<!-- /skilld -->
