# Kangaroo
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oritwoen/kangaroo)

GPU-accelerated Pollard's Kangaroo algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1.

## Features

- **Cross-platform GPU support** via wgpu (Vulkan/Metal/DX12)
  - AMD GPUs (via Vulkan/RADV)
  - NVIDIA GPUs (via Vulkan)
  - Intel GPUs (via Vulkan)
  - Apple Silicon (via Metal)
- **Pure Rust** implementation with WGSL compute shaders
- **Distinguished Points** optimization for efficient collision detection
- **CPU fallback** for testing and comparison
- **Data providers** for puzzle sources (boha integration)

## Why This Project?

Most existing Kangaroo implementations (JeanLucPons/Kangaroo, RCKangaroo, etc.) only support NVIDIA GPUs via CUDA. This implementation uses WebGPU/wgpu which provides cross-platform GPU compute through Vulkan, Metal, and DX12.

## Installation

### Arch Linux (AUR)

```bash
paru -S kangaroo
```

### Cargo

```bash
cargo install kangaroo
```

### From source

```bash
git clone https://github.com/oritwoen/kangaroo
cd kangaroo
cargo build --release
```

### With boha provider

```bash
cargo build --release --features boha
```

## Usage

```bash
kangaroo --pubkey <PUBKEY> --start <START> --range <BITS>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-t, --target` | - | Data provider target (e.g., `boha:b1000/135`) |
| `-p, --pubkey` | - | Target public key (compressed hex, 33 bytes) |
| `-s, --start` | 0 | Start of search range (hex, without 0x prefix) |
| `-r, --range` | 32 | Search range in bits (key is in [start, start + 2^range]) |
| `-d, --dp-bits` | auto | Distinguished point bits |
| `-k, --kangaroos` | auto | Number of parallel kangaroos |
| `--gpu` | 0 | GPU device index |
| `-o, --output` | - | Output file for result |
| `-q, --quiet` | false | Minimal output, just print found key |
| `--max-ops` | 0 | Max operations (0 = unlimited) |
| `--cpu` | false | Use CPU solver instead of GPU |
| `--json` | false | Output benchmark results in JSON format |
| `--list-providers` | false | List available puzzles from providers |

Either `--target` or `--pubkey` is required.

### Examples

**Using data provider (boha):**

```bash
# Solve puzzle using boha data (auto: pubkey, start, range)
kangaroo --target boha:b1000/66

# Override range (search smaller subset)
kangaroo --target boha:b1000/66 --range 60

# List available puzzles
kangaroo --list-providers
```

**Manual parameters:**

```bash
kangaroo \
    --pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
    --start 8000000000 \
    --range 40
```

## How It Works

The Pollard's Kangaroo algorithm solves the discrete logarithm problem in O(√n) time where n is the search range. It works by:

1. **Tame kangaroos** start from a known point and make random jumps
2. **Wild kangaroos** start from the target public key and make the same type of jumps
3. When a wild and tame kangaroo land on the same point (collision), we can compute the private key

**Distinguished Points (DP)** optimization: Instead of storing all visited points, we only store points whose x-coordinate has a specific number of leading zero bits. This dramatically reduces memory usage while still allowing collision detection.

## Performance

Expected operations: ~2^(range_bits/2)

| Range | Expected Ops | Example Time* |
|-------|-------------|---------------|
| 32 bits | ~65K | < 1 second |
| 40 bits | ~1M | seconds |
| 50 bits | ~33M | minutes |
| 60 bits | ~1B | hours |
| 70 bits | ~34B | days |

*Times vary significantly based on GPU performance

## Use Cases

| Use Case | Example |
|----------|---------|
| Partial key decoded | Puzzle gives ~240 bits, need to find remaining ~16 |
| Key in known range | Know key is between X and Y |
| Verify near-solution | Have candidate, search ±N bits around it |

**NOT useful for:**
- Full 256-bit key search (mathematically impossible)
- BIP39 passphrase brute-force (use dictionary attack instead)
- Puzzles without partial key information

## Library Usage

```rust
use kangaroo::{KangarooSolver, GpuContext, parse_pubkey, parse_hex_u256, verify_key};

fn main() -> anyhow::Result<()> {
    let pubkey = parse_pubkey("03...")?;
    let start = parse_hex_u256("8000000000")?;

    let ctx = pollster::block_on(GpuContext::new(0))?;
    let mut solver = KangarooSolver::new(
        ctx,
        pubkey.clone(),
        start,
        40,  // range_bits
        12,  // dp_bits
        1024, // num_kangaroos
    )?;

    loop {
        if let Some(key) = solver.step()? {
            if verify_key(&key, &pubkey) {
                println!("Found: {}", hex::encode(&key));
                break;
            }
        }
    }

    Ok(())
}
```

## Data Providers

Kangaroo supports external data providers for puzzle sources. Providers supply pubkey, key range, and other puzzle metadata.

### boha (optional feature)

[boha](https://github.com/oritwoen/boha) provides crypto puzzle data including Bitcoin Puzzle Transaction (b1000).

Build with boha support:
```bash
cargo build --release --features boha
```

Usage:
```bash
# Solve specific puzzle
kangaroo --target boha:b1000/66

# List solvable puzzles (unsolved with known pubkey)
kangaroo --list-providers
```

Provider validates range overrides - you cannot search outside the puzzle's key range.

## Architecture

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library entry point
├── cli.rs               # CLI utilities
├── provider/
│   ├── mod.rs           # Provider system interface
│   └── boha.rs          # boha provider (feature-gated)
├── cpu/
│   ├── solver.rs        # GPU solver coordination
│   ├── cpu_solver.rs    # Pure CPU solver
│   └── dp_table.rs      # DP collision detection
├── crypto/
│   └── mod.rs           # k256/secp256k1 wrappers
├── gpu/
│   ├── mod.rs           # GPU module
│   ├── pipeline.rs      # Compute pipeline
│   └── buffers.rs       # GPU buffer management
├── gpu_crypto/
│   ├── context.rs       # GPU context abstraction
│   └── shaders/         # WGSL shader sources
│       ├── field.wgsl           # secp256k1 field arithmetic
│       └── curve_jacobian.wgsl  # Jacobian point operations
└── shaders/
    └── kangaroo_jacobian.wgsl  # Main Kangaroo compute shader
```

## Requirements

- Rust 1.70+
- Vulkan-capable GPU (AMD, NVIDIA, Intel) or Metal (macOS)
- GPU drivers installed

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo) - CUDA implementation (NVIDIA only)
- [RCKangaroo](https://github.com/RetiredC/RCKangaroo) - CUDA implementation (NVIDIA only)
- [boha](https://github.com/oritwoen/boha) - Crypto puzzles and bounties data library
