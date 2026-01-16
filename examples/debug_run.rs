use kangaroo::{parse_hex_u256, parse_pubkey, verify_key, GpuBackend, GpuContext, KangarooSolver};
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    let _ = fmt().with_env_filter(EnvFilter::new("debug")).try_init();

    let _puzzle_num = 20;
    let pubkey_hex = "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c";
    let start_hex = "0x80000";
    let range_bits = 20;
    let _expected = "d2c55";
    let num_kangaroos = 256; // Smaller count for debugging

    let pubkey = parse_pubkey(pubkey_hex).expect("valid pubkey");
    let start = parse_hex_u256(start_hex).expect("valid start");
    let dp_bits = 8;

    println!("Initializing context...");
    let ctx = pollster::block_on(GpuContext::new(0, GpuBackend::Auto))?;

    println!("Creating solver...");
    let mut solver = KangarooSolver::new(
        ctx,
        pubkey.clone(),
        start,
        range_bits,
        dp_bits,
        num_kangaroos,
    )?;

    println!("Starting solve...");
    let mut i = 0;
    loop {
        i += 1;
        if i % 10 == 0 {
            println!("Step {}", i);
        }

        if i > 1000 {
            println!("Timeout after 1000 steps");
            break;
        }

        match solver.step() {
            Ok(Some(key)) => {
                let key_hex = hex::encode(&key);
                println!("Found key: {}", key_hex);
                if verify_key(&key, &pubkey) {
                    println!("VERIFIED!");
                } else {
                    println!("MISMATCH!");
                }
                break;
            }
            Ok(None) => continue,
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
