# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.00s | 4,194,304 | 4.21 M/s |
| 40-bit | 4.15s | 29,360,128 | 7.08 M/s |
| 48-bit | 25.16s | 222,298,112 | 8.84 M/s |

*Version: 0.5.0*

### GPU: NVIDIA GeForce RTX 5060

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.51s | 4,194,304 | 2.77 M/s |
| 40-bit | 3.63s | 29,360,128 | 8.10 M/s |
| 48-bit | 20.56s | 222,298,112 | 10.81 M/s |

*Version: 0.5.0*

### Performance History

| Version | 48-bit Rate | Improvement |
|---------|-------------|-------------|
| v0.2.0  | 3.70 M/s    | baseline    |
| v0.3.0  | 5.50 M/s    | +49%        |
| v0.4.0  | 8.84 M/s    | +139%       |

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
