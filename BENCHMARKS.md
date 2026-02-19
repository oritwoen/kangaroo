# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 0.25s | 4,194,304 | 16.72 M/s |
| 40-bit | 1.72s | 29,360,128 | 17.02 M/s |
| 48-bit | 13.03s | 222,298,112 | 17.06 M/s |

*Version: 0.6.0*

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
| v0.2.0 | 3.70 M/s | baseline |
| v0.3.0 | 5.50 M/s | +49% |
| v0.4.0 | 8.84 M/s | +139% |
| v0.6.0 | 17.06 M/s | +361% |

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
