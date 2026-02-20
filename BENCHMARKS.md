# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark --save-benchmarks
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 0.26s | 4,194,304 | 16.42 M/s |
| 40-bit | 0.74s | 12,582,912 | 16.95 M/s |
| 48-bit | 16.28s | 276,824,064 | 17.00 M/s |

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
| v0.6.0 | 17.00 M/s | +359% |

## Contributing

Have different hardware? Run `kangaroo --benchmark --save-benchmarks` and submit a PR with your results!
