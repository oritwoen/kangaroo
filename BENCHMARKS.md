# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate | K |
|-------|------|-----|------|-------|
| 32-bit | 0.37s | 8,388,608 | 22.53 M/s | 128.000 |
| 40-bit | 0.72s | 16,777,216 | 23.14 M/s | 16.000 |
| 48-bit | 17.98s | 419,430,400 | 23.33 M/s | 25.000 |

*Version: 0.7.0*

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
| v0.7.0 | 23.33 M/s | +531% |

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
