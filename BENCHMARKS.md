# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate | K |
|-------|------|-----|------|-------|
| 32-bit | 0.06s | 1,048,576 | 18.03 M/s | 16.000 |
| 40-bit | 0.36s | 7,340,032 | 20.64 M/s | 7.000 |
| 48-bit | 0.87s | 18,874,368 | 21.79 M/s | 1.125 |

*Version: 0.8.0*

### GPU: NVIDIA GeForce RTX 5060

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.51s | 4,194,304 | 2.77 M/s |
| 40-bit | 3.63s | 29,360,128 | 8.10 M/s |
| 48-bit | 20.56s | 222,298,112 | 10.81 M/s |

*Version: 0.5.0*

### Performance History

| Version | 48-bit Time | Improvement |
|---------|-------------|-------------|
| v0.3.0 | 30.35s | baseline |
| v0.4.0 | 25.11s | +17% |
| v0.5.0 | 25.10s | +17% |
| v0.6.0 | 12.93s | +57% |
| v0.7.0 | 12.91s | +57% |
| v0.7.1 | 12.92s | +57% |
| v0.8.0 | 0.87s | +97% |

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
