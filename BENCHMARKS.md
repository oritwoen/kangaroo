# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.29s | 4,194,304 | 3.26 M/s |
| 40-bit | 5.57s | 29,360,128 | 5.27 M/s |
| 48-bit | 40.43s | 222,298,112 | 5.50 M/s |

*Version: 0.3.0 (affine batch addition)*

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
