# Benchmark Results

Run benchmarks on your hardware:

```bash
kangaroo --benchmark
```

## Results

### AMD Radeon RX 6800S (RADV NAVI23)

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.79s | 4194304 | 2.35 M/s |
| 40-bit | 8.78s | 29360128 | 3.34 M/s |
| 48-bit | 60.08s | 222298112 | 3.70 M/s |

*Version: 0.2.0*

## Contributing

Have different hardware? Run `kangaroo --benchmark` and submit a PR with your results!
