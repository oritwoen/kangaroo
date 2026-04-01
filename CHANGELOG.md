## [0.8.0] - 2026-04-01

### Features

- *(gpu)* Allow multiple devices (#69)

### Bug Fixes

- *(crypto)* Avoid panic on oversized U256 hex input (#75)
- *(provider)* Allow exact-fit search ranges for provider bounds (#76)
- *(cli)* Make benchmark dispatch explicit (#77)
- *(solver)* Reset DP counter between calibration probes (#83)
- *(ci)* Gate crates publish with release checks (#82)
- *(solver)* Bound GPU poll waits to prevent indefinite hangs (#85)

### Refactor

- *(deps)* Drop unused dashmap, thiserror, and futures (#84)
- *(solver)* Drop unused SharedResources and dead constructors (#89)
- *(crypto)* Drop dead utils, remove stale dead_code allows (#90)
- *(math)* Drop dead LE arithmetic and DP mask helpers (#92)
- *(provider)* Drop unused is_provider predicate (#93)
- *(dp_table)* Drop dead len and is_empty methods (#94)
- *(dp_table)* Use Neg trait instead of Scalar::ZERO - x (#96)

### Performance

- *(solver)* Tighten walk and DP readback (#87)
- *(dp_table)* Inline StoredDP dist as fixed-size array (#88)
- *(cpu)* Avoid heap alloc in hot loop x-coordinate extraction (#91)
- *(solver)* Pipeline GPU dispatch and DP readback (#95)
- *(cpu)* Cache post-jump affine to skip redundant field inversions (#97)
- *(solver)* Cut startup overhead for small-range solves (#99)
- *(gpu)* Cache compute pipelines across repeated solves (#100)
- *(gpu)* Narrow pipeline cache lock scope during compilation (#101)
## [0.7.1] - 2026-03-08

### Bug Fixes

- *(solver)* Stabilize tiny-range boha targets (#64)
- *(init)* 256-bit range arithmetic (#71)

### Refactor

- *(dp_table)* Dedup conversions (#66)

### Performance

- *(dp_table)* O(1) count_by_type (#68)

### Miscellaneous Tasks

- Remove community section from README.md
- Update README.md
- *(release)* V0.7.1
## [0.7.0] - 2026-02-22

### Features

- *(benchmark)* Optional file output (#51)
- Multi-set kangaroos (#38) (#58)
- *(modular)* Add mod constraint filtering for ECDLP (#59)

### Bug Fixes

- *(cpu)* Unify 8-formula candidates (#53)
- *(cpu)* Enforce timeout budget in CPU solver (#55)
- *(cpu)* Unsigned distance mod n (#56)

### Refactor

- *(shaders)* Loopify 256-bit add/sub paths (#45)

### Performance

- *(gpu)* Dual point evaluation via inverse reuse (#39)
- *(gpu)* Harden wgpu readback path (#44)
- *(shader)* Improve benchmark stability (#47)
- *(negmap)* Optimize walk + cycle guards (#57)
- *(gpu)* Tune WGSL shader hot path (#61)

### Miscellaneous Tasks

- Create FUNDING.yml
- *(release)* V0.7.0
## [0.6.0] - 2026-02-15

### Features

- *(benchmark)* Auto-save results to BENCHMARKS.md (#34)

### Bug Fixes

- Full DP mask in WGSL (#28)

### Refactor

- *(gpu)* Replace Blelloch scan with tree-based batch inversion (#31)
- *(gpu)* Double-buffer DP slots and merge copy encoder (#33)

### Documentation

- *(benchmarks)* Add RTX 5060 results (#26)

### Performance

- *(gpu)* Optimize fe_square (#30)
- *(gpu)* Optimize fe_inv with addition chain — 2x speedup (#32)

### Miscellaneous Tasks

- *(release)* V0.6.0
## [0.5.0] - 2026-01-16

### Features

- *(gpu)* Add --backend flag (#25)

### Other

- *(aur)* Enable all features in PKGBUILD

### Documentation

- Update benchmarks for v0.4.0 with performance history
- Add badges and community section

### Miscellaneous Tasks

- *(release)* V0.5.0
## [0.4.0] - 2026-01-11

### Bug Fixes

- *(provider)* Calculate range_bits from actual bounds

### Refactor

- [**breaking**] Remove Criterion benchmarks in favor of built-in --benchmark
- Change CpuKangarooSolver to accept full 256-bit start value

### Documentation

- Add AGENTS.md and fix README architecture section

### Performance

- Implement parallel batch inversion using Blelloch scan (#21)

### Miscellaneous Tasks

- *(release)* V0.4.0
## [0.3.0] - 2026-01-10

### Features

- Add --benchmark flag and BENCHMARK.md (#16)
- Add affine batch addition mode (~30% faster) (#18)

### Bug Fixes

- Correct misleading comments about Jacobian vs affine coordinates

### Miscellaneous Tasks

- *(release)* V0.3.0
## [0.2.0] - 2026-01-08

### Features

- Add GPU auto-calibration
- *(provider)* Add data provider system (#5)

### Other

- *(wgpu)* Upgrade to v28

### Refactor

- Move GPU solver to dedicated module

### Documentation

- Add AUR installation instructions

### Miscellaneous Tasks

- Remove unused shaders
- Add AUR packaging and CI workflow
- Add `deepwiki` badge
- Add `context7.json`
- Add crates.io publish workflow (#7)
- Add autofix.ci workflow for auto-formatting (#10)
- Add build and clippy workflow (#12)
- Add justfile (#14)
- *(release)* V0.2.0
## [0.1.0] - 2025-12-12

### Features

- Initial release
