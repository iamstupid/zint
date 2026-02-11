# Improvement report: v_003_bench_stability

- zint commit: `a8925898b50be51eabbb3d656878530403e93615`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Makes `bench/bench_zint_vs_bigint.cpp` less noisy by using best-of-5 timing and higher iteration counts for larger sizes.
- Adds a fast-path for `bigint` bitwise `& | ^` when both operands are non-negative (no two's-complement conversion / no extra limb).

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK` (includes bitwise coverage from `v_002_bitwise`).

## Performance

- Benchmark results are now stable run-to-run (CSV/plots reflect best-of-5 trials).
- `mpn_mul` vs legacy: within about ±1.1% on this run (see `bench_mul.csv` / `bench_mul.png`).
- `mpn_tdiv_qr` vs legacy: consistently faster across tested sizes on this run (see `bench_div.csv` / `bench_div.png`).
