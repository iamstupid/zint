# Improvement report: v_008_basecase_dispatch

- zint commit: `db0b89af331a3c83064807927174e006c2e5f6f6`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Fix `mpn_mul_basecase` regression for `n >= 20` by preventing Comba kernels from being inlined into the dispatcher.
- Disable Comba squaring (was consistently slower on `n <= 16`).
- Adjust Comba-mul dispatch to `6 <= n <= 16` (avoids slow `n=2..5` corner).

## Correctness

- `zint/tests/test_zint_correctness.exe` passed (RNG-driven arithmetic + modular checks).

## Performance

- Basecase mul (`mpn_mul_basecase: n x n`): large-size slowdown removed; `n=20..64` now ~parity (ratio ≈ 1.0).
- Basecase sqr (`mpn_sqr_basecase: n^2`): Comba regression removed; `n=2..16` now ~parity (ratio ≈ 1.0).
- End-to-end bigint benches (`bench_mul.csv`, `bench_div.csv`): no degradation vs legacy bigint; small/medium sizes remain faster.
