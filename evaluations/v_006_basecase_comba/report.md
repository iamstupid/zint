# Improvement report: v_006_basecase_comba

- zint commit: `6799711a809683bb2e81efad94c3070b80055c64`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Optimizes basecase multiplication and squaring by adding Comba-style kernels for balanced small sizes (`n <= 16`).
- `mpn_mul_basecase` now uses Comba when `an == bn` and `2 <= n <= 16`.
- `mpn_sqr_basecase` now uses a symmetry-aware Comba squaring kernel for `2 <= n <= 16`.

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK`.

## Performance

- `mpn_mul` vs legacy improves noticeably for sizes that hit Karatsuba leaves heavily (e.g. `n=64..512`), while staying close for larger sizes (see `bench_mul.csv` / `bench_mul.png`).
- `mpn_tdiv_qr` remains comparable/faster vs legacy (see `bench_div.csv` / `bench_div.png`).
