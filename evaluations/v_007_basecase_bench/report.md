# Improvement report: v_007_basecase_bench

- zint commit: `101fbb9fcf0a760d35fe417b48f4b69aa5fb6ad7`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Adds explicit basecase kernel benchmark `bench/bench_basecase.cpp` and wires it into `scripts/run_eval.ps1`.
- Produces `basecase_mul.csv/png` and `basecase_sqr.csv/png` for direct measurement of the “small-n” work we’ve been optimizing.

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK`.

## Performance

- `mpn_mul_basecase` (n×n): Comba-mul helps for `n≈6..16` (e.g. `n=16` is ~0.77×), but is slower at `n=2..4`, and the current dispatch shows a consistent slowdown for `n>=20` even though the classic path is used (see `basecase_mul.csv` / `basecase_mul.png`).
- `mpn_sqr_basecase` (n²): current Comba-sqr is **slower** for `n<=16` (see `basecase_sqr.csv` / `basecase_sqr.png`), so we should not keep using it as-is.
