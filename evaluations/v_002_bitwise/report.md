# Improvement report: v_002_bitwise

- zint commit: `b148df5f2eb7f2a6eba716ec0cdbdceef696a83e`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Adds AVX2 limb-array bitwise kernels: `mpn_and_n`, `mpn_or_n`, `mpn_xor_n`, `mpn_not_n`.
- Adds `zint::bigint` bitwise operators `& | ^ ~` with **infinite two's-complement** semantics for negative values.
- Extends correctness tests to cover bitwise semantics using `int64_t` reference checks, boolean-algebra identities, and low-bit validation via `mod 2^k`.

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK`.
- New coverage includes:
  - `int64_t` reference checks for `& | ^ ~` (two's complement).
  - Identities like `x + y == (x ^ y) + 2*(x & y)` and `x | ~x == -1`.
  - Low-bit checks for `& | ^` under `mod 2^k` for multiple `k` values (1..256).

## Performance

- No expected impact to mul/div kernels; evaluation still compares `mpn_mul` and `mpn_tdiv_qr` vs legacy `bigint/`.
- This run shows `mpn_mul` roughly within a few percent of legacy and `mpn_tdiv_qr` generally comparable, with noticeable run-to-run noise at some sizes (see `bench_*.csv` / `bench_*.png`).
