# Improvement report: v_001_radix_cache

- zint commit: `9b39e5729f7cbbe657730a9a0896edfd53495112`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Refactors decimal radix conversion to avoid persistent/global cached power tables.
- Adds optional injected cache (`zint::bigint::radix_powers_cache`) for reuse across repeated `to_string(10)` / `from_string(..., 10)` calls; default path computes powers on-the-fly and frees them.
- Adds a lightweight RNG (`xoshiro256++`) and a standalone correctness test suite that does not rely on the legacy `bigint/` implementation.
- Adds a benchmark harness vs legacy `bigint/` and an evaluation workflow (`scripts/run_eval.ps1`) that captures CSVs + plots.

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK`.
- Coverage includes: small `int64_t` arithmetic vs builtins (bounded to avoid overflow), congruence checks modulo several primes for `+ - *`, division invariants (`a = q*d + r`, sign(r)=sign(a), `|r|<|d|`), and radix roundtrips (base 10 plus bases 2/3/8/16/36).
- Decimal conversion is tested both with `pow_cache=nullptr` and with an injected `radix_powers_cache`.

## Performance

- `mpn_mul` vs legacy: within ~±3% across the tested sizes (see `bench_mul.csv` / `bench_mul.png`).
- `mpn_tdiv_qr` vs legacy: generally faster; one point at `n=512` was ~+5.7% slower in this run (see `bench_div.csv` / `bench_div.png`).
