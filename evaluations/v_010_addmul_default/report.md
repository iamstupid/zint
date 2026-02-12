# Improvement report: v_010_addmul_default

- zint commit: `e0ffde1ebfc1a2d45efb3353df9d763599e84e66`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Make `zint::mpn_addmul_1` the default ADX/BMI2-dispatching implementation; keep the previous code as `mpn_addmul_1_scalar`.
- Remove the special-case ADX paths in `mul.hpp` so the faster `mpn_addmul_1` is used uniformly across mul/sqr basecases (and thus Karatsuba callers too).
- Update correctness tests and the `bench_addmul_1` microbench to compare `mpn_addmul_1_scalar` vs default `mpn_addmul_1`.

## Correctness

- `zint/tests/test_zint_correctness.exe` passed; includes randomized equivalence checks for `mpn_addmul_1` vs `mpn_addmul_1_scalar`.

## Performance

- `mpn_addmul_1` vs scalar: `n=64` ratio `0.353` (≈2.8× faster), `n=128` ratio `0.330` (≈3.0×). See `addmul_1.csv`/`addmul_1.png`.
- Basecase mul (`mpn_mul_basecase: n x n`): `n=5` ratio `0.699`, `n=64` ratio `0.353` (large win preserved); `n=2` still slower (`1.189`).
- Basecase sqr (`mpn_sqr_basecase: n^2`): `n=16` ratio `0.693`, `n=64` ratio `0.439`.
- End-to-end bigint benches: no degradation overall; e.g. mul `n=512` ratio `0.784`, div `n=64` ratio `0.707`.
