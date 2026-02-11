# Improvement report: v_004_bitwise_bench

- zint commit: `3ac6dd69786d81c751303cbf0dd2b35acb5df5b7`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Adds `bench/bench_bitwise.cpp` to benchmark mpn-level bitwise kernels and bigint bitwise operators.
- Extends `scripts/run_eval.ps1` to run the bitwise benchmarks and generate plots.
- Improves `bigint` bitwise assignment operators `&= |= ^=` to compute in-place (avoids per-op heap allocations); binary ops `& | ^` use the direct compute path.

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK` (includes extensive bitwise semantics coverage).

## Performance

- Core mul/div benchmarks remain comparable to legacy `bigint/` (see `bench_*.csv` / `bench_*.png`).
- `mpn_*` bitwise kernels vs scalar baseline: for medium/large sizes, explicit AVX2 kernels are typically ~2–5× faster (see `bitwise_mpn_*.csv` / `bitwise_mpn_*.png`); very small `n` is overhead-dominated.
- `bigint` bitwise operator timing: `pp` (both non-negative) is roughly linear in limbs; `nn` (both negative) is much slower due to two’s-complement conversion work (see `bitwise_bigint.csv` / `bitwise_bigint.png`).
