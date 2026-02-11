# Improvement report: v_009_addmul_1_adx

- zint commit: `b6a9718f05752ad3d4c4d9675d222b70ed78cd4b`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Add ADX/BMI2 asm kernel for `mpn_addmul_1` (dual carry chains via `ADOX/ADCX` + `MULX`), runtime-gated via CPUID.
- Integrate the ADX `addmul_1` into basecase `mpn_mul_basecase` (`an >= 20`) and basecase squaring off-diagonals (`n >= 4`).
- Add a dedicated microbench for `mpn_addmul_1_fast` and wire it into `scripts/run_eval.ps1` + plots.

## Correctness

- `zint/tests/test_zint_correctness.exe` passed; includes randomized equivalence check `mpn_addmul_1_fast` vs scalar `mpn_addmul_1`.

## Performance

- `mpn_addmul_1_fast` vs scalar: typically **~2.5–3.0× faster** for `n >= 8` (see `addmul_1.csv`/`addmul_1.png`).
- Basecase mul (`mpn_mul_basecase: n x n`): large win from ADX path, e.g. `n=20` ratio ≈ `0.45`, `n=64` ratio ≈ `0.35`.
- Basecase sqr (`mpn_sqr_basecase: n^2`): also improved, e.g. `n=16` ratio ≈ `0.68`, `n=64` ratio ≈ `0.42`.
- End-to-end bigint benches (`bench_mul.csv`, `bench_div.csv`): no degradation; small/medium sizes remain faster than legacy bigint.
