# zint workflow (dev + evaluation)

Policy for each improvement stage:
1. Make code changes.
2. Build + run correctness tests.
3. Build + run performance benchmark vs legacy `bigint/`.
4. Plot performance graphs.
5. Write an improvement report.

All evaluation artifacts live under `evaluations/v_xxx/`.

## Running an evaluation

From the parent repo root (the repo that contains both `bigint/` and this `zint/` submodule):

```powershell
.\zint\scripts\run_eval.ps1 v_001_example
```

This will create:
- `evaluations/v_001_example/tests.txt`
- `evaluations/v_001_example/bench.txt`
- `evaluations/v_001_example/bench_mul.csv`
- `evaluations/v_001_example/bench_div.csv`
- `evaluations/v_001_example/bench_mul.png`
- `evaluations/v_001_example/bench_div.png`
- `evaluations/v_001_example/meta.txt`

Then update `evaluations/v_001_example/report.md` with what changed and what improved.

