# Improvement report: v_005_addlsh

- zint commit: `cef7f076885b68c9a139a3ca3438d7b06b338b44`
- parent commit: `b5fc6948e265a5de98c32e0bc8a5b595a15d00a2`

## Summary

- Adds `zint::mpn_addlsh_n` (fused add with left-shifted addend) in `mpn.hpp`.
- Extends correctness tests with randomized verification against `bigint` arithmetic, including an aliasing case (`rp == ap`).

## Correctness

- `tests/test_zint_correctness.cpp` prints `OK`.

## Performance

- No expected impact to existing mul/div/bitwise benchmarks yet (kernel is newly added; integration into hot paths is next).
