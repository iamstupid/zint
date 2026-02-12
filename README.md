# zint

Header-only arbitrary-precision integer library in C++17, with optional x86-64
ASM kernels (ADX/BMI2) and AVX2-accelerated NTT multiplication.

## Features

- **Sign-magnitude bigint** with full operator overloading (`+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `^`, `~`, comparisons)
- **Multiplication chain**: schoolbook basecase (ASM) &rarr; Karatsuba &rarr; NTT, with dedicated squaring paths
- **Division chain**: single-limb Barrett &rarr; schoolbook (Knuth Algorithm D) &rarr; Newton with precomputed reciprocal
- **Radix conversion**: D&C O(M(n) log n) for all bases 2-64; O(n) bit-extraction for power-of-2 bases; LUT-accelerated basecase
- **Two-tier NTT**: `p30x3` (3 &times; 30-bit primes, u32 Montgomery, up to ~12M elements) and `p50x4` (4 &times; 50-bit primes, FP Barrett, unlimited)
- **AVX2 throughout**: bitwise ops, shifts, carry propagation, comparison, NTT butterflies
- **Thread-local bump allocator** (`scratch.hpp`): zero-overhead temp memory via mark/restore
- **ASM kernels** (optional, MASM): `addmul_1` (ADX dual-carry, 1.7 cyc/limb on Zen 4), `submul_1`, fused `mul_basecase`

## Quick start

zint is header-only. Include a single header:

```cpp
#include "zint/zint.hpp"

int main() {
    zint::bigint a("123456789012345678901234567890");
    zint::bigint b("987654321098765432109876543210");
    zint::bigint c = a * b;
    std::cout << c.to_string() << "\n";       // base 10
    std::cout << c.to_string(16) << "\n";     // base 16
    return 0;
}
```

## Building

### Requirements

- C++17 compiler with AVX2 support
- x86-64 target (SSE2/AVX2 intrinsics)
- Optional: MASM (`ml64`) for ASM kernels

### MSVC (recommended)

```powershell
# Without ASM kernels:
cl /std:c++17 /O2 /EHsc /arch:AVX2 your_program.cpp

# With ASM kernels (requires ml64 on PATH):
ml64 /nologo /c /Fo zint\asm\addmul_1_adx.obj zint\asm\addmul_1_adx.asm
ml64 /nologo /c /Fo zint\asm\submul_1_adx.obj zint\asm\submul_1_adx.asm
ml64 /nologo /c /Fo zint\asm\mul_basecase_adx.obj zint\asm\mul_basecase_adx.asm
cl /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 ^
    your_program.cpp zint\asm\*.obj
```

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

See [CMakeLists.txt](CMakeLists.txt) for options:
- `ZINT_USE_ADX_ASM` &mdash; enable ADX/BMI2 ASM kernels (default: `ON` on Windows/MSVC with MASM available)

### Convenience script (PowerShell)

```powershell
# Build and run a single .cpp file:
.\build.ps1 your_program.cpp your_program.exe

# With ASM kernels:
.\build.ps1 your_program.cpp your_program.exe -Asm
```

## Project structure

```
zint.hpp            Single entry-point header (includes bigint.hpp)
bigint.hpp          Bigint class: operators, radix conversion, bitwise
mpn.hpp             Low-level limb ops: add, sub, mul_1, divrem_1, shifts, AVX2 bitwise
mul.hpp             Multiplication dispatch: basecase, Karatsuba, NTT bridge
div.hpp             Division dispatch: schoolbook, Newton reciprocal
scratch.hpp         Thread-local bump allocator
tuning.hpp          Crossover thresholds (tuned for Zen 4)
rng.hpp             xoshiro256++ PRNG for tests/benchmarks
asm/                Optional MASM kernels (ADX/BMI2)
  addmul_1_adx.asm    rp[] += ap[] * b, dual-carry ADX, 2-way unrolled
  submul_1_adx.asm    rp[] -= ap[] * b, BMI2 MULX
  mul_basecase_adx.asm Fused schoolbook multiply (eliminates per-row overhead)
ntt/                NTT engine (internal, used by mul.hpp)
  api.hpp           Top-level NTT multiply dispatch
  p30x3/            3-prime 30-bit engine (u32 Montgomery, mixed-radix)
  p50x4/            4-prime 50-bit engine (FP Barrett, Bailey 4-step)
  simd/             AVX2 helpers (v4.hpp, avx2.hpp)
```

## Tuning

All crossover thresholds are in [`tuning.hpp`](tuning.hpp). Current values are tuned for AMD Zen 4:

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `KARATSUBA_THRESHOLD` | 32 | Basecase &rarr; Karatsuba (limbs) |
| `NTT_BN_MIN` | 128 | Min smaller operand for NTT |
| `NTT_AREA` | 40960 | `an*bn` crossover area for NTT |
| `SQR_KARATSUBA_THRESHOLD` | 176 | Squaring: basecase &rarr; Karatsuba |
| `SQR_NTT_THRESHOLD` | 224 | Squaring: Karatsuba &rarr; NTT |
| `DIV_DC_THRESHOLD` | 60 | Schoolbook &rarr; Newton division |
| `RADIX_DC_THRESHOLD` | 30 | Basecase &rarr; D&C radix conversion |

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
