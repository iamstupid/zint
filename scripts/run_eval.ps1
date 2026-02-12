param(
    [Parameter(Mandatory=$true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parentRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path

. (Join-Path $PSScriptRoot "msvc_env.ps1")

$evalDir = Join-Path $repoRoot ("evaluations\\" + $Version)
New-Item -ItemType Directory -Force $evalDir | Out-Null

$meta = Join-Path $evalDir "meta.txt"
$testsOut = Join-Path $evalDir "tests.txt"
$benchOut = Join-Path $evalDir "bench.txt"
$bitwiseOut = Join-Path $evalDir "bitwise.txt"
$mulCsv = Join-Path $evalDir "bench_mul.csv"
$divCsv = Join-Path $evalDir "bench_div.csv"
$mulPng = Join-Path $evalDir "bench_mul.png"
$divPng = Join-Path $evalDir "bench_div.png"

$baseOut = Join-Path $evalDir "basecase.txt"
$baseMulCsv = Join-Path $evalDir "basecase_mul.csv"
$baseSqrCsv = Join-Path $evalDir "basecase_sqr.csv"
$baseMulPng = Join-Path $evalDir "basecase_mul.png"
$baseSqrPng = Join-Path $evalDir "basecase_sqr.png"

$addmulOut = Join-Path $evalDir "addmul_1.txt"
$addmulCsv = Join-Path $evalDir "addmul_1.csv"
$addmulPng = Join-Path $evalDir "addmul_1.png"

$mpnAndCsv = Join-Path $evalDir "bitwise_mpn_and.csv"
$mpnOrCsv  = Join-Path $evalDir "bitwise_mpn_or.csv"
$mpnXorCsv = Join-Path $evalDir "bitwise_mpn_xor.csv"
$mpnNotCsv = Join-Path $evalDir "bitwise_mpn_not.csv"
$mpnAndPng = Join-Path $evalDir "bitwise_mpn_and.png"
$mpnOrPng  = Join-Path $evalDir "bitwise_mpn_or.png"
$mpnXorPng = Join-Path $evalDir "bitwise_mpn_xor.png"
$mpnNotPng = Join-Path $evalDir "bitwise_mpn_not.png"
$bigintCsv = Join-Path $evalDir "bitwise_bigint.csv"
$bigintPng = Join-Path $evalDir "bitwise_bigint.png"

Set-Location $repoRoot
$zintHead = (git rev-parse HEAD)

Set-Location $parentRoot
$parentHead = (git rev-parse HEAD)

@(
    "date_utc=" + (Get-Date).ToUniversalTime().ToString("o")
    "parent_repo=" + $parentRoot
    "parent_head=" + $parentHead
    "zint_repo=" + $repoRoot
    "zint_head=" + $zintHead
) | Out-File -Encoding utf8 $meta

# Build optional ADX/BMI2 asm kernels (linked into all executables when enabled).
Set-Location $parentRoot
ml64 /nologo /c /Fo zint\\asm\\addmul_1_adx.obj zint\\asm\\addmul_1_adx.asm | Out-Null

# Build + run tests (must not reference legacy bigint for correctness).
Set-Location $parentRoot
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 zint\tests\test_zint_correctness.cpp zint\\asm\\addmul_1_adx.obj /Fe:zint\tests\test_zint_correctness.exe | Out-File -Encoding utf8 $testsOut
cmd /c "zint\\tests\\test_zint_correctness.exe" | Out-File -Append -Encoding utf8 $testsOut

# Build + run benchmark vs legacy bigint.
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 zint\bench\bench_zint_vs_bigint.cpp zint\\asm\\addmul_1_adx.obj /Fe:zint\bench\bench_zint_vs_bigint.exe | Out-File -Encoding utf8 $benchOut
cmd /c "zint\\bench\\bench_zint_vs_bigint.exe --csv-mul zint\\evaluations\\$Version\\bench_mul.csv --csv-div zint\\evaluations\\$Version\\bench_div.csv" | Out-File -Append -Encoding utf8 $benchOut

# Plot graphs.
python zint\scripts\plot_bench.py --mul $mulCsv --div $divCsv --out-mul $mulPng --out-div $divPng | Out-File -Append -Encoding utf8 $benchOut

# Build + run bitwise benchmarks (mpn + bigint).
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 zint\bench\bench_bitwise.cpp zint\\asm\\addmul_1_adx.obj /Fe:zint\bench\bench_bitwise.exe | Out-File -Encoding utf8 $bitwiseOut
cmd /c "zint\\bench\\bench_bitwise.exe --csv-mpn-and zint\\evaluations\\$Version\\bitwise_mpn_and.csv --csv-mpn-or zint\\evaluations\\$Version\\bitwise_mpn_or.csv --csv-mpn-xor zint\\evaluations\\$Version\\bitwise_mpn_xor.csv --csv-mpn-not zint\\evaluations\\$Version\\bitwise_mpn_not.csv --csv-bigint zint\\evaluations\\$Version\\bitwise_bigint.csv" | Out-File -Append -Encoding utf8 $bitwiseOut

# Plot bitwise graphs.
python zint\scripts\plot_compare.py --csv $mpnAndCsv --out $mpnAndPng --title "mpn_and_n (u64 limbs)" | Out-File -Append -Encoding utf8 $bitwiseOut
python zint\scripts\plot_compare.py --csv $mpnOrCsv  --out $mpnOrPng  --title "mpn_or_n (u64 limbs)"  | Out-File -Append -Encoding utf8 $bitwiseOut
python zint\scripts\plot_compare.py --csv $mpnXorCsv --out $mpnXorPng --title "mpn_xor_n (u64 limbs)" | Out-File -Append -Encoding utf8 $bitwiseOut
python zint\scripts\plot_compare.py --csv $mpnNotCsv --out $mpnNotPng --title "mpn_not_n (u64 limbs)" | Out-File -Append -Encoding utf8 $bitwiseOut
python zint\scripts\plot_bigint_bitwise.py --csv $bigintCsv --out $bigintPng | Out-File -Append -Encoding utf8 $bitwiseOut

# Build + run basecase benchmarks.
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 zint\bench\bench_basecase.cpp zint\\asm\\addmul_1_adx.obj /Fe:zint\bench\bench_basecase.exe | Out-File -Encoding utf8 $baseOut
cmd /c "zint\\bench\\bench_basecase.exe --csv-mul zint\\evaluations\\$Version\\basecase_mul.csv --csv-sqr zint\\evaluations\\$Version\\basecase_sqr.csv" | Out-File -Append -Encoding utf8 $baseOut
python zint\scripts\plot_compare.py --csv $baseMulCsv --out $baseMulPng --title "mpn_mul_basecase: n x n" | Out-File -Append -Encoding utf8 $baseOut
python zint\scripts\plot_compare.py --csv $baseSqrCsv --out $baseSqrPng --title "mpn_sqr_basecase: n^2" | Out-File -Append -Encoding utf8 $baseOut

# Build + run addmul_1 benchmarks (scalar vs mpn_addmul_1_fast).
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 /DZINT_USE_ADX_ASM=1 zint\bench\bench_addmul_1.cpp zint\\asm\\addmul_1_adx.obj /Fe:zint\bench\bench_addmul_1.exe | Out-File -Encoding utf8 $addmulOut
cmd /c "zint\\bench\\bench_addmul_1.exe --csv zint\\evaluations\\$Version\\addmul_1.csv" | Out-File -Append -Encoding utf8 $addmulOut
python zint\scripts\plot_compare.py --csv $addmulCsv --out $addmulPng --title "mpn_addmul_1 (default) vs scalar" | Out-File -Append -Encoding utf8 $addmulOut

# Create a report stub if missing.
$report = Join-Path $evalDir "report.md"
if (!(Test-Path $report)) {
@(
    "# Improvement report: $Version"
    ""
    "- zint commit: ``$zintHead``"
    "- parent commit: ``$parentHead``"
    ""
    "## Summary"
    ""
    "## Correctness"
    ""
    "## Performance"
    ""
) | Out-File -Encoding utf8 $report
}
