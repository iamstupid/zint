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
$mulCsv = Join-Path $evalDir "bench_mul.csv"
$divCsv = Join-Path $evalDir "bench_div.csv"
$mulPng = Join-Path $evalDir "bench_mul.png"
$divPng = Join-Path $evalDir "bench_div.png"

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

# Build + run tests (must not reference legacy bigint for correctness).
Set-Location $parentRoot
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 zint\tests\test_zint_correctness.cpp /Fe:zint\tests\test_zint_correctness.exe | Out-File -Encoding utf8 $testsOut
cmd /c "zint\\tests\\test_zint_correctness.exe" | Out-File -Append -Encoding utf8 $testsOut

# Build + run benchmark vs legacy bigint.
cl /nologo /I. /std:c++17 /O2 /EHsc /arch:AVX2 zint\bench\bench_zint_vs_bigint.cpp /Fe:zint\bench\bench_zint_vs_bigint.exe | Out-File -Encoding utf8 $benchOut
cmd /c "zint\\bench\\bench_zint_vs_bigint.exe --csv-mul zint\\evaluations\\$Version\\bench_mul.csv --csv-div zint\\evaluations\\$Version\\bench_div.csv" | Out-File -Append -Encoding utf8 $benchOut

# Plot graphs.
python zint\scripts\plot_bench.py --mul $mulCsv --div $divCsv --out-mul $mulPng --out-div $divPng | Out-File -Append -Encoding utf8 $benchOut

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
