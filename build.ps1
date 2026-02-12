param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Src,

    [Parameter(Position=1)]
    [string]$Exe,

    [switch]$Asm,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

# Default output name: same as source with .exe extension
if (-not $Exe) {
    $Exe = [System.IO.Path]::ChangeExtension($Src, ".exe")
}

# ── MSVC environment ────────────────────────────────────────────────
$vsBase = 'C:\Program Files\Microsoft Visual Studio'
$msvcBin = $null
foreach ($ed in @('Enterprise','Professional','Community')) {
    $candidates = Get-ChildItem "$vsBase\*\$ed\VC\Tools\MSVC" -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending | Select-Object -First 1
    if ($candidates) {
        $ver = (Get-ChildItem $candidates.FullName | Sort-Object Name -Descending | Select-Object -First 1).Name
        $msvcBin = Join-Path $candidates.FullName "$ver\bin\Hostx64\x64"
        $msvcInc = Join-Path $candidates.FullName "$ver\include"
        $msvcLib = Join-Path $candidates.FullName "$ver\lib\x64"
        break
    }
}

if (-not $msvcBin) {
    Write-Error "Could not find MSVC installation"
    exit 1
}

# Find Windows SDK
$sdkBase = 'C:\Program Files (x86)\Windows Kits\10'
$sdkVer = (Get-ChildItem "$sdkBase\Include" -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name

$env:PATH = "$msvcBin;" + $env:PATH
$env:INCLUDE = "$msvcInc;$sdkBase\Include\$sdkVer\ucrt;$sdkBase\Include\$sdkVer\um;$sdkBase\Include\$sdkVer\shared"
$env:LIB = "$msvcLib;$sdkBase\Lib\$sdkVer\ucrt\x64;$sdkBase\Lib\$sdkVer\um\x64"

# ── Resolve paths ───────────────────────────────────────────────────
$repoRoot = $PSScriptRoot
$parentRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path

# ── Assemble ASM kernels if requested ───────────────────────────────
$asmObjs = @()
$asmDefine = @()

if ($Asm) {
    $asmDir = Join-Path $repoRoot "asm"
    foreach ($asmFile in @("addmul_1_adx.asm", "submul_1_adx.asm", "mul_basecase_adx.asm")) {
        $asmPath = Join-Path $asmDir $asmFile
        $objPath = Join-Path $asmDir ([System.IO.Path]::ChangeExtension($asmFile, ".obj"))
        if (-not (Test-Path $asmPath)) { continue }
        Write-Host "  ASM: $asmFile"
        ml64 /nologo /c /Fo $objPath $asmPath
        if ($LASTEXITCODE -ne 0) { Write-Error "ml64 failed on $asmFile"; exit 1 }
        $asmObjs += $objPath
    }
    $asmDefine = @("/DZINT_USE_ADX_ASM=1")
}

# ── Compile ─────────────────────────────────────────────────────────
$clArgs = @(
    "/nologo", "/I$parentRoot", "/std:c++17", "/O2", "/EHsc", "/arch:AVX2"
) + $asmDefine + @($Src) + $asmObjs + @("/Fe:$Exe")

Write-Host "  CL: cl $($clArgs -join ' ')"
cl @clArgs
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation failed"; exit 1 }

Write-Host "  OK: $Exe"

# ── Run if requested ────────────────────────────────────────────────
if ($Run) {
    Write-Host "--- Running $Exe ---"
    & ".\$Exe"
    Write-Host "--- Exit code: $LASTEXITCODE ---"
}
