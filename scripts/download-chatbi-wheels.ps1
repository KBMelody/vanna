$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$WheelDir = Join-Path $Root "wheels"
$ReqFile = Join-Path $Root "requirements\chatbi-nl2sql-offline.txt"

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

python -m pip download `
  --dest $WheelDir `
  --requirement $ReqFile `
  --only-binary=:all: `
  --platform manylinux2014_x86_64 `
  --implementation cp `
  --python-version 311 `
  --abi cp311 `
  --index-url https://pypi.org/simple

python -m pip download `
  --dest $WheelDir `
  --only-binary=:all: `
  --platform manylinux2014_x86_64 `
  --implementation cp `
  --python-version 311 `
  --abi cp311 `
  --index-url https://pypi.org/simple `
  "uvloop>=0.15.1"

Write-Host "Wheel sync completed: $WheelDir"
