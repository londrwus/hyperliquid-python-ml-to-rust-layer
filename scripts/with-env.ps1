# Windows counterpart of scripts/with-env.sh — runs a command with the repo's
# `.env` loaded into the process environment. Secrets stay off argv and history.
#
#   .\scripts\with-env.ps1 cargo run -p axon-provider-hyperliquid --example wallet_info
#   .\scripts\with-env.ps1 cargo test -p axon-provider-hyperliquid -- --ignored
#
# `.env` is gitignored; copy `.env.example` to start.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$envFile = Join-Path $root '.env'

if (-not (Test-Path $envFile)) {
    Write-Error "with-env.ps1: no .env at $root (copy .env.example and fill it in)"
}

foreach ($line in Get-Content $envFile) {
    $trimmed = $line.Trim()
    if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
    $split = $trimmed.IndexOf('=')
    if ($split -lt 1) { continue }
    $name = $trimmed.Substring(0, $split).Trim()
    $value = $trimmed.Substring($split + 1).Trim().Trim('"')
    Set-Item -Path "env:$name" -Value $value
}

Push-Location $root
try {
    & $Command @CommandArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
