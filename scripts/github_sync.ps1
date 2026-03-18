[CmdletBinding()]
param(
    [string]$Message = "",
    [string]$RemoteName = "origin",
    [string]$Branch = "",
    [string]$TokenEnv = "GITHUB_TOKEN",
    [switch]$AllowEmpty,
    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($GitArgs -join ' ')"
    }
}

function Get-Token {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = [Environment]::GetEnvironmentVariable($Name, "User")
    }
    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = [Environment]::GetEnvironmentVariable($Name, "Machine")
    }
    return $value
}

function Test-StagedLargeFiles {
    param([int64]$MaxSizeBytes = 100MB)
    $staged = & git diff --cached --name-only --diff-filter=ACMR
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect staged files."
    }

    $oversized = @()
    foreach ($path in $staged) {
        if ([string]::IsNullOrWhiteSpace($path)) {
            continue
        }

        # Read size from Git index blob to avoid filesystem path encoding issues.
        $sizeText = ""
        try {
            $sizeText = (& git cat-file -s ":$path" 2>$null).Trim()
        }
        catch {
            continue
        }
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($sizeText)) {
            continue
        }

        [int64]$size = 0
        if (-not [int64]::TryParse($sizeText, [ref]$size)) {
            continue
        }

        if ($size -gt $MaxSizeBytes) {
            $oversized += [pscustomobject]@{
                Path   = $path
                SizeMB = [math]::Round($size / 1MB, 2)
            }
        }
    }

    if ($oversized.Count -gt 0) {
        $rows = $oversized | ForEach-Object { "$($_.Path) ($($_.SizeMB) MB)" }
        $msg = @(
            "Push blocked: staged files larger than 100MB detected."
            "Please move these files to Git LFS or ignore them first:"
            ($rows -join [Environment]::NewLine)
        ) -join [Environment]::NewLine
        throw $msg
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed or not in PATH."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$insideRepo = (& git rev-parse --is-inside-work-tree 2>$null)
if ($LASTEXITCODE -ne 0 -or $insideRepo.Trim() -ne "true") {
    throw "Current directory is not a git repository. Run github_bootstrap.ps1 first."
}

$currentBranch = (& git branch --show-current).Trim()
if ([string]::IsNullOrWhiteSpace($Branch)) {
    $Branch = $currentBranch
}
if ([string]::IsNullOrWhiteSpace($Branch)) {
    throw "Cannot detect branch. Please provide -Branch."
}

$remoteList = @(& git remote)
if (-not ($remoteList -contains $RemoteName)) {
    throw "Remote '$RemoteName' not found. Run github_bootstrap.ps1 first or configure the remote."
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "chore: sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Invoke-Git -GitArgs @("add", "-A")
Test-StagedLargeFiles

$staged = @(& git diff --cached --name-only)
if ($staged.Count -eq 0 -and -not $AllowEmpty) {
    Write-Host "No staged changes. Nothing to commit."
    if (-not $NoPush) {
        Write-Host "Skipping push because there is nothing new."
    }
    exit 0
}

if ($AllowEmpty -and $staged.Count -eq 0) {
    Invoke-Git -GitArgs @("commit", "--allow-empty", "-m", $Message)
}
else {
    Invoke-Git -GitArgs @("commit", "-m", $Message)
}

if (-not $NoPush) {
    $token = Get-Token -Name $TokenEnv
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "Missing token: set environment variable '$TokenEnv' before push."
    }
    $basic = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("x-access-token:$token"))
    & git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" push -u $RemoteName $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }
    Write-Host "Sync complete on branch '$Branch'."
}
else {
    Write-Host "Commit complete. Skipped push due to -NoPush."
}
