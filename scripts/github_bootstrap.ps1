[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoName,

    [string]$Owner = "",

    [ValidateSet("private", "public")]
    [string]$Visibility = "private",

    [string]$Description = "",
    [string]$RemoteName = "origin",
    [string]$DefaultBranch = "main",
    [string]$TokenEnv = "GITHUB_TOKEN",
    [string]$InitialCommitMessage = "chore: initial project import",
    [switch]$SkipInitialCommit,
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
    Write-Host "Initializing git repository..."
    Invoke-Git -GitArgs @("init", "-b", $DefaultBranch)
}

$currentBranch = (& git branch --show-current).Trim()
if ([string]::IsNullOrWhiteSpace($currentBranch) -or $currentBranch -ne $DefaultBranch) {
    Invoke-Git -GitArgs @("checkout", "-B", $DefaultBranch)
}

$hasCommit = $false
try {
    & git rev-parse --verify HEAD *> $null
    $hasCommit = ($LASTEXITCODE -eq 0)
}
catch {
    $hasCommit = $false
}

if (-not $SkipInitialCommit) {
    Invoke-Git -GitArgs @("add", "-A")
    Test-StagedLargeFiles

    $hasStagedChanges = (& git diff --cached --name-only)
    if ($hasStagedChanges.Count -gt 0) {
        Invoke-Git -GitArgs @("commit", "-m", $InitialCommitMessage)
        $hasCommit = $true
    }
    elseif (-not $hasCommit) {
        throw "No commits found and no files staged. Add files or remove -SkipInitialCommit."
    }
}
elseif (-not $hasCommit) {
    throw "No commit exists. Remove -SkipInitialCommit or create an initial commit manually."
}

$token = Get-Token -Name $TokenEnv
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Missing token: set environment variable '$TokenEnv' before running this script."
}

$headers = @{
    Authorization         = "Bearer $token"
    Accept                = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent"          = "time-llm-github-bootstrap"
}

$me = Invoke-RestMethod -Method Get -Uri "https://api.github.com/user" -Headers $headers
$authLogin = $me.login
if ([string]::IsNullOrWhiteSpace($Owner)) {
    $Owner = $authLogin
}

$repoApi = "https://api.github.com/repos/$Owner/$RepoName"
$repoExists = $false
try {
    $null = Invoke-RestMethod -Method Get -Uri $repoApi -Headers $headers
    $repoExists = $true
}
catch {
    $statusCode = 0
    if ($_.Exception.Response) {
        $statusCode = [int]$_.Exception.Response.StatusCode
    }
    if ($statusCode -ne 404) {
        throw
    }
}

if (-not $repoExists) {
    $createUri = if ($Owner -ieq $authLogin) {
        "https://api.github.com/user/repos"
    }
    else {
        "https://api.github.com/orgs/$Owner/repos"
    }

    $body = @{
        name      = $RepoName
        private   = ($Visibility -eq "private")
        auto_init = $false
    }
    if (-not [string]::IsNullOrWhiteSpace($Description)) {
        $body.description = $Description
    }

    $json = $body | ConvertTo-Json -Depth 5
    $null = Invoke-RestMethod -Method Post -Uri $createUri -Headers $headers -Body $json -ContentType "application/json"
    Write-Host "Created repository: $Owner/$RepoName"
}
else {
    Write-Host "Repository already exists: $Owner/$RepoName"
}

$remoteUrl = "https://github.com/$Owner/$RepoName.git"
$existingRemotes = @(& git remote)
if ($existingRemotes -contains $RemoteName) {
    Invoke-Git -GitArgs @("remote", "set-url", $RemoteName, $remoteUrl)
}
else {
    Invoke-Git -GitArgs @("remote", "add", $RemoteName, $remoteUrl)
}

if (-not $NoPush) {
    $basic = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("x-access-token:$token"))
    & git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $basic" push -u $RemoteName $DefaultBranch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }
    Write-Host "Push complete: $remoteUrl ($DefaultBranch)"
}
else {
    Write-Host "Skipped push due to -NoPush."
    Write-Host "Remote configured: $RemoteName -> $remoteUrl"
}
