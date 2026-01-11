# BenderBox Prerequisites Installation Script (Windows PowerShell)
# This script installs required and optional dependencies for BenderBox

param(
    [switch]$All,
    [switch]$Required,
    [switch]$LlamaCli,
    [switch]$Force,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$TOOLS_DIR = "$env:USERPROFILE\.benderbox\tools"
$LLAMA_RELEASE_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  BenderBox Prerequisites Installer (Windows)" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success($message) {
    Write-Host "[OK] $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "[!] $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "[X] $message" -ForegroundColor Red
}

function Write-Info($message) {
    Write-Host "[*] $message" -ForegroundColor Cyan
}

function Show-Help {
    Write-Header
    Write-Host "Usage: install-prerequisites.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -All         Install all prerequisites"
    Write-Host "  -Required    Install only required prerequisites"
    Write-Host "  -LlamaCli    Install llama-cli (for GGUF analysis)"
    Write-Host "  -Force       Force reinstall even if already installed"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\install-prerequisites.ps1 -All"
    Write-Host "  .\install-prerequisites.ps1 -LlamaCli"
    Write-Host ""
}

function Test-Command($command) {
    try {
        $null = Get-Command $command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Get-InstalledVersion($command, $pattern) {
    try {
        $output = & $command 2>&1
        if ($output -match $pattern) {
            return $matches[1]
        }
    } catch {
        return $null
    }
    return $null
}

function Install-LlamaCli {
    param([switch]$Force)

    Write-Info "Checking llama-cli..."

    $llamaPath = "$TOOLS_DIR\llama-cli.exe"

    # Check if already installed
    if ((Test-Path $llamaPath) -and -not $Force) {
        Write-Success "llama-cli is already installed at $llamaPath"
        return $true
    }

    # Also check PATH
    if ((Test-Command "llama-cli") -and -not $Force) {
        $existingPath = (Get-Command "llama-cli").Source
        Write-Success "llama-cli found at $existingPath"
        return $true
    }

    Write-Info "Installing llama-cli..."

    # Create tools directory
    if (-not (Test-Path $TOOLS_DIR)) {
        New-Item -ItemType Directory -Path $TOOLS_DIR -Force | Out-Null
    }

    try {
        # Get latest release info
        Write-Info "Fetching latest release information..."
        $releaseInfo = Invoke-RestMethod -Uri $LLAMA_RELEASE_URL -Headers @{Accept = "application/vnd.github.v3+json"}
        $tagName = $releaseInfo.tag_name -replace "^v", ""

        Write-Info "Latest release: $tagName"

        # Find Windows binary asset
        $asset = $releaseInfo.assets | Where-Object { $_.name -like "*win*avx2*x64*.zip" } | Select-Object -First 1

        if (-not $asset) {
            # Try alternate naming
            $asset = $releaseInfo.assets | Where-Object { $_.name -like "*windows*x64*.zip" } | Select-Object -First 1
        }

        if (-not $asset) {
            Write-Error "Could not find Windows binary in release assets"
            Write-Info "Available assets:"
            $releaseInfo.assets | ForEach-Object { Write-Host "  - $($_.name)" }
            return $false
        }

        $downloadUrl = $asset.browser_download_url
        $fileName = $asset.name

        Write-Info "Downloading $fileName..."

        $tempDir = Join-Path $env:TEMP "benderbox-install"
        $zipPath = Join-Path $tempDir $fileName
        $extractPath = Join-Path $tempDir "extracted"

        # Clean up temp directory
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force
        }
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
        New-Item -ItemType Directory -Path $extractPath -Force | Out-Null

        # Download
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($downloadUrl, $zipPath)

        Write-Info "Extracting archive..."
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

        # Find llama-cli.exe
        $llamaExe = Get-ChildItem -Path $extractPath -Recurse -Filter "llama-cli.exe" | Select-Object -First 1

        if (-not $llamaExe) {
            # Try main.exe (older naming)
            $llamaExe = Get-ChildItem -Path $extractPath -Recurse -Filter "main.exe" | Select-Object -First 1
        }

        if (-not $llamaExe) {
            Write-Error "Could not find llama-cli.exe in archive"
            Write-Info "Archive contents:"
            Get-ChildItem -Path $extractPath -Recurse | ForEach-Object { Write-Host "  - $($_.FullName)" }
            return $false
        }

        # Copy to tools directory
        Copy-Item -Path $llamaExe.FullName -Destination $llamaPath -Force

        # Clean up
        Remove-Item -Path $tempDir -Recurse -Force

        Write-Success "llama-cli installed to $llamaPath"

        # Add to PATH suggestion
        Write-Info "To add to PATH, run:"
        Write-Host "  `$env:PATH += `";$TOOLS_DIR`"" -ForegroundColor Yellow
        Write-Host ""
        Write-Info "Or add permanently via System Properties > Environment Variables"

        return $true

    } catch {
        Write-Error "Installation failed: $_"
        return $false
    }
}

function Check-Python {
    Write-Info "Checking Python..."

    if (Test-Command "python") {
        $version = Get-InstalledVersion "python --version" "Python\s+(\d+\.\d+\.\d+)"
        if ($version) {
            $major = [int]($version.Split('.')[0])
            $minor = [int]($version.Split('.')[1])

            if ($major -ge 3 -and $minor -ge 9) {
                Write-Success "Python $version is installed"
                return $true
            } else {
                Write-Warning "Python $version found, but 3.9+ is required"
                return $false
            }
        }
    }

    Write-Warning "Python not found. Please install Python 3.9+ from https://www.python.org/downloads/"
    return $false
}

function Check-Git {
    Write-Info "Checking Git..."

    if (Test-Command "git") {
        $version = Get-InstalledVersion "git --version" "git version\s+(\d+\.\d+\.\d+)"
        if ($version) {
            Write-Success "Git $version is installed"
            return $true
        }
    }

    Write-Warning "Git not found. Install via: winget install --id Git.Git -e --source winget"
    return $false
}

function Install-PythonDependencies {
    Write-Info "Installing Python dependencies..."

    try {
        & python -m pip install --upgrade pip
        & python -m pip install -e ".[all,dev]"
        Write-Success "Python dependencies installed"
        return $true
    } catch {
        Write-Error "Failed to install Python dependencies: $_"
        return $false
    }
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Header

$results = @{}

# Always check Python (required)
$results["python"] = Check-Python

# Check Git
$results["git"] = Check-Git

# Install llama-cli if requested
if ($All -or $LlamaCli) {
    $results["llama-cli"] = Install-LlamaCli -Force:$Force
}

# Install Python dependencies if Python is available
if ($All -or $Required) {
    if ($results["python"]) {
        $results["pip-deps"] = Install-PythonDependencies
    }
}

# Summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Installation Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$success = 0
$failed = 0

foreach ($item in $results.GetEnumerator()) {
    if ($item.Value) {
        Write-Success "$($item.Key)"
        $success++
    } else {
        Write-Error "$($item.Key)"
        $failed++
    }
}

Write-Host ""
if ($failed -eq 0) {
    Write-Success "All prerequisites installed successfully!"
} else {
    Write-Warning "$failed prerequisite(s) need attention"
}

Write-Host ""
Write-Info "BenderBox tools directory: $TOOLS_DIR"
Write-Host ""
