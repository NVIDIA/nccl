# NCCL Windows Build and Test Script
# Run from the repository root directory

param(
    [switch]$Clean,
    [switch]$BuildOnly,
    [switch]$TestOnly,
    [string]$BuildType = "Release",
    [string]$Generator = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$BuildDir = Join-Path $RepoRoot "build_win"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NCCL Windows Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Configure
if (-not $TestOnly) {
    Write-Host "Configuring CMake..." -ForegroundColor Green
    Push-Location $BuildDir
    try {
        # Find CUDA
        $CudaPath = $env:CUDA_PATH
        if (-not $CudaPath) {
            Write-Host "WARNING: CUDA_PATH not set, CMake will try to find CUDA automatically" -ForegroundColor Yellow
        }
        else {
            Write-Host "Using CUDA from: $CudaPath" -ForegroundColor Gray
        }

        cmake -G $Generator `
            -A x64 `
            -DCMAKE_BUILD_TYPE=$BuildType `
            -DBUILD_TESTS=ON `
            ..

        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
    }
    finally {
        Pop-Location
    }
}

# Build
if (-not $TestOnly) {
    Write-Host "Building NCCL..." -ForegroundColor Green
    Push-Location $BuildDir
    try {
        cmake --build . --config $BuildType --parallel

        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        Write-Host "Build completed successfully!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# Test
if (-not $BuildOnly) {
    Write-Host "Running tests..." -ForegroundColor Green
    Push-Location $BuildDir
    try {
        ctest -C $BuildType --output-on-failure

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Some tests failed!" -ForegroundColor Red
            exit 1
        }
        Write-Host "All tests passed!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build and test completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
