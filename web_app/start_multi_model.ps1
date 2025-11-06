# Quick Start Guide for Multi-Model Lung Cancer Detection Web App

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  Lung Cancer Detection System - Multi-Model Ensemble" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: Please run this script from the web_app directory!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run these commands:" -ForegroundColor Cyan
    Write-Host "  cd web_app" -ForegroundColor White
    Write-Host "  .\start_multi_model.ps1" -ForegroundColor White
    exit 1
}

Write-Host "Checking model files..." -ForegroundColor Cyan

# Check ResNet model
if (Test-Path "models_3d_cnn\best_improved_3d_cnn_model.pth") {
    Write-Host "  [OK] ResNet3D model found" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] ResNet3D model" -ForegroundColor Red
}

# Check DenseNet model
if (Test-Path "densenet3d_attention.pth") {
    Write-Host "  [OK] DenseNet3D model found" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] DenseNet3D model" -ForegroundColor Red
    Write-Host "  Attempting to copy from models directory..." -ForegroundColor Yellow
    if (Test-Path "..\models\densenet3d_attention\best_densenet3d_attention.pth") {
        Copy-Item "..\models\densenet3d_attention\best_densenet3d_attention.pth" "densenet3d_attention.pth"
        Write-Host "  [OK] DenseNet3D model copied successfully" -ForegroundColor Green
    }
}

# Check architecture files
if (Test-Path "densenet3d_architecture.py") {
    Write-Host "  [OK] DenseNet architecture found" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] DenseNet architecture" -ForegroundColor Red
}

# Check model info
if (Test-Path "densenet_model_info.json") {
    Write-Host "  [OK] DenseNet model info found" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] DenseNet model info missing" -ForegroundColor Yellow
    if (Test-Path "..\models\densenet3d_attention\test_results.json") {
        Copy-Item "..\models\densenet3d_attention\test_results.json" "densenet_model_info.json"
        Write-Host "  [OK] Model info copied successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Cyan
Write-Host "  Loading ResNet3D with SE Blocks (83.3% accuracy)" -ForegroundColor White
Write-Host "  Loading DenseNet3D with Attention (95.73% accuracy)" -ForegroundColor White
Write-Host ""
Write-Host "Once started, open your browser to:" -ForegroundColor Yellow
Write-Host "  http://localhost:5000" -ForegroundColor Green -NoNewline
Write-Host " or " -ForegroundColor White -NoNewline
Write-Host "http://127.0.0.1:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Start the app
python app.py
