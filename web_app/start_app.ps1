# Lung Cancer Detection Web Application - Quick Start Script
# PowerShell script to start the Flask application

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Lung Cancer Detection Web Application" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".\venv_frontend\Scripts\Activate.ps1") {
    Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv_frontend\Scripts\Activate.ps1"
} elseif (Test-Path ".\venv_lung_cancer\Scripts\Activate.ps1") {
    Write-Host "[1/3] Activating existing virtual environment..." -ForegroundColor Yellow
    & ".\venv_lung_cancer\Scripts\Activate.ps1"
} else {
    Write-Host "[WARNING] No virtual environment found!" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv_frontend
    & ".\venv_frontend\Scripts\Activate.ps1"
    
    Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements_frontend.txt
}

Write-Host ""
Write-Host "[2/3] Checking dependencies..." -ForegroundColor Yellow
pip show flask > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements_frontend.txt
} else {
    Write-Host "Dependencies OK!" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/3] Starting Flask server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "  Server will start at: http://localhost:5000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""

# Start the Flask app
python app.py
