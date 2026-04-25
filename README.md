# CAD-C Lung Cancer Detection

Multi-model lung nodule malignancy screening web application built with Flask and PyTorch.
The app runs inference with three 3D CNN architectures and exposes a clean web UI plus JSON APIs.

## Overview

This repository is the deployment-ready app for CAD-C. It includes:

- A Flask backend for model loading and inference
- A static frontend (landing, analyze, about pages)
- Three trained 3D models for ensemble-style comparison
- Docker support for local and cloud deployment

Important: this project is for research and educational use. It is not a medical device and must not be used as a standalone clinical diagnostic system.

## Features

- Multi-model prediction using:
	- Improved 3D CNN (Residual + SE)
	- EfficientNet3D-B2
	- DenseNet3D + Attention
- Unified upload-and-predict workflow from the web UI
- Lazy model loading to reduce startup time on hosted platforms
- Health and model metadata API endpoints
- Dockerized deployment on port `10000` by default

## Model Summary

Metrics below are loaded from each model's bundled JSON metadata.

| Model | Parameters | Test Accuracy | Test F1 |
|---|---:|---:|---:|
| Improved 3D CNN (Residual + SE) | 33,378,882 | 83.33% | 0.6667 |
| EfficientNet3D-B2 | 9,215,554 | 93.08% | 0.7326 |
| DenseNet3D + Attention | 672,770 | 95.73% | 0.7775 |

## Repository Structure

```text
CAD-C/
|- app.py                              # Flask backend and API routes
|- static/                             # Frontend pages, CSS, JS
|- models_3d_cnn/                      # Improved 3D CNN model + metadata
|- efficientnet3d_b2.pth               # EfficientNet3D-B2 weights
|- efficientnet_model_info.json
|- densenet3d_attention.pth            # DenseNet3D-Attention weights
|- densenet_model_info.json
|- densenet3d_architecture.py          # Model architecture code
|- efficientnet3d_b2_architecture.py
|- Dockerfile
|- requirements_frontend.txt
|- hf-space-cadc/                      # Hugging Face Space variant
`- CAD-C/                              # Extended research workspace
```

## Quick Start (Local)

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements_frontend.txt
```

### 3) Run the app

```bash
python app.py
```

By default, the app starts on:

- `http://localhost:5000` (local Python run)

You can override port:

```powershell
$env:PORT=10000; python app.py
```

## Quick Start (Docker)

Build image:

```bash
docker build -t cad-c .
```

Run container:

```bash
docker run --rm -p 10000:10000 -e PORT=10000 cad-c
```

Then open:

- `http://localhost:10000`

## API Endpoints

- `GET /api/health`
	- Service status and loaded model count
- `GET /api/models`
	- Available model metadata and performance summary
- `POST /api/predict`
	- Multipart form upload with field name `image`

Example prediction request (PowerShell):

```powershell
curl.exe -X POST "http://localhost:5000/api/predict" `
	-F "image=@C:\path\to\image.png"
```

## Input and Prediction Notes

- Uploaded image is converted to grayscale
- Input is resized to `64 x 64` and expanded to a synthetic `64 x 64 x 64` volume
- Prediction threshold for cancer class is currently set to `0.32`
- Output may include a "Suspicious" class for borderline probabilities

## Git LFS Requirement

Model files (`*.pth`) are tracked with Git LFS.

Before cloning or pulling model weights, ensure Git LFS is installed:

```bash
git lfs install
```

## Development Notes

- CORS is enabled for `/api/*`
- Max upload size is `16 MB`
- Server uses CPU by default
- In production container mode, `gunicorn` is used (`Dockerfile`)

## Related Directories

- `hf-space-cadc/`: Minimal Hugging Face Space deployment package
- `CAD-C/`: Larger research workspace (not required to run this root app)

## License

MIT License. See `CAD-C/LICENSE` for details.
