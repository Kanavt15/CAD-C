# Repository Structure Guide

This document explains the organization and purpose of each directory and file in the CAD-C repository.

## 📂 Directory Structure

### `/assets/`
**Purpose**: Static resources and sample files for testing and demonstration.

- **`sample_images/`**: Test images for quick inference testing
  - Contains various lung CT scan images (healthy and unhealthy samples)
  - Used for validating model predictions
  - Original test images and additional samples for demonstrations

**Usage**: 
```bash
python scripts/test_external_image.py --image assets/sample_images/healthy.jpg
```

---

### `/data/`
**Purpose**: All dataset files, raw data, and processed outputs.

#### `/data/raw/`
Raw LUNA16 dataset and configuration files.

- **`subset0/` to `subset9/`**: LUNA16 CT scan subsets
  - Contains `.mhd` (metadata) and `.raw` (image data) files
  - Each subset contains ~88 CT scans
  - **Note**: Large files, excluded from git (see `.gitignore`)

- **`annotations.csv`**: Ground truth nodule annotations
  - Columns: seriesuid, coordX, coordY, coordZ, diameter_mm
  - True positive nodule locations and sizes

- **`candidates_V2.csv`**: Candidate nodule locations
  - Includes both true positives and false positives
  - Used for training the detection models

- **`*.json`**: Configuration files
  - `enhanced_ensemble_config.json`: Ensemble model configuration
  - `optimal_thresholds.json`: Optimized classification thresholds

#### `/data/processed/`
Generated and cached data from processing pipelines.

- **`patch_cache_enhanced/`**: Pre-extracted image patches
  - Cached 64x64 or 224x224 patches from CT scans
  - Speeds up training by 10-100x
  - Organized by subset and candidate ID

- **`inference_results/`**: Output images from inference
  - Visualization of predictions on test images
  - Saved with filename format: `inference_<original_name>.png`

---

### `/docs/`
**Purpose**: All project documentation and guides.

- **`QUICKSTART.md`**: Quick start guide for new users
- **`MODEL_COMPARISON.md`**: Detailed comparison of model architectures and performance
- **`FRONTEND_README.md`**: Web application setup and usage guide
- **`CONTRIBUTING.md`**: Guidelines for contributing to the project
- **`THRESHOLD_OPTIMIZATION_SUMMARY.md`**: Explanation of threshold optimization techniques
- **`ENHANCEMENT_SUMMARY.md`**: Summary of model enhancements and improvements
- **`GIT_LFS_GUIDE.md`**: Guide for using Git Large File Storage
- **`VGG_VIT_CLEANUP_SUMMARY.md`**: Documentation of removed experimental models
- **`REPOSITORY_SUMMARY.md`**: High-level overview of the repository

**Note**: Keep README.md in root for GitHub display.

---

### `/models/`
**Purpose**: Trained model checkpoints and related artifacts.

#### `/models/models_densenet/`
DenseNet-169 model files.

- **`best_densenet_model.pth`**: Main trained model checkpoint
- **`densenet169_luna16_real_best.pth`**: Fine-tuned version
- **`training_history.png`**: Training/validation curves
- **`*.json`**: Training configuration and results

#### `/models/models_efficientnet/`
EfficientNet-B0 model files.

- **`best_efficientnet_model.pth`**: Trained model checkpoint
- **`training_history.csv`**: Epoch-by-epoch training metrics
- **`confusion_matrix.png`**: Test set confusion matrix
- **`roc_curve.png`**: ROC curve visualization
- **`precision_recall_curve.png`**: PR curve
- **`test_results.json`**: Comprehensive test metrics

#### `/models/models_resnet101/`
ResNet-101 model files (similar structure to EfficientNet).

#### `/models/models_cnn/`
Basic CNN baseline models.

#### `/models/*.pkl`
Supporting model artifacts:
- **`xgboost_smote_model.pkl`**: XGBoost ensemble model
- **`feature_scaler.pkl`**: Feature normalization scaler
- **`feature_names.pkl`**: Feature column names

---

### `/notebooks/`
**Purpose**: Jupyter notebooks for training, experimentation, and analysis.

- **`lung_cancer_densenet.ipynb`**: DenseNet-169 training notebook
  - Data loading and preprocessing
  - Model architecture definition
  - Training loop with focal loss
  - Evaluation and visualization

- **`lung_cancer_efficientnet.ipynb`**: EfficientNet-B0 training
- **`lung_cancer_resnet101.ipynb`**: ResNet-101 training
- **`lung_cancer_detection_cnn.ipynb`**: Basic CNN baseline

**Usage**:
```bash
jupyter notebook notebooks/lung_cancer_densenet.ipynb
```

---

### `/scripts/`
**Purpose**: Python scripts for inference, deployment, and utilities.

#### Core Scripts

- **`inference_ensemble.py`**: 
  - Multi-model ensemble inference
  - Combines predictions from all models
  - Usage: `python scripts/inference_ensemble.py --image path/to/image.jpg`

- **`enhanced_ensemble_deploy.py`**:
  - Production-ready deployment script
  - Advanced preprocessing and postprocessing
  - Configurable via JSON config files

- **`test_external_image.py`**:
  - Test models on external (non-LUNA16) images
  - Handles various image formats
  - Usage: `python scripts/test_external_image.py --image path/to/image.jpg`

#### Optimization Scripts

- **`threshold_optimization.py`**:
  - Find optimal classification thresholds
  - Maximizes F1 score or other metrics
  - Saves results to `optimal_thresholds.json`

- **`simple_threshold_optimization.py`**:
  - Lightweight version for quick optimization

#### Utility Scripts

- **`test_dataset_config.py`**:
  - Validate dataset configuration
  - Check file paths and data integrity

- **`create_flowchart.py`**:
  - Generate project architecture flowcharts
  - Visualize data and model pipelines

---

### `/web_app/`
**Purpose**: Flask-based web application for interactive predictions.

#### Main Files

- **`app.py`**: Flask backend server
  - REST API endpoints for predictions
  - Model loading and management
  - Image preprocessing pipeline

- **`requirements_frontend.txt`**: Web app specific dependencies
  - Flask, Flask-CORS, scipy, etc.

- **`start_app.ps1`**: PowerShell script to start the web app
  ```powershell
  .\web_app\start_app.ps1
  ```

#### `/web_app/static/`
Frontend files served to the browser.

- **`index.html`**: Main web page
  - Upload interface
  - Model selection checkboxes
  - Results display section
  - 3D lung visualization (Sketchfab embed)

- **`css/style.css`**: Stylesheet
  - Dark theme for 3D section
  - Responsive design
  - Color-coded prediction results

- **`js/script.js`**: Client-side JavaScript
  - Image upload handling
  - API communication
  - Dynamic result rendering

#### `/web_app/uploads/`
Temporary storage for uploaded images (excluded from git).

**Access**: Navigate to `http://localhost:5000` after starting the app.

---

### `/tests/`
**Purpose**: Unit tests and integration tests (to be implemented).

**Planned**:
- Model inference tests
- Preprocessing pipeline tests
- API endpoint tests
- Data loading tests

---

## 📄 Root Files

### Configuration Files

- **`requirements.txt`**: Main Python dependencies for the entire project
  ```bash
  pip install -r requirements.txt
  ```

- **`.gitignore`**: Files and directories excluded from version control
  - Large dataset files
  - Cache directories
  - Virtual environments
  - Temporary files

- **`.gitattributes`**: Git LFS configuration for large files

### Documentation

- **`README.md`**: Main project README (you are here!)
- **`LICENSE`**: MIT License

---

## 🔄 Typical Workflow

### 1. Setup
```bash
git clone https://github.com/Kanavt15/CAD-C.git
cd CAD-C
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Training
```bash
jupyter notebook notebooks/lung_cancer_densenet.ipynb
# Train model, save checkpoint to models/models_densenet/
```

### 3. Testing
```bash
python scripts/test_external_image.py --image assets/sample_images/healthy.jpg
```

### 4. Web Interface
```bash
cd web_app
pip install -r requirements_frontend.txt
python app.py
# Open browser to http://localhost:5000
```

### 5. Deployment
```bash
python scripts/enhanced_ensemble_deploy.py --config data/raw/enhanced_ensemble_config.json
```

---

## 🧹 Maintenance

### Cleaning Cache
```bash
# Remove processed data cache
Remove-Item -Recurse data/processed/patch_cache_enhanced/*

# Remove inference results
Remove-Item -Recurse data/processed/inference_results/*
```

### Updating Models
1. Train new model in `notebooks/`
2. Save checkpoint to appropriate `models/` subdirectory
3. Update config files in `data/raw/` if needed
4. Test with `scripts/test_external_image.py`

### Adding Documentation
- Place new `.md` files in `docs/`
- Update main `README.md` if needed
- Link from appropriate sections

---

## 📊 File Size Guidelines

- **Models (`.pth`)**: 50-500 MB each
- **Dataset (subsets)**: ~10 GB per subset
- **Patch cache**: 1-5 GB depending on configuration
- **Keep individual files < 100 MB** for GitHub compatibility
- Use Git LFS for larger files if needed

---

## 🔗 Quick Links

- [Main README](../README.md)
- [Quick Start Guide](QUICKSTART.md)
- [Model Comparison](MODEL_COMPARISON.md)
- [Web App Guide](FRONTEND_README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Last Updated**: October 2025  
**Maintained by**: [Kanavt15](https://github.com/Kanavt15)
