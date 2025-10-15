# Lung Cancer Detection System 🫁# 🫁 LUNA16 Lung Cancer Detection System



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)A comprehensive deep learning system for automated lung cancer detection using the LUNA16 dataset. This project implements and compares state-of-the-art CNN architectures: **LUNA16-DenseNet**, **ResNet-101**, and **EfficientNet-B0** for binary classification of lung nodules.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

A comprehensive deep learning system for lung cancer detection using multiple state-of-the-art CNN architectures (DenseNet-169, EfficientNet-B0, ResNet-101) with ensemble predictions and an interactive web interface.[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



## 📋 Table of Contents---



- [Features](#features)## 📋 Table of Contents

- [Project Structure](#project-structure)

- [Installation](#installation)- [Overview](#-overview)

- [Quick Start](#quick-start)- [Features](#-features)

- [Models](#models)- [Dataset](#-dataset)

- [Web Application](#web-application)- [Model Architectures](#-model-architectures)

- [Documentation](#documentation)- [Installation](#-installation)

- [Contributing](#contributing)- [Project Structure](#-project-structure)

- [License](#license)- [Usage](#-usage)

  - [Training](#training)

## ✨ Features  - [Inference (LUNA16 Dataset)](#inference-luna16-dataset)

  - [Inference (External Images)](#inference-external-images)

- **Multi-Model Architecture**: Ensemble of DenseNet-169, EfficientNet-B0, and ResNet-101- [Results](#-results)

- **Interactive Web Interface**: User-friendly Flask-based web application with 3D visualization- [Technical Details](#-technical-details)

- **Medical Image Processing**: Specialized preprocessing for CT scans with HU normalization- [Contributing](#-contributing)

- **High Performance**: Optimized inference with PyTorch and model checkpointing- [License](#-license)

- **Comprehensive Training**: Jupyter notebooks for model training, fine-tuning, and evaluation- [Acknowledgments](#-acknowledgments)

- **Threshold Optimization**: Advanced threshold optimization for optimal prediction accuracy

- **Deployment Ready**: Production-ready inference scripts and configuration files---



## 📁 Project Structure## 🎯 Overview



```This project tackles the critical challenge of early lung cancer detection using deep learning. By leveraging transfer learning on three powerful CNN architectures, the system achieves high accuracy in distinguishing between cancerous and non-cancerous lung nodules from CT scans.

CAD-C/

├── assets/                          # Sample images and resources### Key Highlights:

│   └── sample_images/              # Test images for inference- 🏆 **94.44% accuracy** with ResNet-101

│       ├── healthy.jpg- ⚡ **Multi-model ensemble** for robust predictions

│       ├── unhealthy.png- 🚀 **Pre-extraction pipeline** for 10-100x faster training

│       └── test_images/- 🖼️ **External image support** for real-world deployment

│- 📊 **Comprehensive visualizations** and metrics

├── data/                            # Data files and datasets

│   ├── raw/                        # Raw LUNA16 dataset---

│   │   ├── subset0/ to subset9/   # LUNA16 CT scan subsets

│   │   ├── annotations.csv         # Ground truth annotations## ✨ Features

│   │   ├── candidates_V2.csv       # Candidate nodules

│   │   └── *.json                 # Configuration files### Core Capabilities

│   └── processed/                  # Processed data- ✅ **Multiple CNN Architectures**: LUNA16-DenseNet, ResNet-101, EfficientNet-B0

│       ├── patch_cache_enhanced/   # Cached image patches- ✅ **Transfer Learning**: Pre-trained on ImageNet, fine-tuned for medical imaging

│       └── inference_results/      # Inference output images- ✅ **Focal Loss**: Handles severe class imbalance (1:500 positive to negative ratio)

│- ✅ **Data Augmentation**: Random flips, rotations, and brightness adjustments

├── docs/                            # Documentation files- ✅ **Memory-Efficient Pre-extraction**: Caches patches in memory for fast training

│   ├── QUICKSTART.md               # Quick start guide- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping

│   ├── MODEL_COMPARISON.md         # Model performance comparison- ✅ **Multi-Model Ensemble**: Combines predictions from all three models

│   ├── FRONTEND_README.md          # Web app documentation- ✅ **External Image Support**: Works with DICOM, JPG, PNG, NPY files

│   ├── CONTRIBUTING.md             # Contribution guidelines- ✅ **Comprehensive Metrics**: AUC-ROC, Precision, Recall, F1, Confusion Matrix

│   └── *.md                       # Additional documentation

│### Visualization

├── models/                          # Trained model checkpoints- 📈 Training history plots (loss, accuracy, AUC, F1)

│   ├── models_densenet/            # DenseNet-169 models- 📊 ROC curves and Precision-Recall curves

│   ├── models_efficientnet/        # EfficientNet-B0 models- 🎨 Confusion matrices with heatmaps

│   ├── models_resnet101/           # ResNet-101 models- 🖼️ Multi-model prediction visualization

│   ├── models_cnn/                 # Basic CNN models

│   └── *.pkl                       # Feature extractors and scalers---

│

├── notebooks/                       # Jupyter notebooks## 📚 Dataset

│   ├── lung_cancer_densenet.ipynb          # DenseNet training

│   ├── lung_cancer_efficientnet.ipynb      # EfficientNet trainingThis project uses the **LUNA16 (Lung Nodule Analysis 2016)** dataset, a subset of the larger LIDC-IDRI dataset.

│   ├── lung_cancer_resnet101.ipynb         # ResNet-101 training

│   └── lung_cancer_detection_cnn.ipynb     # Basic CNN training### Dataset Statistics:

│- **Total Candidates**: 754,975

├── scripts/                         # Python scripts- **Positive Cases (Cancer)**: 1,557 (0.2%)

│   ├── inference_ensemble.py        # Ensemble inference script- **Negative Cases**: 753,418 (99.8%)

│   ├── enhanced_ensemble_deploy.py  # Enhanced deployment script- **CT Scans**: 888 patients

│   ├── threshold_optimization.py    # Threshold optimization- **Format**: MetaImage (.mhd + .raw)

│   ├── test_external_image.py       # External image testing- **Annotations**: XML with radiologist consensus

│   └── *.py                        # Utility scripts

│### Data Processing:

├── web_app/                         # Web application1. **Balanced Sampling**: 3:1 negative to positive ratio (6,228 samples)

│   ├── app.py                      # Flask backend2. **Train/Val/Test Split**: 70% / 15% / 15%

│   ├── static/                     # Frontend files3. **Patch Extraction**: 64×64 pixel patches with 3 consecutive slices

│   │   ├── index.html             # Main HTML page4. **Normalization**: Hounsfield Unit (HU) clipping (-1000 to 400)

│   │   ├── css/style.css          # Stylesheets5. **Augmentation**: Flips, rotations, brightness adjustments

│   │   └── js/script.js           # JavaScript

│   ├── requirements_frontend.txt   # Frontend dependencies**Download Dataset**: [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/)

│   ├── start_app.ps1              # PowerShell startup script

│   └── uploads/                    # Temporary upload folder---

│

├── requirements.txt                 # Main Python dependencies## 🏗️ Model Architectures

├── LICENSE                         # MIT License

└── README.md                       # This file### 1. ResNet-101

```- **Parameters**: 43.5M

- **Depth**: 101 layers with residual connections

## 🚀 Installation- **Strengths**: Deep architecture, skip connections prevent vanishing gradients

- **Performance**: 94.44% accuracy, 0.9772 AUC

### Prerequisites- **Use Case**: Best overall performance



- Python 3.8 or higher### 2. EfficientNet-B0

- CUDA-capable GPU (optional, CPU supported)- **Parameters**: 5.3M (8× smaller than ResNet-101)

- 8GB+ RAM recommended- **Depth**: Variable with compound scaling

- **Strengths**: Efficient, fast inference, low memory usage

### Setup- **Performance**: 96.15% accuracy, 0.9853 AUC, 0.9241 F1

- **Use Case**: Production deployment, mobile/edge devices

1. **Clone the repository**

   ```bash### 3. LUNA16-DenseNet

   git clone https://github.com/Kanavt15/CAD-C.git- **Parameters**: Custom DenseNet-169 architecture

   cd CAD-C- **Training Data**: Real LUNA16 CT scan patches

   ```- **Strengths**: Medical data trained, high performance on real cases

- **Performance**: F1-Score: 0.8071 on real medical data

2. **Create virtual environment**- **Use Case**: Primary model for clinical applications

   ```bash

   python -m venv venv### Ensemble Method

   Combines all three models using:

   # Windows- **Voting**: Majority vote among predictions

   venv\Scripts\activate- **Averaging**: Mean probability across models

   - **Weighted**: Custom weights per model (LUNA16-DenseNet: 50%, ResNet-101: 30%, EfficientNet-B0: 20%)

   # Linux/Mac- **Agreement Score**: Confidence based on model consensus

   source venv/bin/activate

   ```---



3. **Install dependencies**## 🔧 Installation

   ```bash

   pip install -r requirements.txt### Prerequisites

   ```- Python 3.8 or higher

- CUDA-capable GPU (recommended) or CPU

4. **For web application**- 32 GB RAM (for pre-extraction)

   ```bash

   cd web_app### Step 1: Clone Repository

   pip install -r requirements_frontend.txt```bash

   ```git clone https://github.com/Kanavt15/CAD-C.git

cd CAD-C

## 🏃 Quick Start```



### Training Models### Step 2: Create Virtual Environment

```bash

Navigate to the `notebooks/` folder and open any training notebook:# Windows

python -m venv venv_lung_cancer

```bashvenv_lung_cancer\Scripts\activate

jupyter notebook notebooks/lung_cancer_densenet.ipynb

```# Linux/Mac

python3 -m venv venv_lung_cancer

### Running Inferencesource venv_lung_cancer/bin/activate

```

Use the ensemble inference script:

### Step 3: Install Dependencies

```bash```bash

python scripts/inference_ensemble.py --image path/to/ct_scan.jpgpip install -r requirements.txt

``````



### Starting Web Application### Step 4: Download Dataset

1. Register at [LUNA16 Challenge](https://luna16.grand-challenge.org/)

```bash2. Download all subsets (subset0 - subset9)

cd web_app3. Download `annotations.csv` and `candidates_V2.csv`

python app.py4. Extract to project directory

```

### Directory Structure After Setup:

Then open your browser to `http://localhost:5000````

CAD_C/

For detailed instructions, see [docs/QUICKSTART.md](docs/QUICKSTART.md)├── subset0/

├── subset1/

## 🧠 Models├── ...

├── subset9/

### Available Models├── annotations.csv

├── candidates_V2.csv

| Model | Architecture | Input Size | Parameters | Best AUC |└── [other project files]

|-------|-------------|------------|------------|----------|```

| **DenseNet-169** | Dense Convolutional Network | 224×224×3 | ~14M | 0.95+ |

| **EfficientNet-B0** | Efficient CNN | 64×64×3 | ~5M | 0.93+ |---

| **ResNet-101** | Residual Network | 64×64×3 | ~44M | 0.94+ |

## 📁 Project Structure

### Model Features

```

- **DenseNet-169**: Custom medical imaging classification head with BatchNormCAD_C/

- **EfficientNet-B0**: Lightweight with dropout regularization│

- **ResNet-101**: Deep residual learning with enhanced classifier├── README.md                           # This file

- **Ensemble**: Weighted average predictions from all models├── requirements.txt                    # Python dependencies

├── INFERENCE_GUIDE.md                  # Detailed inference instructions

For detailed performance comparison, see [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)├── ENHANCEMENT_SUMMARY.md              # Project enhancements documentation

│

## 🌐 Web Application├── lung_cancer_resnet101.ipynb        # ResNet-101 training notebook

├── lung_cancer_efficientnet.ipynb     # EfficientNet-B0 training notebook

The web application provides:├── lung_cancer_densenet.ipynb         # LUNA16-DenseNet training notebook

│

- **Upload Interface**: Drag-and-drop or click to upload CT scans├── inference_ensemble.py              # Multi-model inference script

- **Model Selection**: Choose individual models or ensemble prediction├── test_external_image.py             # External image testing script

- **Real-time Results**: Instant predictions with confidence scores├── create_flowchart.py                # Visualization utilities

- **3D Visualization**: Interactive 3D lung model (Sketchfab integration)│

- **Visual Feedback**: Color-coded results with probability distributions├── annotations.csv                    # LUNA16 annotations

├── candidates_V2.csv                  # LUNA16 candidates

### Starting the Web App│

├── subset0/                           # CT scans (subset 0)

```bash├── subset1/                           # CT scans (subset 1)

cd web_app├── ...                                # ...

python app.py├── subset9/                           # CT scans (subset 9)

```│

├── models_resnet101/                  # ResNet-101 trained model

Or use the PowerShell script:│   ├── best_resnet101_model.pth

│   ├── training_history.csv

```bash│   ├── test_results.json

.\web_app\start_app.ps1│   └── [visualization plots]

```│

├── models_efficientnet/               # EfficientNet-B0 trained model

See [docs/FRONTEND_README.md](docs/FRONTEND_README.md) for detailed web app documentation.│   └── [similar structure]

│

## 📚 Documentation├── models_densenet/                   # LUNA16-DenseNet trained model

│   └── [LUNA16 real data trained model]

All documentation is available in the `docs/` folder:│

├── inference_results/                 # Inference output visualizations

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Get started quickly│   ├── inference_*.png

- **[MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)** - Compare model performance│   └── ...

- **[FRONTEND_README.md](docs/FRONTEND_README.md)** - Web application guide│

- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - How to contribute└── test_images/                       # External test images

- **[THRESHOLD_OPTIMIZATION_SUMMARY.md](docs/THRESHOLD_OPTIMIZATION_SUMMARY.md)** - Optimization details    ├── test_nodule_64x64.npy

    └── ...

## 🔧 Configuration```



### Data Configuration---



Edit `data/raw/enhanced_ensemble_config.json` to configure:## 🚀 Usage

- Model paths

- Threshold values### Training

- Preprocessing parameters

- Output settings#### Train ResNet-101

```bash

### Model Paths# Open Jupyter Notebook

jupyter notebook lung_cancer_resnet101.ipynb

Update paths in configuration files if models are stored elsewhere:

# Run all cells or execute cell-by-cell

```json# Training takes ~5-10 minutes with pre-extraction

{```

  "models": {

    "densenet": "models/models_densenet/best_densenet_model.pth",#### Train EfficientNet-B0

    "efficientnet": "models/models_efficientnet/best_efficientnet_model.pth",```bash

    "resnet101": "models/models_resnet101/best_resnet101_model.pth"jupyter notebook lung_cancer_efficientnet.ipynb

  }```

}

```#### Train LUNA16-DenseNet

```bash

## 🧪 Testingjupyter notebook lung_cancer_densenet.ipynb

```

Run tests using the test scripts:

### Inference (LUNA16 Dataset)

```bash

# Test with external imageRun inference on a nodule from the LUNA16 dataset:

python scripts/test_external_image.py --image assets/sample_images/healthy.jpg

```bash

# Test dataset configurationpython inference_ensemble.py \

python scripts/test_dataset_config.py  --series_uid 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260 \

```  --coord_x -56.08 \

  --coord_y -67.85 \

## 📊 Dataset  --coord_z -311.92

```

This project uses the **LUNA16** (Lung Nodule Analysis 2016) dataset:

- 888 CT scans from 888 patients**Output:**

- 10 subsets (subset0-subset9)- Individual model predictions with probabilities

- Annotations for nodule locations and sizes- Ensemble voting result

- Candidate nodules for training- Visualization with CT slices and prediction charts

- Detailed results table

Dataset is organized in `data/raw/subset*/`- Clinical recommendation



## 🤝 Contributing### Inference (External Images)



Contributions are welcome! Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.Run inference on external CT scan images:



1. Fork the repository#### Option 1: Single Image (JPG/PNG)

2. Create your feature branch (`git checkout -b feature/AmazingFeature`)```bash

3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)python inference_ensemble.py --image_path path/to/ct_scan.jpg

4. Push to the branch (`git push origin feature/AmazingFeature`)```

5. Open a Pull Request

#### Option 2: NumPy Array

## 📄 License```bash

python inference_ensemble.py --image_path path/to/ct_patch.npy

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.```



## 👥 Authors#### Option 3: DICOM File

```bash

- **Kanav** - [Kanavt15](https://github.com/Kanavt15)python inference_ensemble.py --image_path path/to/scan.dcm

```

## 🙏 Acknowledgments

#### With Custom Options

- LUNA16 Challenge organizers for the dataset```bash

- PyTorch team for the deep learning frameworkpython inference_ensemble.py \

- Open-source community for various tools and libraries  --image_path images.jpg \

  --no-viz              # Disable visualization

## 📧 Contact  --no-save             # Don't save results

```

For questions or issues, please:

- Open an issue on GitHub### Command-Line Options

- Contact: [Your email or contact info]

| Option | Description | Default |

## 🔗 Links|--------|-------------|---------|

| `--series_uid` | LUNA16 series UID | Required for LUNA16 |

- [LUNA16 Challenge](https://luna16.grand-challenge.org/)| `--coord_x/y/z` | World coordinates | Required for LUNA16 |

- [PyTorch Documentation](https://pytorch.org/docs/)| `--image_path` | Path to external image | Required for external |

- [Project Repository](https://github.com/Kanavt15/CAD-C)| `--no-viz` | Disable visualization | False |

| `--no-save` | Don't save results | False |

---

---

**Note**: This is a research project for educational purposes. Not intended for clinical use without proper validation and regulatory approval.

## 📊 Results

### ResNet-101 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 94.44% |
| **AUC-ROC** | 0.9772 |
| **F1 Score** | 0.8903 |
| **Precision** | 87.96% |
| **Recall** | 90.17% |
| **Training Time** | ~2.75 minutes |
| **Epochs** | 22 (early stopping) |

### Confusion Matrix (ResNet-101)
```
                Predicted
              Non-Cancer  Cancer
Actual
Non-Cancer       678        23
Cancer            23        211
```

### Model Comparison

| Model | Parameters | Accuracy | AUC | F1 | Training Time | Inference Time |
|-------|-----------|----------|-----|-----|---------------|----------------|
| **ResNet-101** | 43.5M | 94.44% | 0.9772 | 0.8903 | ~2.75 min | ~8-12 ms |
| **EfficientNet-B0** | 5.3M | 96.15% | 0.9853 | 0.9241 | ~2 min | ~5-8 ms |
| **LUNA16-DenseNet** | Custom | Real Data | Real Data | 0.8071 | Variable | ~40-50 ms |

### Training History
![Training History](models_resnet101/training_history.png)

### ROC Curve
![ROC Curve](models_resnet101/roc_curve.png)

---

## 🔬 Technical Details

### Data Pipeline
1. **Load LUNA16 candidates** from CSV
2. **Balance dataset** (3:1 negative:positive ratio)
3. **Split data** (70/15/15 train/val/test)
4. **Pre-extract patches** (~7.5 minutes for 6,228 samples)
   - Load CT scan (.mhd/.raw)
   - Convert world → voxel coordinates
   - Extract 64×64×3 patches
   - Normalize HU values
   - Cache in memory (292 MB)
5. **Data augmentation** (training only)
   - Random horizontal flip
   - Random vertical flip
   - Random rotation (90°, 180°, 270°)
   - Random brightness adjustment

### Model Architecture

#### ResNet-101
```python
ResNet101 (pretrained=ImageNet)
├── Convolutional backbone (frozen or fine-tuned)
└── Custom classifier:
    ├── Dropout(0.5)
    ├── Linear(2048 → 512)
    ├── ReLU
    ├── Dropout(0.5)
    └── Linear(512 → 2)
```

#### Focal Loss
```python
FL(pt) = -α(1-pt)^γ * log(pt)
α = 0.75  # Weight for positive class
γ = 2.0   # Focusing parameter
```

### Training Configuration
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss Function**: Focal Loss (α=0.75, γ=2.0)
- **Batch Size**: 32
- **Early Stopping**: Patience = 10 epochs
- **Device**: CUDA (GPU) or CPU

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 (6GB) | RTX 3060+ (12GB+) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 100 GB | 200 GB (for full dataset) |
| **CPU** | 4 cores | 8+ cores |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contribution
- [ ] Add more CNN architectures (DenseNet, MobileNet, Inception)
- [ ] Implement 3D CNN for volumetric analysis
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Create web interface (Flask/Streamlit)
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning (Optuna/Ray Tune)
- [ ] Create Docker container for easy deployment
- [ ] Add unit tests and CI/CD pipeline

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LUNA16 Dataset**: [Grand Challenge](https://luna16.grand-challenge.org/)
- **LIDC-IDRI Dataset**: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
- **PyTorch**: Deep learning framework
- **SimpleITK**: Medical image processing
- **Torchvision**: Pre-trained models

### References

1. Setio, A. A. A., et al. (2017). "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge." *Medical Image Analysis*, 42, 1-13.

2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

3. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.

4. Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *arXiv:1409.1556*.

5. Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.

---

## 📧 Contact

**Project Maintainer**: Kanav

- GitHub: [@Kanavt15](https://github.com/Kanavt15)
- Repository: [CAD-C](https://github.com/Kanavt15/CAD-C)

**Project Link**: [https://github.com/Kanavt15/CAD-C](https://github.com/Kanavt15/CAD-C)

---

## 📈 Project Status

- ✅ ResNet-101: **Trained & Tested** (94.44% accuracy, 0.9772 AUC)
- ✅ EfficientNet-B0: **Trained & Tested** (96.15% accuracy, 0.9853 AUC)
- ✅ LUNA16-DenseNet: **Trained & Optimized** (F1: 0.8071 on real medical data) - **Primary Model!**
- ✅ Inference Script: **Complete with Threshold Optimization**
- ✅ External Image Support: **Complete**
- ✅ Documentation: **Complete**

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{lung-cancer-detection-2025,
  author = {Kanav},
  title = {LUNA16 Lung Cancer Detection System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Kanavt15/CAD-C}}
}
```

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ for advancing medical AI research

</div>
