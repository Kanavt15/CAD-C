# 🚀 Quick Start Guide

Get started with LUNA16 Lung Cancer Detection in 5 minutes!

## ⚡ Fast Setup

### 1. Clone and Setup (2 minutes)
```bash
# Clone the repository
git clone https://github.com/yourusername/lung-cancer-detection.git
cd lung-cancer-detection

# Create virtual environment
python -m venv venv_lung_cancer

# Activate (Windows)
venv_lung_cancer\Scripts\activate

# Activate (Linux/Mac)
source venv_lung_cancer/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models (Optional)
If you don't want to train from scratch:
- Download pre-trained models from [Releases](https://github.com/yourusername/lung-cancer-detection/releases)
- Extract to `models_resnet101/`, `models_efficientnet/`, `models_vgg16/`

### 3. Test with External Image (30 seconds)
```bash
# Create a test image
python test_external_image.py

# Run inference
python inference_ensemble.py --image_path test_images/test_nodule_64x64.npy
```

**That's it!** You should see predictions from all three models.

---

## 📊 Training from Scratch

### Prerequisites
- Download LUNA16 dataset (~120 GB)
- 32 GB RAM recommended
- CUDA GPU (optional but recommended)

### Step-by-Step Training

#### 1. Download Dataset
```bash
# Register at https://luna16.grand-challenge.org/
# Download all files:
# - subset0.zip through subset9.zip
# - annotations.csv
# - candidates_V2.csv

# Extract all subsets to project root
```

#### 2. Train ResNet-101 (5-10 minutes)
```bash
# Open Jupyter
jupyter notebook

# Open: lung_cancer_resnet101.ipynb
# Run all cells (Ctrl+Enter through each cell)
```

**Expected Results:**
- Training: ~2.75 minutes (with pre-extraction)
- Test Accuracy: ~94%
- AUC: ~0.97

#### 3. Train EfficientNet-B0 (3-5 minutes)
```bash
# Open: lung_cancer_efficientnet.ipynb
# Run all cells
```

#### 4. Train VGG16 (7-12 minutes)
```bash
# Open: lung_cancer_vgg16.ipynb
# Run all cells
```

---

## 🎯 Usage Examples

### Example 1: LUNA16 Dataset Inference
```bash
python inference_ensemble.py \
  --series_uid 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260 \
  --coord_x -56.08 \
  --coord_y -67.85 \
  --coord_z -311.92
```

### Example 2: External JPG Image
```bash
python inference_ensemble.py --image_path path/to/ct_scan.jpg
```

### Example 3: NumPy Array
```bash
python inference_ensemble.py --image_path path/to/patch.npy
```

### Example 4: Batch Processing (Script)
```python
import glob
from inference_ensemble import run_inference

# Process all images in a folder
for img_path in glob.glob("images/*.jpg"):
    result = run_inference(image_path=img_path, visualize=False)
    print(f"{img_path}: {result['ensemble']['vote_result']}")
```

---

## 📋 Common Issues

### Issue 1: CUDA Out of Memory
**Solution:**
```python
# Edit notebook CONFIG:
CONFIG = {
    'batch_size': 16,  # Reduce from 32
    'preload_patches': False,  # Disable pre-extraction
    ...
}
```

### Issue 2: Missing Models
**Error:** `Model not found: models_resnet101/best_resnet101_model.pth`

**Solution:**
- Train the model first using the Jupyter notebooks
- Or download pre-trained models from releases

### Issue 3: Dataset Not Found
**Error:** `Could not find CT scan with series UID`

**Solution:**
```python
# Verify dataset location in inference_ensemble.py
BASE_DIR = Path(r'e:\Kanav\Projects\CAD_C')  # Update this path
SUBSET_DIRS = [BASE_DIR / f'subset{i}' for i in range(10)]
```

### Issue 4: Import Errors
**Error:** `ModuleNotFoundError: No module named 'SimpleITK'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

## 🎓 Understanding the Output

### Individual Model Predictions
```
ResNet-101: 🔴 CANCER (Prob: 87.32%, Confidence: 87.32%)
EfficientNet-B0: 🔴 CANCER (Prob: 78.91%, Confidence: 78.91%)
VGG16: 🟢 NON-CANCER (Prob: 45.67%, Confidence: 54.33%)
```

### Ensemble Result
```
🎯 ENSEMBLE (Voting): 🔴 CANCER
Average Probability: 70.63%
Weighted Probability: 71.24%
Model Agreement: 66.7%
```

**Interpretation:**
- **Voting Result**: 2 out of 3 models predicted cancer → CANCER
- **Average Probability**: Mean cancer probability across all models
- **Agreement**: 66.7% means 2 models agreed on cancer (not unanimous)

### Confidence Levels
- **> 80%**: High confidence
- **60-80%**: Moderate confidence
- **< 60%**: Low confidence (additional screening recommended)

---

## 🔧 Customization

### Change Model Weights
Edit `inference_ensemble.py`:
```python
# Equal weights (default)
weights = {'ResNet-101': 0.33, 'EfficientNet-B0': 0.33, 'VGG16': 0.34}

# Custom weights (e.g., trust ResNet-101 more)
weights = {'ResNet-101': 0.5, 'EfficientNet-B0': 0.3, 'VGG16': 0.2}
```

### Change Decision Threshold
```python
# In predict_single_model function
prediction = 1 if prob_nodule > 0.4 else 0  # More sensitive (default: 0.5)
prediction = 1 if prob_nodule > 0.6 else 0  # More specific
```

### Disable Visualization
```bash
python inference_ensemble.py --image_path image.jpg --no-viz
```

---

## 📊 Performance Benchmarks

### Training Time (32 GB RAM, RTX 3060)
| Model | Pre-extraction | Training | Total |
|-------|---------------|----------|-------|
| ResNet-101 | 7.5 min | 2.75 min | **~10 min** |
| EfficientNet-B0 | 7.5 min | 1.5 min | **~9 min** |
| VGG16 | 7.5 min | 4 min | **~12 min** |

### Inference Time (Single Image)
| Model | GPU | CPU |
|-------|-----|-----|
| ResNet-101 | 8-12 ms | 80-120 ms |
| EfficientNet-B0 | 5-8 ms | 50-80 ms |
| VGG16 | 15-20 ms | 150-200 ms |
| **Ensemble (All 3)** | **30-40 ms** | **300-400 ms** |

---

## 🎯 Next Steps

### For Researchers
1. ✅ Train all three models
2. ✅ Compare performance metrics
3. ✅ Analyze confusion matrices
4. ✅ Experiment with hyperparameters
5. ✅ Try different architectures

### For Developers
1. ✅ Integrate into existing systems
2. ✅ Create REST API (Flask/FastAPI)
3. ✅ Build web interface (Streamlit)
4. ✅ Deploy to cloud (AWS/Azure/GCP)
5. ✅ Containerize with Docker

### For Medical Professionals
1. ✅ Run inference on clinical cases
2. ✅ Validate against radiologist readings
3. ✅ Integrate into PACS workflow
4. ✅ Generate reports for physicians
5. ✅ Track performance metrics

---

## 📚 Additional Resources

- [Full README](README.md) - Comprehensive documentation
- [Inference Guide](INFERENCE_GUIDE.md) - Detailed inference instructions
- [Enhancement Summary](ENHANCEMENT_SUMMARY.md) - Project improvements
- [LUNA16 Dataset](https://luna16.grand-challenge.org/) - Official dataset page
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework

---

## 🤝 Get Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/lung-cancer-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lung-cancer-detection/discussions)
- **Email**: your.email@example.com

---

<div align="center">

### 🌟 Happy Detecting!

*Early detection saves lives*

</div>
