# Multi-Model Web App Update

## 🎉 What's New

The web application has been updated to support **TWO** state-of-the-art AI models for lung cancer detection!

### Models Available:

#### 1. **ResNet3D with SE Blocks** (Original)
- **Accuracy**: 83.3%
- **F1 Score**: 0.67
- **Precision**: ~55%
- **Recall**: 83% (High sensitivity - catches most cases!)
- **Architecture**: Residual connections + Squeeze-and-Excitation attention
- **Threshold**: 32% (optimized for medical sensitivity - better to flag for review than miss)
- **Best for**: Initial screening where catching potential cases is critical

#### 2. **DenseNet3D with Multi-Head Attention** ⭐ NEW!
- **Accuracy**: 95.73%
- **F1 Score**: 0.7775
- **Precision**: 86.01% (Far fewer false positives!)
- **Recall**: 70.94%
- **Architecture**: Dense feature concatenation + 4-head self-attention
- **Parameters**: 672,770
- **Threshold**: 50% (balanced for optimal precision-recall tradeoff)
- **Best for**: Reliable diagnosis with high confidence

## 🚀 Running the Updated App

### 1. Make sure you have the model files:
```
web_app/
├── densenet3d_architecture.py      ✅ Created
├── densenet3d_attention.pth        ✅ Copied
├── densenet_model_info.json        ✅ Copied
└── models_3d_cnn/
    ├── best_improved_3d_cnn_model.pth   ✅ Existing
    └── model_architecture.py            ✅ Existing
```

### 2. Start the server:
```powershell
cd web_app
python app.py
```

### 3. Access the web app:
Open browser: http://localhost:5000

## 📊 How It Works

### Automatic Ensemble Prediction
When you upload an image, **both models** analyze it simultaneously:
- Each model provides its own prediction
- Results are displayed side-by-side for comparison
- Get more confidence by seeing consensus between models

### Model Comparison
The results page will show:
- **Individual predictions** from each model
- **Confidence scores** (probability of cancer)
- **Model-specific thresholds** explained
- **Performance metrics** for transparency

## 🎯 Key Features

### 1. **Model Selection** (Coming Soon)
Checkboxes in the UI allow you to select which models to use:
- Use both for maximum confidence
- Use DenseNet for highest accuracy
- Use ResNet for faster inference

### 2. **Improved Accuracy**
DenseNet3D achieves **95.73% accuracy** - a significant improvement over the original 83.3%

### 3. **Better Class Balance**
DenseNet was trained with:
- **Focal Loss** (γ=2.0) to handle extreme class imbalance
- **85x class weighting** for positive samples
- Result: Better detection of rare cancer cases

### 4. **Multi-Head Attention**
DenseNet uses 4-head self-attention to:
- Learn spatial relationships in 3D lung scans
- Capture complex patterns
- Improve feature representation

## 📁 Updated Files

### Backend (Python):
- `app.py` - Updated with dual-model support
- `densenet3d_architecture.py` - New DenseNet model definition
- `densenet_model_info.json` - Test results and metrics

### Frontend (HTML/CSS):
- `static/analyze.html` - Updated model info cards with both models
- `static/css/style.css` - Added styles for checkboxes, badges, and model selection

### JavaScript (No changes needed):
- `static/js/script.js` - Already handles multiple results

## 🔧 API Changes

### `/api/models` Endpoint
Now returns info for both models:
```json
{
  "models": [
    {
      "id": "resnet3d",
      "name": "ResNet3D with SE Blocks",
      "accuracy": 83.3,
      "f1_score": 0.67,
      ...
    },
    {
      "id": "densenet3d",
      "name": "DenseNet3D with Multi-Head Attention",
      "accuracy": 95.73,
      "f1_score": 0.7775,
      ...
    }
  ]
}
```

### `/api/predict` Endpoint
Supports model selection (optional):
```javascript
// Default: Use all models
formData.append('image', file);
formData.append('models', 'all');

// Or select specific models
formData.append('models', 'resnet3d,densenet3d');
```

Returns array of results (one per model):
```json
{
  "success": true,
  "results": [
    {
      "model": "ResNet3D with SE Blocks",
      "prediction": "Cancerous",
      "confidence": 65.4,
      ...
    },
    {
      "model": "DenseNet3D with Multi-Head Attention",
      "prediction": "Cancerous",
      "confidence": 82.1,
      ...
    }
  ]
}
```

## 🏆 Performance Comparison

| Metric | ResNet3D | DenseNet3D | Winner |
|--------|----------|------------|--------|
| **Accuracy** | 83.3% | **95.73%** | 🥇 DenseNet |
| **F1 Score** | 0.67 | **0.7775** | 🥇 DenseNet |
| **Precision** | ~55% | **86.01%** | 🥇 DenseNet |
| **Recall** | ~83% | 70.94% | 🥇 ResNet |
| **Parameters** | ~2.5M | 672K | 🥇 DenseNet (smaller!) |

## 💡 Usage Tips

1. **Both models agree** → High confidence in result
2. **Models disagree** → Uncertain case, seek medical review
3. **High DenseNet confidence** → More reliable (better precision)
4. **High ResNet recall** → Good at catching potential cases

## 🐛 Troubleshooting

### Models not loading?
Check that these files exist:
```bash
ls web_app/densenet3d_attention.pth
ls web_app/models_3d_cnn/best_improved_3d_cnn_model.pth
```

### CUDA errors?
App is configured to use CPU by default for compatibility.

### Import errors?
Make sure you're in the web_app directory:
```bash
cd web_app
python app.py
```

## 📚 Architecture Details

### DenseNet3D Features:
- **Dense Blocks**: Features concatenated (not added like ResNet)
- **Transition Layers**: Compress features between blocks
- **Multi-Head Attention**: 4 heads, channels // 4 per head
- **Drop Path**: Stochastic depth (0.1 rate)
- **Focal Loss**: Addresses 89.5% negative / 10.5% positive imbalance

### Training Configuration:
- **Optimizer**: AdamW (lr=0.0001, wd=1e-4)
- **Scheduler**: Cosine Annealing (T_max=60)
- **Gradient Clipping**: max_norm=1.0
- **Mixed Precision**: Enabled (AMP with GradScaler)
- **Batch Size**: 4
- **Epochs**: 43 (early stopped, patience=15)

## 🎓 For Developers

To add more models in the future:
1. Create architecture file in `web_app/`
2. Copy model checkpoint (.pth) to `web_app/`
3. Add loading function in `app.py`
4. Add prediction function
5. Update `/api/models` endpoint
6. Add model card in `analyze.html`

---

**Note**: This update maintains full backward compatibility. The original ResNet model continues to work alongside the new DenseNet model.
