# Web Application Quick Start Guide 🚀

## Starting the App

```bash
cd web_app
python app.py
```

Then open: **http://localhost:5000**

---

## Using the Interface

### 1. Upload Image 📤
- **Drag & drop** image onto upload area
- OR **click "Browse Files"** button
- Formats: JPG, PNG, DICOM, CT scans

### 2. Select Models 🤖
Choose one or more:
- ☐ **DenseNet169** - Medical imaging specialist
- ☐ **EfficientNet-B0** - Fast & efficient
- ☐ **ResNet101** - Deep learning powerhouse

*Tip: Select 2+ models for ensemble prediction!*

### 3. Analyze 🔍
- Click **"Analyze Image"** button
- Wait 2-5 seconds for processing
- Results appear below automatically

### 4. View Results 📊
Each model shows:
- **Prediction**: Cancerous / Non-Cancerous
- **Confidence**: Percentage score
- **Probabilities**: Detailed breakdown
- **Color coding**: 🟢 Green = Healthy, 🔴 Red = Cancerous

---

## Example Results

```
┌─────────────────────────────┐
│ 🧠 DenseNet169              │
├─────────────────────────────┤
│ ✅ Non-Cancerous            │
│ Confidence: 94.52%          │
│                             │
│ ✅ Non-Cancerous: 94.52%    │
│ ⚠️  Cancerous: 5.48%        │
└─────────────────────────────┘

┌─────────────────────────────┐
│ 🧠 Ensemble Prediction      │
├─────────────────────────────┤
│ ✅ Non-Cancerous            │
│ Confidence: 92.15%          │
└─────────────────────────────┘
```

---

## 3D Visualization 🫁

Scroll down to see interactive 3D lung model:
- **Click + drag** to rotate
- **Scroll** to zoom
- **Right-click + drag** to pan

---

## Troubleshooting 🔧

### Models Not Loading?
```bash
# Check if models exist:
ls ../models/models_densenet/best_densenet_model.pth
ls ../models/models_efficientnet/best_efficientnet_model.pth
ls ../models/models_resnet101/best_resnet101_model.pth
```

### Results Not Showing?
1. Open browser console (F12)
2. Check for errors
3. Verify API response in Network tab

### Server Won't Start?
```bash
# Install dependencies:
pip install -r requirements_frontend.txt

# Check if port 5000 is free:
netstat -ano | findstr :5000
```

---

## Keyboard Shortcuts ⌨️

- **Ctrl+O** - Open file browser
- **Ctrl+R** - Reload page
- **F12** - Open developer tools
- **F5** - Refresh

---

## Best Practices 💡

1. **Image Quality**: Use clear, high-resolution CT scans
2. **Multiple Models**: Select all 3 for most accurate results
3. **Compare Results**: Look at individual + ensemble predictions
4. **Medical Advice**: This is for research only, not clinical use

---

## API Endpoints 🔌

For programmatic access:

```python
import requests

# Health check
response = requests.get('http://localhost:5000/api/health')

# Prediction
files = {'image': open('scan.jpg', 'rb')}
data = {'models[]': ['densenet', 'efficientnet', 'resnet101']}
response = requests.post('http://localhost:5000/api/predict', 
                        files=files, data=data)
print(response.json())
```

---

## Support 📧

- **Issues**: [GitHub Issues](https://github.com/Kanavt15/CAD-C/issues)
- **Docs**: See `docs/` folder
- **README**: [Main README](../README.md)

---

**Quick Start Complete! Happy Analyzing! 🎉**
