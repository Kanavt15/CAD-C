# Lung Cancer Detection Web Application

A modern, AI-powered web application for lung cancer detection using three state-of-the-art deep learning models: **DenseNet169**, **EfficientNet-B0**, and **ResNet101**.

## 🎯 Features

- **Multi-Model Analysis**: Analyze medical images using three powerful CNN architectures
- **Ensemble Predictions**: Get consensus predictions from multiple models for higher confidence
- **Interactive 3D Visualization**: Explore lung anatomy with an embedded 3D model
- **Real-time Results**: Get instant predictions with confidence scores and probabilities
- **Modern UI**: Clean, responsive design optimized for medical professionals
- **Drag & Drop Upload**: Easy image upload with preview functionality

## 🏗️ Architecture

### Backend (Flask)
- RESTful API endpoints for model inference
- Supports DenseNet169, EfficientNet-B0, and ResNet101 models
- GPU acceleration support (CUDA)
- Image preprocessing and transformation pipeline

### Frontend
- Pure HTML/CSS/JavaScript (no framework dependencies)
- Responsive design for desktop and mobile
- Real-time feedback and notifications
- Interactive result visualization

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Modern web browser (Chrome, Firefox, Edge, Safari)

## 🚀 Installation

### 1. Clone the Repository

```bash
cd e:\Kanav\Projects\CAD_C
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv_frontend

# Activate virtual environment
.\venv_frontend\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements_frontend.txt
```

### 4. Verify Model Files

Ensure the following model files exist:
- `models_densenet/best_densenet_model.pth`
- `models_efficientnet/best_efficientnet_model.pth`
- `models_resnet101/best_resnet101_model.pth`

## 🎮 Usage

### Starting the Application

1. **Activate Virtual Environment** (if not already activated):
```powershell
.\venv_frontend\Scripts\Activate.ps1
```

2. **Run the Flask Server**:
```powershell
python app.py
```

3. **Open Browser**:
Navigate to `http://localhost:5000`

### Using the Application

1. **Upload Image**:
   - Drag and drop a medical image into the upload area, OR
   - Click "Browse Files" to select an image

2. **Select Models**:
   - Choose one or more models for analysis
   - All models are selected by default

3. **Analyze**:
   - Click "Analyze Image" button
   - Wait for the analysis to complete

4. **View Results**:
   - Individual model predictions with confidence scores
   - Ensemble prediction (if multiple models selected)
   - Probability breakdowns for each class

5. **Explore 3D Visualization**:
   - Interact with the 3D lung model
   - Rotate, zoom, and explore anatomical details

## 📁 Project Structure

```
CAD_C/
├── app.py                          # Flask backend server
├── requirements_frontend.txt       # Python dependencies
├── static/                         # Frontend assets
│   ├── index.html                 # Main HTML page
│   ├── css/
│   │   └── style.css              # Stylesheet
│   └── js/
│       └── script.js              # JavaScript logic
├── models_densenet/               # DenseNet model files
├── models_efficientnet/           # EfficientNet model files
├── models_resnet101/              # ResNet101 model files
└── uploads/                       # Temporary upload directory
```

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and loaded models.

### Get Available Models
```
GET /api/models
```
Returns list of available models.

### Predict
```
POST /api/predict
```
**Form Data**:
- `image`: Image file (required)
- `models[]`: List of model names (optional, defaults to all)

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "model": "densenet",
      "prediction": "Non-Cancerous",
      "confidence": 95.32,
      "probabilities": {
        "non_cancerous": 95.32,
        "cancerous": 4.68
      }
    }
  ]
}
```

## 🎨 Customization

### Changing Port
Edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### API URL (for Frontend)
Edit `static/js/script.js`:
```javascript
const API_URL = 'http://localhost:5000';  // Change URL here
```

### Styling
Modify `static/css/style.css` to customize colors, fonts, and layout.

## 🐛 Troubleshooting

### Models Not Loading
**Issue**: Models not found or failed to load

**Solution**:
- Verify model files exist in correct directories
- Check file paths in `app.py`
- Ensure PyTorch version matches the one used for training

### CORS Errors
**Issue**: Cross-origin errors in browser console

**Solution**:
- Ensure `flask-cors` is installed
- Verify CORS is enabled in `app.py`

### Port Already in Use
**Issue**: Port 5000 already in use

**Solution**:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### CUDA Out of Memory
**Issue**: GPU memory error

**Solution**:
- Reduce batch size (already set to 1)
- Use CPU instead: The app automatically falls back to CPU if CUDA is unavailable

## 🔒 Security Considerations

⚠️ **Important**: This is a development server. For production deployment:

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Add authentication and authorization
3. Implement rate limiting
4. Use HTTPS
5. Validate and sanitize all inputs
6. Set up proper logging and monitoring

## 📊 Model Information

### DenseNet169
- **Architecture**: Dense Convolutional Network
- **Parameters**: ~14M
- **Key Feature**: Dense connections between layers

### EfficientNet-B0
- **Architecture**: Compound scaled CNN
- **Parameters**: ~5M
- **Key Feature**: Balanced depth, width, and resolution

### ResNet101
- **Architecture**: Residual Network
- **Parameters**: ~44M
- **Key Feature**: Skip connections to avoid vanishing gradients

## ⚕️ Medical Disclaimer

**IMPORTANT**: This application is intended for research and educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding a medical condition.

## 📝 License

This project is part of the CAD-C lung cancer detection system. Please refer to the main repository LICENSE file for licensing information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 👥 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

## 🎓 Citation

If you use this application in your research, please cite:

```bibtex
@software{lung_cancer_detection_webapp,
  author = {Your Name},
  title = {Lung Cancer Detection Web Application},
  year = {2025},
  url = {https://github.com/Kanavt15/CAD-C}
}
```

## 🔄 Version History

- **v1.0.0** (2025-10-06): Initial release
  - Three-model support (DenseNet, EfficientNet, ResNet)
  - Ensemble predictions
  - 3D visualization integration
  - Modern responsive UI

---

**Built with ❤️ for advancing medical AI research**
