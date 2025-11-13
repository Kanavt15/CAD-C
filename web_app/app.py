"""
Lung Cancer Detection Web Application
Multi-Model AI System with 3D CNN Architectures
"""

import os
import io
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.ndimage import zoom

# Import model architectures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models_3d_cnn'))
from model_architecture import ImprovedCNN3D_Nodule_Detector
from efficientnet3d_b2_architecture import EfficientNet3D_B2
from densenet3d_architecture import DenseNet3D_Attention

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
PATCH_SIZE = 64
HU_MIN = -1000
HU_MAX = 400
CANCER_THRESHOLD = 0.32

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Device configuration
device = torch.device('cpu')
print(f"Using device: {device}")

# Model paths and info
MODELS_CONFIG = {
    'improved_3d_cnn': {
        'path': 'models_3d_cnn/best_improved_3d_cnn_model.pth',
        'info_path': 'models_3d_cnn/model_info.json',
        'architecture': ImprovedCNN3D_Nodule_Detector,
        'display_name': 'Improved 3D CNN (Residual + SE)'
    },
    'efficientnet3d_b2': {
        'path': 'efficientnet3d_b2.pth',
        'info_path': 'efficientnet_model_info.json',
        'architecture': EfficientNet3D_B2,
        'display_name': 'EfficientNet3D-B2'
    },
    'densenet3d_attention': {
        'path': 'densenet3d_attention.pth',
        'info_path': 'densenet_model_info.json',
        'architecture': DenseNet3D_Attention,
        'display_name': 'DenseNet3D + Attention'
    }
}

# Store loaded models and info
models = {}
model_info = {}

def load_model_info(info_path):
    """Load model information from JSON"""
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}

def load_model(model_key):
    """Load a specific model"""
    config = MODELS_CONFIG[model_key]
    
    try:
        if not os.path.exists(config['path']):
            print(f"⚠ Model file not found: {config['path']}")
            return None
        
        print(f"\nLoading {config['display_name']}...")
        
        # Initialize model
        if model_key == 'improved_3d_cnn':
            model = config['architecture'](in_channels=1, num_classes=2)
        elif model_key == 'efficientnet3d_b2':
            model = config['architecture'](in_channels=1, num_classes=2, 
                                          width_mult=1.1, depth_mult=1.1, dropout_rate=0.3)
        elif model_key == 'densenet3d_attention':
            model = config['architecture'](in_channels=1, num_classes=2, 
                                          growth_rate=16, num_layers=[4, 4, 4], 
                                          num_heads=4, drop_path_rate=0.2)
        
        # Load checkpoint
        checkpoint = torch.load(config['path'], map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Load model info
        info = load_model_info(config['info_path'])
        model_info[model_key] = info
        
        # Extract accuracy and F1 - handle different JSON formats
        if 'test_metrics' in info:
            # DenseNet format
            accuracy = info['test_metrics'].get('accuracy', 0) * 100
            f1 = info['test_metrics'].get('f1', 0)
        else:
            # Standard format
            accuracy = info.get('test_accuracy', 0)
            f1 = info.get('test_f1_score', 0)
        
        print(f"  ✓ {config['display_name']} loaded successfully")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  F1 Score: {f1:.4f}")
        
        return model
        
    except Exception as e:
        print(f"  ✗ Error loading {config['display_name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Load all available models
print("\n" + "="*60)
print("LOADING AI MODELS")
print("="*60)

for model_key in MODELS_CONFIG.keys():
    model = load_model(model_key)
    if model is not None:
        models[model_key] = model

print(f"\n✓ Loaded {len(models)}/{len(MODELS_CONFIG)} models successfully")
print("="*60 + "\n")

def normalize_hu(image):
    """Normalize CT Hounsfield Units to [0, 1] range"""
    image = np.clip(image, HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    return image.astype(np.float32)

def normalize_image(image):
    """Normalize any image to [0, 1] range"""
    if image.max() > 1.0:
        if image.min() < 0:  # Likely HU values
            return normalize_hu(image)
        else:  # Likely 0-255 range
            return (image / 255.0).astype(np.float32)
    return image.astype(np.float32)

def preprocess_image_3d(image_file):
    """
    Preprocess uploaded image for 3D CNN
    Creates a 64x64x64 3D patch from 2D image
    """
    try:
        # Read image
        image = Image.open(image_file).convert('L')  # Convert to grayscale
        image_array = np.array(image).astype(np.float32)
        
        print(f"Original image shape: {image_array.shape}")
        print(f"Original value range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # Normalize the image
        image_array = normalize_image(image_array)
        
        print(f"Normalized value range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # Resize to PATCH_SIZE x PATCH_SIZE if needed
        if image_array.shape[0] != PATCH_SIZE or image_array.shape[1] != PATCH_SIZE:
            zoom_factors = (PATCH_SIZE / image_array.shape[0], PATCH_SIZE / image_array.shape[1])
            image_array = zoom(image_array, zoom_factors, order=1)
            print(f"Resized to: {image_array.shape}")
        
        # Create 3D patch by stacking the 2D image
        # For a real medical image, this would be actual CT slices
        # Here we replicate the image to create a 64x64x64 volume
        patch_3d = np.stack([image_array] * PATCH_SIZE, axis=0)
        
        print(f"Final 3D patch shape: {patch_3d.shape}")
        
        # Convert to tensor: (D, H, W) -> (1, 1, D, H, W)
        img_tensor = torch.from_numpy(patch_3d).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        print(f"Tensor shape: {img_tensor.shape}")
        
        return img_tensor
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_with_model(model_key, img_tensor, threshold=CANCER_THRESHOLD):
    """Make prediction with a specific model"""
    try:
        model = models.get(model_key)
        if model is None:
            return {
                'model': MODELS_CONFIG[model_key]['display_name'],
                'error': 'Model not loaded'
            }
        
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            non_cancerous_prob = probabilities[0][0].item()
            cancerous_prob = probabilities[0][1].item()
            
            # Classification logic
            if cancerous_prob >= threshold:
                predicted_class = 1
                prediction_label = 'Cancerous'
                confidence = cancerous_prob * 100
            elif cancerous_prob >= 0.25:
                predicted_class = 2
                prediction_label = 'Suspicious - Possible Cancer (Needs Review)'
                confidence = cancerous_prob * 100
            else:
                predicted_class = 0
                prediction_label = 'Non-Cancerous'
                confidence = non_cancerous_prob * 100
            
            info = model_info.get(model_key, {})
            
            result = {
                'model': model_key,
                'model_name': MODELS_CONFIG[model_key]['display_name'],
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {
                    'non_cancerous': float(non_cancerous_prob * 100),
                    'cancerous': float(cancerous_prob * 100)
                },
                'threshold': float(threshold * 100),
                'warning': 'Further medical review recommended' if predicted_class == 2 else None,
                'model_info': {
                    'accuracy': info.get('test_accuracy', 0),
                    'f1_score': info.get('test_f1_score', 0),
                    'precision': info.get('test_precision', 0),
                    'recall': info.get('test_recall', 0)
                }
            }
            
            return result
            
    except Exception as e:
        return {
            'model': MODELS_CONFIG[model_key]['display_name'],
            'error': str(e)
        }

@app.route('/')
def index():
    """Serve the landing page"""
    return send_from_directory('static', 'landing.html')

@app.route('/analyze')
def analyze():
    """Serve the analyze page"""
    return send_from_directory('static', 'analyze.html')

@app.route('/about')
def about():
    """Serve the about page"""
    return send_from_directory('static', 'about.html')

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Return information about available models"""
    available_models = []
    
    for model_key, model in models.items():
        info = model_info.get(model_key, {})
        
        # Handle different JSON formats
        if 'test_metrics' in info:
            # DenseNet format
            accuracy = info['test_metrics'].get('accuracy', 0) * 100
            f1_score = info['test_metrics'].get('f1', 0)
            parameters = info.get('total_params', 0)
        else:
            # Standard format
            accuracy = info.get('test_accuracy', 0)
            f1_score = info.get('test_f1_score', 0)
            parameters = info.get('parameters', 0)
        
        available_models.append({
            'id': model_key,
            'name': MODELS_CONFIG[model_key]['display_name'],
            'accuracy': accuracy,
            'f1_score': f1_score,
            'parameters': parameters,
            'description': info.get('description', '')
        })
    
    return jsonify({
        'models': available_models,
        'count': len(available_models)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    print("\n" + "="*60)
    print("Received prediction request")
    
    try:
        # Check if models are loaded
        if len(models) == 0:
            return jsonify({
                'success': False,
                'error': 'No models loaded. Please restart the server.'
            }), 500
        
        # Check if image was uploaded
        if 'image' not in request.files:
            print("Error: No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        print(f"Image filename: {image_file.filename}")
        
        if image_file.filename == '':
            print("Error: No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        # Preprocess image
        print("Preprocessing image...")
        img_tensor = preprocess_image_3d(image_file)
        
        # Make predictions with all models
        print("Making predictions with all models...")
        results = []
        
        for model_key in models.keys():
            result = predict_with_model(model_key, img_tensor, threshold=CANCER_THRESHOLD)
            results.append(result)
            
            if 'error' not in result:
                print(f"✓ {result['model_name']}: {result['prediction']} ({result['confidence']:.2f}%)")
            else:
                print(f"✗ {result['model_name']}: {result['error']}")
        
        print(f"Predictions completed successfully")
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'total_models': len(MODELS_CONFIG),
        'device': str(device),
        'models': [MODELS_CONFIG[k]['display_name'] for k in models.keys()]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Lung Cancer Detection System")
    print("Multi-Model AI Platform")
    print("="*60)
    print(f"Models loaded: {len(models)}/{len(MODELS_CONFIG)}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
