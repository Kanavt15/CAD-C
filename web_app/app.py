"""
Lung Cancer Detection Web Application
Flask backend using Improved 3D CNN with Residual + SE blocks
High-performance model with 83% accuracy
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

# Import model architecture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models_3d_cnn'))
from model_architecture import ImprovedCNN3D_Nodule_Detector

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_DIR = 'models_3d_cnn'
MODEL_PATH = os.path.join(MODELS_DIR, 'best_improved_3d_cnn_model.pth')
MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'model_info.json')

# Model configuration
PATCH_SIZE = 64
HU_MIN = -1000
HU_MAX = 400
# Threshold optimized based on model's precision-recall balance
# Model has 55% precision and 83% recall at default threshold
# Lowering threshold increases sensitivity for cancer detection
CANCER_THRESHOLD = 0.32  # 32% threshold - optimized for medical use (better to flag for review than miss)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enable debug mode
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration - Force CPU due to CUDA compatibility issues
device = torch.device('cpu')
print(f"Using device: {device} (CUDA disabled due to compatibility issues)")

# Load model info
model_info = {}
if os.path.exists(MODEL_INFO_PATH):
    with open(MODEL_INFO_PATH, 'r') as f:
        model_info = json.load(f)
    print("Model Information:")
    print(f"  Name: {model_info.get('model_name', 'Unknown')}")
    print(f"  Test Accuracy: {model_info.get('test_accuracy', 0):.2f}%")
    print(f"  Test F1 Score: {model_info.get('test_f1_score', 0):.4f}")
    print(f"  Parameters: {model_info.get('parameters', 0):,}")

# Load model
model = None

def load_model():
    """Load the improved 3D CNN model"""
    global model
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        print(f"Loading model from: {MODEL_PATH}")
        
        # Initialize model
        model = ImprovedCNN3D_Nodule_Detector(in_channels=1, num_classes=2)
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
            print(f"  Validation F1: {checkpoint.get('val_f1', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print("✓ Improved 3D CNN model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Load model on startup
model_loaded = load_model()

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

def predict_3d_cnn(img_tensor, threshold=CANCER_THRESHOLD):
    """
    Make prediction with the 3D CNN model
    
    Args:
        img_tensor: Input 3D image tensor (1, 1, D, H, W)
        threshold: Probability threshold for cancerous classification
    """
    try:
        print(f"Input tensor shape: {img_tensor.shape}")
        print(f"Threshold: {threshold * 100}%")
        
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get probabilities
            non_cancerous_prob = probabilities[0][0].item()
            cancerous_prob = probabilities[0][1].item()
            
            # Multi-level classification with uncertainty zone
            # 0-25%: Non-Cancerous (confident)
            # 25-32%: Suspicious/Uncertain (possible cancer - needs further review)
            # 32%+: Cancerous (high probability)
            
            if cancerous_prob >= threshold:
                predicted_class = 1
                prediction_label = 'Cancerous'
                confidence = cancerous_prob * 100
            elif cancerous_prob >= 0.25:
                predicted_class = 2  # Suspicious/uncertain
                prediction_label = 'Suspicious - Possible Cancer (Needs Review)'
                confidence = cancerous_prob * 100
            else:
                predicted_class = 0
                prediction_label = 'Non-Cancerous'
                confidence = non_cancerous_prob * 100
            
            result = {
                'model': 'Improved 3D CNN (Residual + SE)',
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {
                    'non_cancerous': float(non_cancerous_prob * 100),
                    'cancerous': float(cancerous_prob * 100)
                },
                'threshold': float(threshold * 100),
                'warning': 'Further medical review recommended' if predicted_class == 2 else None,
                'model_info': {
                    'accuracy': model_info.get('test_accuracy', 0),
                    'f1_score': model_info.get('test_f1_score', 0),
                    'precision': model_info.get('test_precision', 0),
                    'recall': model_info.get('test_recall', 0)
                }
            }
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Cancerous probability: {cancerous_prob * 100:.2f}%")
            print(f"Non-Cancerous probability: {non_cancerous_prob * 100:.2f}%")
            print(f"Threshold used: {threshold * 100:.1f}%")
            print(f"Raw logits: {outputs[0].tolist()}")
            
            return result
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': 'Improved 3D CNN (Residual + SE)',
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
    """Return model information"""
    return jsonify({
        'models': ['improved_3d_cnn'],
        'count': 1,
        'active_model': {
            'name': 'Improved 3D CNN (Residual + SE)',
            'accuracy': model_info.get('test_accuracy', 0),
            'f1_score': model_info.get('test_f1_score', 0),
            'parameters': model_info.get('parameters', 0)
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    print("\n" + "="*60)
    print("Received prediction request")
    
    try:
        # Check if model is loaded
        if model is None or not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
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
        
        # Make prediction
        print("Making prediction...")
        result = predict_3d_cnn(img_tensor, threshold=CANCER_THRESHOLD)
        
        if 'error' in result:
            print(f"Error in prediction: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        print(f"Prediction successful: {result['prediction']}")
        
        return jsonify({
            'success': True,
            'results': [result]  # Keep as list for compatibility with frontend
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
        'model_loaded': model_loaded,
        'device': str(device),
        'model_accuracy': model_info.get('test_accuracy', 0)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Lung Cancer Detection System")
    print("Improved 3D CNN with Residual + SE Blocks")
    print("="*60)
    print(f"Model loaded: {model_loaded}")
    print(f"Device: {device}")
    if model_info:
        print(f"Test Accuracy: {model_info.get('test_accuracy', 0):.2f}%")
        print(f"Test F1 Score: {model_info.get('test_f1_score', 0):.4f}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
