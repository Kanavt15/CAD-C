"""
Lung Cancer Detection Web Application
Flask backend using multiple 3D CNN models:
1. Improved 3D CNN with Residual + SE blocks (83% accuracy)
2. DenseNet3D with Multi-Head Attention (95.73% accuracy, 77.75% F1)
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
from densenet3d_architecture import DenseNet3D_Attention

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_DIR = 'models_3d_cnn'
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'best_improved_3d_cnn_model.pth')
RESNET_INFO_PATH = os.path.join(MODELS_DIR, 'model_info.json')
DENSENET_MODEL_PATH = 'densenet3d_attention.pth'
DENSENET_INFO_PATH = 'densenet_model_info.json'

# Model configuration
PATCH_SIZE = 64
HU_MIN = -1000
HU_MAX = 400
# Thresholds for each model
RESNET_THRESHOLD = 0.32  # 32% - optimized for medical use
DENSENET_THRESHOLD = 0.32  # 32% - same as ResNet for consistent sensitivity

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enable debug mode
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration - Force CPU due to CUDA compatibility issues
device = torch.device('cpu')
print(f"Using device: {device} (CUDA disabled due to compatibility issues)")

# Load model info for both models
resnet_model_info = {}
densenet_model_info = {}

if os.path.exists(RESNET_INFO_PATH):
    with open(RESNET_INFO_PATH, 'r') as f:
        resnet_model_info = json.load(f)
    print("ResNet3D Model Information:")
    print(f"  Name: {resnet_model_info.get('model_name', 'Unknown')}")
    print(f"  Test Accuracy: {resnet_model_info.get('test_accuracy', 0):.2f}%")
    print(f"  Test F1 Score: {resnet_model_info.get('test_f1_score', 0):.4f}")
    print(f"  Parameters: {resnet_model_info.get('parameters', 0):,}")

if os.path.exists(DENSENET_INFO_PATH):
    with open(DENSENET_INFO_PATH, 'r') as f:
        densenet_model_info = json.load(f)
    
    # Extract test metrics from the structure
    test_metrics = densenet_model_info.get('test_metrics', {})
    print("\nDenseNet3D Model Information:")
    print(f"  Name: {densenet_model_info.get('model', 'DenseNet3D-Attention')}")
    print(f"  Test Accuracy: {test_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Test F1 Score: {test_metrics.get('f1', 0):.4f}")
    print(f"  Test Precision: {test_metrics.get('precision', 0):.4f}")
    print(f"  Test Recall: {test_metrics.get('recall', 0):.4f}")
    print(f"  Parameters: {densenet_model_info.get('total_params', 0):,}")

# Load models
resnet_model = None
densenet_model = None

def load_resnet_model():
    """Load the improved 3D CNN ResNet model"""
    global resnet_model
    
    try:
        if not os.path.exists(RESNET_MODEL_PATH):
            raise FileNotFoundError(f"ResNet model file not found: {RESNET_MODEL_PATH}")
        
        print(f"\nLoading ResNet3D model from: {RESNET_MODEL_PATH}")
        
        # Initialize model
        resnet_model = ImprovedCNN3D_Nodule_Detector(in_channels=1, num_classes=2)
        
        # Load checkpoint
        checkpoint = torch.load(RESNET_MODEL_PATH, map_location=device, weights_only=False)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            resnet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
            print(f"  Validation F1: {checkpoint.get('val_f1', 0):.4f}")
        else:
            resnet_model.load_state_dict(checkpoint)
        
        resnet_model.to(device)
        resnet_model.eval()
        
        print("✓ ResNet3D model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading ResNet model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_densenet_model():
    """Load the DenseNet3D with Attention model"""
    global densenet_model
    
    try:
        if not os.path.exists(DENSENET_MODEL_PATH):
            raise FileNotFoundError(f"DenseNet model file not found: {DENSENET_MODEL_PATH}")
        
        print(f"\nLoading DenseNet3D model from: {DENSENET_MODEL_PATH}")
        
        # Initialize model with same config as training
        densenet_model = DenseNet3D_Attention(
            in_channels=1,
            num_classes=2,
            growth_rate=16,
            num_layers=[4, 4, 4],
            num_heads=4,
            drop_path_rate=0.1
        )
        
        # Load checkpoint
        checkpoint = torch.load(DENSENET_MODEL_PATH, map_location=device, weights_only=False)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            densenet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            val_metrics = checkpoint.get('val_metrics', {})
            print(f"  Validation accuracy: {val_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"  Validation F1: {val_metrics.get('f1', 0):.4f}")
        else:
            densenet_model.load_state_dict(checkpoint)
        
        densenet_model.to(device)
        densenet_model.eval()
        
        print("✓ DenseNet3D model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in densenet_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading DenseNet model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Load models on startup
resnet_loaded = load_resnet_model()
densenet_loaded = load_densenet_model()

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

def predict_resnet(img_tensor, threshold=RESNET_THRESHOLD):
    """
    Make prediction with the ResNet3D model
    
    Args:
        img_tensor: Input 3D image tensor (1, 1, D, H, W)
        threshold: Probability threshold for cancerous classification
    """
    try:
        print(f"\n[ResNet] Input tensor shape: {img_tensor.shape}")
        print(f"[ResNet] Threshold: {threshold * 100}%")
        
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = resnet_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get probabilities
            non_cancerous_prob = probabilities[0][0].item()
            cancerous_prob = probabilities[0][1].item()
            
            # Multi-level classification with uncertainty zone
            # Logic: Compare cancerous_prob against threshold
            if cancerous_prob >= threshold:
                # High cancer probability - predict cancerous
                predicted_class = 1
                prediction_label = 'Cancerous'
                confidence = cancerous_prob * 100  # Show how confident we are it IS cancer
            elif cancerous_prob >= 0.25:
                # Moderate cancer probability - uncertain zone
                predicted_class = 2
                prediction_label = 'Suspicious - Possible Cancer (Needs Review)'
                confidence = cancerous_prob * 100  # Show the cancer probability
            else:
                # Low cancer probability - predict non-cancerous
                predicted_class = 0
                prediction_label = 'Non-Cancerous'
                confidence = non_cancerous_prob * 100  # Show how confident we are it's NOT cancer
            
            result = {
                'model': 'ResNet3D with SE Blocks',
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {
                    'non_cancerous': float(non_cancerous_prob * 100),
                    'cancerous': float(cancerous_prob * 100)
                },
                'threshold': float(threshold * 100),
                'warning': 'Further medical review recommended' if predicted_class == 2 else None,
                'model_info': {
                    'accuracy': resnet_model_info.get('test_accuracy', 0),
                    'f1_score': resnet_model_info.get('test_f1_score', 0),
                    'precision': resnet_model_info.get('test_precision', 0),
                    'recall': resnet_model_info.get('test_recall', 0)
                }
            }
            
            print(f"[ResNet] Prediction: {result['prediction']}")
            print(f"[ResNet] Confidence: {result['confidence']:.2f}%")
            print(f"[ResNet] Cancerous probability: {cancerous_prob * 100:.2f}%")
            
            return result
            
    except Exception as e:
        print(f"[ResNet] Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': 'ResNet3D with SE Blocks',
            'error': str(e)
        }


def predict_densenet(img_tensor, threshold=DENSENET_THRESHOLD):
    """
    Make prediction with the DenseNet3D model
    
    Args:
        img_tensor: Input 3D image tensor (1, 1, D, H, W)
        threshold: Probability threshold for cancerous classification
    """
    try:
        print(f"\n[DenseNet] Input tensor shape: {img_tensor.shape}")
        print(f"[DenseNet] Threshold: {threshold * 100}%")
        
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = densenet_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get probabilities
            non_cancerous_prob = probabilities[0][0].item()
            cancerous_prob = probabilities[0][1].item()
            
            # Multi-level classification with uncertainty zone
            # Logic: Compare cancerous_prob against threshold (32% - same as ResNet)
            if cancerous_prob >= threshold:
                # High cancer probability - predict cancerous
                predicted_class = 1
                prediction_label = 'Cancerous'
                confidence = cancerous_prob * 100  # Show how confident we are it IS cancer
            elif cancerous_prob >= 0.25:
                # Moderate cancer probability - uncertain zone
                predicted_class = 2
                prediction_label = 'Suspicious - Possible Cancer (Needs Review)'
                confidence = cancerous_prob * 100  # Show the cancer probability
            else:
                # Low cancer probability - predict non-cancerous
                predicted_class = 0
                prediction_label = 'Non-Cancerous'
                confidence = non_cancerous_prob * 100  # Show how confident we are it's NOT cancer
            
            test_metrics = densenet_model_info.get('test_metrics', {})
            result = {
                'model': 'DenseNet3D with Multi-Head Attention',
                'prediction': prediction_label,
                'confidence': float(confidence),
                'probabilities': {
                    'non_cancerous': float(non_cancerous_prob * 100),
                    'cancerous': float(cancerous_prob * 100)
                },
                'threshold': float(threshold * 100),
                'warning': 'Further medical review recommended' if predicted_class == 2 else None,
                'model_info': {
                    'accuracy': test_metrics.get('accuracy', 0) * 100,
                    'f1_score': test_metrics.get('f1', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0)
                }
            }
            
            print(f"[DenseNet] Prediction: {result['prediction']}")
            print(f"[DenseNet] Confidence: {result['confidence']:.2f}%")
            print(f"[DenseNet] Cancerous probability: {cancerous_prob * 100:.2f}%")
            
            return result
            
    except Exception as e:
        print(f"[DenseNet] Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': 'DenseNet3D with Multi-Head Attention',
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
    models_list = []
    
    if resnet_loaded:
        models_list.append({
            'id': 'resnet3d',
            'name': 'ResNet3D with SE Blocks',
            'accuracy': resnet_model_info.get('test_accuracy', 0),
            'f1_score': resnet_model_info.get('test_f1_score', 0),
            'parameters': resnet_model_info.get('parameters', 0),
            'description': 'Improved 3D CNN with Residual connections and Squeeze-Excitation blocks'
        })
    
    if densenet_loaded:
        test_metrics = densenet_model_info.get('test_metrics', {})
        models_list.append({
            'id': 'densenet3d',
            'name': 'DenseNet3D with Multi-Head Attention',
            'accuracy': test_metrics.get('accuracy', 0) * 100,
            'f1_score': test_metrics.get('f1', 0),
            'parameters': densenet_model_info.get('total_params', 0),
            'description': 'Dense connections with multi-head self-attention for spatial relationships'
        })
    
    return jsonify({
        'models': models_list,
        'count': len(models_list)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests - supports both models"""
    print("\n" + "="*60)
    print("Received prediction request")
    
    try:
        # Check if at least one model is loaded
        if not resnet_loaded and not densenet_loaded:
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
        
        # Get selected models (default to all available)
        selected_models = request.form.get('models', 'all')
        print(f"Selected models: {selected_models}")
        
        # Preprocess image
        print("Preprocessing image...")
        img_tensor = preprocess_image_3d(image_file)
        
        # Make predictions with selected models
        results = []
        
        if (selected_models == 'all' or 'resnet3d' in selected_models) and resnet_loaded:
            print("\nPredicting with ResNet3D...")
            resnet_result = predict_resnet(img_tensor)
            if 'error' not in resnet_result:
                results.append(resnet_result)
        
        if (selected_models == 'all' or 'densenet3d' in selected_models) and densenet_loaded:
            print("\nPredicting with DenseNet3D...")
            densenet_result = predict_densenet(img_tensor)
            if 'error' not in densenet_result:
                results.append(densenet_result)
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'No predictions could be generated'
            }), 500
        
        print(f"\nGenerated {len(results)} prediction(s)")
        
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
        'models': {
            'resnet3d': {
                'loaded': resnet_loaded,
                'accuracy': resnet_model_info.get('test_accuracy', 0) if resnet_loaded else 0
            },
            'densenet3d': {
                'loaded': densenet_loaded,
                'accuracy': densenet_model_info.get('test_metrics', {}).get('accuracy', 0) * 100 if densenet_loaded else 0
            }
        },
        'device': str(device)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Lung Cancer Detection System")
    print("Multi-Model Ensemble System")
    print("="*60)
    print(f"ResNet3D loaded: {resnet_loaded}")
    if resnet_loaded and resnet_model_info:
        print(f"  - Test Accuracy: {resnet_model_info.get('test_accuracy', 0):.2f}%")
        print(f"  - Test F1 Score: {resnet_model_info.get('test_f1_score', 0):.4f}")
    
    print(f"\nDenseNet3D loaded: {densenet_loaded}")
    if densenet_loaded and densenet_model_info:
        test_metrics = densenet_model_info.get('test_metrics', {})
        print(f"  - Test Accuracy: {test_metrics.get('accuracy', 0)*100:.2f}%")
        print(f"  - Test F1 Score: {test_metrics.get('f1', 0):.4f}")
    
    print(f"\nDevice: {device}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
