"""
Lung Cancer Detection Web Application
Flask backend for serving DenseNet, EfficientNet, and ResNet models
"""

import os
import io
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
# Updated paths for reorganized structure
MODELS_DIR = '../models/models_efficientnet'
DENSENET_DIR = '../models/models_densenet'
RESNET_DIR = '../models/models_resnet101'

# Classification threshold
# Probability threshold for cancer classification
# If cancerous probability >= THRESHOLD -> Classified as Cancerous
# If cancerous probability < THRESHOLD -> Classified as Non-Cancerous
CANCER_THRESHOLD = 0.5  # 50% (Original default threshold)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enable debug mode
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration - Force CPU to avoid CUDA compatibility issues
device = torch.device('cpu')
print(f"Using device: {device}")
print(f"Cancer classification threshold: {CANCER_THRESHOLD * 100}%")

# Image preprocessing
def normalize_hu(image):
    """Normalize CT Hounsfield Units to 0-1 range"""
    MIN_HU = -1000
    MAX_HU = 400
    image = np.clip(image, MIN_HU, MAX_HU)
    image = (image - MIN_HU) / (MAX_HU - MIN_HU)
    return image.astype(np.float32)

def normalize_image(image):
    """Normalize any image to 0-1 range"""
    if image.max() > 1.0:
        # Assume it's in 0-255 or HU range
        if image.min() < 0:  # Likely HU values
            return normalize_hu(image)
        else:  # Likely 0-255 range
            return (image / 255.0).astype(np.float32)
    return image.astype(np.float32)

def prepare_densenet_patch(patch, target_size=224):
    """
    Prepare patch specifically for DenseNet-169
    Converts 3D patch to 3-channel ImageNet-normalized input
    """
    # Take middle slice if 3D
    if patch.ndim == 3:
        middle_slice = patch[patch.shape[0] // 2]
    else:
        middle_slice = patch
    
    # Normalize patch
    middle_slice = middle_slice.astype(np.float32)
    middle_slice = np.clip(middle_slice, 0, 1)
    
    # Convert to 3-channel by replicating
    patch_3ch = np.stack([middle_slice] * 3, axis=0)
    
    # Convert to tensor
    patch_tensor = torch.from_numpy(patch_3ch).float()
    
    # Resize to target size
    resize_transform = transforms.Resize((target_size, target_size), antialias=True)
    patch_tensor = resize_transform(patch_tensor)
    
    # Apply ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    patch_tensor = normalize(patch_tensor)
    
    return patch_tensor

# Standard image preprocessing for CNN models
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Model definitions
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet169(pretrained=False)
        num_ftrs = self.densenet.classifier.in_features
        # Custom medical imaging classification head
        self.densenet.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_ftrs // 2),
            nn.Dropout(dropout * 0.7),
            nn.Linear(num_ftrs // 2, num_ftrs // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_ftrs // 4),
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_ftrs // 4, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)

class ResNet101Model(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(ResNet101Model, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Load models
models_dict = {}

def load_models():
    """Load all three models"""
    global models_dict
    
    try:
        # Load EfficientNet
        efficientnet_path = os.path.join(MODELS_DIR, 'best_efficientnet_model.pth')
        if os.path.exists(efficientnet_path):
            efficientnet_model = EfficientNetModel(num_classes=2, dropout=0.3)
            checkpoint = torch.load(efficientnet_path, map_location=device, weights_only=False)
            # Handle checkpoint format (either full checkpoint or just state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                efficientnet_model.load_state_dict(checkpoint)
            efficientnet_model.to(device)
            efficientnet_model.eval()
            models_dict['efficientnet'] = efficientnet_model
            print("✓ EfficientNet model loaded successfully")
        else:
            print(f"⚠ EfficientNet model not found at {efficientnet_path}")
        
        # Load DenseNet
        densenet_path = os.path.join(DENSENET_DIR, 'best_densenet_model.pth')
        if os.path.exists(densenet_path):
            densenet_model = DenseNetModel(num_classes=2, dropout=0.4)
            checkpoint = torch.load(densenet_path, map_location=device, weights_only=False)
            # Handle checkpoint format (either full checkpoint or just state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                densenet_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                densenet_model.load_state_dict(checkpoint)
            densenet_model.to(device)
            densenet_model.eval()
            models_dict['densenet'] = densenet_model
            print("✓ DenseNet model loaded successfully")
        else:
            print(f"⚠ DenseNet model not found at {densenet_path}")
        
        # Load ResNet101
        resnet_path = os.path.join(RESNET_DIR, 'best_resnet101_model.pth')
        if os.path.exists(resnet_path):
            resnet_model = ResNet101Model(num_classes=2, dropout=0.5)
            checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
            # Handle checkpoint format (either full checkpoint or just state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                resnet_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                resnet_model.load_state_dict(checkpoint)
            resnet_model.to(device)
            resnet_model.eval()
            models_dict['resnet101'] = resnet_model
            print("✓ ResNet101 model loaded successfully")
        else:
            print(f"⚠ ResNet101 model not found at {resnet_path}")
        
        if not models_dict:
            print("⚠ WARNING: No models loaded!")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()

# Load models on startup
load_models()

def preprocess_image(image_file):
    """Preprocess uploaded image using inference_ensemble.py logic"""
    try:
        # Read image
        image = Image.open(image_file).convert('L')  # Convert to grayscale
        image_array = np.array(image).astype(np.float32)
        
        print(f"Original image shape: {image_array.shape}")
        print(f"Original value range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # Normalize the image
        image_array = normalize_image(image_array)
        
        print(f"Normalized value range: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # Resize to 64x64 if needed
        if image_array.shape[0] != 64 or image_array.shape[1] != 64:
            from scipy.ndimage import zoom
            zoom_factors = (64 / image_array.shape[0], 64 / image_array.shape[1])
            image_array = zoom(image_array, zoom_factors, order=1)
            print(f"Resized to: {image_array.shape}")
        
        # Create 3-slice stack (replicate for medical imaging)
        NUM_SLICES = 3
        patch = np.stack([image_array] * NUM_SLICES, axis=0)
        
        print(f"Final patch shape: {patch.shape}")
        
        # Convert to tensor
        img_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Add batch dimension
        
        return img_tensor
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_with_model(model, img_tensor, model_name, threshold=0.65):
    """
    Make prediction with a specific model
    
    Args:
        model: The neural network model
        img_tensor: Input image tensor
        model_name: Name of the model
        threshold: Probability threshold for cancerous classification (default: 0.65)
                  Above this threshold -> Cancerous
                  Below this threshold -> Non-Cancerous
    """
    try:
        print(f"    Input tensor shape for {model_name}: {img_tensor.shape}")
        
        with torch.no_grad():
            # Remove batch dimension for processing
            img_tensor_no_batch = img_tensor.squeeze(0)
            
            # Prepare input based on model type
            if 'densenet' in model_name.lower():
                # DenseNet needs 224x224 ImageNet-normalized input
                model_input = prepare_densenet_patch(img_tensor_no_batch.numpy(), target_size=224)
                model_input = model_input.unsqueeze(0).to(device)
            else:
                # CNN models use the standard patch
                model_input = img_tensor.to(device)
            
            print(f"    Model input shape: {model_input.shape}")
            
            outputs = model(model_input)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get probability of cancerous class
            cancerous_prob = probabilities[0][1].item()
            
            # Apply threshold: if cancerous probability >= threshold, classify as Cancerous
            predicted_class = 1 if cancerous_prob >= threshold else 0
            confidence = cancerous_prob * 100 if predicted_class == 1 else (1 - cancerous_prob) * 100
            
            result = {
                'model': model_name,
                'prediction': 'Cancerous' if predicted_class == 1 else 'Non-Cancerous',
                'confidence': float(confidence),
                'probabilities': {
                    'non_cancerous': float(probabilities[0][0].item() * 100),
                    'cancerous': float(probabilities[0][1].item() * 100)
                },
                'threshold': float(threshold * 100)
            }
            
            print(f"    Result: {result['prediction']} ({result['confidence']:.2f}%) [Threshold: {threshold*100}%]")
            return result
            
    except Exception as e:
        print(f"    Error in {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'error': str(e)
        }

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Return list of available models"""
    available = list(models_dict.keys())
    return jsonify({
        'models': available,
        'count': len(available)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    print("\n" + "="*60)
    print("Received prediction request")
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            print("Error: No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        print(f"Image filename: {image_file.filename}")
        
        if image_file.filename == '':
            print("Error: No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        # Get selected models
        selected_models = request.form.getlist('models[]')
        if not selected_models:
            selected_models = list(models_dict.keys())
        
        print(f"Selected models: {selected_models}")
        print(f"Available models: {list(models_dict.keys())}")
        
        # Preprocess image
        print("Preprocessing image...")
        img_tensor = preprocess_image(image_file)
        print(f"Image tensor shape: {img_tensor.shape}")
        
        # Make predictions with selected models
        results = []
        print("Making predictions...")
        for model_name in selected_models:
            if model_name in models_dict:
                print(f"  Predicting with {model_name}...")
                result = predict_with_model(models_dict[model_name], img_tensor, model_name, threshold=CANCER_THRESHOLD)
                results.append(result)
                print(f"  {model_name} result: {result.get('prediction', 'ERROR')}")
            else:
                print(f"Warning: Model '{model_name}' not found in loaded models")
        
        if not results:
            print("Error: No valid models found for prediction")
            return jsonify({
                'success': False,
                'error': 'No valid models found for prediction'
            }), 400
        
        print(f"Got {len(results)} results from models")
        
        # Calculate ensemble prediction if multiple models
        valid_results = [r for r in results if 'error' not in r]
        print(f"Valid results: {len(valid_results)}")
        
        if len(valid_results) > 1:
            print("Calculating ensemble prediction...")
            avg_cancerous = np.mean([r['probabilities']['cancerous'] for r in valid_results])
            avg_non_cancerous = np.mean([r['probabilities']['non_cancerous'] for r in valid_results])
            
            # Apply same threshold for ensemble prediction
            ensemble_pred = 'Cancerous' if avg_cancerous >= (CANCER_THRESHOLD * 100) else 'Non-Cancerous'
            ensemble_confidence = avg_cancerous if ensemble_pred == 'Cancerous' else avg_non_cancerous
            
            ensemble = {
                'model': 'ensemble',
                'prediction': ensemble_pred,
                'confidence': float(ensemble_confidence),
                'probabilities': {
                    'non_cancerous': float(avg_non_cancerous),
                    'cancerous': float(avg_cancerous)
                },
                'threshold': float(CANCER_THRESHOLD * 100)
            }
            results.append(ensemble)
            print(f"  Ensemble result: {ensemble_pred} (Cancerous: {avg_cancerous:.2f}%, Threshold: {CANCER_THRESHOLD*100}%)")
        
        print(f"Returning {len(results)} results")
        response = jsonify({
            'success': True,
            'results': results
        })
        return response
    
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
        'models_loaded': len(models_dict),
        'device': str(device)
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Lung Cancer Detection System")
    print("="*50)
    print(f"Models loaded: {len(models_dict)}")
    print(f"Available models: {list(models_dict.keys())}")
    print(f"Device: {device}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
