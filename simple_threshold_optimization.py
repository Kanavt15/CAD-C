#!/usr/bin/env python3
"""
Simple Threshold Optimization for Lung Cancer Detection Models
==============================================================

This script finds optimal threshold values for each model by testing different
thresholds and evaluating performance on test images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ===========================
# Configuration
# ===========================

BASE_DIR = Path(__file__).parent
DEVICE = torch.device('cpu')  # Force CPU to avoid CUDA compatibility issues

# Model configurations
MODEL_CONFIGS = {
    'LUNA16-DenseNet': {
        'path': 'models_densenet/densenet169_luna16_real_best.pth',
        'description': 'LUNA16-trained DenseNet-169 (F1: 0.8071, Real Data, Primary Model)',
        'weight': 0.5
    },
    'ResNet-101': {
        'path': 'models_resnet101/best_resnet101_model.pth',
        'description': 'Fine-tuned deep residual network with advanced augmentation',
        'weight': 0.3
    },
    'EfficientNet-B0': {
        'path': 'models_efficientnet/best_efficientnet_model.pth',
        'description': 'Fine-tuned efficient compound scaling architecture',
        'weight': 0.2
    }
}

# Test images with ground truth labels
TEST_IMAGES = [
    {'path': 'healthy.jpg', 'label': 0, 'description': 'Healthy lung tissue'},
    {'path': 'download.jpg', 'label': 1, 'description': 'Suspected nodule case'},
    {'path': 'images.jpg', 'label': 1, 'description': 'Medical scan - suspected cancer'},
    {'path': 'unhealty.png', 'label': 1, 'description': 'Unhealthy lung tissue'},
    {'path': 'test_images/test_nodule_64x64.png', 'label': 1, 'description': 'Test nodule 64x64'},
    {'path': 'test_images/test_nodule_256x256.png', 'label': 1, 'description': 'Test nodule 256x256'}
]

# Threshold range to test
THRESHOLD_RANGE = np.arange(0.1, 0.95, 0.05)

# ===========================
# Model Architectures
# ===========================

class ResNet101LungCancer(nn.Module):
    """ResNet-101 model for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.5):
        super(ResNet101LungCancer, self).__init__()
        from torchvision import models
        self.resnet = models.resnet101(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class EfficientNetLungCancer(nn.Module):
    """EfficientNet-B0 model for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.3):
        super(EfficientNetLungCancer, self).__init__()
        from torchvision import models
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


class DenseNet169LungCancer(nn.Module):
    """DenseNet-169 model for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.4):
        super(DenseNet169LungCancer, self).__init__()
        
        # Load pretrained DenseNet-169
        from torchvision import models
        self.densenet = models.densenet169(pretrained=pretrained)
        
        # Get feature dimension (DenseNet-169 has 1664 features)
        num_features = self.densenet.classifier.in_features
        
        # Custom medical imaging classification head
        self.densenet.classifier = nn.Sequential(
            # Batch normalization for stability
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout),
            
            # First reduction layer
            nn.Linear(num_features, num_features // 2),  # 1664 -> 832
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 2),
            nn.Dropout(dropout * 0.7),
            
            # Second reduction layer
            nn.Linear(num_features // 2, num_features // 4),  # 832 -> 416
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 4),
            nn.Dropout(dropout * 0.5),
            
            # Final classification layer
            nn.Linear(num_features // 4, num_classes)  # 416 -> 2
        )
    
    def forward(self, x):
        return self.densenet(x)


# ===========================
# Utility Functions
# ===========================

def load_model(model_name, model_path, device):
    """Load a trained model"""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    if model_name == 'ResNet-101':
        model = ResNet101LungCancer(pretrained=False, num_classes=2)
    elif model_name == 'EfficientNet-B0':
        model = EfficientNetLungCancer(pretrained=False, num_classes=2)
    elif model_name in ['DenseNet-169', 'LUNA16-DenseNet']:
        model = DenseNet169LungCancer(pretrained=False, num_classes=2)
    else:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"✅ {model_name} loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Simple resize using PIL (avoiding cv2)
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray((image * 255).astype(np.uint8))
        resized_image = pil_image.resize((64, 64), PILImage.LANCZOS)
        image_resized = np.array(resized_image, dtype=np.float32) / 255.0
        
        # Create 3-channel patch
        if len(image_resized.shape) == 2:
            patch = np.stack([image_resized] * 3, axis=0)
        else:
            patch = image_resized.transpose(2, 0, 1)
        
        return torch.FloatTensor(patch)
        
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None


def predict_with_threshold(model, patch_tensor, device, threshold):
    """Make prediction with specific threshold"""
    with torch.no_grad():
        patch_tensor = patch_tensor.unsqueeze(0).to(device)
        outputs = model(patch_tensor)
        probs = F.softmax(outputs, dim=1)
        prob_cancer = probs[0, 1].item()
        prediction = 1 if prob_cancer > threshold else 0
    
    return prediction, prob_cancer


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    if len(set(y_true)) < 2:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': 0.0}
    
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy}


def optimize_threshold_for_model(model, model_name, test_data, device):
    """Find optimal threshold for a single model"""
    print(f"\n🔍 Optimizing threshold for {model_name}...")
    
    all_probs = []
    all_labels = []
    
    # Collect predictions for all test images
    for test_item in test_data:
        patch_tensor = preprocess_image(test_item['path'])
        if patch_tensor is not None:
            _, prob = predict_with_threshold(model, patch_tensor, device, 0.5)
            all_probs.append(prob)
            all_labels.append(test_item['label'])
            print(f"   📷 {test_item['path']}: Cancer prob = {prob:.3f}, Label = {test_item['label']}")
    
    if not all_probs:
        print(f"❌ No valid predictions for {model_name}")
        return None
    
    # Test different thresholds
    best_threshold = 0.5
    best_f1 = 0.0
    results = []
    
    for threshold in THRESHOLD_RANGE:
        predictions = [1 if prob > threshold else 0 for prob in all_probs]
        metrics = calculate_metrics(all_labels, predictions)
        
        results.append({
            'threshold': threshold,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy']
        })
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
        
        print(f"   Threshold {threshold:.2f}: F1={metrics['f1_score']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    print(f"✅ Best threshold for {model_name}: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return {
        'model_name': model_name,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'results': results
    }


def main():
    """Main threshold optimization function"""
    print("=" * 80)
    print("         🔍 THRESHOLD OPTIMIZATION FOR LUNG CANCER DETECTION")
    print("=" * 80)
    
    # Check test images
    available_images = []
    for test_item in TEST_IMAGES:
        if Path(test_item['path']).exists():
            available_images.append(test_item)
        else:
            print(f"⚠️  Test image not found: {test_item['path']}")
    
    if not available_images:
        print("❌ No test images found.")
        return
    
    print(f"📋 Found {len(available_images)} test images for optimization")
    
    # Load models
    models = {}
    for model_name, config in MODEL_CONFIGS.items():
        model = load_model(model_name, config['path'], DEVICE)
        if model is not None:
            models[model_name] = model
    
    if not models:
        print("❌ No models could be loaded")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # Optimize thresholds
    optimization_results = {}
    optimal_thresholds = {}
    
    for model_name, model in models.items():
        result = optimize_threshold_for_model(model, model_name, available_images, DEVICE)
        if result:
            optimization_results[model_name] = result
            optimal_thresholds[model_name] = result['best_threshold']
    
    # Print summary
    print("\n" + "=" * 80)
    print("                      🎯 OPTIMIZATION RESULTS")
    print("=" * 80)
    
    for model_name, threshold in optimal_thresholds.items():
        f1_score = optimization_results[model_name]['best_f1']
        print(f"✅ {model_name:<20}: Threshold = {threshold:.2f}, F1-Score = {f1_score:.4f}")
    
    # Save results
    results_file = BASE_DIR / 'optimal_thresholds.json'
    with open(results_file, 'w') as f:
        json.dump({
            'optimal_thresholds': optimal_thresholds,
            'optimization_details': {
                model: {
                    'best_threshold': result['best_threshold'],
                    'best_f1': result['best_f1'],
                    'test_images_count': len(available_images)
                }
                for model, result in optimization_results.items()
            },
            'test_images': [img['path'] for img in available_images]
        }, f, indent=4)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate updated MODEL_CONFIGS code
    print("\n" + "=" * 80)
    print("              📝 UPDATED MODEL_CONFIGS FOR INFERENCE SCRIPT")
    print("=" * 80)
    
    print("MODEL_CONFIGS = {")
    for model_name, config in MODEL_CONFIGS.items():
        if model_name in optimal_thresholds:
            print(f"    '{model_name}': {{")
            print(f"        'path': '{config['path']}',")
            print(f"        'description': '{config['description']}',")
            print(f"        'weight': {config['weight']},")
            print(f"        'threshold': {optimal_thresholds[model_name]:.2f}")
            print("    },")
    print("}")
    
    print("\n✅ Threshold optimization completed successfully!")


if __name__ == "__main__":
    main()