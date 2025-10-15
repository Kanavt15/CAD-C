#!/usr/bin/env python3
"""
Threshold Optimization for Lung Cancer Detection Models
========================================================

This script finds optimal threshold values for each model in the ensemble
by testing different threshold values and evaluating performance metrics.

Author: Enhanced Lung Cancer Detection System
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import json
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# Configuration
# ===========================

BASE_DIR = Path(__file__).parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations
MODEL_CONFIGS = {
    'LUNA16-DenseNet': {
        'path': 'models_densenet/densenet169_luna16_real_best.pth',
        'description': 'LUNA16-trained DenseNet-169 (F1: 0.8071, Real Data, Primary Model)',
        'color': '#1f77b4',
        'weight': 0.5
    },
    'ResNet-101': {
        'path': 'models_resnet101/best_resnet101_model.pth',
        'description': 'Fine-tuned deep residual network with advanced augmentation',
        'color': '#ff7f0e',
        'weight': 0.3
    },
    'EfficientNet-B0': {
        'path': 'models_efficientnet/best_efficientnet_model.pth',
        'description': 'Fine-tuned efficient compound scaling architecture',
        'color': '#2ca02c',
        'weight': 0.2
    }
}

# Test images with known ground truth labels
TEST_IMAGES = [
    # Main directory images
    {'path': 'healthy.jpg', 'label': 0, 'description': 'Healthy lung tissue'},
    {'path': 'download.jpg', 'label': 1, 'description': 'Suspected nodule case'},
    {'path': 'images.jpg', 'label': 1, 'description': 'Medical scan - suspected cancer'},
    {'path': 'unhealty.png', 'label': 1, 'description': 'Unhealthy lung tissue'},
    
    # Test images directory
    {'path': 'test_images/test_nodule_64x64.png', 'label': 1, 'description': 'Test nodule 64x64'},
    {'path': 'test_images/test_nodule_256x256.png', 'label': 1, 'description': 'Test nodule 256x256'}
]

# Threshold range to test
THRESHOLD_RANGE = np.arange(0.1, 0.95, 0.05)

# ===========================
# Model Architectures (copied from inference_ensemble.py)
# ===========================

class ResNet101LungCancer(nn.Module):
    """ResNet-101 with enhanced classifier for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.6):
        super(ResNet101LungCancer, self).__init__()
        
        # Load ResNet-101
        from torchvision import models
        if pretrained:
            self.resnet = models.resnet101(pretrained=True)
        else:
            self.resnet = models.resnet101(pretrained=False)
        
        # Remove the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Enhanced classifier with batch normalization and dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.resnet(x)
        output = self.classifier(features)
        return output


class EfficientNetLungCancer(nn.Module):
    """EfficientNet-B0 with enhanced classifier for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.5):
        super(EfficientNetLungCancer, self).__init__()
        
        # Load EfficientNet-B0
        from torchvision import models
        if pretrained:
            self.efficientnet = models.efficientnet_b0(pretrained=True)
        else:
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # Get number of features from EfficientNet
        num_features = self.efficientnet.classifier[1].in_features
        
        # Replace classifier
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.efficientnet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.efficientnet(x)


class DenseNet169LungCancer(nn.Module):
    """DenseNet-169 with enhanced classifier for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.5):
        super(DenseNet169LungCancer, self).__init__()
        
        # Load DenseNet-169
        from torchvision import models
        if pretrained:
            self.densenet = models.densenet169(pretrained=True)
        else:
            self.densenet = models.densenet169(pretrained=False)
        
        # Get number of features
        num_features = self.densenet.classifier.in_features
        
        # Enhanced classifier
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
    
    # Initialize model
    if model_name == 'ResNet-101':
        model = ResNet101LungCancer(pretrained=False, num_classes=2)
    elif model_name == 'EfficientNet-B0':
        model = EfficientNetLungCancer(pretrained=False, num_classes=2)
    elif model_name in ['DenseNet-169', 'LUNA16-DenseNet']:
        model = DenseNet169LungCancer(pretrained=False, num_classes=2)
    else:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    # Load weights
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
        if Path(image_path).suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            print(f"❌ Unsupported image format: {image_path}")
            return None
        
        # Normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Resize to 64x64 (model input size)
        image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        
        # Create 3-channel patch
        if len(image_resized.shape) == 2:
            patch = np.stack([image_resized] * 3, axis=0)
        else:
            patch = image_resized.transpose(2, 0, 1)
        
        # Convert to tensor
        patch_tensor = torch.FloatTensor(patch)
        
        return patch_tensor
        
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


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate performance metrics"""
    
    if len(set(y_true)) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'specificity': 0.0
        }
    
    # Basic metrics
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def optimize_threshold_for_model(model, model_name, test_data, device):
    """Find optimal threshold for a single model"""
    
    print(f"\n🔍 Optimizing threshold for {model_name}...")
    
    results = []
    all_probs = []
    all_labels = []
    
    # Collect predictions for all test images
    for test_item in test_data:
        patch_tensor = preprocess_image(test_item['path'])
        if patch_tensor is not None:
            _, prob = predict_with_threshold(model, patch_tensor, device, 0.5)  # Dummy threshold
            all_probs.append(prob)
            all_labels.append(test_item['label'])
    
    if not all_probs:
        print(f"❌ No valid predictions for {model_name}")
        return None
    
    # Test different thresholds
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in THRESHOLD_RANGE:
        # Make predictions with current threshold
        predictions = [1 if prob > threshold else 0 for prob in all_probs]
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, predictions, all_probs)
        
        results.append({
            'threshold': threshold,
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy'],
            'specificity': metrics['specificity']
        })
        
        # Track best F1 score
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
    
    print(f"✅ Best threshold for {model_name}: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return {
        'model_name': model_name,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'results': results,
        'probabilities': all_probs,
        'labels': all_labels
    }


def visualize_threshold_optimization(optimization_results, save_path=None):
    """Create comprehensive visualization of threshold optimization"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    models = list(optimization_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. F1-Score vs Threshold for all models
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        thresholds = [r['threshold'] for r in results['results']]
        f1_scores = [r['f1_score'] for r in results['results']]
        ax1.plot(thresholds, f1_scores, 'o-', color=colors[i], linewidth=2, 
                label=f"{model_name} (Best: {results['best_threshold']:.2f})", markersize=4)
        # Mark best threshold
        best_idx = np.argmax(f1_scores)
        ax1.plot(thresholds[best_idx], f1_scores[best_idx], 'o', color=colors[i], 
                markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # 2. Precision vs Threshold
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        thresholds = [r['threshold'] for r in results['results']]
        precisions = [r['precision'] for r in results['results']]
        ax2.plot(thresholds, precisions, 'o-', color=colors[i], linewidth=2, 
                label=model_name, markersize=4)
    
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # 3. Recall vs Threshold
    ax3 = fig.add_subplot(gs[1, :2])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        thresholds = [r['threshold'] for r in results['results']]
        recalls = [r['recall'] for r in results['results']]
        ax3.plot(thresholds, recalls, 'o-', color=colors[i], linewidth=2, 
                label=model_name, markersize=4)
    
    ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax3.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    # 4. Accuracy vs Threshold
    ax4 = fig.add_subplot(gs[1, 2:])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        thresholds = [r['threshold'] for r in results['results']]
        accuracies = [r['accuracy'] for r in results['results']]
        ax4.plot(thresholds, accuracies, 'o-', color=colors[i], linewidth=2, 
                label=model_name, markersize=4)
    
    ax4.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    # 5. ROC Curves
    ax5 = fig.add_subplot(gs[2, :2])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        if len(set(results['labels'])) > 1:  # Need both classes for ROC
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            ax5.plot(fpr, tpr, color=colors[i], linewidth=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax5.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax5.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Precision-Recall Curves
    ax6 = fig.add_subplot(gs[2, 2:])
    for i, (model_name, results) in enumerate(optimization_results.items()):
        if len(set(results['labels'])) > 1:  # Need both classes for PR curve
            precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])
            pr_auc = auc(recall, precision)
            ax6.plot(recall, precision, color=colors[i], linewidth=2, 
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
    
    ax6.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax6.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle('Threshold Optimization Analysis for Lung Cancer Detection Models', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main threshold optimization function"""
    
    print("=" * 80)
    print("         🔍 THRESHOLD OPTIMIZATION FOR LUNG CANCER DETECTION")
    print("=" * 80)
    
    # Check if test images exist
    available_images = []
    for test_item in TEST_IMAGES:
        if Path(test_item['path']).exists():
            available_images.append(test_item)
        else:
            print(f"⚠️  Test image not found: {test_item['path']}")
    
    if not available_images:
        print("❌ No test images found. Please ensure test images are available.")
        return
    
    print(f"📋 Found {len(available_images)} test images for optimization")
    
    # Load all models
    models = {}
    for model_name, config in MODEL_CONFIGS.items():
        model = load_model(model_name, config['path'], DEVICE)
        if model is not None:
            models[model_name] = model
    
    if not models:
        print("❌ No models could be loaded")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # Optimize thresholds for each model
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
    
    # Save results to JSON
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
    
    # Create visualization
    if optimization_results:
        viz_path = BASE_DIR / 'threshold_optimization_analysis.png'
        visualize_threshold_optimization(optimization_results, viz_path)
    
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
            print(f"        'color': '{config['color']}',")
            print(f"        'weight': {config['weight']},")
            print(f"        'threshold': {optimal_thresholds[model_name]:.2f}")
            print("    },")
    print("}")
    
    print("\n✅ Threshold optimization completed successfully!")


if __name__ == "__main__":
    main()