"""
🫁 LUNA16 Lung Cancer Detection - Multi-Model Inference Script

This script takes a CT scan and runs inference through ResNet-101, EfficientNet-B0, 
and VGG16 models to provide comprehensive predictions and ensemble results.

Supports multiple input formats:
1. LUNA16 dataset (series_uid + coordinates)
2. External image file (DICOM, .mhd, .npy, .nii, .nii.gz, PNG, JPG)
3. Direct numpy array

Usage:
    # From LUNA16 dataset
    python inference_ensemble.py --series_uid <SERIES_UID> --coord_x <X> --coord_y <Y> --coord_z <Z>
    
    # From external CT scan (3D volume)
    python inference_ensemble.py --image_path scan.mhd --coord_x <X> --coord_y <Y> --coord_z <Z>
    
    # From external image (2D/3D - will extract patch automatically)
    python inference_ensemble.py --image_path lung_patch.png
    
    # From numpy array
    python inference_ensemble.py --image_path ct_data.npy --is_3d
    
Examples:
    python inference_ensemble.py --series_uid 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260 --coord_x -56.08 --coord_y -67.85 --coord_z -311.92
    python inference_ensemble.py --image_path patient_scan.dcm --slice_idx 150 --center_y 200 --center_x 200
    python inference_ensemble.py --image_path nodule_patch.png
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

warnings.filterwarnings('ignore')

# ===========================
# Configuration
# ===========================

BASE_DIR = Path(r'e:\Kanav\Projects\CAD_C')
SUBSET_DIRS = [BASE_DIR / f'subset{i}' for i in range(10)]

MODEL_CONFIGS = {
    'ResNet-101': {
        'model_path': BASE_DIR / 'models_resnet101' / 'best_resnet101_model.pth',
        'color': '#FF6B6B',
        'description': 'Deep residual network with skip connections'
    },
    'EfficientNet-B0': {
        'model_path': BASE_DIR / 'models_efficientnet' / 'best_efficientnet_model.pth',
        'color': '#4ECDC4',
        'description': 'Efficient compound scaling architecture'
    },
    'VGG16': {
        'model_path': BASE_DIR / 'models_vgg16' / 'best_vgg16_model.pth',
        'color': '#95E1D3',
        'description': 'Classic deep convolutional network'
    }
}

PATCH_SIZE = 64
NUM_SLICES = 3
DEVICE = 'cpu'  # Force CPU to avoid CUDA compatibility issues

# ===========================
# Model Architectures
# ===========================

class ResNet101LungCancer(nn.Module):
    """ResNet-101 model for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.5):
        super(ResNet101LungCancer, self).__init__()
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


class VGG16LungCancer(nn.Module):
    """VGG16 model for lung cancer detection"""
    
    def __init__(self, pretrained=False, num_classes=2, dropout=0.5):
        super(VGG16LungCancer, self).__init__()
        self.vgg = models.vgg16(pretrained=pretrained)
        num_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.vgg(x)


# ===========================
# Utility Functions
# ===========================

def load_external_image(image_path):
    """
    Load external image from various formats
    Supports: DICOM (.dcm), MetaImage (.mhd), NIfTI (.nii, .nii.gz), 
              NumPy (.npy), PNG, JPG, etc.
    
    Returns:
        numpy array of the image data
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return None
    
    file_ext = image_path.suffix.lower()
    
    try:
        # DICOM files
        if file_ext in ['.dcm', '.dicom']:
            try:
                import pydicom
                ds = pydicom.dcmread(str(image_path))
                image_data = ds.pixel_array.astype(np.float32)
                print(f"✅ Loaded DICOM file: {image_data.shape}")
                return image_data
            except ImportError:
                print("❌ pydicom not installed. Install with: pip install pydicom")
                return None
        
        # MetaImage, NIfTI formats (SimpleITK)
        elif file_ext in ['.mhd', '.mha', '.nii', '.gz', '.nrrd']:
            sitk_image = sitk.ReadImage(str(image_path))
            image_data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            print(f"✅ Loaded medical image: {image_data.shape}")
            return image_data
        
        # NumPy array
        elif file_ext == '.npy':
            image_data = np.load(str(image_path)).astype(np.float32)
            print(f"✅ Loaded NumPy array: {image_data.shape}")
            return image_data
        
        # Standard image formats (PNG, JPG, etc.)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            from PIL import Image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            image_data = np.array(img).astype(np.float32) / 255.0
            print(f"✅ Loaded image file: {image_data.shape}")
            return image_data
        
        else:
            print(f"❌ Unsupported file format: {file_ext}")
            print("Supported formats: .dcm, .mhd, .mha, .nii, .nii.gz, .npy, .png, .jpg")
            return None
            
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


def load_ct_scan(series_uid, subset_dirs):
    """Load CT scan from .mhd/.raw files (LUNA16 dataset)"""
    for subset_dir in subset_dirs:
        mhd_path = subset_dir / f"{series_uid}.mhd"
        if mhd_path.exists():
            try:
                ct_scan = sitk.ReadImage(str(mhd_path))
                return ct_scan
            except Exception as e:
                print(f"❌ Error loading {mhd_path}: {e}")
                return None
    return None


def world_to_voxel(world_coords, origin, spacing):
    """Convert world coordinates to voxel coordinates"""
    world_coords = np.array(world_coords)
    origin = np.array(origin)
    spacing = np.array(spacing)
    voxel_coords = (world_coords - origin) / spacing
    return int(voxel_coords[2]), int(voxel_coords[1]), int(voxel_coords[0])


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


def prepare_patch_from_image(image_data, patch_size=64, num_slices=3, 
                             center_z=None, center_y=None, center_x=None):
    """
    Prepare a patch from an image for model inference
    
    Args:
        image_data: numpy array (can be 2D or 3D)
        patch_size: size of the patch (default 64x64)
        num_slices: number of slices needed (default 3)
        center_z, center_y, center_x: center coordinates (if None, use image center)
    
    Returns:
        numpy array of shape (num_slices, patch_size, patch_size)
    """
    
    # Handle different input dimensions
    if image_data.ndim == 2:
        # 2D image - replicate to create slices
        print(f"📐 Input is 2D image: {image_data.shape}")
        
        # Normalize
        image_data = normalize_image(image_data)
        
        # Resize if needed
        if image_data.shape[0] != patch_size or image_data.shape[1] != patch_size:
            from scipy.ndimage import zoom
            zoom_factors = (patch_size / image_data.shape[0], patch_size / image_data.shape[1])
            image_data = zoom(image_data, zoom_factors, order=1)
            print(f"   Resized to: {image_data.shape}")
        
        # Replicate to create slices
        patch = np.stack([image_data] * num_slices, axis=0)
        print(f"   Created patch: {patch.shape}")
        return patch
    
    elif image_data.ndim == 3:
        # 3D volume - extract patch
        print(f"📐 Input is 3D volume: {image_data.shape}")
        
        # Normalize
        image_data = normalize_image(image_data)
        
        # Determine center coordinates
        if center_z is None:
            center_z = image_data.shape[0] // 2
        if center_y is None:
            center_y = image_data.shape[1] // 2
        if center_x is None:
            center_x = image_data.shape[2] // 2
        
        print(f"   Center coordinates: z={center_z}, y={center_y}, x={center_x}")
        
        # Extract patch
        patch = extract_2d_patch(image_data, center_z, center_y, center_x, 
                                patch_size, num_slices)
        
        if patch is None:
            print(f"⚠️  Coordinates out of bounds, using center of volume")
            center_z = image_data.shape[0] // 2
            center_y = image_data.shape[1] // 2
            center_x = image_data.shape[2] // 2
            patch = extract_2d_patch(image_data, center_z, center_y, center_x, 
                                    patch_size, num_slices)
        
        if patch is not None:
            print(f"   Extracted patch: {patch.shape}")
        return patch
    
    else:
        print(f"❌ Unsupported image dimensions: {image_data.ndim}D")
        return None


def extract_2d_patch(ct_array, center_z, center_y, center_x, patch_size=64, num_slices=3):
    """Extract 2D multi-slice patch from CT scan"""
    half_size = patch_size // 2
    half_slices = num_slices // 2
    
    z_start = center_z - half_slices
    z_end = z_start + num_slices
    y_start = center_y - half_size
    y_end = center_y + half_size
    x_start = center_x - half_size
    x_end = center_x + half_size
    
    # Check bounds
    if (z_start < 0 or z_end > ct_array.shape[0] or
        y_start < 0 or y_end > ct_array.shape[1] or
        x_start < 0 or x_end > ct_array.shape[2]):
        return None
    
    patch = ct_array[z_start:z_end, y_start:y_end, x_start:x_end]
    # Note: normalization handled by prepare_patch_from_image
    return patch


def load_model(model_name, model_path, device):
    """Load a trained model"""
    print(f"\n🔄 Loading {model_name}...")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Initialize model
    if model_name == 'ResNet-101':
        model = ResNet101LungCancer(pretrained=False, num_classes=2)
    elif model_name == 'EfficientNet-B0':
        model = EfficientNetLungCancer(pretrained=False, num_classes=2)
    elif model_name == 'VGG16':
        model = VGG16LungCancer(pretrained=False, num_classes=2)
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


def predict_single_model(model, patch_tensor, device):
    """Run inference on a single model"""
    with torch.no_grad():
        patch_tensor = patch_tensor.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(patch_tensor)
        probs = F.softmax(outputs, dim=1)
        prob_nodule = probs[0, 1].item()
        prediction = 1 if prob_nodule > 0.5 else 0
        confidence = prob_nodule if prediction == 1 else (1 - prob_nodule)
    
    return prediction, prob_nodule, confidence


def ensemble_prediction(predictions_dict):
    """Combine predictions from multiple models using voting and averaging"""
    
    # Get all probabilities
    probs = [pred['probability'] for pred in predictions_dict.values()]
    preds = [pred['prediction'] for pred in predictions_dict.values()]
    
    # Voting ensemble
    vote_result = 1 if sum(preds) >= len(preds) / 2 else 0
    
    # Average probability
    avg_prob = np.mean(probs)
    
    # Weighted average (could assign different weights to models)
    # For now, equal weights
    weights = {'ResNet-101': 0.33, 'EfficientNet-B0': 0.33, 'VGG16': 0.34}
    weighted_prob = sum(predictions_dict[model]['probability'] * weights[model] 
                       for model in predictions_dict.keys())
    
    # Confidence based on agreement
    agreement = max(sum(preds), len(preds) - sum(preds)) / len(preds) * 100
    
    return {
        'vote_result': vote_result,
        'average_probability': avg_prob,
        'weighted_probability': weighted_prob,
        'agreement': agreement
    }


def visualize_results(patch, predictions_dict, ensemble_result, save_path=None):
    """Visualize the patch and predictions from all models"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Display the three slices
    for i in range(NUM_SLICES):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(patch[i], cmap='gray')
        ax.set_title(f'Slice {i+1}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Display combined view (middle slice)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(patch[1], cmap='gray')
    ax.set_title('Center Slice', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Model predictions - Individual bars
    ax = fig.add_subplot(gs[1, :2])
    model_names = list(predictions_dict.keys())
    model_probs = [predictions_dict[m]['probability'] * 100 for m in model_names]
    colors = [MODEL_CONFIGS[m]['color'] for m in model_names]
    
    bars = ax.barh(model_names, model_probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Cancer Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Individual Model Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Decision Threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add probability text on bars
    for bar, prob in zip(bars, model_probs):
        ax.text(prob + 2, bar.get_y() + bar.get_height()/2, 
               f'{prob:.1f}%', va='center', fontweight='bold')
    
    # Ensemble result
    ax = fig.add_subplot(gs[1, 2:])
    ensemble_data = [
        ensemble_result['average_probability'] * 100,
        ensemble_result['weighted_probability'] * 100
    ]
    ensemble_labels = ['Average', 'Weighted']
    bars = ax.barh(ensemble_labels, ensemble_data, color=['#FF9999', '#66B2FF'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Cancer Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ensemble Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, prob in zip(bars, ensemble_data):
        ax.text(prob + 2, bar.get_y() + bar.get_height()/2, 
               f'{prob:.1f}%', va='center', fontweight='bold')
    
    # Results table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    table_data = []
    for model_name, pred in predictions_dict.items():
        result = "🔴 CANCER" if pred['prediction'] == 1 else "🟢 NON-CANCER"
        table_data.append([
            model_name,
            f"{pred['probability']*100:.2f}%",
            f"{pred['confidence']*100:.2f}%",
            result
        ])
    
    # Add ensemble row
    ensemble_pred = "🔴 CANCER" if ensemble_result['vote_result'] == 1 else "🟢 NON-CANCER"
    table_data.append([
        "🎯 ENSEMBLE (Voting)",
        f"{ensemble_result['average_probability']*100:.2f}%",
        f"{ensemble_result['agreement']:.1f}%",
        ensemble_pred
    ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Cancer Probability', 'Confidence/Agreement', 'Prediction'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.2, 0.25, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style ensemble row
    for i in range(4):
        table[(len(table_data), i)].set_facecolor('#FFE5B4')
        table[(len(table_data), i)].set_text_props(weight='bold')
    
    plt.suptitle('🫁 Multi-Model Lung Cancer Detection Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Visualization saved to: {save_path}")
    
    plt.show()


def print_summary(predictions_dict, ensemble_result):
    """Print a formatted summary of results"""
    
    print("\n" + "="*80)
    print("🫁 MULTI-MODEL LUNG CANCER DETECTION RESULTS".center(80))
    print("="*80)
    
    # Individual model results
    print("\n📊 INDIVIDUAL MODEL PREDICTIONS:\n")
    table_data = []
    for model_name, pred in predictions_dict.items():
        result = "CANCER" if pred['prediction'] == 1 else "NON-CANCER"
        icon = "🔴" if pred['prediction'] == 1 else "🟢"
        table_data.append([
            model_name,
            MODEL_CONFIGS[model_name]['description'],
            f"{pred['probability']*100:.2f}%",
            f"{pred['confidence']*100:.2f}%",
            f"{icon} {result}"
        ])
    
    print(tabulate(table_data, 
                  headers=['Model', 'Description', 'Cancer Prob.', 'Confidence', 'Prediction'],
                  tablefmt='grid'))
    
    # Ensemble results
    print("\n" + "="*80)
    print("🎯 ENSEMBLE RESULTS:\n")
    
    ensemble_pred = "CANCER" if ensemble_result['vote_result'] == 1 else "NON-CANCER"
    icon = "🔴" if ensemble_result['vote_result'] == 1 else "🟢"
    
    ensemble_data = [
        ["Voting Result", f"{icon} {ensemble_pred}"],
        ["Average Probability", f"{ensemble_result['average_probability']*100:.2f}%"],
        ["Weighted Probability", f"{ensemble_result['weighted_probability']*100:.2f}%"],
        ["Model Agreement", f"{ensemble_result['agreement']:.1f}%"]
    ]
    
    print(tabulate(ensemble_data, tablefmt='grid'))
    
    # Final recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION:\n")
    
    if ensemble_result['vote_result'] == 1:
        if ensemble_result['average_probability'] > 0.8:
            print("⚠️  HIGH CONFIDENCE: Strong indication of cancer. Immediate medical attention recommended.")
        elif ensemble_result['average_probability'] > 0.6:
            print("⚠️  MODERATE CONFIDENCE: Possible cancer detected. Further investigation recommended.")
        else:
            print("⚠️  LOW CONFIDENCE: Cancer detected but with lower confidence. Additional tests advised.")
    else:
        if ensemble_result['average_probability'] < 0.2:
            print("✅ HIGH CONFIDENCE: Strong indication of non-cancerous nodule.")
        elif ensemble_result['average_probability'] < 0.4:
            print("✅ MODERATE CONFIDENCE: Likely non-cancerous. Routine monitoring recommended.")
        else:
            print("⚠️  LOW CONFIDENCE: Non-cancer detected but close to threshold. Consider additional screening.")
    
    print("\n" + "="*80)


# ===========================
# Main Inference Function
# ===========================

def run_inference(series_uid=None, coord_x=None, coord_y=None, coord_z=None, 
                 image_path=None, slice_idx=None, center_y=None, center_x=None,
                 visualize=True, save_viz=True):
    """
    Run inference on a CT scan using all three models
    
    Args:
        series_uid: Series UID of the CT scan (LUNA16 dataset)
        coord_x, coord_y, coord_z: World coordinates of the candidate nodule
        image_path: Path to external image file
        slice_idx: Z-axis slice index for 3D images
        center_y, center_x: Y, X coordinates for patch extraction
        visualize: Whether to show visualization
        save_viz: Whether to save visualization
    """
    
    print("\n" + "="*80)
    print("🫁 LUNG CANCER DETECTION - MULTI-MODEL INFERENCE".center(80))
    print("="*80)
    
    patch = None
    ct_array = None
    
    # ===== Load from external image =====
    if image_path is not None:
        print(f"\n📋 Input Information:")
        print(f"   Image Path: {image_path}")
        print(f"   Device: {DEVICE.upper()}")
        
        print(f"\n📂 Loading image...")
        image_data = load_external_image(image_path)
        
        if image_data is None:
            return None
        
        print(f"✅ Image loaded successfully")
        print(f"   Shape: {image_data.shape}")
        print(f"   Data type: {image_data.dtype}")
        print(f"   Value range: [{image_data.min():.2f}, {image_data.max():.2f}]")
        
        # Prepare patch
        print(f"\n✂️  Preparing patch for inference...")
        patch = prepare_patch_from_image(
            image_data, 
            patch_size=PATCH_SIZE, 
            num_slices=NUM_SLICES,
            center_z=slice_idx,
            center_y=center_y,
            center_x=center_x
        )
        
        if patch is None:
            print(f"❌ Could not prepare patch from image")
            return None
        
        print(f"✅ Patch prepared: {patch.shape}")
    
    # ===== Load from LUNA16 dataset =====
    elif series_uid is not None:
        print(f"\n📋 Input Information:")
        print(f"   Series UID: {series_uid}")
        print(f"   Coordinates: ({coord_x:.2f}, {coord_y:.2f}, {coord_z:.2f})")
        print(f"   Device: {DEVICE.upper()}")
        
        # Load CT scan
        print(f"\n📂 Loading CT scan from LUNA16 dataset...")
        ct_scan = load_ct_scan(series_uid, SUBSET_DIRS)
        
        if ct_scan is None:
            print(f"❌ Could not find CT scan with series UID: {series_uid}")
            return None
        
        print(f"✅ CT scan loaded successfully")
        ct_array = sitk.GetArrayFromImage(ct_scan)
        print(f"   Shape: {ct_array.shape}")
        
        # Convert to voxel coordinates
        origin = ct_scan.GetOrigin()
        spacing = ct_scan.GetSpacing()
        
        voxel_z, voxel_y, voxel_x = world_to_voxel([coord_x, coord_y, coord_z], origin, spacing)
        print(f"\n🎯 Voxel Coordinates: ({voxel_z}, {voxel_y}, {voxel_x})")
        
        # Extract patch
        print(f"\n✂️  Extracting patch...")
        patch = extract_2d_patch(ct_array, voxel_z, voxel_y, voxel_x, 
                                patch_size=PATCH_SIZE, num_slices=NUM_SLICES)
        
        if patch is None:
            print(f"❌ Could not extract patch (coordinates out of bounds)")
            return None
        
        # Normalize the patch
        patch = normalize_hu(patch)
        print(f"✅ Patch extracted: {patch.shape}")
    
    else:
        print("❌ Either series_uid or image_path must be provided")
        return None
    
    # Convert to tensor
    patch_tensor = torch.from_numpy(patch).float()
    
    # Load models and run inference
    print(f"\n🤖 Running inference on all models...")
    predictions_dict = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        model = load_model(model_name, config['model_path'], DEVICE)
        
        if model is not None:
            start_time = time.time()
            prediction, prob_nodule, confidence = predict_single_model(model, patch_tensor, DEVICE)
            inference_time = time.time() - start_time
            
            predictions_dict[model_name] = {
                'prediction': prediction,
                'probability': prob_nodule,
                'confidence': confidence,
                'inference_time': inference_time
            }
            
            result = "CANCER" if prediction == 1 else "NON-CANCER"
            icon = "🔴" if prediction == 1 else "🟢"
            print(f"   {icon} {model_name}: {result} (Prob: {prob_nodule*100:.2f}%, "
                  f"Confidence: {confidence*100:.2f}%, Time: {inference_time*1000:.2f}ms)")
    
    if not predictions_dict:
        print(f"❌ No models could be loaded")
        return None
    
    # Ensemble prediction
    print(f"\n🎯 Computing ensemble prediction...")
    ensemble_result = ensemble_prediction(predictions_dict)
    
    # Print summary
    print_summary(predictions_dict, ensemble_result)
    
    # Visualize
    if visualize:
        save_path = None
        if save_viz:
            output_dir = BASE_DIR / 'inference_results'
            output_dir.mkdir(exist_ok=True)
            if image_path:
                filename = Path(image_path).stem
                save_path = output_dir / f'inference_{filename}.png'
            else:
                save_path = output_dir / f'inference_{series_uid[:20]}.png'
        
        visualize_results(patch, predictions_dict, ensemble_result, save_path)
    
    return {
        'predictions': predictions_dict,
        'ensemble': ensemble_result,
        'patch': patch,
        'input_type': 'external_image' if image_path else 'luna16',
        'coordinates': {
            'world': (coord_x, coord_y, coord_z) if series_uid else None,
            'voxel': (voxel_z, voxel_y, voxel_x) if series_uid and ct_array is not None else None,
            'patch': (slice_idx, center_y, center_x) if image_path else None
        }
    }


# ===========================
# Command Line Interface
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='Run multi-model inference for lung cancer detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LUNA16 dataset (requires series UID and world coordinates)
  python inference_ensemble.py --series_uid 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260 \\
                                --coord_x -56.08 --coord_y -67.85 --coord_z -311.92
  
  # External CT scan (3D volume with coordinates)
  python inference_ensemble.py --image_path patient_scan.mhd --slice_idx 150 --center_y 200 --center_x 200
  
  # External 2D image (automatic patch sizing)
  python inference_ensemble.py --image_path nodule_patch.png
  
  # DICOM file
  python inference_ensemble.py --image_path scan.dcm --slice_idx 100 --center_y 256 --center_x 256
  
  # NumPy array
  python inference_ensemble.py --image_path ct_data.npy --center_y 128 --center_x 128
  
  # Without visualization
  python inference_ensemble.py --image_path scan.png --no-viz
        """
    )
    
    # Input source: LUNA16 dataset OR external image
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--series_uid', type=str,
                            help='Series UID of the CT scan (LUNA16 dataset)')
    input_group.add_argument('--image_path', type=str,
                            help='Path to external image file (DICOM, .mhd, .npy, .png, .jpg, etc.)')
    
    # LUNA16 coordinates (world coordinates)
    parser.add_argument('--coord_x', type=float,
                       help='X coordinate in world coordinates (for LUNA16)')
    parser.add_argument('--coord_y', type=float,
                       help='Y coordinate in world coordinates (for LUNA16)')
    parser.add_argument('--coord_z', type=float,
                       help='Z coordinate in world coordinates (for LUNA16)')
    
    # External image coordinates (voxel/pixel coordinates)
    parser.add_argument('--slice_idx', type=int,
                       help='Z-axis slice index for 3D images (for external images)')
    parser.add_argument('--center_y', type=int,
                       help='Y coordinate for patch center (for external images)')
    parser.add_argument('--center_x', type=int,
                       help='X coordinate for patch center (for external images)')
    
    # Visualization options
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save visualization')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.series_uid and not all([args.coord_x, args.coord_y, args.coord_z]):
        parser.error("--series_uid requires --coord_x, --coord_y, and --coord_z")
    
    # Run inference
    result = run_inference(
        series_uid=args.series_uid,
        coord_x=args.coord_x,
        coord_y=args.coord_y,
        coord_z=args.coord_z,
        image_path=args.image_path,
        slice_idx=args.slice_idx,
        center_y=args.center_y,
        center_x=args.center_x,
        visualize=not args.no_viz,
        save_viz=not args.no_save
    )
    
    if result is None:
        print("\n❌ Inference failed")
        sys.exit(1)
    
    print("\n✅ Inference completed successfully!")


if __name__ == '__main__':
    main()
