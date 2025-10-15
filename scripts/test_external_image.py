"""
Test script for external image inference
Creates a sample CT-like image and tests the inference pipeline
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Create output directory
output_dir = Path(r'e:\Kanav\Projects\CAD_C\test_images')
output_dir.mkdir(exist_ok=True)

print("🔬 Creating test images for inference...\n")

# ====================
# Test 1: Simple 2D image (64x64)
# ====================
print("1️⃣ Creating 64x64 test image...")

# Create a synthetic nodule-like pattern
size = 64
x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
r = np.sqrt(x**2 + y**2)

# Create nodule (bright circular region)
nodule = np.exp(-r**2 / 0.1)  # Gaussian blob
noise = np.random.normal(0, 0.1, (size, size))
image_2d = (nodule + noise) * 0.8 + 0.2

# Normalize to 0-255
image_2d = np.clip(image_2d * 255, 0, 255).astype(np.uint8)

# Save
image_path_2d = output_dir / 'test_nodule_64x64.png'
Image.fromarray(image_2d).save(image_path_2d)
print(f"   ✅ Saved: {image_path_2d}")

# Visualize
plt.figure(figsize=(5, 5))
plt.imshow(image_2d, cmap='gray')
plt.title('Test Nodule (64x64)')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'test_nodule_64x64_preview.png', dpi=150)
plt.close()

# ====================
# Test 2: Larger 2D image (256x256)
# ====================
print("\n2️⃣ Creating 256x256 test image...")

size = 256
x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
r = np.sqrt(x**2 + y**2)

# Create multiple nodules
nodule1 = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.15)
nodule2 = np.exp(-((x+0.7)**2 + (y-0.3)**2) / 0.12) * 0.7
background = 0.3 + 0.05 * np.sin(x * 3) * np.cos(y * 2)
noise = np.random.normal(0, 0.05, (size, size))

image_large = (nodule1 + nodule2 + background + noise) * 0.7 + 0.2

# Normalize
image_large = np.clip(image_large * 255, 0, 255).astype(np.uint8)

# Save
image_path_large = output_dir / 'test_nodule_256x256.png'
Image.fromarray(image_large).save(image_path_large)
print(f"   ✅ Saved: {image_path_large}")

# Visualize
plt.figure(figsize=(7, 7))
plt.imshow(image_large, cmap='gray')
plt.title('Test CT Scan (256x256)')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'test_nodule_256x256_preview.png', dpi=150)
plt.close()

# ====================
# Test 3: 3D volume (numpy array)
# ====================
print("\n3️⃣ Creating 3D test volume (10x128x128)...")

depth, height, width = 10, 128, 128
volume_3d = np.zeros((depth, height, width), dtype=np.float32)

# Create a nodule in the center
for z in range(depth):
    x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    z_normalized = (z - depth/2) / (depth/4)
    r = np.sqrt(x**2 + y**2 + z_normalized**2)
    nodule_slice = np.exp(-r**2 / 0.3) * 0.8
    background = 0.2 + 0.05 * np.random.randn(height, width)
    volume_3d[z] = np.clip(nodule_slice + background, 0, 1)

# Save as numpy
volume_path = output_dir / 'test_volume_3d.npy'
np.save(volume_path, volume_3d)
print(f"   ✅ Saved: {volume_path}")

# Visualize middle slice
plt.figure(figsize=(6, 6))
plt.imshow(volume_3d[depth//2], cmap='gray')
plt.title('Test 3D Volume (Middle Slice)')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'test_volume_3d_preview.png', dpi=150)
plt.close()

# ====================
# Summary
# ====================
print("\n" + "="*70)
print("✅ TEST IMAGES CREATED SUCCESSFULLY")
print("="*70)
print(f"\nTest images saved to: {output_dir}\n")
print("📋 Test cases:")
print(f"   1. 64x64 2D image:    {image_path_2d.name}")
print(f"   2. 256x256 2D image:  {image_path_large.name}")
print(f"   3. 3D volume:         {volume_path.name}")

print("\n🚀 Now you can test with these commands:\n")
print("# Test with 64x64 image (automatic sizing)")
print(f"python inference_ensemble.py --image_path {image_path_2d}\n")

print("# Test with 256x256 image (specify center)")
print(f"python inference_ensemble.py --image_path {image_path_large} --center_y 128 --center_x 128\n")

print("# Test with 3D volume (specify slice and center)")
print(f"python inference_ensemble.py --image_path {volume_path} --slice_idx 5 --center_y 64 --center_x 64\n")

print("="*70)
