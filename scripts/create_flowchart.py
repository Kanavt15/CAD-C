"""
Create a visual flowchart showing the inference pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, '🫁 Multi-Model Lung Cancer Detection Pipeline', 
        fontsize=18, fontweight='bold', ha='center')

# Colors
color_input = '#E8F4F8'
color_process = '#FFF4E6'
color_model = '#E8F5E9'
color_output = '#F3E5F5'

# ===== INPUT LAYER =====
ax.text(5, 10.5, '📥 INPUT', fontsize=14, fontweight='bold', ha='center')

# Input boxes
inputs = [
    ('LUNA16\nDataset', 1, 9.5),
    ('DICOM\nFiles', 2.5, 9.5),
    ('MetaImage\n(.mhd)', 4, 9.5),
    ('NIfTI\n(.nii)', 5.5, 9.5),
    ('NumPy\n(.npy)', 7, 9.5),
    ('Images\n(.png/.jpg)', 8.5, 9.5),
]

for label, x, y in inputs:
    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                         boxstyle="round,pad=0.05", 
                         facecolor=color_input, 
                         edgecolor='#0288D1', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, fontsize=9, ha='center', va='center')

# ===== PREPROCESSING =====
ax.text(5, 8.5, '⚙️ PREPROCESSING', fontsize=14, fontweight='bold', ha='center')

# Preprocessing box
preprocess_steps = [
    '1. Load Image',
    '2. Normalize (0-1)',
    '3. Extract/Resize Patch (64x64)',
    '4. Create 3 Slices',
    '5. Convert to Tensor'
]

box = FancyBboxPatch((2, 6.5), 6, 1.8,
                     boxstyle="round,pad=0.1", 
                     facecolor=color_process, 
                     edgecolor='#F57C00', linewidth=2)
ax.add_patch(box)

for i, step in enumerate(preprocess_steps):
    ax.text(2.2, 8.0 - i*0.32, step, fontsize=9, va='top')

# Arrows from input to preprocessing
for _, x, y in inputs:
    arrow = FancyArrowPatch((x, y-0.4), (x, 8.3),
                          arrowstyle='->', mutation_scale=15,
                          color='#666', linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow)

# Arrow from preprocessing to models
arrow = FancyArrowPatch((5, 6.5), (5, 5.8),
                      arrowstyle='->', mutation_scale=20,
                      color='#666', linewidth=2)
ax.add_patch(arrow)

# ===== MODEL LAYER =====
ax.text(5, 5.5, '🤖 DEEP LEARNING MODELS', fontsize=14, fontweight='bold', ha='center')

models = [
    ('ResNet-101\n44.5M params', 2, 4.2, '#FF6B6B'),
    ('EfficientNet-B0\n5.3M params', 5, 4.2, '#4ECDC4'),
    ('LUNA16-DenseNet\nReal Data Trained', 8, 4.2, '#9B59B6'),
]

for label, x, y, color in models:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                         boxstyle="round,pad=0.05", 
                         facecolor=color, 
                         edgecolor='#333', linewidth=2,
                         alpha=0.7)
    ax.add_patch(box)
    ax.text(x, y, label, fontsize=10, ha='center', va='center', 
           fontweight='bold')

# Arrows to models
for _, x, y, _ in models:
    arrow = FancyArrowPatch((5, 5.3), (x, y+0.5),
                          arrowstyle='->', mutation_scale=15,
                          color='#666', linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow)

# ===== PREDICTIONS =====
ax.text(5, 3.2, '📊 INDIVIDUAL PREDICTIONS', fontsize=14, fontweight='bold', ha='center')

predictions = [
    ('Prob: 98.66%\n🔴 CANCER', 2, 2.2),
    ('Prob: 82.63%\n🔴 CANCER', 5, 2.2),
    ('Prob: 49.94%\n🟢 NON-CANCER', 8, 2.2),
]

for label, x, y in predictions:
    box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                         boxstyle="round,pad=0.05", 
                         facecolor='white', 
                         edgecolor='#666', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, fontsize=9, ha='center', va='center')

# Arrows from models to predictions
for (_, mx, my, _), (_, px, py) in zip(models, predictions):
    arrow = FancyArrowPatch((mx, my-0.5), (px, py+0.35),
                          arrowstyle='->', mutation_scale=12,
                          color='#666', linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow)

# ===== ENSEMBLE =====
ax.text(5, 1.3, '🎯 ENSEMBLE', fontsize=14, fontweight='bold', ha='center')

ensemble_box = FancyBboxPatch((3.5, 0.3), 3, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor=color_output, 
                             edgecolor='#7B1FA2', linewidth=3)
ax.add_patch(ensemble_box)

ensemble_text = 'Voting: 🔴 CANCER\nAvg Prob: 77.07%\nAgreement: 66.7%'
ax.text(5, 0.7, ensemble_text, fontsize=10, ha='center', va='center',
       fontweight='bold')

# Arrows from predictions to ensemble
for _, x, y in predictions:
    arrow = FancyArrowPatch((x, y-0.35), (5, 1.15),
                          arrowstyle='->', mutation_scale=15,
                          color='#666', linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow)

# Legend
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_input, edgecolor='#0288D1', label='Input Formats'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_process, edgecolor='#F57C00', label='Preprocessing'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_model, edgecolor='#333', label='Deep Learning'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=color_output, edgecolor='#7B1FA2', label='Final Output'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=4, 
         frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('e:/Kanav/Projects/CAD_C/inference_pipeline_flowchart.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Flowchart saved: inference_pipeline_flowchart.png")
plt.show()
