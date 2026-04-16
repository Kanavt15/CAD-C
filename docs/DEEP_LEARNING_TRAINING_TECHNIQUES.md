# Deep Learning Algorithms and Training Techniques Used in CAD-C

This document summarizes the deep learning algorithms, model architectures, and training strategies used across this repository for lung nodule/cancer modeling.

---

## 1. Core Problem Setup

### 1.1 Primary task types
1. **3D nodule/cancer classification (binary):**
   - Input: 3D CT patches (commonly `64 x 64 x 64`)
   - Output: class logits for **non-cancerous vs cancerous**

2. **3D segmentation workflow (U-Net notebook path):**
   - Input: volumetric CT regions
   - Output: voxel-wise segmentation masks
   - Used as an advanced/experimental pathway for better region localization

### 1.2 Medical-image preprocessing techniques
- **HU clipping** to lung-relevant window: `[-1000, 400]`
- **Min-max normalization** to `[0, 1]`
- **Patch extraction** around nodules/candidates
- **Patch balancing/pre-extraction caches** to accelerate training and reduce I/O overhead

---

## 2. Classification Architectures Used

## 2.1 Improved 3D CNN (ResNet-style + SE attention)
Implemented in `web_app/models_3d_cnn/model_architecture.py`.

### Key algorithms/blocks
- **3D Convolutions** (`Conv3d`) for volumetric feature extraction
- **Residual learning** (skip connections) to improve gradient flow in deep networks
- **Squeeze-and-Excitation (SE) attention**:
  - Global channel squeeze via `AdaptiveAvgPool3d(1)`
  - Channel reweighting through a bottleneck MLP + sigmoid
  - Improves channel-wise feature selection
- **BatchNorm3d + ReLU** throughout
- **Global average pooling + dropout-regularized FC head**

### Why this matters
- Residuals stabilize deep 3D training.
- SE attention boosts sensitivity to informative channels in noisy CT volumes.
- Dropout and weight decay reduce overfitting on limited positive samples.

---

## 2.2 EfficientNet3D-B2 (compound-scaled MBConv for 3D)
Implemented in `web_app/efficientnet3d_b2_architecture.py`.

### Key algorithms/blocks
- **EfficientNet compound scaling (3D adaptation):**
  - Width and depth multipliers (`width_mult`, `depth_mult`)
- **MBConv3D blocks**:
  - Expansion `1x1x1 conv`
  - **Depthwise 3D convolution**
  - Projection `1x1x1 conv`
- **SE attention inside MBConv**
- **Swish activation**
- **Stochastic Depth** (sample-level block dropping during training)

### Why this matters
- High accuracy/efficiency tradeoff vs heavier 3D backbones.
- Stochastic depth improves regularization and generalization.
- Compound scaling gives principled capacity growth instead of naive widening/deepening.

---

## 2.3 DenseNet3D with Multi-Head Attention
Implemented in `web_app/densenet3d_architecture.py`.

### Key algorithms/blocks
- **Dense connectivity**:
  - Each layer concatenates previous features (`torch.cat`)
  - Strong feature reuse and improved gradient propagation
- **Bottleneck-style Dense layers** (`1x1x1` then `3x3x3`)
- **Transition layers**:
  - Channel compression + average pooling
- **Multi-head self-attention on 3D features**
  - Q/K/V projection via `1x1x1 conv`
  - Scaled dot-product attention across flattened 3D tokens
- **DropPath (stochastic depth variant)**
- **Residual attention fusion** (`x + attention(x)`)

### Why this matters
- Dense feature reuse lowers parameter count while keeping representational power.
- Attention adds global context modeling beyond local convolutional receptive fields.

---

## 3. Segmentation Architecture Used (Advanced Notebook Path)

## 3.1 MONAI U-Net (3D segmentation)
Notebook: `notebooks/lung_nodule_unet.ipynb`.

### Key algorithms/blocks
- **U-Net encoder-decoder** with skip connections
- **Dice-based optimization**:
  - `DiceLoss`
  - `DiceFocalLoss` in improved settings
- **DiceMetric** for validation monitoring

### Why this matters
- Segmentation improves localization quality and can boost downstream classifier quality when used in staged pipelines.

---

## 4. Loss Functions Used

## 4.1 Weighted CrossEntropyLoss
Used in baseline 3D CNN training stages.

- Class weights are computed from data distribution.
- Helps reduce bias toward the majority negative class.

## 4.2 Focal Loss (major technique across improved runs)
Used in improved 3D CNN, EfficientNet3D-B2, and DenseNet3D attention notebooks.

- Formula emphasizes hard examples and down-weights easy examples.
- Typical settings found:
  - `gamma = 2.0`
  - `alpha` as scalar or class-weight tensor

### Why focal loss is critical here
- LUNA-style pipelines are highly imbalanced.
- Focal loss improves sensitivity/recall for minority positive nodules.

## 4.3 Dice + Focal hybrid (segmentation path)
- `DiceFocalLoss` combines overlap optimization (Dice) with hard-example focusing (Focal).

---

## 5. Imbalance Handling Strategies

The project uses multiple techniques together (not just one):

1. **Class weighting** (in CE/Focal alpha)
2. **Positive oversampling / balanced sampling**
3. **Augmentation targeted to positive patches** in some pipelines
4. **Balanced patch extraction/caching** for train/val/test splits

This multi-layer handling is important because class imbalance is severe in lung nodule candidate data.

---

## 6. Optimization and Scheduling Techniques

## 6.1 Optimizers
- **Adam** in baseline stages
- **AdamW** in improved stages (better decoupled weight decay behavior)

## 6.2 Learning-rate schedulers
- **ReduceLROnPlateau** (validation-driven LR decay)
- **CosineAnnealingLR**
- **CosineAnnealingWarmRestarts** (used in improved 3D CNN path)

## 6.3 Early stopping
- Patience-based stopping (commonly `~10-15`) to prevent overfitting and wasted epochs.

## 6.4 Mixed precision training
- **AMP (`autocast`, `GradScaler`)** used in improved notebooks.
- Benefits:
  - Faster training
  - Lower VRAM usage
  - Larger feasible batch sizes

---

## 7. Data Augmentation Techniques

Across notebooks, augmentation includes combinations of:

- Random flips (spatial axis flips)
- Random rotations
- Intensity variation/contrast adjustments
- Noise injection
- Zoom/scale perturbations
- Elastic-like deformation paths in advanced augmentation notebooks
- Random patch offset around nodules (for localization robustness)

In some segmentation steps, augmentation paths were adjusted to avoid specific MONAI-version issues, with alternatives applied during training.

---

## 8. Regularization Techniques

Used throughout architectures/training:

- **Dropout** in classifier heads
- **Weight decay** (`AdamW`)
- **Batch normalization**
- **Stochastic Depth / DropPath**
- **Early stopping**
- **Data augmentation**

These regularizers are especially important due to limited positive medical samples.

---

## 9. Ensemble and Inference-Level Techniques

Although not a training optimizer itself, the repository also applies model-combination methods:

1. **Majority voting**
2. **Probability averaging**
3. **Weighted probability fusion**
4. **Threshold tuning** for operating-point tradeoffs (sensitivity vs specificity)

This improves robustness across architectural biases.

---

## 10. Typical Training Configuration Patterns (Observed)

These vary by notebook, but common patterns are:

- Patch size: `64^3`
- Batch size:
  - Smaller (`~4`) for heavier 3D models
  - Larger (`~16`) in optimized EfficientNet runs
- Learning rate:
  - `1e-3` baseline/fast-convergence runs
  - `1e-4` stabilized runs (especially attention-heavy models)
- Weight decay:
  - Around `1e-4` or `1e-5` in AdamW configs
- Focal gamma:
  - Commonly `2.0`

---

## 11. Practical Interpretation: Why This Stack Works

The project combines:

1. **3D spatial modeling** (Conv3D backbones),
2. **attention mechanisms** (SE + multi-head attention),
3. **imbalance-aware objectives** (Focal, weighted CE),
4. **modern optimization** (AdamW + cosine schedules + AMP),
5. **strong regularization** (dropout, stochastic depth, augmentation),
6. **ensemble decision logic**.

That combination is well-suited for imbalanced, high-dimensional CT data where both sensitivity and calibration matter.

---

## 12. Where Each Technique Appears in This Repo

- **Improved 3D CNN + Residual + SE:**  
  `web_app/models_3d_cnn/model_architecture.py`

- **EfficientNet3D-B2 + MBConv + SE + Stochastic Depth:**  
  `web_app/efficientnet3d_b2_architecture.py`

- **DenseNet3D + Multi-Head Attention + DropPath:**  
  `web_app/densenet3d_architecture.py`

- **Training pipelines (losses, schedulers, AMP, early stopping, augmentation):**  
  `notebooks/lung_cancer_3d_cnn.ipynb`  
  `notebooks/lung_cancer_efficientnet3d_b2.ipynb`  
  `notebooks/lung_cancer_densenet3d_attention.ipynb`

- **Segmentation pathway (MONAI U-Net, Dice/DiceFocal):**  
  `notebooks/lung_nodule_unet.ipynb`

