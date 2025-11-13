# Performance Improvement Strategies for LUNA16 Lung Nodule Detection

## Current Performance
- **Validation Dice**: 0.5999
- **Test Dice**: 0.5235
- **Generalization**: 87.3%

## 🚀 High-Impact Improvements (Ranked by Expected Gain)

### 1. ⭐⭐⭐ Use Real LUNA16 Annotations (Expected +15-25% Dice)
**Problem**: Currently training on synthetic dummy labels (5x5x5 sphere in center)
**Solution**: Extract real nodule patches from annotations.csv

```python
def extract_real_nodule_patches(mhd_path, annotations_df, patch_size=64):
    """Extract patches centered on actual nodule locations"""
    vol, origin, spacing = read_mhd(mhd_path)
    vol = resample_to_spacing(vol, spacing, TARGET_SPACING, order=1)
    vol = normalize_hu(vol)
    
    # Get scan ID from filename
    scan_id = os.path.basename(mhd_path).replace('.mhd', '')
    nodules = annotations_df[annotations_df['seriesuid'] == scan_id]
    
    patches = []
    for _, nodule in nodules.iterrows():
        # Convert world coordinates to voxel coordinates
        world_coord = np.array([nodule['coordX'], nodule['coordY'], nodule['coordZ']])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        
        # Extract patch around nodule
        patch, label = extract_patch_with_mask(vol, voxel_coord, 
                                               nodule['diameter_mm'], 
                                               patch_size)
        if patch is not None:
            patches.append({'image': patch, 'label': label})
    
    return patches
```

**Implementation**: Replace `make_toy_dataset()` with real annotation-based extraction

---

### 2. ⭐⭐⭐ Increase Training Data (Expected +10-20% Dice)
**Problem**: Only 160 training samples (4 batches/epoch)
**Solution**: Extract more patches per CT scan

**Current**:
```python
samples_per_subset=50  # Too few!
n_crops=4  # Way too few patches per scan
```

**Recommended**:
```python
samples_per_subset=500  # 10x increase
# For each nodule: extract positive + negative samples
positive_samples_per_nodule = 5  # With augmentation
negative_samples_per_positive = 3  # Hard negative mining
```

---

### 3. ⭐⭐ Implement Proper Class Balancing (Expected +8-15% Dice)
**Problem**: Severe class imbalance (nodules are <1% of volume)
**Solution**: Balanced sampling + weighted loss

```python
# Weighted sampling
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
positive_ratio = 0.5  # Force 50/50 balance
weights = [1.0 if has_nodule else positive_ratio/negative_ratio 
           for sample in dataset]

sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
loader = DataLoader(dataset, batch_size=2, sampler=sampler)

# Weighted loss
class_weights = torch.tensor([1.0, 10.0]).to(device)  # Weight positives 10x
focal_loss = FocalLoss(alpha=0.75, gamma=3.0)  # Increase gamma for hard examples
```

---

### 4. ⭐⭐ Increase Model Capacity (Expected +5-10% Dice)
**Current**: UNet with [16, 32, 64, 128] channels (1.2M params)
**Recommended**: Deeper/wider architecture

```python
# Option A: Wider channels
final_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),  # 5 levels, wider
    strides=(2, 2, 2, 2),
    num_res_units=3,  # More residual blocks
    dropout=0.3,
)

# Option B: Use attention U-Net
from monai.networks.nets import AttentionUnet
model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
)
```

---

### 5. ⭐⭐ Longer Training (Expected +5-8% Dice)
**Current**: 100 epochs, early stopped at ~84
**Problem**: Dice jumped from 0.01 → 0.60 in epochs 70-84 (still learning!)

```python
FULL_TRAINING_CONFIG = {
    'epochs': 300,  # Triple the epochs
    'early_stopping_patience': 30,  # More patience
    'batch_size': 4,  # Larger batches (if GPU allows)
}

# Implement learning rate warm-up
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [10])
```

---

### 6. ⭐ Optimize Loss Function (Expected +5-8% Dice)
**Current**: 0.6*Dice + 0.4*Focal
**Problem**: Dice loss struggles with small objects

```python
# Combo loss for better small object detection
from monai.losses import DiceFocalLoss, TverskyLoss

# Option A: Tversky Loss (better for imbalanced data)
tversky = TverskyLoss(
    alpha=0.7,  # Penalize false negatives more
    beta=0.3,   # Penalize false positives less
    smooth_nr=1e-5,
    smooth_dr=1e-5
)

# Option B: Combined loss
def advanced_combined_loss(pred, target):
    dice = DiceLoss(sigmoid=True)(pred, target)
    focal = FocalLoss(alpha=0.75, gamma=3.0)(pred, target)
    tversky = TverskyLoss(alpha=0.7, beta=0.3)(pred, target)
    
    # Boundary loss for sharp edges
    boundary = compute_boundary_loss(pred, target)
    
    return 0.4*dice + 0.3*focal + 0.2*tversky + 0.1*boundary
```

---

### 7. ⭐ Multi-Scale Training (Expected +3-7% Dice)
**Current**: Fixed 64³ patches
**Problem**: Misses context for large nodules

```python
# Multi-scale patch extraction
patch_sizes = [48, 64, 96]  # Train on multiple scales

def multi_scale_transform(sample, scales=[48, 64, 96]):
    """Extract patches at multiple scales"""
    transforms = []
    for size in scales:
        t = Compose([
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=(size, size, size),
                pos=2, neg=1,
                num_samples=1
            ),
            # Resize to common size
            Resized(keys=['image', 'label'], spatial_size=(64, 64, 64))
        ])
        transforms.append(t)
    return transforms
```

---

### 8. ⭐ Test-Time Augmentation (Expected +2-5% Dice)
**Inference only** - no retraining needed

```python
def predict_with_tta(model, image, num_augmentations=8):
    """Test-time augmentation for robust predictions"""
    predictions = []
    
    for _ in range(num_augmentations):
        # Random flip
        aug_img = torch.flip(image, dims=[np.random.choice([2,3,4])])
        
        # Random rotation
        angle = np.random.choice([0, 90, 180, 270])
        aug_img = torch.rot90(aug_img, k=angle//90, dims=[2,3])
        
        # Predict
        pred = model(aug_img)
        
        # Reverse augmentation
        pred = torch.rot90(pred, k=-angle//90, dims=[2,3])
        pred = torch.flip(pred, dims=[np.random.choice([2,3,4])])
        
        predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

---

### 9. ⭐ Post-Processing Pipeline (Expected +3-6% Dice)
**Current**: Simple threshold (>0.5)
**Better**: Multi-stage refinement

```python
def post_process_predictions(pred_volume, min_size=20, confidence_threshold=0.6):
    """Refine predictions with morphological operations"""
    from scipy import ndimage
    
    # 1. Threshold
    binary = (pred_volume > confidence_threshold).astype(np.uint8)
    
    # 2. Remove small objects
    labeled, num_features = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(num_features + 1))
    mask_sizes = sizes > min_size
    binary = mask_sizes[labeled]
    
    # 3. Morphological closing (fill holes)
    binary = ndimage.binary_closing(binary, iterations=2)
    
    # 4. Keep only high-confidence regions
    binary = binary * (pred_volume > 0.4)
    
    return binary
```

---

### 10. Deep Supervision (Expected +2-4% Dice)
**Add auxiliary losses at intermediate layers**

```python
class DeepSupervisionUNet(nn.Module):
    def __init__(self, base_unet):
        super().__init__()
        self.unet = base_unet
        
        # Add output heads at each decoder level
        self.aux_heads = nn.ModuleList([
            nn.Conv3d(128, 1, 1),
            nn.Conv3d(64, 1, 1),
            nn.Conv3d(32, 1, 1),
        ])
    
    def forward(self, x):
        # Get intermediate features
        features = self.unet.get_intermediate_features(x)
        
        # Main output
        main_out = self.unet(x)
        
        # Auxiliary outputs
        aux_outs = [head(feat) for head, feat in zip(self.aux_heads, features)]
        
        return main_out, aux_outs

# Loss with deep supervision
def deep_supervision_loss(outputs, target):
    main_out, aux_outs = outputs
    
    main_loss = combined_loss(main_out, target)
    
    aux_loss = 0
    for i, aux_out in enumerate(aux_outs):
        # Resize target to match auxiliary output
        resized_target = F.interpolate(target, size=aux_out.shape[2:])
        aux_loss += combined_loss(aux_out, resized_target) * (0.5 ** i)
    
    return main_loss + 0.3 * aux_loss
```

---

## 📊 Expected Performance with Improvements

| Improvement | Current Dice | Expected Dice | Difficulty |
|-------------|--------------|---------------|------------|
| Baseline | 0.52 | 0.52 | - |
| + Real annotations | 0.52 | 0.70-0.75 | Medium |
| + More training data | 0.70 | 0.77-0.80 | Easy |
| + Class balancing | 0.77 | 0.82-0.85 | Easy |
| + Larger model | 0.82 | 0.85-0.87 | Easy |
| + Longer training | 0.85 | 0.87-0.89 | Easy |
| + Better loss | 0.87 | 0.88-0.90 | Medium |
| + Multi-scale | 0.88 | 0.89-0.91 | Medium |
| + TTA + Post-proc | 0.89 | 0.91-0.93 | Easy |
| + Deep supervision | 0.91 | 0.92-0.94 | Hard |

**Target**: 0.90+ Dice (competitive with SOTA)

---

## 🔧 Quick Wins (Implement Today)

### 1. Fix Data Loading (30 minutes)
```python
# Cell 43: Replace toy dataset loading
annotations_df = pd.read_csv(ANNOTATIONS_CSV)

def load_real_patches(mhd_path, annotations_df, n_positive=10, n_negative=20):
    patches = []
    
    # Load scan
    vol, origin, spacing = read_mhd(mhd_path)
    vol = resample_to_spacing(vol, spacing, TARGET_SPACING)
    vol = normalize_hu(vol)
    
    scan_id = os.path.basename(mhd_path).replace('.mhd', '')
    nodules = annotations_df[annotations_df['seriesuid'] == scan_id]
    
    # Extract positive patches (around nodules)
    for _, nodule in nodules.iterrows():
        for _ in range(n_positive):
            patch = extract_nodule_patch(vol, nodule, patch_size=64, 
                                         origin=origin, spacing=spacing,
                                         augment=True)
            if patch is not None:
                patches.append(patch)
    
    # Extract negative patches (no nodules)
    for _ in range(n_negative):
        patch = extract_random_patch(vol, patch_size=64, 
                                     avoid_nodules=nodules)
        if patch is not None:
            patches.append(patch)
    
    return patches
```

### 2. Increase Training Epochs (1 line change)
```python
FULL_TRAINING_CONFIG = {
    'epochs': 300,  # Change from 100 to 300
    'early_stopping_patience': 30,  # Change from 15 to 30
}
```

### 3. Better Loss Function (5 minutes)
```python
# Replace final_combined_loss
from monai.losses import DiceFocalLoss

def final_combined_loss(pred, target):
    return DiceFocalLoss(
        sigmoid=True,
        focal_weight=torch.tensor([1.0, 10.0]),  # Weight positives
        lambda_dice=0.6,
        lambda_focal=0.4,
        gamma=3.0
    )(pred, target)
```

---

## 📈 Training Monitoring Improvements

```python
# Add to training loop
import wandb  # pip install wandb

wandb.init(project="luna16-detection", config=FULL_TRAINING_CONFIG)

# Log metrics each epoch
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_dice': val_dice,
    'learning_rate': current_lr,
    'epoch': epoch,
    
    # Log predictions periodically
    'predictions': wandb.Image(pred_slice),
    'ground_truth': wandb.Image(label_slice),
})
```

---

## 🎯 Implementation Priority

**Week 1** (Easy wins):
1. ✅ Use real annotations from LUNA16
2. ✅ Increase training data 10x
3. ✅ Train for 300 epochs
4. ✅ Implement class balancing

**Week 2** (Medium difficulty):
5. ✅ Larger model architecture
6. ✅ Better loss function
7. ✅ Multi-scale training

**Week 3** (Advanced):
8. ✅ Test-time augmentation
9. ✅ Deep supervision
10. ✅ Ensemble multiple models

**Expected Final Performance**: 0.90-0.94 Dice (SOTA level)
