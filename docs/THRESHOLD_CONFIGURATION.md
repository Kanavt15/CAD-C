# Cancer Classification Threshold Configuration 🎯

**Date**: October 6, 2025  
**Feature**: Configurable threshold for cancer classification  
**Current Threshold**: **65%**

---

## 📋 Overview

The lung cancer detection system now uses a configurable probability threshold to determine cancer classification. This provides more control over the balance between sensitivity (catching all cancer cases) and specificity (avoiding false positives).

---

## 🎯 How It Works

### Threshold Logic

```python
CANCER_THRESHOLD = 0.65  # 65%

if cancerous_probability >= CANCER_THRESHOLD:
    prediction = "Cancerous"
else:
    prediction = "Non-Cancerous"
```

### Examples

| Cancerous Probability | Non-Cancerous Probability | Prediction (65% threshold) |
|----------------------|--------------------------|----------------------------|
| 70% | 30% | ✅ **Cancerous** |
| 65% | 35% | ✅ **Cancerous** |
| 64% | 36% | ✅ **Non-Cancerous** |
| 55% | 45% | ✅ **Non-Cancerous** |
| 80% | 20% | ✅ **Cancerous** |
| 40% | 60% | ✅ **Non-Cancerous** |

---

## ⚙️ Configuration

### Current Setting

**File**: `web_app/app.py`  
**Line**: ~30

```python
# Classification threshold
# Probability threshold for cancer classification
# If cancerous probability >= THRESHOLD -> Classified as Cancerous
# If cancerous probability < THRESHOLD -> Classified as Non-Cancerous
CANCER_THRESHOLD = 0.65  # 65%
```

### How to Change

Simply edit the `CANCER_THRESHOLD` value and restart the Flask server:

```python
# Conservative (fewer false positives, may miss some cancers)
CANCER_THRESHOLD = 0.75  # 75%

# Balanced (current setting)
CANCER_THRESHOLD = 0.65  # 65%

# Sensitive (catch more cancers, more false positives)
CANCER_THRESHOLD = 0.50  # 50%

# Very sensitive (catch almost all cancers, many false positives)
CANCER_THRESHOLD = 0.40  # 40%
```

---

## 📊 Threshold Impact Analysis

### High Threshold (e.g., 75%)

**Advantages:**
- ✅ Fewer false positives
- ✅ Higher confidence in positive diagnoses
- ✅ Reduces unnecessary follow-up procedures

**Disadvantages:**
- ❌ May miss some early-stage cancers
- ❌ Lower sensitivity
- ❌ More false negatives

**Use Case**: When false positives are costly (e.g., psychological impact, follow-up costs)

---

### Medium Threshold (e.g., 65%) - **CURRENT**

**Advantages:**
- ✅ Balanced approach
- ✅ Good sensitivity and specificity
- ✅ Recommended for general screening

**Disadvantages:**
- ⚠️ May still have some false positives and negatives

**Use Case**: General lung cancer screening with balanced priorities

---

### Low Threshold (e.g., 50%)

**Advantages:**
- ✅ High sensitivity (catches most cancers)
- ✅ Fewer missed diagnoses
- ✅ Good for high-risk populations

**Disadvantages:**
- ❌ More false positives
- ❌ More follow-up tests required
- ❌ Potential for overtreatment

**Use Case**: High-risk populations where missing a cancer is very costly

---

## 🧮 Implementation Details

### Individual Model Predictions

```python
def predict_with_model(model, img_tensor, model_name, threshold=0.65):
    # Get model output probabilities
    probabilities = torch.softmax(outputs, dim=1)
    cancerous_prob = probabilities[0][1].item()
    
    # Apply threshold
    predicted_class = 1 if cancerous_prob >= threshold else 0
    
    # Calculate confidence based on prediction
    if predicted_class == 1:
        confidence = cancerous_prob * 100
    else:
        confidence = (1 - cancerous_prob) * 100
    
    return {
        'prediction': 'Cancerous' if predicted_class == 1 else 'Non-Cancerous',
        'confidence': confidence,
        'probabilities': {
            'non_cancerous': (1 - cancerous_prob) * 100,
            'cancerous': cancerous_prob * 100
        },
        'threshold': threshold * 100
    }
```

### Ensemble Predictions

The ensemble prediction also uses the same threshold:

```python
# Average cancerous probability from all models
avg_cancerous = np.mean([r['probabilities']['cancerous'] for r in valid_results])

# Apply threshold to ensemble
ensemble_pred = 'Cancerous' if avg_cancerous >= (CANCER_THRESHOLD * 100) else 'Non-Cancerous'
```

---

## 🖥️ Frontend Display

The web interface now displays the threshold information in the results:

```
┌─────────────────────────────────┐
│ 🧠 DenseNet169                  │
├─────────────────────────────────┤
│ ✅ Non-Cancerous                │
│ Confidence: 92.15%              │
│                                 │
│ 📏 Classification Threshold: 65%│
│ (≥ 65% cancerous → Cancerous)   │
│                                 │
│ ✅ Non-Cancerous: 92.15%        │
│ ⚠️  Cancerous: 7.85%            │
└─────────────────────────────────┘
```

---

## 📈 Recommended Thresholds by Use Case

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| **General Screening** | 60-65% | Balanced sensitivity/specificity |
| **High-Risk Population** | 50-55% | Prioritize catching all cancers |
| **Follow-up Screening** | 70-75% | Reduce false positives |
| **Research Studies** | 50% | Standard binary classification |
| **Clinical Decision Support** | 65-70% | High confidence needed |

---

## 🔧 Testing Different Thresholds

### Method 1: Change Configuration

1. Edit `web_app/app.py`
2. Change `CANCER_THRESHOLD = 0.65` to desired value
3. Restart Flask server
4. Test with sample images

### Method 2: Add API Parameter (Future Enhancement)

Could add threshold as API parameter:

```python
# In API endpoint
threshold = float(request.form.get('threshold', CANCER_THRESHOLD))
result = predict_with_model(model, img_tensor, model_name, threshold=threshold)
```

Then frontend could have threshold slider.

---

## 📊 Threshold Optimization

### ROC Curve Analysis

The optimal threshold should be determined using:
- **ROC Curve**: Plot True Positive Rate vs False Positive Rate
- **AUC Score**: Area Under Curve
- **Youden's Index**: Maximize (Sensitivity + Specificity - 1)

### Threshold Selection Criteria

```python
# Example optimization code
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)  # Youden's Index
optimal_threshold = thresholds[optimal_idx]
```

---

## 🚨 Important Considerations

### Medical Context

⚠️ **This is a research tool, not a diagnostic tool.**

- Threshold should be validated with clinical data
- Consider consultation with medical professionals
- Follow regulatory guidelines for medical AI
- Document threshold selection rationale

### Performance Metrics

Monitor these metrics when changing threshold:

- **Sensitivity (Recall)**: True Positives / (True Positives + False Negatives)
- **Specificity**: True Negatives / (True Negatives + False Positives)
- **Precision**: True Positives / (True Positives + False Positives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **AUC**: Area Under ROC Curve

---

## 📝 Logging

The system logs threshold information:

```
Console Output:
==================================================
Lung Cancer Detection System
==================================================
Models loaded: 3
Cancer classification threshold: 65.0%
Device: cpu
==================================================

Prediction Output:
  Predicting with densenet...
    Result: Non-Cancerous (92.15%) [Threshold: 65.0%]
  
  Ensemble result: Non-Cancerous (Cancerous: 8.3%, Threshold: 65.0%)
```

---

## 🔄 Version History

| Version | Threshold | Date | Notes |
|---------|-----------|------|-------|
| 1.0 | 50% | 2025-10-01 | Initial implementation |
| 2.0 | 65% | 2025-10-06 | **Current** - Balanced threshold |

---

## 📚 References

1. **Medical Imaging Thresholds**: Best practices from LUNA16 challenge
2. **ROC Analysis**: Fawcett, T. (2006). ROC Analysis
3. **Threshold Selection**: Youden's J statistic method

---

## ✅ Quick Reference

### Change Threshold

```bash
# 1. Edit config
nano web_app/app.py

# 2. Change line ~30
CANCER_THRESHOLD = 0.65  # Change this value

# 3. Restart server
cd web_app
python app.py
```

### Current Settings

- **Threshold**: 65%
- **Logic**: `if cancerous_prob >= 0.65 → Cancerous`
- **Display**: Shown in results card
- **Applies to**: Individual models + Ensemble

---

**Configuration updated**: October 6, 2025  
**Current status**: ✅ Active at 65% threshold  
**Next review**: Validate with clinical data
