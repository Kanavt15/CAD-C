# 🎯 Quick Prediction Logic Summary

## Decision Tree (Both Models)

```
Input Image → Model Prediction → Cancerous Probability
                                         |
                    _____________________|_____________________
                   |                     |                     |
              ≥ 32%                 25% - 32%               < 25%
                   |                     |                     |
            🔴 CANCEROUS         🟡 SUSPICIOUS        🟢 NON-CANCEROUS
                   |                     |                     |
          Confidence = X%        Confidence = X%       Confidence = (100-X)%
          (Cancer prob)          (Cancer prob)         (Non-cancer prob)
                   |                     |                     |
            Flag for review      ⚠️  Medical review     Likely benign
```

---

## Threshold Consistency ✅

Both **ResNet3D** and **DenseNet3D** now use:

| Zone | Cancer Probability | Prediction | Action |
|------|-------------------|------------|--------|
| 🔴 High | **≥ 32%** | Cancerous | Immediate medical review |
| 🟡 Medium | **25% - 32%** | Suspicious | Medical review recommended |
| 🟢 Low | **< 25%** | Non-Cancerous | Likely safe, routine follow-up |

---

## ✅ Fixed Issues

### Before:
- ❌ DenseNet used 35% for suspicious zone (inconsistent with ResNet's 25%)
- ❌ No clear comments explaining the logic

### After:
- ✅ Both models use **25%** as suspicious zone lower bound
- ✅ Both models use **32%** as main cancer threshold  
- ✅ Clear inline comments explaining each decision zone
- ✅ Consistent behavior across both models

---

## 🧠 Understanding "Confidence"

**Important**: Confidence means different things based on the prediction!

### For "Cancerous" or "Suspicious":
```
Confidence = Cancerous Probability
Example: "Cancerous (45% confidence)" 
        → Model detected 45% cancer signal
```

### For "Non-Cancerous":
```
Confidence = Non-Cancerous Probability  
Example: "Non-Cancerous (85% confidence)"
        → Model is 85% confident it's NOT cancer
```

**Why different?**
- We show the **relevant probability** for medical decision-making
- For cancer detection: How strong is the cancer signal?
- For benign cases: How confident are we it's safe?

---

## 📊 Model Comparison

### ResNet3D
- **Accuracy**: 83.3%
- **Recall**: 83% (catches more cancers)
- **Precision**: ~55% (more false positives)
- **Best for**: Initial screening, high sensitivity needed

### DenseNet3D ⭐ (Recommended)
- **Accuracy**: 95.73%
- **Recall**: 71% (good detection rate)
- **Precision**: 86% (fewer false positives)
- **Best for**: Reliable diagnosis, balanced performance

---

## 🎯 Clinical Use Cases

### Use Case 1: Population Screening
- **Goal**: Don't miss any cancer cases
- **Strategy**: Flag anything ≥ 25% (Suspicious or Cancerous)
- **Rationale**: Better to over-detect and review than miss a case

### Use Case 2: Confirmatory Testing
- **Goal**: Reduce false positives
- **Strategy**: Focus on ≥ 32% (Cancerous only)
- **Rationale**: Higher threshold reduces unnecessary biopsies

### Use Case 3: Research/Analysis
- **Goal**: Understand model behavior
- **Strategy**: Examine all three zones separately
- **Rationale**: Study the uncertainty region (25-32%)

---

## 🚀 Quick Start

1. **Upload image** to the web app
2. **Both models analyze** simultaneously
3. **Review results**:
   - If both agree → High confidence
   - If they disagree → Needs medical review
4. **All Cancerous/Suspicious cases** → Send to radiologist

---

## 📝 Code Location

The prediction logic is in `web_app/app.py`:

- **ResNet prediction**: `predict_resnet()` function (lines ~240-290)
- **DenseNet prediction**: `predict_densenet()` function (lines ~305-360)
- **Thresholds defined**: Top of file (lines 20-22)

---

**Last Updated**: After threshold unification (32% for both models)
