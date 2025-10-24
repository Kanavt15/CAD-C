# Cancer Detection Threshold Guide

## Current Configuration

**Threshold:** 35% (Optimized for medical use)

### Why 35%?

In medical diagnosis, it's **better to flag a case for further review** than to miss a cancerous case. This is called "high sensitivity, acceptable specificity."

- **Model Recall:** 83% (catches 83% of cancerous cases)
- **Model Precision:** 55% (when it says cancer, it's correct 55% of the time)

### How Threshold Works

```
If cancerous_probability >= 35% → Classified as "Cancerous"
If cancerous_probability < 35% → Classified as "Non-Cancerous"
```

## Threshold Impact

### Lower Threshold (e.g., 25-35%)
- ✅ **Higher Sensitivity** - Catches more cancerous cases
- ✅ **Fewer False Negatives** - Misses fewer cancer cases
- ⚠️ **More False Positives** - May flag some healthy cases as suspicious
- **Use case:** Screening, initial diagnosis

### Default Threshold (50%)
- **Balanced** approach
- Equal weight to both classes
- **Use case:** General purpose

### Higher Threshold (e.g., 60-75%)
- ✅ **Higher Specificity** - More confident when saying "cancer"
- ✅ **Fewer False Positives** - Less likely to incorrectly flag healthy
- ⚠️ **More False Negatives** - May miss some cancer cases
- **Use case:** Confirmation after other tests

## Adjusting the Threshold

### Option 1: Edit app.py (Permanent)
```python
# In app.py, line ~35
CANCER_THRESHOLD = 0.35  # Change this value (0.0 to 1.0)
```

Then restart the server.

### Option 2: Common Threshold Values

| Threshold | Use Case | Sensitivity | Specificity |
|-----------|----------|-------------|-------------|
| 0.25 | Very High Sensitivity | Very High | Low |
| 0.35 | **Current (Recommended)** | High | Moderate |
| 0.50 | Balanced | Moderate | Moderate |
| 0.65 | High Specificity | Moderate | High |
| 0.75 | Very High Specificity | Low | Very High |

## Understanding the Output

When you get a prediction, you'll see:

```json
{
  "prediction": "Cancerous",  // Based on threshold
  "confidence": 65.5,         // Confidence in the prediction
  "probabilities": {
    "non_cancerous": 34.5,    // Raw model output
    "cancerous": 65.5         // Raw model output (if >= 35%, classified as cancerous)
  },
  "threshold": 35.0           // Threshold used
}
```

### Example Scenarios

**Scenario 1:** Model outputs 40% cancerous
- With 35% threshold → **Classified as: Cancerous** ✅
- With 50% threshold → Classified as: Non-Cancerous ❌

**Scenario 2:** Model outputs 30% cancerous
- With 35% threshold → Classified as: Non-Cancerous
- With 25% threshold → **Classified as: Cancerous** ✅

**Scenario 3:** Model outputs 70% cancerous
- With any threshold ≤ 70% → **Classified as: Cancerous** ✅

## Debugging Predictions

Check the terminal output when making predictions:

```
Prediction: Cancerous
Confidence: 65.50%
Cancerous probability: 65.50%
Non-Cancerous probability: 34.50%
Threshold used: 35.0%
```

### If predictions seem wrong:

1. **Check the raw probabilities** in terminal output
2. **Compare with threshold** - Is cancerous_prob >= threshold?
3. **Adjust threshold** if needed based on your use case
4. **Verify input image** - Is it the correct image?

## Medical Context

⚠️ **Important Medical Note:**

This AI system is a **screening tool**, not a diagnostic tool. It should be used to:
- Flag cases for further review by medical professionals
- Support clinical decision-making
- Prioritize cases for urgent review

It should **NOT** be used as:
- The sole basis for diagnosis
- A replacement for medical imaging specialists
- A definitive diagnostic tool

Always have a qualified radiologist review flagged cases.

## Quick Settings Reference

**For most medical screening:**
```python
CANCER_THRESHOLD = 0.35  # Current setting
```

**For research/testing:**
```python
CANCER_THRESHOLD = 0.50  # Balanced
```

**For secondary confirmation:**
```python
CANCER_THRESHOLD = 0.65  # More conservative
```
