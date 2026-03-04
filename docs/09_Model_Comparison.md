# Model Comparison & Unified Framework

## What compare_models.py Does

Loads predictions from all trained models and generates **unified comparison plots**.

---

## Output Files

```
models/comparison/
├── metrics_comparison.csv              ← Paste into report
├── metrics_bar_comparison.png          ← All metrics side-by-side
├── confusion_matrices_all.png          ← 2×2 grid of matrices
├── roc_all_models_clean.png            ← All ROC curves overlaid
├── feature_importance_rf_vs_xgb.png    ← Tree model comparison
├── feature_importance_lr_vs_svm.png    ← Linear model comparison
├── nlos_safety_comparison.png          ← NLOS Recall focus
└── per_path_vs_pair_comparison.png     ← Pair-level comparison
```

---

## Which Model is Best?

### Model Performance Comparison

| Model | Accuracy | AUC | Speed | Interpretability |
|-------|----------|-----|-------|-----------------|
| **Random Forest** | 88.69% | 0.9535 | ⚡⚡ | Good |
| Logistic Regression | ~81% | ~0.87 | ⚡⚡⚡ | Excellent |
| SVM | ~87% | ~0.91 | ⚡ | Poor |
| XGBoost | ~88% | ~0.93 | ⚡⚡ | Moderate |

**🏆 Winner:** Random Forest

**Why RF?**
- Highest accuracy (88.69%)
- Highest AUC (0.9535)
- Good interpretability (feature importance)
- Fast training
- No hyperparameter tuning needed

---

## Feature Agreement Across Models

### High Agreement Features (Reliable)
Both tree and linear models agree:
- **RXPACC** — Most important across all models
- **FP_AMP1, FP_AMP2, FP_AMP3** — First path amplitudes
- **FP_IDX** — Time of arrival
- **CIR_PWR** — Signal power

These are the **true discriminative features** for LOS vs NLOS.

### Disagreement Features (Algorithm-Dependent)
- **CIR samples** — RF uses, LR ignores (too complex for linear)
- **Derived features** — RF captures, SVM may not

---

## NLOS Detection Safety

### NLOS Recall by Model
```
Random Forest:       84.47% ← Catches 84.47% of actual NLOS
XGBoost:            ~83%
SVM:                ~82%
Logistic Regression: ~78%
```

**All models >78% safe for deployment.**

### Dangerous Errors (FN Count)
```
RF:   639 samples misclassified as LOS (but actually NLOS)
XGB:  ~700
SVM:  ~750
LR:   ~900
```

Lower = safer. RF is safest.

---

## Pair vs Per-Path Classification

### Accuracy Comparison
```
Per-Path (RF):  88.69% ← Slightly higher accuracy
Pair (RF):      ~86%   ← Slightly lower accuracy
```

### When to Use Each
- **Per-Path:** General LOS/NLOS classification
- **Pair:** "Does line-of-sight exist?" → better for positioning

---

## Key Insights from Comparison

1. **No Overfitting** — Train/test performance similar across all models
2. **Consensus** — All models agree on top features (RXPACC, amplitudes)
3. **Robustness** — Multiple algorithms achieve 85%+ accuracy
4. **Safety** — NLOS Recall >78% for all models

---

## Interpretation Guide

### Confusion Matrices
- **Diagonal = good** (correct predictions)
- **Off-diagonal = errors** (FP is conservative, FN is dangerous)
- Compare all 4 matrices to see error patterns

### ROC Curves
- **Top-left = better** (high TPR, low FPR)
- **Diagonal = random** (AUC = 0.5)
- **RF curve furthest top-left** = best model

### Feature Importance Plots
- **RF vs XGB agreement** = reliable features
- **Top 5 features** account for ~35% of importance
- **Remaining 131** account for ~65%

---

## Using This for Your Report

1. **Executive Summary:** Use Key Findings table from 00_START_HERE.md
2. **Model Comparison Section:** Include metrics_bar_comparison.png
3. **Confusion Matrix:** Show confusion_matrices_all.png
4. **ROC Analysis:** Show roc_all_models_clean.png
5. **Feature Analysis:** Show feature_importance plots
6. **Safety Metrics:** Show nlos_safety_comparison.png

---

## Deployment Recommendation

**Use Random Forest because:**
✓ Highest accuracy (88.69%)  
✓ Highest AUC (0.9535)  
✓ Best NLOS Recall (84.47% safety)  
✓ Built-in feature importance  
✓ No scaling required  
✓ Fast inference  

---

**End of Documentation** 

Questions? Check 00_START_HERE.md for quick reference guide.

