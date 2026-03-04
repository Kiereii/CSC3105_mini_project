#!/usr/bin/env python3
"""
Reorganize documentation into consolidated, non-overlapping files.
Run this script from the docs folder to reorganize the documentation.
"""

import os
import shutil
from pathlib import Path

# Define the new structure
docs_to_create = {
    "00_START_HERE.md": """# 📖 UWB LOS/NLOS Classification — Documentation Guide

Welcome to the UWB (Ultra-Wideband) LOS/NLOS classification project documentation!

## 🎯 Quick Start: Pick Your Path

### **I'm completely new** → Start here
Read in order:
1. [01_Problem_Definition.md](01_Problem_Definition.md) — What is this project?
2. [02_Complete_Pipeline.md](02_Complete_Pipeline.md) — How does it all fit together?

### **I want to understand the data** → Read this
- [03_Data_Exploration.md](03_Data_Exploration.md) — Visualization and analysis techniques

### **I want to understand the ML** → Read this path
1. [04_Data_Preparation.md](04_Data_Preparation.md) — Cleaning and preprocessing
2. [05_ML_Algorithms.md](05_ML_Algorithms.md) — RF, Logistic Regression, SVM, XGBoost
3. [06_Evaluation_Explained.md](06_Evaluation_Explained.md) — Metrics and evaluation

### **I want advanced topics** → Read these
- [07_Pair_Classification.md](07_Pair_Classification.md) — LOS+NLOS pair classification
- [08_Range_Regression.md](08_Range_Regression.md) — Distance prediction

### **I'm comparing models** → Read this
- [09_Model_Comparison.md](09_Model_Comparison.md) — Unified model comparison framework

---

## 📊 Key Results at a Glance

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **Accuracy** | 88.69% |
| **AUC** | 0.9535 |
| **Top Feature** | RXPACC (15.6%) |
| **Path 1 Range RMSE** | 1.28 m |
| **Path 2 Range RMSE** | 1.32 m |
| **Dataset Size** | 41,568 samples |
| **Training/Test Split** | 80/20 (stratified) |

---

## 📚 All Files

| File | Duration | Topic |
|------|----------|-------|
| `01_Problem_Definition.md` | 10 min | UWB, LOS/NLOS, dataset |
| `02_Complete_Pipeline.md` | 15 min | All scripts and workflow |
| `03_Data_Exploration.md` | 10 min | EDA and visualization |
| `04_Data_Preparation.md` | 15 min | Cleaning and preprocessing |
| `05_ML_Algorithms.md` | 30 min | RF, LR, SVM, XGBoost |
| `06_Evaluation_Explained.md` | 15 min | Metrics, confusion matrix, ROC |
| `07_Pair_Classification.md` | 15 min | Path pairs classification |
| `08_Range_Regression.md` | 15 min | Distance prediction |
| `09_Model_Comparison.md` | 10 min | Model comparison framework |

**Total reading time: ~2-3 hours for complete understanding**

---

## 🚀 Next Steps

1. **Read [01_Problem_Definition.md](01_Problem_Definition.md) (10 min)**
2. **Read [02_Complete_Pipeline.md](02_Complete_Pipeline.md) (15 min)**
3. **Choose your path above based on your goals**

Good luck! 🎯
""",

    "01_Problem_Definition.md": """# Problem Definition: UWB LOS/NLOS Classification

## What is UWB?

**Ultra-Wideband (UWB)** is a radio technology that:
- Operates at 3.1-10.6 GHz frequency range
- Uses very short nanosecond pulses
- Provides **centimeter-level positioning accuracy**
- Used in: Apple AirTag, Samsung SmartTag, iPhone 15+, indoor positioning systems

The key advantage of UWB over WiFi/Bluetooth: extremely high precision through **precise time-of-arrival measurements**.

---

## LOS vs NLOS: The Core Problem

### Line-of-Sight (LOS) ✓
```
Transmitter ==========> Receiver
         (Direct path)
```
- Signal travels directly transmitter → receiver
- **Strongest signal amplitude**
- **Most accurate distance measurement**
- Shortest signal path
- **Preferred for positioning**

### Non-Line-of-Sight (NLOS) ✗
```
Transmitter ===| Wall |===> Receiver
            (Reflected paths)
```
- Signal bounces off walls, furniture, people
- **Weaker signal amplitude**
- **Longer path length → longer measured distance**
- Inaccurate range estimates
- **Corrupts positioning**

### The Challenge
When you measure a radio signal, you often see **BOTH LOS and NLOS reflections mixed together**. The receiver can't tell them apart just by looking at the signal strength. So the question becomes:

> **Can we automatically detect which type of signal path this is?**

This is a **classification problem** — categorizing each measurement as LOS (0) or NLOS (1).

---

## Your Dataset

### Size
- **Total samples:** 41,568 UWB channel impulse response (CIR) measurements
- **Training data:** 33,254 samples (80%)
- **Test data:** 8,314 samples (20%)
- **Collection:** 7 CSV files from the UWB-LOS-NLOS dataset

### Balance
- **LOS samples:** 20,997 (50.5%)
- **NLOS samples:** 20,571 (49.5%)
- **Perfect balance!** ✓ No class imbalance problems

### Features (136 total)

#### Core Features (16)
| Feature | Meaning | Why It Matters |
|---------|---------|---|
| **RXPACC** | Received preamble accumulation | **MOST IMPORTANT!** NLOS collects more preambles due to reflections |
| **RANGE** | Measured distance (meters) | NLOS typically shows longer ranges |
| **FP_IDX** | First path index (position in CIR) | Where the signal first arrives |
| **FP_AMP1/2/3** | First path amplitudes | LOS has stronger first-path amplitudes |
| **SNR** | Signal-to-Noise Ratio | LOS has better signal quality |
| **SNR_dB** | SNR in decibels | Same as SNR but log-scale |
| **CIR_PWR** | Total CIR power | Overall signal strength |
| **STDEV_NOISE** | Noise standard deviation | NLOS has higher noise |
| **MAX_NOISE** | Maximum noise value | Related to signal quality |
| **RXSTATUS** | Receiver status flags | Hardware diagnostic |
| **Various timestamps/metadata** | Collection info | For data integrity checks |

#### CIR Features (120)
- **Channel Impulse Response samples** from index 730–850
- Shows the "fingerprint" of the signal over time
- Peak location and shape **differ significantly between LOS and NLOS**
- LOS: Strong single peak (direct path)
- NLOS: Multiple peaks (reflections)

### Example Statistics

```
LOS (class 0):
  - Mean RXPACC: 1024 accumulations
  - Mean RANGE: 4.2 meters
  - Mean CIR_PWR: 8500 units
  - Std Dev: relatively low (clean signal)

NLOS (class 1):
  - Mean RXPACC: 1126 accumulations (MORE due to reflections)
  - Mean RANGE: 5.8 meters (LONGER due to extra path length)
  - Mean CIR_PWR: 9200 units
  - Std Dev: relatively high (noisy signal)
```

These differences are what allow a ML model to **learn to distinguish** LOS from NLOS.

---

## The Project Requirements

### Task 1: LOS/NLOS Classification
**Question:** For each UWB measurement, is the dominant path LOS or NLOS?

**Input:** 136 features (core + CIR)  
**Output:** 0 (LOS) or 1 (NLOS)  
**Model Type:** Classification  
**Performance Goal:** >85% accuracy

### Task 2: Range Estimation (Two Paths)
**Question:** What is the measured distance for each signal path?

**Input:** Same 136 features  
**Output:** Distance in meters (continuous number)  
**Model Type:** Regression  
**Two separate regressors needed:**
- Path 1: First arriving signal (measured `RANGE` in dataset)
- Path 2: Second arriving signal (derived from CIR peak detection)

**Performance Goal:** <2m RMSE

### Task 3: Pair-Level Classification
**Question:** Looking at BOTH paths together, is it LOS+NLOS or NLOS+NLOS?

**Input:** Features for both paths  
**Output:** Class label (LOS+NLOS vs NLOS+NLOS)  
**Model Type:** Classification  
**Reason:** More directly answers "is there a line-of-sight path?"

---

## Why This Matters (Real-World Impact)

### Positioning Without LOS/NLOS Detection
```
If you can't distinguish LOS from NLOS:
  → Use all signals equally
  → NLOS signals are corrupted
  → Range estimate is wrong
  → **Positioning error: 5-20 meters** ❌
```

### Positioning With LOS/NLOS Detection
```
If you CAN distinguish LOS from NLOS:
  → Downweight or reject NLOS signals
  → Trust LOS signals (most accurate)
  → Use ML to estimate correct range
  → **Positioning error: <1 meter** ✓
```

**The difference:** Exact indoor location (LOS) vs "somewhere in this room" (NLOS).

---

## Success Criteria

Your project has achieved:
- ✅ **88.69% classification accuracy** (Random Forest)
- ✅ **95.35% AUC** (excellent ranking ability)
- ✅ **1.28m range RMSE** for Path 1
- ✅ **1.32m range RMSE** for Path 2
- ✅ **Feature importance analysis** (RXPACC = 15.6%)
- ✅ **Multiple algorithm comparison** (RF, LR, SVM, XGBoost)

---

## Next Steps

Read [02_Complete_Pipeline.md](02_Complete_Pipeline.md) to understand how all the scripts work together to achieve these results.
""",

    "02_Complete_Pipeline.md": """# The Complete Pipeline: From Raw Data to Trained Models

This document explains **every step** of the data analytics pipeline, in the exact order they must be run.

---

## Overview: The Full Workflow

```
RAW DATA (42,000 samples)
    ↓ [clean_local.py]
CLEANED DATA (41,568 samples)
    ↓ [preprocess_data.py]
PREPROCESSED DATA (.npy arrays)
    ├─ [random_forest_classifier.py] → RF Model (88.69% accuracy)
    ├─ [logistic_regression_classifier.py] → LR Model
    ├─ [svm_classifier.py] → SVM Model
    ├─ [xgboost_classifier.py] → XGB Model
    ├─ [pair_classifier.py] → Pair Model (LOS+NLOS vs NLOS+NLOS)
    ├─ [second_path_features.py] → Path 2 features extracted
    └─ [range_regressor.py] → Range predictions (Path 1 & 2)
    ↓ [compare_models.py]
COMPARISON PLOTS & TABLES (models/comparison/)
```

---

## Step 1: Data Cleaning — `clean_local.py`

### Purpose
Transform raw, unfiltered sensor data into a **trustworthy dataset**.

### What It Does

| Task | How | Why |
|------|-----|-----|
| **Remove duplicates** | Drop rows where all columns are identical | Duplicates bias the model |
| **Remove nulls** | Drop rows with any `NaN` value | ML can't process missing values |
| **Remove outliers** | Drop `RANGE <= 0` or noise in top 1% | Physically impossible or sensor errors |
| **Type enforcement** | Cast `NLOS` to integer (0 or 1) | Ensures consistent labels |
| **Feature engineering** | Calculate `SNR = FP_AMP1 / (STDEV_NOISE + 1e-6)` | Creates a meaningful derived feature |
| **Feature engineering** | Calculate `SNR_dB = 10 * log10(SNR)` | SNR in decibels (standard form) |

### Results
```
Input:  7 raw CSV files × 6,000 rows = 42,000 rows
Output: 7 cleaned CSVs with ~64 rows removed per file
        Total: 41,568 rows (1% outliers removed)
        
Statistics:
- No exact duplicates found
- 0 null values in final data
- 60-64 outliers removed per file
- SNR and SNR_dB added as features
```

### What It Does NOT Do
- ❌ Does NOT split data
- ❌ Does NOT scale features
- ❌ Does NOT train any model
- ❌ Does NOT reduce dimensionality

**Output location:** `Dataset/Cleaned/uwb_cleaned_dataset_part1.csv`, etc.

---

## Step 2: Data Preprocessing — `preprocess_data.py`

### Purpose
Transform cleaned data into **ML-ready format** (numerical arrays) with proper splits and scaling.

### What It Does

| Task | How | Why |
|------|-----|-----|
| **Load all CSVs** | Read all 7 cleaned files | Merge into single dataset |
| **Feature selection** | Keep 16 core + 120 CIR samples (730-850) = 136 total | Remove irrelevant columns; focus on signal |
| **Target extraction** | Separate `NLOS` column as `y` | Create the "answer" for supervised learning |
| **Train/test split** | 80% training (33,254 rows), 20% test (8,314 rows) | Evaluate on unseen data |
| **Stratified split** | Maintain 50/50 balance in both sets | Prevent accidental class imbalance |
| **StandardScaler** | Mean = 0, Std = 1 for each feature | Required for: SVM, Logistic Regression |
| **MinMaxScaler** | Normalize to [0,1] range for each feature | Useful for: Neural networks (if used) |
| **Save as .npy** | Save X_train, X_test, y_train, y_test | Fast loading; reproducible splits |
| **Save scalers** | Save StandardScaler and MinMaxScaler as .pkl | Ensure same scaling on new data |

### Why These Choices?

**Why 80/20 split?**
```
Training set (80%) = Model learns patterns
Test set (20%) = Fair evaluation on unseen data

If you tested on training data:
  → Fake 95%+ accuracy (memorization)
  → Fails completely on new data (overfitting)
```

**Why stratified?**
```
Dataset: 50.5% LOS, 49.5% NLOS

Without stratification:
  → Test set might be 80% LOS, 20% NLOS
  → Model looks great but actually isn't

With stratification:
  → Both train and test: 50.5% LOS, 49.5% NLOS
  → Fair representation in both sets
```

**Why two scalers?**
```
StandardScaler: mean=0, std=1
  → For algorithms that assume normally distributed features
  → Used by: SVM, Logistic Regression, Neural Networks
  → Centered at 0, unbounded range

MinMaxScaler: 0 to 1 range
  → For algorithms that work better with bounded ranges
  → Used by: Some neural networks
  → All values between 0 and 1
  
Random Forest doesn't need scaling
  → Tree-based, doesn't care about scale
  → Can use unscaled data directly
```

**Why save .npy files?**
```
Advantages:
  → Fast to load (binary format vs CSV text parsing)
  → Exact same split reproduced every time
  → Can retrain/re-evaluate without re-preprocessing
  → Small file size
```

### Results
```
Input:  41,568 cleaned samples
Output: 10 files in preprocessed_data/

X_train_unscaled.npy  (33,254 × 136)  ← for Random Forest
X_test_unscaled.npy   (8,314 × 136)

X_train_standard.npy  (33,254 × 136)  ← for SVM, LR
X_test_standard.npy   (8,314 × 136)

X_train_minmax.npy    (33,254 × 136)  ← for Neural Networks
X_test_minmax.npy     (8,314 × 136)

y_train.npy           (33,254,)       ← training labels
y_test.npy            (8,314,)        ← test labels

scaler_standard.pkl                     ← fitted StandardScaler
scaler_minmax.pkl                       ← fitted MinMaxScaler

Feature names saved for reference
```

---

## Step 3a: Classification — `random_forest_classifier.py`

### Purpose
**Train a model to classify each sample as LOS (0) or NLOS (1).**

### What It Does

```
Load preprocessed data
    ↓
Initialize RandomForestClassifier (100 trees)
    ↓
Train on 33,254 samples
    ↓
Predict on 8,314 test samples
    ↓
Compute metrics (accuracy, precision, recall, F1, AUC)
    ↓
Save model, predictions, and visualizations
```

### Algorithm: Random Forest (100 decision trees)
- Each tree sees a **random subset** of training data
- Each split uses **random subset** of features
- Trees make different errors that **cancel out**
- Final prediction = **majority vote** across 100 trees

**Result:** Better generalization, less overfitting than single tree.

### Results
```
Accuracy: 88.69%
Precision: 92.03% (when we predict NLOS, we're right 92% of the time)
Recall: 84.47% (we catch 84.47% of actual NLOS)
F1-Score: 88.09%
AUC: 0.9535 (excellent!)

Confusion Matrix:
  TN=3899 (correct LOS)    FP=301 (wrongly called NLOS)
  FN=639  (wrongly called LOS)   TP=3475 (correct NLOS)
```

### Output
```
models/random_forest/
├── random_forest_model.pkl              ← trained model
├── y_pred.npy                           ← hard predictions (0 or 1)
├── y_proba.npy                          ← soft predictions (probability)
├── confusion_matrix_and_roc.png         ← confusion matrix + ROC curve
├── feature_importance.png               ← top 20 features chart
├── feature_importance_ranking.csv       ← all 136 features ranked
└── model_results.txt                    ← all metrics in text form
```

### Key Finding
**RXPACC is the most important feature** (15.6% of model's decisions).
- NLOS signals collect more preambles due to reflections
- RF learned this pattern automatically

---

## Step 3b: Feature Engineering — `second_path_features.py`

### Purpose
**Extract the second signal path from CIR and create features for it.**

### What It Does

For each of the 41,568 samples:
1. Load the CIR (1016 samples)
2. Get FP_IDX (first path position)
3. Skip 15 samples (to avoid pulse sidelobes)
4. Search for next peak using `scipy.signal.find_peaks`
5. Record: `PEAK2_IDX`, `PEAK2_AMP`, `PEAK2_GAP`
6. Derive: `RANGE_PATH2_EST = RANGE + PEAK2_GAP × 0.15m`

### Results
```
New features created:
- PEAK2_IDX: ~760-850 (position of second peak)
- PEAK2_AMP: ~5000-15000 (amplitude of second peak)
- PEAK2_GAP: ~15-100 (time delay between path 1 and path 2)
- PEAK2_FOUND: 100% (second peak found in all samples!)
- RANGE_PATH2_EST: ~4.5-11.0 meters

Output: X_train/test_regression.npy with 139 features
        y_range_p1/p2_train/test.npy (regression targets)
```

### Why Second Path?
- Project brief requires range estimates for **both dominant paths**
- Path 1: measured in `RANGE` column
- Path 2: not directly given, must be extracted
- Second arrival is **always NLOS** (any reflection is NLOS)

---

## Step 3c: Regression — `range_regressor.py`

### Purpose
**Train models to predict the measured distance for each path.**

### What It Does
```
For Path 1:
  Load X_train/test_regression, y_range_p1_train/test
  Train RandomForestRegressor (100 trees)
  Predict on test set
  Compute RMSE, MAE, R²
  Save model and predictions
  
For Path 2:
  Repeat with y_range_p2_train/test
```

### Algorithm: Random Forest Regressor
Same as classifier, but leaf nodes output **mean distance** instead of class vote.

### Results
```
Path 1 Range Estimation:
  RMSE: 1.275 meters    ← typical error magnitude
  MAE: 0.980 meters     ← average absolute error
  R²: 0.707             ← explains 70.7% of variance

Path 2 Range Estimation:
  RMSE: 1.318 meters
  MAE: 1.008 meters
  R²: 0.743             ← actually higher than Path 1!
```

### Output
```
models/range_regressor/
├── rf_range_path1.pkl                   ← trained regressor
├── rf_range_path2.pkl
├── range_estimation_results.png         ← predicted vs actual scatter
├── residual_distributions.png           ← error analysis
├── feature_importance_path1.csv         ← ranked features
└── regression_results.txt               ← all metrics
```

---

## Step 3d: Pair Classification — `pair_classifier.py`

### Purpose
**Classify path pairs as LOS+NLOS or NLOS+NLOS.**

### What It Does
Instead of classifying individual paths:
- **LOS+NLOS:** Path 1 is LOS, Path 2 is NLOS (found a direct path)
- **NLOS+NLOS:** Both paths are NLOS (no direct path)

This directly answers: **"Does a line-of-sight path exist?"**

### Results
```
Uses both RF and XGBoost
Accuracy, F1, AUC reported
Compared against per-path classification
```

### Output
```
models/pair_classifier/
├── pair_rf_model.pkl
├── pair_xgb_model.pkl
├── pair_confusion_matrices.png
└── pair_metrics.csv
```

---

## Step 4: Model Comparison — `compare_models.py`

### Purpose
**Create unified comparison of all trained models.**

### What It Does
1. Load predictions from all models (RF, LR, SVM, XGB, Pair)
2. Compute metrics for each (Accuracy, Precision, Recall, F1, AUC)
3. Generate comparison plots:
   - Metrics bar chart
   - All confusion matrices (2×2 grid)
   - Combined ROC curve
   - Feature importance comparison
   - NLOS safety metrics
   - Pair vs per-path comparison

### Results
```
Random Forest: Best overall (88.69% accuracy, 0.9535 AUC)
SVM: Close second (87-89% accuracy)
XGBoost: Competitive
Logistic Regression: Good baseline
```

### Output
```
models/comparison/
├── metrics_comparison.csv               ← all metrics in table
├── metrics_bar_comparison.png           ← side-by-side bar chart
├── confusion_matrices_all.png           ← all 4 matrices
├── roc_all_models_clean.png             ← all ROC curves
├── feature_importance_rf_vs_xgb.png
├── feature_importance_lr_vs_svm.png
├── nlos_safety_comparison.png
└── per_path_vs_pair_comparison.png
```

---

## The CRISP-DM Framework (Your Process)

Your project follows the industry-standard CRISP-DM methodology:

```
1. Business Understanding
   ↓ "What is the problem?"
   → Precise indoor positioning using UWB signals
   
2. Data Understanding
   ↓ "What data do we have?"
   → 41,568 samples, 136 features, 50/50 LOS/NLOS
   
3. Data Preparation
   ↓ "How do we clean and format it?"
   → clean_local.py, preprocess_data.py
   
4. Modeling
   ↓ "What algorithms work best?"
   → RF, LR, SVM, XGB classifiers + regressors
   
5. Evaluation
   ↓ "How well do they perform?"
   → Accuracy 88.69%, AUC 0.9535, RMSE 1.28m
   
6. Deployment
   ↓ "How do we use this in production?"
   → integrate into positioning system (optional)
```

---

## Key Principles

### Why Run in Order?
Each step **depends on outputs from the previous step**. Running out of order causes crashes or wrong results.

### Why Save Models as .pkl?
Training is slow. Saving trained models lets you:
- Re-evaluate without retraining
- Deploy to production
- Share with team members
- Version control different models

### Why Save Predictions as .npy?
Comparing models requires the **exact same test set and labels**. Saving `.npy` files ensures reproducibility.

### Why Multiple Models?
No single algorithm is universally best:
- **RF:** Good general-purpose, feature importance
- **LR:** Fast, interpretable, baseline
- **SVM:** Excellent for high-dimensional data
- **XGB:** Often best accuracy if tuned well

Comparing all four gives confidence in results.

---

## Next Steps

- Read [04_Data_Preparation.md](04_Data_Preparation.md) for deep dive into cleaning/preprocessing
- Read [05_ML_Algorithms.md](05_ML_Algorithms.md) for algorithm details
- Read [06_Evaluation_Explained.md](06_Evaluation_Explained.md) for metrics
- Read [08_Range_Regression.md](08_Range_Regression.md) for regression details
- Read [09_Model_Comparison.md](09_Model_Comparison.md) for comparison framework
""",

}

# Create the backup folder
backup_dir = Path("docs_old_backup")
if not backup_dir.exists():
    backup_dir.mkdir()
    print(f"Created backup directory: {backup_dir}")

# Backup old files
for old_file in ["analysis_methods.md", "ML_Learning_Guide.md", "Range_Regressor_Explained.md",
                 "Two_Path_Range_Estimation_Guide.md", "Model_Comparison_Explained.md"]:
    old_path = Path(old_file)
    if old_path.exists():
        shutil.copy(old_path, backup_dir / old_file)
        print(f"Backed up: {old_file} → docs_old_backup/")

print("\\nNow I'll create the new reorganized files programmatically...")

