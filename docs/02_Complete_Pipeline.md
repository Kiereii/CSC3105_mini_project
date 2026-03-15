# The Complete Pipeline: Scripts in Order

## Pipeline Overview

```
RAW DATA (42,000)
    ↓ [clean_local.py]
CLEANED (41,568)
    ↓ [preprocess_data.py]
PREPROCESSED (.npy arrays)
    ├─ [random_forest_classifier.py] → 88.69% accuracy
    ├─ [logistic_regression_classifier.py]
    ├─ [svm_classifier.py]
    ├─ [xgboost_classifier.py]
    ├─ [pair_classifier.py]
    ├─ [second_path_features.py]
    └─ [range_regressor.py]
    ↓ [compare_models.py]
COMPARISON PLOTS
```

---

## Step 1: Data Cleaning

**Script:** `clean_local.py`

**What it does:**
- Remove nulls, duplicates, outliers
- Add SNR features
- Ensure data quality

**Results:**
```
Input:  42,000 rows
Output: 41,568 rows
Removed: 432 rows (1.0%)
Status: ✓ Excellent data quality
```

---

## Step 2: Data Preprocessing

**Script:** `preprocess_data.py`

**What it does:**
- Feature selection (136 features)
- 80/20 stratified train/test split
- StandardScaler & MinMaxScaler
- Save as .npy arrays + .pkl scalers

**Outputs:**
```
preprocessed_data/
├── X_train_unscaled.npy (33254, 136)  ← For Random Forest
├── X_test_unscaled.npy (8314, 136)
├── X_train_standard.npy (33254, 136)  ← For SVM, LR
├── X_test_standard.npy (8314, 136)
├── y_train.npy (33254,)
├── y_test.npy (8314,)
├── scaler_standard.pkl
└── scaler_minmax.pkl
```

---

## Step 3a: Random Forest Classifier

**Script:** `random_forest_classifier.py`

**What it does:**
- Train RandomForestClassifier (100 trees)
- Predict on test set
- Compute metrics & feature importance

**Results:**
```
Accuracy: 88.69%
AUC: 0.9535
Top feature: RXPACC (15.6%)
```

---

## Step 3b: Feature Extraction for Path 2

**Script:** `second_path_features.py`

**What it does:**
- Detect second peak in CIR
- Create PEAK2_IDX, PEAK2_AMP, PEAK2_GAP features
- Derive RANGE_PATH2_EST

**Results:**
```
Second peak found in 100% of samples
New features: 139 total
Ready for range regression
```

---

## Step 3c: Range Regressor

**Script:** `range_regressor.py`

**What it does:**
- Train two RandomForestRegressor models
- One for Path 1 range, one for Path 2 range
- Compute RMSE, MAE, R²

**Results:**
```
Path 1: RMSE 1.28m, MAE 0.98m, R² 0.71
Path 2: RMSE 1.32m, MAE 1.01m, R² 0.74
```

---

## Step 3d: Pair Classifier

**Script:** `pair_classifier.py`

**What it does:**
- Classify path pairs as LOS+NLOS or NLOS+NLOS
- Train RF and XGBoost
- Compare with per-path classification

---

## Step 4: Model Comparison

**Script:** `compare_models.py`

**What it does:**
- Load all model predictions
- Generate unified metrics table
- Create comparison plots

**Outputs:**
```
models/comparison/
├── metrics_comparison.csv
├── metrics_bar_comparison.png
├── confusion_matrices_all.png
├── roc_all_models_clean.png
├── feature_importance_rf_vs_xgb.png
└── nlos_safety_comparison.png
```

---

## CRISP-DM Framework

Your project follows industry standard:

```
1. Business Understanding → Precise indoor positioning
2. Data Understanding → 41,568 samples, 50/50 balanced
3. Data Preparation → Clean, scale, split
4. Modeling → RF, LR, SVM, XGB classifiers + regressors
5. Evaluation → 88.69% accuracy, 1.28m range RMSE
6. Deployment → Ready for production systems
```

---

## Key Principles

✅ **Run in order** — Each step depends on previous outputs  
✅ **Save intermediate results** — .npy + .pkl files enable fast re-evaluation  
✅ **Multiple models** — No single algorithm is universally best  
✅ **Stratified split** — Maintain class balance in train/test  

---

Next: [03_Data_Exploration.md](03_Data_Exploration.md)

